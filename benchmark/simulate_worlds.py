"""Turn each point world's per-frame object points into one body that moves (physlens/objsim.py).

  stage 1  extract   the world's per-frame object points (from export), cleaned of background caught by the mask
  stage 2  combine   one body in its own coordinates + a pose per frame, recursively refined, while it is rigid
  stage 3  analyse   which sides of the body have points and which no frame ever saw
Writes into media/pointworld/<world>: the body as group tag 254 (its real colours) and 253 (coloured by how many
frames saw each spot), and notes.body (poses, the time it stops being rigid, coverage). Run after
export_pointworlds.py / crash3d.py --export and world_events.py.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/simulate_worlds.py [world ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from physlens.objsim import build_body, clean, coverage, heat   # noqa: E402

PW = ROOT / "media" / "pointworld"
BODY, BODY_HEAT, PREDICTED = 254, 253, 252

# Per world: the frames it is rigid in, the reference frame, the face that pins its along-path position for boxy
# bodies whose flat sides let registration slide (None: free), and the clean-up's reach: how far from the frame's
# centre the body extends (radius) and the depth gap that separates it from background (gap: the truck's side is
# seen edge-on, so its own points are sparse in depth).
CONFIG = {
    "truck-sharp": {"rigid": (116, 196), "ref": 176, "anchor": "rear", "radius": 9.5, "gap": 2.0, "prior": "truck2",
                    "label": "the roof is torn open at the beam"},
    "truck": {"rigid": (116, 196), "ref": 176, "anchor": "rear", "radius": 9.5, "gap": 2.0, "label": "the roof is torn open at the beam"},
    "crash-9070": {"rigid": (36, 44), "ref": 43, "anchor": "front", "radius": 3.2, "prior": "crash9070",
                   "truth": {"model": "2015 Lexus NX 200t", "length_m": 4.63, "width_m": 1.814, "speed_kmh": 56.55,
                             "source": "NHTSA test 9070"},
                   "label": "the front crushes against the barrier"},
    "rally-sharp": {"rigid": (166, 184), "ref": 176, "anchor": None, "cut": 0.35, "radius": 2.8, "label": None,
                    "prior": "rally", "truth": {"model": "2019 Toyota Yaris WRC", "length_m": 4.085, "width_m": 1.875,
                                                "source": "published"}},
    "rally": {"rigid": (166, 184), "ref": 176, "anchor": None, "cut": 0.35, "radius": 2.8, "label": None},
}


def load(name):
    h = json.loads((PW / f"{name}.json").read_text())
    n = h["points"]
    raw = np.fromfile(PW / f"{name}.bin", dtype=np.uint8)
    q = raw[: n * 6].view("<i2").reshape(n, 3).astype(np.float64)
    P = np.array(h["origin"]) + q * np.array(h["step"])
    C = raw[n * 6: n * 9].reshape(n, 3)
    tag = raw[n * 9: n * 10]
    return h, P, C, tag


def save(name, h, P, C, tag):
    lo, hi = P.min(0), P.max(0)
    step = np.maximum((hi - lo) / 65000, 1e-6)
    q = np.round((P - lo) / step).astype(np.int64) - 32500
    with open(PW / f"{name}.bin", "wb") as fh:
        fh.write(q.astype("<i2").tobytes()); fh.write(C.astype(np.uint8).tobytes()); fh.write(tag.astype(np.uint8).tobytes())
    h["origin"], h["step"], h["points"] = (lo + 32500 * step).round(6).tolist(), step.round(9).tolist(), int(len(P))
    (PW / f"{name}.json").write_text(json.dumps(h))


CACHE = ROOT / "records" / "real_cache"
MIN_VISIBLE = 0.75     # poses used for speed: frames showing at least this share of the body's length


def visible_share(clouds, poses, length):
    """Per frame: the share of the body's length its points span, in body coordinates."""
    out = []
    for cl, p in zip(clouds, poses):
        if not p["used"]:
            out.append(None)
            continue
        c, s_ = np.cos(p["yaw"]), np.sin(p["yaw"])
        R = np.array([[c, 0, -s_], [0, 1, 0], [s_, 0, c]])
        b = (cl - [p["x"], p["y"], p["z"]]) @ R            # world → body (R is body → world)
        out.append(round(float(np.subtract(*np.percentile(b[:, 0], [98, 2])) / length), 3))
    return out


def predict(name, cfg, body, cov, h):
    """Stage 3b, predicted and labelled: the identified model's published size (Muse Spark, priors.py) gives the
    scale; its width places the mirror plane for the side no frame saw (vehicles are left-right symmetric); speed
    from the poses of frames that show most of the body. Returns (predicted points, colours, notes)."""
    pf = CACHE / cfg["prior"] / "prior.json" if cfg.get("prior") else None
    if pf is None or not pf.exists():
        return None, None, {}
    prior = json.loads(pf.read_text())
    P = body.points
    L = float(np.subtract(*np.percentile(P[:, 0], [99, 1])))
    notes = {"prior": {k: prior.get(k) for k in ("make", "model", "years", "length_m", "width_m", "height_m",
                                                  "confidence", "text_read", "why", "source")},
             "length_measured_ai_scale_m": round(L, 3)}
    k = prior["length_m"] / L if prior.get("length_m") else None
    notes["scale_from_prior"] = round(k, 4) if k else None
    # Speed from the poses (whole body), frames showing most of it.
    good = [p for p in body.poses if p["used"] and (p.get("visible") or 0) >= MIN_VISIBLE]
    if len(good) >= 3:
        xz = np.array([[p["x"], p["z"]] for p in good])
        tt = np.array([p["t"] for p in good])
        d = xz[-1] - xz[0]
        d /= np.linalg.norm(d)
        v = float(np.polyfit(tt, xz @ d, 1)[0])
        notes["speed_from_poses"] = {"frames": [p["frame"] for p in good], "kmh_ai_scale": round(v * 3.6, 2),
                                     "kmh_prior_scale": round(v * 3.6 * k, 2) if k else None}
    pred = cols = None
    hidden = [f for f in ("left", "right") if f in cov["never_seen"]]
    if hidden and prior.get("width_m") and k:
        w = prior["width_m"] / k                          # the published width in this world's units
        if hidden[0] == "right":                          # hidden at +z: the seen (left) side is at −z
            near = float(np.percentile(P[:, 2], 2))
            plane = near + w / 2
            seen = P[:, 2] < plane
            beyond = P[:, 2] > near + w + 0.05
        else:
            near = float(np.percentile(P[:, 2], 98))
            plane = near - w / 2
            seen = P[:, 2] > plane
            beyond = P[:, 2] < near - w - 0.05
        pred = P[seen].copy()
        pred[:, 2] = 2 * plane - pred[:, 2]
        cols = body.colours[seen]
        notes["predicted_side"] = {"side": hidden[0], "points": int(len(pred)), "mirror_plane_z": round(plane, 3),
                                   "width_used_m": round(w, 3), "real_points_beyond_far_side_dropped": int(beyond.sum()),
                                   "how": "mirror symmetry about the car's centre plane, placed half the identified "
                                          "model's published width in from the seen side"}
        notes["_drop"] = beyond
    if cfg.get("truth"):
        tr = cfg["truth"]
        rows = [["Identified by Muse Spark", f"{prior.get('make', '')} {prior.get('model', '')} ({prior.get('confidence')})", tr["model"]],
                ["Length, AI depth scale", f"{L:.2f} m", f"{tr['length_m']} m"]]
        if prior.get("length_m"):
            rows.append(["Length, identified model (predicted)", f"{prior['length_m']:.2f} m", f"{tr['length_m']} m"])
        if prior.get("width_m"):
            rows.append(["Width, identified model (predicted)", f"{prior['width_m']:.2f} m", f"{tr['width_m']} m"])
        sp = notes.get("speed_from_poses")
        if sp and tr.get("speed_kmh"):
            rows.append(["Speed from the body's poses, AI depth scale", f"{sp['kmh_ai_scale']:.1f} km/h", f"{tr['speed_kmh']} km/h"])
            if sp.get("kmh_prior_scale"):
                rows.append(["Speed, scaled by the identified model's length", f"{sp['kmh_prior_scale']:.1f} km/h", f"{tr['speed_kmh']} km/h"])
            rows.append(["Speed, scaled by the true length (check)", f"{sp['kmh_ai_scale'] * tr['length_m'] / L:.1f} km/h", f"{tr['speed_kmh']} km/h"])
        notes["truth_rows"] = rows
    return pred, cols, notes


def simulate(name):
    cfg = CONFIG[name]
    h, P, C, tag = load(name)
    # Drop an earlier run's body groups.
    keep_pts = np.ones(len(P), bool)
    groups = []
    for g in h["groups"]:
        if g["tag"] in (BODY, BODY_HEAT, PREDICTED):
            keep_pts[g["start"]:g["start"] + g["count"]] = False
    if not keep_pts.all():
        idx = np.cumsum(keep_pts) - 1
        for g in h["groups"]:
            if g["tag"] not in (BODY, BODY_HEAT, PREDICTED):
                groups.append({**g, "start": int(idx[g["start"]])})
        P, C, tag = P[keep_pts], C[keep_pts], tag[keep_pts]
        h["groups"] = groups
    cam = None if h["camera"]["moving"] else np.array(h["camera"]["position"])
    moments = [g for g in h["groups"] if g["tag"] != 255]
    lo_f, hi_f = cfg["rigid"]
    sel = [g for g in moments if lo_f <= g["frame"] <= hi_f]
    clouds, cols = [], []
    for g in sel:                                                      # stage 1
        p, c = clean(P[g["start"]:g["start"] + g["count"]], C[g["start"]:g["start"] + g["count"]], cam,
                     gap=cfg.get("gap", 0.6), radius=cfg.get("radius", 4.0))
        clouds.append(p); cols.append(c)
    frames = [g["frame"] for g in sel]
    times = [g["t"] for g in sel]
    centres = np.array([np.median(p, axis=0) for p in clouds])
    if cfg["anchor"] is None:
        # Headings along the track (the body turns): smoothed differences of the centres.
        dv = np.gradient(centres[:, [0, 2]], axis=0)
        heads = np.arctan2(dv[:, 1], dv[:, 0])
        lock = None
    else:
        d = centres[-1, [0, 2]] - centres[0, [0, 2]]
        d /= np.linalg.norm(d)
        heads = np.full(len(sel), np.arctan2(d[1], d[0]))
        a = [p[:, [0, 2]] @ d for p in clouds]
        face = np.array([np.percentile(x, 3) if cfg["anchor"] == "rear" else np.percentile(x, 97) for x in a])
        face = np.maximum.accumulate(face)                         # it doesn't reverse while rigid
        # The anchor face's position stands for every frame's along-path position relative to the reference.
        lock = (d, face - face[frames.index(cfg["ref"])] + centres[frames.index(cfg["ref"]), [0, 2]] @ d)
    body = build_body(clouds, cols, frames, times, frames.index(cfg["ref"]), centres, heads, lock=lock,
                      cut=cfg.get("cut", 0.25))                    # stage 2
    vis = visible_share(clouds, body.poses, float(np.subtract(*np.percentile(body.points[:, 0], [98, 2]))))
    for p, v in zip(body.poses, vis):
        p["visible"] = v
    cov = coverage(body)                                           # stage 3
    pred, pred_c, pnotes = predict(name, cfg, body, cov, h)        # stage 3b, labelled as predicted
    drop = pnotes.pop("_drop", None)
    if drop is not None and drop.any():
        from physlens.objsim import Body
        body = Body(body.points[~drop], body.colours[~drop], body.seen[~drop], body.frames, body.t, body.poses, body.passes)
        cov = coverage(body)
    hc = heat(body.seen)
    start = len(P)
    nb = len(body.points)
    P = np.vstack([P, body.points, body.points] + ([pred] if pred is not None else []))
    C = np.vstack([C, body.colours, hc] + ([pred_c] if pred is not None else []))
    tag = np.concatenate([tag, np.full(nb, BODY), np.full(nb, BODY_HEAT)] + ([np.full(len(pred), PREDICTED)] if pred is not None else []))
    h["groups"] = h["groups"] + [{"tag": BODY, "start": start, "count": nb},
                                 {"tag": BODY_HEAT, "start": start + nb, "count": nb}]
    if pred is not None:
        h["groups"].append({"tag": PREDICTED, "start": start + 2 * nb, "count": int(len(pred))})
    used = [p for p in body.poses if p["used"]]
    h["notes"]["body"] = {"poses": body.poses, "rigid_frames": [lo_f, hi_f],
                          "rigid_until_t": max(p["t"] for p in used) if used else None,
                          "after_rigid": cfg["label"], "points": nb, "frames_used": len(used), "frames": len(sel),
                          "passes": body.passes, "coverage": cov,
                          "seen_max": int(body.seen.max()), "seen_median": float(np.median(body.seen)), **pnotes}
    if pnotes.get("truth_rows"):
        rest = [r for r in h["notes"].get("truth_table", []) if not r[0].startswith(("Identified", "Length,", "Width,", "Speed"))]
        h["notes"]["truth_table"] = pnotes["truth_rows"] + rest
    save(name, h, P, C, tag)
    return h["notes"]["body"]


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in CONFIG if (PW / f"{n}.json").exists()]
    out = {}
    for name in names:
        b = simulate(name)
        out[name] = {k: v for k, v in b.items() if k != "poses"}
        print(f"{name}: body {b['points']:,} points from {b['frames_used']}/{b['frames']} frames;",
              "passes", [p.get("largest_move_m") for p in b["passes"]],
              "| size", b["coverage"]["size_m"], "| never seen:", b["coverage"]["never_seen"])
        print("   faces:", {f: v["covered_pct"] for f, v in b["coverage"]["faces"].items()})
        used = [p for p in b["poses"] if p["used"]]
        print("   rms per frame:", [p["rms"] for p in used])
    (ROOT / "benchmark" / "simulate_worlds.json").write_text(json.dumps(out, indent=1, default=float))
