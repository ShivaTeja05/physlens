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
BODY, BODY_HEAT = 254, 253

# Per world: the frames it is rigid in, the reference frame, the face that pins its along-path position for boxy
# bodies whose flat sides let registration slide (None: free), and the clean-up's reach: how far from the frame's
# centre the body extends (radius) and the depth gap that separates it from background (gap: the truck's side is
# seen edge-on, so its own points are sparse in depth).
CONFIG = {
    "truck-sharp": {"rigid": (116, 196), "ref": 176, "anchor": "rear", "radius": 9.5, "gap": 2.0,
                    "label": "the roof is torn open at the beam"},
    "truck": {"rigid": (116, 196), "ref": 176, "anchor": "rear", "radius": 9.5, "gap": 2.0, "label": "the roof is torn open at the beam"},
    "crash-9070": {"rigid": (36, 44), "ref": 43, "anchor": "front", "radius": 3.2,
                   "label": "the front crushes against the barrier"},
    "rally-sharp": {"rigid": (166, 184), "ref": 176, "anchor": None, "cut": 0.35, "radius": 2.8, "label": None},
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


def simulate(name):
    cfg = CONFIG[name]
    h, P, C, tag = load(name)
    # Drop an earlier run's body groups.
    keep_pts = np.ones(len(P), bool)
    groups = []
    for g in h["groups"]:
        if g["tag"] in (BODY, BODY_HEAT):
            keep_pts[g["start"]:g["start"] + g["count"]] = False
    if not keep_pts.all():
        idx = np.cumsum(keep_pts) - 1
        for g in h["groups"]:
            if g["tag"] not in (BODY, BODY_HEAT):
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
    cov = coverage(body)                                           # stage 3
    hc = heat(body.seen)
    start = len(P)
    nb = len(body.points)
    P = np.vstack([P, body.points, body.points])
    C = np.vstack([C, body.colours, hc])
    tag = np.concatenate([tag, np.full(nb, BODY), np.full(nb, BODY_HEAT)])
    h["groups"] = h["groups"] + [{"tag": BODY, "start": start, "count": nb},
                                 {"tag": BODY_HEAT, "start": start + nb, "count": nb}]
    used = [p for p in body.poses if p["used"]]
    h["notes"]["body"] = {"poses": body.poses, "rigid_frames": [lo_f, hi_f],
                          "rigid_until_t": max(p["t"] for p in used) if used else None,
                          "after_rigid": cfg["label"], "points": nb, "frames_used": len(used), "frames": len(sel),
                          "passes": body.passes, "coverage": cov,
                          "seen_max": int(body.seen.max()), "seen_median": float(np.median(body.seen))}
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
