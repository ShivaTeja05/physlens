"""NHTSA 9070 from both sides: cam15 (real-time, left side) + camera No. 7 "Right Overall" (cam12, 1,000 fps, right side).

  sync      cam12's burned-in clock reads −50 ms at its frame 0 and 0 ms (impact) at frame 50: frame i is
            (i − 50) ms from impact. cam15's impact was found from its own points (crash3d.py); the two are placed on
            one timeline through that moment.
  body      cam12's right-side body (objsim), the same way as cam15's left side
  merge     the two one-sided bodies of the same car joined where both see it (front, bonnet, roof): a similarity
            (scale too: each camera's AI depth has its own scale). The width between the real left and right sides
            is then measured, and the mirror-predicted right side is compared with the real one.
  motion    at 1,000 fps: the approach speed, and after impact the rigid rear section's speed curve, against
            NHTSA's rear-floorpan accelerometers.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/crash_twosided.py [--setup]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
import crash3d as c15   # noqa: E402
from physlens.geometry.client import SharpDepth, sharp_depth   # noqa: E402
from physlens.sharp import fov_x   # noqa: E402
from physlens.vehicle3d import read_at, track   # noqa: E402

VIDEO = ROOT / "datasets" / "nhtsa" / "9070" / "cam12.mp4"
CACHE = ROOT / "records" / "real_cache" / "crash9070_cam12"
IDX = list(range(0, 121, 5)) + list(range(130, 361, 10))   # 5 ms through approach and early crush, then 10 ms (49 frames: under RunPod's 20 MB request limit)
FRAME_AT_IMPACT = 50                  # the burned-in clock reads 0.00 ms here
SHARP_W = 1280


def t_rel(frame):
    """Seconds from impact for a cam12 frame."""
    return (frame - FRAME_AT_IMPACT) / 1000.0


def sharp(views):
    f = CACHE / f"moge3_{SHARP_W}.npz"
    if f.exists():
        z = np.load(f)
        return [SharpDepth(z[f"d{j}"].astype(np.float32), z[f"m{j}"], z[f"n{j}"].astype(np.float32) / 127, z[f"K{j}"])
                for j in range(len(IDX))]
    frames, _ = read_at(VIDEO, IDX, SHARP_W)
    sd = sharp_depth(frames, fov_x=[fov_x(v) for v in views])
    np.savez_compressed(f, **{k: v for j, s in enumerate(sd) for k, v in {
        f"d{j}": s.depth.astype(np.float16), f"m{j}": s.mask, f"n{j}": np.round(s.normal * 127).astype(np.int8),
        f"K{j}": s.intrinsics}.items()})
    return sd


def lens():
    """This high-speed camera's lens. MapAnything alone guesses a 63° field of view; MoGe-3's own estimate is 52°
    (two frames, cached). A single image can't tell a wide lens from a near object, and the two disagree here;
    MoGe-3's is used, given to MapAnything as the intrinsics (checked afterwards: the car's length comes out
    right only with it)."""
    f = CACHE / "lens.json"
    if f.exists():
        return json.loads(f.read_text())
    fr, _ = read_at(VIDEO, [0, 60], SHARP_W)
    sd = sharp_depth(fr)
    fovs = [float(np.degrees(2 * np.arctan(s.depth.shape[1] / 2 / s.intrinsics[0, 0]))) for s in sd]
    out = {"fov_x_deg": float(np.mean(fovs)), "per_frame": fovs, "source": "MoGe-3 (own estimate)"}
    f.write_text(json.dumps(out))
    return out


def geometry_with_lens(frames):
    """MapAnything over the keyframes with the lens given (cached as mapanything.npz, the name track() reads)."""
    from physlens.geometry.client import reconstruct
    f = CACHE / "mapanything.npz"
    if f.exists():
        return
    fov = np.radians(lens()["fov_x_deg"])
    h0, w0 = frames[0].shape[:2]
    fx = w0 / 2 / np.tan(fov / 2)
    K = np.array([[fx, 0, w0 / 2], [0, fx, h0 / 2], [0, 0, 1.0]])
    views = reconstruct(frames, timeout_s=900, intrinsics=[K] * len(frames))
    np.savez_compressed(f, **{k: v for j, vw in enumerate(views) for k, v in {
        f"depth{j}": vw.depth, f"cam{j}": vw.points_cam, f"world{j}": vw.points_world,
        f"conf{j}": vw.confidence if vw.confidence is not None else np.ones_like(vw.depth),
        f"K{j}": vw.intrinsics, f"pose{j}": vw.camera_pose, f"sx{j}": vw.scale_x, f"sy{j}": vw.scale_y}.items()})


def setup():
    CACHE.mkdir(parents=True, exist_ok=True)
    geometry_with_lens(read_at(VIDEO, IDX)[0])
    tr, frames, views = track(VIDEO, IDX, "car", CACHE, sam_video=True)
    sd = sharp(views)
    c15.fix_camera(views)
    ms = [cv2.imread(str(CACHE / "samvideo_car" / f"{j:03d}.png"), 0) for j in range(len(IDX))]
    ms = [m > 0 if m is not None and (m > 0).any() else None for m in ms]
    return tr, frames, views, sd, ms


if __name__ == "__main__" and "--setup" in sys.argv:
    tr, frames, views, sd, ms = setup()
    print("car seen in", sum(m is not None for m in ms), "of", len(IDX), "keyframes")


def sensor_curve():
    """NHTSA's rear-floorpan accelerometers (curves 93 and 94, averaged), integrated from the official speed at
    impact: time (s), speed (m/s) and the rear's travel since impact (m)."""
    S = ROOT / "datasets" / "nhtsa" / "9070" / "sensors"
    a = [np.loadtxt(S / f"floorpan_{c}.tsv") for c in ("093", "094")]
    t = a[0][:, 0]
    g = (a[0][:, 1] + a[1][:, 1]) / 2 * 9.80665
    v0 = c15.TRUTH["impact_kmh"] / 3.6
    i0 = int(np.argmin(np.abs(t)))
    v = np.empty_like(t)
    v[i0] = v0
    dt = np.diff(t)
    v[i0 + 1:] = v0 + np.cumsum((g[i0:-1] + g[i0 + 1:]) / 2 * dt[i0:])
    v[:i0] = v0 - np.cumsum(((g[:i0] + g[1:i0 + 1]) / 2 * dt[:i0])[::-1])[::-1]
    x = np.zeros_like(t)
    x[i0 + 1:] = np.cumsum((v[i0:-1] + v[i0 + 1:]) / 2 * dt[i0:])
    x[:i0] = -np.cumsum(((v[:i0] + v[1:i0 + 1]) / 2 * dt[:i0])[::-1])[::-1]
    return t, v, x


def car_clouds(tr, views, sd, ms, frames_rgb):
    """Per keyframe: the car's points (MoGe-3, scale fitted around the car) with colours, ground frame."""
    from physlens.sharp import align, main_depth_cluster
    T, s = tr.T_ground, tr.scale
    out = []
    for j in range(len(IDX)):
        m = ms[j]
        if m is None:
            out.append(None)
            continue
        roi = c15.mask_at(m, sd[j].depth.shape, dilate=121)
        al = align(views[j], sd[j], roi=roi)
        P = (al.points_world @ T[:3, :3].T + T[:3, 3]) * s
        o = al.valid_loose & np.isfinite(P).all(-1) & c15.mask_at(m, P.shape[:2], erode=3)
        cam = views[j].camera_pose
        zc = (al.points_world[o] - cam[:3, 3]) @ cam[:3, 2]
        keep = main_depth_cluster(zc * s)
        rgb = cv2.resize(frames_rgb[j], (P.shape[1], P.shape[0]), interpolation=cv2.INTER_AREA)[o][:, ::-1]
        out.append((P[o][keep], rgb[keep]))
    return out


def track_section(template, clouds, d, start, search=0.6, cut=0.08):
    """Position along the path of a rigid section (template: ground-frame points at a known position `start`) in
    each frame: the shift along `d` that lays the frame's points on it best (1-D search, then median refinement)."""
    from scipy.spatial import cKDTree
    tree = cKDTree(template)
    out = []
    prev = 0.0
    for c in clouds:
        if c is None:
            out.append(np.nan)
            continue
        P = c[0]
        if len(P) > 6000:
            P = P[np.linspace(0, len(P) - 1, 6000).astype(int)]
        dd3 = np.array([d[0], 0, d[1]])
        best = (np.inf, prev)
        for sft in np.arange(prev - search, prev + search + 1e-9, 0.01):
            dist, _ = tree.query(P - sft * dd3, distance_upper_bound=cut)
            ok = np.isfinite(dist)
            score = (np.mean(dist[ok]) if ok.any() else cut) + cut * (1 - ok.mean())
            if score < best[0]:
                best = (score, sft)
        sft = best[1]
        for _ in range(5):
            dist, j = tree.query(P - sft * dd3, distance_upper_bound=cut)
            ok = np.isfinite(dist)
            if ok.sum() < 50:
                break
            sft += float(np.median(((P[ok] - sft * dd3) - template[j[ok]]) @ dd3))
        out.append(start + sft)
        prev = sft
    return np.array(out)


def analyse():
    from physlens.objsim import build_body, clean, coverage
    cl, cam, T, winfo = car_clouds_moge()
    cl = [None if c is None else clean(c[0], c[1], cam, radius=3.5) for c in cl]
    t = np.array([t_rel(f) for f in IDX])
    pre = [j for j, f in enumerate(IDX) if f <= FRAME_AT_IMPACT and cl[j] is not None]
    med = np.array([np.median(cl[j][0][:, [0, 2]], 0) for j in pre])
    # Direction of travel over the approach (the car moves 0.8 m in these 50 ms).
    d = med[-1] - med[0]
    d /= np.linalg.norm(d)
    front = np.array([np.percentile(c[0][:, [0, 2]] @ d, 98) if c is not None else np.nan for c in cl])
    rear = np.array([np.percentile(c[0][:, [0, 2]] @ d, 2) if c is not None else np.nan for c in cl])
    # Approach: front positions over the 50 ms before impact, straight line.
    A = np.vstack([t[pre], np.ones(len(pre))]).T
    v_front = float(np.linalg.lstsq(A, front[pre], rcond=None)[0][0])
    # Rigid rear section: the rear 40% of the car in the last frame before impact; tracked through the crush.
    j0 = pre[-1]
    P0 = cl[j0][0]
    a0 = P0[:, [0, 2]] @ d
    L0 = float(np.percentile(a0, 98) - np.percentile(a0, 2))
    sec = P0[a0 < np.percentile(a0, 2) + 0.4 * L0]
    r_sec = track_section(sec, cl, d, 0.0)
    r_sec -= np.interp(0.0, t, r_sec)                      # travel since impact
    # Body of the right side, before impact.
    cen = np.array([np.median(cl[j][0], 0) for j in pre])
    heads = np.full(len(pre), np.arctan2(d[1], d[0]))
    face = np.maximum.accumulate(front[pre])
    body = build_body([cl[j][0] for j in pre], [cl[j][1] for j in pre], [IDX[j] for j in pre], [float(t[j]) for j in pre],
                      len(pre) - 1, cen, heads, lock=(d, face - face[-1] + cen[-1, [0, 2]] @ d))
    cov = coverage(body)
    L_body = float(np.subtract(*np.percentile(body.points[:, 0], [99, 1])))
    return {"t": t, "front": front, "rear": rear, "rear_section": r_sec, "v_front": v_front, "d": d, "body": body,
            "coverage": cov, "L_body": L_body, "L_frame": L0, "clouds": cl, "world": winfo}


def merge_sides(L, R, w_over_l=0.39, iters=50):
    """Join a left-side body L (seen side at −z) and a right-side body R (seen side at +z) of the same car, both in
    body coordinates (x forward, y up). Start: R scaled to L's length, its right side placed `w_over_l` × length
    out from L's left side (a typical car's width/length, only a start). Then ICP on what both see (similarity:
    scale, shift; small turn about y), with a shrinking cut-off. Returns R in L's coordinates and the fit."""
    from scipy.spatial import cKDTree
    lL = np.subtract(*np.percentile(L[:, 0], [99, 1]))
    lR = np.subtract(*np.percentile(R[:, 0], [99, 1]))
    s = lL / lR
    zl, zr = np.percentile(L[:, 2], 2), np.percentile(R[:, 2], 98)
    t = np.array([np.percentile(L[:, 0], 50) - s * np.percentile(R[:, 0], 50),
                  np.percentile(L[:, 1], 1) - s * np.percentile(R[:, 1], 1),
                  zl + w_over_l * lL - s * zr])
    R3 = np.eye(3)
    tree = cKDTree(L)
    cut = 0.3
    for _ in range(iters):
        Q = s * R @ R3.T + t
        dist, j = tree.query(Q, distance_upper_bound=cut)
        ok = np.isfinite(dist)
        if ok.sum() < 200:
            break
        X, Y = R[ok], L[j[ok]]
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        Sxz = Xc[:, [0, 2]].T @ Yc[:, [0, 2]]
        ang = np.clip(np.arctan2(Sxz[0, 1] - Sxz[1, 0], Sxz[0, 0] + Sxz[1, 1]), -np.radians(5), np.radians(5))
        c, sn = np.cos(ang), np.sin(ang)
        R3 = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])
        XR = Xc @ R3.T
        s = float((XR * Yc).sum() / (XR * XR).sum())
        t = my - s * mx @ R3.T
        cut = max(cut * 0.92, 0.05)
    Q = s * R @ R3.T + t
    dist, _ = tree.query(Q, distance_upper_bound=0.08)
    ok = np.isfinite(dist)
    return Q, {"scale_R_to_L": round(s, 4), "turn_deg": round(float(np.degrees(np.arctan2(R3[2, 0], R3[0, 0]))), 2),
               "shared_points_within_8cm": int(ok.sum()), "shared_share": round(float(ok.mean()), 3),
               "rms_m": round(float(np.sqrt(np.mean(dist[ok] ** 2))), 4) if ok.any() else None}


def main():
    import simulate_worlds as sw
    from physlens.objsim import Body, coverage, heat
    r = analyse()
    t, xr = r["t"], r["rear_section"]
    ts, vs, xs = sensor_curve()
    TR = c15.TRUTH
    k = TR["car_length_m"] / r["L_body"]                         # this camera's AI scale → metres, by the true length
    out = {"keyframes": IDX, "approach_kmh_ai": round(r["v_front"] * 3.6, 2),
           "approach_kmh_true_length": round(r["v_front"] * 3.6 * k, 2), "truth_kmh": TR["impact_kmh"],
           "body_length_ai_m": round(r["L_body"], 3), "frame_length_ai_m": round(r["L_frame"], 3),
           "coverage": {f: v["covered_pct"] for f, v in r["coverage"]["faces"].items()},
           "never_seen": r["coverage"]["never_seen"], "world": r["world"]}
    # Crush: the rigid rear section's travel since impact vs the accelerometers'.
    after = (t >= 0) & np.isfinite(xr)
    xv = xr * k
    xsens = np.interp(t[after], ts, xs)
    kmax = int(np.nanargmax(np.where(after, xv, np.nan)))
    out["crush"] = {"max_travel_m": round(float(xv[kmax]), 3), "at_ms": round(float(t[kmax] * 1000), 1),
                    "sensor_max_travel_m": round(float(xs.max()), 3), "sensor_at_ms": round(float(ts[np.argmax(xs)] * 1000), 1),
                    "rms_vs_sensor_m": round(float(np.sqrt(np.mean((xv[after] - xsens) ** 2))), 3)}
    late = after & (t > 0.12)
    if late.sum() >= 4:
        out["crush"]["rebound_ms"] = round(float(-np.polyfit(t[late], xv[late], 1)[0]), 2)
        out["crush"]["sensor_rebound_ms_120_300"] = round(float(-np.polyfit(ts[(ts > 0.12)], xs[(ts > 0.12)], 1)[0]), 2)
    out["curve"] = {"t_ms": (t * 1000).round(1).tolist(), "video_m": [None if not np.isfinite(x) else round(float(x), 4) for x in xv],
                    "sensor_t_ms": (ts[::20] * 1000).round(1).tolist(), "sensor_m": xs[::20].round(4).tolist()}
    print(json.dumps({kk: v for kk, v in out.items() if kk not in ("curve", "keyframes")}, indent=1))
    # Merge with cam15's left-side body (crash-9070 world).
    h, P, C, tag = sw.load("crash-9070")
    g = {x["tag"]: x for x in h["groups"]}
    Lb = P[g[sw.BODY]["start"]: g[sw.BODY]["start"] + g[sw.BODY]["count"]]
    Lc = C[g[sw.BODY]["start"]: g[sw.BODY]["start"] + g[sw.BODY]["count"]]
    pred = P[g[sw.PREDICTED]["start"]: g[sw.PREDICTED]["start"] + g[sw.PREDICTED]["count"]] if sw.PREDICTED in g else None
    Rb = r["body"].points
    R_in_L, fit = merge_sides(Lb, Rb)
    k15 = TR["car_length_m"] / float(np.subtract(*np.percentile(Lb[:, 0], [99, 1])))
    Pf = np.vstack([Lb, R_in_L])
    Cf = np.vstack([Lc, r["body"].colours])
    src = np.r_[np.zeros(len(Lb), int), np.ones(len(R_in_L), int)]
    fused = Body(Pf, Cf, np.where(src == 0, 1, 1), [], [], [], [])
    cov = coverage(fused)
    zl, zr = np.percentile(Lb[:, 2], 2), np.percentile(R_in_L[:, 2], 98)
    width = float(zr - zl) * k15
    out["merge"] = fit | {"width_from_both_sides_m": round(width, 3), "true_width_m": 1.814,
                          "coverage_fused": {f: v["covered_pct"] for f, v in cov["faces"].items()},
                          "never_seen_fused": cov["never_seen"]}
    if pred is not None:
        from scipy.spatial import cKDTree
        right_real = R_in_L[R_in_L[:, 2] > (zl + zr) / 2]
        dist, _ = cKDTree(pred).query(right_real)
        out["merge"]["mirror_vs_real_right_side"] = {"median_m": round(float(np.median(dist)) * k15, 3),
                                                     "p90_m": round(float(np.percentile(dist, 90)) * k15, 3),
                                                     "real_points_compared": int(len(right_real))}
    print(json.dumps(out["merge"], indent=1))
    (ROOT / "benchmark" / "crash_twosided.json").write_text(json.dumps(out, indent=1, default=float))
    if "--write" not in sys.argv:
        return out
    # Into the crash world: the body becomes both sides (real); the mirror prediction stays as its own group to
    # compare with; the crush curve and the two-camera results go into the notes.
    keep = np.ones(len(P), bool)
    for x in h["groups"]:
        if x["tag"] in (sw.BODY, sw.BODY_HEAT):
            keep[x["start"]: x["start"] + x["count"]] = False
    idx = np.cumsum(keep) - 1
    h["groups"] = [{**x, "start": int(idx[x["start"]])} for x in h["groups"] if x["tag"] not in (sw.BODY, sw.BODY_HEAT)]
    P, C, tag = P[keep], C[keep], tag[keep]
    hc = heat(np.where(src == 0, 1, 3))                           # colour by camera: left (cam15) / right (cam12)
    start = len(P)
    h["groups"] += [{"tag": sw.BODY, "start": start, "count": int(len(Pf))},
                    {"tag": sw.BODY_HEAT, "start": start + len(Pf), "count": int(len(Pf))}]
    P = np.vstack([P, Pf, Pf]); C = np.vstack([C, Cf, hc])
    tag = np.concatenate([tag, np.full(len(Pf), sw.BODY), np.full(len(Pf), sw.BODY_HEAT)])
    b = h["notes"]["body"]
    b["points"] = int(len(Pf))
    b["sources"] = {"cam15 (left side, 30 fps)": int((src == 0).sum()), "Right Overall (right side, 1,000 fps)": int((src == 1).sum()),
                    "merge": fit}
    b["coverage_one_camera"] = b.get("coverage")
    b["coverage"] = cov
    b["heat_means"] = "camera: blue = left camera (cam15), yellow = right camera (Right Overall)"
    if "predicted_side" in b:
        b["predicted_side"]["compare"] = out["merge"].get("mirror_vs_real_right_side")
    h["notes"]["two_camera"] = {k_: out[k_] for k_ in ("approach_kmh_ai", "approach_kmh_true_length", "crush", "merge")}
    h["notes"]["crush_curve"] = out["curve"]
    h["notes"]["cameras"] = [{"name": "cam15 · left side", "position": h["camera"]["position"], "forward": h["camera"]["forward"]}]
    tt = [row for row in h["notes"].get("truth_table", []) if not row[0].startswith(("Width from both", "Crush, 1,000", "Approach, 1,000", "Stop", "Rebound, 1,000"))]
    tt += [["Approach speed, 1,000 fps camera (true length scale)", f"{out['approach_kmh_true_length']:.1f} km/h", f"{TR['impact_kmh']} km/h"],
           ["Crush, 1,000 fps rear section (true length scale)", f"{out['crush']['max_travel_m']:.2f} m at {out['crush']['at_ms']:.0f} ms",
            f"{out['crush']['sensor_max_travel_m']:.2f} m at {out['crush']['sensor_at_ms']:.0f} ms (accelerometers)"],
           ["Width from both sides (true length scale)", f"{width:.2f} m", "1.814 m (NHTSA)"]]
    if "rebound_ms" in out["crush"]:
        tt.append(["Rebound, 1,000 fps (120–300 ms)", f"{out['crush']['rebound_ms']:.2f} m/s", f"{out['crush']['sensor_rebound_ms_120_300']:.2f} m/s (accelerometers, same window)"])
    h["notes"]["truth_table"] = tt
    sw.save("crash-9070", h, P, C, tag)
    return out



def moge_world():
    """This camera's 3D from MoGe-3 alone (MapAnything's depth is ~2× too far here, MoGe-3's own scale ~0.8×; its
    shape is right: length/height within 3%). One lens for every frame (MoGe-3's own estimate), the camera fixed,
    each frame's depth fitted to the first frame's on the unmoving lab (car masks dilated out). Returns per-frame
    camera-frame points (H, W, 3), validity, colours and masks, and the floor frame T (camera → ground, y up)."""
    from physlens.moving_camera import _plane_ransac
    f = CACHE / "moge_own_1280.npz"
    fov = lens()["fov_x_deg"]
    rgb, _ = read_at(VIDEO, IDX, SHARP_W)
    if f.exists():
        z = np.load(f)
        D = [z[f"d{j}"].astype(np.float32) for j in range(len(IDX))]
        M = [z[f"m{j}"] for j in range(len(IDX))]
        K = z["K0"]
    else:
        sd = sharp_depth(rgb, fov_x=[fov] * len(IDX))
        D, M, K = [s.depth for s in sd], [s.mask for s in sd], sd[0].intrinsics
        np.savez_compressed(f, K0=K, **{f"d{j}": d.astype(np.float16) for j, d in enumerate(D)}, **{f"m{j}": m for j, m in enumerate(M)})
    h, w = D[0].shape
    ms = []
    for j in range(len(IDX)):
        m = cv2.imread(str(CACHE / "samvideo_car" / f"{j:03d}.png"), 0)
        ms.append(None if m is None or not (m > 0).any() else cv2.resize((m > 0).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0)
    static = np.ones((h, w), bool)
    for m in ms:
        if m is not None:
            static &= ~(cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)) > 0)
    scales = []
    for j in range(len(IDX)):
        ok = static & M[j] & M[0] & np.isfinite(D[j]) & np.isfinite(D[0])
        r = D[0][ok] / D[j][ok]
        scales.append(float(np.median(r)))
        D[j] = D[j] * scales[-1]
    ys, xs = np.mgrid[0:h, 0:w]
    def lift(d):
        return np.dstack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d])
    P0 = lift(D[0])
    # Floor: level planes below the camera from the unmoving lab; the lowest large one.
    S = P0[static & M[0]][::7]
    rest, planes = S, []
    up = np.array([0.0, -1.0, 0.0])                       # OpenCV camera: y down
    for _ in range(6):
        if len(rest) < 500:
            break
        n, c, inl = _plane_ransac(rest, 0.02 * float(np.median(D[0])))
        n = n if n @ up > 0 else -n
        planes.append((n, c, int(inl.sum()), float(-c @ n)))
        rest = rest[~inl]
    level = [p for p in planes if p[0] @ up > np.cos(np.radians(35)) and p[3] > 0.2]
    big = max(p[2] for p in level)
    n, c, _, cam_h = max([p for p in level if p[2] >= 0.2 * big], key=lambda p: p[3])
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    zax = np.cross(x, n)
    Rw = np.stack([x, n, zax])
    T = np.eye(4)
    T[:3, :3] = Rw
    T[:3, 3] = -Rw @ c
    info = {"fov_x_deg": round(fov, 2), "frame_scale_to_first": [round(s, 4) for s in scales], "camera_height_ai_m": round(cam_h, 3),
            "camera_tilt_deg": round(float(np.degrees(np.arccos(n @ up))), 1)}
    return [lift(d) for d in D], M, rgb, ms, T, info


def car_clouds_moge():
    """Per keyframe: the car's points (ground frame, MoGe-3's own scale) with colours."""
    from physlens.sharp import main_depth_cluster
    Ps, M, rgb, ms, T, info = moge_world()
    out = []
    for j in range(len(IDX)):
        if ms[j] is None:
            out.append(None)
            continue
        m = cv2.erode(ms[j].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        ok = m & M[j] & np.isfinite(Ps[j]).all(-1)
        Pc = Ps[j][ok]
        keep = main_depth_cluster(Pc[:, 2])
        col = cv2.resize(rgb[j], (Ps[j].shape[1], Ps[j].shape[0]), interpolation=cv2.INTER_AREA)[ok][:, ::-1]
        out.append(((Pc @ T[:3, :3].T + T[:3, 3])[keep], col[keep]))
    cam = T[:3, 3].copy()
    return out, cam, T, info


if __name__ == "__main__" and "--setup" not in sys.argv:
    main()
