"""One car body from two cameras, its hidden side predicted by symmetry, its speed from its poses (NHTSA 9070).

MapAnything can't relate cam15 (side-on) and cam17 (oblique, from the front right) directly: their views share
too little. Both see the same rigid car, so the car is the link:
  1. a body per camera (physlens/objsim.py: one body of real points + a pose per frame, before impact)
  2. the two bodies matched (rotation about the vertical, shift, and scale: each camera's AI depth has its own
     scale), so cam17's points join cam15's body: more of the front and right side
  3. the side no camera saw (the left) predicted by mirror symmetry: cars are left-right symmetric. The mirror
     plane's position comes from the front face, which both cameras see across the car's full width. Predicted
     points are kept apart and labelled; the predicted width is checked against NHTSA's 1.814 m
  4. the impact speed from the body's poses (the whole body, not one edge of it)
Writes benchmark/crash_fusion.json and, into media/pointworld/crash-9070: the fused body (group 254, its heat 253)
and the predicted side (252), with notes.body.sources / notes.body.predicted.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/crash_fusion.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
import crash3d as c15   # noqa: E402
import simulate_worlds as sw   # noqa: E402
from physlens.moving_camera import floor_frame   # noqa: E402
from physlens.objsim import build_body, clean, coverage, heat   # noqa: E402
from physlens.vehicle3d import geometry, read_at, video_masks   # noqa: E402

D = ROOT / "datasets" / "nhtsa" / "9070"
CACHE = c15.CACHE
IDX = c15.IDX
FPS = 30000 / 1001
TRUE_WIDTH = 1.814          # NHTSA VEHWID, test 9070
PREDICTED = 252


def cam17_clouds():
    """cam17's car per keyframe in its own ground frame, with colours (MapAnything at 518 px)."""
    frames, _ = read_at(D / "cam17.mp4", IDX)
    views = geometry(frames, CACHE / "cam17_mapanything.npz")
    ms = video_masks(frames, "car", CACHE / "cam17_samvideo_car")
    c15.fix_camera(views)                                  # bolted down too
    T, _ = floor_frame(views, [v.camera_pose for v in views])
    cam = T[:3, :3] @ views[0].camera_pose[:3, 3] + T[:3, 3]
    out = []
    for j, (v, m, f) in enumerate(zip(views, ms, frames)):
        if m is None:
            out.append(None)
            continue
        h, w = v.depth.shape
        mm = c15.mask_at(m, (h, w), erode=2) & np.isfinite(v.points_world).all(-1) & (v.depth > 0.3)
        P = v.points_world[mm] @ T[:3, :3].T + T[:3, 3]
        rgb = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)[mm][:, ::-1]
        out.append((P, rgb))
    return out, cam


def front_series(clouds, d):
    return np.array([np.percentile(p[:, [0, 2]] @ d, 97) if p is not None else np.nan for p in clouds])


def body_for(clouds, cam, frames, ref_frame, radius=3.2):
    """objsim body over the frames before impact (the front still advancing)."""
    cl, co, fr, tt = [], [], [], []
    for (p, c), f in zip(clouds, frames):
        p, c = clean(p, c, cam, radius=radius)
        cl.append(p); co.append(c); fr.append(f); tt.append(f / FPS)
    cen = np.array([np.median(p, axis=0) for p in cl])
    d = cen[-1, [0, 2]] - cen[0, [0, 2]]
    d /= np.linalg.norm(d)
    heads = np.full(len(cl), np.arctan2(d[1], d[0]))
    face = np.maximum.accumulate(np.array([np.percentile(p[:, [0, 2]] @ d, 97) for p in cl]))
    r = fr.index(ref_frame)
    lock = (d, face - face[r] + cen[r, [0, 2]] @ d)
    return build_body(cl, co, fr, tt, r, cen, heads, lock=lock), d


def match_bodies(A, B, iters=40, cut=0.3):
    """Similarity (scale s, rotation about y, shift) laying body B's points onto body A's, by ICP with Umeyama's
    closed form restricted to a vertical axis. Both bodies already point their x along travel and y up, so the
    start is identity with the scale from their lengths."""
    la = np.subtract(*np.percentile(A[:, 0], [99, 1]))
    lb = np.subtract(*np.percentile(B[:, 0], [99, 1]))
    s, R, t = la / lb, np.eye(3), np.zeros(3)
    tree = cKDTree(A)
    for _ in range(iters):
        Q = s * B @ R.T + t
        d, j = tree.query(Q, distance_upper_bound=cut)
        ok = np.isfinite(d)
        if ok.sum() < 100:
            break
        X, Y = B[ok], A[j[ok]]
        mx, my = X.mean(0), Y.mean(0)
        Xc, Yc = X - mx, Y - my
        # Rotation about y only: angle from the x/z components.
        Sxz = Xc[:, [0, 2]].T @ Yc[:, [0, 2]]
        ang = np.arctan2(Sxz[0, 1] - Sxz[1, 0], Sxz[0, 0] + Sxz[1, 1])
        c, sn = np.cos(ang), np.sin(ang)
        R = np.array([[c, 0, -sn], [0, 1, 0], [sn, 0, c]])
        XR = Xc @ R.T
        s = float((XR * Yc).sum() / (XR * XR).sum())
        t = my - s * mx @ R.T
        cut = max(cut * 0.9, 0.06)
    Q = s * B @ R.T + t
    d, _ = tree.query(Q, distance_upper_bound=0.1)
    ok = np.isfinite(d)
    return s, R, t, {"scale": round(s, 4), "yaw_deg": round(float(np.degrees(np.arctan2(R[2, 0], R[0, 0]))), 2),
                     "overlap_within_10cm": round(float(ok.mean()), 3), "rms_m": round(float(np.sqrt(np.mean(d[ok] ** 2))), 4)}


def mirror(P, C):
    """The left side by symmetry. The near (right, −z) side is seen; the front face is seen across the car's full
    width (both cameras look at it): its lateral extent gives the width, so the mirror plane sits half a width in
    from the near side."""
    x0, x1 = np.percentile(P[:, 0], [1, 99])
    near = float(np.percentile(P[:, 2], 2))
    front = P[(P[:, 0] > x1 - 0.35) & (P[:, 1] > 0.35) & (P[:, 1] < 0.9)]          # bumper / grille height
    w_front = float(np.percentile(front[:, 2], 98) - np.percentile(front[:, 2], 2)) if len(front) > 200 else None
    plane = near + w_front / 2 if w_front else None
    if plane is None:
        return None
    side = P[:, 2] < plane
    M = P[side].copy()
    M[:, 2] = 2 * plane - M[:, 2]
    return {"points": M, "colours": C[side], "plane_z": plane, "width_from_front_m": w_front,
            "near_side_z": near, "front_points": int(len(front))}


def main():
    h, P, C, tag = sw.load("crash-9070")
    b15 = h["notes"]["body"]
    g = {x["tag"]: x for x in h["groups"]}
    A = P[g[sw.BODY]["start"]: g[sw.BODY]["start"] + g[sw.BODY]["count"]]
    Ac = C[g[sw.BODY]["start"]: g[sw.BODY]["start"] + g[sw.BODY]["count"]]
    # The body positions are in world coordinates in the file? No: group 254 holds body coordinates.
    clouds17, cam17 = cam17_clouds()
    # cam17's own impact: the frame its front stops advancing.
    tmp = [cc[0] if cc else None for cc in clouds17]
    med = np.array([np.median(p[:, [0, 2]], 0) if p is not None else [np.nan, np.nan] for p in tmp])
    ok = np.isfinite(med[:, 0])
    i0, i1 = np.nonzero(ok)[0][[2, 8]]
    d17 = med[i1] - med[i0]
    d17 /= np.linalg.norm(d17)
    fr17 = front_series(tmp, d17)
    k_stop = int(np.nanargmax(fr17[: IDX.index(60)]))
    rigid17 = [j for j in range(len(IDX)) if tmp[j] is not None and 36 <= IDX[j] < IDX[k_stop]]
    body17, _ = body_for([clouds17[j] for j in rigid17], cam17, [IDX[j] for j in rigid17], IDX[rigid17[-2]])
    s, R, t, mrep = match_bodies(A, body17.points)
    B_in_A = s * body17.points @ R.T + t
    print("cam17 body", len(body17.points), "points from", sum(p["used"] for p in body17.poses), "frames; impact frame", IDX[k_stop],
          "| match:", mrep)
    # Fused body: cam15's points + cam17's (thinned together: one point per 3 cm cell).
    Pf = np.vstack([A, B_in_A])
    Cf = np.vstack([Ac, body17.colours])
    src = np.r_[np.zeros(len(A), int), np.ones(len(B_in_A), int)]
    _, first = np.unique(np.floor(Pf / 0.03).astype(np.int64), axis=0, return_index=True)
    first.sort()
    Pf, Cf, src = Pf[first], Cf[first], src[first]
    # Seen counts: cam15's from its build; cam17 points count its own frames.
    hA = P[g[sw.BODY_HEAT]["start"]: g[sw.BODY_HEAT]["start"] + g[sw.BODY_HEAT]["count"]]
    seen15 = dict(zip(map(tuple, np.floor(A / 0.03).astype(np.int64)), [1] * len(A)))
    seen = np.where(src == 0, 2, 1)   # placeholder replaced below
    cells = np.floor(Pf / 0.03).astype(np.int64)
    s15 = {tuple(c): n for c, n in zip(np.floor(A / 0.03).astype(np.int64), np.ones(len(A)))}
    seen = np.array([1 + (tuple(c) in s15) for c in cells])
    from physlens.objsim import Body
    fused = Body(Pf, Cf, seen, [], [], [], [])
    cov_before = b15["coverage"]
    cov = coverage(fused)
    mir = mirror(Pf, Cf)
    # Speed from the body's poses (cam15, the frames before impact).
    poses = [p for p in b15["poses"] if p["used"]]
    tt = np.array([p["t"] for p in poses])
    xz = np.array([[p["x"], p["z"]] for p in poses])
    dd = xz[-1] - xz[0]
    dd /= np.linalg.norm(dd)
    v = float(np.polyfit(tt, xz @ dd, 1)[0])
    L15 = float(np.subtract(*np.percentile(A[:, 0], [99, 1])))
    out = {"cam17_body_points": int(len(body17.points)), "cam17_frames_used": int(sum(p["used"] for p in body17.poses)),
           "cam17_impact_frame": IDX[k_stop], "match": mrep, "fused_points": int(len(Pf)),
           "fused_from_cam17": int((src == 1).sum()),
           "coverage_cam15": {k: v_["covered_pct"] for k, v_ in cov_before["faces"].items()},
           "coverage_fused": {k: v_["covered_pct"] for k, v_ in cov["faces"].items()},
           "speed_from_poses_kmh": round(v * 3.6, 2), "body_length_m": round(L15, 3),
           "speed_from_poses_car_length_scale_kmh": round(v * 3.6 * c15.TRUTH["car_length_m"] / L15, 2),
           "truth_kmh": c15.TRUTH["impact_kmh"]}
    if mir:
        w_pred = mir["width_from_front_m"]
        out["predicted"] = {"points": int(len(mir["points"])), "mirror_plane_z": round(mir["plane_z"], 3),
                            "width_from_front_m": round(w_pred, 3),
                            "width_car_length_scale_m": round(w_pred * c15.TRUTH["car_length_m"] / L15, 3),
                            "true_width_m": TRUE_WIDTH}
    print(json.dumps(out, indent=1))
    # Write the fused body and the predicted side into the world.
    keep = np.ones(len(P), bool)
    for x in h["groups"]:
        if x["tag"] in (sw.BODY, sw.BODY_HEAT, PREDICTED):
            keep[x["start"]: x["start"] + x["count"]] = False
    idx = np.cumsum(keep) - 1
    h["groups"] = [{**x, "start": int(idx[x["start"]])} for x in h["groups"] if x["tag"] not in (sw.BODY, sw.BODY_HEAT, PREDICTED)]
    P, C, tag = P[keep], C[keep], tag[keep]
    start = len(P)
    hc = heat(fused.seen)
    add_P = [Pf, Pf] + ([mir["points"]] if mir else [])
    add_C = [Cf, hc] + ([mir["colours"]] if mir else [])
    add_T = [sw.BODY, sw.BODY_HEAT] + ([PREDICTED] if mir else [])
    for pp, cc, tg in zip(add_P, add_C, add_T):
        h["groups"].append({"tag": tg, "start": start, "count": int(len(pp))})
        start += len(pp)
    P = np.vstack([P] + add_P)
    C = np.vstack([C] + add_C)
    tag = np.concatenate([tag] + [np.full(len(pp), tg) for pp, tg in zip(add_P, add_T)])
    b15["points"] = int(len(Pf))
    b15["sources"] = {"cam15": int((src == 0).sum()), "cam17": int((src == 1).sum()), "match": mrep}
    b15["coverage"] = cov
    b15["coverage_one_camera"] = cov_before
    if mir:
        b15["predicted"] = {**out["predicted"], "how": "mirror symmetry: cars are left-right symmetric; the mirror "
                            "plane is half the front face's width in from the seen side"}
    h["notes"]["body"] = b15
    tt_ = h["notes"].get("truth_table", [])
    tt_ = [r for r in tt_ if not r[0].startswith(("Impact speed from the body", "Width predicted"))]
    tt_.insert(2, ["Impact speed from the body's poses (scaled by car length)", f"{out['speed_from_poses_car_length_scale_kmh']:.1f} km/h", f"{c15.TRUTH['impact_kmh']} km/h"])
    if mir:
        tt_.append(["Width predicted by symmetry (scaled by car length)", f"{out['predicted']['width_car_length_scale_m']:.2f} m", f"{TRUE_WIDTH} m (NHTSA)"])
    h["notes"]["truth_table"] = tt_
    sw.save("crash-9070", h, P, C, tag)
    (ROOT / "benchmark" / "crash_fusion.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
