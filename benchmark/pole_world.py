"""NHTSA 15756 (Nissan Kicks sideways into a pole, 31.09 km/h) as a 3D world for the simulator.

The scene and the car's shape come from MoGe-3 (overhead camera, 1,000 fps; one lens for every keyframe, each
frame's depth fitted to the first's on the unmoving floor); SAM 3.1 video tracking marks the car in every keyframe.
The car's motion does NOT come from per-frame depth: the body (the car's points at the first frame) is moved by the
hybrid measurement (pole15756.py: roof features tracked through all 351 frames on the roof plane): its travel at
75° to its axis and its turn after impact. Scale: the car's published width. Frame 50 is impact; times in the world
are seconds from impact.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/pole_world.py
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
import pole15756 as pole   # noqa: E402
from physlens.geometry.client import sharp_depth   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402
from physlens.objsim import Body, coverage, heat   # noqa: E402
from physlens.pointworld import write_world   # noqa: E402
from physlens.sharp import main_depth_cluster   # noqa: E402
from physlens.vehicle3d import video_masks   # noqa: E402

IDX = list(range(0, 121, 5)) + list(range(130, 351, 10))
CACHE = pole.CACHE
NAME = "pole-15756"
TITLE = "NHTSA side pole test 15756: Nissan Kicks sideways into a pole (overhead camera, 1,000 fps)"
BODY, BODY_HEAT = 254, 253


def main():
    res, g = pole.main()
    K, n, c, d, k = g["K"], g["n"], g["c"], g["d"], g["k"]
    w, h = g["size"]
    F = pole.frames()
    kf = [F[i] for i in IDX]
    hs = int(round(kf[0].shape[0] * 960 / kf[0].shape[1] / 2) * 2)          # even: the video encoder needs it
    small = [cv2.resize(f, (960, hs), interpolation=cv2.INTER_AREA) for f in kf]
    ms = video_masks(small, "car", CACHE / "samvideo_car")
    ms = [None if m is None else cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0 for m in ms]
    fov = float(np.degrees(2 * np.arctan(w / 2 / K[0, 0])))
    f = CACHE / "moge_keyframes.npz"
    if f.exists():
        z = np.load(f)
        D = [z[f"d{j}"].astype(np.float32) for j in range(len(IDX))]
        M = [z[f"m{j}"] for j in range(len(IDX))]
    else:
        sd = sharp_depth([cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA) for x in kf], fov_x=[fov] * len(IDX))
        D, M = [s.depth for s in sd], [s.mask for s in sd]
        np.savez_compressed(f, **{f"d{j}": x.astype(np.float16) for j, x in enumerate(D)}, **{f"m{j}": m for j, m in enumerate(M)})
    # Each frame's depth fitted to frame 0's (the hybrid's reference) on the unmoving floor.
    static = np.ones((h, w), bool)
    for m in ms:
        if m is not None:
            static &= ~(cv2.dilate(m.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0)
    D0 = g["depth0"]
    for j in range(len(IDX)):
        ok = static & M[j] & g["valid0"] & np.isfinite(D[j])
        D[j] = D[j] * float(np.median(D0[ok] / D[j][ok]))
    ys, xs = np.mgrid[0:h, 0:w]
    lift = lambda dd: np.dstack([(xs - K[0, 2]) * dd / K[0, 0], (ys - K[1, 2]) * dd / K[1, 1], dd])
    # The floor: the largest plane of the unmoving scene, its normal towards the camera (overhead: up ≈ −z).
    P0 = lift(D0)
    S = P0[static & g["valid0"]][::6]
    fn, fc, _ = _plane_ransac(S, 0.01 * float(np.median(D0)))
    if fn @ (-fc) < 0:
        fn = -fn
    x_ax = d - (d @ fn) * fn
    x_ax /= np.linalg.norm(x_ax)
    z_ax = np.cross(x_ax, fn)
    R = np.stack([x_ax, fn, z_ax])                           # camera → ground (rows: ground axes); x = travel
    to_g = lambda Pc: ((Pc - fc) @ R.T) * k                   # metres, y up, origin on the floor
    rgb = [cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA)[..., ::-1] for x in kf]
    # Static world: frame 0 without the car, its place filled from the last keyframe (the car has moved and turned).
    last = len(IDX) - 1
    car0 = ms[0] if ms[0] is not None else g["car0"]
    keep0 = g["valid0"] & ~(cv2.dilate(car0.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0)
    fill = M[last] & car0 & ~(cv2.dilate(ms[last].astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) if ms[last] is not None else np.zeros_like(car0)
    Pl = lift(D[last])
    P_out = [to_g(P0[keep0]), to_g(Pl[fill])]
    C_out = [rgb[0][keep0], rgb[last][fill]]
    Ps = np.vstack(P_out)
    Cs = np.vstack(C_out)
    groups = [{"tag": 255, "start": 0, "count": int(len(Ps))}]
    tags = [np.full(len(Ps), 255, np.uint8)]
    P_all, C_all = [Ps], [Cs]
    start = len(Ps)
    boxes = []
    t_of = lambda fr: round((fr - pole.IMPACT) / 1000.0, 4)
    # The car as measured in each keyframe (raw frames).
    for j, fr in enumerate(IDX):
        if ms[j] is None:
            continue
        mm = (cv2.erode(ms[j].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & M[j]
        Pc = lift(D[j])[mm]
        keep = main_depth_cluster(Pc[:, 2] * k)
        Pg, Cg = to_g(Pc[keep]), rgb[j][mm][keep]
        P_all.append(Pg); C_all.append(Cg); tags.append(np.full(len(Pg), len(boxes), np.uint8))
        groups.append({"tag": len(boxes), "frame": fr, "t": t_of(fr), "start": start, "count": int(len(Pg))})
        start += len(Pg)
        H = Pg[:, [0, 2]] - Pg[:, [0, 2]].mean(0)
        u = np.linalg.svd(H, full_matrices=False)[2][0]
        a, b = H @ u, H @ np.array([-u[1], u[0]])
        boxes.append({"frame": fr, "t": t_of(fr), "points": int(len(Pg)),
                      "centre": [round(float(Pg[:, 0].mean()), 3), round(float(Pg[:, 1].max() / 2), 3), round(float(Pg[:, 2].mean()), 3)],
                      "size": [round(float(np.ptp(a)), 3), round(float(np.percentile(Pg[:, 1], 99)), 3), round(float(np.ptp(b)), 3)],
                      "yaw": round(float(np.arctan2(u[1], u[0])), 4)})
    # The body: the car's points at frame 0 (all of it in view), in its own coordinates (x along its axis).
    mm0 = (cv2.erode(car0.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0) & g["valid0"]
    Pc0 = P0[mm0]
    keep = main_depth_cluster(Pc0[:, 2] * k)
    B = to_g(Pc0[keep])
    Bc = rgb[0][mm0][keep]
    ax_g = (g["axis"] @ R.T)[[0, 2]]
    ax_g /= np.linalg.norm(ax_g)
    yaw0 = float(np.arctan2(ax_g[1], ax_g[0]))
    cen0 = np.array([B[:, 0].mean(), 0.0, B[:, 2].mean()])
    cy, sy = np.cos(yaw0), np.sin(yaw0)
    R0 = np.array([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])    # body → world (objsim convention)
    Bb = (B - cen0) @ R0                                       # world → body
    body = Body(Bb, Bc, np.ones(len(Bb), int), [], [], [], [])
    cov = coverage(body)
    # Poses from the hybrid measurement: the roof features' travel along the measured direction (x in this frame)
    # and their turn; the body turns about the features' centre.
    curve = res["curve"]
    tc = np.array(curve["t_ms"]) / 1000.0
    trav = np.array(curve["travel_m"])
    yaw_img = np.radians(np.array(curve["yaw_deg"]))
    fc0 = to_g(g["feature_centre0"][None])[0]
    trav0 = trav[0]

    def pose_at(i, sign):
        dy = sign * (yaw_img[i] - yaw_img[0])
        fci = fc0 + np.array([trav[i] - trav0, 0, 0])
        c2, s2 = np.cos(dy), np.sin(dy)
        Rd = np.array([[c2, 0, -s2], [0, 1, 0], [s2, 0, c2]])
        pos = fci + (cen0 - fc0) @ Rd.T
        return pos, yaw0 + dy

    # The turn's sign (image vs ground frame): the one that lays the body on the car as measured at the last keyframe.
    from scipy.spatial import cKDTree
    lastP = P_all[-1]
    tree = cKDTree(lastP[::5])
    best = None
    for sign in (1.0, -1.0):
        i = int(np.argmin(np.abs(tc - t_of(IDX[last]))))
        pos, yw = pose_at(i, sign)
        cyy, syy = np.cos(yw), np.sin(yw)
        Rw = np.array([[cyy, 0, -syy], [0, 1, 0], [syy, 0, cyy]])
        Q = Bb[::20] @ Rw.T + pos
        dist, _ = tree.query(Q)
        sc = float(np.median(dist))
        if best is None or sc < best[0]:
            best = (sc, sign)
    sign = best[1]
    poses = []
    for i in range(0, len(tc), 5):
        pos, yw = pose_at(i, sign)
        poses.append({"frame": int(i), "t": round(float(tc[i]), 4), "used": True, "x": round(float(pos[0]), 4),
                      "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4), "yaw": round(float(yw), 5)})
    # Speed along the way (local straight-line fits over ±5 ms).
    sp_t, sp_v, sp_e = [], [], []
    for i in range(5, len(tc) - 5, 5):
        tt, xx = tc[i - 5:i + 6], trav[i - 5:i + 6]
        A = np.vstack([tt, np.ones(len(tt))]).T
        cf, *_ = np.linalg.lstsq(A, xx, rcond=None)
        r = xx - A @ cf
        se = float(np.sqrt((r @ r) / max(len(tt) - 2, 1) / np.sum((tt - tt.mean()) ** 2)))
        sp_t.append(round(float(tc[i]), 4)); sp_v.append(round(float(cf[0]) * 3.6, 2)); sp_e.append(round(se * 3.6, 2))
    hb = heat(body.seen)
    P_all += [Bb, Bb]; C_all += [Bc, hb]
    tags += [np.full(len(Bb), BODY, np.uint8), np.full(len(Bb), BODY_HEAT, np.uint8)]
    groups += [{"tag": BODY, "start": start, "count": int(len(Bb))}, {"tag": BODY_HEAT, "start": start + len(Bb), "count": int(len(Bb))}]
    T = pole.TRUTH
    events = [
        {"t": -0.05, "kind": "cruise", "label": f"Sliding sideways at {res['speed_kmh']:.1f} km/h",
         "detail": f"roof features tracked at 1,000 fps, at {res['travel_angle_to_car_axis_deg']:.1f}° to the car's axis",
         "truth": f"NHTSA: {T['speed_kmh']} km/h at 75°"},
        {"t": 0.0, "kind": "impact", "label": "Driver's side hits the pole", "detail": "the camera's clock reads 0 ms",
         "window": [0.0, 0.001]},
        {"t": round(res["stops_at_ms"] / 1000, 4), "kind": "stop",
         "label": f"Stops pushing into the pole: {res['travel_after_impact_m']:.2f} m after impact",
         "detail": "its sideways travel peaks, then it rebounds"},
        {"t": 0.3, "kind": "rebound", "label": f"Turned {abs(res['yaw_at_300ms_deg']):.0f}° by 300 ms",
         "detail": "the rear swings round the pole (turn of the tracked roof features)"},
    ]
    truth = [["Speed at impact (hybrid, 1,000 fps)", f"{res['speed_kmh']:.2f} km/h", f"{T['speed_kmh']} km/h (NHTSA)"],
             ["Direction of travel vs the car's axis", f"{res['travel_angle_to_car_axis_deg']:.1f}°", "75° (the test's impact angle)"],
             ["Car length (scaled by its width)", f"{res['length_from_width_scale_m']:.2f} m", f"{T['length_m']} m"],
             ["Travel after impact", f"{res['travel_after_impact_m']:.2f} m, stops at {res['stops_at_ms']:.0f} ms", "(accelerometers not compared yet)"]]
    notes = {"events": events, "truth_table": truth, "speed_curve": {"t": sp_t, "kmh": sp_v, "err_kmh": sp_e},
             "body": {"poses": poses, "rigid_frames": [0, 350], "rigid_until_t": 0.3, "after_rigid": None,
                      "points": int(len(Bb)), "frames_used": 351, "frames": 351, "passes": [], "coverage": cov,
                      "seen_max": 1, "seen_median": 1.0,
                      "how": "the car's points at the first frame, moved by the hybrid measurement (roof features "
                             "tracked through all 351 frames on the roof plane): travel and turn"},
             "scale_from": "the car's published width (1.789 m)",
             "credit": "Video: NHTSA side pole test 15756, camera No. 10 overhead wide view (public)."}
    cam_g = to_g(np.zeros((1, 3)))[0]
    camera = {"position": cam_g.round(3).tolist(), "forward": (np.array([0, 0, 1.0]) @ R.T).round(4).tolist(),
              "fov_deg": round(float(np.degrees(2 * np.arctan(h / 2 / K[1, 1]))), 1), "moving": False}
    meta = {"video": pole.VIDEO.name, "fps": 1000.0, "upsample": 1, "depth_source": "MoGe-3 (overhead camera)",
            "density_note": "one measured point per video pixel (MoGe-3 depth); the car's motion from 2D tracking on its roof plane",
            "scale_from": "the car's published width (1.789 m)"}
    hdr = write_world(NAME, TITLE, P_all, C_all, tags, groups, boxes, ROOT / "media" / "pointworld", "Nissan Kicks", camera, meta, notes)
    print(NAME, hdr["points"], "points;", len(boxes), "moments; turn sign", sign, "fit", round(best[0], 3), "m")


if __name__ == "__main__":
    main()
