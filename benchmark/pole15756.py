"""A third event: NHTSA test 15756, a 2025 Nissan Kicks carried sideways into a rigid pole at 31.09 km/h (FMVSS 214),
measured the hybrid way (physlens/hybrid.py) from the overhead high-speed camera (No. 10, 1,000 fps).

  plane   the car's roof, seen from above: a plane fitted to MoGe-3's points of the car nearest the camera
  track   roof features on the half of the car away from the pole (undamaged), every frame, on that plane
  lens    radial term from physics: the car slides at constant speed for the 50 ms before impact
  scale   the car's published width (1.789 m, NHTSA) against its width in MoGe-3's points (all heights,
          measured across the car on the roof plane)
  after   the car's sideways travel after impact, and its turn (yaw) from the tracked features
Frame 50 is impact (the burned-in clock reads 0 ms). Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/pole15756.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens.geometry.client import sharp_depth   # noqa: E402
from physlens.hybrid import along_from_pairs, fit_distortion, plane_point, radial, track_part   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402

VIDEO = ROOT / "datasets" / "nhtsa" / "15756" / "cam10_overhead_wide.avi"
CACHE = ROOT / "records" / "real_cache" / "pole15756"
IMPACT = 50
TRUTH = {"speed_kmh": 31.09, "length_m": 4.354, "width_m": 1.789, "mass_kg": 1479, "source": "NHTSA test 15756"}


def frames():
    cap = cv2.VideoCapture(str(VIDEO))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(f)
    return out


def main():
    from physlens.perception.sam_image import segment
    CACHE.mkdir(parents=True, exist_ok=True)
    F = frames()
    f = CACHE / "moge0.npz"
    if f.exists():
        z = np.load(f)
        D, Mv, K = z["d"].astype(np.float32), z["m"], z["K"]
    else:
        s = sharp_depth([cv2.resize(F[0], (1280, round(F[0].shape[0] * 1280 / F[0].shape[1])), interpolation=cv2.INTER_AREA)])[0]
        D, Mv, K = s.depth, s.mask, s.intrinsics
        np.savez_compressed(f, d=D.astype(np.float16), m=Mv, K=K)
    h, w = D.shape
    gray = [cv2.cvtColor(cv2.resize(x, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY) for x in F]
    img0 = cv2.resize(F[0], (w, h), interpolation=cv2.INTER_AREA)
    car = max([i.mask for i in segment(img0, "car") if i.mask.sum() > 2000], key=lambda m: m.sum())
    ys, xs = np.mgrid[0:h, 0:w]
    P = np.dstack([(xs - K[0, 2]) * D / K[0, 0], (ys - K[1, 2]) * D / K[1, 1], D])
    inner = (cv2.erode(car.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) & Mv & np.isfinite(D)
    C = P[inner]
    # The roof: the car's points nearest the camera, a plane facing it.
    near = C[C[:, 2] < np.percentile(C[:, 2], 35)]
    n, c, _ = _plane_ransac(near[::3], 0.01 * float(np.median(D[inner])))
    if n[2] > 0:
        n = -n                                            # facing the camera (towards −z)
    # The car's long axis in the image (its mask), carried onto the roof plane.
    pix = np.argwhere(car)[:, ::-1].astype(float)
    u = np.linalg.svd(pix - pix.mean(0), full_matrices=False)[2][0]
    a3 = plane_point(np.array([pix.mean(0), pix.mean(0) + 100 * u]), K, n, c)
    ax = a3[1] - a3[0]
    ax /= np.linalg.norm(ax)
    side = np.cross(n, ax)
    side /= np.linalg.norm(side)
    # Width and length in MoGe-3's units: all the car's points, measured across / along it on the roof plane.
    W_ai = float(np.subtract(*np.percentile(C @ side, [99, 1])))
    L_ai = float(np.subtract(*np.percentile(C @ ax, [99, 1])))
    k = TRUTH["width_m"] / W_ai
    # Roof features on the half away from the pole (the pole is left in this view): right of the car's centre.
    cx = pix[:, 0].mean()
    region0 = (cv2.erode(car.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0) & (xs > cx)

    def region_at(i, sh):
        M = np.float32([[1, 0, sh[0]], [0, 1, sh[1]]])
        return cv2.warpAffine(region0.astype(np.uint8), M, (w, h), flags=cv2.INTER_NEAREST) > 0

    t = (np.arange(len(gray)) - IMPACT) / 1000.0
    r = track_part(gray, region_at, 0, len(gray) - 1, K, n, c, side)
    # The direction of travel: the median plane displacement over the approach (it slides at 75° to its axis).
    steps = []
    for i, pr in enumerate(r["pairs"]):
        if pr is not None and t[i + 1] < -0.003:
            X0, X1 = plane_point(pr[0], K, n, c), plane_point(pr[1], K, n, c)
            steps.append(np.median(X1 - X0, axis=0))
    d = np.sum(steps, axis=0)
    d -= (d @ n) * n
    d /= np.linalg.norm(d)
    steady = (t >= -0.048) & (t <= -0.003)
    lens = fit_distortion(r["pairs"], K, n, c, d, t, steady)
    lens.pop("n"), lens.pop("d")
    und = radial(K, lens["k1"])
    x = along_from_pairs(r["pairs"], K, n, c, d, und) * k
    x -= x[IMPACT]
    x_raw = along_from_pairs(r["pairs"], K, n, c, d) * k
    pre = (t >= -0.048) & (t <= -0.003)
    v = float(np.polyfit(t[pre], x[pre], 1)[0])
    v_raw = float(np.polyfit(t[pre], x_raw[pre], 1)[0])
    # After impact: travel towards the pole until it stops; and the turn from the features (similarity per step).
    after = t >= 0
    k_stop = int(np.argmax(np.where(after, x, -np.inf)))
    yaw = [0.0]
    for pr in r["pairs"]:
        if pr is None or len(pr[0]) < 8:
            yaw.append(yaw[-1])
            continue
        A, _ = cv2.estimateAffinePartial2D(pr[0], pr[1], method=cv2.RANSAC, ransacReprojThreshold=1.5)
        yaw.append(yaw[-1] + (float(np.degrees(np.arctan2(A[1, 0], A[0, 0]))) if A is not None else 0.0))
    yaw = np.array(yaw)
    yaw -= yaw[IMPACT]
    angle_to_axis = float(np.degrees(np.arccos(abs(d @ ax))))
    res = {"camera": "No. 10 overhead wide view, 1,000 fps", "features_median": int(np.median(r["features"])),
           "roof_plane_angle_to_camera_deg": round(float(np.degrees(np.arccos(abs(n @ [0, 0, 1])))), 1),
           "width_ai": round(W_ai, 3), "length_ai": round(L_ai, 3), "scale_from_width": round(k, 4),
           "length_from_width_scale_m": round(L_ai * k, 3), "true_length_m": TRUTH["length_m"],
           "travel_angle_to_car_axis_deg": round(angle_to_axis, 1), "lens": lens,
           "speed_kmh": round(v * 3.6, 2), "speed_kmh_without_lens_fix": round(v_raw * 3.6, 2), "truth_kmh": TRUTH["speed_kmh"],
           "travel_after_impact_m": round(float(x[k_stop]), 3), "stops_at_ms": round(float(t[k_stop] * 1000), 1),
           "yaw_at_300ms_deg": round(float(yaw[-1]), 1), "yaw_at_100ms_deg": round(float(yaw[IMPACT + 100]), 1),
           "curve": {"t_ms": (t * 1000).round(1).tolist(), "travel_m": x.round(4).tolist(), "yaw_deg": yaw.round(2).tolist()}}
    print(json.dumps({kk: vv for kk, vv in res.items() if kk != "curve"}, indent=1))
    (ROOT / "benchmark" / "pole15756.json").write_text(json.dumps(res, default=float))
    # The geometry behind it, for the 3D world (pole_world.py): camera frame of frame 0, MoGe-3 units.
    feat0 = np.argwhere(region0)[:, ::-1].mean(0)
    geom = {"K": K, "n": n, "c": c, "d": d, "axis": ax, "k": k, "depth0": D, "valid0": Mv, "car0": car,
            "feature_centre0": plane_point(feat0[None], K, n, c)[0], "lens_k1": lens["k1"], "size": (w, h)}
    return res, geom


if __name__ == "__main__":
    main()
