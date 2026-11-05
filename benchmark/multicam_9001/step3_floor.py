"""Per camera, from MoGe-3 depth only: the floor plane (RANSAC on non-car points, with the car on its upper side),
a gravity-aligned floor frame (z up, floor at z=0), and the Camry's points in it. Prints the car's apparent
height (MoGe scale) so each camera's scale error shows. Saves out/f{FRAME}_floor_{cam}.npz."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "out"
CAMS = ["cam01_overhead_wide", "cam02", "cam04_struck_side", "cam05_rear_impact", "cam06_front_oblique"]
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10
rng = np.random.default_rng(0)
STRIDE = 3


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth
    return np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1)


def floor_plane(bg, car, up, iters=6000, tol_rel=0.006):
    """Largest plane among background points with >=90% of car points on one side and the car rising above it,
    its normal within 45° of the camera's rough 'up' (side cameras are mounted level, overhead ones look down)."""
    best = None
    for _ in range(iters):
        a, b, c = bg[rng.choice(len(bg), 3, replace=False)]
        n = np.cross(b - a, c - a)
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = n @ a
        hc = car @ n - d
        if np.mean(hc > 0) < 0.5:
            n, d, hc = -n, -d, -hc
        if n @ up < np.cos(np.radians(45)):
            continue
        if np.mean(hc > -0.02 * np.median(np.abs(bg[:, 2]))) < 0.9:
            continue
        tol = tol_rel * np.median(bg[:, 2])
        k = int((np.abs(bg @ n - d) < tol).sum())
        if best is None or k > best[2]:
            best = (n, d, k, tol)
    n, d, k, tol = best
    q = bg[np.abs(bg @ n - d) < tol]
    c0 = q.mean(0)
    _, _, vt = np.linalg.svd(q - c0)
    n2 = vt[-1] if vt[-1] @ n > 0 else -vt[-1]
    return n2, float(n2 @ c0), k / len(bg)


summary = {}
for cam in CAMS:
    z = np.load(OUT / f"f{FRAME}_in_{cam}.npz")
    P = backproject(z["depth"], z["K"])
    valid = z["valid"] & (z["depth"] > 0)
    car = z["car"]
    near_car = cv2.dilate(car.astype(np.uint8), np.ones((31, 31), np.uint8)) > 0
    sl = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    Pv, vv, cv_, nc = P[sl], valid[sl], car[sl], near_car[sl]
    bg = Pv[vv & ~nc]
    carp = Pv[vv & cv_]
    bgs = bg[rng.choice(len(bg), min(len(bg), 40000), replace=False)]
    up = np.array([0, 0, -1.0]) if cam in ("cam01_overhead_wide", "cam02") else np.array([0, -1.0, 0])
    n, d, share = floor_plane(bgs, carp, up)
    # Floor frame: z = n (up), x = camera forward projected on the floor, origin = camera foot on the floor.
    fwd = np.array([0, 0, 1.0]) - n * n[2]
    if np.linalg.norm(fwd) < 0.2:            # looking straight down: use the image's up direction instead
        fwd = np.array([0, -1.0, 0]) - n * (-n[1])
    x = fwd / np.linalg.norm(fwd)
    y = np.cross(n, x)
    Rw = np.stack([x, y, n])                 # rows: camera -> floor frame rotation
    origin = d * n                           # foot of the camera on the floor plane (camera at 0)
    to_floor = lambda p: (p - origin) @ Rw.T   # noqa: E731
    car_f = to_floor(carp)
    h98 = float(np.quantile(car_f[:, 2], 0.98))
    # The camera sits at the origin of its own frame; its height above the floor is -(n · origin) = -d.
    summary[cam] = {"floor_share": round(share, 3), "camera_height": round(float(-d), 3),
                    "tilt_from_up_deg": round(float(np.degrees(np.arccos(np.clip(n @ up, -1, 1)))), 1),
                    "car_h98": round(h98, 3), "car_points": int(len(car_f)),
                    "car_below_floor_pct": round(float(np.mean(car_f[:, 2] < -0.05) * 100), 1)}
    allf = to_floor(Pv[vv])
    cols = z["img"][sl][vv]
    np.savez_compressed(OUT / f"f{FRAME}_floor_{cam}.npz", car=car_f.astype(np.float32), all=allf.astype(np.float32),
                        rgb=cols, iscar=cv_[vv], R=Rw, origin=origin, n=n, d=d)
print(json.dumps(summary, indent=1))
