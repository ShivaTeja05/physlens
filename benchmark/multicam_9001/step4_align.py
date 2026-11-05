"""Align cameras in the floor plane: each camera scaled so the Camry's roof is 1.47 m (Toyota's published height),
then yaw + floor shift (3 DOF, floors already level) found by ICP from a full grid of starting yaws, pairwise.
Prints a pair table: inlier share within 4 cm and RMS, best vs second-best yaw (to spot symmetric mismatches)."""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "out"
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10
CAMS = ["cam01_overhead_wide", "cam02", "cam05_rear_impact", "cam06_front_oblique"]
ROOF = 1.47
rng = np.random.default_rng(0)


def load(cam):
    z = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    car = z["car"].astype(float)
    s = ROOF / np.quantile(car[:, 2], 0.98)
    car = car * s
    car = car[(car[:, 2] > 0.25) & (car[:, 2] < 1.6)]
    return car[rng.choice(len(car), min(len(car), 8000), replace=False)], s


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def icp(src, tree, tgt, t0, iters=40, tol=0.04):
    th, tx = t0, np.zeros(3)
    # start by matching centroids at this yaw
    tx[:2] = tgt[:, :2].mean(0) - (src @ rot(th).T)[:, :2].mean(0)
    for _ in range(iters):
        p = src @ rot(th).T + tx
        d, j = tree.query(p, distance_upper_bound=0.5)
        m = np.isfinite(d)
        if m.sum() < 50:
            break
        a, b = p[m][:, :2], tgt[j[m]][:, :2]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        dth = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        th += dth
        R2 = rot(dth)[:2, :2]
        tx[:2] = R2 @ (tx[:2] - ca) + cb if False else (R2 @ tx[:2]) + (cb - R2 @ ca)
    p = src @ rot(th).T + tx
    d, _ = tree.query(p)
    return th, tx, float(np.mean(d < tol)), float(np.sqrt(np.mean(np.minimum(d, 0.3) ** 2)))


clouds = {c: load(c) for c in CAMS}
print({c: round(s, 3) for c, (_, s) in clouds.items()}, "(scale applied to MoGe depth)")
pairs = {}
for a, b in combinations(CAMS, 2):
    src, tgt = clouds[a][0], clouds[b][0]
    tree = cKDTree(tgt)
    runs = sorted((icp(src, tree, tgt, t0) for t0 in np.radians(np.arange(0, 360, 15))), key=lambda r: -r[2])
    best = runs[0]
    second = next((r for r in runs[1:] if abs(((r[0] - best[0] + np.pi) % (2 * np.pi)) - np.pi) > np.radians(20)), None)
    pairs[f"{a}->{b}"] = {"yaw_deg": round(float(np.degrees(best[0]) % 360), 1), "shift": np.round(best[1][:2], 3).tolist(),
                         "inliers": round(best[2], 3), "rms": round(best[3], 3),
                         "second_inliers": round(second[2], 3) if second else None,
                         "second_yaw": round(float(np.degrees(second[0]) % 360), 1) if second else None}
    print(f"{a[:14]:14s} -> {b[:14]:14s} yaw {pairs[f'{a}->{b}']['yaw_deg']:6.1f}  in4cm {best[2]:.2f}  rms {best[3]:.3f}"
          f"   2nd: yaw {pairs[f'{a}->{b}']['second_yaw']} in {pairs[f'{a}->{b}']['second_inliers']}")
json.dump(pairs, open(OUT / f"pairs_f{FRAME}.json", "w"), indent=1)
