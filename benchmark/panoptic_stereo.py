"""Dense, measured 3D of CMU Panoptic 161029_sports1 by classic stereo between neighbouring calibrated HD cameras:
each pair is rectified with Panoptic's own calibration (lens distortion removed in the same step), matched by
semi-global block matching (OpenCV SGBM) along the rectified lines, and triangulated with the exact camera matrices.
No learned depth: every point is two calibrated rays meeting. Pairs one above the other are matched on transposed
images (their epipolar lines are columns).
Usage: backend/.venv/bin/python benchmark/panoptic_stereo.py [frame ...]   (CPU only; default: frame 30)
Writes datasets/panoptic/161029_sports1/stereo/f{frame}.npz (points in the Panoptic world frame, metres, y down).
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panoptic_ball import D, cameras  # noqa: E402
from panoptic_ball_sam import frame  # noqa: E402

HOLDOUT = "00_15"                              # never used for stereo: it checks the result independently


def pick_pairs(cams, max_base=1.3, max_angle=32.0):
    """Neighbouring cameras (the dome's HD cameras sit ~1.25 m apart, one per panel), each camera in at most 2 pairs,
    gentlest angle first."""
    import itertools
    names = [n for n in cams if n != HOLDOUT]
    cand = []
    for a, b in itertools.combinations(names, 2):
        base = np.linalg.norm(cams[a]["C"] - cams[b]["C"])
        ang = np.degrees(np.arccos(np.clip(cams[a]["R"][2] @ cams[b]["R"][2], -1, 1)))
        if base < max_base and ang < max_angle:
            cand.append((ang, a, b))
    used, pairs = {}, []
    for ang, a, b in sorted(cand):
        if used.get(a, 0) < 2 and used.get(b, 0) < 2:
            pairs.append((a, b)); used[a] = used.get(a, 0) + 1; used[b] = used.get(b, 0) + 1
    return pairs


PAIRS: list = []
SCALE = 0.5                                    # match at half resolution (960x540)
OUT = D / "stereo"
OUT.mkdir(exist_ok=True)


def inside_volume(n=4000, seed=0):
    """Sample points of the capture volume (people and floor), to set each pair's disparity search range."""
    rng = np.random.default_rng(seed)
    r, a = 2.4 * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    return np.column_stack([r * np.cos(a), -rng.random(n) * 2.4, r * np.sin(a)])


def pair_points(cams, a, b, i):
    A, B = cams[a], cams[b]
    size = (1920, 1080)
    R = B["R"] @ A["R"].T
    T = B["t"] - R @ A["t"]
    # flags=0: the rectified principal points may differ, which keeps disparities small for converging cameras
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(A["K"], A["dist"], B["K"], B["dist"], size, R, T, alpha=1, flags=0)
    m1 = cv2.initUndistortRectifyMap(A["K"], A["dist"], R1, P1, size, cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(B["K"], B["dist"], R2, P2, size, cv2.CV_32FC1)
    g1 = cv2.remap(cv2.cvtColor(frame(a, i), cv2.COLOR_BGR2GRAY), *m1, cv2.INTER_LINEAR)
    g2 = cv2.remap(cv2.cvtColor(frame(b, i), cv2.COLOR_BGR2GRAY), *m2, cv2.INTER_LINEAR)
    c1 = cv2.remap(frame(a, i), *m1, cv2.INTER_LINEAR)
    vertical = abs(P2[1, 3]) > abs(P2[0, 3])
    S = np.diag([SCALE, SCALE, 1.0])
    P1s, P2s = S @ P1, S @ P2
    g1, g2, c1 = (cv2.resize(x, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA) for x in (g1, g2, c1))
    if vertical:
        g1, g2, c1 = g1.T.copy(), g2.T.copy(), np.ascontiguousarray(c1.transpose(1, 0, 2))
    # which image is "left": its features sit at larger x than in the other (disparity = x_left - x_right > 0)
    first_is_left = (P2[1, 3] if vertical else P2[0, 3]) < 0
    L, Rr = (g1, g2) if first_is_left else (g2, g1)
    # disparity range from the capture volume projected into both rectified views
    X = inside_volume()
    xr = []
    for P, Rk, C in ((P1s, R1, A), (P2s, R2, B)):
        Xc = (A["R"] @ X.T + A["t"]).T                                   # world -> cam a (rect frames are both built off cam a)
        Xr = (R1 @ Xc.T).T
        p = (P @ np.column_stack([Xr, np.ones(len(Xr))]).T).T
        xr.append((p[:, 1] if vertical else p[:, 0]) / p[:, 2])
    d = (xr[0] - xr[1]) if first_is_left else (xr[1] - xr[0])
    lo, hi = np.percentile(d, [1, 99])
    mind = int(np.floor(lo / 16) * 16) - 16
    num = int(min(np.ceil((hi - mind) / 16) * 16 + 16, 512))
    if num < 16 or mind + num >= L.shape[1] or mind <= -L.shape[1]:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"skipped": f"disparity range {lo:.0f}..{hi:.0f}"}
    sgbm = cv2.StereoSGBM_create(minDisparity=mind, numDisparities=num, blockSize=5, P1=8 * 25, P2=32 * 25,
                                 disp12MaxDiff=1, uniquenessRatio=12, speckleWindowSize=120, speckleRange=2,
                                 mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(L, Rr).astype(np.float32) / 16.0
    valid = disp > mind
    ys, xs = np.nonzero(valid)
    if len(xs) == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), {"matched_pct": 0.0, "points": 0}
    dd = disp[ys, xs]
    # pixel in the left image and its match in the right one, back in (row, col) of the untransposed pictures
    xl, xrr = xs.astype(np.float64), xs - dd
    if vertical:
        pl = np.stack([ys, xl], 1)          # transposed: x is the original row direction
        pr = np.stack([ys, xrr], 1)
        pl, pr = pl.astype(np.float64), pr.astype(np.float64)
    else:
        pl = np.stack([xl, ys], 1).astype(np.float64)
        pr = np.stack([xrr, ys], 1).astype(np.float64)
    PL, PR = (P1s, P2s) if first_is_left else (P2s, P1s)
    Xh = cv2.triangulatePoints(PL, PR, pl.T, pr.T)
    Xrect = (Xh[:3] / Xh[3]).T
    Xcam = (R1.T @ Xrect.T).T
    Xw = (A["R"].T @ (Xcam.T - A["t"])).T
    if first_is_left:
        cols = (c1[ys, xs] if not vertical else c1[ys, xs])
    else:                                   # colour from the second view's rectified image is not computed; use grey
        cols = np.repeat(L[ys, xs][:, None], 3, 1)
    # keep the capture volume and a bit of the dome
    r, h = np.hypot(Xw[:, 0], Xw[:, 2]), -Xw[:, 1]
    keep = (r < 3.2) & (h > -0.2) & (h < 3.5)
    return Xw[keep].astype(np.float32), cols[keep][:, ::-1].astype(np.uint8), {"disp_range": [mind, mind + num], "vertical": bool(vertical),
                                                                               "matched_pct": round(100 * valid.mean(), 1), "points": int(keep.sum())}


if __name__ == "__main__":
    cams = cameras()
    PAIRS[:] = pick_pairs(cams)
    print(f"{len(PAIRS)} stereo pairs from {len(cams)} cameras (held out: {HOLDOUT})")
    for i in [int(x) for x in sys.argv[1:]] or [30]:
        pts, rgb, src = [], [], []
        for k, (a, b) in enumerate(PAIRS):
            X, C, info = pair_points(cams, a, b, i)
            print(f"frame {i} pair {a}-{b}: {info}")
            pts.append(X); rgb.append(C); src.append(np.full(len(X), k, np.uint8))
        np.savez_compressed(OUT / f"f{i:03d}.npz", pts=np.concatenate(pts), rgb=np.concatenate(rgb), pair=np.concatenate(src))
