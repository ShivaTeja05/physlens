"""Motion measured in 2D, made metric in 3D.

Per-frame 3D (one depth estimate per frame) jitters by centimetres, more than a crash moves between 1 ms frames.
Image tracking doesn't: features on an object followed from frame to frame land within a fraction of a pixel. So:
  1. features on a rigid part of the object (say, the car's rear door) are followed in the image through every
     frame (pyramidal Lucas–Kanade, forward–backward checked, re-seeded inside the part as they're lost)
  2. the 3D world gives what it measures well: the lens and the plane the part moves in (the car's side: a
     vertical plane along its path). Each feature's ray meets that plane at a point in metres, so its move along
     the path comes out in metres, with no per-frame depth involved
  3. the frame-to-frame move is the median over features seen in both frames, summed up: re-seeding doesn't jump
  4. lens distortion (one radial term) can be found from physics instead of a calibration target: over a stretch
     where the object is known to move at constant speed (a car rolling into a barrier), the right correction is
     the one that makes its measured speed constant (`fit_distortion`)
The plane and lens come from the AI depth, so the result carries its scale; a known length (an identified
model's, or a measured reference) corrects it, and both are reported.
"""
from __future__ import annotations

import cv2
import numpy as np


def plane_point(px: np.ndarray, K: np.ndarray, n: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Where the rays through pixels `px` (N, 2) meet the plane through `c` with normal `n` (camera frame)."""
    rays = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(px))]
    s = (c @ n) / (rays @ n)
    return rays * s[:, None]


def fit_side_plane(P: np.ndarray, d: np.ndarray, iters: int = 300, thresh: float = 0.03, seed: int = 0):
    """A vertical-ish plane through points P (camera frame) containing direction d (camera frame): the side of a
    car moving along d. RANSAC over pairs (the plane holds d), refit on the inliers. Returns (normal, point)."""
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        a, b = P[rng.choice(len(P), 2, replace=False)]
        n = np.cross(d, b - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        k = int((np.abs((P - a) @ n) < thresh).sum())
        if k > best_n:
            best, best_n = (n, a), k
    n, a = best
    inl = P[np.abs((P - a) @ n) < thresh]
    c = inl.mean(0)
    # Refit: normal ⟂ d, best fitting the inliers.
    Q = inl - c
    Q = Q - np.outer(Q @ d, d)
    n = np.linalg.svd(Q, full_matrices=False)[2][-1]
    n = n - (n @ d) * d
    n /= np.linalg.norm(n)
    return n, c


def track_part(gray: list[np.ndarray], region_at, start: int, end: int, K: np.ndarray, n: np.ndarray, c: np.ndarray,
               d: np.ndarray, max_pts: int = 300, min_pts: int = 60, fb_px: float = 0.7, scale: float = 1.0,
               undistort=None) -> dict:
    """Follow a rigid part from frame `start` to `end` (indices into `gray`). `region_at(i, shift_px)` gives the
    part's mask in frame i (it may use the part's accumulated image shift to move a seed region along). Returns
    per frame: the part's move along `d` in metres (on the plane), the features used and the image shift."""
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    undistort = undistort or (lambda p: p)
    pairs: list = []
    along = [0.0]
    used = [0]
    shift = np.zeros(2)
    shifts = [shift.copy()]

    def seed(i, sh):
        m = region_at(i, sh)
        p = cv2.goodFeaturesToTrack(gray[i], maxCorners=max_pts, qualityLevel=0.01, minDistance=5,
                                    mask=m.astype(np.uint8) * 255)
        return np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)

    pts = seed(start, shift)
    for i in range(start + 1, end + 1):
        if len(pts) < min_pts:
            pts = np.concatenate([pts, seed(i - 1, shift)]) if len(pts) else seed(i - 1, shift)
        if not len(pts):
            along.append(along[-1]); used.append(0); shifts.append(shift.copy())
            pairs.append(None)
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[i - 1], gray[i], pts, None, **lk)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[i], gray[i - 1], nxt, None, **lk)
        ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < fb_px)
        p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
        if len(p0) < 8:
            along.append(along[-1]); used.append(int(len(p0))); shifts.append(shift.copy())
            pairs.append(None)
            pts = np.zeros((0, 1, 2), np.float32)
            continue
        pairs.append((p0, p1))
        X0, X1 = plane_point(undistort(p0), K, n, c), plane_point(undistort(p1), K, n, c)
        step = float(np.median((X1 - X0) @ d)) * scale
        along.append(along[-1] + step)
        used.append(int(len(p0)))
        shift = shift + np.median(p1 - p0, axis=0)
        shifts.append(shift.copy())
        # Keep only features still on the part (a crushing front, a background point the window slid onto).
        m = region_at(i, shift)
        inside = m[np.clip(p1[:, 1].astype(int), 0, m.shape[0] - 1), np.clip(p1[:, 0].astype(int), 0, m.shape[1] - 1)]
        pts = p1[inside].reshape(-1, 1, 2).astype(np.float32)
    return {"along_m": np.array(along), "features": np.array(used), "shift_px": np.array(shifts), "pairs": pairs}


def radial(K: np.ndarray, k1: float):
    """Undistortion with one radial term, r measured in focal lengths: p_u = c + (p − c)(1 + k1 r²)."""
    cx, cy, f = K[0, 2], K[1, 2], K[0, 0]

    def u(p):
        q = (p - [cx, cy]) / f
        r2 = (q ** 2).sum(1, keepdims=True)
        return q * (1 + k1 * r2) * f + [cx, cy]
    return u


def along_from_pairs(pairs, K, n, c, d, undistort=None) -> np.ndarray:
    """Re-run step 3 on tracked feature pairs (no re-tracking): the move along d per frame, summed."""
    undistort = undistort or (lambda p: p)
    along = [0.0]
    for pr in pairs:
        if pr is None:
            along.append(along[-1])
            continue
        p0, p1 = pr
        X0, X1 = plane_point(undistort(p0), K, n, c), plane_point(undistort(p1), K, n, c)
        along.append(along[-1] + float(np.median((X1 - X0) @ d)))
    return np.array(along)


def _turn(v: np.ndarray, up: np.ndarray, a: float) -> np.ndarray:
    """v turned by angle a about the unit axis up (Rodrigues)."""
    return v * np.cos(a) + np.cross(up, v) * np.sin(a) + up * (up @ v) * (1 - np.cos(a))


def fit_distortion(pairs, K, n, c, d, t: np.ndarray, steady: np.ndarray, up: np.ndarray | None = None,
                   k1s=np.linspace(-0.4, 0.4, 17), turns_deg=np.arange(-30, 30.1, 1.0)) -> dict:
    """The radial term (and, given `up`, the side plane's turn about the vertical) that make the motion over
    `steady` (frames known to be at constant speed) most uniform: smallest residual of a straight line through
    position against time, relative to the distance covered. A plane turned the wrong way, like a distorting
    lens, makes equal steps look unequal across the picture."""
    best = None
    for k1 in k1s:
        und = radial(K, k1)
        for a in (turns_deg if up is not None else [0.0]):
            nn, dd = (_turn(n, up, np.radians(a)), _turn(d, up, np.radians(a))) if up is not None else (n, d)
            x = along_from_pairs(pairs, K, nn, c, dd, und)
            tt, xx = t[steady], x[steady]
            A = np.vstack([tt, np.ones(len(tt))]).T
            coef, *_ = np.linalg.lstsq(A, xx, rcond=None)
            res = xx - A @ coef
            score = float(np.sqrt(np.mean(res ** 2)) / max(abs(xx[-1] - xx[0]), 1e-9))
            if best is None or score < best[0]:
                best = (score, float(k1), float(a), nn, dd)
    return {"k1": best[1], "plane_turn_deg": best[2], "relative_residual": round(best[0], 5), "n": best[3], "d": best[4]}
