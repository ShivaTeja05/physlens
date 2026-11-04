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


def plane_point(px: np.ndarray, K: np.ndarray, n: np.ndarray, c: np.ndarray, offset=0.0) -> np.ndarray:
    """Where the rays through pixels `px` (N, 2) meet the plane through `c` with unit normal `n` (camera frame),
    moved `offset` along n (a number, or one per pixel: points of a body at different heights)."""
    rays = np.c_[(px[:, 0] - K[0, 2]) / K[0, 0], (px[:, 1] - K[1, 2]) / K[1, 1], np.ones(len(px))]
    s = (c @ n + offset) / (rays @ n)
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
               undistort=None, height_at=None) -> dict:
    """Follow a rigid part from frame `start` to `end` (indices into `gray`). `region_at(i, shift_px)` gives the
    part's mask in frame i (it may use the part's accumulated image shift to move a seed region along). With
    `height_at(i, shift_px)` (a per-pixel map of heights along n above the plane), each feature moves in its own
    plane at its own height, so a part spanning heights (a cab's roof and windscreen) isn't measured on one.
    Returns per frame: the part's move along `d` in metres, the features used and the image shift."""
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
        off = 0.0
        if height_at is not None:
            hm = height_at(i - 1, shift)
            off = hm[np.clip(p0[:, 1].astype(int), 0, hm.shape[0] - 1), np.clip(p0[:, 0].astype(int), 0, hm.shape[1] - 1)]
            keep = np.isfinite(off)
            p0, p1, off = p0[keep], p1[keep], off[keep]
            if len(p0) < 8:
                along.append(along[-1]); used.append(int(len(p0))); shifts.append(shift.copy())
                pairs.append(None)
                pts = np.zeros((0, 1, 2), np.float32)
                continue
        pairs.append((p0, p1))
        X0, X1 = plane_point(undistort(p0), K, n, c, off), plane_point(undistort(p1), K, n, c, off)
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


def lens_from_lines(gray: np.ndarray, K: np.ndarray, mask: np.ndarray, min_len: int | None = None,
                    k1s=np.round(np.arange(-0.4, 0.41, 0.02), 2), segments=None) -> dict:
    """The radial term (as `radial`) that makes the scene's long straight lines straight: floor seams, slab edges.
    Segments are found in `mask` (the floor, away from anything moving or curved), each refined to sub-pixel edge
    points across its length, then each line's own best term is found; the median over lines is returned, so one
    line that isn't really straight can't drag it."""
    g = cv2.GaussianBlur(gray.astype(np.float32), (5, 5), 0)
    gx, gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=5), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=5)
    e = cv2.Canny(g.astype(np.uint8), 40, 120) * (mask > 0)
    min_len = min_len or int(0.2 * min(gray.shape))
    segs = cv2.HoughLinesP(e.astype(np.uint8), 1, np.pi / 720, 80, minLineLength=min_len, maxLineGap=12)
    if segments is not None:                                             # rough segments given (hand-picked)
        segs = np.asarray(segments, float).reshape(-1, 4)
    segs = [] if segs is None else segs.reshape(-1, 4).astype(float)
    # One per physical line: drop segments nearly collinear with a longer one already kept.
    segs = sorted(segs, key=lambda s: -np.hypot(s[2] - s[0], s[3] - s[1]))
    kept = []
    for s in segs:
        u = np.array([s[2] - s[0], s[3] - s[1]]) / np.hypot(s[2] - s[0], s[3] - s[1])
        if any(abs(u @ ku) > 0.995 and abs((s[:2] - ks[:2]) @ np.array([-ku[1], ku[0]])) < 6 for ks, ku in kept):
            continue
        kept.append((s, u))
    h, w = gray.shape
    band = 12 if segments is not None else 6                            # hand-picked ends are rougher
    lines = []
    for s, u in kept:
        nrm = np.array([-u[1], u[0]])
        L = np.hypot(s[2] - s[0], s[3] - s[1])
        pts = []
        for a in np.arange(0, L, 3.0):
            p = s[:2] + a * u
            offs = np.arange(-band, band + 1)
            q = p[None] + offs[:, None] * nrm[None]
            xi, yi = np.clip(q[:, 0].round().astype(int), 0, w - 1), np.clip(q[:, 1].round().astype(int), 0, h - 1)
            prof = np.abs(gx[yi, xi] * nrm[0] + gy[yi, xi] * nrm[1])
            j = int(np.argmax(prof))
            if 0 < j < len(prof) - 1:
                a0, b0, c0 = prof[j - 1:j + 2]
                den = a0 - 2 * b0 + c0
                pts.append(q[j] + (0.5 * (a0 - c0) / den if den else 0.0) * nrm)
        if len(pts) > 30:
            lines.append(np.array(pts))

    def rms(p):
        for _ in range(2):
            c = p.mean(0)
            v = np.linalg.svd(p - c, full_matrices=False)[2][-1]
            r = (p - c) @ v
            p = p[np.abs(r) < 3 * np.median(np.abs(r)) + 0.3]
        c = p.mean(0)
        return float(np.sqrt(np.mean(((p - c) @ np.linalg.svd(p - c, full_matrices=False)[2][-1]) ** 2)))

    best = [float(k1s[int(np.argmin([rms(radial(K, k1)(p)) for k1 in k1s]))]) for p in lines]
    return {"k1": float(np.median(best)) if best else 0.0, "per_line": best, "lines": len(lines),
            "segments": [[round(v) for v in s] for s, _ in kept[:len(lines)]]}


def _rigid2d(a: np.ndarray, b: np.ndarray, thresh: float, iters: int = 200, seed: int = 0):
    """Rotation angle and translation taking 2D points a onto b (a rigid move), RANSAC over pairs, refit on inliers
    (Kabsch). Returns (theta, t, inlier mask)."""
    rng = np.random.default_rng(seed)

    def fit(p, q):
        pc, qc = p.mean(0), q.mean(0)
        H = (p - pc).T @ (q - qc)
        th = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return th, qc - R @ pc, R

    best = None
    for _ in range(iters):
        i, j = rng.choice(len(a), 2, replace=False)
        th, t, R = fit(a[[i, j]], b[[i, j]])
        inl = np.linalg.norm(a @ R.T + t - b, axis=1) < thresh
        if best is None or inl.sum() > best.sum():
            best = inl
    th, t, R = fit(a[best], b[best])
    inl = np.linalg.norm(a @ R.T + t - b, axis=1) < thresh
    th, t, _ = fit(a[inl], b[inl])
    return float(th), t, inl


def track_rigid(gray: list[np.ndarray], region0: np.ndarray, frames: list[int], K: np.ndarray, n: np.ndarray,
                c: np.ndarray, e1: np.ndarray, e2: np.ndarray, height0: np.ndarray | None = None, undistort=None,
                max_pts: int = 400, min_pts: int = 80, fb_px: float = 0.7, thresh_m: float = 0.01) -> dict:
    """A rigid body that slides and turns on a plane, followed through `frames` (indices into `gray`, in order; the
    first is the seed). Features in `region0` (the body in the seed frame) are followed in the image; each one's ray
    meets the plane (moved along n by its height in `height0`, a per-pixel map at the seed frame), giving 2D points
    (along e1, e2) in metres; the body's move per step is the rigid move that best takes them from one frame to the
    next (RANSAC, `thresh_m`). The region and heights are carried by the body's image motion as features are lost
    and re-seeded. Returns per frame the body's pose (turn angle, and where the seed frame's plane origin went):
    a point x (2D, seed frame) is at R(theta) x + t."""
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    undistort = undistort or (lambda p: p)
    h, w = gray[frames[0]].shape
    A = np.eye(3)                                    # seed-frame pixels -> current pixels (similarity)
    th, t = 0.0, np.zeros(2)
    out_th, out_t, used = [0.0], [t.copy()], [0]

    def carried(m, fill):
        return cv2.warpAffine(m, A[:2].astype(np.float32), (w, h), flags=cv2.INTER_NEAREST, borderValue=fill)

    def seed(i):
        m = carried(region0.astype(np.uint8), 0)
        p = cv2.goodFeaturesToTrack(gray[i], maxCorners=max_pts, qualityLevel=0.01, minDistance=5, mask=m * 255)
        return np.zeros((0, 1, 2), np.float32) if p is None else p.astype(np.float32)

    def on_plane(p, hm):
        off = 0.0 if hm is None else hm[np.clip(p[:, 1].astype(int), 0, h - 1), np.clip(p[:, 0].astype(int), 0, w - 1)]
        X = plane_point(undistort(p), K, n, c, off)
        return np.c_[X @ e1, X @ e2], (np.isfinite(off) if hm is not None else np.ones(len(p), bool))

    pts = seed(frames[0])
    for a, b in zip(frames[:-1], frames[1:]):
        if len(pts) < min_pts:
            more = seed(a)
            pts = np.concatenate([pts, more]) if len(pts) else more
        ok_step = False
        if len(pts) >= 8:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(gray[a], gray[b], pts, None, **lk)
            back, st2, _ = cv2.calcOpticalFlowPyrLK(gray[b], gray[a], nxt, None, **lk)
            ok = (st.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm((back - pts).reshape(-1, 2), axis=1) < fb_px)
            p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
            hm = None if height0 is None else carried(height0.astype(np.float32), np.nan)
            if len(p0) >= 8:
                q0, f0 = on_plane(p0, hm)
                q1, _ = on_plane(p1, hm)
                keep = f0 & np.isfinite(q0).all(1) & np.isfinite(q1).all(1)
                p0, p1, q0, q1 = p0[keep], p1[keep], q0[keep], q1[keep]
            if len(p0) >= 8:
                dth, dt, inl = _rigid2d(q0, q1, thresh_m)
                if inl.sum() >= 8:
                    R = np.array([[np.cos(dth), -np.sin(dth)], [np.sin(dth), np.cos(dth)]])
                    th, t = th + dth, R @ t + dt
                    M, _ = cv2.estimateAffinePartial2D(p0[inl], p1[inl], method=cv2.LMEDS)
                    if M is not None:
                        A = np.r_[M, [[0, 0, 1]]] @ A
                    m = carried(region0.astype(np.uint8), 0) > 0
                    inside = m[np.clip(p1[inl, 1].astype(int), 0, h - 1), np.clip(p1[inl, 0].astype(int), 0, w - 1)]
                    pts = p1[inl][inside].reshape(-1, 1, 2).astype(np.float32)
                    out_th.append(th); out_t.append(t.copy()); used.append(int(inl.sum()))
                    ok_step = True
        if not ok_step:
            out_th.append(np.nan); out_t.append(np.full(2, np.nan)); used.append(0)
            pts = np.zeros((0, 1, 2), np.float32)
            break                                    # lost: stop rather than guess
    pad = len(frames) - len(out_th)
    return {"theta": np.r_[out_th, [np.nan] * pad], "t": np.r_[np.array(out_t), np.full((pad, 2), np.nan)],
            "features": np.r_[used, [0] * pad]}
