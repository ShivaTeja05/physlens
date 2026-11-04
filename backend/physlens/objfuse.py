"""One object, seen in many frames, merged into one cloud in its own frame: its size from all sides at once.

Each keyframe sees only part of a vehicle (the rear, then the side, then mostly the roof...). Its points from every
keyframe are brought onto a reference keyframe by a rigid move on the ground (a turn about the vertical and a
shift; vehicles stay on the road), found by ICP (iterative closest point) with a robust cut-off, starting from the
track. The merged cloud is then measured: length along the direction of travel, width across it, height from the
road. Points that don't agree with the others after registration (stray background) fall away in the trimming.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def _yaw(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _thin(P: np.ndarray, cell: float) -> np.ndarray:
    k = np.floor(P / cell).astype(np.int64)
    _, first = np.unique(k, axis=0, return_index=True)
    return P[np.sort(first)]


def register(src: np.ndarray, dst_tree: cKDTree, dst: np.ndarray, t0: np.ndarray, iters: int = 30,
             cut: float = 0.25, yaw0: float = 0.0, pivot: np.ndarray | None = None,
             lock_dir: np.ndarray | None = None) -> tuple[np.ndarray, float, float]:
    """Yaw + horizontal shift moving `src` onto `dst` (ground frame, y up). Starts from turning by `yaw0` about
    `pivot` (the source's centre) and shifting by `t0`. Returns (4x4, rms, overlap).
    `lock_dir` (unit, ground plane): keep the shift along this direction at its starting value. A box's long flat
    sides let ICP slide frames along them with no penalty, stretching the merged object; the track measures that
    component better (from a face seen in every frame)."""
    T = np.eye(4)
    c = src.mean(0) if pivot is None else pivot
    T[:3, :3] = _yaw(yaw0)
    T[:3, 3] = c - _yaw(yaw0) @ c
    T[[0, 2], 3] += t0
    along0 = float(T[[0, 2], 3] @ lock_dir) if lock_dir is not None else 0.0
    for _ in range(iters):
        P = src @ T[:3, :3].T + T[:3, 3]
        d, j = dst_tree.query(P, distance_upper_bound=cut)
        ok = np.isfinite(d)
        if ok.sum() < 30:
            break
        a, b = P[ok][:, [0, 2]], dst[j[ok]][:, [0, 2]]
        ca, cb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
        R2 = (U @ Vt).T
        if np.linalg.det(R2) < 0:
            Vt[-1] *= -1
            R2 = (U @ Vt).T
        ang = np.arctan2(R2[1, 0], R2[0, 0])
        dT = np.eye(4)
        dT[:3, :3] = _yaw(-ang)
        shift = cb - (R2 @ ca)
        dT[[0, 2], 3] = shift
        T = dT @ T
        if lock_dir is not None:
            t = T[[0, 2], 3]
            T[[0, 2], 3] = t - (t @ lock_dir - along0) * lock_dir
        if abs(ang) < 1e-5 and np.linalg.norm(shift) < 1e-4:
            break
    P = src @ T[:3, :3].T + T[:3, 3]
    d, _ = dst_tree.query(P, distance_upper_bound=cut)
    ok = np.isfinite(d)
    return T, float(np.sqrt(np.mean(d[ok] ** 2))) if ok.any() else float("inf"), float(ok.mean())


def fuse(clouds: list[np.ndarray], centres: np.ndarray, ref: int, cell: float = 0.03, cut: float = 0.25,
         min_overlap: float = 0.25, headings: np.ndarray | None = None, max_turn_deg: float = 15.0,
         lock_dir: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Merge per-keyframe clouds into keyframe `ref`'s frame. Keyframes are added outward from `ref`, each
    registered against everything merged so far (so a view sharing little with `ref` still finds its neighbours).
    `centres`: the track's positions, the starting guess for each shift. `headings` (radians, per keyframe, from
    the track): the starting guess for each turn; a registration that ends up more than `max_turn_deg` away from
    it is rejected (vehicles don't spin between keyframes; a big jump means ICP matched the wrong side)."""
    merged = _thin(clouds[ref], cell)
    report = [{"k": ref, "rms": 0.0, "overlap": 1.0, "used": True}]
    order = sorted(range(len(clouds)), key=lambda k: abs(k - ref))[1:]
    for k in order:
        src = _thin(clouds[k], cell)
        if len(src) < 50:
            continue
        tree = cKDTree(merged)
        t0 = (centres[ref] - centres[k])[[0, 2]]
        yaw0 = 0.0 if headings is None else float(headings[k] - headings[ref])
        T, rms, ov = register(src, tree, merged, t0, cut=cut, yaw0=yaw0, pivot=centres[k], lock_dir=lock_dir)
        yaw = float(np.arctan2(T[0, 2], T[0, 0]))
        turn_ok = abs(np.degrees(np.angle(np.exp(1j * (yaw - yaw0))))) <= max_turn_deg
        used = ov >= min_overlap and rms < cut * 0.6 and turn_ok
        report.append({"k": k, "rms": round(rms, 3), "overlap": round(ov, 2), "used": used,
                       "yaw_deg": round(float(np.degrees(yaw)), 2), "yaw_guess_deg": round(float(np.degrees(yaw0)), 2)})
        if used:
            merged = _thin(np.vstack([merged, src @ T[:3, :3].T + T[:3, 3]]), cell)
    return merged, report


def local_ground(static: np.ndarray, footprint: np.ndarray, r_in: float = 0.5, r_out: float = 4.0) -> float:
    """Height of the ground right around an object: the low static points (road) in a ring around its footprint.
    One plane for the whole scene can sit a few tens of centimetres off where the road dips or a verge rises."""
    c = footprint[:, [0, 2]].mean(0)
    rad = np.percentile(np.linalg.norm(footprint[:, [0, 2]] - c, axis=1), 95)
    dist = np.linalg.norm(static[:, [0, 2]] - c, axis=1)
    ring = static[(dist > rad + r_in) & (dist < rad + r_out)]
    if len(ring) < 50:
        return 0.0
    low = ring[ring[:, 1] < np.percentile(ring[:, 1], 40)]   # the ground, not posts, people or walls
    return float(np.median(low[:, 1]))


def size(cloud: np.ndarray, heading: np.ndarray, trim: float = 1.0, ground_y: float | None = None) -> dict:
    """Length (along `heading`), width, height from the ground (`ground_y` when known, else y = 0 or the cloud's
    base if below it)."""
    h = heading / np.linalg.norm(heading)
    side = np.array([-h[1], h[0]])
    H = cloud[:, [0, 2]]
    a, b = H @ h, H @ side
    lo, hi = trim, 100 - trim
    top = float(np.percentile(cloud[:, 1], hi))
    base = ground_y if ground_y is not None else min(float(np.percentile(cloud[:, 1], lo)), 0.0)
    return {"length": round(float(np.percentile(a, hi) - np.percentile(a, lo)), 3),
            "width": round(float(np.percentile(b, hi) - np.percentile(b, lo)), 3),
            "height": round(top - base, 3), "top": round(top, 3), "points": int(len(cloud))}


def local_plane(static: np.ndarray, footprint: np.ndarray, r_in: float = 0.3, r_out: float = 5.0,
                thresh: float = 0.04, iters: int = 400, seed: int = 0) -> tuple[np.ndarray, np.ndarray] | None:
    """The road surface right around an object, as a plane (unit normal pointing up, a point on it). The scene's
    one ground plane can be a verge or bank tilted against the road; measured against it, a car's width leaks
    into its height. RANSAC over the low static points in a ring around the footprint."""
    c = footprint[:, [0, 2]].mean(0)
    rad = np.percentile(np.linalg.norm(footprint[:, [0, 2]] - c, axis=1), 95)
    dist = np.linalg.norm(static[:, [0, 2]] - c, axis=1)
    ring = static[(dist > rad + r_in) & (dist < rad + r_out)]
    base = np.percentile(footprint[:, 1], 2)
    ring = ring[np.abs(ring[:, 1] - base) < 0.5]         # near the object's base: the road, not posts or banks above
    if len(ring) < 100:
        return None
    rng = np.random.default_rng(seed)
    if len(ring) > 50000:
        ring = ring[rng.choice(len(ring), 50000, replace=False)]
    best, best_n = None, 0
    for _ in range(iters):
        a, b, d = ring[rng.choice(len(ring), 3, replace=False)]
        n = np.cross(b - a, d - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.8:                               # roads are within ~35° of level
            continue
        k = int((np.abs((ring - a) @ n) < thresh).sum())
        if k > best_n:
            best, best_n = (n, a), k
    if best is None:
        return None
    n, a = best
    inl = ring[np.abs((ring - a) @ n) < thresh]
    c0 = inl.mean(0)
    n = np.linalg.svd(inl - c0)[2][-1]
    return (n if n[1] > 0 else -n), c0


def size_on_plane(cloud: np.ndarray, heading: np.ndarray, plane: tuple[np.ndarray, np.ndarray], trim: float = 1.0) -> dict:
    """Length × width × height with 'up' = the local road's normal and height measured from the road."""
    n, c0 = plane
    h3 = np.array([heading[0], 0.0, heading[1]])
    h3 = h3 - (h3 @ n) * n
    h3 /= np.linalg.norm(h3)
    s3 = np.cross(n, h3)
    rel = cloud - c0
    a, b, up = rel @ h3, rel @ s3, rel @ n
    lo, hi = trim, 100 - trim
    tilt = float(np.degrees(np.arccos(np.clip(n[1], -1, 1))))
    return {"length": round(float(np.percentile(a, hi) - np.percentile(a, lo)), 3),
            "width": round(float(np.percentile(b, hi) - np.percentile(b, lo)), 3),
            "height": round(float(np.percentile(up, hi)), 3), "road_tilt_vs_scene_ground_deg": round(tilt, 1),
            "points": int(len(cloud))}
