"""Physics from a moving camera: take the camera's own motion out, so what's left is how the object really moved.

    1. camera     keyframes go to MapAnything (no intrinsics or poses given: it estimates both, plus depth), all in
                  one shared frame; poses in between are interpolated (rotation slerp, position spline)
    2. gravity    the floor is the largest plane in the point cloud whose normal points up for the camera; "up" and
                  height zero come from it
    3. object     centre and apparent radius in every frame (here an orange-colour mask; SAM 3.1 masks in the app)
    4. size       the object's real radius from MapAnything depth at the keyframes; distance in every other frame
                  from its apparent size (distance = focal × radius ÷ radius in pixels), smoothed over time
    5. world      the ray through the object's centre × distance, through the camera pose, then into the floor frame

The result is the object's path in metres in a gravity-aligned world frame, with nothing assumed except that the
camera is a pinhole and the object is round.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation, Slerp


def read_frames(video: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    return frames, fps


def track_colour(frames: list[np.ndarray], lo=(3, 120, 60), hi=(22, 255, 255)) -> list[tuple[float, float, float] | None]:
    """(u, v, radius px) of the largest blob in the colour range, per frame. Radius from area (robust to edges)."""
    out = []
    for f in frames:
        m = cv2.inRange(cv2.cvtColor(f, cv2.COLOR_BGR2HSV), np.array(lo), np.array(hi))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        if n < 2:
            out.append(None)
            continue
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = stats[k, cv2.CC_STAT_AREA]
        out.append(None if area < 12 else (float(cent[k][0]), float(cent[k][1]), float(np.sqrt(area / np.pi))))
    return out


def interpolate_poses(key_idx: list[int], key_poses: list[np.ndarray], n: int) -> np.ndarray:
    """Cam-to-world 4×4 for every frame 0..n−1 from keyframe poses."""
    R = Rotation.from_matrix([p[:3, :3] for p in key_poses])
    slerp = Slerp(key_idx, R)
    spline = CubicSpline(key_idx, np.array([p[:3, 3] for p in key_poses]), axis=0)
    idx = np.clip(np.arange(n), key_idx[0], key_idx[-1])
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, :3] = slerp(idx).as_matrix()
    out[:, :3, 3] = spline(idx)
    return out


def refine_poses(frames, keys, views, poses: np.ndarray, f: float, sightings) -> tuple[np.ndarray, int]:
    """Every frame's own camera pose: background features matched to the nearest keyframe, whose MapAnything depth
    makes each match a known 3D point, then a perspective solve (PnP with RANSAC). Interpolating between keyframes
    flattens the camera's swings (a camera following a falling object turns along a curve); this measures them."""
    h, w = frames[0].shape[:2]
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    sift = cv2.SIFT_create(1500)
    match = cv2.BFMatcher(cv2.NORM_L2)

    def feats(i):
        m = np.full((h, w), 255, np.uint8)
        if sightings[i] is not None:                       # leave the moving object out: only the static world counts
            u, v, r = sightings[i]
            cv2.circle(m, (int(u), int(v)), int(2 * r + 4), 0, -1)
        return sift.detectAndCompute(cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY), m)

    key_feats = {k: feats(k) for k in keys}
    out, solved = poses.copy(), 0
    for i in range(len(frames)):
        if i in key_feats:
            continue
        j = int(np.argmin([abs(k - i) for k in keys]))
        k, vw = keys[j], views[j]
        kp_k, des_k = key_feats[k]
        kp_i, des_i = feats(i)
        if des_i is None or des_k is None:
            continue
        good = [a for a, b in match.knnMatch(des_i, des_k, k=2) if a.distance < 0.75 * b.distance]
        if len(good) < 12:
            continue
        P3 = np.array([vw.point_at(*kp_k[m.trainIdx].pt, world=True) for m in good], np.float64)
        p2 = np.array([kp_i[m.queryIdx].pt for m in good], np.float64)
        T0 = np.linalg.inv(poses[i])                       # world-to-camera start guess from interpolation
        rvec0, _ = cv2.Rodrigues(T0[:3, :3])
        ok, rvec, tvec, inl = cv2.solvePnPRansac(P3, p2, K, None, rvec0, T0[:3, 3].copy(), useExtrinsicGuess=True,
                                                 reprojectionError=2.0, iterationsCount=200)
        if not ok or inl is None or len(inl) < 10:
            continue
        Rw, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = Rw.T, -Rw.T @ tvec.ravel()
        out[i] = T
        solved += 1
    return out, solved


def _plane_ransac(P: np.ndarray, thresh: float, iters: int = 400, seed: int = 0):
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        nrm = np.cross(b - a, c - a)
        if np.linalg.norm(nrm) < 1e-9:
            continue
        nrm /= np.linalg.norm(nrm)
        inl = np.abs((P - a) @ nrm) < thresh
        if inl.sum() > best_n:
            best, best_n = (nrm, a), int(inl.sum())
    nrm, a = best
    inl = np.abs((P - a) @ nrm) < thresh
    # Least-squares refit on the inliers.
    c = P[inl].mean(0)
    nrm = np.linalg.svd(P[inl] - c, full_matrices=False)[2][-1]
    return nrm, c, np.abs((P - c) @ nrm) < thresh


def floor_frame(views, key_poses: list[np.ndarray]) -> tuple[np.ndarray, dict]:
    """World(MapAnything) → floor frame: y up along the floor normal, height 0 on the floor, x along the first
    camera's right. Two largest planes are found; the floor is the one whose normal is closest to the cameras' up
    (more planes are searched when neither is close to level)."""
    pts = []
    for v in views:
        P = v.points_world[::6, ::6].reshape(-1, 3)
        ok = np.isfinite(P).all(1)
        if v.confidence is not None:
            c = v.confidence[::6, ::6].reshape(-1)
            ok &= c >= np.percentile(c[ok], 30)
        pts.append(P[ok])
    P = np.concatenate(pts)
    depth_scale = float(np.median([np.median(v.depth) for v in views]))
    thresh = 0.01 * depth_scale
    cam_up = -np.mean([p[:3, 1] for p in key_poses], axis=0)          # OpenCV y is down
    cam_up /= np.linalg.norm(cam_up)
    planes, rest = [], P
    for _ in range(2):
        nrm, c, inl = _plane_ransac(rest, thresh)
        planes.append((nrm, c, int(inl.sum())))
        rest = rest[~inl]
    nrm, c, count = max(planes, key=lambda p: abs(p[0] @ cam_up))
    # When a large object or a wall fills the view, the floor may not be among the two largest planes: keep
    # looking (up to six planes).
    if abs(nrm @ cam_up) < np.cos(np.radians(10)):
        for _ in range(4):
            if len(rest) < 500:
                break
            n2, c2, inl = _plane_ransac(rest, thresh)
            planes.append((n2, c2, int(inl.sum())))
            rest = rest[~inl]
        # The floor is level and BELOW the cameras (a ceiling or light grid is level too, above them): among the
        # level planes with a fair share of points, the lowest one.
        cam_c = np.mean([p[:3, 3] for p in key_poses], axis=0)
        level = []
        for n2, c2, k2 in planes:
            if abs(n2 @ cam_up) > np.cos(np.radians(35)):
                n2 = n2 if n2 @ cam_up > 0 else -n2
                height = float((cam_c - c2) @ n2)
                if height > 0.1 * depth_scale:
                    level.append((n2, c2, k2, height))
        if level:
            big = max(p[2] for p in level)
            nrm, c, count, _ = max([p for p in level if p[2] >= 0.2 * big], key=lambda p: p[3])
    if nrm @ cam_up < 0:
        nrm = -nrm
    x = key_poses[0][:3, 0] - (key_poses[0][:3, 0] @ nrm) * nrm
    x /= np.linalg.norm(x)
    z = np.cross(x, nrm)
    Rw = np.stack([x, nrm, z])                                         # rows: new axes in old coordinates
    T = np.eye(4)
    T[:3, :3] = Rw
    T[:3, 3] = -Rw @ c
    tilt = float(np.degrees(np.arccos(np.clip(abs(nrm @ cam_up), -1, 1))))
    return T, {"floor_points": count, "points": len(P), "camera_tilt_from_floor_up_deg": round(tilt, 1)}


@dataclass
class WorldTrack:
    t: np.ndarray
    p: np.ndarray                 # (n, 3) metres, floor frame (y up), NaN where unseen
    cam: np.ndarray               # (n, 3) camera centres, floor frame
    radius_m: float
    focal_px: float
    info: dict = field(default_factory=dict)


def radius_from_floor(t: np.ndarray, A: np.ndarray, B: np.ndarray, fps: float, r0: float) -> float | None:
    """The radius r for which the contacts found in the track p = A + r·B sit at height r (centre one radius above
    the floor). Contacts are where neighbouring free-flight arcs meet (`drop_physics`), not the lowest frame."""
    from scipy.optimize import brentq

    def gap(r):
        c = drop_physics(t, A[:, 1] + r * B[:, 1], r, fps).get("contact_heights") or []
        return float(np.median(c)) - r if c else np.nan

    grid = r0 * np.geomspace(0.25, 4, 33)
    gaps = np.array([gap(r) for r in grid])
    roots = [(grid[i], grid[i + 1]) for i in range(len(grid) - 1)
             if np.isfinite(gaps[i]) and np.isfinite(gaps[i + 1]) and gaps[i] * gaps[i + 1] <= 0]
    if not roots:
        return None
    lo, hi = min(roots, key=lambda ab: abs(np.log(np.sqrt(ab[0] * ab[1]) / r0)))
    return float(brentq(gap, lo, hi, xtol=1e-5)) if gap(lo) * gap(hi) < 0 else float(lo)


def lift(frames: list[np.ndarray], fps: float, keys: list[int], views, sightings,
         focal_px: float | None = None, known_radius_m: float | None = None) -> WorldTrack:
    """focal_px: the lens focal length in pixels when the device knows it (phone video metadata, glasses).
    known_radius_m: the object's real radius when the user knows it (a tennis ball is 3.35 cm). MapAnything's sense
    of scale is an estimate; one known size replaces it, and every length (camera path too) is rescaled to match."""
    n = len(frames)
    key_poses = [v.camera_pose for v in views]
    poses = interpolate_poses(keys, key_poses, n)
    T_floor, finfo = floor_frame(views, key_poses)
    # Focal length (original pixels) as MapAnything estimated it.
    f_est = float(np.median([v.intrinsics[0, 0] / v.scale_x for v in views]))
    f = focal_px or f_est
    poses, n_pnp = refine_poses(frames, keys, views, poses, f, sightings)
    h, w = frames[0].shape[:2]
    cx, cy = w / 2, h / 2
    # The object's radius in metres: depth to its front surface at each keyframe, plus the radius itself.
    radii = []
    for k, v in zip(keys, views):
        s = sightings[k]
        if s is None:
            continue
        u, vv, r_px = s
        zs = [v.point_at(u + du, vv + dv)[2] for du in (-0.3 * r_px, 0, 0.3 * r_px) for dv in (-0.3 * r_px, 0, 0.3 * r_px)]
        z_front = float(np.median(zs))
        radii.append(z_front * r_px / (f - r_px))
    r_depth = float(np.median(radii))
    # Distance from apparent size, smoothed over time (the distance to an object changes smoothly; its outline in
    # pixels jitters by a fraction of a pixel). Every position is then linear in the unknown radius: p = A + r·B.
    seen = np.array([s is not None for s in sightings])
    idx = np.nonzero(seen)[0]
    u = np.array([sightings[i][0] for i in idx]); v = np.array([sightings[i][1] for i in idx])
    r_px = np.array([sightings[i][2] for i in idx])
    win = min(15, len(idx) - (1 - len(idx) % 2))
    log_r = savgol_filter(np.log(r_px), win, 2) if win >= 5 else np.log(r_px)
    per_r = f / np.exp(log_r)                                          # distance per metre of radius
    ray = np.stack([(u - cx) / f * per_r, (v - cy) / f * per_r, per_r], 1)
    Rf, tf = T_floor[:3, :3], T_floor[:3, 3]
    A = poses[idx, :3, 3] @ Rf.T + tf
    B = np.einsum("ij,njk,nk->ni", Rf, poses[idx, :3, :3], ray)
    t_all = np.arange(n) / fps
    # A small object's depth blends with what's behind it, so the depth-based radius is only a first guess. The floor
    # is the better ruler: at every bounce the object's centre sits exactly one radius above the floor plane.
    r_m, how = r_depth, "depth at keyframes"
    r_floor = radius_from_floor(t_all[idx], A, B, fps, r_depth)
    if r_floor is not None:
        r_m, how = r_floor, "bounces touch the floor"
    p = np.full((n, 3), np.nan)
    p[idx] = A + r_m * B
    cam = poses[:, :3, 3] @ Rf.T + tf
    scale, r_scene = 1.0, r_m
    if known_radius_m:
        # The floor frame's origin is on the floor, so a uniform scale about it keeps the floor at height 0.
        scale = known_radius_m / r_m
        p, cam, r_m = p * scale, cam * scale, known_radius_m
    info = {**finfo, "keyframes": len(keys), "frames_posed_by_pnp": n_pnp, "radius_m": round(r_m, 4), "radius_from": how,
            "radius_from_depth_m": round(r_depth, 4),
            "radius_scene_units": round(r_scene, 4), "scale_from": "known object size" if known_radius_m else "MapAnything",
            "scale_applied": round(scale, 4), "radius_spread_m": round(float(np.std(radii)), 4),
            "focal_px": round(f, 1), "focal_from": "device" if focal_px else "MapAnything",
            "focal_mapanything_px": round(f_est, 1)}
    return WorldTrack(np.arange(n) / fps, p, cam, r_m, f, info)


def run(video: Path, step: int = 10, cache: Path | None = None, focal_px: float | None = None,
        known_radius_m: float | None = None, intrinsics: np.ndarray | None = None) -> tuple[WorldTrack, list]:
    """The full moving-camera pass on a clip. MapAnything results are cached (one GPU call per clip)."""
    from .geometry.client import ViewGeometry, reconstruct
    frames, fps = read_frames(video)
    keys = list(range(0, len(frames), step))
    if keys[-1] != len(frames) - 1:
        keys.append(len(frames) - 1)
    sightings = track_colour(frames)
    # intrinsics: the device's 3×3 camera matrix (original pixels), given to MapAnything so poses, depth and focal
    # all agree with the real lens.
    tag = "_K" if intrinsics is not None else ""
    f = cache / f"{video.stem}_mapanything_{step}{tag}.npz" if cache else None
    if f is not None and f.exists():
        z = np.load(f)
        views = [ViewGeometry(z[f"depth{j}"], z[f"cam{j}"], z[f"world{j}"], z[f"conf{j}"], None, z[f"K{j}"],
                              z[f"pose{j}"], float(z[f"sx{j}"]), float(z[f"sy{j}"])) for j in range(len(keys))]
    else:
        views = reconstruct([frames[k] for k in keys], timeout_s=900,
                            intrinsics=None if intrinsics is None else [intrinsics] * len(keys))
        if f is not None:
            f.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, **{k: val for j, vw in enumerate(views) for k, val in {
                f"depth{j}": vw.depth, f"cam{j}": vw.points_cam, f"world{j}": vw.points_world,
                f"conf{j}": vw.confidence if vw.confidence is not None else np.ones_like(vw.depth),
                f"K{j}": vw.intrinsics, f"pose{j}": vw.camera_pose, f"sx{j}": vw.scale_x, f"sy{j}": vw.scale_y}.items()})
    if intrinsics is not None and focal_px is None:
        focal_px = float(intrinsics[0][0])
    return lift(frames, fps, keys, views, sightings, focal_px, known_radius_m), sightings


def drop_physics(t: np.ndarray, y: np.ndarray, radius_m: float, fps: float) -> dict:
    """Gravity (one g for every flight) and bounciness at each contact, from a height track in metres."""
    from .physics.models import consistent_flights, fit_free_fall
    from .physics.segment import find_bounces, split_flights
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    flights = consistent_flights(t, y, split_flights(t, find_bounces(t, y, radius_m), fps))
    if not flights:
        return {"g": None, "flights": 0}
    fit = fit_free_fall(t, y, flights)
    g, g_err = fit.shared["g"]
    ratios, heights = [], []
    for (fa, pa), (fb, pb) in zip(zip(flights, fit.per_flight), zip(flights[1:], fit.per_flight[1:])):
        if fb.index != fa.index + 1:
            continue
        # Where the two arcs meet is the contact; speeds just before and after it give the bounciness.
        ta, tb = fa.t_start, fb.t_start
        ya = lambda s: pa["y0"] + pa["v0"] * (s - ta) - 0.5 * g * (s - ta) ** 2
        yb = lambda s: pb["y0"] + pb["v0"] * (s - tb) - 0.5 * g * (s - tb) ** 2
        grid = np.linspace(t[fa.stop - 1], t[fb.start], 400)
        tc = grid[int(np.argmin(np.abs(ya(grid) - yb(grid))))]
        v_in, v_out = pa["v0"] - g * (tc - ta), pb["v0"] - g * (tc - tb)
        if v_in < 0 < v_out:
            ratios.append(round(float(v_out / -v_in), 3))
            heights.append(float(ya(tc)))
    return {"g": round(float(g), 3), "g_err": round(float(g_err), 3), "flights": len(flights),
            "restitution": ratios, "contact_heights": heights, "rmse_m": round(float(fit.rmse), 4)}


def to_scene(wt: WorldTrack, label: str, phys: dict, video: str) -> dict:
    """A physlens-scene/1 scene from a moving-camera world track. Unlike one fixed camera, depth is measured, so
    x, y and z are all measured. The horizontal axes are turned so the object's sideways motion runs along x."""
    ok = np.isfinite(wt.p).all(1)
    P = wt.p[ok]
    d = P[-1, [0, 2]] - P[0, [0, 2]]
    a = np.arctan2(d[1], d[0])
    c, s = np.cos(a), np.sin(a)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    P = P @ Ry.T
    P[:, [0, 2]] -= P[0, [0, 2]]
    t = wt.t[ok]
    win = min(9, len(t) - (1 - len(t) % 2))
    V = savgol_filter(P, win, 2, deriv=1, delta=float(np.median(np.diff(t))), axis=0)
    g = phys.get("g") or 9.81
    e = phys.get("restitution") or []
    atoms = [{"t": round(float(ti), 4), "p": [round(float(x), 4) for x in p], "v": [round(float(x), 3) for x in v],
              "q": [0, 0, 0, 1], "err": 0.02, "measured": ["x", "y", "z", "vx", "vy", "vz"], "assumed": ["q"]}
             for ti, p, v in zip(t, P, V)]
    return {"format": "physlens-scene/1", "capture": f"moving:{Path(video).stem}", "experiment": "drop",
            "source_video": video, "fps": float(round(1 / np.median(np.diff(wt.t)), 3)),
            "world": {"gravity": [0.0, -abs(g), 0.0],
                      "gravity_source": f"measured in the floor frame ({phys.get('g')} ± {phys.get('g_err')} m/s²)",
                      "ground_y": 0.0, "obstacles": []},
            "objects": [{"id": "ball", "label": label, "shape": "sphere",
                         "size": {"value": [round(wt.radius_m, 4)], "source": "measured (MapAnything depth at keyframes)"},
                         "mass_kg": {"value": None, "source": "unknown (not needed for a drop)"},
                         "restitution": {"value": float(np.median(e)) if e else 0.7,
                                         "source": f"measured (median of {len(e)} contacts)" if e else "assumed"},
                         "friction": {"value": 0.5, "source": "assumed"}}],
            "atoms": {"ball": atoms}, "moments": [],
            "camera": {"path": [[round(float(x), 4) for x in (cp @ Ry.T)] for cp in wt.cam[::2]],
                       "source": "MapAnything poses (keyframes) + interpolation, floor frame"},
            "notes": "Filmed with a moving camera: the camera's motion was measured and removed; depth comes from "
                     "MapAnything and the object's apparent size. Orientation is not measured."}
