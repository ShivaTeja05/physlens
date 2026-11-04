"""From per-frame points to one object that moves: the simulator's body, its motion, and what was never seen.

Three stages, each on its own:
  1. extract   each frame's object points, cleaned of background the mask caught (`clean`)
  2. combine   every frame's points registered into ONE body in its own coordinates, and a pose (position and
               heading on the ground) for every frame (`build_body`). Recursive: register every frame against the
               body, rebuild the body from all of them, repeat until no pose moves. The simulator then plays this
               one body of real points along the measured poses: no box, no drawn shape.
  3. analyse   where information is missing (`coverage`): for each side of the body, how much of it has points
               and from how many frames; a side no frame saw stays empty and is reported, never filled in.
Body coordinates: x forward (direction of travel), y up, z to the left; origin on the ground under the body's
centre. A pose maps body → world (ground frame): world = R(yaw) · body + (x, y, z).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from .objfuse import register


def clean(P: np.ndarray, C: np.ndarray, camera: np.ndarray | None, gap: float = 0.6,
          radius: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """Stage 1 clean-up. With a fixed camera: keep the main cluster in distance from the camera (background caught
    by the mask sits metres behind, past a gap). Always: drop points far from the frame's median position."""
    if len(P) < 50:
        return P, C
    keep = np.ones(len(P), bool)
    if camera is not None:
        dist = np.linalg.norm(P - camera, axis=1)
        order = np.argsort(dist)
        ds = dist[order]
        jumps = np.nonzero(np.diff(ds) > gap)[0]
        # Split into runs at the gaps; the run holding most points is the object.
        bounds = np.r_[0, jumps + 1, len(ds)]
        best = int(np.argmax(np.diff(bounds)))
        sel = np.zeros(len(P), bool)
        sel[order[bounds[best]:bounds[best + 1]]] = True
        keep &= sel
    m = np.median(P[keep][:, [0, 2]], axis=0)
    keep &= np.linalg.norm(P[:, [0, 2]] - m, axis=1) < radius
    return P[keep], C[keep]


def _yaw_of(R: np.ndarray) -> float:
    """Yaw of a rotation about y such that body x maps to world (cos yaw, 0, sin yaw)."""
    return float(np.arctan2(R[2, 0], R[0, 0]))


def _rot(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]])


def _thin(P: np.ndarray, cell: float) -> np.ndarray:
    _, first = np.unique(np.floor(P / cell).astype(np.int64), axis=0, return_index=True)
    return np.sort(first)


@dataclass
class Body:
    points: np.ndarray                 # (n, 3) body coordinates
    colours: np.ndarray                # (n, 3) uint8
    seen: np.ndarray                   # (n,) how many frames had a point in this point's 3 cm cell
    frames: list[int]                  # the frames whose poses were solved
    t: list[float]
    poses: list[dict]                  # per frame: x, y, z, yaw (radians), rms, overlap, used
    passes: list[dict] = field(default_factory=list)


def build_body(clouds: list[np.ndarray], colours: list[np.ndarray], frames: list[int], times: list[float],
               ref: int, centres: np.ndarray, headings: np.ndarray, lock: tuple[np.ndarray, np.ndarray] | None = None,
               cut: float = 0.25, cell: float = 0.03, passes: int = 4, tol: float = 0.01,
               max_turn_deg: float = 15.0) -> Body:
    """Stage 2. `centres` (n, 3) and `headings` (n, radians) from the track: the starting guess for each frame.
    `lock`: (direction, along-path positions) for boxy bodies whose flat sides let ICP slide them along the path;
    the along-path shift then comes from a face seen in every frame (the truck's rear, the car's front).
    Pass 1 adds frames outward from `ref`, each registered against everything merged so far; later passes register
    every frame against the whole body and rebuild it, until no frame's pose changes by more than `tol` metres."""
    n = len(clouds)
    if lock is not None:
        d, along = lock
        cen = centres.copy()
        cen[:, [0, 2]] += ((along - cen[:, [0, 2]] @ d)[:, None]) * d
    else:
        cen = centres

    # T[k]: world points of frame k → world frame of the reference frame.
    T = [None] * n
    T[ref] = np.eye(4)
    order = sorted(range(n), key=lambda k: abs(k - ref))

    def merge(ks):
        P = np.vstack([clouds[k] @ T[k][:3, :3].T + T[k][:3, 3] for k in ks])
        C = np.vstack([colours[k] for k in ks])
        F = np.concatenate([np.full(len(clouds[k]), k) for k in ks])
        return P, C, F

    hist = []
    used = [ref]
    Pm = clouds[ref][_thin(clouds[ref], cell)]
    rep = {ref: {"rms": 0.0, "overlap": 1.0}}
    for k in order[1:]:
        tree = cKDTree(Pm)
        t0 = (cen[ref] - cen[k])[[0, 2]]
        # register() turns by a about y with x → (cos a, −sin a): turning frame k's heading onto the reference's
        # is a = h_k − h_ref in that convention.
        a0 = float(headings[k] - headings[ref])
        Tk, rms, ov = register(clouds[k][_thin(clouds[k], cell)], tree, Pm, t0, cut=cut, yaw0=a0, pivot=cen[k],
                               lock_dir=lock[0] if lock is not None else None)
        a = float(np.arctan2(Tk[0, 2], Tk[0, 0]))
        turn_off = abs(np.degrees(np.angle(np.exp(1j * (a - a0)))))
        ok = ov >= 0.25 and rms < cut * 0.6 and turn_off <= max_turn_deg
        rep[k] = {"rms": rms, "overlap": ov}
        if ok:
            T[k] = Tk
            used.append(k)
            Pk = clouds[k] @ Tk[:3, :3].T + Tk[:3, 3]
            Pm = np.vstack([Pm, Pk])
            Pm = Pm[_thin(Pm, cell)]
    hist.append({"pass": 1, "frames_used": len(used), "points": int(len(Pm))})

    for p in range(2, passes + 1):
        P, C, F = merge(used)
        keep = _thin(P, cell)
        tree = cKDTree(P[keep])
        moved = 0.0
        newT = {}
        for k in used:
            if k == ref:
                continue
            src = clouds[k][_thin(clouds[k], cell)]
            # Start from the current pose (register's start: turn a about pivot c, then shift t0) and re-solve
            # against the whole body built from every frame.
            a_cur = float(np.arctan2(T[k][0, 2], T[k][0, 0]))
            c = src.mean(0)
            t0 = (T[k][:3, 3] - c + _rot_reg(a_cur) @ c)[[0, 2]]
            Tk, rms, ov = register(src, tree, P[keep], t0, cut=cut, yaw0=a_cur, pivot=c,
                                   lock_dir=lock[0] if lock is not None else None)
            newT[k] = Tk
            moved = max(moved, float(np.max(np.linalg.norm((src @ Tk[:3, :3].T + Tk[:3, 3]) -
                                                            (src @ T[k][:3, :3].T + T[k][:3, 3]), axis=1))))
            rep[k] = {"rms": rms, "overlap": ov}
        for k, Tk in newT.items():
            T[k] = Tk
        P, C, F = merge(used)
        hist.append({"pass": p, "frames_used": len(used), "points": int(len(_thin(P, cell))), "largest_move_m": round(moved, 4)})
        if moved < tol:
            break

    P, C, F = merge(used)
    # Body frame: at the reference frame, x along its heading, origin on the ground under the body's centre.
    yaw_ref = float(headings[ref])
    Rb = _rot(yaw_ref)                                     # body → world at ref
    lo, hi = np.percentile(P @ Rb, [1, 99], axis=0)        # extents in body axes (rotated into them)
    Pb0 = P @ Rb                                           # world(ref) → body axes: R^T applied as row-vector @ R
    centre_b = np.array([(lo[0] + hi[0]) / 2, np.percentile(Pb0[:, 1], 0.5), (lo[2] + hi[2]) / 2])
    Pb = Pb0 - centre_b
    origin_w = Rb @ centre_b                               # body origin in world(ref) coordinates
    keep = _thin(Pb, cell)
    # How many frames saw each kept point's cell.
    cells = np.floor(Pb / cell).astype(np.int64)
    pairs = np.unique(np.c_[cells, F], axis=0)
    cid, counts = np.unique(pairs[:, :3], axis=0, return_counts=True)
    lookup = {tuple(c): n for c, n in zip(cid, counts)}
    seen = np.array([lookup[tuple(c)] for c in cells[keep]])
    poses = []
    for k in range(n):
        if T[k] is None:
            poses.append({"frame": frames[k], "t": round(times[k], 4), "used": False,
                          "rms": round(rep.get(k, {}).get("rms", float("nan")), 3)})
            continue
        Tinv = np.linalg.inv(T[k])                         # world(ref) → world(k)
        Rw = Tinv[:3, :3] @ Rb                             # body → world(k)
        pos = Tinv[:3, :3] @ origin_w + Tinv[:3, 3]
        poses.append({"frame": frames[k], "t": round(times[k], 4), "used": True, "x": round(float(pos[0]), 4),
                      "y": round(float(pos[1]), 4), "z": round(float(pos[2]), 4), "yaw": round(_yaw_of(Rw), 5),
                      "rms": round(rep[k]["rms"], 3), "overlap": round(rep[k]["overlap"], 2)})
    return Body(Pb[keep], C[keep], seen, [frames[k] for k in range(n)], [times[k] for k in range(n)], poses, hist)


def _rot_reg(yaw: float) -> np.ndarray:
    """objfuse.register's yaw convention (rotation about y by `yaw`)."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


FACES = {"front": (0, +1), "rear": (0, -1), "left": (2, +1), "right": (2, -1), "top": (1, +1)}


def coverage(body: Body, frames_of: np.ndarray | None = None, cell: float = 0.1, shell: float = 0.25) -> dict:
    """Stage 3. For each side of the body: the share of its outline (bounding rectangle) with points within
    `shell` of that side, the area covered, and how many frames contributed points there. A side seen from no
    frame is listed under "never_seen"."""
    P = body.points
    lo, hi = np.percentile(P, [1, 99], axis=0)
    hi[1] = np.percentile(P[:, 1], 99)
    out, never = {}, []
    for name, (ax, sgn) in FACES.items():
        edge = hi[ax] if sgn > 0 else lo[ax]
        near = np.abs(P[:, ax] - edge) < shell
        others = [a for a in range(3) if a != ax]
        span = [(lo[a], hi[a]) for a in others]
        n_cells = [max(int(np.ceil((b - a) / cell)), 1) for a, b in span]
        Q = P[near][:, others]
        # Only the side's interior counts: its border cells also hold the edges of the neighbouring sides.
        m = int(np.ceil(shell / cell))
        inner = [(m, max(nc - m, m + 1)) for nc in n_cells]
        n_inner = (inner[0][1] - inner[0][0]) * (inner[1][1] - inner[1][0])
        if len(Q):
            ij = np.floor((Q - [span[0][0], span[1][0]]) / cell).astype(int)
            ok = (ij[:, 0] >= inner[0][0]) & (ij[:, 0] < inner[0][1]) & (ij[:, 1] >= inner[1][0]) & (ij[:, 1] < inner[1][1])
            covered = len(np.unique(ij[ok], axis=0))
        else:
            covered = 0
        frac = covered / max(n_inner, 1)
        seen_by = int(np.max(body.seen[near])) if near.any() else 0
        out[name] = {"covered_pct": round(100 * frac, 1), "area_m2": round(covered * cell * cell, 2),
                     "most_frames_on_a_spot": seen_by, "points": int(near.sum())}
        if frac < 0.1:
            never.append(name)
    size = (hi - lo)
    return {"faces": out, "never_seen": never, "size_m": {"length": round(float(size[0]), 3),
            "height": round(float(hi[1] - max(lo[1], 0.0)), 3), "width": round(float(size[2]), 3)}}


def heat(seen: np.ndarray) -> np.ndarray:
    """Colours for 'how many frames saw this spot': 1 frame dark blue → many frames yellow."""
    x = np.clip(np.log1p(seen) / np.log1p(max(int(np.percentile(seen, 98)), 2)), 0, 1)
    stops = np.array([[30, 40, 140], [40, 150, 200], [80, 210, 120], [250, 220, 60]], float)
    pos = x * (len(stops) - 1)
    i = np.minimum(pos.astype(int), len(stops) - 2)
    f = (pos - i)[:, None]
    return (stops[i] * (1 - f) + stops[i + 1] * f).astype(np.uint8)
