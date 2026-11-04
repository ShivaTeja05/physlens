"""Physical events in a 3D world, found from the measured points instead of labelled by hand.

A camera rarely sees the part of an object that matters: the truck's box front hits the beam, but a camera behind
it sees only the rear door and the roof at a grazing angle. A rigid body solves that, as in a simulator: an object's
full length is measured in the frames where all of it is in view, then the whole body is placed at every frame from
the face the camera does see (the rear). Its hidden front is then known, and so is the moment it reaches a
structure measured in the same 3D world.

  rigid_body    length (and the front of its tallest section) from the frames where it is fully in view
  contact       when that front reaches a plane across the path, with ± from the length's and the plane's spread
  motion_events cruising, braking onset, impact-like jumps in deceleration, stop, from a fitted speed curve
  deformation   when a body stops being rigid: its far end stops while its near end keeps coming (crush)
  height_check  whether the measured height says it should have cleared; a contact that happened anyway bounds
                the true height from below
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def along(P: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Position along the ground direction `d` (unit, (x, z)) of points in the ground frame (y up)."""
    return P[:, [0, 2]] @ d


def ends(clouds: list[np.ndarray], d: np.ndarray, lo: float = 3, hi: float = 97) -> tuple[np.ndarray, np.ndarray]:
    """Rear and front of each frame's visible points along the direction of travel (robust percentiles)."""
    a = [along(c, d) for c in clouds]
    return np.array([np.percentile(x, lo) for x in a]), np.array([np.percentile(x, hi) for x in a])


@dataclass
class RigidBody:
    length: float                  # rear to front, metres
    length_err: float
    section_front: float           # rear to the front of the tallest section (a truck's box, not its cab)
    section_front_err: float
    top: float                     # height of the tallest section (ground frame y)
    frames_used: list[int]

    def as_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def rigid_body(clouds: list[np.ndarray], frames: list[int], d: np.ndarray, full: float = 0.04,
               bin_m: float = 0.25, drop: float = 0.3) -> RigidBody:
    """The body's length from the frames whose visible length is within `full` of the longest seen (fully in view),
    and the front of its tallest section: the last stretch along the body whose top is within `drop` of the
    highest top (a box truck's cab sits lower than its box)."""
    rear, front = ends(clouds, d)
    vis = front - rear
    sel = np.nonzero(vis >= vis.max() * (1 - full))[0]
    L = vis[sel]
    fronts, tops = [], []
    for k in sel:
        c = clouds[k]
        a = along(c, d) - rear[k]
        edges = np.arange(0, vis[k] - bin_m / 2, bin_m)    # within the body's robust ends (stray points past them)
        prof = np.array([np.percentile(c[(a >= e) & (a < e + bin_m), 1], 98) if ((a >= e) & (a < e + bin_m)).sum() > 30
                         else np.nan for e in edges])
        top = np.nanmax(prof)
        tall = np.nonzero(prof >= top - drop)[0]
        fronts.append(min(edges[tall[-1]] + bin_m, vis[k]))
        tops.append(top)
    fronts = np.array(fronts)
    spread = lambda x: float(max(np.std(x), np.ptp(x) / 2)) if len(x) > 1 else 0.0
    return RigidBody(float(np.median(L)), spread(L), float(np.median(fronts)), max(spread(fronts), bin_m / 2),
                     float(np.median(tops)), [frames[k] for k in sel])


def crossing(t: np.ndarray, x: np.ndarray, target: float) -> float | None:
    """First time the (increasing) series x(t) reaches `target`, linearly interpolated."""
    above = np.nonzero(x >= target)[0]
    if not len(above):
        return None
    k = above[0]
    if k == 0:
        return float(t[0])
    return float(t[k - 1] + (target - x[k - 1]) / (x[k] - x[k - 1]) * (t[k] - t[k - 1]))


def contact(t: np.ndarray, rear: np.ndarray, offset: float, offset_err: float, plane: float, plane_err: float) -> dict:
    """When the body's front (rear + offset) reaches the plane at `plane` along the path, and the window it could
    fall in given both spreads. The rear series is made non-decreasing first (a vehicle doesn't reverse here; small
    backward steps are measurement noise)."""
    r = np.maximum.accumulate(rear)
    t_mid = crossing(t, r, plane - offset)
    t_early = crossing(t, r, plane - offset - np.hypot(offset_err, plane_err))
    t_late = crossing(t, r, plane - offset + np.hypot(offset_err, plane_err))
    return {"t": t_mid, "t_earliest": t_early, "t_latest": t_late,
            "front_offset_m": round(offset, 3), "plane_along_m": round(plane, 3)}


def motion_events(t: np.ndarray, v: np.ndarray, brake_ms2: float = 0.8, hold_s: float = 0.25,
                  stop_ms: float = 0.15, jump_ms2: float = 3.0) -> list[dict]:
    """Events read off a speed curve sampled on a fine grid: peak speed, braking onset (deceleration above
    `brake_ms2` held for `hold_s`), jumps in deceleration (a sudden extra `jump_ms2` within ~0.1 s: something
    pushed back), and the stop."""
    out = []
    dt = float(np.median(np.diff(t)))
    a = np.gradient(v, t)
    k_peak = int(np.argmax(v))
    out.append({"kind": "peak_speed", "t": float(t[k_peak]), "speed_ms": float(v[k_peak])})
    hold = max(int(round(hold_s / dt)), 1)
    for k in range(k_peak, len(t) - hold):
        if np.all(a[k:k + hold] < -brake_ms2):
            out.append({"kind": "braking", "t": float(t[k]), "speed_ms": float(v[k]),
                        "decel_ms2": float(-np.mean(a[k:k + hold * 2]))})
            break
    w = max(int(round(0.1 / dt)), 1)
    for k in range(w, len(t) - w):
        if a[k - w] - a[k + w] > jump_ms2 and v[k] > stop_ms:
            out.append({"kind": "decel_jump", "t": float(t[k]), "speed_ms": float(v[k]),
                        "extra_decel_ms2": float(a[k - w] - a[k + w])})
            break
    moving = np.nonzero(v > stop_ms)[0]
    if len(moving) and moving[-1] < len(t) - 1:
        out.append({"kind": "stop", "t": float(t[moving[-1] + 1])})
    return out


def deformation(t: np.ndarray, rear: np.ndarray, front: np.ndarray, length: float, shrink: float = 0.08) -> dict | None:
    """When a body stops being rigid: its visible length drops by more than `shrink` (fraction) while both ends are
    in view: the front has stopped (it hit something) and the rear is still coming. Returns the first such moment
    and the largest shortening (crush)."""
    vis = front - rear
    bad = np.nonzero(vis < length * (1 - shrink))[0]
    if not len(bad):
        return None
    k = bad[0]
    return {"t": float(t[k]), "max_crush_m": float(length - vis.min()), "t_max_crush": float(t[int(np.argmin(vis))])}


def height_check(top: float, ground: float, clearance_top: float, clearance_ground: float, hit: bool,
                 err: float = 0.0, signed: float | None = None) -> dict:
    """Measured height vs a clearance (`err`: the combined ± of the two). If the body measured lower than the
    clearance by more than that and still hit, the height measurement is low: the contact bounds the true height
    from below (by the signed clearance when there is one, a better number than our own)."""
    h = top - ground
    c = clearance_top - clearance_ground
    out = {"measured_height_m": round(h, 3), "clearance_m": round(c, 3), "margin_m": round(c - h, 3), "err_m": round(err, 3)}
    if hit:
        floor = signed if signed is not None else c
        out["verdict"] = "consistent" if h + err >= floor else "height reads low"
        out["true_height_at_least_m"] = round(floor, 3)
        if h + err < floor:
            out["measured_low_by_at_least_pct"] = round((floor - h - err) / floor * 100, 1)
    return out


@dataclass
class Refined:
    template: np.ndarray           # the body's points in its own frame: x from the rear along travel, y up, z across
    rear: np.ndarray               # rear position along the path at each frame (after registration)
    length: float
    top: float
    iterations: int
    history: list[dict]


def refine_rigid(clouds: list[np.ndarray], d: np.ndarray, rear0: np.ndarray, seed: list[int], iters: int = 12,
                 cell: float = 0.03, cut: float = 0.15, tol: float = 0.005, search: float = 1.0,
                 build: list[int] | None = None) -> Refined:
    """Recursive rigid-body completion. Start from the frames in `seed` (fully in view); then repeat:
      1. place the current body in every frame: the shift along the path that best lays the frame's points on it
         (1-D search, then median nearest-neighbour refinement), so frames that see only part of it (the rear
         from far away, the roof from above, the side as it turns) are placed by what they do see;
      2. rebuild the body from every frame's points in its own frame (voxel-thinned: each surface once);
    until no frame moves by more than `tol`. Each pass adds the points the last pass couldn't place, so the body
    ends with every point any frame saw. Surfaces no frame ever saw stay empty: repetition can't invent them.
    `build`: the frames whose points may join the body (default all); the rest are only placed against it (frames
    where the mask also catches something the body touches, like a beam it is driving under)."""
    from scipy.spatial import cKDTree

    side = np.array([-d[1], d[0]])
    lat0 = np.median(np.concatenate([clouds[k][:, [0, 2]] @ side for k in seed]))

    def body(k, r):
        c = clouds[k]
        return np.c_[c[:, [0, 2]] @ d - r, c[:, 1], c[:, [0, 2]] @ side - lat0]

    def thin(P):
        _, first = np.unique(np.floor(P / cell).astype(np.int64), axis=0, return_index=True)
        return P[np.sort(first)]

    rear = rear0.astype(float).copy()
    tmpl = thin(np.vstack([body(k, rear[k]) for k in seed]))
    hist = []
    for it in range(iters):
        tree = cKDTree(tmpl)
        new = rear.copy()
        for k in range(len(clouds)):
            P = body(k, rear[k])
            if len(P) > 4000:
                P = P[np.linspace(0, len(P) - 1, 4000).astype(int)]
            best = (np.inf, 0.0)
            for s in np.arange(-search, search + 1e-9, 0.05):
                dd, _ = tree.query(P - [s, 0, 0], distance_upper_bound=cut)
                ok = np.isfinite(dd)
                score = (np.mean(dd[ok]) if ok.any() else cut) + cut * (1 - ok.mean())
                if score < best[0]:
                    best = (score, s)
            s = best[1]
            for _ in range(5):                   # refine below the search step
                dd, j = tree.query(P - [s, 0, 0], distance_upper_bound=cut)
                ok = np.isfinite(dd)
                if ok.sum() < 30:
                    break
                s += float(np.median((P[ok] - [s, 0, 0] - tmpl[j[ok]])[:, 0]))
            new[k] = rear[k] + s
        moved = float(np.max(np.abs(new - rear)))
        rear = new
        tmpl = thin(np.vstack([body(k, rear[k]) for k in (build if build is not None else range(len(clouds)))]))
        a = tmpl[:, 0]
        L = float(np.percentile(a, 99) - np.percentile(a, 1))
        hist.append({"iteration": it + 1, "points": int(len(tmpl)), "length_m": round(L, 3),
                     "top_m": round(float(np.percentile(tmpl[:, 1], 99)), 3), "largest_move_m": round(moved, 4)})
        search = 0.3                             # later passes only correct small residuals
        if moved < tol:
            break
    a = tmpl[:, 0]
    return Refined(tmpl, rear, float(np.percentile(a, 99) - np.percentile(a, 1)), float(np.percentile(tmpl[:, 1], 99)),
                   len(hist), hist)
