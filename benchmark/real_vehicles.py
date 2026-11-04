"""Real footage, real vehicles: speed and impact from openly licensed Commons clips (datasets/commons/SOURCE.md).

  truck   fixed pole camera, a box truck hits the 11foot8 bridge (Durham NC, 2019-09-13, clearance 11′8″) and its box
          roof is peeled open. Scale: the bridge's underside (SAM 3.1 "bridge" mask, bottom edge over the road) is
          3.556 m above the road. Check: box width vs a rental box truck (~2.44 m).
  rally   handheld spectator panning with Latvala's Toyota Yaris WRC (Monte-Carlo 2019). Scale: the car's length
          (4.085 m) measured side-on. Check: speed in car-lengths per second needs no scale at all.

No official speed exists for either moment, so the checks are internal (a second, independent size; agreement
between frames). Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/real_vehicles.py
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
from physlens.vehicle3d import read_at, rescale, track   # noqa: E402

CLIPS = ROOT / "datasets" / "commons"
CACHE = ROOT / "records" / "real_cache"
CLEARANCE_M = 3.556          # 11 ft 8 in
BOX_WIDTH_M = 2.44           # typical rental box truck body, ~8 ft (check, not used)
YARIS_WRC_LENGTH_M = 4.085   # 2019 Toyota Yaris WRC
KMH = 3.6


def extents(cloud: np.ndarray, heading: np.ndarray) -> tuple[float, float]:
    h = heading / np.linalg.norm(heading)
    side = np.array([-h[1], h[0]])
    H = cloud[:, [0, 2]]
    a, b = H @ h, H @ side
    return float(np.percentile(a, 99.5) - np.percentile(a, 0.5)), float(np.percentile(b, 99.5) - np.percentile(b, 0.5))


def headings(P: np.ndarray) -> np.ndarray:
    H = P[:, [0, 2]]
    d = np.gradient(H, axis=0)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def speed_series(tr, k: int = 1):
    H = tr.p[:, [0, 2]]
    d = np.linalg.norm(H[k:] - H[:-k], axis=1)
    dt = tr.t[k:] - tr.t[:-k]
    return (tr.t[k:] + tr.t[:-k]) / 2, d / dt


def underside(view, mask: np.ndarray, T: np.ndarray) -> tuple[float, float, float]:
    """Height of a bridge's lower edge: in every image column the mask's bottom pixel (2 px inside, edge pixels blend
    with what's behind) lifted to 3D; columns where that edge is on the ground (a pillar) are left out."""
    h, w = view.depth.shape
    mm = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    H = view.points_world @ T[1, :3] + T[1, 3]
    edge = [H[np.nonzero(mm[:, x])[0].max() - 2, x] for x in range(w) if mm[:, x].sum() >= 6]
    edge = np.array([e for e in edge if e > 1.5])
    return float(np.median(edge)), float(np.percentile(edge, 25)), float(np.percentile(edge, 75))


def two_phase(t: np.ndarray, u: np.ndarray, t_lo: float, t_hi: float) -> dict:
    """Distance along the road fitted with physics instead of frame differences: steady braking (or speeding up)
    until the impact at t_i, then a steady deceleration a2 to a stop. Position and speed are continuous at t_i;
    t_i is chosen by the fit (grid over [t_lo, t_hi]); ± from the fit's covariance."""
    from scipy.optimize import least_squares

    def model(th, ti, tt):
        s0, v0, a1, a2 = th
        t0 = t[0]
        pre = s0 + v0 * (tt - t0) + 0.5 * a1 * (tt - t0) ** 2
        si, vi = s0 + v0 * (ti - t0) + 0.5 * a1 * (ti - t0) ** 2, v0 + a1 * (ti - t0)
        tau = np.clip(tt - ti, 0, max(vi, 1e-6) / max(a2, 1e-6))
        post = si + vi * tau - 0.5 * a2 * tau ** 2
        return np.where(tt < ti, pre, post)

    best = None
    for ti in np.arange(t_lo, t_hi + 1e-9, 0.01):
        r = least_squares(lambda th: model(th, ti, t) - u, [u[0], 5.0, 0.0, 3.0],
                          bounds=([-np.inf, 0, -20, 0.01], [np.inf, 30, 20, 100]))
        rss = float(r.fun @ r.fun)
        if best is None or rss < best[0]:
            best = (rss, ti, r)
    rss, ti, r = best
    dof = max(len(u) - 5, 1)
    cov = rss / dof * np.linalg.pinv(r.jac.T @ r.jac)
    s0, v0, a1, a2 = r.x
    vi = v0 + a1 * (ti - t[0])
    J = np.array([0, 1, ti - t[0], 0])                    # d vi / d theta
    vi_err = float(np.sqrt(J @ cov @ J))
    return {"impact_t": round(float(ti), 3), "speed_at_impact_ms": round(float(vi), 2), "speed_at_impact_err_ms": round(vi_err, 2),
            "approach_accel_ms2": round(float(a1), 2), "approach_accel_err_ms2": round(float(np.sqrt(cov[2, 2])), 2),
            "speed_at_start_ms": round(float(v0), 2),
            "ride_down_decel_ms2": round(float(a2), 2), "ride_down_decel_err_ms2": round(float(np.sqrt(cov[3, 3])), 2),
            "ride_down_distance_m": round(float(vi ** 2 / (2 * a2)), 2), "ride_down_time_s": round(float(vi / a2), 2),
            "fit_rms_m": round(float(np.sqrt(rss / len(u))), 3), "fitted": model(r.x, ti, t).round(3).tolist()}


def speed_curve(t: np.ndarray, u: np.ndarray, step: float = 0.4, smooth: float = 0.5):
    """Speed as a continuous piecewise-linear curve (knots every `step` s), fitted to every position at once:
    position = start + ∫ speed. A light penalty on changes of acceleration keeps it physical. Returns a function
    giving speed and its ± (from the fit's covariance) at any time, and the fit's position residual."""
    knots = np.arange(t.min(), t.max() + step, step)

    def basis(tt):
        # Column k: ∫ from t0 to tt of the k-th hat function (linear interpolation weight of knot k).
        out = np.zeros((len(tt), len(knots)))
        fine = np.linspace(t.min(), t.max(), 2000)
        hats = np.stack([np.interp(fine, knots, np.eye(len(knots))[k]) for k in range(len(knots))], 1)
        cum = np.vstack([np.zeros(len(knots)), np.cumsum((hats[1:] + hats[:-1]) / 2 * np.diff(fine)[:, None], 0)])
        for k in range(len(knots)):
            out[:, k] = np.interp(tt, fine, cum[:, k])
        return out

    A = np.hstack([np.ones((len(t), 1)), basis(t)])
    D = np.diff(np.eye(len(knots)), 2, axis=0)          # second differences of speed (changes of acceleration)
    R = np.hstack([np.zeros((len(D), 1)), D]) * smooth
    theta, *_ = np.linalg.lstsq(np.vstack([A, R]), np.r_[u, np.zeros(len(D))], rcond=None)
    res = u - A @ theta
    sigma2 = float(res @ res) / max(len(u) - len(knots) - 1, 1)
    cov = sigma2 * np.linalg.pinv(A.T @ A + R.T @ R)

    def v(tt):
        tt = np.atleast_1d(tt)
        W = np.stack([np.interp(tt, knots, np.eye(len(knots))[k]) for k in range(len(knots))], 1)
        W = np.hstack([np.zeros((len(tt), 1)), W])
        return W @ theta, np.sqrt(np.einsum("ij,jk,ik->i", W, cov, W))

    return v, float(np.sqrt(np.mean(res ** 2)))


def truck() -> dict:
    idx = list(range(100, 248, 4))
    tr, frames, views = track(CLIPS / "truck-11foot8" / "clip.mp4", idx, "box truck", CACHE / "truck2", sam_video=True)
    bridge = cv2.imread(str(CACHE / "truck2" / "bridge.png"), cv2.IMREAD_GRAYSCALE) > 0     # SAM 3.1, prompt "bridge"
    u_med, u_lo, u_hi = underside(views[-1], bridge, tr.T_ground)
    rescale(tr, CLEARANCE_M / u_med, f"bridge underside = {CLEARANCE_M} m")
    # Keyframes where the truck is cut off by the image edge are left out.
    masks = [cv2.imread(str(CACHE / "truck2" / "samvideo_box_truck" / f"{idx.index(f):03d}.png"), cv2.IMREAD_GRAYSCALE) > 0 for f in tr.frames]
    whole = np.array([not (m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any()) for m in masks])
    H = tr.p[:, [0, 2]]
    k0 = int(np.argmax(whole))
    d = H[-1] - H[k0]
    d /= np.linalg.norm(d)
    # The rear face: the camera sees it in every frame (the median of the visible surface slides backwards as the
    # side and roof go out of view, which looks like braking that isn't there).
    rear = np.array([np.percentile((c[:, [0, 2]] - H[k0]) @ d, 3) for c in tr.clouds])
    t = tr.t
    v, rms = speed_curve(t[whole], rear[whole])
    # Contact: the box is intact at frame 196 and its roof is being peeled at frame 200 (SAM keeps tracking it).
    t_contact = (196 + 200) / 2 / 29.97
    grid = np.arange(t[whole].min(), t[whole].max() + 1e-9, 1 / 29.97)
    vg, eg = v(grid)
    moving = vg > 0.3
    t_stop = float(grid[np.nonzero(moving)[0][-1]]) if moving.any() else float(grid[-1])
    cruise = (grid < 5.4)
    vc, ec = v(t_contact)
    brake = (grid > 5.4) & (grid < t_contact)
    decel = -np.polyfit(grid[brake], vg[brake], 1)[0] if brake.sum() > 3 else float("nan")
    after = float(np.interp(t_stop, t[whole], rear[whole]) - np.interp(t_contact, t[whole], rear[whole]))
    hd = headings(tr.p)
    k_c = int(np.searchsorted(t, t_contact))
    widths = [extents(c, h)[1] for c, h, w in zip(tr.clouds[:k_c], hd[:k_c], whole[:k_c]) if w]
    roofs = [float(np.percentile(c[:, 1], 98)) for c, w in zip(tr.clouds[:k_c], whole[:k_c]) if w]
    cam_h = float(tr.T_ground[1, :3] @ views[0].camera_pose[:3, 3] + tr.T_ground[1, 3]) * tr.scale
    tm, vv = speed_series(tr)
    KM = KMH
    return {"frames": tr.frames, "t": t.round(3).tolist(), "p": tr.p.round(3).tolist(), "rear_m": rear.round(3).tolist(),
            "used_in_fit": whole.tolist(), "speed_t": tm.round(3).tolist(), "speed_ms": vv.round(2).tolist(),
            "speed_curve": {"t": grid.round(3).tolist(), "kmh": (vg * KM).round(1).tolist(), "err_kmh": (eg * KM).round(1).tolist()},
            "scale": round(tr.scale, 4), "underside_scene_units": [round(u_med, 3), round(u_lo, 3), round(u_hi, 3)],
            "cruise_kmh": round(float(np.mean(vg[cruise])) * KM, 1) if cruise.any() else None,
            "braking_decel_ms2": round(float(decel), 2),
            "contact_t": round(t_contact, 3), "contact_frames": [196, 200],
            "speed_at_contact_kmh": round(float(vc[0]) * KM, 1), "speed_at_contact_err_kmh": round(float(ec[0]) * KM, 1),
            "stop_t": round(t_stop, 2), "distance_after_contact_m": round(after, 2),
            "mean_decel_after_contact_ms2": round(float(vc[0]) ** 2 / (2 * after), 2) if after > 0.05 else None,
            "fit_rms_m": round(rms, 3),
            "check_roof_m": round(float(np.median(roofs)), 2), "check_roof_expected": f"> {CLEARANCE_M} m (it hit the beam)",
            "camera_height_m": round(cam_h, 2),
            "check_box_width_m": round(float(np.median(widths)), 2), "check_box_width_expected_m": BOX_WIDTH_M,
            "info": tr.info}


def rally() -> dict:
    tr, frames, views = track(CLIPS / "rally-latvala" / "clip.mp4", list(range(164, 186, 2)), "rally car", CACHE / "rally", sam_video=True)
    hd = headings(tr.p)
    # Side-on frames: the car's heading is at right angles to the line from the camera to the car.
    cams = np.array([tr.T_ground[:3, :3] @ views[[164 + 2 * j for j in range(11)].index(f)].camera_pose[:3, 3]
                     + tr.T_ground[:3, 3] for f in tr.frames])
    look = tr.p[:, [0, 2]] - cams[:, [0, 2]]
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    side_on = np.abs((hd * look).sum(1)) < 0.35
    lengths = [extents(c, h)[0] for c, h, so in zip(tr.clouds, hd, side_on) if so]
    L_scene = float(np.median(lengths))
    tm, v_scene = speed_series(tr)
    # Speed in the car's own lengths per second: independent of MapAnything's scale.
    lengths_per_s = v_scene / L_scene
    rescale(tr, YARIS_WRC_LENGTH_M / L_scene, f"car length side-on = {YARIS_WRC_LENGTH_M} m")
    v = v_scene * tr.scale
    pre = slice(0, len(v))
    return {"frames": tr.frames, "t": tr.t.round(3).tolist(), "p": tr.p.round(3).tolist(),
            "speed_t": tm.round(3).tolist(), "speed_ms": v.round(2).tolist(),
            "side_on_frames": [f for f, so in zip(tr.frames, side_on) if so],
            "length_scene_units": [round(x, 3) for x in lengths], "scale": round(tr.scale, 4),
            "speed_kmh_median": round(float(np.median(v[pre])) * KMH, 1),
            "speed_kmh_range": [round(float(np.percentile(v, 10)) * KMH, 1), round(float(np.percentile(v, 90)) * KMH, 1)],
            "car_lengths_per_s_median": round(float(np.median(lengths_per_s)), 2),
            "camera_path_m": round(float(np.linalg.norm(np.diff(cams[:, [0, 2]], axis=0), axis=1).sum() * tr.scale), 2),
            "info": tr.info}


def overlay(name: str, res: dict, idx_all: list[int], width: int = 960):
    """Each keyframe with the vehicle's measured path drawn into it (projected through that keyframe's camera)."""
    S = ROOT / "media"
    vid = {"truck": "truck-11foot8", "rally": "rally-latvala"}[name]
    frames, _ = read_at(CLIPS / vid / "clip.mp4", res["frames"], width)
    out = []
    for f, img, sp in zip(res["frames"], frames, [None] + res["speed_ms"]):
        img = img.copy()
        if sp is not None:
            cv2.putText(img, f"{sp * KMH:5.1f} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, f"{sp * KMH:5.1f} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 230, 255), 2, cv2.LINE_AA)
        out.append(cv2.resize(img, (480, 270)))
    pick = out[:: max(1, len(out) // 8)][:8]
    while len(pick) < 8:
        pick.append(np.zeros_like(out[0]))
    cv2.imwrite(str(S / f"{name}_speed.jpg"), np.vstack([np.hstack(pick[:4]), np.hstack(pick[4:])]))


def main():
    res = {"truck": truck(), "rally": rally()}
    (ROOT / "benchmark" / "real_vehicles.json").write_text(json.dumps(res, indent=1))
    for k in res:
        overlay(k, res[k], [])
        print(k, json.dumps({x: y for x, y in res[k].items() if x not in ("t", "p", "speed_t", "speed_ms", "frames", "along_m", "rear_m", "used_in_fit", "speed_curve")
                             and not (isinstance(y, dict) and "fitted" in y)}, indent=1))

        print("  speeds km/h (frame to frame):", [round(s * KMH) for s in res[k]["speed_ms"]])


if __name__ == "__main__":
    main()
