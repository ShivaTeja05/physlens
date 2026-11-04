"""The 9070 crash at 1,000 fps ("Right Overall", cam12), measured the hybrid way (physlens/hybrid.py): the car's
rigid rear part followed in the image through all 361 frames, made metric on the car's side plane from the 3D world
(MoGe-3 lens and depth, crash_twosided.moge_world). Compared with NHTSA's rear-floorpan accelerometers.

The side plane comes from the picture: the wheels are the same size, so their apparent sizes give the hubs'
relative distances, hence the car's direction and side (MoGe-3's own plane was 50° off here). The lens's radial
term from physics: the car rolls at constant speed for the 50 ms before impact. Scale: the wheelbase (hub to hub
on that plane) set to the published 2.670 m (NHTSA's WHLBAS, the one reference length). Frame 50 is impact.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/hybrid_9070.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
import crash_twosided as ct   # noqa: E402
from physlens.hybrid import along_from_pairs, fit_distortion, fit_side_plane, plane_point, radial, track_part   # noqa: E402

WHEELBASE_M = 2.670
OUT = ROOT / "benchmark" / "hybrid_9070.json"


def all_frames(size):
    cap = cv2.VideoCapture(str(ct.VIDEO))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        out.append(cv2.cvtColor(cv2.resize(f, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY))
    return out


def main():
    from physlens.perception.sam_image import segment
    Ps, M, rgb, ms, T, info = ct.moge_world()
    h, w = Ps[0].shape[:2]
    z = np.load(ct.CACHE / "moge_own_1280.npz")
    K = z["K0"].astype(float)
    gray = all_frames((w, h))
    up = T[1, :3]                                       # the ground frame's up, in camera coordinates
    # The car's direction: the long axis of its points at impact, level (floor plane), pointing the way it moves.
    j50 = ct.IDX.index(50)
    car50 = Ps[j50][ms[j50] & M[j50]]
    car0 = Ps[0][ms[0] & M[0]]
    H = car50 - np.outer(car50 @ up, up)
    d = np.linalg.svd(H - H.mean(0), full_matrices=False)[2][0]
    d -= (d @ up) * up
    d /= np.linalg.norm(d)
    if (np.median(car50, 0) - np.median(car0, 0)) @ d < 0:
        d = -d
    n_ai, c_ai = fit_side_plane(car50[np.linspace(0, len(car50) - 1, min(len(car50), 20000)).astype(int)], d,
                                thresh=0.02 * float(np.median(car50[:, 2])))
    # The side plane from the picture itself: the two wheels are the same size, so their apparent heights give
    # their hubs' relative distances (z = f·D/h, D one wheel diameter); the hubs fix the car's direction and the
    # plane (holding that direction and up). MoGe-3's own plane is kept to compare.
    f40 = cv2.resize(rgb[ct.IDX.index(40)], (w, h), interpolation=cv2.INTER_AREA)
    wheels = sorted([i.mask for i in segment(f40, "wheel") if i.mask.sum() > 500], key=lambda m: -m.sum())[:2]
    cen = np.array([np.argwhere(m).mean(0)[::-1] for m in wheels])          # (x, y) pixels
    hts = np.array([np.ptp(np.nonzero(m)[0]) + 1 for m in wheels], float)
    order = np.argsort(cen[:, 0])                                           # rear (left), front (right)
    cen, hts, wheels = cen[order], hts[order], [wheels[i] for i in order]
    rays = np.c_[(cen[:, 0] - K[0, 2]) / K[0, 0], (cen[:, 1] - K[1, 2]) / K[1, 1], np.ones(2)]
    Xh = rays * (K[0, 0] / hts)[:, None]                                    # hubs, in wheel diameters
    d = Xh[1] - Xh[0]
    d -= (d @ up) * up
    d /= np.linalg.norm(d)
    n = np.cross(d, up)
    n /= np.linalg.norm(n)
    c = Xh[0]
    plane_angle = float(np.degrees(np.arccos(abs(n @ np.array([0, 0, 1.0])))))
    ai_angle = float(np.degrees(np.arccos(abs(n_ai @ np.array([0, 0, 1.0])))))

    # The rigid rear part: the car's rear 35% in the image (it starts cut by the left edge), above the wheels, moved
    # along with the part's own image shift; inside the car's SAM mask at the nearest keyframe.
    m0 = ms[0]
    cols = np.nonzero(m0.any(0))[0]
    x_rear, x_front = cols.min(), cols.max()
    span = x_front - x_rear
    kf = np.array(ct.IDX)

    # Above the wheels: a rolling tyre's points move slower than the car (its bottom is momentarily still on the
    # floor), which would pull the median down while the car rolls.
    wheel_top = int(min(np.nonzero(m.any(1))[0].min() for m in wheels)) - 4

    def region_at(i, sh):
        j = int(np.argmin(np.abs(kf - i)))
        m = ms[j] if ms[j] is not None else m0
        box = np.zeros_like(m)
        a = int(max(x_rear + 4 + sh[0], 2))
        b = int(min(x_rear + 0.35 * span + sh[0], w - 2))
        box[:max(int(wheel_top + sh[1]), 1), a:b] = True
        er = cv2.erode(m.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        return er & box

    r = track_part(gray, region_at, 0, len(gray) - 1, K, n, c, d)
    t = (np.arange(len(gray)) - ct.FRAME_AT_IMPACT) / 1000.0
    # Lens distortion from physics: the car rolls at constant speed in the 50 ms before impact.
    steady = (t >= -0.048) & (t <= -0.003)
    lensfit = fit_distortion(r["pairs"], K, n, c, d, t, steady)
    und = radial(K, lensfit["k1"])
    raw = r["along_m"]
    n0, d0 = n, d
    lensfit.pop("n"), lensfit.pop("d")
    along = along_from_pairs(r["pairs"], K, n, c, d, und)
    Xw = plane_point(und(cen), K, n, c)
    wb_ai = float(abs((Xw[1] - Xw[0]) @ d))
    k = WHEELBASE_M / wb_ai
    x = (along - along[ct.FRAME_AT_IMPACT])
    ts, vs, xs = ct.sensor_curve()
    xm = x * k
    sel = (t >= 0) & (t <= 0.3)
    s_on = np.interp(t[sel], ts, xs)
    pre = (t >= -0.05) & (t < -0.002)
    v_app = float(np.polyfit(t[pre], xm[pre], 1)[0])
    kmax = int(np.argmax(np.where(sel, xm, -np.inf)))
    # Speed from the curve (central differences over ±3 ms) vs the accelerometers'.
    v_vid = np.gradient(cv2.GaussianBlur(xm.reshape(1, -1), (1, 7), 0).ravel(), t)
    late = (t > 0.12) & (t <= 0.3)
    xr = (raw - raw[ct.FRAME_AT_IMPACT])
    Xw0 = plane_point(cen, K, n0, c)
    k0 = WHEELBASE_M / float(abs((Xw0[1] - Xw0[0]) @ d0))
    res = {"side_plane": "from the wheels' apparent sizes (hub distances), not the AI depth",
           "side_plane_angle_to_camera_deg": round(plane_angle, 1), "ai_side_plane_angle_deg": round(ai_angle, 1),
           "wheel_heights_px": hts.tolist(), "region": "rear 35% of the car, above the wheels", "lens_distortion": lensfit, "approach_kmh_without_distortion_fix": round(float(np.polyfit(t[pre], xr[pre] * k0, 1)[0]) * 3.6, 2),
           "max_crush_m_without_distortion_fix": round(float(np.max(np.where(sel, xr * k0, -np.inf))), 3),
           "wheelbase_in_wheel_diameters": round(wb_ai, 4), "implied_wheel_diameter_m": round(k, 3), "scale_from_wheelbase": round(k, 4), "side_plane_distance_wheel_diameters": round(float(c @ n), 3),
           "features_median": int(np.median(r["features"])), "features_min": int(r["features"].min()),
           "approach_kmh": round(v_app * 3.6, 2), "truth_kmh": ct.c15.TRUTH["impact_kmh"],
           "max_crush_m": round(float(xm[kmax]), 3), "max_crush_at_ms": round(float(t[kmax] * 1000), 1),
           "sensor_max_crush_m": round(float(xs.max()), 3), "sensor_at_ms": round(float(ts[np.argmax(xs)] * 1000), 1),
           "rms_vs_sensor_m": round(float(np.sqrt(np.mean((xm[sel] - s_on) ** 2))), 4),
           "rebound_ms": round(float(-np.polyfit(t[late], xm[late], 1)[0]), 2),
           "sensor_rebound_ms": round(float(-np.polyfit(ts[(ts > 0.12) & (ts <= 0.3)], xs[(ts > 0.12) & (ts <= 0.3)], 1)[0]), 2),
           "speed_rms_vs_sensor_ms": round(float(np.sqrt(np.mean((v_vid[sel] - np.interp(t[sel], ts, vs)) ** 2))), 3),
           "curve": {"t_ms": (t * 1000).round(1).tolist(), "video_m": xm.round(4).tolist(),
                     "sensor_t_ms": (ts[::20] * 1000).round(1).tolist(), "sensor_m": xs[::20].round(4).tolist(),
                     "video_speed_ms": v_vid.round(3).tolist(), "sensor_speed_ms": vs[::20].round(3).tolist()}}
    res["features_per_frame"] = r["features"].tolist()
    print(json.dumps({kk: v for kk, v in res.items() if kk not in ("curve", "features_per_frame")}, indent=1))
    OUT.write_text(json.dumps(res, default=float))
    if "--write" in sys.argv:
        write_world(res)
    return res


def write_world(res):
    """The curve and the results into the crash world (media/pointworld/crash-9070.json notes)."""
    f = ROOT / "media" / "pointworld" / "crash-9070.json"
    h = json.loads(f.read_text())
    n = h["notes"]
    n["crush_curve"] = {k: res["curve"][k] for k in ("t_ms", "video_m", "sensor_t_ms", "sensor_m")}
    n["crush_curve_how"] = ("1,000 fps 'Right Overall' camera: the car's rear part followed in the image through all "
                            "361 frames, made metric on the car's side plane (found from the wheels' apparent sizes); "
                            "lens term from the constant-speed approach; scale from the wheelbase (2.670 m)")
    rows = [r for r in n.get("truth_table", []) if not r[0].startswith(("Hybrid", "Approach speed, 1,000", "Crush, 1,000", "Width from both", "Rebound, 1,000"))]
    rows += [["Hybrid, 1,000 fps: approach speed", f"{res['approach_kmh']:.1f} km/h", f"{res['truth_kmh']} km/h (speed trap)"],
             ["Hybrid, 1,000 fps: largest crush", f"{res['max_crush_m']:.2f} m at {res['max_crush_at_ms']:.0f} ms",
              f"{res['sensor_max_crush_m']:.2f} m at {res['sensor_at_ms']:.0f} ms (accelerometers)"],
             ["Hybrid, 1,000 fps: crush curve vs accelerometers, 0–300 ms", f"within {res['rms_vs_sensor_m'] * 100:.1f} cm (RMS)", "—"],
             ["Hybrid, 1,000 fps: rebound (120–300 ms)", f"{res['rebound_ms']:.2f} m/s", f"{res['sensor_rebound_ms']:.2f} m/s (same window)"]]
    n["truth_table"] = rows
    ev = [e for e in n.get("events", []) if e.get("kind") != "stop_hybrid"]
    ev.append({"t": round(1.4731 + res["max_crush_at_ms"] / 1000, 4), "kind": "stop",
               "label": f"Car stops crushing: {res['max_crush_m']:.2f} m at {res['max_crush_at_ms']:.0f} ms after impact",
               "detail": "from the 1,000 fps camera (hybrid 2D tracking + 3D plane)",
               "truth": f"accelerometers: {res['sensor_max_crush_m']:.2f} m at {res['sensor_at_ms']:.0f} ms"})
    n["events"] = sorted([e for e in ev if not (e.get("kind") == "stop" and e.get("label", "").startswith("Most crushed"))], key=lambda e: e["t"])
    f.write_text(json.dumps(h))


if __name__ == "__main__":
    main()
