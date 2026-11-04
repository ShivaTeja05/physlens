"""Crash reconstruction from a fixed side camera: approach speed, impact, crush, rebound.

The vehicle's front and rear edges are tracked in every frame (its centre moves as it crushes; the rear doesn't).
Before impact a towed crash-test car coasts at constant speed and is rigid, which calibrates the camera itself:
  * if the track isn't quite parallel to the image, the car appears to speed up or slow down as it crosses. A 1-D
    perspective correction is fitted so that the front edge moves at constant speed and the car's length stays
    constant, and it's kept only if it clearly improves the fit (on NHTSA test 9070 it isn't needed);
  * the car's known length turns corrected pixels into metres.
Then: impact = when the front edge reaches where it stops; crush = how much further the rear moves after impact
(sampled at 30 fps, so a lower bound); rebound = the car's speed as it leaves the barrier.

The approach speed is locked as a prediction before impact and checked against the official test speed when given.
"""
from __future__ import annotations


import numpy as np

from .common import ObjectTrack, Timeline, add_tracks, discover_law, explain, interval, measurement
from ..physics import calibrate_ruler

GLOSSARY = """- speed_kmh, speed_ms: approach speed measured from the video. official_kmh: the test lab's measured speed.
- crush_m: how much the car shortened at its most compressed (at least, from 30 fps samples).
- rebound_ms: speed as it bounced back off the barrier. restitution: rebound speed ÷ approach speed.
- decel_g: average deceleration while the front crushed, in multiples of Earth's gravity."""


def template(ev: dict) -> str:
    s = f"The car hit the barrier at {ev['speed_kmh']:.1f} km/h, measured from the video"
    s += f" (the lab measured {ev['official_kmh']:.2f})." if "official_kmh" in ev else "."
    if "crush_m" in ev:
        s += f" Its front crushed by at least {ev['crush_m']:.2f} m"
        s += f" while it slowed at about {ev['decel_g']:.0f} g on average." if "decel_g" in ev else "."
    if "rebound_ms" in ev:
        s += f" It bounced back at about {ev['rebound_ms']:.1f} m/s."
    return s


EDGE_SYSTEMATIC = 0.02   # where a cut-out's edges sit vs the official bumper-to-bumper length (blur, mirrors)


def _perspective(x: np.ndarray, p: float, cx: float, half_w: float) -> np.ndarray:
    """Position along the track, if the track isn't quite parallel to the image: a 1-D projective correction."""
    return (x - cx) / (1 + p * (x - cx) / half_w) + cx


def _fit(tf, xf, tr, xr, p, cx, hw):
    """Least squares for front_u = a + v t and rear_u = a + L + v t (constant speed, rigid car), given correction p."""
    uf, ur = _perspective(xf, p, cx, hw), _perspective(xr, p, cx, hw)
    A = np.vstack([np.c_[np.ones_like(tf), tf, np.zeros_like(tf)], np.c_[np.ones_like(tr), tr, np.ones_like(tr)]])
    b = np.r_[uf, ur]
    beta, *_ = np.linalg.lstsq(A, b, rcond=None)
    r = b - A @ beta
    return beta, float(r @ r), A


def analyse(tracks: dict[str, ObjectTrack], card, params: dict, ledger, agent, log, video: dict,
            px_per_m=None) -> dict:
    car = tracks["car"]
    fps, W, H = video["fps"], video["width"], video["height"]
    L = float(params.get("car_length_m", 4.6))
    boxes = np.array([[o[0][0], o[1][0], o[0][1], o[2][1]] for o in car.outlines], float)   # x0, x1, y0, y1
    x0, x1, t = boxes[:, 0], boxes[:, 1], car.t

    # Direction of travel, then which edge leads.
    leftwards = np.median(np.diff((x0 + x1)[: max(4, len(x0) // 6)])) < 0
    front, rear = (x0, x1) if leftwards else (x1, x0)
    sign = -1.0 if leftwards else 1.0

    # The barrier face: where the front edge comes to rest at its furthest (the median of its five furthest values,
    # so one noisy frame doesn't set it). The approach is everything before the front first gets there.
    far = np.sort(sign * front)[-5:]
    barrier_x = sign * float(np.median(far))
    step = float(np.median(np.abs(np.diff(front[: max(4, len(front) // 6)]))))
    reached = np.nonzero(sign * (front - barrier_x) >= -max(25.0, 0.15 * step))[0]
    if len(reached) == 0 or reached[0] < 4:
        raise RuntimeError("couldn't find the impact: too little approach before the front stops")
    i_c = int(reached[0])
    appr = np.arange(len(t)) < i_c
    length = np.abs(rear - front)
    front_ok = appr & (front > 2) & (front < W - 3)
    # A rear edge counts only when the whole car is in view (a car still entering has a false "rear").
    rear_ok = appr & (rear > 2) & (rear < W - 3) & (length >= 0.85 * length[appr].max())
    tf, xf, tr, xr = t[front_ok], front[front_ok], t[rear_ok], rear[rear_ok]
    if len(tf) < 4 or len(tr) < 2:
        raise RuntimeError("too few frames of the approach (with the whole car in view) to measure its speed")

    # Is the track parallel to the image? Fit a perspective correction and keep it only if it clearly helps.
    cx, hw = W / 2, W / 2
    n_obs = len(tf) + len(tr)
    b0, rss0, A0 = _fit(tf, xf, tr, xr, 0.0, cx, hw)
    ps = np.linspace(-0.3, 0.3, 601)
    rss = np.array([_fit(tf, xf, tr, xr, p, cx, hw)[1] for p in ps])
    p_best = float(ps[int(np.argmin(rss))])
    s2_p = float(rss.min()) / max(n_obs - 4, 1)
    use_p = (rss0 - rss.min()) / max(s2_p, 1e-9) > 4.0        # more than ~2σ better
    p_used = p_best if use_p else 0.0
    beta, rss_used, A = _fit(tf, xf, tr, xr, p_used, cx, hw)
    a, v_px, Lpx = beta
    s2 = max(rss_used / max(n_obs - 3, 1), 1.0)               # at least a pixel of edge noise
    cov = np.linalg.inv(A.T @ A) * s2
    b_alt = _fit(tf, xf, tr, xr, p_best, cx, hw)[0] if not use_p else b0
    model_spread = abs(abs(b_alt[1]) / b_alt[2] - abs(v_px) / Lpx) / (abs(v_px) / Lpx)

    # Monte Carlo: fit scatter, the edge-definition systematic, and the with/without-correction spread.
    rng = np.random.default_rng(11)
    n = 4000
    P = rng.multivariate_normal(beta, cov, size=n)
    sys_ = rng.uniform(-EDGE_SYSTEMATIC, EDGE_SYSTEMATIC, n) + rng.uniform(-model_spread, model_spread, n)
    ppm_s = P[:, 2] / L
    speed_s = np.abs(P[:, 1]) / ppm_s * (1 + sys_)
    ppm = Lpx / L
    speed, speed_ci = interval(speed_s)
    kmh, kmh_ci = speed * 3.6, (speed_ci[0] * 3.6, speed_ci[1] * 3.6)
    kbest = p_used
    k_lo = k_hi = p_used
    _undistort = lambda x, k, cx_, hw_: _perspective(np.asarray(x, float), k, cx_, hw_)

    # Impact instant: the front reaches where it then stays.
    after = np.arange(len(t)) >= i_c
    stop_u = float(_undistort(np.array([barrier_x]), kbest, cx, hw)[0])
    t_imp = (stop_u - a) / v_px
    rear_ok_any = (rear > 2) & (rear < W - 3)
    rear_at_imp = a + Lpx + v_px * t_imp
    # Crush: how much further the rear travels after impact, until it turns back (sampled: a lower bound).
    win = after & rear_ok_any & (t <= t_imp + 0.25)
    ev = {"speed_kmh": kmh, "speed_ms": speed}
    if params.get("official_kmh"):
        ev["official_kmh"] = float(params["official_kmh"])
    moments = [{"kind": "approach", "t": float(tf[0]), "speed_in": speed}]
    t_min = None
    if win.sum() >= 2:
        ru = _undistort(rear[win], kbest, cx, hw)
        travel = sign * (ru - rear_at_imp)              # positive = rear kept moving towards the barrier
        j = int(np.argmax(travel))
        crush = float(travel[j]) / ppm
        t_min = float(t[win][j])
        if crush > 0.02:
            ev["crush_m"] = crush
            dt_c = max(t_min - t_imp, 0.5 / fps)
            ev["decel_g"] = speed / dt_c / 9.81
            moments.append({"kind": "max_crush", "t": t_min, "note": f"at least {crush:.2f} m"})
    moments.insert(1, {"kind": "impact", "t": float(t_imp), "speed_in": speed})
    # Rebound: once the front leaves the barrier, its speed away from it over the next few frames.
    left = np.nonzero(after & (front > 2) & (front < W - 3) & (sign * (_undistort(front, kbest, cx, hw) - stop_u) < -6))[0]
    if len(left) >= 3:
        sel = left[:5]
        vb = np.polyfit(t[sel], _undistort(front[sel], kbest, cx, hw), 1)[0]
        rebound = abs(vb) / ppm
        if 0 < rebound < speed:
            ev["rebound_ms"] = rebound
            ev["restitution"] = rebound / speed
            moments[1]["speed_out"] = rebound
            moments.append({"kind": "rebound", "t": float(t[sel[0]]), "speed_out": rebound})
    moments.append({"kind": "rest", "t": float(t[-1])})

    # Timeline.
    cal = calibrate_ruler(ppm, H)
    tl = Timeline()
    tl.add(t[0], "narration", text=card.narration["intro"])
    tl.add(t[0], "calibration", method=f"car length {L:.2f} m" + (f", perspective corrected (p = {kbest:+.3f})" if use_p else ", no perspective correction needed (tested)"),
           px_per_m=round(ppm, 3), rel_uncertainty=round(float(np.std(ppm_s) / ppm), 4))
    add_tracks(tl, [car], cal, fps)
    t_lock = float(tf[-1])
    discover_law(tl, t_lock, [(tf, sign * _undistort(xf, kbest, cx, hw) / ppm)], fps, "x",
                 "Approach: along the track", "car", "law_x", float(np.std(ppm_s) / ppm))
    items = [measurement("Approach speed", kmh, float(np.std(speed_s)) * 3.6, "km/h"),
             measurement("Scale (car length)", ppm, float(np.std(ppm_s)), "px/m")]
    if "crush_m" in ev:
        items.append(measurement("Crush (at least)", ev["crush_m"], 1.0 / ppm, "m"))
        items.append(measurement("Average deceleration", ev["decel_g"], None, "g"))
    if "rebound_ms" in ev:
        items.append(measurement("Rebound speed", ev["rebound_ms"], None, "m/s"))
    # Where the car's own instruments are known (crash tests), show them beside the video's numbers.
    for key, label, unit in (("sensor_crush_m", "Crush, car's accelerometer", "m"),
                             ("sensor_decel_g", "Avg deceleration, accelerometer", "g"),
                             ("sensor_rebound_ms", "Rebound, accelerometer", "m/s")):
        if params.get(key):
            items.append(measurement(label, float(params[key]), None, unit))
            ev[key] = float(params[key])
    tl.add(t_lock, "measurement", items=items)
    tl.add(t_lock, "run_state", state="locked", message="Impact speed locked")
    pred = ledger.lock({"id": "impact_speed", "text": f"Hits the barrier at {kmh:.1f} km/h", "quantity": "speed",
                        "value": round(kmh, 2), "unit": "km/h", "ci90": [round(kmh_ci[0], 2), round(kmh_ci[1], 2)],
                        "target_t": round(float(t_imp), 4), "based_on_until_t": round(t_lock, 4)})
    tl.add(t_lock, "prediction", **pred)
    tl.add(t_lock, "narration", text=card.narration["locked"].format(kmh=kmh))
    verdicts = []
    if "official_kmh" in ev:
        v = ledger.verify("impact_speed", round(ev["official_kmh"], 2))
        tl.add(max(t_imp, t_lock + 0.01), "verdict", **v)
        tl.add(max(t_imp, t_lock + 0.01), "narration",
               text=card.narration["hit" if v["hit"] else "miss"].format(kmh=ev["official_kmh"]))
        verdicts.append(v)
    for m in moments:
        tl.add(m["t"], "moment", object="car", **{k: v for k, v in m.items() if k != "t"})
    # The reconstruction: both edges in corrected metres from the barrier face (barrier at 0, car on the + side).
    recon = []
    in_view = lambda x: 2 < x < W - 3
    for k_, (ti, f_, r_) in enumerate(zip(t, front, rear)):
        whole = length[k_] >= 0.85 * length[appr].max() or not appr[k_]
        recon.append([round(float(ti), 4),
                      round(float(sign * (stop_u - _undistort(np.array([f_]), kbest, cx, hw)[0]) / ppm), 3) if in_view(f_) else None,
                      round(float(sign * (stop_u - _undistort(np.array([r_]), kbest, cx, hw)[0]) / ppm), 3) if in_view(r_) and whole else None])
    height_px = float(np.median(boxes[rear_ok, 3] - boxes[rear_ok, 2])) if rear_ok.any() else 0.35 * Lpx
    tl.add(float(t[0]), "reconstruction", object="car", label=car.label, length_m=round(L, 3),
           # (a floor reflection can inflate the box's height: then a typical car's proportion is used)
           height_m=round(height_px / ppm if height_px / ppm < 0.4 * L else 0.36 * L, 3),
           mass_kg=params.get("mass_kg"), frames=recon,
           impact_t=round(float(t_imp), 4), speed_ms=round(speed, 3), crush_m=round(ev.get("crush_m", 0.0), 3),
           max_crush_t=round(t_min, 4) if t_min else None, rebound_ms=round(ev.get("rebound_ms", 0.0), 3))
    out = explain(agent, tl, max(t_imp, t_lock) + 0.2, ev, template, GLOSSARY, ["impact_speed"])
    tl.emit(log)
    return {"experiment": "crash", "speed_kmh": kmh, "speed_ci_kmh": kmh_ci, "perspective": kbest, "impact_t": float(t_imp),
            "crush_m": ev.get("crush_m"), "rebound_ms": ev.get("rebound_ms"), "verdicts": verdicts,
            "t_end": float(t[-1]), **out}
