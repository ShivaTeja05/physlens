"""Weigh it by watching: a known weight and a mystery weight bounce on identical springs. The mystery's mass comes
from rhythm alone, no scale needed.

Physics: a mass on a spring bobs with angular frequency ω = √(k/m). Same spring, so m_mystery = m_ref · (ω_ref/ω)².
Periods don't depend on the camera's pixel scale at all. A second, independent estimate uses how much further the
heavier weight stretches its spring: m = m_ref + k·Δstretch/g (this one does need the metre scale). After the lock
the fitted motion also predicts when the mystery weight's next peaks come.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import curve_fit

from ..physics import STANDARD_GRAVITY
from .common import (ObjectTrack, Timeline, add_tracks, calibrate, discover_law, explain, interval, measurement,
                     samples)

GLOSSARY = """- period_ref_s, period_mystery_s: time for one bob of the known and the mystery weight, seconds.
- ref_mass_g: the known weight. mass_pred_g: mystery mass from the rhythm (same spring: m = m_ref·(T/T_ref)²).
- mass_static_g: a second estimate from how much further the mystery stretches its spring.
- mass_true_g: the kitchen-scale reading, if given. spring_k: spring stiffness, N/m."""


def template(ev: dict) -> str:
    s = (f"Same spring, different rhythm: the mystery bobs every {ev['period_mystery_s']:.3f} s against "
         f"{ev['period_ref_s']:.3f} s for the {ev['ref_mass_g']:.0f} g reference, so it weighs about "
         f"{ev['mass_pred_g']:.0f} g")
    if "mass_true_g" in ev:
        s += f" (scale: {ev['mass_true_g']:.0f} g)."
    else:
        s += "."
    if "mass_static_g" in ev:
        s += f" The extra stretch of its spring says {ev['mass_static_g']:.0f} g."
    return s


def _damped(t, c, A, gamma, omega, phi):
    return c + A * np.exp(-gamma * t) * np.cos(omega * t + phi)


def fit_oscillation(t: np.ndarray, y: np.ndarray):
    """Damped cosine fit. The FFT peak seeds the frequency so the fit doesn't lock onto a harmonic."""
    yc = y - y.mean()
    dt = float(np.median(np.diff(t)))
    freqs = np.fft.rfftfreq(len(yc) * 8, dt)
    spec = np.abs(np.fft.rfft(yc * np.hanning(len(yc)), len(yc) * 8))
    spec[freqs < 0.3] = 0
    w0 = 2 * math.pi * freqs[int(np.argmax(spec))]
    A0 = (yc.max() - yc.min()) / 2
    best = None
    for phi0 in np.linspace(0, 2 * math.pi, 8, endpoint=False):
        try:
            p, cov = curve_fit(_damped, t, y, p0=[y.mean(), A0, 0.1, w0, phi0], maxfev=20000)
        except RuntimeError:
            continue
        res = float(np.sum((_damped(t, *p) - y) ** 2))
        if p[1] < 0:   # keep A positive so peaks are at ωt+φ = 2πn
            p[1], p[4] = -p[1], p[4] + math.pi
        if best is None or res < best[2]:
            best = (p, cov, res)
    if best is None:
        raise RuntimeError("couldn't fit the oscillation")
    return best[0], best[1]


def _peaks(t: np.ndarray, y: np.ndarray, period: float) -> list[float]:
    """Times of the crests, refined with a parabola through each crest and its neighbours. A crest must be the
    highest point within half a period either side and above the midline, so tracking jitter isn't a peak."""
    out = []
    mid = float(np.median(y))
    for i in range(1, len(y) - 1):
        near = np.abs(t - t[i]) <= period / 2
        if y[i] > mid and y[i] >= y[near].max() and y[i] > y[i - 1]:
            a = 0.5 * (y[i - 1] - 2 * y[i] + y[i + 1])
            b = 0.5 * (y[i + 1] - y[i - 1])
            shift = -b / (2 * a) if a != 0 else 0.0
            out.append(float(t[i] + shift * (t[i + 1] - t[i])))
    return out


def analyse(tracks: dict[str, ObjectTrack], card, params: dict, ledger, agent, log, video: dict,
            px_per_m=None) -> dict:
    ref, mys = tracks["ref"], tracks["mystery"]
    fps, H = video["fps"], video["height"]
    cal = calibrate([ref, mys], H, params.get("weight_diameter_m"), px_per_m)
    m_ref = float(params["reference_mass_g"]) / 1000
    t_lock = float(params.get("lock_after_s", 2.0))
    g0 = STANDARD_GRAVITY

    tl = Timeline()
    tl.add(ref.t[0], "narration", text=card.narration["intro"])
    tl.add(ref.t[0], "calibration", method=cal.method, px_per_m=round(cal.px_per_m, 3),
           rel_uncertainty=round(cal.rel_uncertainty, 4))
    add_tracks(tl, [ref, mys], cal, fps)

    fits = {}
    for tr in (ref, mys):
        _, y = cal.to_world(tr.x_px, tr.y_px)
        early = tr.until(t_lock)
        p, cov = fit_oscillation(tr.t[early], y[early])
        fits[tr.id] = (p, cov, y)
        name = "Known weight" if tr.id == "ref" else "Mystery weight"
        discover_law(tl, t_lock, [(tr.t[early], y[early])], fps, "y", f"{name}, up and down", tr.id, scale_rel=cal.rel_uncertainty)
    (pr, cr, _), (pm, cm, ym) = fits["ref"], fits["mystery"]

    Sr = samples(pr, cr, seed=3)
    Sm = samples(pm, cm, seed=4)
    ratio = (Sr[:, 3] / Sm[:, 3]) ** 2
    mass_g = m_ref * ratio * 1000
    # Real springs aren't ideal (their own mass adds about a third of it to each weight, coils aren't perfectly
    # linear), so the interval never claims better than ±2% (override with params["mass_systematic_frac"]).
    floor = float(params.get("mass_systematic_frac", 0.02)) * float(np.median(mass_g))
    mass_val, mass_ci = interval(mass_g, floor=floor)
    k_spring = m_ref * pr[3] ** 2
    T_ref, T_mys = 2 * math.pi / pr[3], 2 * math.pi / pm[3]
    # Independent check from stretch: the heavier weight hangs lower by Δ = g·Δm/k (needs the metre scale).
    stretch = pr[0] - pm[0]
    mass_static = (m_ref + k_spring * stretch / g0) * 1000

    tl.add(t_lock, "measurement", items=[
        measurement("Reference period", T_ref, 2 * math.pi * math.sqrt(cr[3, 3]) / pr[3] ** 2, "s"),
        measurement("Mystery period", T_mys, 2 * math.pi * math.sqrt(cm[3, 3]) / pm[3] ** 2, "s"),
        measurement("Spring stiffness", k_spring, None, "N/m"),
        measurement("Mass from stretch (check)", mass_static, None, "g"),
    ])
    tl.add(t_lock, "run_state", state="locked", message="Mass and next peaks locked")

    based = round(float(mys.t[mys.until(t_lock)][-1]), 4)
    mass_pred = ledger.lock({"id": "mass", "text": f"Mystery weighs {mass_val:.0f} g", "quantity": "mass",
                             "value": round(mass_val, 1), "unit": "g",
                             "ci90": [round(mass_ci[0], 1), round(mass_ci[1], 1)],
                             "target_t": round(float(mys.t[-1]), 4), "based_on_until_t": based,
                             "object": "mystery"})
    # Next peaks of the mystery weight after the lock: ωt + φ = 2πn.
    w, phi = pm[3], pm[4]
    n0 = math.ceil((w * t_lock + phi) / (2 * math.pi))
    n_target = n0 + 2
    peak_times = (2 * math.pi * n_target - Sm[:, 4]) / Sm[:, 3]
    pk_val, pk_ci = interval(peak_times, floor=1.0 / fps)
    ghost = []
    x_m, _ = cal.to_world(mys.x_px, mys.y_px)
    for tt in np.linspace(t_lock, pk_val, 40):
        gx, gy = cal.to_image(np.array(float(np.median(x_m))), np.array(_damped(tt, *pm)))
        ghost.append([round(float(tt), 4), round(float(gx), 1), round(float(gy), 1)])
    peak_pred = ledger.lock({"id": "peak", "text": f"Mystery's 3rd next peak at {pk_val:.3f} s", "quantity": "peak_time",
                             "value": round(pk_val, 4), "unit": "s", "ci90": [round(pk_ci[0], 4), round(pk_ci[1], 4)],
                             "target_t": round(pk_val, 4), "based_on_until_t": based, "ghost": ghost,
                             "object": "mystery"})
    tl.add(t_lock, "prediction", **mass_pred)
    tl.add(t_lock, "prediction", **peak_pred)
    tl.add(t_lock, "narration", text=card.narration["locked"].format(mass=mass_val))

    ev = {"period_ref_s": T_ref, "period_mystery_s": T_mys, "ref_mass_g": m_ref * 1000, "mass_pred_g": mass_val,
          "mass_static_g": mass_static, "spring_k": k_spring}
    verdicts = []
    after = mys.t > t_lock
    actual_peaks = [p for p in _peaks(mys.t[after], ym[after], T_mys) if p > t_lock]
    t_explain = pk_val + 0.1
    if len(actual_peaks) >= 3:
        v = ledger.verify("peak", round(actual_peaks[2], 4))
        tl.add(actual_peaks[2], "verdict", **v)
        verdicts.append(v)
    if params.get("true_mass_g"):
        truth = float(params["true_mass_g"])
        v = ledger.verify("mass", round(truth, 1))
        t_reveal = min(float(mys.t[-1]), t_explain)
        tl.add(t_reveal, "verdict", **v)
        tl.add(t_reveal, "narration", text=card.narration["hit" if v["hit"] else "miss"].format(mass=truth))
        verdicts.append(v)
        ev["mass_true_g"] = truth
    out = explain(agent, tl, t_explain, ev, template, GLOSSARY, ["mass", "peak"])
    tl.emit(log)
    return {"experiment": "springs", "periods": (T_ref, T_mys), "mass_pred_g": mass_val, "mass_ci_g": mass_ci,
            "mass_static_g": mass_static, "spring_k": k_spring, "verdicts": verdicts,
            "t_end": max(float(mys.t[-1]), t_explain), **out}
