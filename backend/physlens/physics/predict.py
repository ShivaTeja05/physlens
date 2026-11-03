"""Bounce physics: restitution from observed contacts, and predictions for the next flight.

All predictions are computed with Monte Carlo over the free-fall fit's parameter covariance, so every
number comes with a 90% interval.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import FitResult
from .segment import Flight

N_SAMPLES = 4000


@dataclass
class Estimate:
    value: float
    ci90: tuple[float, float]
    std: float

    @classmethod
    def from_samples(cls, s: np.ndarray) -> "Estimate":
        s = s[np.isfinite(s)]
        lo, hi = np.percentile(s, [5, 95])
        return cls(float(np.median(s)), (float(lo), float(hi)), float(np.std(s)))


@dataclass
class ContactReading:
    impact_speed: Estimate       # m/s, speed arriving at the floor
    restitution: Estimate        # e = speed leaving / speed arriving


@dataclass
class BounceLaw:
    name: str
    equation: str
    weight: float                # relative support, 0..1


@dataclass
class BouncePrediction:
    restitution: Estimate        # bounciness used for the predicted bounce
    contacts: list[ContactReading]  # what each observed contact before the lock showed
    laws: list[BounceLaw]
    chosen_law: str
    launch_speed: Estimate       # speed leaving the lock contact, m/s
    peak_height: Estimate        # apex above the contact height, m
    peak_time: Estimate          # video time of the apex
    landing_time: Estimate       # video time of the next contact
    contact_height: float        # world y of the ball centre at the lock contact, m
    contact_time: float
    ghost: list[tuple[float, float, float]]  # (t, x_m, y_m) along the predicted flight


def _sample_theta(fit: FitResult, rng: np.random.Generator) -> np.ndarray:
    return rng.multivariate_normal(fit.theta, fit.cov, size=N_SAMPLES, method="eigh")


def _poly(theta, j: int, flight: Flight):
    """Flight j as y(t) = −½·g·t² + b·t + c in video time (vectorised over parameter samples)."""
    g = theta[..., 0]
    y0 = theta[..., 1 + 2 * j]
    v0 = theta[..., 2 + 2 * j]
    s = flight.t_start
    return g, v0 + g * s, y0 - v0 * s - 0.5 * g * s ** 2


def contact_samples(theta, flights: list[Flight], j_before: int):
    """Contact between flight j_before and the next one, where the two fitted curves meet. Both share g,
    so their difference is linear in t. Returns (time, height, restitution e = −v_out / v_in)."""
    g, b1, c1 = _poly(theta, j_before, flights[j_before])
    _, b2, c2 = _poly(theta, j_before + 1, flights[j_before + 1])
    tc = (c2 - c1) / (b1 - b2)
    yc = -0.5 * g * tc ** 2 + b1 * tc + c1
    e = -(b2 - g * tc) / (b1 - g * tc)
    return tc, yc, e


def restitution_samples(theta, flights: list[Flight], j_before: int) -> np.ndarray:
    return contact_samples(theta, flights, j_before)[2]


CONSTANT = BounceLaw("constant_restitution", "e = constant", 1.0)
RESTITUTION_SYSTEMATIC = 0.01


def _bounce_laws(readings: list[tuple[np.ndarray, np.ndarray]], v_next: np.ndarray):
    """Pick the bounce law from the contacts seen so far and return (laws, chosen name, e samples for the
    next contact). One contact only supports a constant e. With two, we check whether e held constant:
    the constant law is scored by how well the earlier e predicts the later one; the speed-dependent law
    pays an Akaike-style penalty for its extra parameter. If e moved, it is extrapolated linearly in
    impact speed from the last two contacts (real balls usually lose more energy in harder impacts)."""
    v_last, e_last = readings[-1]
    if len(readings) < 2:
        return [CONSTANT], CONSTANT.name, e_last
    v_prev, e_prev = readings[-2]
    # Statistical spreads alone can be tiny (±0.0005 on a clean clip), so add a systematic floor per contact
    # (tracking bias, ball spin, uneven floor) before deciding that bounciness really changed.
    var = np.var(e_prev) + np.var(e_last) + 2 * RESTITUTION_SYSTEMATIC ** 2
    z = (np.median(e_last) - np.median(e_prev)) / np.sqrt(var)
    support = np.array([np.exp(-0.5 * z ** 2), np.exp(-1.0)])
    w_const, w_speed = support / support.sum()
    laws = [BounceLaw("constant_restitution", "e = constant", float(w_const)),
            BounceLaw("speed_dependent_restitution", "e = e₀ + k·v", float(w_speed))]
    if w_speed > w_const:
        slope = (e_last - e_prev) / (v_last - v_prev)
        return laws, laws[1].name, e_last + slope * (v_next - v_last)
    return laws, laws[0].name, e_last


def predict_next_flight(fit: FitResult, t: np.ndarray, flights: list[Flight], vx: float, x_at_contact: float,
                        seed: int = 0) -> BouncePrediction:
    """Uses only the flights in `fit` (everything before the lock). The last flight ends at the lock
    contact; the floor height comes from the contact before it, and the bounce law from all earlier
    contacts."""
    if len(flights) < 2 or flights[-1].bounce_after is None:
        raise ValueError("need two complete flights before the lock contact")
    rng = np.random.default_rng(seed)
    theta = _sample_theta(fit, rng)
    last = len(flights) - 1

    readings = []
    for j in range(last):
        tc, yc, e = contact_samples(theta, flights, j)
        g_s, b_s, _ = _poly(theta, j, flights[j])
        readings.append((g_s * tc - b_s, e))   # impact speed (positive, m/s), restitution
        contact_y = yc

    # The incoming flight reaches the floor height on its way down.
    g, b, c = _poly(theta, last, flights[last])
    t_contact_s = (b + np.sqrt(b ** 2 + 2 * g * (c - contact_y))) / g
    v_impact = g * t_contact_s - b
    laws, chosen, e = _bounce_laws(readings, v_impact)
    v_out = e * v_impact
    t_contact = float(np.median(t_contact_s))

    peak_h = v_out ** 2 / (2 * g)
    peak_t = t_contact_s + v_out / g
    land_t = t_contact_s + 2 * v_out / g

    v_mid, g_mid, y_mid = float(np.median(v_out)), float(np.median(g)), float(np.median(contact_y))
    duration = 2 * v_mid / g_mid
    ghost = []
    for tau in np.linspace(0.0, duration, 48):
        ghost.append((t_contact + tau, x_at_contact + vx * tau, y_mid + v_mid * tau - 0.5 * g_mid * tau ** 2))

    return BouncePrediction(
        restitution=Estimate.from_samples(e),
        contacts=[ContactReading(Estimate.from_samples(v), Estimate.from_samples(r)) for v, r in readings],
        laws=laws,
        chosen_law=chosen,
        launch_speed=Estimate.from_samples(v_out),
        peak_height=Estimate.from_samples(peak_h),
        peak_time=Estimate.from_samples(peak_t),
        landing_time=Estimate.from_samples(land_t),
        contact_height=y_mid,
        contact_time=t_contact,
        ghost=ghost,
    )


def measure_flight(t: np.ndarray, y: np.ndarray, flight: Flight, contact_height: float) -> dict[str, float]:
    """What actually happened in one flight: apex height above the contact height, apex time, and the
    time it came back down to that height. Uses a local parabola fit for sub-frame precision."""
    tt, yy = t[flight.start:flight.stop], y[flight.start:flight.stop]
    a, b, c = np.polyfit(tt, yy, 2)
    t_peak = -b / (2 * a)
    y_peak = c - b ** 2 / (4 * a)
    roots = np.roots([a, b, c - contact_height])
    roots = np.real(roots[np.isreal(roots)])
    t_land = float(roots.max()) if roots.size else float("nan")
    return {"peak_height": float(y_peak - contact_height), "peak_time": float(t_peak), "landing_time": t_land}
