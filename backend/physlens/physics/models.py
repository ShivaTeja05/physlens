"""Candidate laws for vertical motion between contacts, fitted jointly across all flights.

Each model shares its physical constants (g, k) across flights and gives every flight its own launch
height y0 and launch speed v0. Times inside a flight are measured from that flight's first sample.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from .segment import Flight


@dataclass
class FitResult:
    model: str
    equation: str
    shared: dict[str, tuple[float, float]]          # physical constants: name -> (value, 1-sigma)
    per_flight: list[dict[str, float]]              # y0, v0 for each flight
    theta: np.ndarray                               # full parameter vector
    cov: np.ndarray                                 # its covariance
    rss: float
    n: int
    k: int
    window: tuple[float, float]
    aic: float = field(init=False)

    def __post_init__(self):
        self.aic = self.n * np.log(max(self.rss, 1e-300) / self.n) + 2 * self.k

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.rss / self.n))


def _stack(t: np.ndarray, y: np.ndarray, flights: list[Flight]):
    taus, ys, which = [], [], []
    for j, f in enumerate(flights):
        taus.append(t[f.start:f.stop] - f.t_start)
        ys.append(y[f.start:f.stop])
        which.append(np.full(f.stop - f.start, j))
    return np.concatenate(taus), np.concatenate(ys), np.concatenate(which)


def _linear_fit(A: np.ndarray, y: np.ndarray):
    theta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ theta
    rss = float(resid @ resid)
    dof = max(len(y) - A.shape[1], 1)
    cov = (rss / dof) * np.linalg.pinv(A.T @ A)
    return theta, cov, rss


def fit_constant_velocity(t, y, flights) -> FitResult:
    tau, ys, which = _stack(t, y, flights)
    F = len(flights)
    A = np.zeros((len(ys), 2 * F))
    for j in range(F):
        m = which == j
        A[m, 2 * j] = 1.0
        A[m, 2 * j + 1] = tau[m]
    theta, cov, rss = _linear_fit(A, ys)
    per = [{"y0": theta[2 * j], "v0": theta[2 * j + 1]} for j in range(F)]
    return FitResult("constant_velocity", "y = y0 + v0·t", {}, per, theta, cov, rss, len(ys), A.shape[1],
                     (float(t[flights[0].start]), float(t[flights[-1].stop - 1])))


def fit_free_fall(t, y, flights) -> FitResult:
    """y = y0 + v0·τ − ½·g·τ²  with one g for every flight (linear least squares)."""
    tau, ys, which = _stack(t, y, flights)
    F = len(flights)
    A = np.zeros((len(ys), 1 + 2 * F))
    A[:, 0] = -0.5 * tau ** 2
    for j in range(F):
        m = which == j
        A[m, 1 + 2 * j] = 1.0
        A[m, 2 + 2 * j] = tau[m]
    theta, cov, rss = _linear_fit(A, ys)
    per = [{"y0": theta[1 + 2 * j], "v0": theta[2 + 2 * j]} for j in range(F)]
    shared = {"g": (float(theta[0]), float(np.sqrt(cov[0, 0])))}
    return FitResult("free_fall", "y = y0 + v0·t − ½·g·t²", shared, per, theta, cov, rss, len(ys), A.shape[1],
                     (float(t[flights[0].start]), float(t[flights[-1].stop - 1])))


def _drag_curve(tau, y0, v0, g, k):
    # Solution of y'' = −g − k·y'. expm1 keeps it accurate for small k.
    decay = -np.expm1(-k * tau) / k
    return y0 + (v0 + g / k) * decay - (g / k) * tau


def fit_linear_drag(t, y, flights, start: FitResult) -> FitResult:
    """y'' = −g − k·y'  (air drag proportional to speed), initialised from the free-fall fit."""
    tau, ys, which = _stack(t, y, flights)
    F = len(flights)
    x0 = np.concatenate([[start.shared["g"][0], 0.05], np.ravel([[p["y0"], p["v0"]] for p in start.per_flight])])
    lower = np.concatenate([[0.0, 1e-4], np.full(2 * F, -np.inf)])
    upper = np.concatenate([[np.inf, 50.0], np.full(2 * F, np.inf)])

    def residuals(theta):
        g, k = theta[0], theta[1]
        y0 = theta[2::2][which]
        v0 = theta[3::2][which]
        return _drag_curve(tau, y0, v0, g, k) - ys

    sol = least_squares(residuals, x0, bounds=(lower, upper))
    rss = float(sol.fun @ sol.fun)
    dof = max(len(ys) - len(x0), 1)
    cov = (rss / dof) * np.linalg.pinv(sol.jac.T @ sol.jac)
    theta = sol.x
    per = [{"y0": theta[2 + 2 * j], "v0": theta[3 + 2 * j]} for j in range(F)]
    shared = {"g": (float(theta[0]), float(np.sqrt(cov[0, 0]))), "k": (float(theta[1]), float(np.sqrt(cov[1, 1])))}
    return FitResult("linear_drag", "y'' = −g − k·y'", shared, per, theta, cov, rss, len(ys), len(x0),
                     (float(t[flights[0].start]), float(t[flights[-1].stop - 1])))


def consistent_flights(t, y, flights, min_samples: int = 6) -> list[Flight]:
    """Drops flights that don't look like clean free flight on their own (merged tiny bounces, the ball
    being held, tracking glitches): each flight is fitted alone and must agree with the others."""
    if not flights:
        return []
    single = [(f, fit_free_fall(t, y, [f])) for f in flights if f.stop - f.start >= min_samples]
    if not single:
        return []
    g_med = np.median([r.shared["g"][0] for _, r in single])
    rmse_med = np.median([r.rmse for _, r in single])
    return [f for f, r in single
            if abs(r.shared["g"][0] - g_med) <= 0.2 * abs(g_med) and r.rmse <= 3 * rmse_med]


def fit_all(t, y, flights) -> list[FitResult]:
    ff = fit_free_fall(t, y, flights)
    return [fit_constant_velocity(t, y, flights), ff, fit_linear_drag(t, y, flights, ff)]


def akaike_weights(fits: list[FitResult]) -> list[float]:
    aic = np.array([f.aic for f in fits])
    rel = np.exp(-0.5 * (aic - aic.min()))
    return list(rel / rel.sum())


def fit_horizontal(t, x, flights) -> list[float]:
    """Horizontal speed in each flight (no horizontal force while airborne)."""
    out = []
    for f in flights:
        tt, xx = t[f.start:f.stop], x[f.start:f.stop]
        out.append(float(np.polyfit(tt - f.t_start, xx, 1)[0]))
    return out
