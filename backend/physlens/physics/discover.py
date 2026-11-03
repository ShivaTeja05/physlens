"""Law discovery: find the equation of motion from the track instead of picking one from a fixed list.

Sparse regression in the spirit of SINDy (Brunton, Proctor & Kutz, 2016). Velocity and acceleration come from
Savitzky-Golay derivatives of each continuous stretch of track (never across a bounce). Acceleration along one axis is
then written as a combination of a small library of physical terms:

    1        a constant pull (gravity, a slope)
    q        a pull back towards a centre (a spring: Hooke's law)
    v        drag that grows with speed
    |v|·v    drag that grows with speed squared (air at higher speeds)

Every combination is fitted by least squares (12 candidates including "no force at all"; a spring always comes
with its centre). A candidate counts only if
each of its terms is clearly non-zero (|coef| ≥ 4 standard errors), and the simplest well-supported one wins by BIC.
The output stays a formula a person can read and check: no network, no memory, just what this video shows.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import savgol_filter

TERMS = ("1", "q", "v", "|v|v")
SHORTLIST = 4  # equations re-judged on raw positions
MIN_T = 4.0   # each kept term must be at least this many standard errors from zero


@dataclass
class Candidate:
    terms: tuple[str, ...]
    coef: dict[str, tuple[float, float]]   # term → (value, standard error)
    rss: float
    bic: float
    valid: bool
    weight: float = 0.0


@dataclass
class Law:
    axis: str                 # "y", "x" or "s" (along a slope)
    terms: dict[str, tuple[float, float]]
    n: int
    rmse: float               # metres, position residual of the winning equation
    candidates: list[Candidate] = field(default_factory=list)

    @property
    def n_searched(self) -> int:
        return len(self.candidates)

    def equation(self, unit_pos: str = "") -> str:
        """Readable equation, e.g. 'a = −9.80' or 'a = −88.7·(y − y₀) − 0.21·v'."""
        q = {"y": "y", "x": "x", "s": "s"}.get(self.axis, "q")
        parts = []
        c = self.terms.get("1", (0.0, 0.0))[0]
        if "q" in self.terms:
            k = -self.terms["q"][0]
            parts.append(f"{_num(-k)}·({q} − {q}₀)")
        elif "1" in self.terms:
            parts.append(_num(c))
        if "v" in self.terms:
            parts.append(f"{_num(self.terms['v'][0])}·v")
        if "|v|v" in self.terms:
            parts.append(f"{_num(self.terms['|v|v'][0])}·|v|v")
        if not parts:
            return "a = 0"
        s = "a = " + parts[0]
        for p in parts[1:]:
            s += (" − " + p[1:]) if p.startswith("−") else (" + " + p)
        return s

    def meaning(self) -> list[str]:
        """Plain words for each term the data needed."""
        out = []
        if not self.terms:
            out.append("No force along this direction: it keeps a constant speed.")
        if "q" in self.terms:
            w2 = -self.terms["q"][0]
            if w2 > 0:
                out.append(f"Pulled back towards a centre in proportion to distance (Hooke's law): "
                           f"ω² = {w2:.1f} /s², one bob every {2 * math.pi / math.sqrt(w2):.3f} s.")
            else:
                out.append("Pushed away from a centre in proportion to distance (unstable).")
        elif "1" in self.terms:
            c, e = self.terms["1"]
            if self.axis == "y":
                out.append(f"A constant pull {'down' if c < 0 else 'UP'} of {abs(c):.2f} ± {e:.2f} m/s²"
                           + (": gravity." if c < 0 else ": gravity points up, so the picture is upside down."))
            else:
                out.append(f"A constant push of {abs(c):.2f} ± {e:.2f} m/s² along the motion (a slope or a steady force).")
        if "v" in self.terms:
            k = -self.terms["v"][0]
            out.append(f"Drag in proportion to speed: k = {k:.2f} /s." if k > 0
                       else "A push that grows with speed: not passive physics (sped up, reversed, or driven).")
        if "|v|v" in self.terms:
            k = -self.terms["|v|v"][0]
            out.append(f"Drag in proportion to speed squared (air): c = {k:.3f} /m." if k > 0
                       else "A push growing with speed squared: not passive physics.")
        return out

    def to_event(self, id_: str, label: str, obj: str | None = None) -> dict:
        ev = {"id": id_, "label": label, "axis": self.axis, "equation": self.equation(),
              "terms": [{"term": k, "value": round(v, 4), "err": round(e, 4)} for k, (v, e) in self.terms.items()],
              "meaning": self.meaning(), "searched": self.n_searched, "samples": self.n,
              "rmse": round(self.rmse, 4),
              "alternatives": [{"name": _describe(c.terms), "equation": Law(self.axis, _centred(c.coef), self.n, 0.0).equation(),
                                "score": round(c.weight, 4)} for c in sorted([c for c in self.candidates if c.valid], key=lambda c: -c.weight)[:3]]}
        if obj:
            ev["object"] = obj
        return ev

    def race(self, label: str, top: int = 4) -> dict:
        """The search as a hypothesis race: the best few equations and how strongly the data prefer each."""
        best = sorted([c for c in self.candidates if c.valid], key=lambda c: -c.weight)[:top]
        return {"topic": f"discover:{self.axis}", "label": label, "candidates": [
            {"name": _describe(c.terms), "equation": Law(self.axis, _centred(c.coef), self.n, 0.0).equation(),
             "score": round(c.weight, 4)} for c in best]}


def _centred(coef: dict) -> dict:
    return {k: v for k, v in coef.items() if not (k == "1" and "q" in coef)}


def _num(x: float) -> str:
    a = abs(x)
    s = f"{a:.3g}" if a < 0.1 else f"{a:.2f}" if a < 100 else f"{a:.0f}"
    return ("−" if x < 0 else "") + s


def _describe(terms: tuple[str, ...]) -> str:
    names = {"1": "constant pull", "q": "spring", "v": "drag ∝ v", "|v|v": "drag ∝ v²"}
    shown = [t for t in terms if not (t == "1" and "q" in terms)]   # a spring's constant is just its centre
    return " + ".join(names[t] for t in shown) if shown else "no force"


def derivatives(t: np.ndarray, q: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smoothed position, velocity and acceleration of one continuous stretch of track. Returns empty arrays if the
    stretch is too short to differentiate twice."""
    win = max(7, int(round(0.08 * fps)) | 1)
    if len(q) < win + 2:
        win = len(q) - (1 - len(q) % 2) - 2  # largest odd window that still leaves a trimmed core
    if win < 5:
        return np.array([]), np.array([]), np.array([])
    dt = 1.0 / fps
    # Quartic fits keep fast wobbles (springs) from being smoothed flat; short windows can only afford cubics.
    order = 4 if win >= 9 else 3
    qs = savgol_filter(q, win, order)
    v = savgol_filter(q, win, order, deriv=1, delta=dt)
    a = savgol_filter(q, win, order, deriv=2, delta=dt)
    h = win // 2       # the ends of a stretch have one-sided windows: drop them
    core = slice(h, len(q) - h) if len(q) > 2 * h + 3 else slice(0, len(q))
    return qs[core], v[core], a[core]


def _refine(c: Candidate, pieces, q_centre: float) -> Candidate | None:
    """Fit candidate equation c directly to positions: its coefficients plus a start position and speed for each
    stretch. Returns it with coefficients, errors and BIC from the position residuals."""
    from scipy.optimize import least_squares

    pieces = [(t[::2], q[::2]) if len(t) > 160 else (t, q) for t, q in pieces if len(t) >= 5]
    if not pieces:
        return None
    terms = c.terms

    idx = {k: i for i, k in enumerate(terms)}

    def accel(qq, vv, beta):
        a = beta[idx["1"]] if "1" in idx else 0.0
        if "q" in idx:
            a += beta[idx["q"]] * (qq - q_centre)
        if "v" in idx:
            a += beta[idx["v"]] * vv
        return a + beta[idx["|v|v"]] * abs(vv) * vv

    def simulate(t, q0, v0, beta):
        if "|v|v" not in terms:
            # Linear equation: step it exactly with the matrix exponential (evenly spaced samples).
            from scipy.linalg import expm
            co = dict(zip(terms, beta))
            M = np.array([[0.0, 1.0, 0.0], [co.get("q", 0.0), co.get("v", 0.0), co.get("1", 0.0)], [0.0, 0.0, 0.0]])
            steps: dict[float, np.ndarray] = {}   # one matrix per distinct frame gap (a missed frame is a 2-frame step)
            z = np.array([q0 - q_centre, v0, 1.0])
            out = np.empty(len(t))
            for i in range(len(t)):
                out[i] = z[0] + q_centre
                if i + 1 < len(t):
                    dt = round(float(t[i + 1] - t[i]), 9)
                    if dt not in steps:
                        steps[dt] = expm(M * dt)
                    z = steps[dt] @ z
            return out
        # Speed-squared drag: fixed-step Runge-Kutta, 4 sub-steps per frame (much faster than an adaptive solver here).
        qq, vv = q0, v0
        out = np.empty(len(t))
        for i in range(len(t)):
            out[i] = qq
            if i + 1 == len(t):
                break
            n_sub = max(4, int(round(4 * (t[i + 1] - t[i]) / float(np.median(np.diff(t))))))
            h = float(t[i + 1] - t[i]) / n_sub
            for _ in range(n_sub):
                k1q, k1v = vv, accel(qq, vv, beta)
                k2q, k2v = vv + 0.5 * h * k1v, accel(qq + 0.5 * h * k1q, vv + 0.5 * h * k1v, beta)
                k3q, k3v = vv + 0.5 * h * k2v, accel(qq + 0.5 * h * k2q, vv + 0.5 * h * k2v, beta)
                k4q, k4v = vv + h * k3v, accel(qq + h * k3q, vv + h * k3v, beta)
                qq += h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
                vv += h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
            if not np.isfinite(qq) or abs(qq) > 1e6:
                return np.full(len(t), 1e3)
        return out

    def residual(p):
        beta = p[:len(terms)]
        out = []
        with np.errstate(over="ignore", invalid="ignore"):   # a bad trial can blow up; simulate() catches it
            for j, (t, q) in enumerate(pieces):
                out.append(simulate(t, p[len(terms) + 2 * j], p[len(terms) + 2 * j + 1], beta) - q)
        return np.concatenate(out)

    p0 = [c.coef[k][0] for k in terms]
    for t, q in pieces:
        m = min(len(t), 7)
        cf = np.polyfit(t[:m] - t[0], q[:m], 2)
        p0 += [float(cf[2]), float(cf[1])]
    try:
        fit = least_squares(residual, np.array(p0, float), x_scale="jac", max_nfev=400)
    except (ValueError, FloatingPointError):
        return None
    r = fit.fun
    n, k = len(r), len(p0)
    rss = float(r @ r)
    try:
        cov = np.linalg.inv(fit.jac.T @ fit.jac) * rss / max(n - k, 1)
        se = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        return None
    coef = {kk: (float(fit.x[i]), float(se[i])) for i, kk in enumerate(terms)}
    valid = all(abs(b) >= MIN_T * e for kk, (b, e) in coef.items() if kk != "1" or "q" not in terms)
    bic = n * math.log(max(rss / n, 1e-300)) + k * math.log(n)
    return Candidate(terms, coef, rss, bic, valid)


def discover(segments: list[tuple[np.ndarray, np.ndarray]], fps: float, axis: str = "y",
             speed_segments: list[np.ndarray] | None = None, terms: tuple[str, ...] = TERMS,
             scale_rel: float = 0.0) -> Law:
    """segments: continuous (t, q) stretches in metres (no bounces inside). speed_segments: optional full speed |v|
    for each stretch (for 2D drag); defaults to |v| along this axis. scale_rel: relative uncertainty of the
    pixel-to-metre scale, added to the terms whose value depends on it (a constant pull in m/s², speed-squared drag
    in 1/m). A spring's ω² and speed-drag's k are in 1/s² and 1/s and don't depend on the scale."""
    # Derivatives need evenly spaced samples: split a stretch wherever the tracker lost the object for a frame or more.
    pieces, speeds = [], []
    for i, (t, q) in enumerate(segments):
        t, q = np.asarray(t, float), np.asarray(q, float)
        cuts = np.nonzero(np.diff(t) > 1.5 / fps)[0] + 1
        for lo, hi in zip(np.r_[0, cuts], np.r_[cuts, len(t)]):
            pieces.append((t[lo:hi], q[lo:hi]))
            speeds.append(None if speed_segments is None else np.asarray(speed_segments[i], float)[lo:hi])
    Q, V, A, S = [], [], [], []
    for i, (t, q) in enumerate(pieces):
        qs, v, a = derivatives(t, q, fps)
        if not len(a):
            continue
        Q.append(qs); V.append(v); A.append(a)
        if speed_segments is not None:
            sp = speeds[i]
            h = (len(sp) - len(a)) // 2
            S.append(sp[h:h + len(a)])
        else:
            S.append(np.abs(v))
    if not A:
        raise ValueError("not enough continuous track to find a law")
    q, v, a, sp = (np.concatenate(z) for z in (Q, V, A, S))
    n = len(a)
    cols = {"1": np.ones(n), "q": q - q.mean(), "v": v, "|v|v": sp * v}

    cands = []
    for k in range(len(terms) + 1):
        for combo in itertools.combinations(terms, k):
            if "q" in combo and "1" not in combo:
                continue   # a spring always comes with its centre (the constant), so q alone isn't a separate law
            if combo:
                X = np.stack([cols[c] for c in combo], axis=1)
                beta, *_ = np.linalg.lstsq(X, a, rcond=None)
                resid = a - X @ beta
            else:
                beta, resid, X = np.array([]), a, np.zeros((n, 0))
            rss = float(resid @ resid)
            dof = max(n - len(combo), 1)
            sigma2 = rss / dof
            se = np.sqrt(np.diag(sigma2 * np.linalg.pinv(X.T @ X))) if combo else np.array([])
            coef = {c: (float(b), float(e)) for c, b, e in zip(combo, beta, se)}
            # The constant is always allowed (it's where the centre of a spring sits); other terms must be clear.
            valid = all(abs(b) >= MIN_T * e for c, (b, e) in coef.items() if c != "1" or "q" not in combo)
            bic = n * math.log(max(rss / n, 1e-300)) + len(combo) * math.log(n)
            cands.append(Candidate(combo, coef, rss, bic, valid))

    # Derivatives are biased by smoothing (fast wobbles flattened, low frame rates) and noisy, so they only shortlist.
    # The shortlist is then judged on the raw positions: each equation is integrated from fitted starting points
    # and fitted to where the object actually was.
    # The shortlist always keeps the simplest laws too, so an over-eager derivative fit can't crowd them out.
    shortlist = sorted(cands, key=lambda c: c.bic)[:SHORTLIST]
    shortlist += [c for c in cands if len(c.terms) <= 1 and c not in shortlist]
    q_centre = float(q.mean())
    # Positions can be fitted straight across a missed frame (unlike derivatives), so refine on whole stretches.
    whole = [(np.asarray(t, float), np.asarray(q, float)) for t, q in segments]
    refined = [(c, _refine(c, whole, q_centre)) for c in shortlist]
    if any(r is not None for _, r in refined):
        # Everything not judged on positions (outside the shortlist, or its position fit failed) can't win.
        for c in cands:
            c.valid = False
        cands = [r if r is not None else c for c, r in refined] + [c for c in cands if c not in shortlist]
    ok = [c for c in cands if c.valid]
    b0 = min(c.bic for c in ok)
    for c in ok:
        c.weight = math.exp(-0.5 * (c.bic - b0))
    total = sum(c.weight for c in ok)
    for c in ok:
        c.weight /= total
    best = min(ok, key=lambda c: c.bic)
    # A spring's constant is its centre, not a separate force: a = −ω²(q − q₀).
    n_pos = sum(len(t) for t, _ in whole)
    coef = {k: (v, math.hypot(e, abs(v) * scale_rel) if k in ("1", "|v|v") else e)
            for k, (v, e) in _centred(best.coef).items()}
    return Law(axis, coef, n_pos, math.sqrt(best.rss / n_pos), cands)
