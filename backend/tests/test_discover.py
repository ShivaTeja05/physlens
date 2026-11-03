"""Law discovery must find the right equation of motion, not just a good fit, from noisy tracks."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp

from physlens.physics.discover import discover

NOISE_M = 0.0005   # 0.5 mm, about a quarter pixel at 500 px/m


def track(f, T, fps, seed, y0=1.0, v0=1.5):
    rng = np.random.default_rng(seed)
    t = np.arange(0, T, 1 / fps)
    s = solve_ivp(lambda _, z: [z[1], f(z[0], z[1])], (0, T), [y0, v0], t_eval=t, rtol=1e-10, atol=1e-12)
    return t, s.y[0] + rng.normal(0, NOISE_M, len(t))


CASES = [
    # name, a(y, v), duration, fps, expected terms {term: value}
    ("free fall", lambda y, v: -9.81, 0.6, 60, {"1": -9.81}),
    ("free fall at 25 fps", lambda y, v: -9.81, 0.6, 25, {"1": -9.81}),
    ("slow motion 2x", lambda y, v: -9.81 / 4, 0.8, 60, {"1": -2.4525}),
    ("upside down", lambda y, v: 9.81, 0.6, 60, {"1": 9.81}),
    ("drag in proportion to speed", lambda y, v: -9.81 - 0.8 * v, 0.6, 60, {"1": -9.81, "v": -0.8}),
    ("air drag (speed squared)", lambda y, v: -9.81 - 0.25 * abs(v) * v, 0.6, 60, {"1": -9.81, "|v|v": -0.25}),
    ("damped spring", lambda y, v: -200 * (y - 0.3) - 0.4 * v, 2.0, 120, {"q": -200, "v": -0.4}),
    ("damped spring at 25 fps", lambda y, v: -80 * (y - 0.3) - 0.4 * v, 2.0, 25, {"q": -80, "v": -0.4}),
    ("no force", lambda y, v: 0.0, 0.6, 60, {}),
]


@pytest.mark.parametrize("name,f,T,fps,expected", CASES, ids=[c[0] for c in CASES])
def test_finds_the_law(name, f, T, fps, expected):
    law = discover([track(f, T, fps, seed) for seed in (1, 2)], fps)
    assert set(law.terms) == set(expected), f"{name}: found {law.equation()}"
    for term, true in expected.items():
        value, err = law.terms[term]
        assert abs(value - true) <= max(4 * err, 0.02 * abs(true)), f"{name}: {term} = {value} ± {err}, true {true}"


def test_split_at_tracking_gaps():
    """A missed frame must not be differentiated across (it would look like a huge acceleration)."""
    t, y = track(lambda y, v: -9.81, 0.8, 60, seed=3)
    keep = np.ones(len(t), bool)
    keep[20:23] = False
    law = discover([(t[keep], y[keep])], 60)
    assert set(law.terms) == {"1"} and abs(law.terms["1"][0] + 9.81) < 0.05


def test_readable_output():
    law = discover([track(lambda y, v: -200 * (y - 0.3) - 0.4 * v, 2.0, 120, seed=4)], 120)
    assert law.equation().startswith("a = −200·(y − y₀)")
    assert "Hooke's law" in law.meaning()[0]
    ev = law.to_event("law_y", "test")
    assert ev["equation"] == law.equation() and ev["alternatives"][0]["score"] > 0.9
