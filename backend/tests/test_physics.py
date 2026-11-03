import numpy as np
import pytest

from physlens.physics import (
    akaike_weights, consistent_flights, find_bounces, fit_all, fit_free_fall, fit_horizontal, measure_flight,
    predict_next_flight, split_flights,
)
from physlens.synth import DropScene, simulate

PIXEL_NOISE_M = 0.5 / 480  # half a pixel of centroid noise at 480 px/m


def observe(scene: DropScene, seed: int = 1):
    t, x, y = simulate(scene)
    rng = np.random.default_rng(seed)
    return t, x + rng.normal(0, PIXEL_NOISE_M, x.shape), y + rng.normal(0, PIXEL_NOISE_M, y.shape)


def flights_for(scene: DropScene):
    t, x, y = observe(scene)
    bounces = find_bounces(t, y, scene.ball_diameter / 2)
    return t, x, y, consistent_flights(t, y, split_flights(t, bounces, scene.fps))


@pytest.mark.parametrize("scene,expected_g", [
    (DropScene(), 9.81),
    (DropScene(time_scale=2.0, duration=6.0), 9.81 / 4),
    (DropScene(g=1.81, duration=7.0, restitution=0.7), 1.81),
])
def test_recovers_apparent_gravity(scene, expected_g):
    t, x, y, flights = flights_for(scene)
    g, g_err = fit_free_fall(t, y, flights).shared["g"]
    assert abs(g - expected_g) / expected_g < 0.01
    assert g_err < 0.01 * expected_g


def test_free_fall_wins_the_hypothesis_race():
    t, x, y, flights = flights_for(DropScene())
    fits = fit_all(t, y, flights)
    weights = dict(zip([f.model for f in fits], akaike_weights(fits)))
    assert weights["constant_velocity"] < 1e-6
    assert weights["free_fall"] > weights["constant_velocity"]
    assert weights["free_fall"] + weights["linear_drag"] > 0.999


def test_predicts_the_next_bounce_from_earlier_flights_only():
    scene = DropScene()
    t, x, y, flights = flights_for(scene)
    seen = flights[:2]                       # release flight + first bounce, lock at the second contact
    fit = fit_free_fall(t, y, seen)
    vx = fit_horizontal(t, x, seen)[-1]
    contact = seen[-1].bounce_after
    pred = predict_next_flight(fit, t, seen, vx, float(x[contact]))

    assert abs(pred.restitution.value - scene.restitution) < 0.02
    actual = measure_flight(t, y, flights[2], pred.contact_height)
    lo, hi = pred.peak_height.ci90
    assert lo <= actual["peak_height"] <= hi
    lo, hi = pred.landing_time.ci90
    assert lo <= actual["landing_time"] <= hi
