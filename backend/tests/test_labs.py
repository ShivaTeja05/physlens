"""Stage experiments on rendered clips with known physics (offline colour tracker, template explanations)."""
import pytest

from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger, prediction_hash
from physlens.perception import ColorBlobTracker
from physlens.pipeline import run_pipeline
from physlens.synth_labs import (ProjectileScene, RaceScene, SpringScene, render_projectile, render_race,
                                 render_springs)

OFFLINE = lambda spec: ColorBlobTracker(color=spec.color)


def run(card, path, **params):
    log = EventLog(card)
    summary = run_pipeline(path, CARDS[card], log, Ledger(), params=params, tracker_for=OFFLINE)
    for p in log.of_type("prediction"):
        # Uses only frames up to the moment it's locked, and the outcome comes after that.
        assert p["based_on_until_t"] <= p["t"] < p["target_t"]
        assert p["hash"] == prediction_hash(p)
    assert log.of_type("explain")[0]["verified"]
    return summary, log


def test_catch_the_cup(tmp_path):
    scene = ProjectileScene()
    s, log = run("cup", render_projectile(scene, tmp_path / "cup.mp4"))
    truth_cm = (scene.landing_x - scene.x_edge) * 100
    lo, hi = s["landing_ci_cm"]
    assert lo <= truth_cm <= hi
    assert all(v["hit"] for v in s["verdicts"]) and len(s["verdicts"]) == 2
    assert abs(s["g_measured"] - scene.g) / scene.g < 0.03
    laws = {e["id"]: e for e in log.of_type("law")}
    assert laws["law_x"]["terms"] == []                               # sideways: no force
    assert [t["term"] for t in laws["law_y"]["terms"]] == ["1"]       # up and down: gravity only
    assert abs(laws["law_y"]["terms"][0]["value"] + scene.g) / scene.g < 0.03


def test_mystery_cans(tmp_path):
    scene = RaceScene()
    s, log = run("cans", render_race(scene, tmp_path / "cans.mp4"))
    assert abs(s["k"]["red"] - scene.k_a) < 0.05 and abs(s["k"]["blue"] - scene.k_b) < 0.05
    assert s["winner_pred"] == "red"
    assert abs(s["finish_pred"]["red"] - scene.finish_time(scene.k_a)) < 0.01
    assert all(v["hit"] for v in s["verdicts"]) and len(s["verdicts"]) == 3
    for e in log.of_type("law"):                                       # a constant push down the slope
        assert [t["term"] for t in e["terms"]] == ["1"]


def test_weigh_it_by_watching(tmp_path):
    scene = SpringScene()
    s, log = run("weigh", render_springs(scene, tmp_path / "springs.mp4"), true_mass_g=scene.m_mystery * 1000)
    assert abs(s["mass_pred_g"] - 250) < 5
    assert abs(s["mass_static_g"] - 250) < 10
    w2 = {e["object"]: -next(t["value"] for t in e["terms"] if t["term"] == "q") for e in log.of_type("law")}
    assert abs(w2["ref"] / w2["mystery"] - 2.5) < 0.05                 # same spring: ω² ∝ 1/m, 250 g vs 100 g
    assert all(v["hit"] for v in s["verdicts"]) and len(s["verdicts"]) == 2
