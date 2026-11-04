"""Crash reconstruction on real crash-test footage with known answers (NHTSA test 9070, camera 15; offline)."""
import json
from pathlib import Path

import pytest

from physlens import recorder
from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger
from physlens.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[2]
CLIP = ROOT / "samples" / "crash-frontal.mp4"
pytestmark = pytest.mark.skipif(not CLIP.exists(), reason="NHTSA crash sample not present")


@pytest.fixture(scope="module")
def crash(tmp_path_factory):
    params = json.loads((ROOT / "samples/settings.json").read_text())["crash-frontal"]["params"]
    log = EventLog("crash-test", tmp_path_factory.mktemp("log") / "events.jsonl")   # written to disk, as in the app
    summary = run_pipeline(CLIP, CARDS["crash"], log, Ledger(), params=params)
    return params, summary, log, tmp_path_factory.mktemp("crash")


def test_speed_matches_the_lab(crash):
    params, s, log, _ = crash
    official = params["official_kmh"]
    assert abs(s["speed_kmh"] - official) / official < 0.04            # within 4% of the lab's speed trap
    lo, hi = s["speed_ci_kmh"]
    assert lo <= official <= hi                                        # and the band contains it
    assert s["verdicts"] and s["verdicts"][0]["hit"]
    pred = log.of_type("prediction")[0]
    assert pred["based_on_until_t"] < pred["target_t"]                  # locked from the approach, before impact


def test_physics_is_consistent_with_the_cars_instruments(crash):
    params, s, log, _ = crash
    assert s["crush_m"] is not None and s["crush_m"] <= params["sensor_crush_m"] + 0.05   # a lower bound at 30 fps
    assert [e["equation"] for e in log.of_type("law")] == ["a = 0"]     # the released car coasts: no force
    kinds = [e["kind"] for e in log.of_type("moment")]
    assert kinds[:2] == ["approach", "impact"] and "rest" in kinds


def test_reconstruction_and_record(crash):
    params, s, log, tmp = crash
    rec = log.of_type("reconstruction")[0]
    assert abs(rec["length_m"] - params["car_length_m"]) < 1e-6 and rec["impact_t"] > 0
    before = [f for f in rec["frames"] if f[0] < rec["impact_t"] and f[1] is not None]
    assert before and all(f[1] > -0.05 for f in before)               # the car is on its side of the barrier
    db = recorder.connect(tmp / "rec.db")
    recorder.record_run("crash-test", log.events, tmp, {}, db=db, experiment="crash")
    out = recorder.export("crash-test", db)
    impact = next(e for e in out["events"] if e["kind"] == "impact")
    assert abs(impact["speed_in"] * 3.6 - params["official_kmh"]) / params["official_kmh"] < 0.04
