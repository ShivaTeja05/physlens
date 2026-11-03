"""End to end on rendered clips: tracking → physics → ledger → explanation."""
from pathlib import Path

import pytest

from physlens.agent import check_numbers
from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger, prediction_hash
from physlens.pipeline import run_pipeline
from physlens.synth import SAMPLES, render

CASES = {"drop-real": (9.81, 1.0), "drop-slowmo-2x": (9.81 / 4, 2.0), "drop-low-gravity": (1.81, 2.33)}


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    folder = tmp_path_factory.mktemp("clips")
    return {name: render(SAMPLES[name], folder / f"{name}.mp4") for name in CASES}


@pytest.mark.parametrize("name", list(CASES))
def test_run(clips, name):
    expected_g, expected_factor = CASES[name]
    log = EventLog(name)
    ledger = Ledger()
    summary = run_pipeline(clips[name], CARDS["drop"], log, ledger)

    assert abs(summary["g"] - expected_g) < 2 * summary["g_err"]
    assert abs(summary["slowmo_factor"] - expected_factor) < 0.05
    assert all(v["hit"] for v in summary["verdicts"]) and len(summary["verdicts"]) == 2
    assert summary["bounce_law"] == "constant_restitution"   # the test clips really are constant

    # Every prediction was locked before the outcome it predicts, from earlier samples only.
    for p in log.of_type("prediction"):
        assert p["based_on_until_t"] < p["t"] <= p["target_t"]
        assert p["hash"] == prediction_hash(p)
    verdict_times = {v["id"]: v["t"] for v in log.of_type("verdict")}
    assert all(verdict_times[p["id"]] > p["t"] for p in log.of_type("prediction"))

    # Timeline is in video order, and the explanation passed the evidence check.
    times = [e["t"] for e in log.events if e["type"] != "run_state"]
    assert times == sorted(times)
    explain = log.of_type("explain")[0]
    assert explain["verified"]


def test_verifier_rejects_numbers_not_in_evidence():
    evidence = {"g": 2.466, "memory_g": 9.81}
    assert check_numbers("Gravity here is 2.47 m/s², not 9.81 (or 9.8 rounded).", evidence)[0]
    ok, bad = check_numbers("Gravity here is 2.47 m/s², close to Mars at 3.7.", evidence)
    assert not ok and bad == ["3.7"]
