"""Trick clips where answering from memory is wrong. Offline (OpenCV motion tracker, template explanations).

Real-footage variants come from the classroom clip (see benchmark/clips, built by the benchmark notes in
benchmark/README.md); they are skipped if absent. Hidden-variable clips are rendered here.
"""
from pathlib import Path

import pytest

from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger
from physlens.perception import MotionBlobTracker
from physlens.pipeline import run_pipeline
from physlens.synth import DropScene, render

CLIPS = Path(__file__).resolve().parents[2] / "benchmark/clips"
RULER = {"px_per_m": 161.143}


def run(path: Path, **kw):
    return run_pipeline(path, CARDS["drop"], EventLog(path.stem), Ledger(), tracker=MotionBlobTracker(color="orange"), **kw)


def real(name: str) -> Path:
    path = CLIPS / f"{name}.mp4"
    if not path.exists():
        pytest.skip(f"{path.name} not built")
    return path


def test_slow_motion_shows_a_quarter_of_gravity():
    s = run(real("real-slowmo-2x"), **RULER)
    assert abs(s["g"] - 9.81 / 4) < 2 * s["g_err"] and abs(s["slowmo_factor"] - 2) < 0.1


def test_sped_up_shows_four_times_gravity():
    s = run(real("real-speedup-2x"), **RULER)
    assert abs(s["g"] - 9.81 * 4) < 2 * s["g_err"] and abs(s["slowmo_factor"] - 0.5) < 0.05


def test_upside_down_is_noticed_and_flipped():
    s = run(real("real-upside-down"), **RULER)
    assert s["upside_down"]
    assert abs(s["g"] - 9.81) < 2 * s["g_err"]
    assert all(v["hit"] for v in s["verdicts"]) and s["verdicts"]


def test_reversed_clip_is_flagged_and_not_predicted():
    s = run(real("real-reversed"), **RULER)
    assert s["reversed"] and s["restitution"] > 1.03
    assert s["verdicts"] == [] and s["peak_pred_cm"] is None
    assert "backwards" in s["explanation"]


@pytest.mark.xfail(reason="known limit: at 12.5 fps motion blur biases g ~7% high with a too-small error bar",
                   strict=True)
def test_low_frame_rate_still_measures_gravity():
    s = run(real("real-12fps"), **RULER)
    assert abs(s["g"] - 9.81) < 2 * s["g_err"]


@pytest.fixture(scope="module")
def hidden(tmp_path_factory):
    folder = tmp_path_factory.mktemp("hidden")
    return {
        "bouncy": render(DropScene(restitution=0.85, fps=60, duration=5.0), folder / "bouncy.mp4"),
        "dead": render(DropScene(restitution=0.55, fps=60, duration=3.0), folder / "dead.mp4"),
        "mars": render(DropScene(g=3.71, restitution=0.75, fps=60, duration=6.0), folder / "mars.mp4"),
    }


def test_identical_looking_balls_get_their_own_bounciness(hidden):
    bouncy, dead = run(hidden["bouncy"]), run(hidden["dead"])
    assert abs(bouncy["contacts"][-1][1] - 0.85) < 0.02
    assert abs(dead["contacts"][-1][1] - 0.55) < 0.02
    assert all(v["hit"] for v in bouncy["verdicts"] + dead["verdicts"])


def test_mars_gravity_is_measured_not_remembered(hidden):
    s = run(hidden["mars"])
    assert abs(s["g"] - 3.71) < 2 * s["g_err"]
    assert "either" in s["explanation"]   # can't tell slow motion from weaker gravity in one clip
