"""Real footage regression test: a classroom ball drop with a teacher's hand-tracked ground truth.

Skipped when the clip isn't present (see real_clips/compadre-bouncing-ball/SOURCE.md to fetch it).
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger
from physlens.perception import MotionBlobTracker
from physlens.pipeline import run_pipeline

CLIP = Path(__file__).resolve().parents[2] / "real_clips/compadre-bouncing-ball/extracted"
VIDEO = CLIP / "videos/BouncingBall_ballbouncelookang01_x264.mp4"
TRK = CLIP / "BouncingBall_ballbouncelookang01_x264.trk"
PX_PER_M = 161.143  # 1.000 m tape in the teacher's Tracker file

pytestmark = pytest.mark.skipif(not VIDEO.exists(), reason="real clip not downloaded")


def teacher_points() -> np.ndarray:
    root = ET.parse(TRK).getroot()
    num = lambda el, n: float(el.find(f"property[@name='{n}']").text)
    pm = next(o for o in root.iter("object") if o.get("class", "").endswith("PointMass"))
    return np.array([(num(f, "x"), num(f, "y")) for f in pm.iter("object")
                     if f.get("class", "").endswith("PointMass$FrameData")])


def test_tracker_matches_hand_clicks():
    result = MotionBlobTracker(color="orange").track(VIDEO, "ball")
    teacher = teacher_points()
    errors = [np.hypot(d.x_px - teacher[i, 0], d.y_px - teacher[i, 1]) for i, d in enumerate(result.frames) if d]
    assert len(errors) >= 50                 # free flight is covered
    assert np.median(errors) < 3.5           # pixels
    assert max(errors) < 10                  # never on the wrong object


def test_measures_gravity_and_calls_the_bounce():
    summary = run_pipeline(VIDEO, CARDS["drop"], EventLog("real"), Ledger(), px_per_m=PX_PER_M)
    assert abs(summary["g"] - 9.81) < 2 * summary["g_err"]
    assert all(v["hit"] for v in summary["verdicts"]) and len(summary["verdicts"]) == 2
    # This ball loses more energy in harder impacts (the teacher's clicks show it too), so the
    # constant-bounciness law must be rejected.
    assert summary["bounce_law"] == "speed_dependent_restitution"
    (v1, e1), (v2, e2) = summary["contacts"][:2]
    assert v1 > v2 and e1 < e2
