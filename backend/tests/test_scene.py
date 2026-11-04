"""Atomic scenes: a record becomes world + objects + atoms that say what was measured and what assumed."""
from pathlib import Path

from physlens import recorder, scene
from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger
from physlens.perception.motion_tracker import MotionBlobTracker
from physlens.pipeline import run_pipeline
from physlens.synth import DropScene, render


def test_drop_scene(tmp_path):
    clip = render(DropScene(), tmp_path / "drop.mp4")
    log = EventLog("drop-scene")
    run_pipeline(clip, CARDS["drop"], log, Ledger(), tracker=MotionBlobTracker())
    db = recorder.connect(tmp_path / "rec.db")
    recorder.record_run("drop-scene", log.events, tmp_path, {}, db=db, experiment="drop")
    sc = scene.from_record(recorder.export("drop-scene", db), log.events)
    assert sc["format"] == "physlens-scene/1"
    assert abs(sc["world"]["gravity"][1] + 9.81) < 0.3 and sc["world"]["gravity_source"].startswith("measured")
    ball = sc["objects"][0]
    assert ball["shape"] == "sphere" and ball["restitution"]["source"].startswith("measured")
    assert abs(ball["restitution"]["value"] - 0.78) < 0.03
    atoms = sc["atoms"][ball["id"]]
    assert len(atoms) > 50 and all("z" in a["assumed"] and "y" in a["measured"] for a in atoms)
    assert min(a["p"][1] for a in atoms) > -0.02          # the ground is at y = 0, nothing below it


def test_mujoco_fit_reproduces_the_drop(tmp_path):
    """MuJoCo, fitted to the atoms of a rendered drop (bounciness 0.78), follows it to within a few cm."""
    import pytest
    pytest.importorskip("mujoco")
    from physlens import sim_mujoco
    clip = render(DropScene(), tmp_path / "drop.mp4")
    log = EventLog("drop-mj")
    run_pipeline(clip, CARDS["drop"], log, Ledger(), tracker=MotionBlobTracker())
    db = recorder.connect(tmp_path / "rec.db")
    recorder.record_run("drop-mj", log.events, tmp_path, {}, db=db, experiment="drop")
    out = sim_mujoco.fit(scene.from_record(recorder.export("drop-mj", db), log.events))
    assert out["rms_m"] < 0.03 and out["rms_m"] <= out["rms_unfitted_m"]
    assert 0.72 < out["params"]["e0"] < 0.82
    import mujoco
    mujoco.MjModel.from_xml_string(out["mjcf"])              # the exported model opens in MuJoCo
