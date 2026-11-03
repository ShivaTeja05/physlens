"""The recorder keeps an event's physics, and a replay from the record reproduces it (offline, free)."""
from pathlib import Path

import pytest

from physlens import recorder
from physlens.events import EventLog
from physlens.experiments import CARDS
from physlens.ledger import Ledger
from physlens.perception.motion_tracker import MotionBlobTracker
from physlens.pipeline import run_pipeline
from physlens.synth import DropScene, render

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def drop_record(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("rec")
    scene = DropScene()
    clip = render(scene, tmp / "drop.mp4")
    log = EventLog("drop-test")
    run_pipeline(clip, CARDS["drop"], log, Ledger(), tracker=MotionBlobTracker())
    db = recorder.connect(tmp / "test.db")
    recorder.record_run("drop-test", log.events, tmp, {"/media": tmp}, db=db, experiment="drop")
    return scene, recorder.export("drop-test", db), db


def test_record_has_the_physics(drop_record):
    scene, rec, _ = drop_record
    contacts = [e for e in rec["events"] if e["kind"] == "contact" and e["ratio"]]
    assert len(contacts) >= 4
    for e in contacts:                       # every bounce keeps the true bounciness
        assert abs(e["ratio"] - scene.restitution) < 0.03
    assert rec["laws"] and rec["laws"][0]["equation"].startswith("a = −9.")
    assert rec["predictions"] and all(p["sha256"] for p in rec["predictions"])
    kinds = [e["kind"] for e in rec["events"]]
    assert kinds[0] == "start" and kinds[-1] == "end" and "peak" in kinds


def test_list_and_rewrite(drop_record):
    _, rec, db = drop_record
    rows = recorder.list_captures(db)
    assert rows[0]["id"] == "drop-test" and rows[0]["contacts"] >= 4
    n_states = len(rec["states"]["ball"])
    recorder.record_run("drop-test", [], Path("."), {}, db=db)        # re-recording replaces, never duplicates
    assert db.execute("SELECT COUNT(*) FROM state WHERE capture = 'drop-test'").fetchone()[0] == 0 < n_states
