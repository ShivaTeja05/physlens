"""The engine's guards: only local pages may drive it, uploads are checked, bad numbers are refused.

Offline and free: nothing here reaches a paid model (every request is refused before a job starts)."""
import pytest
from fastapi.testclient import TestClient

from physlens import api

client = TestClient(api.app)


def test_foreign_origin_is_refused_before_anything_runs():
    r = client.post("/api/scan", data={"sample": "anything"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403
    assert client.get("/api/records", headers={"Origin": "https://evil.example"}).status_code == 403


@pytest.mark.parametrize("origin", ["http://localhost:5173", "http://127.0.0.1:5173", "http://[::1]:8000"])
def test_local_pages_are_allowed(origin):
    r = client.get("/api/health", headers={"Origin": origin})
    assert r.status_code == 200
    assert r.headers["access-control-allow-origin"] == origin


def test_dns_rebinding_host_is_refused():
    assert client.get("/api/health", headers={"Host": "evil.example"}).status_code == 400


def test_websockets_refuse_foreign_origins():
    from starlette.websockets import WebSocketDisconnect
    for path in ("/api/voice", "/api/runs/anything/stream"):
        with pytest.raises(WebSocketDisconnect) as closed:
            with client.websocket_connect(path, headers={"Origin": "https://evil.example"}) as ws:
                ws.receive_json()
        assert closed.value.code == 4403


def test_file_that_is_not_a_video_gets_a_clear_400():
    r = client.post("/api/runs", data={"experiment_id": "drop"}, files={"file": ("holiday.mp4", b"not a video", "video/mp4")})
    assert r.status_code == 400
    assert "Couldn't read that file as a video" in r.json()["detail"]
    assert not list(api.MEDIA.glob("holiday-*.mp4")), "a failed upload must not leave a file behind"


def test_oversized_upload_is_refused(monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 10)
    r = client.post("/api/scan", files={"file": ("big.jpg", b"x" * 100, "image/jpeg")})
    assert r.status_code == 413


def test_image_that_is_not_an_image_gets_a_clear_400():
    r = client.post("/api/scan", files={"file": ("photo.jpg", b"\x00" * 3000, "image/jpeg")})
    assert r.status_code == 400
    assert "JPG or PNG" in r.json()["detail"]


@pytest.mark.parametrize("form", [{"diameter_cm": "-6.5"}, {"px_per_m": "0"}, {"params": '{"table_height_m": -1}'},
                                  {"params": "not json"}, {"experiment_id": "nope"}])
def test_bad_numbers_and_experiments_are_refused_before_a_run_exists(form):
    before = set(api.RUNS)
    r = client.post("/api/runs", data={"sample": "drop-real", **form})
    assert r.status_code in (400, 404)
    assert set(api.RUNS) == before


def test_busy_engine_says_so(monkeypatch):
    import threading
    monkeypatch.setattr(api, "JOBS", threading.BoundedSemaphore(1))
    api.JOBS.acquire()
    r = client.post("/api/scan", files={"file": ("p.png", _png(), "image/png")})
    assert r.status_code == 429
    assert "already running" in r.json()["detail"]


def _png() -> bytes:
    import cv2
    import numpy as np
    return cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))[1].tobytes()
