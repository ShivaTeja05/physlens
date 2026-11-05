"""Orchestrator API: the only thing the browser talks to.

  uvicorn physlens.api:app --reload --port 8000
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles

from .config import load_env
from .agent import ScientistAgent
from .events import EventLog
from .experiments import CARDS
from .ledger import Ledger
from .perception import default_tracker
from .pipeline import run_pipeline
from .synth import FFMPEG_INPUT, ffmpeg_exe

ROOT = Path(__file__).resolve().parents[2]
load_env()
SAMPLES = ROOT / "samples"
MEDIA = ROOT / "media"
REPLAYS = ROOT / "replays"
for folder in (SAMPLES, MEDIA, REPLAYS):
    folder.mkdir(exist_ok=True)


@dataclass
class Run:
    id: str
    experiment: str
    source: str
    log: EventLog
    status: str = "running"
    summary: Optional[dict] = None
    error: Optional[str] = None
    subscribers: list[tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)


app = FastAPI(title="PhysLens orchestrator")

# Only pages served from this machine may drive the engine. Without this, any website open in the same browser could
# POST scans (spending the Meta budget) or read records and media: CORS alone doesn't stop a simple POST from
# running, and doesn't apply to WebSockets at all. The host check stops DNS rebinding (evil.example → 127.0.0.1).
# PHYSLENS_ALLOWED_ORIGINS adds more origins (comma separated), for a deployed web app.
LOCAL_ORIGIN = re.compile(r"^https?://(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$")
EXTRA_ORIGINS = {o.strip() for o in os.environ.get("PHYSLENS_ALLOWED_ORIGINS", "").split(",") if o.strip()}


def origin_allowed(origin: Optional[str]) -> bool:
    """No Origin header (curl, the tests, a top-level page load) is allowed; a browser page must be local."""
    return origin is None or bool(LOCAL_ORIGIN.match(origin)) or origin in EXTRA_ORIGINS


@app.middleware("http")
async def refuse_foreign_origins(request, call_next):
    if not origin_allowed(request.headers.get("origin")):
        return JSONResponse({"detail": "requests from other websites are not allowed"}, status_code=403)
    return await call_next(request)


app.add_middleware(CORSMiddleware, allow_origin_regex=LOCAL_ORIGIN.pattern, allow_origins=sorted(EXTRA_ORIGINS),
                   allow_methods=["*"], allow_headers=["*"])
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "[::1]", "testserver",
                                                         *(h.strip() for h in os.environ.get("PHYSLENS_ALLOWED_HOSTS", "").split(",") if h.strip())])
app.mount("/samples", StaticFiles(directory=SAMPLES), name="samples")
app.mount("/media", StaticFiles(directory=MEDIA), name="media")
RUNS: dict[str, Run] = {}
MAX_VIDEO_BYTES = 300 * 1024 * 1024   # a few minutes of phone video; anything bigger is not a lab clip
MAX_IMAGE_BYTES = 25 * 1024 * 1024
MAX_AUDIO_BYTES = 10 * 1024 * 1024
# Runs and scans call paid models from worker threads; the budget check before each call can't see calls still in
# flight, so this bounds how far parallel jobs could pass the cap (and how much one browser can queue up).
JOBS = threading.BoundedSemaphore(2)
BUSY = "Two analyses are already running. Wait for one to finish, then try again."


async def _read_upload(file: UploadFile, limit: int, what: str) -> bytes:
    """The upload's bytes, refusing (413) past `limit` without reading the rest into memory."""
    chunks, size = [], 0
    while chunk := await file.read(1 << 20):
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"That {what} is over {limit // (1024 * 1024)} MB. Trim it or send a smaller one.")
        chunks.append(chunk)
    return b"".join(chunks)


def _broadcast(run: Run, event: dict) -> None:
    with run.lock:
        subs = list(run.subscribers)
    for loop, queue in subs:
        loop.call_soon_threadsafe(queue.put_nowait, event)


def _transcode(src: Path, dst: Path) -> None:
    """Uploads can be any phone format; the browser needs H.264 in MP4. Baseline profile with no B-frames and a
    90 kHz time base: with ffmpeg's defaults (High profile, B-frames, 12800 time base) SAM 3.1 processed and billed a
    70-frame clip as 1,680 frames, 24× too many."""
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", *FFMPEG_INPUT, "-i", str(src), "-an", "-c:v", "libx264",
                    "-profile:v", "baseline", "-bf", "0", "-pix_fmt", "yuv420p", "-crf", "18",
                    "-video_track_timescale", "90000", "-movflags", "+faststart", str(dst)], check=True)


def _sample_settings() -> dict:
    path = SAMPLES / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _process(run: Run, video: Path, url: str, px_per_m: Optional[float], diameter_m: Optional[float],
             params: dict) -> None:
    """Runs on a worker thread holding one JOBS slot, which it releases at the end."""
    try:
        card = CARDS.get(run.experiment)
        if run.experiment == "auto":
            card = _detect_experiment(run, video)
        ledger = Ledger(REPLAYS / f"{run.id}.ledger.jsonl")
        run.summary = run_pipeline(video, card, run.log, ledger, video_url=url,
                                   px_per_m=px_per_m, diameter_m=diameter_m, params=params)
        run.status = "done"
    except Exception as exc:
        run.status, run.error = "error", str(exc)
    finally:
        JOBS.release()
        _save_record(run)
        _broadcast(run, {"type": "_end", "status": run.status, "error": run.error})


def _save_record(run: Run) -> None:
    """Every run goes into the recorder (records/physlens.db), so it can be found, replayed and compared later."""
    from . import recorder
    try:
        recorder.record_run(run.id, run.log.events, REPLAYS, {"/samples": SAMPLES, "/media": MEDIA},
                            experiment=run.experiment, status=run.status)
    except Exception as exc:   # a record that can't be written must never break the run itself
        print(f"recorder: couldn't record {run.id}: {exc}")


def _detect_experiment(run: Run, video: Path):
    """Muse Spark looks at the middle of the clip and picks the experiment (and, for a drop, what to track)."""
    import dataclasses

    import cv2

    from .scan import SURFACES, _slug, look
    run.log.emit("run_state", 0, state="created", message="Muse Spark is working out which experiment this is…")
    import numpy as np
    # Three moments of the clip side by side, so Muse Spark can see what moves.
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shots = []
    for f in (0.25, 0.5, 0.75):
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(n * f), 0))
        ok, frame = cap.read()
        if ok:
            shots.append(cv2.resize(frame, (480, round(frame.shape[0] * 480 / frame.shape[1]))))
    cap.release()
    if not shots:
        raise RuntimeError("couldn't read the clip")
    sheet = np.hstack([np.pad(s, ((0, 0), (0, 8), (0, 0))) for s in shots])
    seen = look(sheet, frames=len(shots))
    exp = seen.get("video_experiment", "none")
    if exp not in CARDS:
        raise RuntimeError(f"This looks like {seen.get('title') or 'something else'}, not one of the lab experiments. "
                           "Try Scan mode for a still, or pick the experiment yourself.")
    card = CARDS[exp]
    tracked = ""
    if card.kind == "drop":
        moving = [seen.get("moving_object") or ""] + [p["name"] for p in seen["parts"]
                                                       if not set(_slug(p["name"]).split("_")) & SURFACES]
        moving = [m.strip() for m in moving if m and m.strip()]
        if moving:
            tracked = moving[0]
            card = dataclasses.replace(card, prompt=tracked, color=None)
    run.experiment = exp
    run.log.emit("run_state", 0, state="created", experiment=exp,
                 message=f"Detected: {seen.get('title', 'experiment')} → {card.title}"
                         + (f", tracking the {tracked}" if tracked else ""))
    return card


@app.get("/api/health")
def health():
    return {"tracker": default_tracker().name, "agent": ScientistAgent().name}


@app.get("/api/budget")
def spend():
    from . import budget
    return budget.summary()


@app.get("/api/experiments")
def experiments():
    return [{"id": c.id, "title": c.title, "hook": c.hook, "kind": c.kind, "params": c.params,
             "objects": [o.prompt for o in c.objects] or [c.prompt]} for c in CARDS.values()]


@app.get("/api/samples")
def samples():
    settings = _sample_settings()
    return [{"name": p.stem, "url": f"/samples/{p.name}", **settings.get(p.stem, {})}
            for p in sorted(SAMPLES.glob("*.mp4"))]


SCAN_SAMPLES = SAMPLES / "scan"
SCAN_CACHE = REPLAYS / "scans"
SCAN_SAMPLES.mkdir(exist_ok=True)


@app.get("/api/scan-samples")
def scan_samples():
    meta = json.loads((SCAN_SAMPLES / "settings.json").read_text()) if (SCAN_SAMPLES / "settings.json").exists() else {}
    return [{"name": p.stem, "url": f"/samples/scan/{p.name}", **meta.get(p.stem, {})}
            for p in sorted(SCAN_SAMPLES.glob("*.jpg"))]


SCANS: dict[str, dict] = {}


@app.get("/api/scan/{job_id}")
def scan_status(job_id: str):
    """Progress of a scan: stage (looking, outlining, explaining, done, error) and what's known so far."""
    if job_id not in SCANS:
        raise HTTPException(404, "unknown scan")
    return SCANS[job_id]


@app.post("/api/scan")
async def create_scan(file: Optional[UploadFile] = File(None), sample: Optional[str] = Form(None),
                      reference: Optional[str] = Form(None)):
    """One still image (a webcam frame, an upload, or a sample) → what it is, its parts, measurements, how it works.
    `reference`: optional JSON {"part": id or name, "size_m": 2.0, "axis": "height"|"width"} to get metres."""
    import cv2
    import numpy as np
    from .scan import scan
    if sample:
        path = SCAN_SAMPLES / f"{Path(sample).name}.jpg"
        if not path.exists():
            raise HTTPException(404, f"unknown scan sample {sample}")
        img = cv2.imread(str(path))
    elif file is not None:
        img = cv2.imdecode(np.frombuffer(await _read_upload(file, MAX_IMAGE_BYTES, "image"), np.uint8), cv2.IMREAD_COLOR)
    else:
        raise HTTPException(400, "send an image file or a sample name")
    if img is None:
        raise HTTPException(400, "Couldn't read that file as an image. Send a JPG or PNG photo.")
    if img.shape[1] > 1280:   # plenty for SAM and Muse Spark, and keeps uploads quick
        img = cv2.resize(img, (1280, round(img.shape[0] * 1280 / img.shape[1])), interpolation=cv2.INTER_AREA)
    ref = None
    if reference:
        try:
            ref = json.loads(reference)
        except ValueError:
            raise HTTPException(400, "reference must be JSON")
    if not JOBS.acquire(blocking=False):
        raise HTTPException(429, BUSY)
    job_id = uuid.uuid4().hex[:10]
    if sample:
        url = f"/samples/scan/{Path(sample).name}.jpg"
    else:
        url = f"/media/scan-{job_id}.jpg"
        cv2.imwrite(str(MEDIA / f"scan-{job_id}.jpg"), img)
    SCANS[job_id] = {"stage": "queued", "result": {"image": {"width": img.shape[1], "height": img.shape[0], "url": url}}}

    def progress(stage, partial):
        SCANS[job_id] = {"stage": stage, "result": {**partial, "image": {**partial.get("image", {}), "url": url}}}

    def work():
        try:
            result = scan(img, ref, SCAN_CACHE, progress)
            progress("done", result)
        except Exception as exc:
            SCANS[job_id] = {**SCANS[job_id], "stage": "error", "error": str(exc)[:300]}
        finally:
            JOBS.release()

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/ask")
async def ask_question(context: str = Form(...), question: Optional[str] = Form(None),
                       audio: Optional[UploadFile] = File(None), reset: bool = Form(False)):
    """A question about one scan ("scan:<id>"), lab run ("run:<id>") or 3D world ("world:<name>"), typed or spoken. Answers come from what that
    scan or run measured, and the conversation is remembered per context so follow-ups stay consistent."""
    from . import ask as asking
    kind, _, cid = context.partition(":")
    if kind == "scan":
        f = SCAN_CACHE / f"scan-{Path(cid).name}.json"
        if not f.exists():
            raise HTTPException(404, "unknown scan")
        facts = asking.scan_facts(json.loads(f.read_text()))
    elif kind == "run":
        run = RUNS.get(cid)
        replay = REPLAYS / f"{Path(cid).name}.jsonl"
        if run is not None:
            events, exp = run.log.events, run.experiment
        elif replay.exists():   # the engine restarted since: the run's saved events
            events = [json.loads(l) for l in replay.read_text().splitlines() if l.strip()]
            exp = next((e["experiment"] for e in events if e.get("experiment")), "experiment")
        else:
            raise HTTPException(404, "unknown run")
        facts = asking.run_facts(events, CARDS[exp].title if exp in CARDS else exp)
        try:
            from . import recorder
            facts |= asking.record_facts(recorder.export(cid))
        except KeyError:
            pass
    elif kind == "world":
        f = MEDIA / "pointworld" / f"{Path(cid).name}.json"
        if not f.exists():
            raise HTTPException(404, "unknown world")
        facts = asking.world_facts(json.loads(f.read_text()))
    else:
        raise HTTPException(400, "context must be scan:<id>, run:<id> or world:<name>")
    if reset:
        asking.reset(context)
    heard = None
    if audio is not None:
        try:
            clip = await _read_upload(audio, MAX_AUDIO_BYTES, "recording")
            heard = await asyncio.to_thread(asking.transcribe, clip, audio.filename or "question.webm")
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(502, f"couldn't transcribe: {str(exc)[:200]}")
        question = heard
    if not question or not question.strip():
        raise HTTPException(400, "ask a question")
    try:
        out = await asyncio.to_thread(asking.ask, context, facts, question.strip())
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))
    return {"question": question.strip(), "heard": heard, **out}


@app.get("/api/worlds/{name}/specs")
def world_specs_cached(name: str):
    """The specs agent's last lookup for this world, if one was made (free: no model calls)."""
    f = MEDIA / "pointworld" / f"{Path(name).name}.specs.json"
    if not f.exists():
        raise HTTPException(404, "no lookup yet")
    return json.loads(f.read_text())


@app.post("/api/worlds/{name}/specs")
async def world_specs(name: str, refresh: bool = Form(False)):
    """Published specs of the world's object, looked up by the Muse Spark specs agent (NHTSA, VIN decoder, Wikipedia),
    each number checked against the tool results, cross-checked between sources and against the video's measurement.
    Costs a few Muse Spark calls; cached per world."""
    from . import specs
    pw = MEDIA / "pointworld"
    if not (pw / f"{Path(name).name}.json").exists():
        raise HTTPException(404, "unknown world")
    try:
        return await asyncio.to_thread(specs.for_world, pw, Path(name).name, refresh)
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


@app.websocket("/api/voice")
async def voice(ws: WebSocket):
    """Push-to-talk: the app streams 16 kHz mono PCM16 while the button is held, then sends {"type":"stop"}.
    Words come back as they are recognised ({"type":"partial"}), then {"type":"final"}."""
    from .voice import VoiceSession
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=4403)
        return
    await ws.accept()

    async def partial(text: str) -> None:
        await ws.send_json({"type": "partial", "text": text})

    session = VoiceSession(partial)
    try:
        await session.start()
        await ws.send_json({"type": "ready"})
        while True:
            msg = await ws.receive()
            if msg.get("bytes"):
                await session.feed(msg["bytes"])
            elif msg.get("text") and json.loads(msg["text"]).get("type") == "stop":
                await ws.send_json({"type": "final", "text": await session.finish()})
                break
            elif msg.get("type") == "websocket.disconnect":
                break
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        try:
            await ws.send_json({"type": "error", "message": str(exc)[:200]})
        except Exception:
            pass
    finally:
        try:
            await ws.close()
        except Exception:
            pass


@app.get("/api/records")
def records(limit: int = 50):
    """Every recorded run, newest first, with counts of objects, contacts and prediction hits."""
    from . import recorder
    return recorder.list_captures(limit=limit)


@app.get("/api/records/{capture_id}")
def record(capture_id: str):
    """One full record: capture details, objects, key moments, laws, predictions and every state."""
    from . import recorder
    try:
        return recorder.export(capture_id)
    except KeyError:
        raise HTTPException(404, "unknown record")


@app.post("/api/records/{capture_id}/whatif")
def what_if(capture_id: str, object: str = Form(...), gravity: Optional[float] = Form(None),
            restitution: Optional[float] = Form(None), speed_factor: float = Form(1.0), drag: Optional[float] = Form(None),
            horizon: Optional[float] = Form(None)):
    """Replay a record from its stored state with the law found for it, and the same with one thing changed."""
    from . import recorder, replay
    try:
        rec = recorder.export(capture_id)
    except KeyError:
        raise HTTPException(404, "unknown record")
    try:
        out = replay.what_if(rec, object, gravity=gravity, restitution=restitution, speed_factor=speed_factor,
                             drag=drag, horizon=horizon)
    except replay.Unsupported as exc:
        raise HTTPException(422, str(exc))
    out["px_from_m"] = rec["px_from_m"]
    return out


@app.get("/api/records/{capture_id}/scene")
def record_scene(capture_id: str):
    """The record as an atomic scene (world, objects, atoms over time, moments) for the 3D world."""
    from . import scene
    try:
        return scene.build(capture_id, REPLAYS)
    except KeyError:
        raise HTTPException(404, "unknown record")


SIM_DIR = ROOT / "records" / "sim"


@app.post("/api/records/{capture_id}/mujoco")
async def record_mujoco(capture_id: str):
    """Fit a MuJoCo simulation to the record's atoms (system identification); saves the fitted MJCF model."""
    from . import scene, sim_mujoco
    try:
        sc = scene.build(capture_id, REPLAYS)
    except KeyError:
        raise HTTPException(404, "unknown record")
    if not sc["objects"]:
        raise HTTPException(422, "this record has no objects to simulate")
    out = await asyncio.to_thread(sim_mujoco.fit, sc)
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    name = Path(capture_id).name
    (SIM_DIR / f"{name}.xml").write_text(out["mjcf"])
    (SIM_DIR / f"{name}.json").write_text(json.dumps({k: v for k, v in out.items() if k != "track"}, default=float))
    return json.loads(json.dumps(out, default=float))


@app.get("/api/records/{capture_id}/mujoco.xml")
def record_mujoco_xml(capture_id: str):
    """The fitted MuJoCo model (MJCF), to open in any MuJoCo viewer."""
    from fastapi.responses import PlainTextResponse
    f = SIM_DIR / f"{Path(capture_id).name}.xml"
    if not f.exists():
        raise HTTPException(404, "fit the record first")
    return PlainTextResponse(f.read_text(), media_type="application/xml",
                             headers={"Content-Disposition": f'attachment; filename="{f.name}"'})


@app.get("/api/records/{capture_id}/events")
def record_events(capture_id: str):
    """A past run's full event stream, so the app can open it again."""
    f = REPLAYS / f"{Path(capture_id).name}.jsonl"
    if not f.exists():
        raise HTTPException(404, "unknown record")
    return [json.loads(l) for l in f.read_text().splitlines() if l.strip()]


@app.get("/api/runs")
def runs():
    return [{"id": r.id, "experiment": r.experiment, "source": r.source, "status": r.status} for r in RUNS.values()]


@app.post("/api/runs")
async def create_run(experiment_id: str = Form("drop"), sample: Optional[str] = Form(None),
                     file: Optional[UploadFile] = File(None), px_per_m: Optional[float] = Form(None),
                     diameter_cm: Optional[float] = Form(None), params: Optional[str] = Form(None)):
    """Scale: `px_per_m` from a ruler in the shot, or `diameter_cm` of the ball. Samples carry their own.
    `params`: JSON of experiment settings (e.g. a taped table height), merged over the card's defaults."""
    run_params: dict = {}
    for name, value in (("px_per_m", px_per_m), ("diameter_cm", diameter_cm)):
        if value is not None and not value > 0:
            raise HTTPException(400, f"{name} must be a positive number")
    if experiment_id not in CARDS and experiment_id != "auto":
        raise HTTPException(404, f"unknown experiment {experiment_id}")
    extra: dict = {}
    if params:
        try:
            extra = {k: float(v) for k, v in json.loads(params).items() if v not in (None, "")}
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(400, "params must be a JSON object of numbers")
        if any(not v > 0 for v in extra.values()):   # lengths and masses: zero or negative would give nonsense
            raise HTTPException(400, "every measurement must be a positive number")
    if sample:
        video = SAMPLES / f"{Path(sample).name}.mp4"
        if not video.exists():
            raise HTTPException(404, f"unknown sample {sample}")
        base, url = sample, f"/samples/{video.name}"
        settings = _sample_settings().get(sample, {})
        px_per_m = px_per_m or settings.get("px_per_m")
        experiment_id = settings.get("experiment", experiment_id)
        run_params.update(settings.get("params", {}))
    elif file is not None:
        base = re.sub(r"[^a-zA-Z0-9_-]", "-", Path(file.filename or "upload").stem)[:40] or "upload"
        raw = MEDIA / f"_raw_{uuid.uuid4().hex}{Path(file.filename or '').suffix}"
        video = MEDIA / f"{base}-{uuid.uuid4().hex[:6]}.mp4"
        try:
            raw.write_bytes(await _read_upload(file, MAX_VIDEO_BYTES, "video"))
            await asyncio.to_thread(_transcode, raw, video)
        except subprocess.CalledProcessError:
            video.unlink(missing_ok=True)
            raise HTTPException(400, "Couldn't read that file as a video. Send an MP4, MOV or WebM clip.")
        finally:
            raw.unlink(missing_ok=True)
        url = f"/media/{video.name}"
    else:
        raise HTTPException(400, "send a sample name or a video file")

    run_params.update(extra)   # the user's measurements win over a sample's stored settings
    if not JOBS.acquire(blocking=False):
        raise HTTPException(429, BUSY)
    run_id = f"{base}-{uuid.uuid4().hex[:6]}"
    run = Run(run_id, experiment_id, url, EventLog(run_id, REPLAYS / f"{run_id}.jsonl"))
    run.log.subscribe(lambda e: _broadcast(run, e))
    RUNS[run_id] = run
    diameter_m = diameter_cm / 100 if diameter_cm else None
    threading.Thread(target=_process, args=(run, video, url, px_per_m, diameter_m, run_params), daemon=True).start()
    return {"run_id": run_id}


def _get(run_id: str) -> Run:
    if run_id not in RUNS:
        raise HTTPException(404, f"unknown run {run_id}")
    return RUNS[run_id]


@app.get("/api/runs/{run_id}")
def run_status(run_id: str):
    r = _get(run_id)
    return {"id": r.id, "status": r.status, "summary": r.summary, "error": r.error}


@app.get("/api/runs/{run_id}/events")
def run_events(run_id: str):
    return _get(run_id).log.events


@app.websocket("/api/runs/{run_id}/stream")
async def stream(ws: WebSocket, run_id: str):
    """Sends every event so far, then new ones as they happen, then {"type": "_end"}."""
    if not origin_allowed(ws.headers.get("origin")):
        await ws.close(code=4403)
        return
    await ws.accept()
    if run_id not in RUNS:
        await ws.close(code=4404)
        return
    run = RUNS[run_id]
    queue: asyncio.Queue = asyncio.Queue()
    with run.lock:
        backlog = list(run.log.events)
        finished = run.status != "running"
        run.subscribers.append((asyncio.get_running_loop(), queue))
    try:
        for event in backlog:
            await ws.send_json(event)
        if finished:
            await ws.send_json({"type": "_end", "status": run.status, "error": run.error})
        else:
            # An event emitted while we were subscribing can be in the backlog and the queue; the log
            # broadcasts the same dict object it stores, so identity tells them apart.
            already_sent = {id(e) for e in backlog}
            while True:
                event = await queue.get()
                if id(event) in already_sent:
                    continue
                await ws.send_json(event)
                if event.get("type") == "_end":
                    break
        await ws.close()
    except WebSocketDisconnect:
        pass
    finally:
        with run.lock:
            run.subscribers = [s for s in run.subscribers if s[1] is not queue]
