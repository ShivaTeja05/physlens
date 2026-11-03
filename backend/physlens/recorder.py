"""The black-box recorder: every run's physical record in one SQLite file, so events can be found, cut down, compared
and re-simulated later.

    capture     one analysed video: fingerprint, frame rate, scale and its uncertainty, experiment, models used
    object      each tracked thing in a capture
    state       an object's state at each sampled instant: position, velocity (metres, m/s, y up), ± position
    event       key moments: start, contact (bounce), peak, end, with speed into and out of each contact
    law         equations of motion found from the motion (law discovery), with parameters ± errors
    prediction  what was called before it happened (locked, fingerprinted) and what actually happened

Built from the run's event log (`replays/<run>.jsonl`), so old runs can be imported too. One file:
`records/physlens.db`; `export(capture_id)` gives the same record as JSON.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "records" / "physlens.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS capture (
  id TEXT PRIMARY KEY, recorded_at TEXT, experiment TEXT, title TEXT, source_url TEXT, video_sha256 TEXT,
  fps REAL, width INTEGER, height INTEGER, duration REAL,
  scale_method TEXT, px_per_m REAL, scale_rel_err REAL, gravity_up INTEGER,
  px_from_m TEXT,                 -- JSON [ax, bx, ay, by]: x_px = ax·x_m + bx, y_px = ay·y_m + by (for drawing)
  tracker TEXT, agent TEXT, status TEXT);
CREATE TABLE IF NOT EXISTS object (
  capture TEXT, id TEXT, label TEXT, radius_m REAL, PRIMARY KEY (capture, id));
CREATE TABLE IF NOT EXISTS state (
  capture TEXT, object TEXT, t REAL, x_m REAL, y_m REAL, vx REAL, vy REAL, pos_err_m REAL, x_px REAL, y_px REAL);
CREATE INDEX IF NOT EXISTS state_by_object ON state (capture, object, t);
CREATE TABLE IF NOT EXISTS event (
  capture TEXT, object TEXT, t REAL, kind TEXT, x_m REAL, y_m REAL,
  speed_in REAL, speed_out REAL, ratio REAL, note TEXT);
CREATE TABLE IF NOT EXISTS law (
  capture TEXT, object TEXT, id TEXT, axis TEXT, label TEXT, equation TEXT, terms TEXT, until_t REAL, rmse_m REAL);
CREATE TABLE IF NOT EXISTS prediction (
  capture TEXT, id TEXT, text TEXT, quantity TEXT, value REAL, unit TEXT, ci_low REAL, ci_high REAL,
  locked_t REAL, based_on_until_t REAL, target_t REAL, sha256 TEXT, actual REAL, hit INTEGER);
"""


def connect(path: Path = DB) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    return db


def _sha256(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _arc(t: np.ndarray, y: np.ndarray, sel: np.ndarray, g: Optional[float]) -> Optional[np.ndarray]:
    """Free-flight arc through the positions in `sel` as polynomial coefficients (highest first). With a known g only
    height and speed are fitted, which is what a few frames at 25 fps can support."""
    if sel.sum() < 4:      # a 3-frame arc can't pin a speed (it gave "bounced back faster" on a real clip)
        return None
    if g is None:
        return np.polyfit(t[sel], y[sel], 2)
    b, c = np.polyfit(t[sel], y[sel] + 0.5 * g * t[sel] ** 2, 1)
    return np.array([-0.5 * g, b, c])


def key_moments(t: np.ndarray, y: np.ndarray, vy: np.ndarray, k: int = 3, g: Optional[float] = None) -> list[dict]:
    """Start, contacts (lowest point where falling turns to rising), peaks (highest point where rising turns to
    falling) and end, from an object's states (y up). At a contact, the falling and rising arcs are fitted to the
    positions either side (with the found gravity, if any), the contact is where they meet, and the speeds in and out
    are the arcs' speeds at that instant. Smoothed velocities would smear across the impact."""
    turns = []
    for i in range(k, len(y) - k):
        win = y[i - k:i + k + 1]
        if y[i] == win.min() and y[i - k] - y[i] > 0.005 and y[i + k] - y[i] > 0.005:
            turns.append(("contact", i))
        elif y[i] == win.max() and y[i] - y[i - k] > 0.005 and y[i] - y[i + k] > 0.005:
            turns.append(("peak", i))
    dedup = []
    for kind, i in turns:   # two turns of the same kind within a few samples are one event
        if dedup and dedup[-1][0] == kind and i - dedup[-1][1] <= k:
            continue
        dedup.append((kind, i))
    out = [{"kind": "start", "i": 0}]
    idx = np.arange(len(y))
    for n, (kind, i) in enumerate(dedup):
        if kind == "peak":
            out.append({"kind": "peak", "i": i})
            continue
        lo = dedup[n - 1][1] if n > 0 else 0
        hi = dedup[n + 1][1] if n + 1 < len(dedup) else len(y) - 1
        # The lowest sample may already be bouncing back or still falling: leave it and its neighbours to the arcs
        # only if they clearly belong (the contact sits between frames).
        down = _arc(t, y, (idx >= max(lo, i - 8)) & (idx < i), g)
        up = _arc(t, y, (idx > i) & (idx <= min(hi, i + 8)), g)
        e = {"kind": "contact", "i": i}
        if down is not None and up is not None:
            roots = np.roots(down - up)
            roots = [float(r.real) for r in roots if abs(r.imag) < 1e-9 and t[max(i - 1, 0)] - 1e-6 <= r.real <= t[min(i + 1, len(t) - 1)] + 1e-6]
            t_c = min(roots, key=lambda r: abs(r - t[i])) if roots else float(t[i])
            v_in, v_out = float(np.polyval(np.polyder(down), t_c)), float(np.polyval(np.polyder(up), t_c))
            if v_in < 0 < v_out:
                e.update(speed_in=-v_in, speed_out=v_out, ratio=v_out / -v_in, t=t_c,
                         y=float(np.polyval(down, t_c)))
        out.append(e)
    out.append({"kind": "end", "i": len(y) - 1})
    return out


def record_run(run_id: str, events: list[dict], replays: Path, roots: dict[str, Path], db: Optional[sqlite3.Connection] = None,
               experiment: Optional[str] = None, title: Optional[str] = None, status: str = "done",
               recorded_at: Optional[str] = None) -> str:
    """Write (or rewrite) one run's record. `roots` maps URL prefixes ('/samples', '/media') to folders, to find the
    video for its fingerprint."""
    db = db or connect()
    created = next((e for e in events if e["type"] == "run_state" and e.get("video")), None)
    video = (created or {}).get("video", {})
    url = video.get("url", "")
    path = next((roots[p] / url[len(p) + 1:] for p in roots if url.startswith(p + "/")), None)
    msgs = [e.get("message", "") for e in events if e["type"] == "run_state"]
    tracker = next((m.split("tracker:")[1].split("·")[0].strip() for m in msgs if "tracker:" in m), None)
    agent = next((m.split("agent:")[1].strip() for m in msgs if "agent:" in m), None)
    exp = experiment or next((e["experiment"] for e in events if e.get("experiment")), None)
    cal = next((e for e in events if e["type"] == "calibration"), {})

    tracks = [e for e in events if e["type"] == "track"]
    per: dict[str, list] = {}
    labels: dict[str, str] = {}
    for e in tracks:
        for o in e["objects"]:
            if o.get("x_m") is None:
                continue
            per.setdefault(o["id"], []).append((e["t"], o["x_m"], o["y_m"], o.get("vx", 0.0), o.get("vy", 0.0),
                                                o["x_px"], o["y_px"], o.get("r_px")))
            labels[o["id"]] = o.get("label", o["id"])
    # Pixel ↔ metre mapping (affine per axis), recovered from the samples themselves.
    xs = np.array([s for v in per.values() for s in v], float) if per else np.zeros((0, 8))
    px_from_m = None
    gravity_up = None
    if len(xs) >= 2:
        ax, bx = np.polyfit(xs[:, 1], xs[:, 5], 1)
        ay, by = np.polyfit(xs[:, 2], xs[:, 6], 1)
        px_from_m = [float(ax), float(bx), float(ay), float(by)]
        gravity_up = int(ay > 0)   # y_m up with image y down means ay < 0 (the usual case)
    ppm = cal.get("px_per_m")

    with db:
        for table in ("capture", "object", "state", "event", "law", "prediction"):
            db.execute(f"DELETE FROM {table} WHERE {'id' if table == 'capture' else 'capture'} = ?", (run_id,))
        db.execute("INSERT INTO capture VALUES (?, COALESCE(?, datetime('now')), ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                   (run_id, recorded_at, exp, title, url, _sha256(path) if path else None, video.get("fps"), video.get("width"),
                    video.get("height"), video.get("duration"), cal.get("method"), ppm, cal.get("rel_uncertainty"),
                    gravity_up, json.dumps(px_from_m), tracker, agent, status))
        for oid, rows in per.items():
            a = np.array(rows, float)
            r_px = np.nanmedian(np.where(a[:, 7] > 0, a[:, 7], np.nan)) if len(a) else np.nan
            db.execute("INSERT INTO object VALUES (?, ?, ?, ?)",
                       (run_id, oid, labels[oid], float(r_px / ppm) if ppm and np.isfinite(r_px) else None))
            # Position uncertainty: a pixel of tracking, plus the scale's share of the distance from the origin.
            err = np.hypot(1.0 / ppm if ppm else 0.0, (cal.get("rel_uncertainty") or 0.0) * np.hypot(a[:, 1], a[:, 2]))
            db.executemany("INSERT INTO state VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           [(run_id, oid, *map(float, r[:5]), float(e), float(r[5]), float(r[6])) for r, e in zip(a, err)])
            law_y = next((e for e in events if e["type"] == "law" and e.get("object") == oid and e["axis"] == "y"), None)
            g = next((-t["value"] for t in law_y["terms"] if t["term"] == "1"), None) if law_y and \
                not any(t["term"] == "q" for t in law_y["terms"]) else None
            # Bounces and peaks are vertical motion; analyses that report their own moments keep those instead.
            vertical = [] if any(e["type"] == "moment" for e in events) else key_moments(a[:, 0], a[:, 2], a[:, 4], g=g)
            for m in vertical:
                i = m["i"]
                db.execute("INSERT INTO event VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (run_id, oid, m.get("t", float(a[i, 0])), m["kind"], float(a[i, 1]), m.get("y", float(a[i, 2])),
                            m.get("speed_in"), m.get("speed_out"), m.get("ratio"),
                            "contact between frames, from the arcs either side" if "t" in m else None))
        for e in events:
            if e["type"] == "moment":   # key moments an analysis reports itself (impact, max crush, rebound...)
                db.execute("INSERT INTO event VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (run_id, e.get("object"), e["t"], e["kind"], None, None, e.get("speed_in"),
                            e.get("speed_out"), (e["speed_out"] / e["speed_in"]) if e.get("speed_in") and e.get("speed_out") else None,
                            e.get("note")))
        for e in events:
            if e["type"] == "law":
                db.execute("INSERT INTO law VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (run_id, e.get("object"), e["id"], e["axis"], e["label"], e["equation"],
                            json.dumps(e["terms"]), e["t"], e.get("rmse")))
        verdicts = {e["id"]: e for e in events if e["type"] == "verdict"}
        for p in (e for e in events if e["type"] == "prediction"):
            v = verdicts.get(p["id"], {})
            lo, hi = (p.get("ci90") or [None, None])[:2]
            db.execute("INSERT INTO prediction VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (run_id, p["id"], p["text"], p.get("quantity"), p["value"], p.get("unit"), lo, hi, p["t"],
                        p.get("based_on_until_t"), p.get("target_t"), p.get("hash"), v.get("actual"),
                        None if not v else int(bool(v.get("hit")))))
    return run_id


def import_replays(replays: Path, roots: dict[str, Path], experiments: dict[str, str] | None = None) -> list[str]:
    """Record every saved run that has states (runs that failed before tracking have nothing to record)."""
    db = connect()
    done = []
    for f in sorted(replays.glob("*.jsonl")):
        if f.name.endswith(".ledger.jsonl"):
            continue
        events = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not any(e["type"] == "track" for e in events):
            continue
        stem = f.stem
        exp = next((v for k, v in (experiments or {}).items() if stem.startswith(k + "-")), None)
        ok = any(e["type"] == "run_state" and e.get("state") == "done" for e in events)
        import datetime
        when = datetime.datetime.fromtimestamp(f.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        record_run(stem, events, replays, roots, db, experiment=exp, status="done" if ok else "error", recorded_at=when)
        done.append(stem)
    return done


def export(capture_id: str, db: Optional[sqlite3.Connection] = None) -> dict:
    """One record as plain JSON (for a file, the app, or the assistant)."""
    db = db or connect()
    cap = db.execute("SELECT * FROM capture WHERE id = ?", (capture_id,)).fetchone()
    if cap is None:
        raise KeyError(capture_id)
    rows = lambda q: [dict(r) for r in db.execute(q, (capture_id,)).fetchall()]
    out = dict(cap)
    out["px_from_m"] = json.loads(out["px_from_m"]) if out["px_from_m"] else None
    out["objects"] = rows("SELECT id, label, radius_m FROM object WHERE capture = ?")
    out["events"] = rows("SELECT object, t, kind, x_m, y_m, speed_in, speed_out, ratio FROM event WHERE capture = ? ORDER BY t")
    out["laws"] = [{**r, "terms": json.loads(r["terms"])} for r in
                   rows("SELECT object, id, axis, label, equation, terms, until_t, rmse_m FROM law WHERE capture = ?")]
    out["predictions"] = rows("SELECT id, text, value, unit, ci_low, ci_high, based_on_until_t, target_t, sha256, actual, hit "
                              "FROM prediction WHERE capture = ?")
    out["states"] = {o["id"]: [dict(r) for r in db.execute(
        "SELECT t, x_m, y_m, vx, vy, pos_err_m FROM state WHERE capture = ? AND object = ? ORDER BY t",
        (capture_id, o["id"])).fetchall()] for o in out["objects"]}
    return out


def list_captures(db: Optional[sqlite3.Connection] = None, limit: int = 50) -> list[dict]:
    db = db or connect()
    return [dict(r) for r in db.execute(
        """SELECT c.id, c.recorded_at, c.experiment, c.source_url, c.fps, c.duration, c.status,
                  (SELECT COUNT(*) FROM object o WHERE o.capture = c.id) AS objects,
                  (SELECT COUNT(*) FROM event e WHERE e.capture = c.id AND e.kind = 'contact') AS contacts,
                  (SELECT COUNT(*) FROM prediction p WHERE p.capture = c.id AND p.hit = 1) AS hits,
                  (SELECT COUNT(*) FROM prediction p WHERE p.capture = c.id AND p.hit IS NOT NULL) AS verdicts
           FROM capture c ORDER BY c.recorded_at DESC, c.id LIMIT ?""", (limit,)).fetchall()]
