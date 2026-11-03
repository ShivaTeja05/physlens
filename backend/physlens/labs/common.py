"""Shared pieces for the stage experiments: tracks per object, a timeline that writes events in video order,
Monte Carlo intervals and resolution floors."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from ..events import EventLog
from ..perception import TrackResult
from ..physics import Calibration, Estimate, calibrate_known_object, calibrate_ruler

UI_TRACK_RATE = 60.0
N_SAMPLES = 4000


@dataclass
class ObjectTrack:
    id: str
    label: str
    idx: np.ndarray      # frame indices where the object was found
    t: np.ndarray        # seconds
    x_px: np.ndarray
    y_px: np.ndarray
    r_px: np.ndarray
    outlines: list

    @classmethod
    def from_result(cls, oid: str, label: str, result: TrackResult) -> "ObjectTrack":
        idx = np.array([i for i, d in enumerate(result.frames) if d is not None], int)
        if len(idx) < max(10, 0.3 * len(result.frames)):
            raise RuntimeError(f"the {label} was only found in {len(idx)} of {len(result.frames)} frames")
        dets = [result.frames[i] for i in idx]
        return cls(oid, label, idx, idx / result.fps, np.array([d.x_px for d in dets]),
                   np.array([d.y_px for d in dets]), np.array([d.r_px for d in dets]), [d.outline for d in dets])

    def until(self, t_max: float) -> np.ndarray:
        return self.t <= t_max


def calibrate(tracks: list[ObjectTrack], height: int, diameter_m: Optional[float], px_per_m: Optional[float]) -> Calibration:
    if px_per_m:
        return calibrate_ruler(px_per_m, height)
    return calibrate_known_object(np.concatenate([tr.r_px for tr in tracks]), diameter_m, height)


@dataclass
class Timeline:
    items: list = field(default_factory=list)

    def add(self, t: float, type_: str, **fields) -> None:
        self.items.append((float(t), type_, fields))

    def emit(self, log: EventLog) -> None:
        for t, type_, fields in sorted(self.items, key=lambda p: p[0]):
            log.emit(type_, t, **fields)


def add_tracks(tl: Timeline, tracks: list[ObjectTrack], cal: Calibration, fps: float) -> None:
    """One track event per ~1/60 s of video, holding every object seen in that frame."""
    step = max(1, int(round(fps / UI_TRACK_RATE)))
    frames = sorted({int(i) for tr in tracks for i in tr.idx if i % step == 0})
    lookup = {tr.id: {int(i): k for k, i in enumerate(tr.idx)} for tr in tracks}
    world = {tr.id: cal.to_world(tr.x_px, tr.y_px) for tr in tracks}
    vel = {}
    for tr in tracks:
        x, y = world[tr.id]
        vel[tr.id] = (np.gradient(x, tr.t), np.gradient(y, tr.t)) if len(tr.t) > 2 else (np.zeros(len(x)),) * 2
    for f in frames:
        objs = []
        for tr in tracks:
            k = lookup[tr.id].get(f)
            if k is None:
                continue
            objs.append({"id": tr.id, "label": tr.label, "x_px": round(float(tr.x_px[k]), 1),
                         "y_px": round(float(tr.y_px[k]), 1), "r_px": round(float(tr.r_px[k]), 2),
                         "x_m": round(float(world[tr.id][0][k]), 4), "y_m": round(float(world[tr.id][1][k]), 4),
                         "vx": round(float(vel[tr.id][0][k]), 3), "vy": round(float(vel[tr.id][1][k]), 3),
                         "outline": tr.outlines[k]})
        if objs:
            tl.add(f / fps, "track", objects=objs)


def polyfit_cov(t: np.ndarray, v: np.ndarray, deg: int):
    """Least-squares polynomial with its covariance (highest power first, like np.polyfit)."""
    coef, cov = np.polyfit(t, v, deg, cov=True)
    return coef, cov


def samples(mean, cov, n: int = N_SAMPLES, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).multivariate_normal(np.asarray(mean, float), np.asarray(cov, float), size=n,
                                                           method="eigh")


def interval(values: np.ndarray, floor: float = 0.0) -> tuple[float, tuple[float, float]]:
    """Median and 90% interval, widened by a resolution floor (what the video can't resolve)."""
    e = Estimate.from_samples(np.asarray(values, float))
    return e.value, (e.ci90[0] - floor, e.ci90[1] + floor)


def measurement(label: str, value: float, err: Optional[float], unit: str) -> dict:
    # Uncertainty to 2 significant figures: a period known to ±0.00006 s mustn't print as ±0.
    return {"label": label, "value": round(float(value), 6), "err": None if err is None else float(f"{float(err):.2g}"),
            "unit": unit}


def discover_law(tl: Timeline, t: float, segments, fps: float, axis: str, label: str, obj: str | None = None,
                 id_: str | None = None, scale_rel: float = 0.0):
    """Search for the equation of motion along one axis and put it on the timeline. Returns the law, or None when the
    track is too short to tell."""
    from ..physics.discover import discover
    try:
        law = discover(segments, fps, axis=axis, scale_rel=scale_rel)
    except ValueError:
        return None
    tl.add(t, "law", **law.to_event(id_ or f"law_{obj or 'obj'}_{axis}", label, obj))
    return law


def explain(agent, tl: Timeline, t: float, evidence: dict, template: Callable[[dict], str], glossary: str,
            evidence_ids: list[str]) -> dict:
    ex = agent.explain(evidence, template=template, glossary=glossary)
    tl.add(t, "explain", text=ex.text, evidence=evidence_ids, verified=ex.verified, source=ex.source)
    tl.add(t, "narration", text=ex.text)
    return {"explanation": ex.text, "explanation_source": ex.source}
