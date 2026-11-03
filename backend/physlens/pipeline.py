"""One experiment run: video in, event stream out.

Order of work: track → calibrate → split into flights → fit → lock predictions → verify → explain.
Predictions are computed only from samples before the lock contact; the verification samples are never
seen by the fit. Timeline events carry the video time they belong to, so the stage display can play them
in sync with the video.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from .agent import ScientistAgent
from .events import EventLog
from .experiments import ExperimentCard
from .ledger import Ledger
from .perception import Tracker, default_tracker
from .physics.discover import discover
from .physics import (
    akaike_weights, calibrate_known_object, calibrate_ruler, consistent_flights, find_bounces, fit_all, fit_free_fall,
    fit_horizontal, measure_flight, predict_next_flight, split_flights,
)

UI_TRACK_RATE = 60.0  # track events per second of video sent to the stage


@dataclass
class _Pending:
    t: float
    type: str
    fields: dict


def video_info(path: Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    info = {"fps": fps, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps}
    cap.release()
    return info


def _hypothesis(fits) -> dict:
    return {"candidates": [{"name": f.model, "equation": f.equation, "score": round(float(w), 4),
                            "aic": round(float(f.aic), 2)} for f, w in zip(fits, akaike_weights(fits))]}


def _fit_event(fit_id: str, fit, extra_rel_err: float = 0.0) -> dict:
    params = {}
    for name, (value, err) in fit.shared.items():
        if name == "g":
            err = math.hypot(err, value * extra_rel_err)
        params[name] = [round(value, 4), round(err, 4)]
    return {"id": fit_id, "model": fit.model, "params": params, "rmse_m": round(fit.rmse, 5),
            "window": [round(fit.window[0], 4), round(fit.window[1], 4)]}


def _choose_lock(flights, max_contacts: int):
    """The first run of back-to-back flights. Lock after `max_contacts` contacts if the run is long enough
    to also verify the next flight, otherwise after as many as it allows (at least two)."""
    run = flights[:1]
    for f in flights[1:]:
        if run[-1].bounce_after is not None and run[-1].bounce_after == f.bounce_before:
            run.append(f)
        elif len(run) >= 2:
            break
        else:
            run = [f]
    complete = [f for f in run if f.bounce_after is not None]
    if len(complete) < 2:
        raise RuntimeError("need at least two clean bounces to measure and predict")
    k = max(2, min(max_contacts, len(run) - 1)) if len(run) >= 3 else 2
    k = min(k, len(complete))
    seen = run[:k]
    nxt = run[k] if k < len(run) else None
    return seen, nxt


def _gravity_points_up(idx: np.ndarray, y_px: np.ndarray) -> bool:
    """In a normal picture a falling object speeds up downwards, so its image-y second difference is
    positive in free flight. Negative means the picture is upside down. The median ignores bounces."""
    ok = np.nonzero((np.diff(idx)[:-1] == 1) & (np.diff(idx)[1:] == 1))[0] + 1
    if len(ok) < 5:
        return False
    accel = y_px[ok + 1] - 2 * y_px[ok] + y_px[ok - 1]
    return float(np.median(accel)) < 0


def _smooth(a: np.ndarray, n: int = 5) -> np.ndarray:
    if len(a) < n:
        return a
    pad = n // 2
    padded = np.pad(a, pad, mode="edge")
    return np.convolve(padded, np.ones(n) / n, mode="valid")


def run_pipeline(video: Path, card: ExperimentCard, log: EventLog, ledger: Optional[Ledger] = None,
                 tracker: Optional[Tracker] = None, agent: Optional[ScientistAgent] = None,
                 video_url: str = "", px_per_m: Optional[float] = None,
                 diameter_m: Optional[float] = None, params: Optional[dict] = None,
                 tracker_for: Optional[Callable] = None) -> dict:
    """Scale comes from a ruler in the shot (`px_per_m`) if given, otherwise from the tracked object's
    known diameter (`diameter_m`, defaulting to the experiment card's).

    Stage experiments with several objects (card.kind != "drop") track each object separately:
    `tracker_for(object_spec)` returns its tracker (default: SAM 3.1 by name when configured, otherwise the
    offline colour tracker). `params` override the card's defaults, e.g. a taped table height."""
    ledger = ledger or Ledger()
    agent = agent or ScientistAgent()
    if card.kind != "drop":
        return _run_lab(video, card, log, ledger, agent, video_url, px_per_m, {**card.params, **(params or {})},
                        tracker_for)
    tracker = tracker or default_tracker(card.color)
    info = video_info(video)
    log.emit("run_state", 0, state="created", video={"url": video_url, **info},
             message=f"tracker: {tracker.name} · agent: {agent.name}")
    try:
        log.emit("run_state", 0, state="tracking", message=f"Following the {card.prompt}")
        result = tracker.track(video, card.prompt)
        log.emit("run_state", 0, state="measuring", message="Fitting physical laws")
        summary = _analyse(result, card, ledger, agent, log, px_per_m, diameter_m or card.reference_diameter_m)
        log.emit("run_state", summary["t_end"], state="done", message="Run complete")
        return summary
    except Exception as exc:  # surface any failure to the stage instead of hanging
        log.emit("run_state", 0, state="error", message=str(exc))
        raise


def _lab_tracker(spec):
    """SAM 3.1 by name when a Meta key is configured (and not forced offline), otherwise the colour tracker,
    which can tell several objects apart by their colour hint."""
    import os
    from .perception import ColorBlobTracker
    if os.environ.get("PHYSLENS_TRACKER", "").lower() != "motion":
        try:
            from .perception.sam_tracker import SamTracker
            return SamTracker(color=spec.color)
        except RuntimeError:
            pass
    return ColorBlobTracker(color=spec.color)


def _run_lab(video, card, log, ledger, agent, video_url, px_per_m, params, tracker_for) -> dict:
    from .labs import projectile, race, springs
    from .labs.common import ObjectTrack
    labs = {"projectile": projectile, "race": race, "springs": springs}
    tracker_for = tracker_for or _lab_tracker
    info = video_info(video)
    names = " · ".join(f"{o.prompt}: {tracker_for(o).name}" for o in card.objects)
    log.emit("run_state", 0, state="created", video={"url": video_url, **info},
             message=f"{card.title} · {names} · agent: {agent.name}")
    try:
        from concurrent.futures import ThreadPoolExecutor
        log.emit("run_state", 0, state="tracking", message="Following the " + " and the ".join(o.prompt for o in card.objects))
        # Each object is its own SAM call: run them side by side rather than one after another.
        with ThreadPoolExecutor(max_workers=max(len(card.objects), 1)) as pool:
            results = list(pool.map(lambda spec: tracker_for(spec).track(video, spec.prompt), card.objects))
        tracks = {spec.id: ObjectTrack.from_result(spec.id, spec.prompt, r) for spec, r in zip(card.objects, results)}
        log.emit("run_state", 0, state="measuring", message="Fitting physical laws")
        summary = labs[card.kind].analyse(tracks, card, params, ledger, agent, log, info, px_per_m=px_per_m)
        log.emit("run_state", summary["t_end"], state="done", message="Run complete")
        return summary
    except Exception as exc:
        log.emit("run_state", 0, state="error", message=str(exc))
        raise


def _analyse(result, card: ExperimentCard, ledger: Ledger, agent: ScientistAgent, log: EventLog,
             px_per_m: Optional[float], diameter_m: float) -> dict:
    fps, H = result.fps, result.height
    idx = np.array([i for i, d in enumerate(result.frames) if d is not None])
    if len(idx) < max(12, 0.35 * len(result.frames)):
        raise RuntimeError(f"the {card.prompt} was only found in {len(idx)} of {len(result.frames)} frames")
    dets = [result.frames[i] for i in idx]
    t = idx / fps
    x_px = np.array([d.x_px for d in dets])
    y_px = np.array([d.y_px for d in dets])
    r_px = np.array([d.r_px for d in dets])
    if px_per_m:
        rough = calibrate_ruler(px_per_m, H)
        radius_m = float(np.median(r_px)) / px_per_m
    else:
        # Bounce detection only needs a rough scale; the final scale uses frames before the lock only.
        rough = calibrate_known_object(r_px, diameter_m, H)
        radius_m = diameter_m / 2
    upside_down = _gravity_points_up(idx, y_px)
    rough.flipped = upside_down
    _, y_rough = rough.to_world(x_px, y_px)
    min_samples = 4 if fps < 40 else 6   # low frame rates give short flights
    flights = consistent_flights(t, y_rough, split_flights(t, find_bounces(t, y_rough, radius_m), fps, min_samples),
                                 min_samples)

    seen, nxt = _choose_lock(flights, card.lock_after_contacts)

    lock_idx = seen[-1].bounce_after
    cal = rough if px_per_m else calibrate_known_object(r_px[: lock_idx + 1], diameter_m, H)
    cal.flipped = upside_down
    x, y = cal.to_world(x_px, y_px)

    timeline: list[_Pending] = []
    add = lambda t_, type_, **f: timeline.append(_Pending(float(t_), type_, f))

    add(t[0], "narration", text=card.narration["intro"])
    add(t[0], "calibration", method=cal.method, px_per_m=round(cal.px_per_m, 3),
        rel_uncertainty=round(cal.rel_uncertainty, 4))
    if upside_down:
        t_dir = t[min(8, len(t) - 1)]
        add(t_dir, "memory_vs_eyes", quantity="direction", memory_value=1, eyes_value=-1,
            memory_text="Things fall down", eyes_text="Here the ball falls up: the picture is upside down")
        add(t_dir, "narration", text=card.narration["upside_down"])

    xs, ys = _smooth(x), _smooth(y)
    vx_all, vy_all = np.gradient(xs, t), np.gradient(ys, t)
    step = max(1, int(round(fps / UI_TRACK_RATE)))
    for k in range(0, len(t), step):
        d = dets[k]
        add(t[k], "track", objects=[{
            "id": "ball", "label": card.prompt, "x_px": round(d.x_px, 1), "y_px": round(d.y_px, 1),
            "r_px": round(d.r_px, 2), "x_m": round(float(x[k]), 4), "y_m": round(float(y[k]), 4),
            "vx": round(float(vx_all[k]), 3), "vy": round(float(vy_all[k]), 3), "outline": d.outline,
        }])

    # First contact: one flight seen.
    t_first = t[seen[0].bounce_after]
    fit_first = fit_free_fall(t, y, seen[:1])
    add(t_first, "fit", **_fit_event("fit_first", fit_first, cal.rel_uncertainty))
    add(t_first, "hypothesis", **_hypothesis(fit_all(t, y, seen[:1])))
    add(t_first, "narration", text=card.narration["measuring"].format(g=fit_first.shared["g"][0]))

    # Lock: two flights seen. Everything below uses samples before the lock contact only.
    fit_lock = fit_free_fall(t, y, seen)
    vx = fit_horizontal(t, x, seen)[-1]
    pred = predict_next_flight(fit_lock, t, seen, vx, float(x[lock_idx]))
    t_lock = pred.contact_time
    used_until = float(t[seen[-1].stop - 1])
    g, g_stat = fit_lock.shared["g"]
    g_err = math.hypot(g_stat, g * cal.rel_uncertainty)
    gap_sigma = abs(g - card.memory_g) / g_err
    slowmo = math.sqrt(card.memory_g / g) if g > 0 else float("nan")

    add(t_lock, "fit", **_fit_event("fit_lock", fit_lock, cal.rel_uncertainty))
    try:
        law = discover([(t[f.start:f.stop], y[f.start:f.stop]) for f in seen], fps, axis="y",
                       scale_rel=cal.rel_uncertainty)
        add(t_lock, "law", **law.to_event("law_y", "Between bounces: up and down", "ball"))
    except ValueError:
        pass
    add(t_lock, "hypothesis", **_hypothesis(fit_all(t, y, seen)))
    add(t_lock, "hypothesis", topic="bounce", candidates=[
        {"name": law.name, "equation": law.equation, "score": round(law.weight, 4)} for law in pred.laws])
    speed_dependent = pred.chosen_law == "speed_dependent_restitution"
    if speed_dependent:
        add(t_lock, "narration", text=card.narration["bounce_law"])
    add(t_lock, "memory_vs_eyes", quantity="g", unit="m/s²", memory_value=card.memory_g,
        eyes_value=round(g, 3), eyes_err=round(g_err, 3),
        memory_text=f"Falling things accelerate at {card.memory_g:.2f} m/s²",
        eyes_text=f"This video shows {g:.2f} ± {g_err:.2f} m/s²")

    max_e = max(c.restitution.value for c in pred.contacts)
    reversed_clip = max_e > 1.03   # a passive ball can't leave a bounce faster than it arrived
    evidence = {"g": g, "g_err": g_err, "memory_g": card.memory_g, "gravity_gap_sigma": gap_sigma,
                "slowmo_factor": slowmo, "restitution_pct": pred.restitution.value * 100,
                "speed_dependent": float(speed_dependent), "upside_down": float(upside_down),
                "reversed": float(reversed_clip)}
    for n, c in enumerate(pred.contacts[-2:], start=1):
        evidence[f"contact{n}_restitution_pct"] = c.restitution.value * 100
        evidence[f"contact{n}_impact_speed"] = c.impact_speed.value

    if reversed_clip:
        gain = (max_e - 1) * 100
        evidence["energy_gain_pct"] = gain
        evidence["restitution_pct"] = max_e * 100
        add(t_lock, "memory_vs_eyes", quantity="bounce", unit="%", memory_value=100.0, eyes_value=round(max_e * 100, 1),
            memory_text="A bouncing ball always leaves slower than it arrived",
            eyes_text=f"This ball leaves each bounce {gain:.0f}% faster than it arrived")
        add(t_lock, "narration", text=card.narration["reversed"].format(gain=gain))
        return _finish(timeline, log, agent, evidence, t, y, flights, cal, add, t_lock + 0.05,
                       summary={"g": g, "g_err": g_err, "slowmo_factor": slowmo, "restitution": max_e,
                                "bounce_law": pred.chosen_law, "upside_down": upside_down, "reversed": True,
                                "contacts": [(round(c.impact_speed.value, 3), round(c.restitution.value, 4))
                                             for c in pred.contacts],
                                "peak_pred_cm": None, "peak_ci90_cm": None, "verdicts": [],
                                "flights_used": len(flights), "tracker": result.label})

    ghost = []
    for tg, xg, yg in pred.ghost:
        gx, gy = cal.to_image(np.array(xg), np.array(yg))
        ghost.append([round(tg, 4), round(float(gx), 1), round(float(gy), 1)])
    # Intervals can't be tighter than what the video can resolve: one pixel in height, one frame in time.
    px_floor_cm = 100.0 / cal.px_per_m
    frame_floor_s = 1.0 / fps
    peak_cm = pred.peak_height.value * 100
    lo, hi = pred.peak_height.ci90
    peak = ledger.lock({
        "id": "peak", "text": f"Next bounce peaks at {peak_cm:.1f} cm", "quantity": "peak_height",
        "value": round(peak_cm, 2), "unit": "cm",
        "ci90": [round(lo * 100 - px_floor_cm, 2), round(hi * 100 + px_floor_cm, 2)],
        "target_t": round(pred.peak_time.value, 4), "based_on_until_t": round(used_until, 4), "ghost": ghost,
    })
    lo, hi = pred.landing_time.ci90
    landing = ledger.lock({
        "id": "landing", "text": f"It lands again at {pred.landing_time.value:.3f} s", "quantity": "landing_time",
        "value": round(pred.landing_time.value, 4), "unit": "s",
        "ci90": [round(lo - frame_floor_s, 4), round(hi + frame_floor_s, 4)],
        "target_t": round(pred.landing_time.value, 4), "based_on_until_t": round(used_until, 4),
    })
    add(t_lock, "run_state", state="locked", message="Predictions locked")
    add(t_lock, "prediction", **peak)
    add(t_lock, "prediction", **landing)
    add(t_lock, "narration", text=card.narration["locked"].format(peak_cm=peak_cm))

    evidence["peak_pred_cm"] = peak_cm
    verdicts = []
    t_explain = t_lock + 0.05
    if nxt is not None:
        actual = measure_flight(t, y, nxt, pred.contact_height)
        actual_cm = actual["peak_height"] * 100
        v_peak = ledger.verify("peak", round(actual_cm, 2))
        add(actual["peak_time"], "verdict", **v_peak)
        add(actual["peak_time"], "narration",
            text=card.narration["hit" if v_peak["hit"] else "miss"].format(actual_cm=actual_cm))
        verdicts.append(v_peak)
        evidence["peak_actual_cm"] = actual_cm
        t_explain = actual["peak_time"] + 0.05
        if math.isfinite(actual["landing_time"]):
            v_land = ledger.verify("landing", round(actual["landing_time"], 4))
            add(actual["landing_time"], "verdict", **v_land)
            verdicts.append(v_land)
            t_explain = actual["landing_time"] + 0.05

    return _finish(timeline, log, agent, evidence, t, y, flights, cal, add, t_explain,
                   summary={"g": g, "g_err": g_err, "slowmo_factor": slowmo, "restitution": pred.restitution.value,
                            "bounce_law": pred.chosen_law, "upside_down": upside_down, "reversed": False,
                            "contacts": [(round(c.impact_speed.value, 3), round(c.restitution.value, 4))
                                         for c in pred.contacts],
                            "peak_pred_cm": peak_cm, "peak_ci90_cm": peak["ci90"], "verdicts": verdicts,
                            "flights_used": len(flights), "tracker": result.label})


def _finish(timeline, log, agent, evidence, t, y, flights, cal, add, t_explain, summary) -> dict:
    """Final law race over every flight, the explanation, and the timeline written out in video order."""
    all_fits = fit_all(t, y, flights)
    t_end = float(t[-1])
    add(t_end, "hypothesis", **_hypothesis(all_fits))
    add(t_end, "fit", **_fit_event("fit_all", all_fits[1], cal.rel_uncertainty))
    explanation = agent.explain(evidence)
    add(t_explain, "explain", text=explanation.text, evidence=["fit_lock", "peak", "landing"],
        verified=explanation.verified, source=explanation.source)
    add(t_explain, "narration", text=explanation.text)
    for p in sorted(timeline, key=lambda p: p.t):
        log.emit(p.type, p.t, **p.fields)
    return {**summary, "explanation": explanation.text, "explanation_source": explanation.source,
            "t_end": max(t_end, t_explain)}
