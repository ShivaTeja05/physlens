"""Mystery Cans: two identical-looking cans roll down the same slope. Where each hides its mass decides how fast it
speeds up, so the first part of the race predicts the winner and both finish times.

Physics: a can rolling without slipping down a slope θ accelerates at a = g·sinθ / (1 + k), where k = I / (m r²)
says where the mass sits: about 0.5 for a solid can, less with weight packed at the centre, about 1 with weight at
the rim. Each can's acceleration is fitted from the first stretch of its run; the finish line is a measured distance
from the start. g is Earth's (a slope can't separate g from k).
"""
from __future__ import annotations

import math

import numpy as np

from ..physics import STANDARD_GRAVITY
from .common import (ObjectTrack, Timeline, add_tracks, calibrate, discover_law, explain, interval, measurement,
                     samples)

EARLY_FRACTION = 0.35   # predictions lock once both cans have covered this share of the course
SHAPES = [("weight at the centre", 0.25), ("evenly spread (solid)", 0.5), ("weight at the rim (hollow)", 1.0)]

GLOSSARY = """- For each can (a, b): accel = acceleration down the slope (m/s²), k = I/(m·r²): about 0.5 for a solid
  can, smaller when the mass sits near the centre, about 1 when it sits at the rim.
- winner, margin_pred_s: who was called to win and by how much, locked after the first part of the race.
- margin_actual_s: the real winning margin. slope_deg: ramp angle."""


def template(ev: dict) -> str:
    centre, rim = (ev["name_a"], ev["name_b"]) if ev["k_a"] < ev["k_b"] else (ev["name_b"], ev["name_a"])
    s = (f"The {ev['name_a']} can speeds up at {ev['accel_a']:.2f} m/s² and the {ev['name_b']} at {ev['accel_b']:.2f}, "
         f"so the {centre} can hides its weight nearer the centre and the {rim} can nearer the rim "
         f"(k {ev['k_a']:.2f} vs {ev['k_b']:.2f}).")
    s += f" Called: {ev['winner']} wins by {ev['margin_pred_s']:.2f} s"
    if "margin_actual_s" in ev:
        s += f"; it won by {ev['margin_actual_s']:.2f} s."
    else:
        s += "."
    return s


def _course(tr: ObjectTrack, x: np.ndarray, y: np.ndarray, mask: np.ndarray):
    """Direction of travel (principal axis of the early track), slope angle, and distance along it from the start."""
    P = np.stack([x[mask], y[mask]], axis=1)
    d = np.linalg.svd(P - P.mean(axis=0))[2][0]
    if d[0] < 0:
        d = -d
    s = (np.stack([x, y], axis=1) - np.array([x[0], y[0]])) @ d
    return d, math.degrees(math.atan2(-d[1], d[0])), s


def _finish_time(coef_samples: np.ndarray, distance: float) -> np.ndarray:
    a2, b, c = coef_samples[:, 0], coef_samples[:, 1], coef_samples[:, 2] - distance
    disc = np.maximum(b ** 2 - 4 * a2 * c, 0)
    return (-b + np.sqrt(disc)) / (2 * a2)


def analyse(tracks: dict[str, ObjectTrack], card, params: dict, ledger, agent, log, video: dict,
            px_per_m=None) -> dict:
    ids = [o.id for o in card.objects]
    names = {o.id: o.prompt.split()[0] for o in card.objects}
    fps, H = video["fps"], video["height"]
    cans = [tracks[i] for i in ids]
    cal = calibrate(cans, H, params.get("can_diameter_m"), px_per_m)
    L = float(params["finish_distance_m"])
    g0 = STANDARD_GRAVITY

    tl = Timeline()
    tl.add(cans[0].t[0], "narration", text=card.narration["intro"])
    tl.add(cans[0].t[0], "calibration", method=cal.method, px_per_m=round(cal.px_per_m, 3),
           rel_uncertainty=round(cal.rel_uncertainty, 4))
    add_tracks(tl, cans, cal, fps)

    world = {tr.id: cal.to_world(tr.x_px, tr.y_px) for tr in cans}
    # Lock once BOTH cans have covered the early stretch.
    t_reach = {}
    for tr in cans:
        x, y = world[tr.id]
        _, _, s = _course(tr, x, y, np.ones(len(x), bool))
        hit = np.nonzero(s >= EARLY_FRACTION * L)[0]
        if len(hit) == 0:
            raise RuntimeError(f"the {tr.label} never covered {EARLY_FRACTION:.0%} of the course")
        t_reach[tr.id] = float(tr.t[hit[0]])
    t_lock = max(t_reach.values())

    fits, ev = {}, {}
    for tr in cans:
        x, y = world[tr.id]
        early = tr.until(t_lock)
        d, slope, s = _course(tr, x, y, early)
        coef, cov = np.polyfit(tr.t[early], s[early], 2, cov=True)   # s = ½a t² + v t + s0
        S = samples(coef, cov, seed=hash(tr.id) % 1000)
        scale = np.random.default_rng(7).normal(1.0, cal.rel_uncertainty, len(S))
        accel = 2 * S[:, 0] * scale
        sin_t = math.sin(math.radians(slope))
        k = g0 * sin_t / accel - 1
        # If the pixel scale is off by `scale`, the taped course length looks like L / scale in our metres.
        finish = _finish_time(S, L / scale)
        fits[tr.id] = {"coef": coef, "accel": accel, "k": k, "finish": finish, "slope": slope, "dir": d, "s": s}
        discover_law(tl, t_lock, [(tr.t[early], s[early])], fps, "s", f"{names[tr.id].capitalize()} can, along the slope",
                     tr.id, scale_rel=cal.rel_uncertainty)
        ev[f"accel_{'a' if tr.id == ids[0] else 'b'}"] = float(np.median(accel))
        ev[f"k_{'a' if tr.id == ids[0] else 'b'}"] = float(np.median(k))

    a_id, b_id = ids
    margin = fits[b_id]["finish"] - fits[a_id]["finish"]      # > 0: a wins
    m_val, m_ci = interval(np.abs(margin), floor=1.0 / fps)
    winner_id = a_id if np.median(margin) > 0 else b_id
    loser_id = b_id if winner_id == a_id else a_id
    ev.update(name_a=names[a_id], name_b=names[b_id], winner=names[winner_id], margin_pred_s=m_val,
              slope_deg=fits[a_id]["slope"])

    items = []
    for tr in cans:
        f = fits[tr.id]
        items.append(measurement(f"{names[tr.id].capitalize()} can: acceleration", float(np.median(f["accel"])),
                                 float(np.std(f["accel"])), "m/s²"))
        items.append(measurement(f"{names[tr.id].capitalize()} can: k (mass placement)", float(np.median(f["k"])),
                                 float(np.std(f["k"])), ""))
        sigma = max(float(np.std(f["k"])), 0.08)
        w = np.array([math.exp(-0.5 * ((float(np.median(f["k"])) - kk) / sigma) ** 2) for _, kk in SHAPES])
        w /= w.sum()
        tl.add(t_lock, "hypothesis", topic=f"mass:{tr.id}", label=f"Where is the {names[tr.id]} can's weight?",
               candidates=[{"name": n, "equation": f"k ≈ {kk}", "score": round(float(s), 4)}
                           for (n, kk), s in zip(SHAPES, w)])
    items.append(measurement("Slope", fits[a_id]["slope"], None, "°"))
    tl.add(t_lock, "measurement", items=items)
    tl.add(t_lock, "run_state", state="locked", message="Race result locked")

    based = round(t_lock, 4)
    preds = []
    for tr in cans:
        f = fits[tr.id]
        ft, ft_ci = interval(f["finish"], floor=1.0 / fps)
        d = f["dir"]
        x, y = world[tr.id]
        ghost = []
        for tt in np.linspace(t_lock, ft, 24):
            s_pred = np.polyval(f["coef"], tt)
            gx, gy = cal.to_image(np.array(x[0] + d[0] * s_pred), np.array(y[0] + d[1] * s_pred))
            ghost.append([round(float(tt), 4), round(float(gx), 1), round(float(gy), 1)])
        p = ledger.lock({"id": f"finish_{tr.id}", "text": f"{names[tr.id].capitalize()} finishes at {ft:.3f} s",
                         "quantity": "finish_time", "value": round(ft, 4), "unit": "s",
                         "ci90": [round(ft_ci[0], 4), round(ft_ci[1], 4)], "target_t": round(ft, 4),
                         "based_on_until_t": based, "ghost": ghost, "object": tr.id})
        preds.append(p)
        tl.add(t_lock, "prediction", **p)
    win = ledger.lock({"id": "winner", "text": f"{names[winner_id].capitalize()} wins by {m_val:.2f} s",
                       "quantity": "margin", "value": round(m_val, 4), "unit": "s",
                       "ci90": [round(m_ci[0], 4), round(m_ci[1], 4)],
                       "target_t": round(float(np.median(fits[loser_id]["finish"])), 4), "based_on_until_t": based,
                       "winner": winner_id})
    tl.add(t_lock, "prediction", **win)
    tl.add(t_lock, "narration", text=card.narration["locked"].format(winner=names[winner_id], margin=m_val))

    # What actually happened: when each can's distance first passed the finish line (interpolated).
    actual = {}
    for tr in cans:
        s = fits[tr.id]["s"]
        cross = np.nonzero(s >= L)[0]
        if len(cross) and cross[0] > 0:
            i = cross[0]
            actual[tr.id] = float(tr.t[i - 1] + (L - s[i - 1]) / (s[i] - s[i - 1]) * (tr.t[i] - tr.t[i - 1]))
    verdicts = []
    t_explain = t_lock + 0.1
    for tr in cans:
        if tr.id in actual:
            v = ledger.verify(f"finish_{tr.id}", round(actual[tr.id], 4))
            tl.add(actual[tr.id], "verdict", **v)
            verdicts.append(v)
    if len(actual) == 2:
        true_winner = a_id if actual[a_id] < actual[b_id] else b_id
        m_act = abs(actual[a_id] - actual[b_id])
        v = ledger.verify("winner", round(m_act, 4))
        v["hit"] = bool(v["hit"] and true_winner == winner_id)
        t_done = max(actual.values())
        tl.add(t_done, "verdict", **v)
        tl.add(t_done, "narration", text=card.narration["hit" if v["hit"] else "miss"].format(
            winner=names[true_winner], margin=m_act))
        verdicts.append(v)
        ev["margin_actual_s"] = m_act
        t_explain = t_done + 0.1
    out = explain(agent, tl, t_explain, ev, template, GLOSSARY, [p["id"] for p in preds] + ["winner"])
    tl.emit(log)
    return {"experiment": "race", "k": {i: float(np.median(fits[i]["k"])) for i in ids},
            "accel": {i: float(np.median(fits[i]["accel"])) for i in ids},
            "finish_pred": {i: float(np.median(fits[i]["finish"])) for i in ids}, "finish_actual": actual,
            "winner_pred": winner_id, "verdicts": verdicts,
            "t_end": max(float(max(tr.t[-1] for tr in cans)), t_explain), **out}
