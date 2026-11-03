"""Catch the cup: a ball rolls along a table and off the edge. From the roll alone (before it falls), predict
where it lands, so a cup can be placed there in time.

Physics: on the table the ball moves at constant speed v. After the edge it keeps v horizontally and falls the
table height h, taking √(2h/g). Landing distance from the edge = v·√(2h/g). The table height is measured by the
presenter with a tape (the camera can't see the floor's depth reliably); g is Earth's, because no fall has been seen
when the prediction locks. The fall itself then measures g as a check.
"""
from __future__ import annotations

import math

import numpy as np

from ..physics import STANDARD_GRAVITY
from .common import (ObjectTrack, discover_law, Timeline, add_tracks, calibrate, explain, interval, measurement, polyfit_cov,
                     samples)

GLOSSARY = """- roll_speed: speed of the ball along the table, m/s. table_height_cm: table surface to floor, cm.
- landing_pred_cm: where the ball was called to land, measured from the table edge, cm, locked before it fell.
- landing_actual_cm: where it really landed, cm from the edge. fall_time_s: predicted time in the air.
- g_measured: gravity measured during the fall (a check; the prediction assumed g_assumed)."""


def template(ev: dict) -> str:
    s = (f"The ball left the table at {ev['roll_speed']:.2f} m/s from {ev['table_height_cm']:.0f} cm up, so it was "
         f"called to land {ev['landing_pred_cm']:.1f} cm past the edge")
    if "landing_actual_cm" in ev:
        s += f", and it landed at {ev['landing_actual_cm']:.1f} cm."
    else:
        s += "."
    if "g_measured" in ev:
        s += f" Its fall measured gravity at {ev['g_measured']:.2f} m/s²."
    return s


def analyse(tracks: dict[str, ObjectTrack], card, params: dict, ledger, agent, log, video: dict,
            px_per_m=None) -> dict:
    ball = tracks["ball"]
    fps, H = video["fps"], video["height"]
    cal = calibrate([ball], H, params.get("ball_diameter_m"), px_per_m)
    x, y = cal.to_world(ball.x_px, ball.y_px)
    t = ball.t
    h = float(params["table_height_m"])
    r_m = float(np.median(ball.r_px)) / cal.px_per_m
    g0 = STANDARD_GRAVITY

    tl = Timeline()
    tl.add(t[0], "narration", text=card.narration["intro"])
    tl.add(t[0], "calibration", method=cal.method, px_per_m=round(cal.px_per_m, 3),
           rel_uncertainty=round(cal.rel_uncertainty, 4))
    add_tracks(tl, [ball], cal, fps)

    # On the table: height flat. The first sample clearly below that height means it has left the edge.
    n0 = max(8, len(t) // 8)
    table_y = float(np.median(y[:n0]))
    noise = float(np.std(y[:n0]))
    drop = max(0.25 * r_m, 6 * noise, 0.003)
    falling = np.nonzero(table_y - y > drop)[0]
    if len(falling) == 0:
        raise RuntimeError("the ball never left the table")
    launch = int(falling[0])
    on_table = np.arange(0, max(launch - 2, 4))
    if len(on_table) < 6:
        raise RuntimeError("too little of the roll was seen before the edge")

    # Roll speed (constant on the table) and the moment it left the edge, from samples up to the lock only.
    coef, cov = polyfit_cov(t[on_table], x[on_table], 1)
    v, x0 = coef
    first_fall = [i for i in range(launch, min(launch + 3, len(t))) if table_y - y[i] > noise]
    t_edge = float(np.mean([t[i] - math.sqrt(2 * (table_y - y[i]) / g0) for i in first_fall]))
    t_lock = float(t[launch])

    # Monte Carlo: roll fit, tape error on the table height (±0.5 cm), pixel-scale error.
    S = samples(coef, cov)
    hs = np.random.default_rng(1).normal(h, 0.005, len(S))
    scale = np.random.default_rng(2).normal(1.0, cal.rel_uncertainty, len(S))
    vs = S[:, 0] * scale
    fall = np.sqrt(2 * hs / g0)
    d = vs * fall
    land_t = t_edge + fall
    px_cm = 100.0 / cal.px_per_m
    d_cm, d_ci = interval(d * 100, floor=px_cm)
    land_s, land_ci = interval(land_t, floor=1.0 / fps)

    x_edge = x0 + v * t_edge
    fall_mid = math.sqrt(2 * h / g0)
    ghost = []
    for tau in np.linspace(0, fall_mid, 40):
        gx, gy = cal.to_image(np.array(x_edge + v * tau), np.array(table_y - 0.5 * g0 * tau ** 2))
        ghost.append([round(t_edge + tau, 4), round(float(gx), 1), round(float(gy), 1)])
    land_px = cal.to_image(np.array(x_edge + v * fall_mid), np.array(table_y - h))

    tl.add(t_lock, "measurement", items=[
        measurement("Roll speed", v, math.sqrt(cov[0, 0]), "m/s"),
        measurement("Table height (tape)", h * 100, 0.5, "cm"),
        measurement("Time in the air", fall_mid, None, "s"),
    ])
    tl.add(t_lock, "run_state", state="locked", message="Landing spot locked")
    pred = ledger.lock({"id": "landing", "text": f"Lands {d_cm:.1f} cm past the edge", "quantity": "landing_distance",
                        "value": round(d_cm, 2), "unit": "cm", "ci90": [round(d_ci[0], 2), round(d_ci[1], 2)],
                        "target_t": round(land_s, 4), "based_on_until_t": round(float(t[on_table[-1]]), 4),
                        "ghost": ghost, "target_px": [round(float(land_px[0]), 1), round(float(land_px[1]), 1)]})
    when = ledger.lock({"id": "landing_time", "text": f"Touches down at {land_s:.3f} s", "quantity": "landing_time",
                        "value": round(land_s, 4), "unit": "s", "ci90": [round(land_ci[0], 4), round(land_ci[1], 4)],
                        "target_t": round(land_s, 4), "based_on_until_t": round(float(t[on_table[-1]]), 4)})
    tl.add(t_lock, "prediction", **pred)
    tl.add(t_lock, "prediction", **when)
    tl.add(t_lock, "narration", text=card.narration["locked"].format(cm=d_cm))

    evidence = {"roll_speed": v, "table_height_cm": h * 100, "landing_pred_cm": d_cm, "fall_time_s": fall_mid,
                "g_assumed": g0}
    verdicts = []
    # What actually happened: fit the flight (x linear, y quadratic) and find where it meets the floor height.
    floor_y = table_y - h
    flight = np.nonzero((np.arange(len(t)) >= launch) & (y > floor_y + 0.3 * r_m))[0]
    t_explain = land_s + 0.1
    if len(flight) >= 5:
        cx, _ = polyfit_cov(t[flight], x[flight], 1)
        cy, _ = polyfit_cov(t[flight], y[flight], 2)
        roots = np.roots([cy[0], cy[1], cy[2] - floor_y])
        roots = np.real(roots[np.isreal(roots)])
        roots = roots[roots > t[launch]]
        t_flight_end = float(t[flight[-1]])
        discover_law(tl, t_flight_end, [(t[flight], x[flight])], fps, "x", "In the air: sideways", "ball", "law_x",
                     cal.rel_uncertainty)
        discover_law(tl, t_flight_end, [(t[flight], y[flight])], fps, "y", "In the air: up and down", "ball", "law_y",
                     cal.rel_uncertainty)
        if roots.size:
            t_land = float(roots.min())
            d_actual = (np.polyval(cx, t_land) - x_edge) * 100
            g_meas = -2 * cy[0]
            evidence.update(landing_actual_cm=d_actual, g_measured=g_meas)
            v1 = ledger.verify("landing", round(float(d_actual), 2))
            v2 = ledger.verify("landing_time", round(t_land, 4))
            tl.add(t_land, "verdict", **v1)
            tl.add(t_land, "verdict", **v2)
            tl.add(t_land, "narration", text=card.narration["hit" if v1["hit"] else "miss"].format(cm=d_actual))
            tl.add(t_land, "memory_vs_eyes", quantity="g", unit="m/s²", memory_value=g0, eyes_value=round(g_meas, 3),
                   memory_text=f"Prediction assumed Earth's {g0:.2f} m/s²",
                   eyes_text=f"The fall measured {g_meas:.2f} m/s²")
            verdicts = [v1, v2]
            t_explain = t_land + 0.1
    out = explain(agent, tl, t_explain, evidence, template, GLOSSARY, ["landing", "landing_time"])
    tl.emit(log)
    return {"experiment": "projectile", "roll_speed": v, "landing_pred_cm": d_cm, "landing_ci_cm": d_ci,
            "verdicts": verdicts, "g_measured": evidence.get("g_measured"), "t_end": max(float(t[-1]), t_explain), **out}
