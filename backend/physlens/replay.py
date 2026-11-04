"""Replay and what-if: re-simulate a recorded event from its stored state and the law found for it, then change one
thing and see what would have happened.

    replay   start from the recorded state, apply the law found from the motion (and the measured bounciness at
             each contact), and compare with what really happened: how well the record explains the event.
    what-if  the same, with gravity, bounciness, launch speed or drag changed.

Supported: vertical motion under a found law (drops, bounces, flights, springs), with sideways motion at its
recorded speed. Everything comes from the record (`recorder.export`), so it works on any past run.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class Unsupported(ValueError):
    pass


def _params(rec: dict, obj: str) -> dict:
    """What the record measured: the vertical law, bounciness, floor height, sideways speed, start state."""
    states = rec["states"].get(obj)
    if not states:
        raise Unsupported(f"no recorded states for {obj}")
    t = np.array([s["t"] for s in states]); x = np.array([s["x_m"] for s in states])
    y = np.array([s["y_m"] for s in states]); vy = np.array([s["vy"] for s in states])
    contacts = [e for e in rec["events"] if e["object"] == obj and e["kind"] == "contact"]
    law = next((l for l in rec["laws"] if l["object"] == obj and l["axis"] == "y"), None)
    if law is not None:
        terms = {t_["term"]: t_["value"] for t_ in law["terms"]}
        equation = law["equation"]
    else:
        # Older records have no stored law: find it now from the recorded flights between contacts.
        from .physics.discover import discover
        cuts = [0] + [int(np.argmin(np.abs(t - c["t"]))) for c in contacts] + [len(t) - 1]
        segs = [(t[a + 1:b], y[a + 1:b]) for a, b in zip(cuts[:-1], cuts[1:]) if b - a > 8]
        if not segs:
            raise Unsupported("no clean stretch of motion to find a law from")
        rate = 1.0 / float(np.median(np.diff(t)))
        found = discover(segs, rate, axis="y", scale_rel=rec.get("scale_rel_err") or 0.0)
        terms = {k: v for k, (v, _) in found.terms.items()}
        equation = found.equation() + " (found now from the record)"
    ratios = [e["ratio"] for e in contacts if e.get("ratio") and 0 < e["ratio"] < 1]
    # Where the motion starts: the first sample that is clearly falling or flying (before that it may be held).
    moving = np.nonzero(np.abs(vy) > 0.1)[0]
    i0 = int(moving[0]) if len(moving) else 0
    # Start velocity from the positions of the first stretch (a free-flight arc), not from smoothed velocities.
    first_contact = next((int(np.argmin(np.abs(t - c["t"]))) for c in contacts if c["t"] > t[i0]), len(t) - 1)
    sel = slice(i0, max(min(first_contact, i0 + 10), i0 + 3))
    a2, b, c = np.polyfit(t[sel] - t[i0], y[sel], 2)
    vx = float(np.polyfit(t[sel], x[sel], 1)[0]) if len(t[sel]) >= 2 else 0.0
    return {
        "terms": terms, "law": equation, "t0": float(t[i0]), "y0": float(c), "vy0": float(b),
        "x0": float(x[i0]), "vx": vx, "t_end": float(t[-1]),
        "floor": float(np.median([e["y_m"] for e in contacts])) if contacts else None,
        "restitution": float(np.median(ratios)) if ratios else None,
        # Each recorded contact's own bounciness, in order (a real ball's changes with impact speed); a replay uses
        # these, a what-if replaces them with one value.
        "per_contact": [e.get("ratio") if e.get("ratio") and 0 < e["ratio"] < 1 else None
                        for e in contacts if e["t"] > t[i0]],
        "centre": float(np.mean(y)),          # a spring's equilibrium (its law is written about the centre)
        "observed": {"t": t, "x": x, "y": y},
    }


def simulate(p: dict, gravity: Optional[float] = None, restitution: Optional[float] = None,
             speed_factor: float = 1.0, drag: Optional[float] = None, horizon: Optional[float] = None,
             fps: float = 240.0) -> dict:
    """Integrate the found law from the start state; a contact with the floor reverses the speed × restitution."""
    terms = dict(p["terms"])
    if gravity is not None:
        if "1" not in terms or "q" in terms:
            raise Unsupported("this law has no constant pull to change")
        terms["1"] = -abs(gravity)
    if drag is not None:
        terms["v"] = -abs(drag)
    floor = p["floor"]
    if floor is not None and p["restitution"] is None and restitution is None:
        raise Unsupported("the record has bounces but no reliable bounciness")
    recorded = [] if restitution is not None else list(p.get("per_contact", []))

    def bounciness(n: int) -> float:
        if restitution is not None:
            return restitution
        return recorded[n] if n < len(recorded) and recorded[n] is not None else p["restitution"]

    def accel(y, v):
        a = terms.get("1", 0.0) + terms.get("v", 0.0) * v + terms.get("|v|v", 0.0) * abs(v) * v
        if "q" in terms:
            a = terms["q"] * (y - p["centre"]) + terms.get("v", 0.0) * v
        return a

    dt = 1.0 / fps
    t_end = p["t0"] + (horizon if horizon else p["t_end"] - p["t0"])
    t, y, v = p["t0"], p["y0"], p["vy0"] * speed_factor
    vx = p["vx"] * speed_factor
    T, Y, X, contacts = [t], [y], [p["x0"]], []
    while t < t_end:
        # Velocity Verlet: accurate for constant and spring-like pulls at this step size.
        a0 = accel(y, v)
        y_new = y + v * dt + 0.5 * a0 * dt * dt
        v_new = v + 0.5 * (a0 + accel(y_new, v + a0 * dt)) * dt
        if floor is not None and y_new < floor and v_new < 0:
            # Time of impact within the step, then bounce.
            frac = (y - floor) / max(y - y_new, 1e-12)
            t_hit = t + frac * dt
            v_hit = v + a0 * frac * dt
            e = bounciness(len(contacts))
            contacts.append({"t": t_hit, "speed_in": -v_hit, "speed_out": -v_hit * e})
            y_new, v_new = floor, -v_hit * e
            if -v_hit * e < 0.05:          # comes to rest
                t = t_hit; T.append(t); Y.append(floor); X.append(X[-1] + vx * (t - T[-2])); break
        t += dt
        y, v = y_new, v_new
        T.append(t); Y.append(y); X.append(X[-1] + vx * dt)
    T, Y, X = np.array(T), np.array(Y), np.array(X)
    peaks = [i for i in range(1, len(Y) - 1) if Y[i] >= Y[i - 1] and Y[i] > Y[i + 1]]
    return {"t": T, "x": X, "y": Y, "contacts": contacts,
            "peaks": [{"t": float(T[i]), "height_m": float(Y[i] - (floor if floor is not None else 0.0))} for i in peaks],
            "params": {"gravity": -terms.get("1", 0.0) if "q" not in terms else None,
                       "restitution": restitution if restitution is not None else p["restitution"],
                       "speed_factor": speed_factor, "drag": -terms.get("v", 0.0) if "v" in terms else 0.0}}


def _observed_peaks(p: dict) -> list[dict]:
    o = p["observed"]
    floor = p["floor"] if p["floor"] is not None else 0.0
    y, t = o["y"], o["t"]
    return [{"t": float(t[i]), "height_m": float(y[i] - floor)} for i in range(1, len(y) - 1)
            if y[i] >= y[i - 1] and y[i] > y[i + 1] and t[i] > p["t0"] + 0.05]


def what_if(rec: dict, obj: str, **changes) -> dict:
    """Replay (no changes) and the what-if, side by side, with a plain summary of what changed."""
    p = _params(rec, obj)
    base = simulate(p)
    changed = {k: v for k, v in changes.items() if v is not None and not (k == "speed_factor" and v == 1.0)}
    alt = simulate(p, **changes) if changed else base
    o = p["observed"]
    # How well the replay explains reality: RMS gap over the recorded stretch.
    m = (o["t"] >= p["t0"]) & (o["t"] <= base["t"][-1])
    rms = float(np.sqrt(np.mean((np.interp(o["t"][m], base["t"], base["y"]) - o["y"][m]) ** 2))) if m.any() else None
    obs_peaks = _observed_peaks(p)

    lines = []
    if rms is not None:
        lines.append(f"Replaying the record ({p['law']}"
                     + (", each bounce's measured bounciness" if p["restitution"] else "")
                     + f") follows the real path to within {rms * 100:.1f} cm (RMS).")
    if changed:
        label = {"gravity": ("gravity", " m/s²", base["params"]["gravity"]), "restitution": ("bounciness", "", p["restitution"]),
                 "speed_factor": ("launch speed ×", "", 1.0), "drag": ("drag", " /s", base["params"]["drag"])}
        lines.append("What if: " + ", ".join(f"{label[k][0]} {v:g}{label[k][1]} instead of {label[k][2]:.2f}{label[k][1]}"
                                             for k, v in changed.items() if k in label) + ".")
        word = "high" if p["floor"] is not None else "at y ="
        for n, (b, a) in enumerate(zip(base["peaks"][:3], alt["peaks"][:3]), start=1):
            lines.append(f"Peak {n}: {word} {a['height_m'] * 100:.0f} cm at {a['t']:.2f} s, instead of "
                         f"{b['height_m'] * 100:.0f} cm at {b['t']:.2f} s.")
        if len(alt["contacts"]) != len(base["contacts"]):
            n = len(alt["contacts"])
            lines.append(f"{n} bounce{'' if n == 1 else 's'} in the same time instead of {len(base['contacts'])}.")

    def pack(sim):
        return {"t": [round(float(v), 4) for v in sim["t"][::2]], "x_m": [round(float(v), 4) for v in sim["x"][::2]],
                "y_m": [round(float(v), 4) for v in sim["y"][::2]], "peaks": sim["peaks"], "contacts": sim["contacts"],
                "params": sim["params"]}

    return {"object": obj, "measured": {"law": p["law"], "restitution": p["restitution"], "gravity": base["params"]["gravity"],
                                        "start_t": p["t0"], "floor_m": p["floor"]},
            "replay": pack(base), "what_if": pack(alt) if changed else None, "rms_m": rms,
            "observed_peaks": obs_peaks, "summary": lines}
