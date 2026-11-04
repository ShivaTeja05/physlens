"""Atomic scenes: any recorded event as data a physics engine can replay and re-simulate.

    world    gravity vector, ground plane, fixed obstacles (a crash barrier)
    objects  shape (sphere | box), size, mass, bounciness and friction, each with where the value came from
    atoms    one object at one instant: t, position (x, y, z), velocity, orientation, ± position, and which of
             those were measured (from the video) and which assumed (a single camera sees no depth: z = 0)
    moments  contacts, peaks, impacts... from the recorder

Axes: metres, x along the motion in the picture, y up, z towards the camera. Built from a record
(`recorder.export`) plus, for crashes, the run's reconstruction event (front and rear edges, so the car's length
over time, i.e. its crush, is kept). Plain JSON: `GET /api/records/{id}/scene`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

FORMAT = "physlens-scene/1"


def _src(value, source: str) -> dict:
    return {"value": value, "source": source}


def from_record(rec: dict, events: Optional[list[dict]] = None) -> dict:
    """The record of one run as an atomic scene."""
    events = events or []
    recon = next((e for e in events if e.get("type") == "reconstruction"), None)
    g = 9.81
    g_src = "assumed (Earth)"
    for law in rec.get("laws", []):
        terms = {t["term"]: t["value"] for t in law["terms"]}
        if law["axis"] == "y" and "1" in terms and "q" not in terms:
            g, g_src = -terms["1"], f"measured (law found: {law['equation']})"
    world = {"gravity": [0.0, -abs(g), 0.0], "gravity_source": g_src, "ground_y": 0.0, "obstacles": []}
    objects, atoms = [], {}

    if recon is not None:
        # Crash: x = distance from the barrier face (car on the + side), the car's centre and length per frame.
        L, h = recon["length_m"], recon["height_m"]
        world["obstacles"].append({"id": "barrier", "shape": "box", "size": [0.6, max(1.2 * h, 1.5), 3.0],
                                   "position": [-0.3, max(1.2 * h, 1.5) / 2, 0.0], "source": "fixed (crash barrier face at x = 0)"})
        width = 0.39 * L
        objects.append({"id": recon["object"], "label": recon["label"], "shape": "box",
                        "size": _src([L, h, width], "length given; height measured; width assumed (typical car)"),
                        "mass_kg": _src(recon.get("mass_kg"), "given (test data)" if recon.get("mass_kg") else "unknown"),
                        "restitution": _src(round(recon["rebound_ms"] / recon["speed_ms"], 3) if recon.get("rebound_ms") else 0.1,
                                            "measured (rebound ÷ impact speed)" if recon.get("rebound_ms") else "assumed"),
                        "friction": _src(0.8, "assumed (tyres on concrete)")})
        rows, prev = [], None
        for t, front, rear in recon["frames"]:
            if front is None and rear is None:
                continue
            f = front if front is not None else rear - L
            r = rear if rear is not None else front + L
            x, length = (f + r) / 2, r - f
            rows.append([round(t, 4), round(x, 4), round(h / 2, 4), 0.0, round(length, 4)])
        out = []
        for k, (t, x, y, z, length) in enumerate(rows):
            nb = rows[max(k - 1, 0)], rows[min(k + 1, len(rows) - 1)]
            vx = (nb[1][1] - nb[0][1]) / (nb[1][0] - nb[0][0]) if nb[1][0] > nb[0][0] else 0.0
            out.append({"t": t, "p": [x, y, z], "v": [round(vx, 3), 0.0, 0.0], "q": [0, 0, 0, 1], "length": length,
                        "err": 0.01, "measured": ["x", "vx", "length"], "assumed": ["y", "z", "q"]})
        atoms[recon["object"]] = out
        # After impact the car doesn't roll freely: its wheels and crushed front drag. The effective friction comes
        # from the atoms: rebound speed² ÷ (2 g × the distance it rebounds before stopping).
        if recon.get("rebound_ms") and len(out) > 3:
            x_min = min(a["p"][0] for a in out)
            d = out[-1]["p"][0] - x_min
            if d > 0.05:
                mu = recon["rebound_ms"] ** 2 / (2 * abs(world["gravity"][1]) * d)
                objects[-1]["post_impact_friction"] = _src(round(min(mu, 1.5), 3),
                                                           f"measured (rebound {recon['rebound_ms']:.2f} m/s, stops after {d:.2f} m)")
    else:
        floor = None
        for o in rec.get("objects", []):
            states = rec["states"].get(o["id"], [])
            contacts = [e for e in rec["events"] if e["object"] == o["id"] and e["kind"] == "contact"]
            ratios = sorted(e["ratio"] for e in contacts if e.get("ratio") and 0 < e["ratio"] < 1)
            floor = min(e["y_m"] for e in contacts if e.get("y_m") is not None) if contacts else None
            r = o.get("radius_m") or 0.03
            objects.append({"id": o["id"], "label": o["label"], "shape": "sphere",
                            "size": _src([r], "measured (outline area)"),
                            "mass_kg": _src(None, "unknown (not needed: a drop's motion doesn't depend on mass)"),
                            "restitution": _src(round(ratios[len(ratios) // 2], 3) if ratios else 0.7,
                                                f"measured (median of {len(ratios)} contacts)" if ratios else "assumed"),
                            "friction": _src(0.5, "assumed")})
            base = (floor - r) if floor is not None else 0.0      # put the ground at y = 0
            atoms[o["id"]] = [{"t": s["t"], "p": [round(s["x_m"] - states[0]["x_m"], 4), round(s["y_m"] - base, 4), 0.0],
                               "v": [s["vx"], s["vy"], 0.0], "q": [0, 0, 0, 1], "err": s.get("pos_err_m"),
                               "measured": ["x", "y", "vx", "vy"], "assumed": ["z", "vz", "q"]} for s in states]
    moments = [{k: e[k] for k in ("object", "t", "kind", "speed_in", "speed_out", "ratio") if e.get(k) is not None}
               for e in rec.get("events", [])]
    return {"format": FORMAT, "capture": rec["id"], "experiment": rec.get("experiment"),
            "source_video_sha256": rec.get("video_sha256"), "fps": rec.get("fps"),
            "world": world, "objects": objects, "atoms": atoms, "moments": moments,
            "notes": "One camera sees no depth: z and orientation are assumed, and labelled so in every atom."}


def build(capture_id: str, replays: Path) -> dict:
    from . import recorder
    rec = recorder.export(capture_id)
    f = replays / f"{Path(capture_id).name}.jsonl"
    events = [json.loads(l) for l in f.read_text().splitlines() if l.strip()] if f.exists() else []
    return from_record(rec, events)
