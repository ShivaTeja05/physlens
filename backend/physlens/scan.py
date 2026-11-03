"""Scan mode: point the camera at anything and PhysLens says what it is, finds its parts, measures what the pixels can
honestly give, and explains how it works.

    look     Muse Spark (vision) names the scene and the parts worth outlining, how they relate (load, light, power,
             motion...), and how the thing works. That last part is background knowledge and is labelled as such.
    find     SAM 3.1 outlines every instance of each part.
    measure  Deterministic: counts, rows and columns of repeated parts (windows → floors), share of the frame, a round
             body's lit fraction (the Moon's phase), and metres when the user gives one known size.
    explain  Muse Spark writes what was SEEN using only the measured numbers; every sentence goes through the
             evidence-or-reject verifier. Unsupported sentences are dropped, never shown as fact.

Memory and eyes stay apart on screen: "measured" claims carry numbers from this image; "how it works" is what a
textbook would say, tagged as knowledge.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .agent import verifier

SCENE_TYPES = ("structure", "astronomy", "motion", "activity", "device", "nature", "other")
EXPERIMENTS = ("none", "drop", "cup", "cans", "weigh", "free_motion")
RELATION_KINDS = ("load", "force", "energy", "light", "motion", "signal", "flow")
MAX_PARTS = 6
MAX_INSTANCES = 80
PALETTE = ["#3ee6ff", "#ffc857", "#ff6b7d", "#4ade80", "#b18cff", "#5aa9ff"]

LOOK_PROMPT = """You are the eyes of PhysLens, a camera that explains how the physical world works.
Look at the image and answer with ONE JSON object, no prose around it:
{
 "scene_type": one of %(types)s,
 "title": "2-5 words naming what this is",
 "summary": "one sentence: what is in front of the camera",
 "parts": [ up to %(max_parts)d things worth outlining, most important first:
   {"id": "p1", "name": "a short plain noun SAM can find in the picture, e.g. 'window', 'column', 'moon', 'bicycle chain'",
    "count": "one" or "many", "role": "what this part does, under 12 words"} ],
 "relations": [ how parts act on each other: {"from": "p1", "to": "p2", "kind": one of %(kinds)s,
   "label": "under 6 words, e.g. 'carries weight to'"} ],
 "how_it_works": [ 3-5 short steps explaining the physics or engineering of what you see, each
   {"text": "...", "parts": ["p1"]} ],
 "measure_ideas": [ up to 3 things a camera could measure here to check the explanation ],
 "moving_object": "the one object whose motion the physics is about, as a short noun SAM can find (e.g. 'orange
   ball'), or empty if nothing is moving",
 "video_experiment": one of %(experiments)s (which PhysLens video experiment would suit this, if any),
 "video_hint": "one sentence: what to film to measure the physics here, or empty"
}
Rules: only name parts that are visible. In how_it_works do NOT state sizes, counts or distances of THIS scene
(those are measured separately); general physics facts are fine."""

EXPLAIN_PROMPT = """You are PhysLens. You receive what is in a picture and the numbers PhysLens MEASURED from it.
Reply with ONE JSON object:
{"headline": "one keynote-style line tying what was measured to how it works",
 "seen": ["2-5 short sentences, each stating one measured fact and what it means"]}
Rules: every number you write must appear in "measurements" (rounding is fine). Write no other numbers at all, not
even general facts. Say "about" for estimates. Use plain words."""


def _client():
    key = os.environ.get("META_API_KEY")
    model = os.environ.get("MUSE_MODEL", "muse-spark-1.3")
    if not key:
        return None, model
    from openai import OpenAI
    return OpenAI(api_key=key, base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1")), model


def _jpeg_url(img: np.ndarray, width: int = 768) -> str:
    import base64
    if img.shape[1] > width:
        img = cv2.resize(img, (width, round(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
    _, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode()


def _json_reply(client, model: str, messages: list, what: str, effort: str = "low") -> dict:
    from . import budget
    budget.check(0.03)
    r = client.chat.completions.create(model=model, messages=messages, max_tokens=6000,
                                       reasoning_effort=os.environ.get("MUSE_EFFORT", effort))
    budget.record_tokens(model, r.usage, what=what)
    raw = r.choices[0].message.content or ""
    return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])


def look(img: np.ndarray, frames: int = 1) -> dict:
    client, model = _client()
    if client is None:
        raise RuntimeError("Scan needs Muse Spark: set META_API_KEY in backend/.env")
    note = (f"\nThe image shows {frames} moments of ONE video side by side, left to right in time order. Use the "
            "differences between them to see what moves.\n") if frames > 1 else ""
    prompt = note + LOOK_PROMPT % {"types": "|".join(SCENE_TYPES), "max_parts": MAX_PARTS, "kinds": "|".join(RELATION_KINDS),
                            "experiments": "|".join(EXPERIMENTS)}
    out = _json_reply(client, model, [{"role": "user", "content": [
        {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": _jpeg_url(img)}}]}], "scan: look")
    # Keep the reply to the shape the rest relies on.
    parts = []
    for i, p in enumerate(out.get("parts", [])[:MAX_PARTS]):
        name = str(p.get("name", "")).strip()
        if name:
            parts.append({"id": str(p.get("id") or f"p{i + 1}"), "name": name, "role": str(p.get("role", "")),
                          "many": str(p.get("count", "one")).lower() == "many"})
    ids = {p["id"] for p in parts}
    out["parts"] = parts
    out["relations"] = [r for r in out.get("relations", []) if r.get("from") in ids and r.get("to") in ids][:8]
    out["scene_type"] = out.get("scene_type") if out.get("scene_type") in SCENE_TYPES else "other"
    out["video_experiment"] = out.get("video_experiment") if out.get("video_experiment") in EXPERIMENTS else "none"
    return out


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")[:24] or "part"


def _rows(values: np.ndarray, sizes: np.ndarray) -> int:
    """How many rows (or columns) repeated parts sit in: centres closer than ~0.6 of a part's size share a row."""
    if len(values) < 2:
        return len(values)
    v = np.sort(values)
    gap = 0.6 * float(np.median(sizes))
    return int(1 + np.sum(np.diff(v) > gap))


SURFACES = {"wall", "floor", "ground", "sky", "ceiling", "road", "water", "grass", "table", "street", "field", "sea",
            "background", "room", "pavement", "sidewalk", "carpet"}


def _tilt_deg(mask: np.ndarray) -> Optional[tuple[float, list[float]]]:
    """Lean of an elongated shape's long axis from vertical (or horizontal for wide shapes), in degrees, and the axis
    direction for drawing; None if the shape isn't clearly elongated."""
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    P = np.stack([xs - xs.mean(), ys - ys.mean()])
    evals, evecs = np.linalg.eigh(np.cov(P))
    if evals[1] < 6.0 * max(evals[0], 1e-9):      # clearly long and thin (long axis ≥ 2.4 × the short one)
        return None
    vx, vy = evecs[:, 1]
    ang = math.degrees(math.atan2(abs(vx), abs(vy)))   # 0 = vertical
    return (ang if ang <= 45 else 90 - ang), [round(float(vx), 4), round(float(vy), 4)]


def lit_fraction(gray: np.ndarray, mask: np.ndarray, seed: int = 0
                 ) -> Optional[tuple[float, tuple[float, float, float], tuple[float, float]]]:
    """Lit share (%) of a round body's disc, the disc (cx, cy, r), and a (low, high) range for the share.

    The sunlit edge of a Moon or planet is always an arc of its full circle, whatever the phase; the shadow line
    (terminator) is a half-ellipse. So the disc is found by a robust (RANSAC) circle fit to the lit region's outline,
    which locks onto the circular edge and ignores the terminator. Then every pixel in the disc brighter than the night
    side counts as lit: near the terminator the Sun is low and the ground is dim but still sunlit, and dark lava plains
    are sunlit too. SAM's outline only says where to look."""
    H, W = gray.shape
    ys, xs = np.nonzero(mask)
    if len(xs) < 50:
        return None
    pad = int(0.25 * max(np.ptp(xs), np.ptp(ys)))
    x0, y0 = max(int(xs.min()) - pad, 0), max(int(ys.min()) - pad, 0)
    x1, y1 = min(int(xs.max()) + pad + 1, W), min(int(ys.max()) + pad + 1, H)
    roi = gray[y0:y1, x0:x1].astype(np.float32)
    peak = float(np.percentile(gray[mask], 99))
    bright = (roi > 0.15 * peak).astype(np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bright)
    if n < 2:
        return None
    lit = labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cnts, _ = cv2.findContours(lit.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = max(cnts, key=len)[:, 0, :].astype(float)
    # Outline points on the ROI's border are where the picture cuts the disc off, not its edge.
    on_edge = (pts[:, 0] <= 0) | (pts[:, 1] <= 0) | (pts[:, 0] >= roi.shape[1] - 1) | (pts[:, 1] >= roi.shape[0] - 1)
    pts = pts[~on_edge]
    if len(pts) < 20:
        return None
    if len(pts) > 1500:
        pts = pts[np.linspace(0, len(pts) - 1, 1500).astype(int)]
    extent = max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
    rng = np.random.default_rng(seed)
    best, best_n = None, 0
    for _ in range(600):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
        if abs(d) < 1e-9:
            continue
        ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
        uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
        r = float(np.hypot(a[0] - ux, a[1] - uy))
        if not 0.45 * extent <= r <= 1.2 * extent:
            continue
        inl = np.abs(np.hypot(pts[:, 0] - ux, pts[:, 1] - uy) - r) < max(1.5, 0.01 * r)
        if inl.sum() > best_n:
            best, best_n = inl, int(inl.sum())
    if best is None or best_n < 15:
        return None
    # Least-squares circle through the edge points (Kåsa fit).
    P = pts[best]
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
    cx, cy = sol[0], sol[1]
    r = float(np.sqrt(sol[2] + cx ** 2 + cy ** 2))
    # Count lit pixels over the whole disc (the picture's edge may cut it; only the visible part is scored).
    yy, xx = np.mgrid[0:H, 0:W]
    disc = (xx - (cx + x0)) ** 2 + (yy - (cy + y0)) ** 2 <= r ** 2
    v = gray[disc].astype(np.float32)
    # Just above the night side, which may glow faintly with Earthshine and has its own noise: its level and spread
    # come from the dark pixels (median and MAD). If no night side is visible (nearly full), 5% of the peak.
    dark = v[v < 0.25 * peak]
    if len(dark) > 0.02 * len(v):
        level = float(np.median(dark))
        spread = 1.4826 * float(np.median(np.abs(dark - level)))
        thrs = [level + k * spread + f * peak for k, f in ((4, 0.03), (2, 0.015), (6, 0.06))]
    else:
        thrs = [0.05 * peak, 0.025 * peak, 0.1 * peak]
    # Near the terminator the ground fades smoothly from sunlit to night, so where "lit" ends is a judgement call:
    # the range covers a stricter and a looser threshold.
    share = [min(100 * float((v > t).sum()) / max(float(math.pi * r ** 2), 1.0), 100.0) for t in thrs]
    return share[0], (float(cx + x0), float(cy + y0), r), (min(share), max(share))


def find_and_measure(img: np.ndarray, parts: list[dict], scene_type: str,
                     reference: Optional[dict] = None, relations: Optional[list] = None) -> tuple[list[dict], dict]:
    from .perception.sam_image import outline, segment
    H, W = img.shape[:2]

    def run(p):
        try:
            return segment(img, p["name"], what="scan")
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=MAX_PARTS) as pool:
        found = list(pool.map(run, parts))

    out, evidence, masks = [], {}, {}
    for k, (p, inst) in enumerate(zip(parts, found)):
        inst = [i for i in inst if i.mask.sum() >= 0.00002 * H * W]
        if not p["many"]:
            inst = sorted(inst, key=lambda i: -i.mask.sum())[:1]
        inst = inst[:MAX_INSTANCES]
        union = np.zeros((H, W), bool)
        for i in inst:
            union |= i.mask
        # Two names for the same thing ("motorcycle", "scooter") would outline it twice: keep the first.
        if union.any() and any((union & m).sum() / max((union | m).sum(), 1) > 0.6 for m in masks.values()):
            continue
        items = []
        for i in inst:
            ys, xs = np.nonzero(i.mask)
            items.append({"box": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                          "center": [round(float(xs.mean()), 1), round(float(ys.mean()), 1)],
                          "area": int(i.mask.sum()), "outline": outline(i.mask)})
        part = {**p, "color": PALETTE[len(out) % len(PALETTE)], "instances": items, "measures": []}
        out.append(part)
        if not items:
            continue
        masks[p["id"]] = union
        s = _slug(p["name"])
        boxes = np.array([it["box"] for it in items], float)
        x0, y0, x1, y1 = boxes[:, 0].min(), boxes[:, 1].min(), boxes[:, 2].max(), boxes[:, 3].max()
        part["union_box"] = [int(x0), int(y0), int(x1), int(y1)]
        if p["many"]:
            evidence[f"{s}_count"] = len(items)
            part["measures"].append({"label": "count", "value": len(items), "unit": ""})
            if len(items) >= 3 and scene_type == "structure":   # rows mean floors; for players or craters, nothing
                cy = np.array([it["center"][1] for it in items]); hh = boxes[:, 3] - boxes[:, 1] + 1
                cx = np.array([it["center"][0] for it in items]); ww = boxes[:, 2] - boxes[:, 0] + 1
                rows, cols = _rows(cy, hh), _rows(cx, ww)
                evidence[f"{s}_rows"], evidence[f"{s}_columns"] = rows, cols
                part["measures"] += [{"label": "rows", "value": rows, "unit": ""},
                                     {"label": "columns", "value": cols, "unit": ""}]
                part["row_lines"] = _row_lines(cy, hh)
        else:
            # A lean means something for objects, not for surfaces or regions that run off the picture's edges.
            edges = (x0 <= 2) + (y0 <= 2) + (x1 >= W - 3) + (y1 >= H - 3)
            tilt = None if (set(_slug(p["name"]).split("_")) & SURFACES or edges >= 2) else _tilt_deg(union)
            if tilt is not None:
                evidence[f"{s}_tilt_deg"] = round(tilt[0], 1)
                part["axis"] = tilt[1]
                part["measures"].append({"label": "tilt from upright", "value": round(tilt[0], 1), "unit": "°"})
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            if scene_type == "structure" and bh > bw:
                evidence[f"{s}_height_to_width"] = round(bh / bw, 2)
                part["measures"].append({"label": "height ÷ width", "value": round(bh / bw, 2), "unit": "×"})
            if scene_type == "astronomy":
                got = lit_fraction(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), union)
                lit = got[0] if got else 0.0
                if got:
                    part["disc"] = [round(v, 1) for v in got[1]]
                    evidence[f"{s}_lit_pct_low"], evidence[f"{s}_lit_pct_high"] = (round(x, 1) for x in got[2])
                if 3 < lit <= 105:
                    evidence[f"{s}_lit_pct"] = round(min(lit, 100.0), 1)
                    part["measures"].append({"label": "disc lit", "value": round(min(lit, 100.0), 1), "unit": "%",
                                             "note": f"{got[2][0]:.0f}–{got[2][1]:.0f}% depending on where the shadow line is drawn"})

    # Sub-parts: how high a part sits on the thing it belongs to (a seat on a bike, a window band on a tower).
    singles = [q for q in out if q.get("union_box") and not q["many"] and not set(_slug(q["name"]).split("_")) & SURFACES]
    if singles:
        big = max(singles, key=lambda q: (q["union_box"][3] - q["union_box"][1]) * (q["union_box"][2] - q["union_box"][0]))
        bx0, by0, bx1, by1 = big["union_box"]
        for q in out:
            if q is big or not q.get("union_box"):
                continue
            x0, y0, x1, y1 = q["union_box"]
            inside = x0 >= bx0 - 5 and x1 <= bx1 + 5 and y0 >= by0 - 5 and y1 <= by1 + 5
            if inside and by1 > by0:
                frac = 100 * (by1 - (y0 + y1) / 2) / (by1 - by0)
                evidence[f"{_slug(q['name'])}_height_up_{_slug(big['name'])}_pct"] = round(frac, 0)
                q["measures"].append({"label": f"height up the {big['name']}", "value": round(frac), "unit": "%"})

    # Contact: do the parts a relation links actually touch? (Eyes checking what the model claimed.)
    kernel = np.ones((7, 7), np.uint8)
    for r in relations or []:
        a, b = masks.get(r["from"]), masks.get(r["to"])
        if a is not None and b is not None:
            r["contact"] = bool((cv2.dilate(a.astype(np.uint8), kernel) > 0)[b].any())

    # One known size turns pixels into metres for everything at about the same distance.
    if reference and reference.get("size_m"):
        ref = next((q for q in out if q["id"] == reference.get("part") or q["name"] == reference.get("part")), None)
        if ref and ref.get("union_box"):
            axis = 3 if reference.get("axis", "height") == "height" else 2
            b = ref["instances"][0]["box"]
            px_per_m = (b[axis] - b[axis - 2] + 1) / float(reference["size_m"])
            evidence["reference_size_m"] = float(reference["size_m"])
            for q in out:
                if q.get("union_box"):
                    x0, y0, x1, y1 = q["union_box"]
                    s = _slug(q["name"])
                    h_m, w_m = (y1 - y0 + 1) / px_per_m, (x1 - x0 + 1) / px_per_m
                    evidence[f"{s}_height_m"], evidence[f"{s}_width_m"] = round(h_m, 2), round(w_m, 2)
                    q["measures"] += [{"label": "height", "value": round(h_m, 2), "unit": "m",
                                       "note": f"scaled from the {ref['name']} ({reference['size_m']} m)"},
                                      {"label": "width", "value": round(w_m, 2), "unit": "m"}]
    return out, evidence


def _row_lines(cy: np.ndarray, hh: np.ndarray) -> list[float]:
    """Centre height of each row of repeated parts (for drawing floor lines)."""
    order = np.argsort(cy)
    v, gap = cy[order], 0.6 * float(np.median(hh))
    rows, cur = [], [v[0]]
    for a, b in zip(v[:-1], v[1:]):
        if b - a > gap:
            rows.append(float(np.mean(cur))); cur = []
        cur.append(b)
    rows.append(float(np.mean(cur)))
    return [round(r, 1) for r in rows]


def _name(parts: list[dict], pid: str) -> str:
    return next((p["name"] for p in parts if p["id"] == pid), pid)


def template_seen(parts: list[dict], evidence: dict) -> list[str]:
    out = []
    for p in parts:
        s = _slug(p["name"])
        if f"{s}_rows" in evidence:
            out.append(f"Counted {evidence[f'{s}_count']} {p['name']}s in {evidence[f'{s}_rows']} rows.")
        elif f"{s}_count" in evidence:
            out.append(f"Counted {evidence[f'{s}_count']} {p['name']}s.")
        if f"{s}_lit_pct" in evidence:
            out.append(f"About {evidence[f'{s}_lit_pct']:.0f}% of the {p['name']}'s disc is lit.")
        if f"{s}_tilt_deg" in evidence:
            out.append(f"The {p['name']} leans about {evidence[f'{s}_tilt_deg']:.0f}° from upright.")
        if f"{s}_height_m" in evidence:
            out.append(f"The {p['name']} is about {evidence[f'{s}_height_m']:.1f} m tall.")
    return out[:5]


def explain(seen_by: dict, parts: list[dict], evidence: dict) -> tuple[str, list[dict], str]:
    """Returns (headline, seen sentences with their verification, source)."""
    fallback = template_seen(parts, evidence)
    client, model = _client()
    if client is None or not evidence:
        return seen_by.get("summary", ""), [{"text": t, "verified": verifier.check(t, evidence)[0]} for t in fallback], "template"
    payload = {"title": seen_by.get("title"), "scene_type": seen_by.get("scene_type"), "summary": seen_by.get("summary"),
               "parts": [{"name": p["name"], "role": p["role"]} for p in parts], "measurements": evidence,
               "contacts_seen": [f"{_name(parts, r['from'])} {'touches' if r['contact'] else 'does not touch'} "
                                 f"{_name(parts, r['to'])}" for r in seen_by.get("relations", []) if "contact" in r]}
    try:
        out = _json_reply(client, model, [{"role": "system", "content": EXPLAIN_PROMPT},
                                          {"role": "user", "content": json.dumps(payload)}], "scan: explain")
    except Exception:
        return seen_by.get("summary", ""), [{"text": t, "verified": verifier.check(t, evidence)[0]} for t in fallback], "template"
    headline = str(out.get("headline", ""))
    if not verifier.check(headline, evidence)[0]:
        headline = seen_by.get("summary", "")
    seen = [{"text": str(t), "verified": True} for t in out.get("seen", []) if verifier.check(str(t), evidence)[0]]
    if not seen:
        return headline, [{"text": t, "verified": verifier.check(t, evidence)[0]} for t in fallback], "template"
    return headline, seen[:5], "muse_spark"


PHASES = [("new", 0, 3), ("crescent", 3, 45), ("quarter", 45, 55), ("gibbous", 55, 97), ("full", 97, 101)]
PHASE_WORDS = {"new": "new", "crescent": "crescent", "quarter": "quarter", "half": "quarter", "gibbous": "gibbous",
               "full": "full"}


def word_checks(seen_by: dict, parts: list[dict], evidence: dict) -> list[dict]:
    """Where the model's words make a claim the measurements can test, test it. So far: a Moon's phase name against
    its measured lit share (crescent < 45% < quarter < 55% < gibbous)."""
    out = []
    words = f"{seen_by.get('title', '')} {seen_by.get('summary', '')}".lower()
    named = next((PHASE_WORDS[w] for w in PHASE_WORDS if re.search(rf"\b{w}\b", words)), None)
    for p in parts:
        s = _slug(p["name"])
        if f"{s}_lit_pct" not in evidence or not named:
            continue
        lo, hi = evidence[f"{s}_lit_pct_low"], evidence[f"{s}_lit_pct_high"]
        fits = [n for n, a, b in PHASES if lo < b and hi >= a]
        ok = named in fits
        out.append({"ok": ok, "claim": f"Muse Spark called it {named}",
                    "text": (f"Measured {evidence[f'{s}_lit_pct']:.0f}% lit ({lo:.0f}–{hi:.0f}%): "
                             + (f"consistent with {named}." if ok else f"that's {' or '.join(fits)}, not {named}."))})
    return out


def scan(img: np.ndarray, reference: Optional[dict] = None, cache_dir: Optional[Path] = None,
         progress=None) -> dict:
    """The whole scan for one image. `progress(stage, partial_result)` is called after each stage. Results are
    cached by image (and reference) so a demo never pays twice."""
    say = progress or (lambda stage, partial: None)
    key = hashlib.sha256(cv2.imencode(".png", img)[1].tobytes() + json.dumps(reference or {}, sort_keys=True).encode()).hexdigest()[:16]
    if cache_dir:
        f = cache_dir / f"scan-{key}.json"
        if f.exists():
            return json.loads(f.read_text())
    H, W = img.shape[:2]
    result = {"id": key, "image": {"width": W, "height": H}}
    say("looking", result)
    seen_by = look(img)
    result.update(scene_type=seen_by["scene_type"], title=seen_by.get("title", ""), summary=seen_by.get("summary", ""),
                  parts=[{**p, "color": PALETTE[i % len(PALETTE)], "instances": [], "measures": []}
                         for i, p in enumerate(seen_by["parts"])],
                  relations=seen_by["relations"], how_it_works=seen_by.get("how_it_works", [])[:5],
                  measure_ideas=seen_by.get("measure_ideas", [])[:3], video_experiment=seen_by["video_experiment"],
                  video_hint=seen_by.get("video_hint", ""))
    say("outlining", result)
    parts, evidence = find_and_measure(img, seen_by["parts"], seen_by["scene_type"], reference, seen_by["relations"])
    kept = {p["id"] for p in parts}
    result.update(parts=parts, evidence=evidence,
                  relations=[r for r in seen_by["relations"] if r["from"] in kept and r["to"] in kept])
    say("explaining", result)
    headline, seen, source = explain(seen_by, parts, evidence)
    result.update(headline=headline, seen=seen, explain_source=source, checks=word_checks(seen_by, parts, evidence))
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / f"scan-{key}.json").write_text(json.dumps(result))
    say("done", result)
    return result
