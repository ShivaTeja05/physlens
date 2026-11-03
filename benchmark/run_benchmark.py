"""Memory vs eyes benchmark: plain Muse Spark vs PhysLens on trick clips with known answers.

Every clip is built so that answering from memory ("gravity is 9.81", "things fall down", "bounces lose
energy") gives the wrong answer. Both systems get the same scale information.

  cd backend && .venv/bin/python ../benchmark/run_benchmark.py
Writes benchmark/results.json and benchmark/RESULTS.md. Uses the Meta Model API (SAM 3.1 + Muse Spark).
"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))

from physlens.config import load_env  # noqa: E402

load_env()

import os  # noqa: E402

from openai import OpenAI  # noqa: E402

from physlens import budget  # noqa: E402
from physlens.events import EventLog  # noqa: E402
from physlens.experiments import CARDS  # noqa: E402
from physlens.ledger import Ledger  # noqa: E402
from physlens.perception import default_tracker  # noqa: E402
from physlens.pipeline import run_pipeline  # noqa: E402

CLIPS = HERE / "clips"
REAL_SCALE = "The tape on the wall is exactly 1 metre long, which is 161 pixels in the 720x576 frame."
SYNTH_SCALE = "The ball is 6.5 cm in diameter; the frame is 1280x720."

# name: (file, scale text for the model, PhysLens scale kwargs, truth)
CASES = {
    "real":         (str(HERE.parent / "samples/real-classroom-drop.mp4"), REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81, "playback": "real-time", "upside_down": False}),
    "slow-mo 2×":   ("real-slowmo-2x.mp4", REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81 / 4, "playback": "slow-motion", "upside_down": False}),
    "sped up 2×":   ("real-speedup-2x.mp4", REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81 * 4, "playback": "sped-up", "upside_down": False}),
    "reversed":     ("real-reversed.mp4", REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81, "playback": "reversed", "upside_down": False}),
    "upside down":  ("real-upside-down.mp4", REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81, "playback": "real-time", "upside_down": True}),
    "12 fps":       ("real-12fps.mp4", REAL_SCALE, {"px_per_m": 161.143},
                     {"g": 9.81, "playback": "real-time", "upside_down": False}),
    "Mars gravity": ("mars-drop.mp4", SYNTH_SCALE, {"diameter_m": 0.065},
                     {"g": 3.71, "upside_down": False}),
    "bouncy twin":  ("twin-bouncy.mp4", SYNTH_SCALE, {"diameter_m": 0.065},
                     {"g": 9.81, "e": 0.85, "playback": "real-time", "upside_down": False}),
    "dead twin":    ("twin-dead.mp4", SYNTH_SCALE, {"diameter_m": 0.065},
                     {"g": 9.81, "e": 0.55, "playback": "real-time", "upside_down": False}),
}

QUESTION = ("This video shows a ball dropped and bouncing. {scale} Watch the video and answer from what you see. "
            'Reply with JSON only: {{"gravity_m_s2": number, "speed_kept_per_bounce": number between 0 and 2, '
            '"playback": "real-time"|"slow-motion"|"sped-up"|"reversed", "upside_down": true|false, '
            '"reasoning": short string}}')


def ask_muse(client: OpenAI, video: Path, scale: str) -> dict:
    with video.open("rb") as f:
        fid = client.files.create(file=f, purpose="user_data").id
    t0 = time.time()
    model = os.environ.get("MUSE_MODEL", "muse-spark-1.3")
    budget.check(0.10)
    r = client.responses.create(model=model, input=[
        {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": QUESTION.format(scale=scale)},
            {"type": "input_video", "file_id": fid}]}])
    budget.record_tokens(model, r.usage, what="baseline question")
    text = r.output_text or ""
    m = re.search(r"\{.*\}", text, re.S)
    out = json.loads(m.group()) if m else {}
    out["_seconds"] = round(time.time() - t0, 1)
    return out


def physlens(video: Path, kwargs: dict) -> dict:
    t0 = time.time()
    tracker = default_tracker(CARDS["drop"].color)
    s = run_pipeline(video, CARDS["drop"], EventLog(video.stem), Ledger(), tracker=tracker, **kwargs)
    if s["reversed"]:
        playback = "reversed"
    elif s["slowmo_factor"] > 1.1 and abs(s["g"] - 9.81) > 3 * s["g_err"]:
        playback = "slow-motion"
    elif s["slowmo_factor"] < 0.9 and abs(s["g"] - 9.81) > 3 * s["g_err"]:
        playback = "sped-up"
    else:
        playback = "real-time"
    contacts = s.get("contacts") or []
    return {"gravity_m_s2": s["g"], "g_err": s["g_err"], "speed_kept_per_bounce": contacts[-1][1] if contacts else None,
            "playback": playback, "upside_down": s["upside_down"], "verdicts": s["verdicts"],
            "tracker": getattr(tracker, "name", "?"), "explanation": s["explanation"],
            "_seconds": round(time.time() - t0, 1)}


def score(answer: dict, truth: dict) -> dict:
    out = {}
    g = answer.get("gravity_m_s2")
    if isinstance(g, (int, float)) and g > 0:
        out["g_rel_err"] = abs(g - truth["g"]) / truth["g"]
        out["g_ok"] = out["g_rel_err"] <= 0.10
    else:
        out["g_ok"] = False
    if "e" in truth:
        e = answer.get("speed_kept_per_bounce")
        out["e_ok"] = isinstance(e, (int, float)) and abs(e - truth["e"]) <= 0.05
    if "playback" in truth:
        out["playback_ok"] = answer.get("playback") == truth["playback"]
    out["upside_ok"] = bool(answer.get("upside_down")) == truth["upside_down"]
    return out


def main() -> None:
    client = OpenAI(api_key=os.environ["META_API_KEY"], base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"))
    only = set(sys.argv[1:])   # e.g. `run_benchmark.py real` re-runs one clip and keeps the others' results
    saved = HERE / "results.json"
    results = json.loads(saved.read_text()) if only and saved.exists() else {}
    for name, (file, scale_text, kwargs, truth) in CASES.items():
        if only and name not in only:
            continue
        video = (CLIPS / file).resolve()
        row = {"truth": truth}
        for system, fn in [("muse_spark", lambda: ask_muse(client, video, scale_text)),
                           ("physlens", lambda: physlens(video, kwargs))]:
            try:
                row[system] = fn()
                row[system + "_score"] = score(row[system], truth)
            except Exception as exc:  # record failures instead of stopping the run
                row[system] = {"error": str(exc)[:300]}
                row[system + "_score"] = {"g_ok": False}
        results[name] = row
        print(f"{name:14} muse g={row['muse_spark'].get('gravity_m_s2')}  physlens g={row['physlens'].get('gravity_m_s2')}",
              flush=True)
    results["_budget"] = budget.summary()
    (HERE / "results.json").write_text(json.dumps(results, indent=2, default=float))
    write_markdown({k: v for k, v in results.items() if not k.startswith("_")})
    print("spend:", results["_budget"])


def _fmt_g(a: dict) -> str:
    g = a.get("gravity_m_s2")
    if not isinstance(g, (int, float)):
        return "error" if "error" in a else "—"
    return f"{g:.2f} ± {a['g_err']:.2f}" if "g_err" in a else f"{g:.2f}"


def write_markdown(results: dict) -> None:
    mark = lambda ok: "✓" if ok else "✗"
    lines = ["# Memory vs eyes benchmark", "",
             "Plain Muse Spark (`muse-spark-1.3`, video input) vs PhysLens (SAM 3.1 tracking + physics fits + "
             "Muse Spark explanation) on clips built so that answering from memory is wrong. Same scale info for both. "
             "Gravity counts as correct within 10%.", "",
             "| Clip | Truth | Muse Spark: gravity | PhysLens: gravity | Muse Spark: trick spotted | PhysLens: trick spotted |",
             "|---|---|---|---|---|---|"]
    totals = {"muse_spark": [0, 0], "physlens": [0, 0]}
    for name, row in results.items():
        t = row["truth"]
        truth = f"g = {t['g']:.2f}" + (f", e = {t['e']}" if "e" in t else "") + \
            (f", {t['playback']}" if t.get("playback") not in (None, "real-time") else "") + (", upside down" if t["upside_down"] else "")
        cells = []
        for system in ("muse_spark", "physlens"):
            sc = row[system + "_score"]
            checks = [v for k, v in sc.items() if k.endswith("_ok")]
            totals[system][0] += sum(checks)
            totals[system][1] += len(checks)
            cells.append(sc)
        def trick(system):
            a, sc = row[system], row[system + "_score"]
            if "error" in a:
                return "error"
            bits = []
            if "playback" in t:
                bits.append(f"{a.get('playback')} {mark(sc.get('playback_ok'))}")
            if "e" in t:
                e = a.get("speed_kept_per_bounce")
                bits.append(f"e = {e:.2f} {mark(sc.get('e_ok'))}" if isinstance(e, (int, float)) else "e = ? ✗")
            if t["upside_down"] or a.get("upside_down"):
                bits.append(f"upside down {mark(sc.get('upside_ok'))}")
            return ", ".join(bits) or "—"
        lines.append(f"| {name} | {truth} | {_fmt_g(row['muse_spark'])} {mark(row['muse_spark_score'].get('g_ok'))} | "
                     f"{_fmt_g(row['physlens'])} {mark(row['physlens_score'].get('g_ok'))} | {trick('muse_spark')} | {trick('physlens')} |")
    lines += ["", f"**Checks passed:** Muse Spark {totals['muse_spark'][0]}/{totals['muse_spark'][1]} · "
                  f"PhysLens {totals['physlens'][0]}/{totals['physlens'][1]}", ""]
    (HERE / "RESULTS.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
