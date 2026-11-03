"""Ask PhysLens: questions about a scan or a lab run, answered from what was measured, with memory.

Each answer is grounded in one context: a scan (parts, measurements, checks, how it works) or a run (gravity, laws,
predictions and their verdicts...). The conversation so far goes with every question, so a follow-up ("and the
other one?") stays consistent with what was already said. Muse Spark answers only what was asked, in one to three
spoken sentences. Every number it says must be one PhysLens measured; if not, it gets one chance to fix that, and
otherwise the answer is shown as unverified.
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .agent import verifier

MAX_TURNS = 8   # conversation remembered per context

SYSTEM = """You are PhysLens, a camera that measures the physical world, speaking with the person using it.
You get FACTS (what PhysLens saw and measured in this scene or video) and the conversation so far.
Reply with ONE JSON object: {"answer": "...", "parts": [ids of parts the answer is about, if any]}.
Rules for "answer":
- Answer only what was asked, in 1-3 short spoken sentences. No lists, no markdown, no preamble.
- Every number you say must appear in FACTS (rounding is fine). If what they ask wasn't measured, say so plainly, then
  you may explain from general physics in words, starting with "From general physics," and without numbers.
- Stay consistent with what you said earlier in the conversation; if FACTS show you were wrong, say so.
- Refer to things by their names in FACTS. Say "about" for estimates and give the range when FACTS have one."""

CONVERSATIONS: dict[str, list[dict]] = {}


def _numbers(obj, out: dict, prefix: str = "") -> None:
    """Every number in the facts, flattened, so the verifier can check answers against them."""
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out[prefix or "value"] = float(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _numbers(v, out, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _numbers(v, out, f"{prefix}[{i}]")


def scan_facts(result: dict) -> dict:
    return {
        "kind": "still image scan", "title": result.get("title"), "scene_type": result.get("scene_type"),
        "summary": result.get("summary"),
        "parts": [{"id": p["id"], "name": p["name"], "role": p["role"], "found": len(p["instances"]),
                   "measured": [{k: m[k] for k in ("label", "value", "unit", "note") if k in m} for m in p["measures"]]}
                  for p in result.get("parts", [])],
        "relations": [{"from": r["from"], "to": r["to"], "label": r["label"], "kind": r.get("kind"),
                       **({"touching_seen": r["contact"]} if "contact" in r else {})} for r in result.get("relations", [])],
        "measurements": result.get("evidence", {}),
        "checks_of_model_words": result.get("checks", []),
        "how_it_works_background_knowledge": [s["text"] for s in result.get("how_it_works", [])],
        "already_told_the_user": [result.get("headline")] + [s["text"] for s in result.get("seen", [])],
    }


def run_facts(events: list[dict], title: str) -> dict:
    """The parts of a run's event stream a person would ask about."""
    facts: dict = {"kind": "video experiment", "experiment": title}
    for e in events:
        t = e["type"]
        if t == "calibration":
            facts["scale"] = {"method": e["method"], "px_per_m": e["px_per_m"], "relative_uncertainty": e["rel_uncertainty"]}
        elif t == "fit" and e.get("params"):
            facts.setdefault("fits", []).append({"model": e["model"], "params": e["params"], "at_t": e["t"]})
        elif t == "law":
            facts.setdefault("laws_found", []).append({k: e[k] for k in ("label", "equation", "meaning", "searched")})
        elif t == "measurement":
            facts["measured"] = e["items"]
        elif t == "memory_vs_eyes":
            facts.setdefault("memory_vs_eyes", []).append({k: e[k] for k in ("quantity", "memory_value", "eyes_value")
                                                          if k in e} | {k: e[k] for k in ("eyes_err", "eyes_text") if k in e})
        elif t == "prediction":
            facts.setdefault("predictions", []).append({k: e[k] for k in ("id", "text", "value", "unit", "ci90",
                                                                          "based_on_until_t", "target_t") if k in e})
        elif t == "verdict":
            facts.setdefault("verdicts", []).append({k: e[k] for k in ("id", "actual", "hit") if k in e})
        elif t == "hypothesis":
            best = max(e["candidates"], key=lambda c: c["score"])
            facts.setdefault("law_races", []).append({"topic": e.get("topic", "motion"), "winner": best["name"],
                                                      "winner_score": best["score"]})
        elif t == "explain":
            facts["explanation_already_given"] = e["text"]
    return facts


def record_facts(rec: dict) -> dict:
    """The recorder's key moments (contacts with speeds in and out, peaks) for the assistant."""
    return {"key_moments": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in e.items()
                             if k in ("object", "t", "kind", "y_m", "speed_in", "speed_out", "ratio") and v is not None}
                            for e in rec["events"] if e["kind"] in ("contact", "peak", "start", "end")],
            "video_sha256": rec.get("video_sha256")}


def ask(context_key: str, facts: dict, question: str) -> dict:
    from openai import OpenAI

    from . import budget
    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("Asking needs Muse Spark: set META_API_KEY in backend/.env")
    client = OpenAI(api_key=key, base_url=os.environ.get("META_API_BASE", "https://api.meta.ai/v1"))
    model = os.environ.get("MUSE_MODEL", "muse-spark-1.3")
    evidence: dict = {}
    _numbers(facts, evidence)
    history = CONVERSATIONS.setdefault(context_key, [])
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "FACTS:\n" + json.dumps(facts, default=str)}]
    for turn in history[-MAX_TURNS:]:
        messages += [{"role": "user", "content": turn["q"]}, {"role": "assistant", "content": json.dumps({"answer": turn["a"]})}]
    messages.append({"role": "user", "content": question})

    def call(msgs) -> dict:
        budget.check(0.02)
        r = client.chat.completions.create(model=model, messages=msgs, max_tokens=3000,
                                           reasoning_effort=os.environ.get("MUSE_ASK_EFFORT", "low"))
        budget.record_tokens(model, r.usage, what="ask")
        raw = r.choices[0].message.content or ""
        try:
            return json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        except ValueError:
            return {"answer": raw.strip()}

    out = call(messages)
    answer = str(out.get("answer", "")).strip()
    ok, bad = verifier.check(answer, evidence)
    if not ok:
        # One chance to fix numbers that weren't measured.
        out = call(messages + [{"role": "assistant", "content": json.dumps(out)},
                               {"role": "user", "content": f"These numbers are not in FACTS: {', '.join(bad)}. Answer "
                                                           "again using only measured numbers, or none."}])
        answer = str(out.get("answer", "")).strip()
        ok, bad = verifier.check(answer, evidence)
    history.append({"q": question, "a": answer})
    parts = [p for p in out.get("parts", []) if isinstance(p, str)]
    return {"answer": answer, "parts": parts, "verified": ok, "unverified_numbers": bad, "turn": len(history)}


def reset(context_key: str) -> None:
    CONVERSATIONS.pop(context_key, None)


def transcribe(audio: bytes, filename: str) -> str:
    """Speech to text for recorded audio (any format ffmpeg reads): decoded to 16 kHz mono PCM and streamed to
    Muse Voice, which only works in realtime."""
    import asyncio
    import subprocess
    import tempfile
    from pathlib import Path

    from .synth import ffmpeg_exe
    from .voice import transcribe_pcm
    src = Path(tempfile.mkdtemp()) / (Path(filename).name or "audio")
    src.write_bytes(audio)
    pcm = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(src), "-f", "s16le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    src.unlink(missing_ok=True)
    return asyncio.run(transcribe_pcm(pcm))
