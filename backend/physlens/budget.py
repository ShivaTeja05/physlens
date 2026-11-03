"""Spending meter for Meta Model API calls.

Every call is priced from the usage the API reports and appended to backend/.spend.jsonl. Before a call,
`check()` refuses it if the running total plus the call's estimate would pass the cap
(PHYSLENS_BUDGET_USD, default $50). Prices from dev.meta.ai/docs/pricing-rate-limits (Standard mode).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

SPEND_FILE = Path(__file__).resolve().parents[1] / ".spend.jsonl"
DEFAULT_CAP_USD = 50.0

# USD per 1M tokens: (input, cached input, output). Reasoning tokens are billed as output.
TOKEN_PRICES = {
    "muse-spark-1.3": (1.25, 0.15, 4.25),
    "muse-spark-1.2": (1.25, 0.15, 4.25),
    "muse-spark-1.1": (1.25, 0.15, 4.25),
    "muse-spark-1.3-contributor": (0.10, 0.002, 0.20),
    "muse-spark-1.2-contributor": (0.10, 0.002, 0.20),
}
SAM_PER_FRAME = 0.20 / 1000
SAM_PER_IMAGE = 2.50 / 1000
SAM_MIN_FRAMES = 50

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def cap() -> float:
    return float(os.environ.get("PHYSLENS_BUDGET_USD", DEFAULT_CAP_USD))


def spent() -> float:
    if not SPEND_FILE.exists():
        return 0.0
    return sum(json.loads(line)["usd"] for line in SPEND_FILE.read_text().splitlines() if line.strip())


def check(estimate_usd: float = 0.0) -> None:
    total = spent()
    if total + estimate_usd > cap():
        raise BudgetExceeded(f"budget cap reached: ${total:.2f} spent of ${cap():.2f} "
                             f"(this call would add ~${estimate_usd:.3f})")


def sam_video_estimate(frames: int) -> float:
    return max(frames, SAM_MIN_FRAMES) * SAM_PER_FRAME


def token_cost(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    p_in, p_cached, p_out = TOKEN_PRICES.get(model, TOKEN_PRICES["muse-spark-1.3"])
    fresh = max(input_tokens - cached_tokens, 0)
    return (fresh * p_in + cached_tokens * p_cached + output_tokens * p_out) / 1e6


def record(model: str, usd: float, **detail) -> float:
    row = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "model": model, "usd": round(usd, 6), **detail}
    with _lock, SPEND_FILE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return usd


def record_tokens(model: str, usage, what: str) -> float:
    """Accepts Responses usage (input_tokens/output_tokens) or Chat Completions usage (prompt/completion)."""
    if usage is None:
        return 0.0
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    details = getattr(usage, "input_tokens_details", None) or getattr(usage, "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return record(model, token_cost(model, inp, out, cached), what=what, input_tokens=inp, output_tokens=out)


def record_sam(model: str, usage, what: str) -> float:
    frames = getattr(usage, "video_frames_processed", None) if usage is not None else None
    if frames:
        return record(model, max(frames, SAM_MIN_FRAMES) * SAM_PER_FRAME, what=what, frames=frames)
    return record(model, SAM_PER_IMAGE, what=what, images=1)


def summary() -> dict:
    rows = [json.loads(l) for l in SPEND_FILE.read_text().splitlines() if l.strip()] if SPEND_FILE.exists() else []
    by_model: dict[str, float] = {}
    for r in rows:
        by_model[r["model"]] = by_model.get(r["model"], 0.0) + r["usd"]
    return {"spent_usd": round(sum(by_model.values()), 4), "cap_usd": cap(), "calls": len(rows),
            "by_model": {k: round(v, 4) for k, v in by_model.items()}}
