"""Prediction ledger: a prediction is hashed and time-stamped when it is locked, before the outcome is
processed, and appended to a file that is never rewritten. Anyone can re-hash a prediction to check it
wasn't edited after the fact."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Envelope fields added when the prediction travels as an event are not part of what was predicted.
UNHASHED = {"hash", "locked_at", "run_id", "t", "type"}


def prediction_hash(prediction: dict) -> str:
    body = {k: v for k, v in prediction.items() if k not in UNHASHED}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Ledger:
    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.locked: dict[str, dict] = {}
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def lock(self, prediction: dict) -> dict:
        if prediction["id"] in self.locked:
            raise ValueError(f"prediction {prediction['id']} is already locked")
        stamped = dict(prediction)
        stamped["hash"] = prediction_hash(prediction)
        stamped["locked_at"] = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        self.locked[prediction["id"]] = stamped
        self._append({"event": "lock", **stamped})
        return stamped

    def verify(self, prediction_id: str, actual: float) -> dict:
        p = self.locked[prediction_id]
        lo, hi = p["ci90"]
        verdict = {"id": prediction_id, "actual": actual, "hit": bool(lo <= actual <= hi),
                   "error": actual - p["value"]}
        self._append({"event": "verify", **verdict})
        return verdict

    def _append(self, record: dict) -> None:
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
