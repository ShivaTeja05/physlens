"""Event stream: the single contract between the engine, the stage display and the narrator.

Mirrors ../../contract/events.schema.json. Keep the two in sync when adding fields.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional


class EventLog:
    """Collects events for one run, writes them to a .jsonl replay file and notifies listeners."""

    def __init__(self, run_id: str, path: Optional[Path] = None):
        self.run_id = run_id
        self.path = path
        self.events: list[dict] = []
        self._listeners: list[Callable[[dict], None]] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("")

    def subscribe(self, fn: Callable[[dict], None]) -> None:
        self._listeners.append(fn)

    def emit(self, type_: str, t: float, **fields) -> dict:
        event = {"run_id": self.run_id, "t": round(float(t), 4), "type": type_, **fields}
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a") as f:
                f.write(json.dumps(event) + "\n")
        for fn in self._listeners:
            fn(event)
        return event

    def of_type(self, type_: str) -> list[dict]:
        return [e for e in self.events if e["type"] == type_]


def load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
