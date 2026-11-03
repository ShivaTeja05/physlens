"""Tracker interface. SAM 3.1 and the OpenCV fallback both return this shape."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class Detection:
    x_px: float
    y_px: float
    r_px: float                      # equivalent radius from mask area
    outline: list[tuple[int, int]] = field(default_factory=list)  # simplified mask contour for overlays


@dataclass
class TrackResult:
    fps: float
    width: int
    height: int
    label: str
    frames: list[Optional[Detection]]  # one entry per video frame; None when the object wasn't found

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps


class Tracker(Protocol):
    name: str

    def track(self, video: Path, prompt: str) -> TrackResult:
        """Follow the object described by `prompt` (e.g. "orange ball") through every frame."""
        ...
