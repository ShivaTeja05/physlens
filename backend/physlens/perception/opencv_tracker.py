"""Fallback tracker for development without a GPU: finds the most saturated blob (or a given colour).

Good enough for a bright ball on a plain background. SAM 3.1 replaces it for real scenes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Detection, TrackResult
from .measure import soft_measure

# OpenCV hue runs 0-179.
HUE_RANGES = {
    "orange": [(5, 25)],
    "red": [(0, 8), (170, 179)],
    "yellow": [(22, 38)],
    "green": [(40, 85)],
    "blue": [(95, 130)],
}


class ColorBlobTracker:
    name = "opencv-color"

    def __init__(self, color: Optional[str] = None, min_saturation: int = 90, min_value: int = 70,
                 min_area_px: int = 40):
        self.color = color
        self.min_saturation = min_saturation
        self.min_value = min_value
        self.min_area_px = min_area_px

    def _mask(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        mask = (s >= self.min_saturation) & (v >= self.min_value)
        if self.color in HUE_RANGES:
            in_hue = np.zeros_like(mask)
            for lo, hi in HUE_RANGES[self.color]:
                in_hue |= (h >= lo) & (h <= hi)
            mask &= in_hue
        mask = mask.astype(np.uint8) * 255
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    def _detect(self, frame: np.ndarray, near: Optional[Detection]) -> Optional[Detection]:
        mask = self._mask(frame)
        chroma = frame.max(axis=2).astype(np.int16) - frame.min(axis=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best, best_score = None, -np.inf
        for c in contours:
            if cv2.contourArea(c) < self.min_area_px:
                continue
            measured = soft_measure(chroma, c)
            if measured is None:
                continue
            cx, cy, area = measured
            # Prefer large blobs, and stay with the object we were following.
            score = area if near is None else area - 50.0 * np.hypot(cx - near.x_px, cy - near.y_px)
            if score > best_score:
                best_score = score
                outline = cv2.approxPolyDP(c, 1.5, True).reshape(-1, 2)
                best = Detection(cx, cy, float(np.sqrt(area / np.pi)), [(int(x), int(y)) for x, y in outline[:48]])
        return best

    def track(self, video: Path, prompt: str) -> TrackResult:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames: list[Optional[Detection]] = []
        last = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            det = self._detect(frame, last)
            frames.append(det)
            last = det or last
        cap.release()
        return TrackResult(fps=fps, width=width, height=height, label=prompt, frames=frames)
