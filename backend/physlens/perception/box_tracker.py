"""Background-subtraction box tracker for a fixed camera that starts on an empty scene (crash-test side views).

The empty scene is the median of the first frames. In every frame, pixels that differ from it are cleaned up, and
the biggest blob is the vehicle. The detection carries its bounding box as a 4-corner outline, so an analysis can use
the front and rear edges (a car's centre moves as it crushes; its rear doesn't).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Detection, TrackResult


class BoxTracker:
    name = "opencv-box"

    def __init__(self, empty_fraction: float = 0.12, threshold: int = 40, min_area_frac: float = 0.01):
        self.empty_fraction = empty_fraction     # leading share of the clip assumed to show the empty scene
        self.threshold = threshold
        self.min_area_frac = min_area_frac

    def track(self, video: Path, prompt: str = "") -> TrackResult:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(cv2.GaussianBlur(f, (5, 5), 0))
        cap.release()
        if not frames:
            raise RuntimeError(f"cannot read {video}")
        H, W = frames[0].shape[:2]
        n_bg = max(3, int(len(frames) * self.empty_fraction))
        bg = np.median(np.stack(frames[:n_bg]), axis=0).astype(np.int16)
        kernel = np.ones((9, 9), np.uint8)
        out: list[Optional[Detection]] = []
        for f in frames:
            diff = (np.abs(f.astype(np.int16) - bg).max(axis=2) > self.threshold).astype(np.uint8)
            diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, kernel)
            n, labels, stats, _ = cv2.connectedComponentsWithStats(diff)
            if n < 2:
                out.append(None)
                continue
            k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            x, y, w, h, area = stats[k]
            if area < self.min_area_frac * H * W:
                out.append(None)
                continue
            # The blob's horizontal extent, measured where the blob is solid (its middle third in height), so a
            # shadow or floor reflection hanging off the bottom doesn't widen it.
            blob = labels[y:y + h, x:x + w] == k
            band = blob[h // 3: 2 * h // 3]
            cols = np.nonzero(band.sum(axis=0) > max(3, 0.08 * band.shape[0]))[0]
            x0, x1 = (x + int(cols[0]), x + int(cols[-1])) if len(cols) else (x, x + w - 1)
            x0, x1, y, h = int(x0), int(x1), int(y), int(h)      # plain ints: these go into JSON event logs
            box = [(x0, y), (x1, y), (x1, y + h - 1), (x0, y + h - 1)]
            out.append(Detection((x0 + x1) / 2, y + h / 2, (x1 - x0) / 2, box))
        return TrackResult(fps=fps, width=W, height=H, label=prompt or "vehicle", frames=out)
