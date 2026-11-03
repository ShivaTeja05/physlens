"""Fallback tracker for real footage without a GPU: follows the small, round thing that moves.

With a fixed camera, the median of frames spread across the clip is the empty background. Anything that
differs from it is moving. Among the moving blobs we pick the one that is round, small, matches the colour
hint if one is given, and continues the path we were already following. SAM 3.1 replaces this for scenes
with several moving objects or a moving camera.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Detection, TrackResult
from .measure import soft_measure
from .opencv_tracker import HUE_RANGES

BACKGROUND_SAMPLES = 60


def _read_background(video: Path, n_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    picks = np.unique(np.linspace(0, max(n_frames - 1, 0), min(BACKGROUND_SAMPLES, max(n_frames, 1))).astype(int))
    frames = []
    for i in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if not frames:
        raise FileNotFoundError(f"cannot read frames from {video}")
    return np.median(np.stack(frames), axis=0).astype(np.int16)


class MotionBlobTracker:
    name = "opencv-motion"

    def __init__(self, color: Optional[str] = None, threshold: int = 28, min_area_px: int = 6,
                 max_area_frac: float = 0.02, min_circularity: float = 0.3):
        self.color = color
        self.threshold = threshold
        self.min_area_px = min_area_px
        self.max_area_frac = max_area_frac
        self.min_circularity = min_circularity

    def _color_match(self, hsv: np.ndarray, contour: np.ndarray) -> float:
        if self.color not in HUE_RANGES:
            return 0.0
        mask = np.zeros(hsv.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        h, s = hsv[..., 0][mask > 0], hsv[..., 1][mask > 0]
        if h.size == 0:
            return 0.0
        in_hue = np.zeros(h.shape, bool)
        for lo, hi in HUE_RANGES[self.color]:
            in_hue |= (h >= lo) & (h <= hi)
        return float(np.mean(in_hue & (s >= 60)))

    @staticmethod
    def _looks_like_shadow(frame: np.ndarray, background: np.ndarray, contour: np.ndarray) -> bool:
        """Shadows make the background darker without adding colour; a real object usually adds colour or
        brightness."""
        mask = np.zeros(frame.shape[:2], np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        sel = mask > 0
        f, b = frame[sel].astype(np.int16), background[sel]
        d_bright = float(np.mean(f.max(axis=1) - b.max(axis=1)))
        d_chroma = float(np.mean((f.max(axis=1) - f.min(axis=1)) - (b.max(axis=1) - b.min(axis=1))))
        return d_bright < -8 and d_chroma < 6

    def _candidates(self, frame: np.ndarray, background: np.ndarray):
        diff = np.abs(frame.astype(np.int16) - background).max(axis=2)
        mask = (diff > self.threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = self.max_area_frac * frame.shape[0] * frame.shape[1]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if area < self.min_area_px or area > max_area or perimeter == 0:
                continue
            circularity = 4 * np.pi * area / perimeter ** 2
            if circularity < self.min_circularity:
                continue
            if self._looks_like_shadow(frame, background, c):
                continue
            measured = soft_measure(diff, c, min_contrast=self.threshold * 0.5)
            if measured is None:
                continue
            out.append((c, measured, circularity, self._color_match(hsv, c)))
        return out

    def track(self, video: Path, prompt: str) -> TrackResult:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {video}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        background = _read_background(video, n)

        per_frame: list[list[_Blob]] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            blobs = []
            for c, (cx, cy, area), circ, colour in self._candidates(frame, background):
                outline = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
                det = Detection(cx, cy, float(np.sqrt(area / np.pi)), [(int(x), int(y)) for x, y in outline[:48]])
                blobs.append(_Blob(len(per_frame), det, circ + 2.0 * colour))
            per_frame.append(blobs)
        cap.release()

        frames: list[Optional[Detection]] = [None] * len(per_frame)
        for blob in _best_path(per_frame, float(np.hypot(width, height))):
            frames[blob.frame] = blob.det
        return TrackResult(fps=fps, width=width, height=height, label=prompt, frames=frames)


class _Blob:
    __slots__ = ("frame", "det", "quality")

    def __init__(self, frame: int, det: Detection, quality: float):
        self.frame, self.det, self.quality = frame, det, quality


MAX_GAP = 5          # frames a track may go unseen while linking (motion blur, occlusion)
CHAIN_MAX_GAP = 8    # frames allowed between two tracks joined into one path (fast blur near a bounce)
MIN_EXTENT_FRAC = 0.03  # a track that barely moves and stays this small (× diagonal) is jitter, not the object


def _predict(tr: list[_Blob], frame: int) -> tuple[float, float, float]:
    """Constant-velocity guess of where a track is at `frame`, and its speed in px/frame."""
    last = tr[-1]
    if len(tr) < 2:
        return last.det.x_px, last.det.y_px, 0.0
    prev = tr[-2]
    step = max(last.frame - prev.frame, 1)
    vx, vy = (last.det.x_px - prev.det.x_px) / step, (last.det.y_px - prev.det.y_px) / step
    gap = frame - last.frame
    return last.det.x_px + vx * gap, last.det.y_px + vy * gap, float(np.hypot(vx, vy))


def _gate(speed: float, gap: int, diag: float, known: bool) -> float:
    """How far a blob may be from the prediction and still belong to the track. A new track has no
    velocity yet, so it gets a fixed allowance. Something standing still must stay put. Something moving
    may change speed (gravity, bounces) and drift further the longer it has been unseen."""
    if not known:
        return 0.05 * diag * np.sqrt(gap)
    if speed < 2.0:
        return 6.0 * gap
    # Gravity changes the speed by up to ~0.003 × diagonal per frame² at 25 fps.
    return (10.0 + 0.8 * speed) * gap + 0.003 * diag * gap ** 2


def _link_tracklets(per_frame: list[list[_Blob]], diag: float) -> list[list[_Blob]]:
    """Frame-to-frame linking. Blobs are assigned to tracks jointly (Hungarian), so two tracks
    passing near each other can't swap identities."""
    from scipy.optimize import linear_sum_assignment

    active: list[list[_Blob]] = []
    done: list[list[_Blob]] = []
    for i, blobs in enumerate(per_frame):
        matched: set[int] = set()
        if active and blobs:
            cost = np.full((len(active), len(blobs)), 1e9)
            for a, tr in enumerate(active):
                px, py, speed = _predict(tr, i)
                limit = _gate(speed, i - tr[-1].frame, diag, len(tr) >= 2)
                for b, blob in enumerate(blobs):
                    d = np.hypot(blob.det.x_px - px, blob.det.y_px - py)
                    if d <= limit:
                        cost[a, b] = d
            rows, cols = linear_sum_assignment(cost)
            for a, b in zip(rows, cols):
                if cost[a, b] < 1e9:
                    active[a].append(blobs[b])
                    matched.add(b)
        still_active = []
        for tr in active:
            (still_active if i - tr[-1].frame <= MAX_GAP else done).append(tr)
        active = still_active + [[blobs[b]] for b in range(len(blobs)) if b not in matched]
    return done + active


def _score(tr: list[_Blob], diag: float) -> float:
    xs = np.array([b.det.x_px for b in tr])
    ys = np.array([b.det.y_px for b in tr])
    steps = np.hypot(np.diff(xs), np.diff(ys)) / np.maximum(np.diff([b.frame for b in tr]), 1)
    if max(np.ptp(xs), np.ptp(ys)) < MIN_EXTENT_FRAC * diag and (not len(steps) or np.median(steps) < 1.2):
        return -np.inf
    motion = np.minimum(steps / (0.02 * diag), 2.0).sum() if len(steps) else 0.0
    return sum(b.quality for b in tr) + motion


def _best_path(per_frame: list[list[_Blob]], diag: float) -> list[_Blob]:
    """Highest-scoring chain of non-overlapping tracklets, where each joins the previous one's end."""
    tracks = [tr for tr in _link_tracklets(per_frame, diag) if len(tr) >= 3]
    scored = [(tr, _score(tr, diag)) for tr in tracks]
    scored = [(tr, s) for tr, s in scored if np.isfinite(s)]
    if not scored:
        return []
    scored.sort(key=lambda p: p[0][0].frame)
    best_total = [s for _, s in scored]
    back = [-1] * len(scored)
    for j, (tj, sj) in enumerate(scored):
        for i in range(j):
            ti = scored[i][0]
            gap = tj[0].frame - ti[-1].frame
            if gap <= 0 or gap > CHAIN_MAX_GAP:
                continue
            # A bounce reverses the motion, so accept a join near either the predicted or the last position.
            px, py, speed = _predict(ti, tj[0].frame)
            start = tj[0].det
            jump = min(np.hypot(start.x_px - px, start.y_px - py),
                       np.hypot(start.x_px - ti[-1].det.x_px, start.y_px - ti[-1].det.y_px))
            if jump <= _gate(speed, gap, diag, len(ti) >= 2) and best_total[i] + sj > best_total[j]:
                best_total[j], back[j] = best_total[i] + sj, i
    j = int(np.argmax(best_total))
    chain = []
    while j >= 0:
        chain = scored[j][0] + chain
        j = back[j]
    return chain
