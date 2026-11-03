"""Sub-pixel blob measurement shared by the fallback trackers."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def soft_measure(signal: np.ndarray, contour: np.ndarray, min_contrast: float = 20.0
                 ) -> Optional[tuple[float, float, float]]:
    """Sub-pixel centroid and area of one blob. Edge pixels are only partly covered by the object, so
    each pixel is weighted by where its signal (chroma, or difference from the background) sits between
    the surroundings and the blob's interior. Returns (cx, cy, area) or None if the blob has no contrast."""
    x, y, w, h = cv2.boundingRect(contour)
    pad = 4
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, signal.shape[1]), min(y + h + pad, signal.shape[0])
    s = signal[y0:y1, x0:x1].astype(float)
    inside = np.zeros(s.shape, np.uint8)
    cv2.drawContours(inside, [contour - [x0, y0]], -1, 255, -1)
    core = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=2) > 0
    if not core.any():
        core = cv2.erode(inside, np.ones((3, 3), np.uint8), iterations=1) > 0
    ring = cv2.dilate(inside, np.ones((3, 3), np.uint8), iterations=3) == 0
    if not core.any() or not ring.any():
        return None
    s_fg, s_bg = np.median(s[core]), np.median(s[ring])
    if s_fg - s_bg < min_contrast:
        return None
    weight = np.clip((s - s_bg) / (s_fg - s_bg), 0.0, 1.0)
    weight[ring] = 0.0
    area = weight.sum()
    if area <= 0:
        return None
    ys, xs = np.mgrid[y0:y1, x0:x1]
    return float((weight * xs).sum() / area), float((weight * ys).sum() / area), float(area)
