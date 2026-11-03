"""Pixel <-> metre conversion.

The image has y pointing down; the world frame has y pointing up with its origin at the image bottom edge.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mask edges are blurry by about a pixel, so a single-object calibration is never better than this.
MIN_REL_UNCERTAINTY = 0.01


@dataclass
class Calibration:
    method: str
    px_per_m: float
    rel_uncertainty: float
    image_height: int
    flipped: bool = False   # the picture is upside down, so world "up" is image "down"

    def to_world(self, x_px: np.ndarray, y_px: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        up = y_px if self.flipped else self.image_height - y_px
        return x_px / self.px_per_m, up / self.px_per_m

    def to_image(self, x_m: np.ndarray, y_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        y_px = y_m * self.px_per_m if self.flipped else self.image_height - y_m * self.px_per_m
        return x_m * self.px_per_m, y_px


def calibrate_ruler(px_per_m: float, image_height: int, rel_uncertainty: float = 0.015) -> Calibration:
    """Scale measured from a ruler or tape in the shot (pixels between two marks a known distance apart).
    The default uncertainty allows for a pixel or two of clicking error on a ~150 px ruler."""
    if px_per_m <= 0:
        raise ValueError("px_per_m must be positive")
    return Calibration(method="ruler", px_per_m=px_per_m, rel_uncertainty=rel_uncertainty, image_height=image_height)


def calibrate_known_object(radii_px: np.ndarray, diameter_m: float, image_height: int) -> Calibration:
    """Scale from an object of known size, e.g. a 6.5 cm ball tracked through the clip."""
    diameters = 2.0 * np.asarray(radii_px, dtype=float)
    diameters = diameters[np.isfinite(diameters) & (diameters > 0)]
    if diameters.size == 0:
        raise ValueError("no valid object sizes to calibrate from")
    mean_d = float(np.median(diameters))
    spread = float(np.std(diameters) / mean_d) if diameters.size > 1 else 1.0
    return Calibration(
        method="known_object",
        px_per_m=mean_d / diameter_m,
        rel_uncertainty=max(MIN_REL_UNCERTAINTY, spread),
        image_height=image_height,
    )
