"""Scan-mode measurements that don't need the Meta API."""
import math

import numpy as np
import pytest

from physlens.scan import lit_fraction


def moon(k: float, r: int = 180, size: int = 480, earthshine: float = 0.05, seed: int = 0):
    """A Moon with illuminated fraction k (lit on the right), dark lava plains, a faint night side and noise.
    For orthographic viewing the lit area is exactly k of the disc."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] - size / 2
    disc = xx ** 2 + yy ** 2 <= r ** 2
    a = abs(2 * k - 1) * r                       # the terminator's half-ellipse semi-axis
    ell = (xx / max(a, 1e-6)) ** 2 + (yy / r) ** 2 <= 1
    lit = disc & ((xx >= 0) | ell) if k >= 0.5 else disc & (xx >= 0) & ~ell
    img = np.where(disc, 255 * earthshine, 0.0)
    img[lit] = 220
    maria = ((xx - 40) ** 2 + (yy + 50) ** 2 < 55 ** 2) | ((xx - 90) ** 2 + (yy - 60) ** 2 < 35 ** 2)
    img[lit & maria] = 120                       # dark plains, still sunlit
    img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
    return img, lit, 100 * lit.sum() / disc.sum()


@pytest.mark.parametrize("k", [0.12, 0.3, 0.5, 0.7, 0.95])
def test_lit_fraction_any_phase(k):
    img, lit, truth = moon(k)
    # SAM's outline is only a hint: give it a slightly shrunken version of the lit part.
    from scipy.ndimage import binary_erosion
    got = lit_fraction(img, binary_erosion(lit, iterations=6))
    assert got is not None
    pct, (cx, cy, r), (lo, hi) = got
    assert abs(pct - truth) < 3, f"k={k}: measured {pct:.1f}%, truth {truth:.1f}%"
    assert lo - 1 <= truth <= hi + 1
    assert abs(r - 180) < 4 and math.hypot(cx - 240, cy - 240) < 4


def test_whole_disc_outline():
    """When SAM outlines the whole disc (night side faintly visible), the night side still doesn't count as lit."""
    img, lit, truth = moon(0.5, earthshine=0.1)
    yy, xx = np.mgrid[0:480, 0:480] - 240
    pct, _, _ = lit_fraction(img, xx ** 2 + yy ** 2 <= 175 ** 2)
    assert abs(pct - truth) < 3
