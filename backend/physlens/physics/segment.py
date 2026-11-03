"""Split a bouncing trajectory into free-flight segments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.signal import find_peaks


@dataclass
class Flight:
    index: int
    start: int  # first sample used for fitting (inclusive)
    stop: int   # last sample used for fitting (exclusive)
    t_start: float
    bounce_before: Optional[int]  # sample index of the contact that launched this flight
    bounce_after: Optional[int]   # sample index of the contact that ended it


def find_bounces(t: np.ndarray, y: np.ndarray, radius_m: float) -> list[int]:
    """Indices of floor contacts: local minima of height that sit near the floor, plus contacts hidden
    in a tracking gap (the ball is often blurred out of sight exactly when it hits the floor)."""
    floor = float(np.percentile(y, 2))
    prominence = max(0.3 * radius_m, 0.005)
    dt = float(np.median(np.diff(t)))
    min_gap = max(3, int(round(0.03 / dt)))
    minima, _ = find_peaks(-y, prominence=prominence, distance=min_gap)
    contacts = {int(i) for i in minima if y[i] < floor + 2.0 * radius_m}

    # Falling before a gap and rising after it means it bounced while unseen. The contact is recorded at
    # the first sample after the gap; the fit finds the true contact time from where the curves meet.
    for i in range(1, len(t) - 2):
        if t[i + 1] - t[i] > 1.5 * dt and y[i] - y[i - 1] < 0 < y[i + 2] - y[i + 1]:
            if not any(abs(c - (i + 1)) < min_gap for c in contacts):
                contacts.add(i + 1)
    return sorted(contacts)


def split_flights(t: np.ndarray, bounces: list[int], fps: float, min_samples: int = 6) -> list[Flight]:
    """Free-flight windows between contacts. A few frames either side of each contact are dropped
    because the ball squashes while it touches the floor."""
    guard = max(1, int(round(0.008 * fps)))
    dt = float(np.median(np.diff(t)))
    # A contact hidden in a tracking gap sits between samples, so nothing on either side needs trimming.
    hidden = {c for c in bounces if c > 0 and t[c] - t[c - 1] > 1.5 * dt}
    contacts: list[Optional[int]] = [None, *bounces, None]
    flights = []
    for before, after in zip(contacts[:-1], contacts[1:]):
        if before is None:
            start = 0
        else:
            start = before if before in hidden else before + guard + 1
        if after is None:
            stop = len(t)
        else:
            stop = after if after in hidden else after - guard
        if stop - start >= min_samples:
            flights.append(Flight(len(flights), start, stop, float(t[start]), before, after))
    return flights
