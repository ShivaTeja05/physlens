"""NHTSA accelerometer curves (UDS-1992 files from the test's UDS zip) and the speed change they integrate to.

UDS-1992 layout, as found in these files: a 1,024-byte header (first and last sample index as big-endian int32 at
byte 790 and 794, sample spacing as float32 at 798), then 512-byte records, each "Y DATA" + 6 spaces and 125
big-endian float32 samples (g). Checked against NHTSA's text export of test 8555 curve 7.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/nhtsa_sensors.py 8555 4 7 10
"""
import glob
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = 9.80665


def uds(path):
    """Time (s) and acceleration (g) of one UDS curve."""
    b = Path(path).read_bytes()
    i0, i1 = np.frombuffer(b, ">i4", 2, 790)
    dt = float(np.frombuffer(b, ">f4", 1, 798)[0])
    vals = b"".join(b[o + 12:o + 512] for o in range(1024, len(b), 512) if b[o:o + 6] == b"Y DATA")
    return np.arange(i0, i1 + 1) * dt, np.frombuffer(vals, ">f4", i1 - i0 + 1).astype(float)


def curve(test, ch):
    return uds(glob.glob(str(ROOT / "datasets" / "nhtsa" / str(test) / "uds" / f"v{test:05d}_{ch:03d}*.uds"))[0])


def speed_change(t, g, t_end=0.25):
    """Speed change (m/s) from t = 0, the pre-impact mean removed: time (s) and the running change."""
    g = g - g[t < -0.002].mean()
    m = (t >= 0) & (t <= t_end)
    return t[m], np.r_[0, np.cumsum((g[m][1:] + g[m][:-1]) / 2 * np.diff(t[m]))] * G


if __name__ == "__main__":
    test = int(sys.argv[1])
    for ch in map(int, sys.argv[2:]):
        t, v = speed_change(*curve(test, ch))
        print(f"{test} curve {ch}: speed change {v[-1] * 3.6:+.1f} km/h by {t[-1] * 1000:.0f} ms")
