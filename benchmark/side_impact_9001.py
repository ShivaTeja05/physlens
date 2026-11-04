"""NHTSA test 9001 (NCAP side): a 1,362 kg crabbed moving barrier hits a standing 1,666 kg 2015 Toyota Camry at
62.19 km/h. Video (overhead camera 2, 1,000 fps, time stamped) against the cars' own accelerometers.

  * Before impact: the barrier face's motion (median optical flow of its features), scaled by the face's standard
    width (1,676 mm), gives speed AND direction: the barrier is crabbed 27°.
  * After impact: the same features (barrier) and features on the car's front and rear, followed through the crash
    with a forward-backward check.
  * Ground truth: barrier and car centre-of-gravity accelerometers (binary UDS files: 512-byte blocks, each with an
    8-byte marker), integrated from the official starting speeds.

  cd backend && .venv/bin/python ../benchmark/side_impact_9001.py
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "datasets" / "nhtsa" / "9001"
FACE_WIDTH_M = 1.676          # FMVSS 214 moving deformable barrier face width
M_BARRIER, M_CAR = 1362.0, 1666.0
LK = dict(winSize=(25, 25), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))


def frames() -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(DATA / "cam02.mp4"))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))


def face_width_px(f: np.ndarray) -> int:
    """The white barrier face at contact (time 0 = file frame 50)."""
    H = f.shape[0]
    band = f[int(H * 0.68): int(H * 0.93)] > 170
    cols = np.nonzero(band.sum(axis=0) > 0.4 * band.shape[0])[0]
    return int(np.ptp(cols))


def follow(F, seed, region, moving=None, end=200):
    """Median per-millisecond motion (dx, dy in px) of features seeded in `region`, followed frame to frame."""
    mask = np.zeros_like(F[0])
    x0, y0, x1, y1 = region
    mask[y0:y1, x0:x1] = 255
    p = cv2.goodFeaturesToTrack(F[seed], 500, 0.01, 6, mask=mask)
    if moving is not None:
        q, st, _ = cv2.calcOpticalFlowPyrLK(F[seed], F[seed + 1], p, None, **LK)
        d = (q - p)[:, 0]
        p = p[(st[:, 0] == 1) & (np.abs(d[:, 1]) > moving)]
    out = []
    for i in range(seed, min(seed + end, len(F) - 1)):
        q, st, _ = cv2.calcOpticalFlowPyrLK(F[i], F[i + 1], p, None, **LK)
        back, st2, _ = cv2.calcOpticalFlowPyrLK(F[i + 1], F[i], q, None, **LK)
        good = (st[:, 0] == 1) & (st2[:, 0] == 1) & (np.linalg.norm((back - p)[:, 0], axis=1) < 0.5)
        if good.sum() < 6:
            break
        out.append(np.median((q - p)[good][:, 0], axis=0))
        p = q[good]
    return np.array(out)


def uds(channel: str, n: int = 3500) -> np.ndarray:
    raw = (DATA / "sensors" / f"ch{channel}.uds").read_bytes()
    pos, data = raw.index(b"Y DATA") + 12, bytearray()
    while len(data) < 4 * n and pos < len(raw):
        nxt = (pos // 512 + 1) * 512
        data += raw[pos:nxt]
        pos = nxt + 8
    return np.frombuffer(bytes(data[: 4 * n]), dtype=">f4").astype(float)


def sensor_dv(channel: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(-500, 3000) * 1e-4
    b, a = butter(2, 100 / (0.5 / 1e-4))
    g = filtfilt(b, a, uds(channel))
    return t, np.cumsum(np.where(t >= 0, g, 0.0)) * 9.81 * 1e-4 * 3.6       # km/h change


def main() -> None:
    F = frames()
    ppm = face_width_px(F[50]) / FACE_WIDTH_M
    kmh = lambda px_per_ms: px_per_ms / ppm * 1000 * 3.6
    pre = follow(F, 41, (300, 700, 1300, 1024), moving=3.0, end=8)
    dx, dy = np.median(pre, axis=0)
    speed, angle = kmh(math.hypot(dx, dy)), math.degrees(math.atan2(abs(dx), abs(dy)))
    print(f"scale {ppm:.0f} px/m (face {FACE_WIDTH_M} m)")
    print(f"BEFORE: barrier {speed:.1f} km/h at {angle:.1f}° crab  (official 62.19 km/h at 27.0°);"
          f" straight in {kmh(-dy):.1f} (official 55.41), along the car {kmh(dx):.1f} (official 28.23)")

    bar = follow(F, 45, (300, 700, 1300, 1024), moving=3.0)
    car_f = follow(F, 48, (0, 40, 400, 470))
    car_r = follow(F, 48, (1180, 40, 1536, 470))
    t, dv_bar = sensor_dv("056")
    _, dv_car = sensor_dv("034")
    at = lambda arr, seed, ms: kmh(-np.median(arr[ms - (seed - 50) - 5: ms - (seed - 50) + 5, 1]))
    print("AFTER (straight-in direction, km/h):   video            sensors (centre of gravity)")
    for ms in (20, 40, 60, 80, 100):
        sb = 55.41 + dv_bar[np.argmin(np.abs(t - ms / 1000))]
        sc = dv_car[np.argmin(np.abs(t - ms / 1000))]
        car_mid = (at(car_f, 48, ms) + at(car_r, 48, ms)) / 2 if ms - 2 < len(car_r) else float("nan")
        print(f"  +{ms:3d} ms   barrier {at(bar, 45, ms):5.1f}   car ≈ {car_mid:5.1f}  |  barrier {sb:5.1f}   car {sc:5.1f}")
    sb, sc = 55.41 + dv_bar[np.argmin(np.abs(t - 0.1))], dv_car[np.argmin(np.abs(t - 0.1))]
    p0, p1 = M_BARRIER * 55.41, M_BARRIER * sb + M_CAR * sc
    print(f"MOMENTUM (sensors, +100 ms): {p1 / p0 * 100:.0f}% of the barrier's straight-in momentum is in the two bodies;"
          f" a perfectly shared speed would be {M_BARRIER * 55.41 / (M_BARRIER + M_CAR):.1f} km/h")


if __name__ == "__main__":
    main()
