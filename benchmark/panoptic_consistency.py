"""Keep only the stereo points that the whole dome agrees on. A real point on a person or the ball must land on that
person (a moving, foreground pixel) in every calibrated HD camera that sees it; a wrong stereo match lands on the
empty dome in most of them. Foreground per camera = difference to the clip's median frame (people and ball move).
One camera (HOLDOUT) is never used, neither for stereo nor for this vote, and checks the result independently.
Usage: backend/.venv/bin/python benchmark/panoptic_consistency.py [frame ...]   (after panoptic_stereo.py)
Writes datasets/panoptic/161029_sports1/stereo/f{frame}_kept.npz.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panoptic_ball import D, cameras  # noqa: E402

HOLDOUT = "00_15"
SCALE = 0.5
OUT = D / "stereo"
_bg: dict[str, np.ndarray] = {}


def video(name):
    cap = cv2.VideoCapture(str(D / "hd" / f"hd_{name}.mp4"))
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    return out


def foreground(name, frames: dict[int, np.ndarray], i: int) -> np.ndarray:
    if name not in _bg:
        _bg[name] = np.median(np.stack(frames[name][::6]), 0).astype(np.int16)
    m = (np.abs(frames[name][i].astype(np.int16) - _bg[name]) > 22).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(m, np.ones((9, 9), np.uint8)) > 0


def votes(cams, P, masks, names):
    """For each point: in how many of `names` it is in view, and in how many of those it lands on foreground."""
    seen = np.zeros(len(P), int)
    fg = np.zeros(len(P), int)
    for n in names:
        c = cams[n]
        Xc = (c["R"] @ P.T + c["t"]).T
        front = Xc[:, 2] > 0.2
        px, _ = cv2.projectPoints(P.astype(np.float64), cv2.Rodrigues(c["R"])[0], c["t"], c["K"], c["dist"])
        px = (px.reshape(-1, 2) * SCALE).round().astype(int)
        h, w = masks[n].shape
        ok = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        seen += ok
        hit = np.zeros(len(P), bool)
        hit[ok] = masks[n][px[ok, 1], px[ok, 0]]
        fg += hit
    return seen, fg


if __name__ == "__main__":
    cams = cameras()
    voters = [n for n in cams if n != HOLDOUT]
    frames = {n: video(n) for n in cams}
    for i in [int(x) for x in sys.argv[1:]] or [30]:
        z = np.load(OUT / f"f{i:03d}.npz")
        P, rgb, pair = z["pts"].astype(np.float64), z["rgb"], z["pair"]
        h, r = -P[:, 1], np.hypot(P[:, 0], P[:, 2])
        area = (r < 2.2) & (h > 0.05) & (h < 2.6)               # people and ball, not the dome or floor
        P, rgb, pair = P[area], rgb[area], pair[area]
        masks = {n: foreground(n, frames, i) for n in cams}
        seen, fg = votes(cams, P, masks, voters)
        keep = (seen >= 6) & (fg >= 0.85 * seen)
        s_h, f_h = votes(cams, P, masks, [HOLDOUT])
        before = f_h[s_h > 0].mean() * 100 if (s_h > 0).any() else float("nan")
        after = f_h[keep & (s_h > 0)].mean() * 100 if (keep & (s_h > 0)).any() else float("nan")
        print(f"frame {i}: {len(P):,} stereo points in the capture area -> kept {keep.sum():,} "
              f"(seen by >=6 cameras and on a person/ball in >=85% of them)")
        print(f"   held-out camera {HOLDOUT}: {before:.0f}% of all points land on a person/ball, {after:.0f}% of the kept ones")
        np.savez_compressed(OUT / f"f{i:03d}_kept.npz", pts=P[keep].astype(np.float32), rgb=rgb[keep], seen=seen[keep], fg=fg[keep], pair=pair[keep])
