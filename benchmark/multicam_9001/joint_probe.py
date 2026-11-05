"""Probe: one MapAnything call on five synced NHTSA 9001 cameras at one pre-impact instant (unknown poses).
Saves poses, per-view points and a cache for the next steps."""
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens.geometry.client import reconstruct  # noqa: E402

D = ROOT / "datasets" / "nhtsa" / "9001"
OUT = Path(__file__).parent / "out"
OUT.mkdir(exist_ok=True)
CAMS = ["cam01_overhead_wide", "cam02", "cam04_struck_side", "cam05_rear_impact", "cam06_front_oblique"]
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def frame(cam, i):
    cap = cv2.VideoCapture(str(D / f"{cam}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    assert ok, (cam, i)
    return im


ims = [frame(c, FRAME) for c in CAMS]
for c, im in zip(CAMS, ims):
    cv2.imwrite(str(OUT / f"f{FRAME}_{c}.jpg"), im)
t0 = time.time()
views = reconstruct(ims, target_width=518, timeout_s=900)
print(f"GPU call {time.time() - t0:.0f} s")
summary = {}
for c, v in zip(CAMS, views):
    P = v.camera_pose
    pts = v.points_world.reshape(-1, 3)
    ok = np.isfinite(pts).all(1)
    if v.valid is not None:
        ok &= v.valid.reshape(-1).astype(bool)
    summary[c] = {"position": np.round(P[:3, 3], 3).tolist(), "forward": np.round(P[:3, 2], 3).tolist(),
                  "fov_x_deg": round(float(np.degrees(2 * np.arctan(v.depth.shape[1] / 2 / v.intrinsics[0, 0]))), 1),
                  "median_depth_m": round(float(np.nanmedian(v.depth)), 2), "points": int(ok.sum())}
    np.savez_compressed(OUT / f"f{FRAME}_{c}.npz", pts=v.points_world.astype(np.float32), depth=v.depth,
                        conf=v.confidence if v.confidence is not None else np.zeros(1), K=v.intrinsics, pose=P)
(OUT / f"f{FRAME}_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
