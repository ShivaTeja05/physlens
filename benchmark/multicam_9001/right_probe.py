"""The Camry's right side from NHTSA 9001's documentation video: one MapAnything call on the slow pan along the right
side (0:39.5 to 0:45.5, 2 frames a second), where neighbouring frames overlap so the camera path can be solved.
Saves each frame's points, confidence, pose and image for registration to the car frame."""
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

VIDEO = ROOT / "datasets" / "nhtsa" / "9001" / "doc_video.mp4"
OUT = Path(__file__).parent / "out" / "right"
OUT.mkdir(parents=True, exist_ok=True)
TIMES = [round(39.5 + 0.5 * k, 2) for k in range(13)]

cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS)
ims = []
for t in TIMES:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, im = cap.read()
    assert ok, t
    ims.append(im)
    cv2.imwrite(str(OUT / f"t{t:05.2f}.jpg"), im)
t0 = time.time()
views = reconstruct(ims, target_width=518, timeout_s=900)
print(f"GPU call {time.time() - t0:.0f} s for {len(ims)} frames")
summary = []
for t, im, v in zip(TIMES, ims, views):
    np.savez_compressed(OUT / f"t{t:05.2f}.npz", pts=v.points_world.astype(np.float32), conf=v.confidence if v.confidence is not None else np.zeros(1),
                        K=v.intrinsics, pose=v.camera_pose, sx=v.scale_x, sy=v.scale_y,
                        img=cv2.resize(im, (v.depth.shape[1], v.depth.shape[0])))
    summary.append({"t": t, "position": np.round(v.camera_pose[:3, 3], 3).tolist(),
                    "fov_x": round(float(np.degrees(2 * np.arctan(v.depth.shape[1] / 2 / v.intrinsics[0, 0]))), 1)})
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary))
