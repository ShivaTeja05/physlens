"""The right-side pan again, with one fixed pinhole lens for all 13 frames (MapAnything's own guesses ranged 79–104°
within one continuous pan). Usage: right_fixed_lens.py <fov_deg> ... ; writes out/right_fov<deg>/ like right_probe."""
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

RD = Path(__file__).parent / "out" / "right"
TIMES = [round(39.5 + 0.5 * k, 2) for k in range(13)]
ims = [cv2.imread(str(RD / f"t{t:05.2f}.jpg")) for t in TIMES]
h, w = ims[0].shape[:2]
for fov in [float(a) for a in sys.argv[1:]]:
    out = Path(__file__).parent / "out" / f"right_fov{int(fov)}"
    out.mkdir(parents=True, exist_ok=True)
    f = w / 2 / np.tan(np.radians(fov / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1.0]])
    t0 = time.time()
    views = reconstruct(ims, target_width=518, timeout_s=900, intrinsics=[K] * len(ims))
    summary = []
    for t, im, v in zip(TIMES, ims, views):
        np.savez_compressed(out / f"t{t:05.2f}.npz", pts=v.points_world.astype(np.float32),
                            conf=v.confidence if v.confidence is not None else np.zeros(1), K=v.intrinsics,
                            pose=v.camera_pose, sx=v.scale_x, sy=v.scale_y,
                            img=cv2.resize(im, (v.depth.shape[1], v.depth.shape[0])))
        summary.append({"t": t, "position": np.round(v.camera_pose[:3, 3], 3).tolist(),
                        "fov_x": round(float(np.degrees(2 * np.arctan(v.depth.shape[1] / 2 / v.intrinsics[0, 0]))), 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"fov {fov:.0f}: {time.time() - t0:.0f} s; returned fov {summary[0]['fov_x']}..{summary[-1]['fov_x']}", flush=True)
