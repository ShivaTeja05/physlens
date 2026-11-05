"""Per camera, one pre-impact frame: the Camry's mask (SAM 3.1, darkest 'car' instance: the Camry is black, the
barrier is not) and MoGe-3 metric depth with its own lens. Cached in out/f{FRAME}_in_{cam}.npz."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens.geometry.client import sharp_depth  # noqa: E402
from physlens.perception.sam_image import segment  # noqa: E402

D = ROOT / "datasets" / "nhtsa" / "9001"
OUT = Path(__file__).parent / "out"
CAMS = ["cam01_overhead_wide", "cam02", "cam04_struck_side", "cam05_rear_impact", "cam06_front_oblique"]
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10


def frame(cam, i):
    cap = cv2.VideoCapture(str(D / f"{cam}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    assert ok
    return im


ims = {c: frame(c, FRAME) for c in CAMS}
masks, info = {}, {}
for c, im in ims.items():
    inst = segment(im, "car", what=f"9001 multicam {c}")
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    cand = [(float(gray[i.mask].mean()), int(i.mask.sum()), i) for i in inst if i.mask.sum() > 0.02 * gray.size]
    info[c] = [{"mean_gray": round(g, 1), "area": a, "conf": i.confidence} for g, a, i in cand]
    if not cand:
        print(c, "no car found")
        continue
    g, a, best = min(cand, key=lambda t: t[0])
    masks[c] = best.mask
    vis = im.copy()
    vis[best.mask] = (0.5 * vis[best.mask] + [0, 90, 255]).clip(0, 255).astype(np.uint8)
    cv2.imwrite(str(OUT / f"mask_f{FRAME}_{c}.jpg"), cv2.resize(vis, None, fx=0.4, fy=0.4))
print(json.dumps(info, indent=1))

depths = sharp_depth([ims[c] for c in CAMS], fov_x=None)
for c, d in zip(CAMS, depths):
    np.savez_compressed(OUT / f"f{FRAME}_in_{c}.npz", depth=d.depth, valid=d.mask, K=d.intrinsics,
                        normal=d.normal if d.normal is not None else np.zeros(1),
                        car=masks.get(c, np.zeros(d.depth.shape, bool)), img=cv2.resize(ims[c], (d.depth.shape[1], d.depth.shape[0])))
    fov = np.degrees(2 * np.arctan(d.depth.shape[1] / 2 / d.intrinsics[0, 0]))
    print(c, "depth", d.depth.shape, f"fov {fov:.1f}°", f"car px {int(masks.get(c, np.zeros(1)).sum())}")
