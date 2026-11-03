"""M1: measure static things in metres from ONE ordinary video frame.

Ground truth: the 1.000 m tape on the wall in the real classroom clip, whose endpoints the teacher marked in
their Tracker file. MapAnything (Modal GPU) gives metric 3D for every pixel; the tape's length is the distance
between its two endpoints in 3D. Bonus: SAM 3.1 outlines a door, and we read off its height.

  cd backend && .venv/bin/python ../benchmark/geometry_m1.py
"""
from __future__ import annotations

import base64
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from physlens.config import load_env  # noqa: E402

load_env()
from physlens import budget  # noqa: E402
from physlens.geometry.client import reconstruct  # noqa: E402

VIDEO = HERE.parent / "samples/real-classroom-drop.mp4"
TRK = HERE.parent / "real_clips/compadre-bouncing-ball/extracted/BouncingBall_ballbouncelookang01_x264.trk"


def tape_endpoints() -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(TRK).getroot()
    num = lambda el, n: float(el.find(f"property[@name='{n}']").text)
    tape = next(o for o in root.iter("object") if o.get("class", "").endswith("TapeMeasure$FrameData"))
    return np.array([num(tape, "x1"), num(tape, "y1")]), np.array([num(tape, "x2"), num(tape, "y2")])


def frame(index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    assert ok
    return img


def sam_image_mask(img: np.ndarray, prompt: str):
    """SAM 3.1 on one image (inline data URL). Returns the largest mask as a full-size boolean array, or None."""
    import os

    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
    ok, png = cv2.imencode(".png", img)
    url = "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()
    budget.check(budget.SAM_PER_IMAGE)
    r = client.responses.create(model="sam-3.1", input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt}, {"type": "input_image", "image_url": url}]}],
        metadata={"mask_encoding": "one_bit"})
    budget.record("sam-3.1", budget.SAM_PER_IMAGE, what=f"image: {prompt}")
    parser = image_segmentation_format().create_parser()
    for _ in parser.push(r.output_text or ""):
        pass
    masks = [rec for rec in parser.finish(CompletedOutcome()).result.records if rec.kind == "mask"]
    if not masks:
        return None
    best = max(masks, key=lambda m: m.mask.width * m.mask.height)
    raster = np.frombuffer(decode_mask_to_raster(best.mask), np.uint8).reshape(best.mask.height, best.mask.width)
    full = np.zeros(img.shape[:2], bool)
    b = best.bounds
    full[b.top:b.top + raster.shape[0], b.left:b.left + raster.shape[1]] = raster > 0
    return full


def content_rows(img: np.ndarray) -> tuple[int, int]:
    """Rows that hold picture rather than baked-in black letterbox bars."""
    bright = img.mean(axis=(1, 2)) > 12
    rows = np.nonzero(bright)[0]
    return int(rows[0]), int(rows[-1]) + 1


def object_height(view, mask: np.ndarray, y_offset: int) -> float:
    """Height of an upright object, robust to its edges.

    Depth right on an outline often belongs to the wall behind, so: take the object's typical depth from its
    eroded interior, measure its pixel height between the 2nd and 98th percentile rows, and convert with the
    camera's vertical focal length (rescaled from model pixels to original pixels)."""
    interior = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    if interior.sum() < 50:
        interior = mask
    ys, xs = np.nonzero(interior)
    depths = [view.point_at(x, y - y_offset)[2] for x, y in zip(xs[::7], ys[::7])]
    z = float(np.median(depths))
    rows = np.nonzero(mask.any(axis=1))[0]
    top, bottom = np.percentile(rows, 2), np.percentile(rows, 98)
    fy = view.intrinsics[1, 1] / view.scale_y
    return float((bottom - top) * z / fy)


def main() -> None:
    img = frame(40)
    y0, y1 = content_rows(img)
    print(f"picture rows {y0}-{y1} of {img.shape[0]} (cropping baked-in black bars); running MapAnything…", flush=True)
    view = reconstruct([img[y0:y1]], timeout_s=1800)[0]
    a, b = tape_endpoints()
    pa, pb = view.point_at(a[0], a[1] - y0), view.point_at(b[0], b[1] - y0)
    tape = float(np.linalg.norm(pa - pb))
    print(f"\nTAPE  true 1.000 m · measured {tape:.3f} m · error {100 * (tape - 1):+.1f}%")
    print(f"      camera-to-wall distance ≈ {float(np.mean([pa[2], pb[2]])):.2f} m")

    results = {"tape_true_m": 1.0, "tape_measured_m": tape, "wall_distance_m": float(np.mean([pa[2], pb[2]]))}
    for prompt in ["door", "person"]:
        mask = sam_image_mask(img, prompt)
        if mask is None:
            print(f"{prompt.upper():5} SAM found no match")
            continue
        h = object_height(view, mask, y0)
        results[f"{prompt}_height_m"] = h
        print(f"{prompt.upper():5} height ≈ {h:.2f} m  (SAM mask {int(mask.sum())} px)")
    (HERE / "geometry_m1.json").write_text(json.dumps(results, indent=2))
    print("\nspend:", budget.summary())


if __name__ == "__main__":
    main()
