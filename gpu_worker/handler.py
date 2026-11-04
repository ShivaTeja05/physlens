"""RunPod Serverless handler: images in, metric 3D geometry out.

Two tasks:
  (default)  Meta MapAnything (Apache-2.0 weights): many views → depth, camera poses, intrinsics in one shared frame.
  "moge"     Microsoft MoGe-3 ViT-L (MIT): one image at a time → sharp metric depth, normals and a valid-pixel mask at
             the image's own resolution. Pass the field of view MapAnything found (fov_x, degrees) so both agree.

MoGe request  {"input": {"task": "moge", "images": [base64 ...], "fov_x": [deg, ...] or deg (optional),
                         "resolution_level": 0-9 (default 9)}}
MoGe response {"views": [{"npz": base64, "size": [W, H]}, ...]}: depth (float16, metres, 0 where invalid),
              mask (bool), normal (int8 × 3, ×127, OpenCV camera axes), intrinsics (3x3, pixels).

MapAnything:

Request  {"input": {"images": [base64 PNG/JPEG, ...], "size": [width, height],
                    "views": [{"intrinsics": 3x3, "camera_pose": 4x4}, ...]}}   # "views" optional
         Images must already be resized to `size`, both multiples of 14 (the PhysLens client does this).
         Optional per-view extras, when the device knows them (phone/glasses AR tracking):
           intrinsics   pinhole matrix at `size` pixels
           camera_pose  OpenCV cam-to-world (+X right, +Y down, +Z forward), translation in metres
         Poses fix the metric scale; intrinsics fix the field of view.
Response {"views": [{"npz": base64, "model_size": [W, H]}, ...]}
         Each npz holds depth_z (float16, metres), conf (float16), mask (bool), intrinsics (3x3),
         camera_pose (4x4 OpenCV cam-to-world) and metric_scale. Points are rebuilt client-side from depth +
         intrinsics, which keeps responses small.
"""
import base64
import io
import tempfile
from pathlib import Path

import numpy as np
import runpod
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images, preprocess_inputs
from PIL import Image

MODEL_ID = "facebook/map-anything-apache"
model = MapAnything.from_pretrained(MODEL_ID).to("cuda").eval()
_moge = None


def _moge_model():
    """MoGe-3 is loaded on first use, so MapAnything-only workers start as fast as before."""
    global _moge
    if _moge is None:
        from huggingface_hub import hf_hub_download
        from moge.model.v3 import MoGeModel
        _moge = MoGeModel.from_pretrained(hf_hub_download("Ruicheng/moge-3-vitl", "model.pt")).to("cuda").eval()
    return _moge


def moge_job(inp):
    images = [Image.open(io.BytesIO(base64.b64decode(d))).convert("RGB") for d in inp["images"]]
    fov = inp.get("fov_x")
    fovs = fov if isinstance(fov, list) else [fov] * len(images)
    level = int(inp.get("resolution_level", 9))
    m = _moge_model()
    out = []
    for img, fx in zip(images, fovs):
        t = torch.tensor(np.asarray(img, dtype=np.float32) / 255, device="cuda").permute(2, 0, 1)
        with torch.no_grad():
            r = m.infer(t, fov_x=fx, resolution_level=level, use_fp16=True)
        depth = r["depth"].float().cpu().numpy()
        mask = r["mask"].cpu().numpy().astype(bool) if "mask" in r else np.isfinite(depth)
        depth = np.where(mask & np.isfinite(depth), depth, 0).astype(np.float16)
        H, W = depth.shape
        K = r["intrinsics"].float().cpu().numpy().astype(np.float64)
        K[0] *= W
        K[1] *= H
        packed = {"depth": depth, "mask": mask, "intrinsics": K}
        if r.get("normal") is not None:
            n = np.nan_to_num(r["normal"].float().cpu().numpy())
            packed["normal"] = np.clip(np.round(n * 127), -127, 127).astype(np.int8)
        buf = io.BytesIO()
        np.savez_compressed(buf, **packed)
        out.append({"npz": base64.b64encode(buf.getvalue()).decode(), "size": [W, H]})
    return {"views": out}


def _views_with_extras(images: list[Image.Image], extras: list[dict]) -> list[dict]:
    raw = []
    for img, ex in zip(images, extras):
        v = {"img": torch.from_numpy(np.array(img, dtype=np.uint8))}
        if ex.get("intrinsics") is not None:
            v["intrinsics"] = torch.tensor(ex["intrinsics"], dtype=torch.float32)
        if ex.get("camera_pose") is not None:
            v["camera_poses"] = torch.tensor(ex["camera_pose"], dtype=torch.float32)
            v["is_metric_scale"] = torch.tensor([True])
        raw.append(v)
    return preprocess_inputs(raw)


def handler(job):
    inp = job["input"]
    if inp.get("task") == "moge":
        return moge_job(inp)
    size = tuple(int(v) for v in inp["size"])
    if any(v % 14 for v in size):
        return {"error": f"size {size} must be multiples of 14"}
    images = [Image.open(io.BytesIO(base64.b64decode(d))).convert("RGB") for d in inp["images"]]
    extras = inp.get("views")
    if extras is not None and len(extras) != len(images):
        return {"error": "views must have one entry per image"}

    if extras:
        views = _views_with_extras(images, extras)
    else:
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, img in enumerate(images):
                p = Path(tmp) / f"{i:04d}.png"
                img.save(p)
                paths.append(str(p))
            views = load_images(paths, resize_mode="fixed_size", size=size)
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                            apply_mask=True, mask_edges=True)

    out = []
    for pred in preds:
        arr = lambda k: pred[k][0].float().cpu().numpy()
        depth = arr("depth_z")[..., 0]
        packed = {
            "depth_z": depth.astype(np.float16),
            "intrinsics": arr("intrinsics").astype(np.float64),
            "camera_pose": arr("camera_poses").astype(np.float64),
            "metric_scale": np.float64(pred["metric_scaling_factor"][0]) if "metric_scaling_factor" in pred else np.nan,
        }
        if "conf" in pred:
            packed["conf"] = arr("conf").squeeze().astype(np.float16)
        if "mask" in pred:
            packed["mask"] = arr("mask").squeeze().astype(bool)
        buf = io.BytesIO()
        np.savez_compressed(buf, **packed)
        out.append({"npz": base64.b64encode(buf.getvalue()).decode(), "model_size": [depth.shape[1], depth.shape[0]]})
    return {"views": out}


runpod.serverless.start({"handler": handler})
