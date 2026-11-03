"""MapAnything on a Modal GPU: photos or video frames in, metric 3D geometry out.

Deploy once:   .venv/bin/modal deploy physlens/geometry/modal_app.py
Then call it from anywhere with physlens.geometry.client.reconstruct(...).

Uses Meta's Apache-2.0 checkpoint (facebook/map-anything-apache). The weights are cached in a Modal Volume, so
only the first cold start downloads them.
"""
from __future__ import annotations

import io

import modal

MODEL_ID = "facebook/map-anything-apache"
REPO = "https://github.com/facebookresearch/map-anything.git"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch==2.6.0", "torchvision==0.21.0", "numpy<2.3", "pillow", "huggingface_hub")
    .run_commands(f"git clone --depth 1 {REPO} /opt/map-anything", "cd /opt/map-anything && pip install -e .")
    .env({"HF_HOME": "/cache/hf"})
)
cache = modal.Volume.from_name("physlens-hf-cache", create_if_missing=True)
app = modal.App("physlens-geometry", image=image)


@app.cls(gpu="L4", volumes={"/cache": cache}, timeout=900, scaledown_window=120)
class Geometry:
    @modal.enter()
    def load(self):
        import torch
        from mapanything.models import MapAnything

        self.torch = torch
        self.model = MapAnything.from_pretrained(MODEL_ID).to("cuda").eval()
        cache.commit()

    @modal.method()
    def reconstruct(self, images: list[bytes], size: tuple[int, int]) -> list[dict]:
        """`images` are PNG/JPEG bytes already resized to `size` = (width, height), both multiples of 14.
        Returns one dict per view with metric depth, camera-frame points, intrinsics and cam-to-world pose."""
        import tempfile
        from pathlib import Path

        import numpy as np
        from mapanything.utils.image import load_images
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, data in enumerate(images):
                p = Path(tmp) / f"{i:04d}.png"
                Image.open(io.BytesIO(data)).convert("RGB").save(p)
                paths.append(str(p))
            views = load_images(paths, resize_mode="fixed_size", size=size)
            with self.torch.no_grad():
                preds = self.model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype="bf16",
                                         apply_mask=True, mask_edges=True)

        out = []
        for pred in preds:
            arr = lambda k: pred[k][0].float().cpu().numpy()
            out.append({
                "depth_z": arr("depth_z")[..., 0].astype(np.float32),        # metres, (H, W)
                "pts3d_cam": arr("pts3d_cam").astype(np.float32),            # metres, camera frame (H, W, 3)
                "pts3d": arr("pts3d").astype(np.float32),                    # metres, world frame (H, W, 3)
                "conf": arr("conf").astype(np.float32) if "conf" in pred else None,
                "mask": arr("mask")[..., 0].astype(bool) if "mask" in pred else None,
                "intrinsics": arr("intrinsics").astype(np.float64),          # (3, 3) at model resolution
                "camera_pose": arr("camera_poses").astype(np.float64),       # (4, 4) OpenCV cam-to-world
                "metric_scale": float(pred["metric_scaling_factor"][0]) if "metric_scaling_factor" in pred else None,
                "model_size": tuple(int(v) for v in arr("pts3d").shape[1::-1]),  # (W, H)
            })
        return out
