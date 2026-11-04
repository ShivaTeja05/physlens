"""Calls the MapAnything GPU worker and maps its outputs back to original image pixels.

The worker runs on RunPod Serverless (RUNPOD_ENDPOINT_ID + RUNPOD_API_KEY in backend/.env, see gpu_worker/).
Images are resized here to a size MapAnything accepts (multiples of
14, about 518 px wide), so original pixels map to model pixels by a plain scale per axis, with no hidden crop.
"""
from __future__ import annotations

import base64
import io
import os
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

PATCH = 14
RUNPOD_API = "https://api.runpod.ai/v2"


@dataclass
class ViewGeometry:
    depth: np.ndarray        # metres, model resolution (H, W)
    points_cam: np.ndarray   # metres, camera frame (H, W, 3)
    points_world: np.ndarray
    confidence: np.ndarray | None
    valid: np.ndarray | None
    intrinsics: np.ndarray
    camera_pose: np.ndarray  # OpenCV cam-to-world
    scale_x: float           # model px per original px
    scale_y: float

    def point_at(self, x_px: float, y_px: float, world: bool = False) -> np.ndarray:
        """3D point (metres) under an original-image pixel."""
        u = int(np.clip(round(x_px * self.scale_x), 0, self.depth.shape[1] - 1))
        v = int(np.clip(round(y_px * self.scale_y), 0, self.depth.shape[0] - 1))
        return (self.points_world if world else self.points_cam)[v, u]


def model_size(width: int, height: int, target_width: int = 518) -> tuple[int, int]:
    w = max(PATCH, round(target_width / PATCH) * PATCH)
    h = max(PATCH, round(target_width * height / width / PATCH) * PATCH)
    return w, h


def backproject(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Camera-frame points from Z-depth and pinhole intrinsics (OpenCV: x right, y down, z forward)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def _payload(images, target_width):
    pil = [im if isinstance(im, Image.Image) else Image.fromarray(im[..., ::-1]) for im in images]  # BGR → RGB
    w0, h0 = pil[0].size
    size = model_size(w0, h0, target_width)
    encoded = []
    for im in pil:
        buf = io.BytesIO()
        im.convert("RGB").resize(size, Image.LANCZOS).save(buf, format="PNG")
        encoded.append(buf.getvalue())
    return encoded, size, (w0, h0)


def _runpod(encoded: list[bytes], size: tuple[int, int], timeout_s: float = 600,
            extras: list[dict] | None = None) -> list[dict]:
    import httpx

    endpoint, key = os.environ["RUNPOD_ENDPOINT_ID"], os.environ["RUNPOD_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}
    body = {"input": {"images": [base64.b64encode(b).decode() for b in encoded], "size": list(size)}}
    if extras:
        body["input"]["views"] = extras
    with httpx.Client(timeout=120) as http:
        r = http.post(f"{RUNPOD_API}/{endpoint}/runsync", json=body, headers=headers)
        if r.status_code == 409:
            # The endpoint is mid-update (a new release rolling out); wait briefly and try once more.
            time.sleep(20)
            r = http.post(f"{RUNPOD_API}/{endpoint}/runsync", json=body, headers=headers)
        if r.status_code >= 300:
            raise RuntimeError(f"RunPod HTTP {r.status_code}: {r.text[:300]}")
        job = r.json()
        deadline = time.time() + timeout_s
        # runsync returns early while a cold worker starts; keep polling the job until it finishes.
        while job.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() < deadline:
            time.sleep(2)
            job = http.get(f"{RUNPOD_API}/{endpoint}/status/{job['id']}", headers=headers).json()
    if job.get("status") != "COMPLETED":
        raise RuntimeError(f"RunPod job ended with {job.get('status')}: {str(job.get('error') or job)[:300]}")
    output = job["output"]
    if "error" in output:
        raise RuntimeError(f"GPU worker error: {output['error']}")
    views = []
    for v in output["views"]:
        z = np.load(io.BytesIO(base64.b64decode(v["npz"])))
        depth = z["depth_z"].astype(np.float32)
        K, pose = z["intrinsics"], z["camera_pose"]
        pts_cam = backproject(depth, K)
        pts_world = pts_cam @ pose[:3, :3].T + pose[:3, 3]
        views.append({"depth_z": depth, "pts3d_cam": pts_cam, "pts3d": pts_world,
                      "conf": z["conf"].astype(np.float32) if "conf" in z else None,
                      "mask": z["mask"] if "mask" in z else None, "intrinsics": K, "camera_pose": pose,
                      "model_size": tuple(v["model_size"])})
    return views


def _require_worker() -> None:
    if not (os.environ.get("RUNPOD_ENDPOINT_ID") and os.environ.get("RUNPOD_API_KEY")):
        raise RuntimeError("3D needs the GPU worker: set RUNPOD_ENDPOINT_ID and RUNPOD_API_KEY in backend/.env "
                           "(see gpu_worker/README.md)")


def reconstruct(images: list[np.ndarray | Image.Image], target_width: int = 518, timeout_s: float = 600,
                intrinsics: list[np.ndarray] | None = None,
                camera_poses: list[np.ndarray] | None = None) -> list[ViewGeometry]:
    """Metric 3D for one photo, or for several views of the same scene (video frames, two viewpoints).

    Optional device knowledge (phone or glasses AR tracking), one per image:
      intrinsics    3x3 pinhole matrix in ORIGINAL image pixels (rescaled here to the model size)
      camera_poses  4x4 OpenCV cam-to-world, translation in metres. These fix the metric scale.
    `timeout_s` covers a cold start of the GPU worker."""
    encoded, size, (w0, h0) = _payload(images, target_width)
    extras = None
    if intrinsics is not None or camera_poses is not None:
        sx, sy = size[0] / w0, size[1] / h0
        extras = []
        for i in range(len(encoded)):
            ex = {}
            if intrinsics is not None:
                K = np.array(intrinsics[i], float).copy()
                K[0, :] *= sx
                K[1, :] *= sy
                ex["intrinsics"] = K.tolist()
            if camera_poses is not None:
                ex["camera_pose"] = np.array(camera_poses[i], float).tolist()
            extras.append(ex)
    _require_worker()
    results = _runpod(encoded, size, timeout_s, extras)
    views = []
    for r in results:
        mw, mh = r["model_size"]
        views.append(ViewGeometry(r["depth_z"], r["pts3d_cam"], r["pts3d"], r["conf"], r["mask"], r["intrinsics"],
                                  r["camera_pose"], scale_x=mw / w0, scale_y=mh / h0))
    return views


# --- MoGe-3: sharp per-frame depth at the image's own resolution ----------------------------------------------

@dataclass
class SharpDepth:
    depth: np.ndarray            # metres (MoGe's own scale), 0 where invalid, original image size (H, W)
    mask: np.ndarray             # bool, valid pixels
    normal: np.ndarray | None    # (H, W, 3) unit normals, OpenCV camera axes
    intrinsics: np.ndarray       # 3x3, pixels


def sharp_depth(images: list[np.ndarray], fov_x: list[float] | float | None = None, batch: int = 6,
                timeout_s: float = 900, resolution_level: int = 9) -> list[SharpDepth]:
    """MoGe-3 on the RunPod worker, a few images per job (each image's maps are about a megabyte compressed).
    `fov_x` (degrees): give MapAnything's field of view so both models use the same lens."""
    import httpx

    _require_worker()

    endpoint, key = os.environ["RUNPOD_ENDPOINT_ID"], os.environ["RUNPOD_API_KEY"]
    headers = {"Authorization": f"Bearer {key}"}
    fovs = fov_x if isinstance(fov_x, list) else [fov_x] * len(images)
    out: list[SharpDepth] = []
    for lo in range(0, len(images), batch):
        enc = []
        for im in images[lo:lo + batch]:
            pil = im if isinstance(im, Image.Image) else Image.fromarray(im[..., ::-1])
            buf = io.BytesIO()
            pil.convert("RGB").save(buf, format="JPEG", quality=95)
            enc.append(base64.b64encode(buf.getvalue()).decode())
        body = {"input": {"task": "moge", "images": enc, "fov_x": fovs[lo:lo + batch], "resolution_level": resolution_level}}
        with httpx.Client(timeout=120) as http:
            r = http.post(f"{RUNPOD_API}/{endpoint}/runsync", json=body, headers=headers)
            if r.status_code == 409:
                time.sleep(20)
                r = http.post(f"{RUNPOD_API}/{endpoint}/runsync", json=body, headers=headers)
            if r.status_code >= 300:
                raise RuntimeError(f"RunPod HTTP {r.status_code}: {r.text[:300]}")
            job = r.json()
            deadline = time.time() + timeout_s
            while job.get("status") in ("IN_QUEUE", "IN_PROGRESS") and time.time() < deadline:
                time.sleep(2)
                job = http.get(f"{RUNPOD_API}/{endpoint}/status/{job['id']}", headers=headers).json()
        if job.get("status") != "COMPLETED":
            raise RuntimeError(f"RunPod job ended with {job.get('status')}: {str(job.get('error') or job)[:300]}")
        if "error" in job["output"]:
            raise RuntimeError(f"GPU worker error: {job['output']['error']}")
        for v in job["output"]["views"]:
            z = np.load(io.BytesIO(base64.b64decode(v["npz"])))
            n = z["normal"].astype(np.float32) / 127 if "normal" in z else None
            out.append(SharpDepth(z["depth"].astype(np.float32), z["mask"].astype(bool), n, z["intrinsics"]))
    return out
