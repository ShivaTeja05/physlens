"""Vehicles (and any large object) tracked in 3D from real footage, fixed or moving camera.

    keyframes → MapAnything (depth + camera poses in one shared frame)  +  SAM 3.1 mask of the object per keyframe
    → the object's 3D points (MapAnything points inside its mask) → its position (median point) per keyframe
    → a gravity-aligned ground frame (road plane) → one known size fixes the scale → speeds, stops, decelerations.

Unlike a small ball, a vehicle covers thousands of depth pixels, so its depth is measured directly rather than
inferred from its apparent size. The median of the visible surface sits a little off the true centre and shifts
slightly as the view angle changes; that is part of the stated error.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .moving_camera import floor_frame


def read_at(video: Path, idx: list[int], width: int = 960) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out = []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, f = cap.read()
        if not ok:
            raise ValueError(f"no frame {i} in {video}")
        h, w = f.shape[:2]
        out.append(cv2.resize(f, (width, round(h * width / w)), interpolation=cv2.INTER_AREA) if w != width else f)
    return out, fps


def geometry(frames: list[np.ndarray], cache: Path):
    from .geometry.client import ViewGeometry, reconstruct
    if cache.exists():
        z = np.load(cache)
        return [ViewGeometry(z[f"depth{j}"], z[f"cam{j}"], z[f"world{j}"], z[f"conf{j}"], None, z[f"K{j}"], z[f"pose{j}"],
                             float(z[f"sx{j}"]), float(z[f"sy{j}"])) for j in range(len(frames))]
    views = reconstruct(frames, timeout_s=900)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **{k: v for j, vw in enumerate(views) for k, v in {
        f"depth{j}": vw.depth, f"cam{j}": vw.points_cam, f"world{j}": vw.points_world,
        f"conf{j}": vw.confidence if vw.confidence is not None else np.ones_like(vw.depth),
        f"K{j}": vw.intrinsics, f"pose{j}": vw.camera_pose, f"sx{j}": vw.scale_x, f"sy{j}": vw.scale_y}.items()})
    return views


def masks(frames: list[np.ndarray], prompt: str, cache: Path) -> list[np.ndarray | None]:
    """One SAM 3.1 mask per frame: the instance nearest the previous one (largest in the first frame)."""
    from .perception.sam_image import segment
    cache.mkdir(parents=True, exist_ok=True)
    out, prev = [], None
    for j, f in enumerate(frames):
        p = cache / f"{j:03d}.png"
        if p.exists():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) > 0
            out.append(m if m.any() else None)
        else:
            inst = [i.mask for i in segment(f, prompt) if i.mask.sum() > 200]
            m = None
            if inst:
                if prev is None:
                    m = max(inst, key=lambda a: a.sum())
                else:
                    c0 = np.argwhere(prev).mean(0)
                    m = min(inst, key=lambda a: np.linalg.norm(np.argwhere(a).mean(0) - c0))
            cv2.imwrite(str(p), (m if m is not None else np.zeros(f.shape[:2], bool)).astype(np.uint8) * 255)
            out.append(m)
        if out[-1] is not None:
            prev = out[-1]
    return out


def points_in(view, mask: np.ndarray, erode: int = 3, jump: float = 0.03) -> np.ndarray:
    """MapAnything world points under a mask. Left out: the mask's edge (eroded), and "flying pixels" — pixels on a
    depth jump, whose depth blends the object with what's behind it and smears the object into a long tail."""
    h, w = view.depth.shape
    m = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
    if erode:
        m = cv2.erode(m, np.ones((erode, erode), np.uint8))
    z = view.depth.astype(np.float32)
    grad = np.maximum(np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))) / 8
    ok = (m > 0) & (grad < jump * z)
    if view.confidence is not None:
        c = view.confidence
        ok &= c >= np.percentile(c[m > 0], 20) if (m > 0).any() else True
    P = view.points_world[ok]
    return P[np.isfinite(P).all(1)]


@dataclass
class Track3D:
    frames: list[int]
    t: np.ndarray
    p: np.ndarray                    # (n, 3) median visible point, ground frame (y up), metres after scaling
    clouds: list[np.ndarray]         # per-frame object points, ground frame
    T_ground: np.ndarray             # MapAnything world → ground frame
    scale: float = 1.0
    info: dict = field(default_factory=dict)


def track(video: Path, idx: list[int], prompt: str, cache: Path, width: int = 960) -> tuple[Track3D, list, list]:
    frames, fps = read_at(video, idx, width)
    views = geometry(frames, cache / "mapanything.npz")
    ms = masks(frames, prompt, cache / f"sam_{prompt.replace(' ', '_')}")
    T, finfo = floor_frame(views, [v.camera_pose for v in views])
    keep, P, clouds = [], [], []
    for j, (v, m) in enumerate(zip(views, ms)):
        if m is None:
            continue
        pts = points_in(v, m)
        if len(pts) < 30:
            continue
        g = pts @ T[:3, :3].T + T[:3, 3]
        keep.append(j)
        clouds.append(g)
        P.append(np.median(g, axis=0))
    info = {**finfo, "keyframes": len(idx), "seen": len(keep), "prompt": prompt}
    tr = Track3D([idx[j] for j in keep], np.array([idx[j] for j in keep]) / fps, np.array(P), clouds, T, info=info)
    return tr, frames, views


def rescale(tr: Track3D, factor: float, why: str) -> None:
    tr.p = tr.p * factor
    tr.clouds = [c * factor for c in tr.clouds]
    tr.scale *= factor
    tr.info["scale_from"] = why
    tr.info["scale_applied"] = round(tr.scale, 4)


def speeds(tr: Track3D, window: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal speed (m/s) between keyframes `window` apart (central), with the time at the middle."""
    H = tr.p[:, [0, 2]]
    v, tm = [], []
    for i in range(len(H) - window):
        v.append(np.linalg.norm(H[i + window] - H[i]) / (tr.t[i + window] - tr.t[i]))
        tm.append((tr.t[i + window] + tr.t[i]) / 2)
    return np.array(tm), np.array(v)


def save(tr: Track3D, out: Path, extra: dict) -> None:
    out.write_text(json.dumps({"frames": tr.frames, "t": tr.t.round(4).tolist(), "p": tr.p.round(4).tolist(),
                               "info": tr.info, **extra}, indent=1))
