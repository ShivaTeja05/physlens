"""Sharper point clouds: MoGe-3's full-resolution depth, put into MapAnything's world.

MapAnything is the backbone: one shared frame for every view, camera poses, metric scale, all at 518 px. MoGe-3
looks at each frame on its own at the video's resolution and draws much finer, cleaner geometry, but in its own
scale. Per frame, MoGe's depth is fitted to MapAnything's (one scale factor, robust, on pixels both trust), then
every MoGe pixel is lifted with MapAnything's intrinsics (rescaled to full size) and camera pose.

Flying pixels are dropped where MoGe's depth jumps between neighbours or its surface is seen almost edge-on (normal
nearly at right angles to the line of sight) — except where the surface is genuinely smooth, which keeps grazing
roofs and roads.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry.client import SharpDepth, backproject


@dataclass
class Aligned:
    points_world: np.ndarray     # (H, W, 3) MapAnything world frame
    valid: np.ndarray            # (H, W) bool
    valid_loose: np.ndarray      # (H, W) bool: depth jumps removed, but surfaces seen edge-on kept (object clouds)
    scale: float                 # MapAnything depth ÷ MoGe depth
    agree: float                 # median relative difference after alignment, on shared pixels
    K: np.ndarray                # intrinsics used (full resolution)


def fov_x(view) -> float:
    w = view.depth.shape[1]
    return float(np.degrees(2 * np.arctan(w / 2 / view.intrinsics[0, 0])))


def align(view, sharp: SharpDepth, jump: float = 0.04, grazing_deg: float = 80.0, roi: np.ndarray | None = None) -> Aligned:
    """roi: optional (H, W) bool region to fit the scale on (an object and its surroundings). Monocular depth bends
    a little with distance, so a scale fitted on the whole picture (mostly far background) can sit several percent
    off at a nearby object; fitted around the object, it carries MapAnything's scale onto the object itself."""
    H, W = sharp.depth.shape
    # MapAnything depth at full resolution (nearest: compare like with like, no blended edges).
    zm = cv2.resize(view.depth.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    conf = None
    if view.confidence is not None:
        conf = cv2.resize(view.confidence.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    zs = sharp.depth
    both = sharp.mask & (zs > 0) & np.isfinite(zm) & (zm > 0)
    if conf is not None:
        both &= conf >= np.percentile(conf[both], 40)
    # Away from edges of either map, where both are reliable.
    gm = np.abs(cv2.Laplacian(zm, cv2.CV_32F)) < 0.02 * zm
    sel = both & gm
    if roi is not None and (sel & roi).sum() > 500:
        sel = sel & roi
    ratio = zm[sel] / zs[sel]
    s = float(np.median(ratio))
    for _ in range(2):                                   # trim and refit
        keep = np.abs(np.log(ratio / s)) < 0.15
        s = float(np.median(ratio[keep]))
    z = zs * s
    agree = float(np.median(np.abs(z[sel] - zm[sel]) / zm[sel]))
    # MapAnything's lens, at full resolution.
    K = view.intrinsics.astype(np.float64).copy()
    K[0] *= W / view.depth.shape[1]
    K[1] *= H / view.depth.shape[0]
    pc = backproject(z, K)
    # Flying pixels: depth jumps, unless the surface is smooth (second difference small).
    gx = np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)) / 8
    gy = np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3)) / 8
    lap = np.abs(cv2.Laplacian(z, cv2.CV_32F))
    edge = (np.maximum(gx, gy) > jump * z) & (lap > 0.5 * jump * z)
    valid = sharp.mask & (z > 0) & ~edge
    valid_loose = valid.copy()
    if sharp.normal is not None:
        rays = pc / np.maximum(np.linalg.norm(pc, axis=-1, keepdims=True), 1e-9)
        cosang = np.abs((rays * sharp.normal).sum(-1))
        valid &= (cosang > np.cos(np.radians(grazing_deg))) | (lap < 0.1 * jump * z)
    pose = view.camera_pose
    pw = pc @ pose[:3, :3].T + pose[:3, 3]
    return Aligned(pw, valid, valid_loose, s, agree, K)


def main_depth_cluster(points_cam_z: np.ndarray, gap: float = 0.6) -> np.ndarray:
    """Pixels of one object belong to one stretch of depth; a mask that spills onto the background behind it picks
    up points metres further away, separated by a gap. Keep the stretch holding the most pixels."""
    z = points_cam_z
    order = np.argsort(z)
    zs = z[order]
    cuts = np.nonzero(np.diff(zs) > gap)[0]
    starts, ends = np.r_[0, cuts + 1], np.r_[cuts + 1, len(zs)]
    k = int(np.argmax(ends - starts))
    keep = np.zeros(len(z), bool)
    keep[order[starts[k]:ends[k]]] = True
    return keep
