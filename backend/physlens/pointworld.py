"""Point worlds: everything the camera saw, as millions of small coloured 3D points, with each object's points kept
together so its size can be read straight off them.

Every pixel MapAnything sees becomes one point (metres, ground frame: y up, the road or floor at y = 0). An object's
points (SAM 3.1 mask) are stored once per keyframe, so its motion is a sequence of point clouds; the static world
around it is stored once (fixed camera) or fused from every keyframe (moving camera). Points on depth jumps
("flying pixels", which blend an edge with what's behind it) and low-confidence points are left out.

`upsample` > 1 adds points between the measured ones (depth interpolated, colour from the video at that pixel):
denser to look at, but not new measurements. The header says so.

Output, served from /media/pointworld/:  {name}.json (header)  +  {name}.bin:
    positions  int16 × 3 per point, quantised over the scene's bounds (header: origin, step)
    colours    uint8 × 3
    tag        uint8: 255 = static world, k = the object's points at keyframe k
Points are ordered static first, then by keyframe, so each group is one contiguous range (header: groups).
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .geometry.client import backproject
from .vehicle3d import Track3D, read_at

STATIC = 255


def _view_points(view, frame: np.ndarray, T: np.ndarray, scale: float, upsample: int, max_depth: float,
                 jump: float = 0.03) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All good points of one keyframe: (points in ground metres, colours, pixel grid validity) at the output size."""
    h, w = view.depth.shape
    W, H = w * upsample, h * upsample
    z = cv2.resize(view.depth.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) if upsample > 1 else view.depth.astype(np.float32)
    K = view.intrinsics.astype(np.float64).copy()
    K[:2] *= upsample
    if upsample > 1:
        K[0, 2] += (upsample - 1) / 2       # pixel centres move when the grid is refined
        K[1, 2] += (upsample - 1) / 2
    pc = backproject(z, K)
    pose = view.camera_pose
    pw = pc @ pose[:3, :3].T + pose[:3, 3]
    pg = (pw @ T[:3, :3].T + T[:3, 3]) * scale
    grad = np.maximum(np.abs(cv2.Sobel(z, cv2.CV_32F, 1, 0, ksize=3)), np.abs(cv2.Sobel(z, cv2.CV_32F, 0, 1, ksize=3))) / 8
    ok = np.isfinite(pg).all(-1) & (grad < jump * upsample * z) & (z * scale < max_depth)
    if view.confidence is not None:
        c = cv2.resize(view.confidence.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        ok &= c >= np.percentile(c, 15)
    rgb = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
    return pg, rgb, ok


def _sharp_points(al, frame: np.ndarray, T: np.ndarray, scale: float, max_depth: float):
    """The same three outputs from a MoGe-3 view aligned to MapAnything (physlens.sharp.align): real samples at the
    frame's own resolution instead of interpolated ones."""
    H, W = al.valid.shape
    pg = (al.points_world @ T[:3, :3].T + T[:3, 3]) * scale
    ok = al.valid & np.isfinite(pg).all(-1)
    rgb = cv2.cvtColor(cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA) if frame.shape[:2] != (H, W) else frame,
                       cv2.COLOR_BGR2RGB)
    return pg, rgb, ok


def _mask_at(mask: np.ndarray | None, shape: tuple[int, int], dilate: int = 0) -> np.ndarray:
    if mask is None:
        return np.zeros(shape, bool)
    m = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if dilate:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m > 0


def oriented_box(cloud: np.ndarray, heading: np.ndarray | None = None) -> dict:
    """Length × width × height of an object from its points: its direction of travel (or the ground-plane principal
    axis) gives length, across it width, base to top height. Extremes are trimmed at 0.5 % / 99.5 %."""
    H = cloud[:, [0, 2]]
    c = H.mean(0)
    if heading is None or np.linalg.norm(heading) < 1e-9:
        heading = np.linalg.svd(H - c, full_matrices=False)[2][0]
    h = heading / np.linalg.norm(heading)
    side = np.array([-h[1], h[0]])
    a, b = (H - c) @ h, (H - c) @ side
    a0, a1 = np.percentile(a, [0.5, 99.5])
    b0, b1 = np.percentile(b, [0.5, 99.5])
    top = float(np.percentile(cloud[:, 1], 99.5))
    # Height from the ground plane, or from the object's own lowest points when they sit below it (the plane found
    # can be a verge or bank slightly above the road the object stands on).
    base = min(float(np.percentile(cloud[:, 1], 0.5)), 0.0)
    centre = c + h * (a0 + a1) / 2 + side * (b0 + b1) / 2
    return {"centre": [round(float(centre[0]), 3), round((top + base) / 2, 3), round(float(centre[1]), 3)],
            "size": [round(float(a1 - a0), 3), round(top - base, 3), round(float(b1 - b0), 3)],   # length, height, width
            "yaw": round(float(np.arctan2(h[1], h[0])), 4)}


def export(name: str, title: str, video: Path, tr: Track3D, views, masks: list, idx: list[int], out_dir: Path,
           moving_camera: bool, upsample: int = 2, max_depth: float = 60.0, width: int = 960,
           notes: dict | None = None, object_label: str = "object", sharp: list | None = None,
           sharp_object: list | None = None) -> dict:
    """sharp: optional MoGe-3 views aligned to MapAnything (one per keyframe), used instead of MapAnything's own
    518-px depth. Frames are then read at `width` to colour every real sample. sharp_object: the same views with
    the scale fitted around the object (physlens.sharp.align roi=...), used for the object's own points."""
    frames, fps = read_at(video, idx, width)
    T, s = tr.T_ground, tr.scale
    seen = {f: k for k, f in enumerate(tr.frames)}          # video frame → index in the track
    P_all, C_all, tags, groups = [], [], [], []

    # The static world.
    if moving_camera:
        statics = list(range(len(idx)))
    else:
        # One camera: the first keyframe, with the pixels the object hides filled in from a later keyframe.
        statics = [0, max(j for j, f in enumerate(idx) if f in seen)]
    def points_of(j, for_object=False):
        if sharp is not None:
            al = sharp_object[j] if (for_object and sharp_object is not None) else sharp[j]
            pg, rgb, ok = _sharp_points(al, frames[j], T, s, max_depth)
            if for_object:
                ok = ok | (al.valid_loose & np.isfinite(pg).all(-1))
            cam = (T[:3, :3] @ views[j].camera_pose[:3, 3] + T[:3, 3]) * s
            return pg, rgb, ok & (np.linalg.norm(pg - cam, axis=-1) < max_depth)
        return _view_points(views[j], frames[j], T, s, upsample, max_depth)

    covered = None
    for j in statics:
        pg, rgb, ok = points_of(j)
        obj = _mask_at(masks[j], ok.shape, dilate=3 * (1 if sharp is not None else upsample) * max(1, ok.shape[1] // 518))
        keep = ok & ~obj
        if not moving_camera and covered is not None:
            keep &= ~covered                                  # only fill what the first view couldn't see
        if not moving_camera:
            covered = (covered | keep) if covered is not None else keep
        P_all.append(pg[keep]); C_all.append(rgb[keep])
    P = np.concatenate(P_all); C = np.concatenate(C_all)
    if moving_camera:
        # Fusing views repeats the same surfaces: keep one point per 1.5 cm cell.
        cell = np.floor(P / 0.015).astype(np.int64)
        _, first = np.unique(cell, axis=0, return_index=True)
        first.sort()
        P, C = P[first], C[first]
    P_out, C_out = [P], [C]
    tags.append(np.full(len(P), STATIC, np.uint8))
    groups.append({"tag": STATIC, "start": 0, "count": int(len(P))})

    # The object, once per keyframe.
    boxes = []
    heads = np.gradient(tr.p[:, [0, 2]], axis=0) if len(tr.p) > 1 else np.zeros((len(tr.p), 2))
    start = len(P)
    for j, f in enumerate(idx):
        if f not in seen or masks[j] is None:
            continue
        k = seen[f]
        pg, rgb, ok = points_of(j, for_object=True)
        m = ok & _mask_at(cv2.erode(masks[j].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0, ok.shape)
        if m.sum() < 50:
            continue
        cloud = pg[m]
        P_out.append(cloud); C_out.append(rgb[m]); tags.append(np.full(len(cloud), len(boxes), np.uint8))
        groups.append({"tag": len(boxes), "frame": f, "t": round(f / fps, 4), "start": start, "count": int(len(cloud))})
        start += len(cloud)
        boxes.append({"frame": f, "t": round(f / fps, 4), "points": int(len(cloud)), **oriented_box(cloud, heads[k])})
    P = np.concatenate(P_out).astype(np.float64)
    C = np.concatenate(C_out).astype(np.uint8)
    tag = np.concatenate(tags)

    lo, hi = P.min(0), P.max(0)
    step = np.maximum((hi - lo) / 65000, 1e-6)
    q = np.round((P - lo) / step).astype(np.int64) - 32500
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{name}.bin", "wb") as fh:
        fh.write(q.astype("<i2").tobytes()); fh.write(C.tobytes()); fh.write(tag.tobytes())

    cam = views[0].camera_pose
    cam_pos = (T[:3, :3] @ cam[:3, 3] + T[:3, 3]) * s
    fwd = T[:3, :3] @ cam[:3, 2]
    sizes = np.array([b["size"] for b in boxes]) if boxes else np.zeros((0, 3))
    header = {"format": "physlens-pointworld/1", "name": name, "title": title, "video": str(video.name), "fps": fps,
              "points": int(len(P)), "static_points": groups[0]["count"], "object_points": int(len(P) - groups[0]["count"]),
              "upsample": 1 if sharp is not None else upsample,
              "depth_source": "MoGe-3 at full resolution, aligned to MapAnything" if sharp is not None else "MapAnything (518 px)",
              "density_note": ("one measured point per video pixel (MoGe-3 depth, fitted to MapAnything's scale and poses)"
                               if sharp is not None else
                               f"{upsample}× denser than measured: in-between points are interpolated, not new measurements"
                               if upsample > 1 else "one point per measured pixel"),
              "origin": (lo + 32500 * step).round(6).tolist(), "step": step.round(9).tolist(),
              "groups": groups, "object": object_label, "boxes": boxes,
              "object_size_median_m": {"length": round(float(np.median(sizes[:, 0])), 2) if len(sizes) else None,
                                       "height": round(float(np.median(sizes[:, 1])), 2) if len(sizes) else None,
                                       "width": round(float(np.median(sizes[:, 2])), 2) if len(sizes) else None},
              "camera": {"position": cam_pos.round(3).tolist(), "forward": fwd.round(4).tolist(),
                         "fov_deg": round(float(np.degrees(2 * np.arctan(views[0].depth.shape[0] / 2 / views[0].intrinsics[1, 1]))), 1),
                         "moving": moving_camera},
              "scale_from": tr.info.get("scale_from", "MapAnything"), "notes": notes or {}}
    (out_dir / f"{name}.json").write_text(json.dumps(header, indent=1))
    index = out_dir / "index.json"
    items = json.loads(index.read_text()) if index.exists() else []
    items = [i for i in items if i["name"] != name] + [{"name": name, "title": title, "points": header["points"]}]
    index.write_text(json.dumps(items, indent=1))
    return header
