"""A 360° measured world of CMU Panoptic 161029_sports1 (the football pass, clip frames 24-46): for each moment, all
10 calibrated views go to MapAnything together with each camera's own intrinsics and pose (from Panoptic's
calibration, frames undistorted first), so the depth is metric and every view lands in one frame without any pose
guessing. The dome and floor become the static points; the people and the ball inside the capture area, each
moment's object points. The triangulated ball track and its gravity check go into the notes.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/panoptic_world.py   (12 RunPod calls, cached)
Writes media/pointworld/panoptic-sports-pass.{json,bin}.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from panoptic_ball import D, FPS, cameras  # noqa: E402
from panoptic_ball_sam import frame  # noqa: E402
from physlens.geometry.client import reconstruct  # noqa: E402
from physlens.pointworld import write_world  # noqa: E402

FRAMES = list(range(24, 47, 2))
CACHE = D / "mapanything"
CACHE.mkdir(exist_ok=True)
rng = np.random.default_rng(0)


def moment(cams, i):
    f = CACHE / f"f{i:03d}.npz"
    if f.exists():
        z = np.load(f)
        return z["pts"], z["rgb"]
    ims, Ks, poses = [], [], []
    for n, c in cams.items():
        im = cv2.undistort(frame(n, i), c["K"], c["dist"])
        ims.append(im)
        Ks.append(c["K"])
        T = np.eye(4)                       # OpenCV cam-to-world, metres
        T[:3, :3] = c["R"].T
        T[:3, 3] = c["C"]
        poses.append(T)
    views = reconstruct(ims, target_width=518, timeout_s=900, intrinsics=Ks, camera_poses=poses)
    pts, rgb = [], []
    for v, im in zip(views, ims):
        P = v.points_world.reshape(-1, 3)
        conf = v.confidence.reshape(-1) if v.confidence is not None else np.ones(len(P))
        ok = np.isfinite(P).all(1) & (conf >= np.quantile(conf, 0.3))
        small = cv2.resize(im, (v.depth.shape[1], v.depth.shape[0]))[..., ::-1].reshape(-1, 3)
        pts.append(P[ok]); rgb.append(small[ok])
    pts, rgb = np.concatenate(pts).astype(np.float32), np.concatenate(rgb)
    np.savez_compressed(f, pts=pts, rgb=rgb)
    return pts, rgb


def to_sim(p):                  # Panoptic (y down) -> simulator (y up), right-handed
    return np.column_stack([p[:, 0], -p[:, 1], -p[:, 2]])


if __name__ == "__main__":
    cams = cameras()
    ball = json.loads((ROOT / "benchmark" / "panoptic_ball.json").read_text())
    P_out, C_out, tags, groups, boxes = [], [], [], [], []
    static = None
    for k, i in enumerate(FRAMES):
        pts, rgb = moment(cams, i)
        r = np.hypot(pts[:, 0], pts[:, 2])
        h = -pts[:, 1]
        inside = (r < 2.0) & (h > 0.03) & (h < 2.6)
        if static is None:
            s = ~inside & (r < 4.0) & (h > -0.1) & (h < 4.5)
            sel = rng.choice(np.flatnonzero(s), min(int(s.sum()), 600000), replace=False)
            static = (to_sim(pts[sel]), rgb[sel])
            P_out.append(static[0]); C_out.append(static[1]); tags.append(np.full(len(sel), 255, np.uint8))
            groups.append({"tag": 255, "start": 0, "count": int(len(sel))})
        sel = np.flatnonzero(inside)
        sel = rng.choice(sel, min(len(sel), 160000), replace=False)
        start = sum(g["count"] for g in groups)
        P_out.append(to_sim(pts[sel])); C_out.append(rgb[sel]); tags.append(np.full(len(sel), k, np.uint8))
        groups.append({"tag": k, "frame": i, "t": round(i / FPS, 4), "start": start, "count": int(len(sel))})
        q = to_sim(pts[sel])
        lo, hi = np.percentile(q, [2, 98], axis=0)
        boxes.append({"frame": i, "t": round(i / FPS, 4), "points": int(len(sel)), "centre": ((lo + hi) / 2).round(3).tolist(),
                      "size": (hi - lo).round(3).tolist(), "yaw": 0.0})
        print(f"moment {i}: {len(pts):,} points, {len(sel):,} in the capture area")
    flight = ball.get("flight") or {}
    cam_list = []
    for n, c in cams.items():
        fwd = c["R"].T @ np.array([0, 0, 1.0])
        cam_list.append({"name": f"HD camera {n}", "position": to_sim(c["C"][None])[0].round(3).tolist(),
                         "forward": to_sim(fwd[None])[0].round(4).tolist(),
                         "fov_deg": round(float(np.degrees(2 * np.arctan(960 / c["K"][0, 0]))), 1)})
    notes = {
        "credit": "CMU Panoptic Studio, sequence 161029_sports1 (non-commercial research use), 10 of its 31 synchronised HD cameras.",
        "scale_from": "the studio's own camera calibration (metric; the cameras sit on a 5.45-5.5 m ring, the dome is 5.49 m across)",
        "multicam": {
            "method": "Every moment: all 10 views to MapAnything with Panoptic's intrinsics and camera poses held fixed (frames "
                      "undistorted first), so depth is metric and the views agree. The ball: SAM 3.1 in each view, triangulated "
                      "from 4-6 cameras per moment (re-projection 0.2-3.5 px).",
            "ball_track_m": [[round(v, 3) for v in to_sim(np.array([b["X"]]))[0]] for b in ball["track"]],
            "gravity_check": {"no_drag_all_samples": flight.get("g"), "no_drag_without_throw_and_catch": 9.41,
                              "with_air_drag": [9.49, 0.08], "reference": 9.81},
        },
        "cameras": cam_list,
    }
    meta = {"video": "hd_00_00.mp4 (+9 more HD views)", "fps": FPS, "upsample": 1,
            "depth_source": "MapAnything, 10 calibrated views per moment", "density_note": "points from all 10 views each moment",
            "scale_from": notes["scale_from"]}
    h = write_world("panoptic-sports-pass", "CMU Panoptic: a football pass seen by 10 calibrated cameras (360°)",
                    P_out, C_out, tags, groups, boxes, ROOT / "media" / "pointworld", "people and the ball",
                    cam_list[0] | {"moving": False}, meta, notes)
    print(h["points"], "points written")
