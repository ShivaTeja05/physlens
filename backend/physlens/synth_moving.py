"""A drop filmed by a moving, handheld-style camera, with known physics and known camera poses (ground truth for the
moving-camera pipeline).

A small ray tracer: a textured floor (y = 0) and back wall (z = −2.5), an orange ball dropped from 1.5 m that bounces
with restitution 0.78 under g = 9.81. The camera slides 1.2 m sideways, bobs, and pans to keep the ball near the
middle with a little hand shake. Axes: metres, y up; camera poses are OpenCV-style cam-to-world (x right, y down,
z forward).
"""
from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .synth import ffmpeg_exe


@dataclass
class MovingScene:
    W: int = 640
    H: int = 480
    f: float = 560.0
    fps: float = 60.0
    duration: float = 2.6
    g: float = 9.81
    e: float = 0.78
    r: float = 0.06
    y0: float = 1.5
    vx: float = 0.25


def ball_path(sc: MovingScene) -> np.ndarray:
    """[t, x, y, z] of the ball centre at every frame (fine-stepped bounces)."""
    dt = 1e-4
    t, x, y, vy = 0.0, -0.3, sc.y0, 0.0
    out, nxt = [], 0.0
    while t <= sc.duration + 1e-9:
        if t >= nxt - 1e-12:
            out.append([t, x, y, 0.0])
            nxt += 1.0 / sc.fps
        vy -= sc.g * dt
        y += vy * dt
        x += sc.vx * dt
        if y < sc.r and vy < 0:
            y, vy = sc.r, -vy * sc.e
        t += dt
    return np.array(out)


def camera_pose(sc: MovingScene, t: float, look_at: np.ndarray) -> np.ndarray:
    """Cam-to-world 4×4 (OpenCV axes) for a camera sliding sideways, bobbing, panning towards `look_at`."""
    rng = np.random.default_rng(int(t * 1000) + 7)
    c = np.array([-0.6 + 1.2 * t / sc.duration, 1.0 + 0.05 * math.sin(2 * math.pi * 1.3 * t), 3.0 - 0.3 * t / sc.duration])
    target = 0.6 * look_at + 0.4 * np.array([0.0, 0.7, 0.0]) + rng.normal(0, 0.01, 3)   # hand shake
    fwd = target - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 1.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = right, down, fwd, c
    return T


def _texture(n: int = 512, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.random((n // 16, n // 16))
    tex = np.kron(base, np.ones((16, 16)))
    blobs = rng.random((n, n))
    from scipy.ndimage import gaussian_filter
    tex = 0.6 * tex + 0.4 * gaussian_filter(blobs, 2)
    return (tex - tex.min()) / (tex.max() - tex.min())


def render(sc: MovingScene, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    path = ball_path(sc)
    tex_floor, tex_wall = _texture(seed=3), _texture(seed=11)
    u, v = np.meshgrid(np.arange(sc.W) + 0.5, np.arange(sc.H) + 0.5)
    dirs_cam = np.stack([(u - sc.W / 2) / sc.f, (v - sc.H / 2) / sc.f, np.ones_like(u)], -1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=-1, keepdims=True)
    proc = subprocess.Popen([ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                             "-s", f"{sc.W}x{sc.H}", "-r", str(sc.fps), "-i", "-", "-an", "-c:v", "libx264",
                             "-profile:v", "baseline", "-bf", "0", "-pix_fmt", "yuv420p", "-crf", "16",
                             "-video_track_timescale", "90000", str(out)], stdin=subprocess.PIPE)
    poses = []
    for t, bx, by, bz in path:
        T = camera_pose(sc, t, np.array([bx, by, bz]))
        poses.append(T.tolist())
        o = T[:3, 3]
        d = dirs_cam @ T[:3, :3].T
        img = np.zeros((sc.H, sc.W, 3))
        depth = np.full((sc.H, sc.W), np.inf)
        # Floor y = 0.
        tf = np.where(d[..., 1] < -1e-6, -o[1] / np.where(d[..., 1] < -1e-6, d[..., 1], -1), np.inf)
        p = o + d * tf[..., None]
        p = np.nan_to_num(p, posinf=0.0, neginf=0.0)      # rays that miss the floor (masked out below)
        val = tex_floor[((p[..., 0] * 40) % 512).astype(int), ((p[..., 2] * 40) % 512).astype(int)]
        m = tf < depth
        img[m] = (0.25 + 0.6 * val[m])[:, None] * np.array([0.85, 0.82, 0.78])
        depth[m] = tf[m]
        # Back wall z = −2.5.
        tw = np.where(d[..., 2] < -1e-6, (-2.5 - o[2]) / np.where(d[..., 2] < -1e-6, d[..., 2], -1), np.inf)
        p = np.nan_to_num(o + d * tw[..., None], posinf=0.0, neginf=0.0)
        val = tex_wall[((p[..., 0] * 40) % 512).astype(int), ((p[..., 1] * 40) % 512).astype(int)]
        m = (tw < depth) & (p[..., 1] > 0)
        img[m] = (0.2 + 0.5 * val[m])[:, None] * np.array([0.55, 0.62, 0.75])
        depth[m] = tw[m]
        # The ball.
        oc = o - np.array([bx, by, bz])
        b = (d * oc).sum(-1)
        disc = b * b - (oc @ oc - sc.r ** 2)
        ts = np.where(disc > 0, -b - np.sqrt(np.maximum(disc, 0)), np.inf)
        m = (ts > 0) & (ts < depth)
        if m.any():
            nrm = (o + d[m] * ts[m][:, None] - np.array([bx, by, bz])) / sc.r
            shade = 0.35 + 0.65 * np.clip(nrm @ np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5]), 0, 1)
            img[m] = shade[:, None] * np.array([1.0, 0.45, 0.1])
        proc.stdin.write((np.clip(img, 0, 1) * 255).astype(np.uint8).tobytes())
    proc.stdin.close()
    proc.wait()
    truth = {"fps": sc.fps, "intrinsics": [[sc.f, 0, sc.W / 2], [0, sc.f, sc.H / 2], [0, 0, 1]], "g": sc.g, "e": sc.e,
             "ball_radius_m": sc.r, "ball_world": path.tolist(), "camera_poses_cam2world": poses}
    out.with_suffix(".truth.json").write_text(json.dumps(truth))
    return truth
