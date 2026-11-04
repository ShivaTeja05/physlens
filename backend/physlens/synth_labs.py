"""Test clips with known physics for the stage experiments: catch the cup, Mystery Cans, weigh it by watching.

Same rules as synth.py: exact physics, sub-pixel anti-aliased discs, H.264 output. World y points up; image rows
point down.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .synth import draw_ball, ffmpeg_exe

BG = 18
ORANGE, RED, BLUE, GREEN = (40, 140, 255), (40, 40, 230), (230, 120, 40), (60, 190, 60)


def _encode(frames: Iterator[np.ndarray], fps: float, size: tuple[int, int], out: Path) -> Path:
    tmp = Path(tempfile.mkdtemp()) / "raw.avi"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
    for frame in frames:
        writer.write(frame)
    writer.release()
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(out)], check=True)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return out


# --- Catch the cup ------------------------------------------------------------------------------------------

@dataclass
class ProjectileScene:
    """A ball rolls along a table at constant speed, leaves the edge, falls to the floor and slides on."""
    g: float = 9.81
    ball_diameter: float = 0.04
    vx: float = 1.1               # m/s along the table
    table_height: float = 0.75    # m, table surface above the floor (what a presenter measures with a tape)
    x_start: float = 0.6          # m, ball centre at t = 0
    x_edge: float = 1.3           # m, table edge
    fps: float = 240.0
    tail: float = 0.4             # s of video after landing
    width: int = 1280
    height: int = 720
    px_per_m: float = 500.0
    floor_px: int = 680

    @property
    def landing_x(self) -> float:
        return self.x_edge + self.vx * np.sqrt(2 * self.table_height / self.g)

    def positions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.ball_diameter / 2
        t_edge = (self.x_edge - self.x_start) / self.vx
        t_fall = np.sqrt(2 * self.table_height / self.g)
        t = np.arange(0, t_edge + t_fall + self.tail, 1 / self.fps)
        x, y = np.empty_like(t), np.empty_like(t)
        for i, ti in enumerate(t):
            if ti <= t_edge:
                x[i], y[i] = self.x_start + self.vx * ti, self.table_height + r
            elif ti <= t_edge + t_fall:
                tau = ti - t_edge
                x[i], y[i] = self.x_edge + self.vx * tau, self.table_height + r - 0.5 * self.g * tau ** 2
            else:
                x[i], y[i] = self.landing_x + 0.35 * self.vx * (ti - t_edge - t_fall), r
        return t, x, y


def render_projectile(scene: ProjectileScene, out: Path) -> Path:
    t, x, y = scene.positions()
    ppm, r_px = scene.px_per_m, scene.ball_diameter / 2 * scene.px_per_m
    table_top = int(round(scene.floor_px - scene.table_height * ppm))

    def frames():
        for xm, ym in zip(x, y):
            f = np.full((scene.height, scene.width, 3), BG, np.uint8)
            cv2.rectangle(f, (0, table_top), (int(scene.x_edge * ppm), scene.floor_px), (55, 55, 55), -1)
            cv2.line(f, (0, scene.floor_px), (scene.width, scene.floor_px), (80, 80, 80), 2)
            draw_ball(f, xm * ppm + 0.5, scene.floor_px - ym * ppm + 0.5, r_px, ORANGE)
            yield f
    return _encode(frames(), scene.fps, (scene.width, scene.height), out)


# --- Mystery Cans -------------------------------------------------------------------------------------------

@dataclass
class RaceScene:
    """Two identical-looking cans roll from rest down parallel ramps. Their hidden mass placement differs:
    k = I / (m r²) (0.5 = solid, smaller = weight packed at the centre, 1 = weight at the rim)."""
    g: float = 9.81
    angle_deg: float = 12.0
    can_diameter: float = 0.066
    k_a: float = 0.40             # red can
    k_b: float = 0.90             # blue can
    finish_distance: float = 1.0  # m along the ramp from the start
    fps: float = 240.0
    tail: float = 0.3
    width: int = 1280
    height: int = 720
    px_per_m: float = 600.0
    start_px: tuple[float, float] = (240.0, 160.0)   # can A centre at t = 0 (image px)
    lane_gap_px: float = 230.0                       # can B starts this far below can A

    def accel(self, k: float) -> float:
        return self.g * np.sin(np.radians(self.angle_deg)) / (1 + k)

    def finish_time(self, k: float) -> float:
        return float(np.sqrt(2 * self.finish_distance / self.accel(k)))

    def positions(self, k: float, lane: int):
        t = np.arange(0, max(self.finish_time(self.k_a), self.finish_time(self.k_b)) + self.tail, 1 / self.fps)
        s = 0.5 * self.accel(k) * t ** 2
        th = np.radians(self.angle_deg)
        x0, y0 = self.start_px[0], self.start_px[1] + lane * self.lane_gap_px
        return t, x0 + s * np.cos(th) * self.px_per_m, y0 + s * np.sin(th) * self.px_per_m


def render_race(scene: RaceScene, out: Path) -> Path:
    th = np.radians(scene.angle_deg)
    r_px = scene.can_diameter / 2 * scene.px_per_m
    lanes = [(scene.k_a, 0, RED), (scene.k_b, 1, BLUE)]
    paths = [scene.positions(k, lane) for k, lane, _ in lanes]
    normal = np.array([np.sin(th), -np.cos(th)])   # image-space normal pointing up-left (away from the ramp)

    def frames():
        for i in range(len(paths[0][0])):
            f = np.full((scene.height, scene.width, 3), BG, np.uint8)
            for lane in (0, 1):
                x0, y0 = scene.start_px[0], scene.start_px[1] + lane * scene.lane_gap_px
                a = np.array([x0, y0]) - normal * r_px - np.array([np.cos(th), np.sin(th)]) * 40
                b = a + np.array([np.cos(th), np.sin(th)]) * (scene.finish_distance * scene.px_per_m + 160)
                cv2.line(f, tuple(int(v) for v in a), tuple(int(v) for v in b), (90, 90, 90), 3)
                fin = np.array([x0, y0]) + np.array([np.cos(th), np.sin(th)]) * scene.finish_distance * scene.px_per_m
                p, q = fin - normal * (r_px + 6), fin + normal * (r_px * 2.2)
                cv2.line(f, tuple(int(v) for v in p), tuple(int(v) for v in q), (200, 200, 200), 2)
            for (k, lane, color), (_, xs, ys) in zip(lanes, paths):
                draw_ball(f, xs[i] + 0.5, ys[i] + 0.5, r_px, color)
            yield f
    return _encode(frames(), scene.fps, (scene.width, scene.height), out)


# --- Weigh it by watching -----------------------------------------------------------------------------------

@dataclass
class SpringScene:
    """Two weights bounce on identical springs. The reference mass is known; the other is the mystery."""
    g: float = 9.81
    spring_k: float = 20.0        # N/m
    m_ref: float = 0.100          # kg, green
    m_mystery: float = 0.250      # kg, red
    weight_diameter: float = 0.04
    rest_length: float = 0.20     # m, unloaded spring
    amplitude: float = 0.03       # m, initial pull
    damping: float = 0.25         # 1/s
    fps: float = 120.0
    duration: float = 4.0
    width: int = 1280
    height: int = 720
    px_per_m: float = 900.0
    ceiling_px: int = 70
    x_m: tuple[float, float] = (0.50, 0.90)

    def omega(self, m: float) -> float:
        return float(np.sqrt(self.spring_k / m))

    def heights(self, m: float, phase: float = 0.0):
        """Weight centre in metres below the ceiling over time."""
        t = np.arange(0, self.duration, 1 / self.fps)
        eq = self.rest_length + m * self.g / self.spring_k
        return t, eq + self.amplitude * np.exp(-self.damping * t) * np.cos(self.omega(m) * t + phase)


def render_springs(scene: SpringScene, out: Path) -> Path:
    ppm, r_px = scene.px_per_m, scene.weight_diameter / 2 * scene.px_per_m
    series = [(scene.x_m[0], *scene.heights(scene.m_ref), GREEN), (scene.x_m[1], *scene.heights(scene.m_mystery, 0.6), RED)]

    def spring(f, x_px, top, bottom):
        pts = [(x_px, top)]
        n = 14
        for j in range(1, n):
            yy = top + (bottom - top) * j / n
            pts.append((x_px + (10 if j % 2 else -10), yy))
        pts.append((x_px, bottom))
        cv2.polylines(f, [np.array(pts, np.int32)], False, (150, 150, 150), 2, lineType=cv2.LINE_AA)

    def frames():
        for i in range(len(series[0][1])):
            f = np.full((scene.height, scene.width, 3), BG, np.uint8)
            cv2.line(f, (250, scene.ceiling_px), (1050, scene.ceiling_px), (120, 120, 120), 4)
            for x_m, _, h, color in series:
                x_px = x_m * ppm + 0.5
                y_px = scene.ceiling_px + h[i] * ppm + 0.5
                spring(f, x_px, scene.ceiling_px, y_px - r_px)
                draw_ball(f, x_px, y_px, r_px, color)
            yield f
    return _encode(frames(), scene.fps, (scene.width, scene.height), out)


LAB_SAMPLES: dict[str, tuple[Callable, object]] = {
    "cup-catch": (render_projectile, ProjectileScene()),
    "mystery-cans": (render_race, RaceScene()),
    "weigh-springs": (render_springs, SpringScene()),
}


def render_lab_samples(folder: Path) -> list[Path]:
    return [fn(scene, folder / f"{name}.mp4") for name, (fn, scene) in LAB_SAMPLES.items()]
