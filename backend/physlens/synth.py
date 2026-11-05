"""Test clips with known physics: a ball dropped onto a floor, bouncing with fixed restitution.

`time_scale` > 1 plays the clip in slow motion (2.0 = half speed), which is how we build the
"memory vs eyes" test: the true gravity is 9.81 but the video shows 9.81 / time_scale².
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DropScene:
    g: float = 9.81            # m/s², real gravity
    restitution: float = 0.78
    drop_height: float = 1.0   # m, ball centre above the floor contact height at release
    vx: float = 0.12           # m/s, gentle sideways drift
    ball_diameter: float = 0.065
    time_scale: float = 1.0    # video seconds per real second
    fps: float = 240.0         # video frame rate
    duration: float = 3.0      # video seconds
    width: int = 1280
    height: int = 720
    px_per_m: float = 480.0
    floor_px: int = 660        # image row of the floor line
    x0_px: float = 420.0


# Input options for ffmpeg on anything a user uploaded: real media containers from local files only, so a crafted
# playlist or concat file can't make ffmpeg open other files on this machine.
FFMPEG_INPUT = ["-protocol_whitelist", "file", "-format_whitelist",
                "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts,flv,ogg,wav,mp3,aac,flac"]


def ffmpeg_exe() -> str:
    """The ffmpeg bundled with imageio-ffmpeg, so encoding doesn't depend on the system install."""
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def simulate(scene: DropScene) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact positions at every video frame. Returns (t_video, x_m, y_m) with y = height of the ball
    centre above the floor contact height."""
    t_video = np.arange(0.0, scene.duration, 1.0 / scene.fps)
    t_real = t_video / scene.time_scale
    g, e = scene.g, scene.restitution

    # Contact times and launch speeds of each flight, in real time.
    t_contact = np.sqrt(2 * scene.drop_height / g)
    starts, speeds, heights = [0.0], [0.0], [scene.drop_height]
    v = g * t_contact
    t_now = t_contact
    while t_now < t_real[-1] + 1 and v > 0.05:
        v *= e
        starts.append(t_now)
        speeds.append(v)
        heights.append(0.0)
        t_now += 2 * v / g

    y = np.empty_like(t_real)
    for i, tr in enumerate(t_real):
        j = np.searchsorted(starts, tr, side="right") - 1
        tau = tr - starts[j]
        y[i] = max(heights[j] + speeds[j] * tau - 0.5 * g * tau ** 2, 0.0)
    x = scene.x0_px / scene.px_per_m + scene.vx * t_real
    return t_video, x, y


def draw_ball(frame: np.ndarray, cx: float, cy: float, r: float, color: tuple[int, int, int], ss: int = 8) -> None:
    """Anti-aliased disc with exact area: each pixel is shaded by the fraction of its ss×ss sub-samples
    inside the circle. (cv2.circle's anti-aliasing draws discs about half a pixel too big.)"""
    x0, x1 = int(np.floor(cx - r - 1)), int(np.ceil(cx + r + 1))
    y0, y1 = int(np.floor(cy - r - 1)), int(np.ceil(cy + r + 1))
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, frame.shape[1]), min(y1, frame.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    offsets = (np.arange(ss) + 0.5) / ss
    xs = x0 + (np.arange(x1 - x0)[:, None] + offsets[None, :]).ravel()
    ys = y0 + (np.arange(y1 - y0)[:, None] + offsets[None, :]).ravel()
    inside = ((xs[None, :] - cx) ** 2 + (ys[:, None] - cy) ** 2) <= r * r
    cover = inside.reshape(y1 - y0, ss, x1 - x0, ss).mean(axis=(1, 3))[..., None]
    roi = frame[y0:y1, x0:x1].astype(float)
    frame[y0:y1, x0:x1] = np.round(roi * (1 - cover) + np.array(color, float) * cover).astype(np.uint8)


def render(scene: DropScene, out_path: Path) -> Path:
    """Draws the clip and encodes it as browser-friendly H.264."""
    t, x, y = simulate(scene)
    r_px = scene.ball_diameter * scene.px_per_m / 2
    contact_row = scene.floor_px - r_px
    tmp = Path(tempfile.mkdtemp()) / "raw.avi"
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"MJPG"), scene.fps, (scene.width, scene.height))
    for xm, ym in zip(x, y):
        frame = np.full((scene.height, scene.width, 3), 18, np.uint8)
        cv2.line(frame, (0, scene.floor_px), (scene.width, scene.floor_px), (70, 70, 70), 2)
        # Pixel centres sit at integer coordinates in OpenCV; shift by half a pixel to match draw_ball.
        cx, cy = xm * scene.px_per_m + 0.5, contact_row - ym * scene.px_per_m + 0.5
        draw_ball(frame, cx, cy, r_px, (40, 140, 255))
        writer.write(frame)
    writer.release()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-crf", "18", "-movflags", "+faststart", str(out_path)], check=True)
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return out_path


SAMPLES = {
    "drop-real": DropScene(),
    "drop-slowmo-2x": DropScene(time_scale=2.0, duration=6.0),
    "drop-low-gravity": DropScene(g=1.81, duration=7.0, restitution=0.7),
}


def render_samples(folder: Path) -> list[Path]:
    return [render(scene, folder / f"{name}.mp4") for name, scene in SAMPLES.items()]
