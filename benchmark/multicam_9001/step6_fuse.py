"""Put each placed camera's MoGe-3 points into the car frame (scaled so the camera sits at its solved height) and
measure the car: front face (front camera) to rear face (rear camera) against NHTSA's 4.850 m, and width against
1.816 m. No keypoint lies on either end, so the length is an independent check."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
FRAME = 10
poses = json.loads((OUT / "poses_car.json").read_text())


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def car_points(cam, stride=2):
    zi = np.load(OUT / f"f{FRAME}_in_{cam}.npz")
    zf = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    P = backproject(zi["depth"], zi["K"])[::stride, ::stride]
    ok = zi["valid"][::stride, ::stride] & (zi["depth"][::stride, ::stride] > 0)
    rgb = zi["img"][::stride, ::stride][ok]
    P = P[ok]
    R, n, d = zf["R"], zf["n"], float(zf["d"])
    pose = poses[cam]
    s = pose["h"] / (-d)
    q = s * ((P - d * n) @ R.T)
    c, si = np.cos(pose["yaw"]), np.sin(pose["yaw"])
    xy = q[:, :2] @ np.array([[c, -si], [si, c]]).T + pose["t"]
    return np.column_stack([xy, q[:, 2]]), rgb


pts = {cam: car_points(cam) for cam in poses}
box = lambda p: (np.abs(p[:, 0]) < 2.8) & (np.abs(p[:, 1]) < 1.05) & (p[:, 2] > 0.08) & (p[:, 2] < 1.6)  # noqa: E731
mid = lambda p: (np.abs(p[:, 1]) < 0.5) & (p[:, 2] > 0.35) & (p[:, 2] < 0.9)  # noqa: E731
front = pts["cam06_front_oblique"][0]
rear = pts["cam05_rear_impact"][0]
f_in, r_in = front[box(front)], rear[box(rear)]
x_front = float(np.quantile(f_in[mid(f_in)][:, 0], 0.98))
x_rear = float(np.quantile(r_in[mid(r_in)][:, 0], 0.02))
# width: the rear face spans the full width; take its 1st..99th percentile of y at bumper height
rb = r_in[(r_in[:, 0] < x_rear + 0.35) & (r_in[:, 2] > 0.4) & (r_in[:, 2] < 1.0)]
w_rear = float(np.quantile(rb[:, 1], 0.99) - np.quantile(rb[:, 1], 0.01))
fb = f_in[(f_in[:, 0] > x_front - 0.35) & (f_in[:, 2] > 0.4) & (f_in[:, 2] < 0.9)]
w_front = float(np.quantile(fb[:, 1], 0.99) - np.quantile(fb[:, 1], 0.01))
roof_f = float(np.quantile(f_in[:, 2], 0.99))
roof_r = float(np.quantile(r_in[:, 2], 0.99))
print(f"front face x = {x_front:+.3f} m   rear face x = {x_rear:+.3f} m")
print(f"length {x_front - x_rear:.3f} m   (NHTSA data sheet 4.850)")
print(f"width at rear bumper {w_rear:.3f} m, at front bumper {w_front:.3f} m   (data sheet 1.816)")
print(f"roof height: front camera {roof_f:.3f} m, rear camera {roof_r:.3f} m   (Toyota spec 1.470)")
print(f"overhangs: front {x_front - 2.773 / 2:.3f} m, rear {-2.773 / 2 - x_rear:.3f} m")
np.savez_compressed(OUT / "fused_f10.npz", **{f"{c}_xyz": p[box(p)].astype(np.float32) for c, (p, _) in pts.items()},
                    **{f"{c}_rgb": rgb[box(p)] for c, (p, rgb) in pts.items()})
