"""The 9001 Camry's measured surfaces, cleaned and dense: every pixel of the two placed cameras (front oblique, rear),
minus the points that are not surface:
  - flying pixels: at a depth jump (the car's edge against the background) a pixel's depth falls between the two
    surfaces and lands in mid-air; dropped where the relative depth change to any neighbour exceeds 4%;
  - strays: points with too few neighbours within 3 cm (isolated specks);
  - grazing views: surfaces seen nearly edge-on (normal more than 80° from the camera ray) have unreliable depth.
Writes the world again (front + rear cameras), and prints how much each step removed and how consistent the two
cameras are where both see the car (median distance to the other camera's points)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from physlens.pointworld import write_world  # noqa: E402

OUT = Path(__file__).parent / "out"
FRAME = 10
CAMS = ["cam06_front_oblique", "cam05_rear_impact"]
NICE = {"cam06_front_oblique": "Front oblique (No. 6)", "cam05_rear_impact": "Rear (No. 5)"}
poses = json.loads((OUT / "poses_car.json").read_text())


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def to_sim(p):
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def camera_points(cam):
    zi = np.load(OUT / f"f{FRAME}_in_{cam}.npz")
    zf = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    depth, K, valid = zi["depth"].astype(np.float64), zi["K"], zi["valid"] & (zi["depth"] > 0)
    h, w = depth.shape
    # flying pixels: relative depth jump to any 8-neighbour
    jump = np.zeros_like(depth)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        sh = np.roll(np.roll(depth, dy, 0), dx, 1)
        jump = np.maximum(jump, np.abs(depth - sh) / np.maximum(depth, 1e-6))
    flying = jump > 0.04
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    P = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)
    # grazing: normal from the depth map's neighbours vs the viewing ray
    dx_ = np.roll(P, -1, 1) - P
    dy_ = np.roll(P, -1, 0) - P
    n = np.cross(dx_, dy_)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    ray = P / (np.linalg.norm(P, axis=-1, keepdims=True) + 1e-12)
    grazing = np.abs(np.einsum("ijk,ijk->ij", n, ray)) < np.cos(np.radians(80))
    steps = {"pixels": int(valid.sum()), "flying": int((valid & flying).sum()), "grazing": int((valid & ~flying & grazing).sum())}
    ok = valid & ~flying & ~grazing
    rgb = zi["img"][ok][:, ::-1]
    R, nn, d = zf["R"], zf["n"], float(zf["d"])
    pose = poses[cam]
    s = pose["h"] / (-d)
    q = s * ((P[ok] - d * nn) @ R.T)
    xy = q[:, :2] @ rot(pose["yaw"]).T + pose["t"]
    p = np.column_stack([xy, q[:, 2]])
    return p, rgb, steps


cars, labs = [], []
for cam in CAMS:
    p, rgb, steps = camera_points(cam)
    in_car = (np.abs(p[:, 0]) < 2.6) & (np.abs(p[:, 1]) < 1.0) & (p[:, 2] > 0.08) & (p[:, 2] < 1.6)
    car, crgb = p[in_car], rgb[in_car]
    # strays: fewer than 6 neighbours within 3 cm
    cnt = np.array([len(x) for x in cKDTree(car).query_ball_point(car, 0.03, workers=-1)])
    stray = cnt < 6
    steps["strays"] = int(stray.sum())
    steps["car_points_kept"] = int((~stray).sum())
    cars.append((car[~stray], crgb[~stray]))
    around = (np.abs(p[:, 0]) < 5) & (p[:, 1] > -3) & (p[:, 1] < 4) & (p[:, 2] > -0.05) & (p[:, 2] < 2.5) & ~in_car
    sel = np.flatnonzero(around)
    sel = np.random.default_rng(0).choice(sel, min(len(sel), 250000), replace=False)
    labs.append((p[sel], rgb[sel]))
    print(cam, steps)

# consistency where both cameras see the car: each point's distance to the other camera's points (only where close)
a, b = cars[0][0], cars[1][0]
da, _ = cKDTree(b).query(a, distance_upper_bound=0.15)
db, _ = cKDTree(a).query(b, distance_upper_bound=0.15)
shared = np.concatenate([da[np.isfinite(da)], db[np.isfinite(db)]])
print(f"overlap: {len(shared):,} points within 15 cm of the other camera's surface; median gap {np.median(shared) * 100:.1f} cm, "
      f"80th pct {np.percentile(shared, 80) * 100:.1f} cm")

Pc = np.concatenate([c[0] for c in cars]); Cc = np.concatenate([c[1] for c in cars])
Pw = np.concatenate([l[0] for l in labs]); Cw = np.concatenate([l[1] for l in labs])
Pc_s, Pw_s = to_sim(Pc), to_sim(Pw)
lo, hi = np.percentile(Pc_s, [1, 99], axis=0)
box = {"frame": FRAME, "t": -0.04, "points": int(len(Pc)), "centre": ((lo + hi) / 2).round(3).tolist(), "size": (hi - lo).round(3).tolist(), "yaw": 0.0}
hdr = json.loads((ROOT / "media" / "pointworld" / "side-9001-multicam.json").read_text())
notes = hdr["notes"]
notes["multicam"]["cleaning"] = ("Every pixel of the two cameras, minus flying pixels at depth jumps (>4% relative), surfaces seen "
                                 "nearly edge-on (>80°), and isolated strays (<6 neighbours within 3 cm).")
notes["multicam"]["overlap_gap_cm"] = {"median": round(float(np.median(shared) * 100), 1), "p80": round(float(np.percentile(shared, 80) * 100), 1)}
groups = [{"tag": 255, "start": 0, "count": int(len(Pw_s))}, {"tag": 0, "frame": FRAME, "t": -0.04, "start": int(len(Pw_s)), "count": int(len(Pc_s))}]
meta = {k: hdr[k] for k in ("video", "fps", "upsample", "depth_source", "density_note", "scale_from") if k in hdr}
meta["density_note"] = "every pixel of each camera, cleaned of flying pixels, edge-on surfaces and strays"
h = write_world("side-9001-multicam", hdr["title"], [Pw_s, Pc_s], [Cw, Cc],
                [np.full(len(Pw_s), 255, np.uint8), np.zeros(len(Pc_s), np.uint8)], groups, [box],
                ROOT / "media" / "pointworld", hdr["object"], hdr["camera"], meta, notes)
print(h["points"], "points;", h["object_points"], "car points; car box", box["size"])
