"""Write the four-camera Camry (NHTSA 9001, 40 ms before impact) as a simulator world: the lab around it as the
static group, the car's points as one moment, each camera drawn where it was solved. Simulator axes: x forward,
y up, z = the car's right (car frame y is its left)."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from physlens.pointworld import write_world  # noqa: E402

OUT = Path(__file__).parent / "out"
FRAME, ROOF = 10, 1.47
# The overhead cameras are left out: MoGe-3 flattens the glossy roof seen from above into a sheet that runs past the
# car, so their points are not accurate enough to show as measured.
CAMS = ["cam06_front_oblique", "cam05_rear_impact"]
NICE = {"cam06_front_oblique": "Front oblique (No. 6)", "cam05_rear_impact": "Rear (No. 5)",
        "cam01_overhead_wide": "Overhead wide (No. 1)", "cam02": "Overhead close-up (No. 2)"}
poses = {**json.loads((OUT / "poses_car.json").read_text()), **json.loads((OUT / "poses_overhead.json").read_text())}


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def to_sim(p):                      # car frame (x fwd, y left, z up) -> simulator (x fwd, y up, z right)
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


world, car, world_c, car_c, cams = [], [], [], [], []
for cam in CAMS:
    zi = np.load(OUT / f"f{FRAME}_in_{cam}.npz")
    zf = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    P = backproject(zi["depth"], zi["K"])[::2, ::2]
    ok = zi["valid"][::2, ::2] & (zi["depth"][::2, ::2] > 0)
    rgb = zi["img"][::2, ::2][ok][:, ::-1]                       # BGR -> RGB
    R, n, d = zf["R"], zf["n"], float(zf["d"])
    pose = poses[cam]
    s = pose["h"] / (-d)
    q = s * ((P[ok] - d * n) @ R.T)
    xy = q[:, :2] @ rot(pose["yaw"]).T + pose["t"]
    p = np.column_stack([xy, q[:, 2]])
    in_car = (np.abs(p[:, 0]) < 2.6) & (np.abs(p[:, 1]) < 1.0) & (p[:, 2] > 0.08) & (p[:, 2] < 1.6)
    around = (np.abs(p[:, 0]) < 5) & (p[:, 1] > -3) & (p[:, 1] < 4) & (p[:, 2] > -0.05) & (p[:, 2] < 2.5) & ~in_car
    car.append(to_sim(p[in_car])); car_c.append(rgb[in_car])
    world.append(to_sim(p[around])); world_c.append(rgb[around])
    fwd_floor = R @ np.array([0, 0, 1.0])
    fwd = np.array([*(rot(pose["yaw"]) @ fwd_floor[:2]), fwd_floor[2]])
    pos = np.array([*pose["t"], pose["h"]])
    fov = float(np.degrees(2 * np.arctan(zi["depth"].shape[1] / 2 / zi["K"][0, 0])))
    cams.append({"name": NICE[cam], "position": to_sim(pos[None])[0].round(3).tolist(),
                 "forward": to_sim(fwd[None])[0].round(4).tolist(), "fov_deg": round(fov, 1)})

Pw, Cw = np.concatenate(world), np.concatenate(world_c)
Pc, Cc = np.concatenate(car), np.concatenate(car_c)
lo, hi = np.percentile(Pc, [1, 99], axis=0)
box = {"frame": FRAME, "t": -0.04, "points": int(len(Pc)), "centre": ((lo + hi) / 2).round(3).tolist(),
       "size": (hi - lo).round(3).tolist(), "yaw": 0.0}
groups = [{"tag": 255, "start": 0, "count": int(len(Pw))},
          {"tag": 0, "frame": FRAME, "t": -0.04, "start": int(len(Pw)), "count": int(len(Pc))}]
notes = {
    "credit": "Video: NHTSA vehicle crash test 9001 (public), cameras No. 5 and 6, both at -40 ms on the shared 1,000 fps clock.",
    "scale_from": "the Camry's wheels (wheelbase 2.773 m, NHTSA data sheet; track and tyre, Toyota's published figures) and its roof (1.47 m, published)",
    "multicam": {
        "method": "Each camera placed on its own against the parked Camry: its floor plane and lens from MoGe-3, then the wheel "
                  "contact points and centres marked in the frame (front camera) or the rear tyres plus the roof height (rear camera), "
                  "so the two cameras need not share a view. The car's length was not used and is the check.",
        "checks": {"length_m": [4.863, 4.850], "rear_bumper_width_m": 1.664, "data_sheet_width_m": 1.816},
        "coverage_pct": {"front": 54.2, "rear": 89.6, "left (struck side)": 92.4, "right": 16.3, "top": 64.0},
        "not_seen": "No camera faced the car's right side, and the overhead cameras' depth of the glossy roof was too flat to use: right side and much of the roof are not measured.",
    },
    "cameras": cams,
}
meta = {"video": "cam06_front_oblique.mp4 (+ cam05_rear_impact.mp4)", "fps": 1000.0, "upsample": 1,
        "depth_source": "MoGe-3, two synced cameras placed by the car's wheels", "density_note": "one measured point per 2x2 pixels of each camera",
        "scale_from": notes["scale_from"]}
h = write_world("side-9001-multicam", "NHTSA side test 9001: Camry from two synced cameras, front and rear (before impact)",
                [Pw, Pc], [Cw, Cc], [np.full(len(Pw), 255, np.uint8), np.zeros(len(Pc), np.uint8)], groups, [box],
                ROOT / "media" / "pointworld", "Toyota Camry", cams[0] | {"moving": False}, meta, notes)
print(h["points"], "points:", h["static_points"], "lab +", h["object_points"], "car; car box", box["size"])
