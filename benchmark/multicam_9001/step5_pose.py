"""Place each camera against the Camry's own wheels. Per camera the floor plane (direction only) and lens come from
MoGe-3; a marked pixel's ray meets a plane at the keypoint's known height (tyre contact z=0, wheel centre z=r), so
the unknowns are the camera's height h, yaw and floor shift (4). Car frame: origin on the floor under the middle of
the wheelbase, x forward, y to the car's left (the struck side), z up.
Known: wheelbase 2.773 m (NHTSA data sheet); track 1.585 front / 1.575 rear and tyre 205/65R16 (r ≈ 0.325 m loaded):
Toyota's published 2015 Camry LE figures. Keypoint pixels were read from zoomed frames (full 1536x1024)."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

OUT = Path(__file__).parent / "out"
FRAME = 10
WB, TF, TR, RW = 2.773, 1.585, 1.575, 0.325
CAR = {"FL_contact": (WB / 2, TF / 2, 0.0), "FL_centre": (WB / 2, TF / 2, RW),
       "RL_contact": (-WB / 2, TR / 2, 0.0), "RL_centre": (-WB / 2, TR / 2, RW),
       "RR_contact": (-WB / 2, -TR / 2, 0.0), "FR_contact": (WB / 2, -TF / 2, 0.0)}
# (pixel u, v, weight): lower weight where the wheel is partly hidden
MARKS = {
    "cam06_front_oblique": {"FL_contact": (1090, 873, 1.0), "FL_centre": (1082, 712, 1.0),
                            "RL_contact": (1471, 617, 0.5), "RL_centre": (1474, 524, 0.5)},
    "cam05_rear_impact": {"RL_contact": (362, 858, 1.0), "RR_contact": (1200, 813, 0.6)},
}


def rays(cam, px):
    z = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    K = np.load(OUT / f"f{FRAME}_in_{cam}.npz")["K"]
    R = z["R"]
    uv1 = np.array([[u, v, 1.0] for u, v in px])
    r = uv1 @ np.linalg.inv(K).T
    return r @ R.T, z                     # directions in the camera's floor frame (z up)


def rotz(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


ROOF = 1.47           # Toyota's published height
# Cameras with too few wheel marks to fix their own height take it from the roof line they see (98th percentile of
# the Camry's points), which scales with the camera height.
HEIGHT_FROM_ROOF = {"cam05_rear_impact"}


def roof_height(cam):
    z = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    return ROOF / float(np.quantile(z["car"][:, 2], 0.98)) * (-float(z["d"]))


def solve(cam):
    names = list(MARKS[cam])
    px = [MARKS[cam][n][:2] for n in names]
    w = np.array([MARKS[cam][n][2] for n in names])
    f, z = rays(cam, px)
    X = np.array([CAR[n] for n in names])
    h_fixed = roof_height(cam) if cam in HEIGHT_FROM_ROOF else None

    def resid(q):
        h, yaw, tx, ty = q
        if h_fixed is not None:
            h = h_fixed
        lam = (X[:, 2] - h) / f[:, 2]
        p = lam[:, None] * f[:, :2]
        pc = p @ rotz(yaw).T + [tx, ty]
        return ((pc - X[:, :2]) * w[:, None]).ravel()

    best = None
    for yaw0 in np.radians(np.arange(0, 360, 20)):
        for h0 in (0.8, 1.2, 1.8, 2.5, 4.0):
            r = least_squares(resid, [h0, yaw0, 0, 0], bounds=([0.2, -10, -20, -20], [8, 10, 20, 20]))
            if best is None or r.cost < best.cost:
                best = r
    h, yaw, tx, ty = best.x
    if h_fixed is not None:
        h = h_fixed
    err = np.sqrt(np.mean((best.fun / np.repeat(w, 2)) ** 2))
    moge_h = -float(z["d"])
    return {"h": h, "yaw": yaw, "t": [tx, ty], "rms_m": err, "scale_vs_moge": h / moge_h,
            "camera_position": [*(rotz(yaw) @ [0, 0] + [tx, ty]), h]}


res = {cam: solve(cam) for cam in MARKS}
for cam, r in res.items():
    print(f"{cam:22s} height {r['h']:.2f} m  yaw {np.degrees(r['yaw']) % 360:6.1f}°  at ({r['t'][0]:+.2f}, {r['t'][1]:+.2f})"
          f"  keypoint rms {r['rms_m'] * 100:.1f} cm  MoGe scale x{r['scale_vs_moge']:.2f}")
json.dump({c: {k: (float(v) if np.isscalar(v) else [float(x) for x in v]) for k, v in r.items()} for c, r in res.items()},
          open(OUT / "poses_car.json", "w"), indent=1)
