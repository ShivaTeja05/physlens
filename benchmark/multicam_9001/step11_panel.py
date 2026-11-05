"""A labelled geometric correction for the 9001 Camry's glossy left side. MoGe-3 reads the studio lights reflected in the
black paint as surface behind the paint, so the front-oblique camera's side panel lands ~10 cm inside the car, while
its depth on matte surfaces (the tyres) matches the exact ray geometry within 0.5-3%. The correction: the front
camera's painted surfaces on the left side (not the tyres) are moved along that camera's own rays by one factor k,
chosen so the lower doors between the wheels sit at the tyres' outer faces (0.895 m = half the 1.585 m track plus half
the 205 mm tyre). One number, shape kept; the world says it is a correction, not a measurement.
Then: the gap to the rear camera where both see the car, the width and the length, before and after."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from physlens.pointworld import write_world  # noqa: E402

HERE = Path(__file__).parent
src = (HERE / "step10_clean_car.py").read_text().split("cars, labs = [], []")[0]
g = {"__file__": str(HERE / "step10_clean_car.py")}
exec(compile(src, "s10", "exec"), g)
OUT, poses = g["OUT"], g["poses"]
WB, TF, TR, RW, TYRE_W = 2.773, 1.585, 1.575, 0.325, 0.205
TARGET_Y = TF / 2 + TYRE_W / 2            # 0.895 m: the tyres' outer faces


def in_car(p):
    return (np.abs(p[:, 0]) < 2.6) & (np.abs(p[:, 1]) < 1.0) & (p[:, 2] > 0.08) & (p[:, 2] < 1.6)


def strays(P):
    cnt = np.array([len(x) for x in cKDTree(P).query_ball_point(P, 0.03, workers=-1)])
    return cnt >= 6


clouds, cols, labs = {}, {}, []
for cam in g["CAMS"]:
    p, rgb, _ = g["camera_points"](cam)
    m = in_car(p)
    keep = strays(p[m])
    clouds[cam], cols[cam] = p[m][keep], rgb[m][keep]
    around = (np.abs(p[:, 0]) < 5) & (p[:, 1] > -3) & (p[:, 1] < 4) & (p[:, 2] > -0.05) & (p[:, 2] < 2.5) & ~m
    sel = np.random.default_rng(0).choice(np.flatnonzero(around), min(int(around.sum()), 250000), replace=False)
    labs.append((p[sel], rgb[sel]))

F = clouds["cam06_front_oblique"]
Cf = np.array([*poses["cam06_front_oblique"]["t"], poses["cam06_front_oblique"]["h"]])     # the front camera in the car frame
wheels = np.array([[WB / 2, RW], [-WB / 2, RW]])
near_wheel = np.min(np.hypot(F[:, [0]] - wheels[:, 0], F[:, [2]] - wheels[:, 1]), axis=1) < 0.40
painted_left = (F[:, 1] > 0.3) & ~near_wheel
doors = lambda P: (np.abs(P[:, 0]) < 0.9) & (P[:, 2] > 0.35) & (P[:, 2] < 0.75) & (P[:, 1] > 0.5)   # noqa: E731


def corrected(k):
    Q = F.copy()
    Q[painted_left] = Cf + k * (F[painted_left] - Cf)
    return Q


ks = np.arange(0.85, 1.0001, 0.0025)
y_med = [float(np.median(corrected(k)[doors(corrected(k))][:, 1])) for k in ks]
k = float(ks[int(np.argmin(np.abs(np.array(y_med) - TARGET_Y)))])
Fc = corrected(k)


def report(label, Fx):
    R = clouds["cam05_rear_impact"]
    d1, _ = cKDTree(R).query(Fx, distance_upper_bound=0.15)
    d2, _ = cKDTree(Fx).query(R, distance_upper_bound=0.15)
    d = np.concatenate([d1[np.isfinite(d1)], d2[np.isfinite(d2)]])
    allp = np.concatenate([Fx, R])
    mid = allp[(np.abs(allp[:, 1]) < 0.5) & (allp[:, 2] > 0.35) & (allp[:, 2] < 0.9)]
    length = np.quantile(Fx[(np.abs(Fx[:, 1]) < 0.5) & (Fx[:, 2] > 0.35) & (Fx[:, 2] < 0.9)][:, 0], 0.98) - \
        np.quantile(R[(np.abs(R[:, 1]) < 0.5) & (R[:, 2] > 0.35) & (R[:, 2] < 0.9)][:, 0], 0.02)
    side = Fx[doors(Fx)][:, 1]
    print(f"{label:7s} gap where both cameras see the car: median {np.median(d) * 100:4.1f} cm, 80th pct {np.percentile(d, 80) * 100:4.1f} cm"
          f" | left doors at y = {np.median(side):.3f} m | length {length:.3f} m | overlap points {len(d):,}")
    return float(np.median(d) * 100), float(np.percentile(d, 80) * 100)


print(f"correction factor k = {k:.4f} along the front camera's rays, on {painted_left.sum():,} painted left-side points "
      f"(of {len(F):,}); target: doors at the tyres' outer faces, y = {TARGET_Y:.3f} m")
before = report("before", F)
after = report("after", Fc)

# write the world with the corrected front camera surface
to_sim = g["to_sim"]
Pc = np.concatenate([Fc, clouds["cam05_rear_impact"]]); Cc = np.concatenate([cols["cam06_front_oblique"], cols["cam05_rear_impact"]])
Pw = np.concatenate([l[0] for l in labs]); Cw = np.concatenate([l[1] for l in labs])
Pc_s, Pw_s = to_sim(Pc), to_sim(Pw)
lo, hi = np.percentile(Pc_s, [1, 99], axis=0)
box = {"frame": 10, "t": -0.04, "points": int(len(Pc)), "centre": ((lo + hi) / 2).round(3).tolist(), "size": (hi - lo).round(3).tolist(), "yaw": 0.0}
hdr = json.loads((ROOT / "media" / "pointworld" / "side-9001-multicam.json").read_text())
notes = hdr["notes"]
notes["multicam"]["corrections"] = {
    "glossy_side_panel": (f"CORRECTED, not measured: the front-oblique camera's painted left side (not the tyres) moved along that "
                          f"camera's rays by k = {k:.4f}, so the lower doors sit at the tyres' outer faces ({TARGET_Y:.3f} m). "
                          "MoGe-3 reads the lights reflected in the black paint as surface behind it; on matte tyres its depth "
                          "matches the exact geometry within 0.5-3%."),
    "gap_cm_before_after": [round(before[0], 1), round(after[0], 1)]}
notes["multicam"]["overlap_gap_cm"] = {"median": round(after[0], 1), "p80": round(after[1], 1)}
groups = [{"tag": 255, "start": 0, "count": int(len(Pw_s))}, {"tag": 0, "frame": 10, "t": -0.04, "start": int(len(Pw_s)), "count": int(len(Pc_s))}]
meta = {k_: hdr[k_] for k_ in ("video", "fps", "upsample", "depth_source", "density_note", "scale_from") if k_ in hdr}
h = write_world("side-9001-multicam", hdr["title"], [Pw_s, Pc_s], [Cw, Cc],
                [np.full(len(Pw_s), 255, np.uint8), np.zeros(len(Pc_s), np.uint8)], groups, [box],
                ROOT / "media" / "pointworld", hdr["object"], hdr["camera"], meta, notes)
print("world written; car box", box["size"])
