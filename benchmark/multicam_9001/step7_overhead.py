"""Add the overhead cameras: each one's floor is level (3.5° / 5.6°) and its scale comes from the roof height, so
only yaw and floor shift remain. ICP (full grid of starting yaws) of its Camry points against the top surfaces
(z > 0.9 m) of the car already placed from the wheel-marked cameras. Then coverage per side, measured the same way
as objsim.coverage (10 cm cells, 25 cm shell, side interiors), for both the two-camera and the all-camera car."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

OUT = Path(__file__).parent / "out"
FRAME = 10
ROOF = 1.47
rng = np.random.default_rng(0)
fused = np.load(OUT / "fused_f10.npz")
placed = ["cam06_front_oblique", "cam05_rear_impact"]
ref = np.concatenate([fused[f"{c}_xyz"] for c in placed])
top = ref[ref[:, 2] > 0.5]
tree = cKDTree(top)


def backproject(depth, K):
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    return np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)


def floor_points(cam, stride=2, only_car=True):
    zi = np.load(OUT / f"f{FRAME}_in_{cam}.npz")
    zf = np.load(OUT / f"f{FRAME}_floor_{cam}.npz")
    P = backproject(zi["depth"], zi["K"])[::stride, ::stride]
    ok = zi["valid"][::stride, ::stride] & (zi["depth"][::stride, ::stride] > 0)
    if only_car:
        ok &= zi["car"][::stride, ::stride]
    R, n, d = zf["R"], zf["n"], float(zf["d"])
    s = ROOF / float(np.quantile(zf["car"][:, 2], 0.98))
    return s * ((P[ok] - d * n) @ R.T), zi["img"][::stride, ::stride][ok], s, -d * s


def rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def icp(src, yaw, iters=50):
    t = top[:, :2].mean(0) - (src[:, :2] @ rot(yaw).T).mean(0)
    for _ in range(iters):
        p = src[:, :2] @ rot(yaw).T + t
        d, j = tree.query(np.column_stack([p, src[:, 2]]), distance_upper_bound=0.4)
        m = np.isfinite(d)
        if m.sum() < 100:
            return yaw, t, 0.0, 9.0
        a, b = p[m], top[j[m]][:, :2]
        ca, cb = a.mean(0), b.mean(0)
        H = (a - ca).T @ (b - cb)
        dy = np.arctan2(H[0, 1] - H[1, 0], H[0, 0] + H[1, 1])
        yaw += dy
        t = rot(dy) @ t + (cb - rot(dy) @ ca)
    p = np.column_stack([src[:, :2] @ rot(yaw).T + t, src[:, 2]])
    d, _ = tree.query(p)
    return yaw, t, float(np.mean(d < 0.05)), float(np.sqrt(np.mean(np.minimum(d, 0.3) ** 2)))


poses = json.loads((OUT / "poses_car.json").read_text())
extra = {}
for cam in ["cam01_overhead_wide", "cam02"]:
    car, _, s, h = floor_points(cam)
    car = car[car[:, 2] > 0.5]
    src = car[rng.choice(len(car), min(len(car), 6000), replace=False)]
    runs = sorted((icp(src, y0) for y0 in np.radians(np.arange(0, 360, 10))), key=lambda r: -r[2])
    best = runs[0]
    second = next((r for r in runs[1:] if abs(((r[0] - best[0] + np.pi) % (2 * np.pi)) - np.pi) > np.radians(25)), None)
    print(f"{cam:22s} yaw {np.degrees(best[0]) % 360:6.1f}°  inliers<5cm {best[2]:.2f}  rms {best[3]:.3f} m"
          f"   next-best other yaw: {np.degrees(second[0]) % 360:.0f}° at {second[2]:.2f}   camera height {h:.2f} m")
    extra[cam] = {"yaw": float(best[0]), "t": [float(x) for x in best[1]], "h": float(h), "inliers": best[2], "rms": best[3]}
json.dump(extra, open(OUT / "poses_overhead.json", "w"), indent=1)

# Put every camera's points (whole frame) into the car frame and keep the car's box.
allp, allc, src_id = [], [], []
for i, cam in enumerate(placed + list(extra)):
    if cam in extra:
        p, rgb, s, h = floor_points(cam, only_car=False)
        pose = extra[cam]
        p = np.column_stack([p[:, :2] @ rot(pose["yaw"]).T + pose["t"], p[:, 2]])
    else:
        p, rgb = fused[f"{cam}_xyz"].astype(float), fused[f"{cam}_rgb"]
    m = (np.abs(p[:, 0]) < 2.6) & (np.abs(p[:, 1]) < 1.0) & (p[:, 2] > 0.08) & (p[:, 2] < 1.6)
    allp.append(p[m]); allc.append(rgb[m]); src_id.append(np.full(m.sum(), i))
P, C, S = np.concatenate(allp), np.concatenate(allc), np.concatenate(src_id)
np.savez_compressed(OUT / "fused_all_f10.npz", xyz=P.astype(np.float32), rgb=C, cam=S, names=np.array(placed + list(extra)))

FACES = {"front": (0, 1), "rear": (0, -1), "left": (1, 1), "right": (1, -1), "top": (2, 1)}


def coverage(P, cell=0.1, shell=0.25):
    lo, hi = np.percentile(P, [1, 99], axis=0)
    out = {}
    for name, (ax, sgn) in FACES.items():
        edge = hi[ax] if sgn > 0 else lo[ax]
        near = np.abs(P[:, ax] - edge) < shell
        others = [a for a in range(3) if a != ax]
        span = [(lo[a], hi[a]) for a in others]
        n_cells = [max(int(np.ceil((b - a) / cell)), 1) for a, b in span]
        m = int(np.ceil(shell / cell))
        inner = [(m, max(nc - m, m + 1)) for nc in n_cells]
        n_inner = (inner[0][1] - inner[0][0]) * (inner[1][1] - inner[1][0])
        ij = np.floor((P[near][:, others] - [span[0][0], span[1][0]]) / cell).astype(int)
        ok = (ij[:, 0] >= inner[0][0]) & (ij[:, 0] < inner[0][1]) & (ij[:, 1] >= inner[1][0]) & (ij[:, 1] < inner[1][1])
        out[name] = round(100 * len(np.unique(ij[ok], axis=0)) / max(n_inner, 1), 1)
    return out


print("coverage, front+rear cameras:", coverage(P[S < 2]))
print("coverage, all four cameras:  ", coverage(P))
print("old single-pair 9070 fusion:  {'front': 49.0, 'rear': 38.5, 'left': 17.7, 'right': 39.0, 'top': 46.2}")
lo, hi = np.percentile(P, [1, 99], axis=0)
print("fused extent (1-99%):", np.round(hi - lo, 3), " data sheet L x W: 4.850 x 1.816, height 1.470")
