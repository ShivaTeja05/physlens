"""Register the documentation video's right-side reconstruction to the car frame by the right wheels (front-right at
0:40.0, rear-right at 0:45.5: tyre contacts and wheel centres), with a similarity transform (scale, rotation, shift).
Checks that were not fitted: the floor comes out level, the roof height (1.47 m published), the length, and where the
front and rear faces sit against the crash cameras' measurements (+2.550 / -2.313 m)."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
TAG = sys.argv[1] if len(sys.argv) > 1 else "right"     # reconstruction folder under out/
RD = OUT / TAG
WB, TF, TR, RW = 2.773, 1.585, 1.575, 0.325
MARKS = [("t40.00", (447, 356), (WB / 2, -TF / 2, 0.0)), ("t40.00", (447, 284), (WB / 2, -TF / 2, RW)),
         ("t45.50", (262, 349), (-WB / 2, -TR / 2, 0.0)), ("t45.50", (262, 275), (-WB / 2, -TR / 2, RW))]
TIMES = [f"t{39.5 + 0.5 * k:05.2f}" for k in range(13)]
data = {t: np.load(RD / f"{t}.npz") for t in TIMES}


def point_at(t, u, v, r=2):
    z = data[t]
    P = z["pts"]
    x, y = int(round(u * float(z["sx"]))), int(round(v * float(z["sy"])))
    patch = P[max(y - r, 0):y + r + 1, max(x - r, 0):x + r + 1].reshape(-1, 3)
    return np.median(patch[np.isfinite(patch).all(1)], 0)


A = np.array([point_at(t, *uv) for t, uv, _ in MARKS])
B = np.array([X for _, _, X in MARKS])
# Up from the wheels themselves: each tyre's contact-to-centre line is vertical. Averaged over both wheels and made
# perpendicular to the line between the contacts. (A RANSAC floor was tried and found a plane 45-68 cm off the tyre
# contacts: this reconstruction's floor is bent, most likely by the documentation camera's ~100° distorted lens.)
c_f, c_r = A[0], A[2]
axis = (c_f - c_r) / np.linalg.norm(c_f - c_r)
up = (A[1] - A[0]) / np.linalg.norm(A[1] - A[0]) + (A[3] - A[2]) / np.linalg.norm(A[3] - A[2])
n = up - (up @ axis) * axis
n /= np.linalg.norm(n)
x_ax = axis - (axis @ n) * n
x_ax /= np.linalg.norm(x_ax)
y_ax = np.cross(n, x_ax)
R = np.stack([x_ax, y_ax, n])            # rows: reconstruction -> car frame axes
s = WB / np.linalg.norm((c_f - c_r) - ((c_f - c_r) @ n) * n)
mid = (c_f + c_r) / 2
t = np.array([0.0, -(TF + TR) / 4, 0.0]) - s * R @ mid
res = (s * A @ R.T + t) - B
print("keypoint residuals (cm):", np.round(np.linalg.norm(res, axis=1) * 100, 1), " wheel-centre heights:", np.round((s * A @ R.T + t)[[1, 3], 2], 3), "(0.325)")
P_all, C_all, F_all = [], [], []
for k, tt in enumerate(TIMES):
    z = data[tt]
    P = z["pts"].reshape(-1, 3)
    conf = z["conf"].reshape(-1) if z["conf"].size == len(P) else np.ones(len(P))
    rgb = z["img"].reshape(-1, 3)[:, ::-1]
    ok = np.isfinite(P).all(1) & (conf >= np.quantile(conf, 0.35))
    Q = s * P[ok] @ R.T + t
    P_all.append(Q); C_all.append(rgb[ok]); F_all.append(np.full(len(Q), k))
P, C, F = np.concatenate(P_all), np.concatenate(C_all), np.concatenate(F_all)

# Floor check: points near z=0 away from the car should be flat and level
floor = P[(np.abs(P[:, 2]) < 0.15) & ((P[:, 1] < -1.2) & (P[:, 1] > -3.5)) & (np.abs(P[:, 0]) < 3)]
if len(floor) > 200:
    c0 = floor.mean(0)
    n = np.linalg.svd(floor - c0, full_matrices=False)[2][-1]
    n *= np.sign(n[2])
    print(f"floor beside the car: tilt {np.degrees(np.arccos(abs(n[2]))):.1f}° from level, mean height {c0[2] * 100:+.1f} cm, "
          f"spread {np.std(floor[:, 2]) * 100:.1f} cm ({len(floor)} points)")
car = P[(np.abs(P[:, 0]) < 2.7) & (P[:, 1] > -1.05) & (P[:, 1] < 0.1) & (P[:, 2] > 0.08) & (P[:, 2] < 1.6)]
side = car[(car[:, 1] < -0.5)]
mid = side[(side[:, 2] > 0.35) & (side[:, 2] < 0.9)]
print(f"right side: roof {np.quantile(car[:, 2], 0.99):.3f} m (published 1.47), "
      f"front end {np.quantile(mid[:, 0], 0.99):+.3f} m (crash front camera +2.550), rear end {np.quantile(mid[:, 0], 0.01):+.3f} m (crash rear camera -2.313)")
print(f"length from the right side {np.quantile(mid[:, 0], 0.99) - np.quantile(mid[:, 0], 0.01):.3f} m (data sheet 4.850); "
      f"side plane y = {np.median(side[(side[:, 2] > 0.4) & (side[:, 2] < 0.8)][:, 1]):+.3f} m (half width -0.908)")
np.savez_compressed(OUT / f"{TAG}_car.npz", xyz=car.astype(np.float32),
                    rgb=C[(np.abs(P[:, 0]) < 2.7) & (P[:, 1] > -1.05) & (P[:, 1] < 0.1) & (P[:, 2] > 0.08) & (P[:, 2] < 1.6)])
json.dump({"s": float(s), "R": R.tolist(), "t": t.tolist()}, open(OUT / f"{TAG}_transform.json", "w"))
