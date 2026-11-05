"""Does every camera see the same floor? RANSAC a plane per camera (normal close to the overhead camera's view
direction), then compare normals (degrees) and heights (metres) across cameras. Agreement = trustworthy joint poses."""
import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "out"
FRAME = int(sys.argv[1]) if len(sys.argv) > 1 else 10
tag = sys.argv[2] if len(sys.argv) > 2 else ""
s = json.loads((OUT / f"f{FRAME}{tag}_summary.json").read_text())
down = np.array(s["cam01_overhead_wide"]["forward"], float)
down /= np.linalg.norm(down)
rng = np.random.default_rng(0)


def ransac_floor(p, iters=3000, tol=0.03):
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if n @ down < 0:
            n = -n
        if n @ down < np.cos(np.radians(35)):     # roughly horizontal only
            continue
        d = n @ a
        inl = np.abs(p @ n - d) < tol
        if inl.sum() > best_n:
            best, best_n = (n, d), inl.sum()
    n, d = best
    inl = np.abs(p @ n - d) < tol
    # refine by least squares on inliers
    q = p[inl]
    c0 = q.mean(0)
    _, _, vt = np.linalg.svd(q - c0)
    n = vt[-1] * np.sign(vt[-1] @ down)
    return n, n @ c0, inl.mean()


res = {}
for c in s:
    z = np.load(OUT / f"f{FRAME}{tag}_{c}.npz")
    p = z["pts"].reshape(-1, 3)
    conf = z["conf"].reshape(-1) if z["conf"].size == p.shape[0] else np.ones(len(p))
    ok = np.isfinite(p).all(1) & (conf >= np.quantile(conf, 0.3))
    p = p[ok]
    p = p[rng.choice(len(p), min(len(p), 60000), replace=False)]
    n, d, frac = ransac_floor(p)
    res[c] = {"n": n, "d": d, "frac": frac}

ref = np.mean([r["n"] for r in res.values()], 0)
ref /= np.linalg.norm(ref)
print(f"{'camera':22s} {'tilt vs mean':>12s} {'height (m)':>11s} {'floor share':>11s}")
for c, r in res.items():
    ang = np.degrees(np.arccos(np.clip(r["n"] @ ref, -1, 1)))
    print(f"{c:22s} {ang:11.2f}° {r['d']:11.3f} {r['frac']:10.1%}")
ds = [r["d"] for r in res.values()]
print(f"height spread: {max(ds) - min(ds):.3f} m")
json.dump({c: {"normal": r["n"].tolist(), "d": float(r["d"]), "share": float(r["frac"])} for c, r in res.items()},
          open(OUT / f"floors_f{FRAME}{tag}.json", "w"), indent=1)
