"""The documentation camera's radial term from straight lines in the lab (hybrid.lens_from_lines), on several frames
of the right-side pan, outside the car (top band: curtain rail, wall edges; bottom band: floor seams). Lens from
MapAnything's field of view (~100°). The 8555 rule: use the term only when the lines agree (IQR of per-line k1 < 0.1)."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path("/Users/gaddamshivateja/Meta-developer-hackathon/physlens")
sys.path.insert(0, str(ROOT / "backend"))
from physlens.hybrid import lens_from_lines, radial  # noqa: E402

RD = Path(__file__).parent / "out" / "right"
FOV = float(sys.argv[1]) if len(sys.argv) > 1 else 100.0
k1s = np.round(np.arange(-0.8, 0.81, 0.02), 2)
per, used = [], {}
for t in ["39.50", "40.00", "41.00", "42.00", "43.00", "44.00", "45.50"]:
    im = cv2.imread(str(RD / f"t{t}.jpg"))
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    f = w / 2 / np.tan(np.radians(FOV / 2))
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    mask = np.zeros_like(g)
    mask[: int(0.28 * h)] = 255          # above the car: rails, wall edges
    mask[int(0.86 * h):] = 255           # below it: floor seams
    mask[-24:] = 0                       # no burned-in text at the very bottom
    r = lens_from_lines(g, K, mask, min_len=int(0.18 * w), k1s=k1s)
    per += r["per_line"]
    used[t] = {"lines": r["lines"], "per_line": r["per_line"]}
    vis = im.copy()
    for x0, y0, x1, y1 in r["segments"]:
        cv2.line(vis, (x0, y0), (x1, y1), (0, 0, 255), 2)
    cv2.imwrite(str(RD / f"lines_t{t}.jpg"), vis)
per = np.array(per)
q1, q3 = np.percentile(per, [25, 75]) if len(per) else (0, 0)
out = {"fov_deg": FOV, "lines": int(len(per)), "k1_median": float(np.median(per)) if len(per) else None,
       "k1_iqr": round(float(q3 - q1), 3), "per_frame": used}
print(json.dumps({k: v for k, v in out.items() if k != "per_frame"}))
for t, u in used.items():
    print(t, u["lines"], u["per_line"])
(RD / "lens.json").write_text(json.dumps(out, indent=1))
