"""Compare size-measurement variants on Objectron with honest error bars.

Reads objectron_m1_<mode>.json for each mode given. For each: median size error over the scored sides with a 95%
bootstrap interval (resampling whole clips, since sides of one object share its errors), sides within 10% / 20%, and
the median scale. Then, on the clips every mode measured, the paired difference from the first mode (also bootstrapped
by clip): a variant only really beats the baseline if that interval sits below zero. Also a per-category table.

  python compare_objectron.py tapname motion box tapname_tri fused_median tapname_median
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
B = 4000
rng = np.random.default_rng(0)


def load(mode: str) -> dict[str, list[float]]:
    rows = json.loads((HERE / f"objectron_m1_{mode}.json").read_text())["rows"]
    return {r["clip"]: [abs(s["ratio"] - 1) for s in r["sides"] if s["scored"]]
            for r in rows if "sides" in r and any(s["scored"] for s in r["sides"])}


def scale(mode: str) -> float:
    rows = json.loads((HERE / f"objectron_m1_{mode}.json").read_text())["rows"]
    return float(np.median([r["median_ratio"] for r in rows if r.get("median_ratio")]))


def boot_median(per_clip: list[list[float]]) -> tuple[float, float, float]:
    flat = np.concatenate(per_clip)
    idx = rng.integers(0, len(per_clip), (B, len(per_clip)))
    meds = [np.median(np.concatenate([per_clip[i] for i in row])) for row in idx]
    return float(np.median(flat)), float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def main(modes: list[str]) -> None:
    data = {m: load(m) for m in modes}
    print(f"{'mode':16} {'clips':>5} {'sides':>5}  {'median error (95% CI)':>24}  {'≤10%':>6} {'≤20%':>6}  scale")
    for m in modes:
        per = list(data[m].values())
        flat = np.concatenate(per)
        med, lo, hi = boot_median(per)
        print(f"{m:16} {len(per):5} {len(flat):5}  {med*100:8.1f}%  ({lo*100:4.1f}–{hi*100:4.1f}%)    "
              f"{(flat <= .1).mean()*100:5.0f}% {(flat <= .2).mean()*100:5.0f}%  ×{scale(m):.2f}")

    base = modes[0]
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"\nPaired vs {base} on the {len(common)} clips every mode measured (per-clip median error, negative = better):")
    for m in modes[1:]:
        diff = np.array([np.median(data[m][c]) - np.median(data[base][c]) for c in common])
        boots = [np.median(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(B)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "better" if hi < 0 else "worse" if lo > 0 else "no clear difference"
        print(f"  {m:16} {np.median(diff)*100:+6.1f} points  (95% CI {lo*100:+.1f} to {hi*100:+.1f})  → {verdict}")

    print(f"\nPer category, median error ({' / '.join(modes)}):")
    cats = sorted({c.rsplit('_batch-', 1)[0] for c in common})
    for cat in cats:
        vals = []
        for m in modes:
            e = np.concatenate([data[m][c] for c in common if c.startswith(cat + "_batch-")])
            vals.append(f"{np.median(e)*100:5.1f}%")
        n = sum(c.startswith(cat + "_batch-") for c in common)
        print(f"  {cat:11} ({n:2} clips)  " + "  ".join(vals))


if __name__ == "__main__":
    main(sys.argv[1:] or ["tapname", "motion"])
