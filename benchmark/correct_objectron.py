"""Does a scale correction (or a near/far rule) really help, on clips it was never fitted on?

The size test shows objects coming out systematically small (median scale about ×0.82). A single correction factor
c = 1 / median(measured/true) fixes that on the clips it was fitted on by construction, so it's only worth something
if it holds on clips it hasn't seen. Two held-out schemes:

  5-fold by clip         fit on 4/5 of the clips, score the other 1/5, five times.
  leave one category out fit on 8 categories, score the 9th: the honest test for a new kind of object.

Also the near/far rule: use the phone-motion triangulation (tapname_tri) only when MapAnything puts the object
nearer than d0, otherwise tapname. d0 and both corrections are chosen on the training clips only.

  python correct_objectron.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(0)
B = 4000


def load(mode: str) -> dict[str, dict]:
    rows = json.loads((HERE / f"objectron_m1_{mode}.json").read_text())["rows"]
    out = {}
    for r in rows:
        ratios = [s["ratio"] for s in r.get("sides", []) if s["scored"]]
        if ratios:
            out[r["clip"]] = {"ratios": np.array(ratios), "dist": r["measured_distance_m"],
                              "cat": r["clip"].rsplit("_batch-", 1)[0]}
    return out


def fit_c(rows: list[dict]) -> float:
    return 1.0 / float(np.median(np.concatenate([r["ratios"] for r in rows])))


def fit_dist(rows: list[dict]) -> tuple[float, float]:
    """log(measured/true) = a + b·log(distance), fitted on per-side ratios. Close objects come out small."""
    x = np.concatenate([np.full(len(r["ratios"]), np.log(r["dist"])) for r in rows])
    y = np.log(np.concatenate([r["ratios"] for r in rows]))
    b, a = np.polyfit(x, y, 1)
    return float(a), float(b)


def dist_corrected(data: dict, folds) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    errs, fits = [], []
    for train, test in folds:
        a, b = fit_dist([data[x] for x in train])
        fits.append((a, b))
        errs += [np.abs(data[x]["ratios"] * np.exp(-(a + b * np.log(data[x]["dist"]))) - 1) for x in test]
    return errs, fits


def score(per_clip: list[np.ndarray]) -> str:
    flat = np.concatenate(per_clip)
    idx = rng.integers(0, len(per_clip), (B, len(per_clip)))
    meds = [np.median(np.concatenate([per_clip[i] for i in row])) for row in idx]
    lo, hi = np.percentile(meds, [2.5, 97.5])
    return (f"{np.median(flat)*100:5.1f}% ({lo*100:4.1f}–{hi*100:4.1f}%)  ≤10% {(flat <= .1).mean()*100:3.0f}%  "
            f"≤20% {(flat <= .2).mean()*100:3.0f}%")


def errors(rows: list[dict], c: float = 1.0) -> list[np.ndarray]:
    return [np.abs(r["ratios"] * c - 1) for r in rows]


def folds_by_clip(clips: list[str], k: int = 5):
    order = list(rng.permutation(clips))
    for i in range(k):
        test = set(order[i::k])
        yield [c for c in clips if c not in test], sorted(test)


def folds_by_category(clips: list[str], data: dict):
    for cat in sorted({data[c]["cat"] for c in clips}):
        yield [c for c in clips if data[c]["cat"] != cat], [c for c in clips if data[c]["cat"] == cat]


def corrected(data: dict, folds) -> tuple[list[np.ndarray], list[float]]:
    errs, cs = [], []
    for train, test in folds:
        c = fit_c([data[x] for x in train])
        cs.append(c)
        errs += errors([data[x] for x in test], c)
    return errs, cs


def near_far(name: dict, tri: dict, folds, candidates=np.arange(0.2, 2.01, 0.1)):
    """Per fold: corrections for both variants and the switch distance d0, all from the training clips."""
    errs, picks = [], []
    for train, test in folds:
        c_n, c_t = fit_c([name[x] for x in train]), fit_c([tri[x] for x in train])

        def errs_for(clips, d0):
            return [np.abs(tri[x]["ratios"] * c_t - 1) if name[x]["dist"] < d0 else np.abs(name[x]["ratios"] * c_n - 1)
                    for x in clips]

        d0 = min(candidates, key=lambda d: float(np.median(np.concatenate(errs_for(train, d)))))
        picks.append(round(float(d0), 1))
        errs += errs_for(test, d0)
    return errs, picks


def main() -> None:
    name, tri = load("tapname"), load("tapname_tri")
    clips = sorted(set(name) & set(tri))
    print(f"{len(clips)} clips measured by both variants\n")

    for label, data in (("name + tap (tapname)", name), ("phone-motion triangulation (tapname_tri)", tri)):
        rows = [data[c] for c in clips]
        print(f"{label}")
        print(f"  no correction           {score(errors(rows))}")
        e, cs = corrected(data, folds_by_clip(clips))
        print(f"  5-fold by clip          {score(e)}   c = {min(cs):.2f}–{max(cs):.2f}")
        e, cs = corrected(data, folds_by_category(clips, data))
        print(f"  leave one category out  {score(e)}   c = {min(cs):.2f}–{max(cs):.2f}")
        print(f"  (in-sample, optimistic) {score(errors(rows, fit_c(rows)))}")
        e, f = dist_corrected(data, folds_by_clip(clips))
        print(f"  by distance, 5-fold     {score(e)}   b = {min(x[1] for x in f):.2f}–{max(x[1] for x in f):.2f}")
        e, f = dist_corrected(data, folds_by_category(clips, data))
        print(f"  by distance, leave-cat  {score(e)}   b = {min(x[1] for x in f):.2f}–{max(x[1] for x in f):.2f}\n")

    print("near/far rule: triangulation when MapAnything says the object is nearer than d0, else name + tap")
    e, d = near_far(name, tri, folds_by_clip(clips))
    print(f"  5-fold by clip          {score(e)}   d0 = {sorted(set(d))} m")
    e, d = near_far(name, tri, folds_by_category(clips, name))
    print(f"  leave one category out  {score(e)}   d0 = {sorted(set(d))} m")

    print("\nper category, leave one category out: uncorrected name+tap / name+tap ×c / triangulation ×c / near-far /"
          " name+tap by distance / triangulation by distance")
    e_0 = [np.abs(name[c]["ratios"] - 1) for _, test in folds_by_category(clips, name) for c in test]
    e_n, _ = corrected(name, folds_by_category(clips, name))
    e_t, _ = corrected(tri, folds_by_category(clips, tri))
    e_r, _ = near_far(name, tri, folds_by_category(clips, name))
    e_nd, _ = dist_corrected(name, folds_by_category(clips, name))
    e_td, _ = dist_corrected(tri, folds_by_category(clips, tri))
    order = [c for _, test in folds_by_category(clips, name) for c in test]
    for cat in sorted({name[c]["cat"] for c in clips}):
        idx = [i for i, c in enumerate(order) if name[c]["cat"] == cat]
        med = lambda E: np.median(np.concatenate([E[i] for i in idx])) * 100
        print(f"  {cat:11} " + "  ".join(f"{med(E):5.1f}%" for E in (e_0, e_n, e_t, e_r, e_nd, e_td)))


if __name__ == "__main__":
    main()
