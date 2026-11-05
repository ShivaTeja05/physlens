"""A thrown ball in CMU Panoptic Studio (161029_sports1, 0:80-0:92) seen by 10 synchronised, calibrated HD cameras:
found in each view, triangulated with the cameras' own calibration (no pose guessing), outliers rejected by
re-projection error, and the free flight checked against gravity (9.81 m/s²) as the accuracy test.

Detection per view: the moving parts of the picture (difference to the clip's median frame), kept where the blob
is ball-sized and roughly round, away from the people (largest moving blobs). Every candidate is then tested in 3D:
a ball position must re-project onto a candidate in at least 3 views within a few pixels.
Usage: python3 benchmark/panoptic_ball.py   (needs datasets/panoptic/161029_sports1, see its SOURCE.md)
Writes benchmark/panoptic_ball.json.
"""
import itertools
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "datasets" / "panoptic" / "161029_sports1"
OUT = ROOT / "benchmark" / "panoptic_ball.json"
FPS = 30000 / 1001
T0 = 80.0                      # clip start in the sequence (s)


def cameras():
    calib = json.loads((D / "calibration_161029_sports1.json").read_text())
    cams = {}
    for c in calib["cameras"]:
        if c["type"] != "hd" or not (D / "hd" / f"hd_{c['name']}.mp4").exists():
            continue
        K, dist = np.array(c["K"]), np.array(c["distCoef"])
        R, t = np.array(c["R"]), np.array(c["t"]).reshape(3, 1) / 100.0      # cm -> m
        cams[c["name"]] = {"K": K, "dist": dist, "R": R, "t": t, "P": K @ np.hstack([R, t]),
                           "C": (-R.T @ t).ravel()}
    return cams


def frames(name, stride=1):
    cap = cv2.VideoCapture(str(D / "hd" / f"hd_{name}.mp4"))
    out = []
    i = 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % stride == 0:
            out.append(im)
        i += 1
    return out


def candidates(ims, K, dist, max_per_frame=6):
    """Per frame: undistorted centres of ball-sized, roundish moving blobs (not the people)."""
    small = [cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), None, fx=0.5, fy=0.5) for im in ims]
    bg = np.median(np.stack(small[:: max(len(small) // 40, 1)]), 0).astype(np.uint8)
    out = []
    for g in small:
        diff = cv2.GaussianBlur(cv2.absdiff(g, bg), (5, 5), 0)
        _, m = cv2.threshold(diff, 28, 255, cv2.THRESH_BINARY)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m)
        cs = []
        for k in range(1, n):
            x, y, w, h, a = stats[k]
            if 12 <= a <= 900 and 0.5 <= w / max(h, 1) <= 2.0 and a / (w * h) > 0.45:
                cs.append(cent[k] * 2.0)                     # back to full resolution
        cs = np.array(cs[:max_per_frame]) if cs else np.zeros((0, 2))
        if len(cs):
            cs = cv2.undistortPoints(cs.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
        out.append(cs)
    return out


def triangulate(Ps, xs):
    A = []
    for P, (u, v) in zip(Ps, xs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    X = np.linalg.svd(np.array(A))[2][-1]
    return X[:3] / X[3]


def reproject(P, X):
    x = P @ np.append(X, 1.0)
    return x[:2] / x[2]


def ball_at(cams, cand, f, tol_px=6.0):
    """Best 3D point for frame f: try pairs of candidates from two views, keep the one most views agree with."""
    names = [n for n in cams if len(cand[n][f])]
    best = None
    for a, b in itertools.combinations(names, 2):
        for xa in cand[a][f]:
            for xb in cand[b][f]:
                X = triangulate([cams[a]["P"], cams[b]["P"]], [xa, xb])
                if not (-0.5 < -X[1] < 4.0):                 # above the floor, below the dome top (y points down)
                    continue
                support = {}
                for n in names:
                    d = np.linalg.norm(cand[n][f] - reproject(cams[n]["P"], X), axis=1)
                    j = int(np.argmin(d))
                    if d[j] < tol_px:
                        support[n] = cand[n][f][j]
                if len(support) >= 3 and (best is None or len(support) > len(best[1])):
                    best = (X, support)
    if best is None:
        return None
    X = triangulate([cams[n]["P"] for n in best[1]], list(best[1].values()))
    err = [float(np.linalg.norm(reproject(cams[n]["P"], X) - x)) for n, x in best[1].items()]
    return {"X": X, "views": sorted(best[1]), "reproj_px": float(np.sqrt(np.mean(np.square(err))))}


def fit_flight(ts, Xs, min_len=10):
    """Longest run of consecutive samples whose height fits a parabola; g from its curvature."""
    best = None
    step = float(np.median(np.diff(ts))) if len(ts) > 1 else 1 / FPS     # sampling interval (every frame or every 2nd)
    for i in range(len(ts)):
        for j in range(i + min_len, len(ts) + 1):
            t, h = np.array(ts[i:j]), -np.array([X[1] for X in Xs[i:j]])      # height = -y
            if np.max(np.diff(t)) > 1.6 * step:                 # no gaps in the run
                continue
            c = np.polyfit(t, h, 2)
            res = h - np.polyval(c, t)
            if np.sqrt(np.mean(res ** 2)) < 0.02 and (best is None or j - i > best[1] - best[0]):
                best = (i, j, c, float(np.sqrt(np.mean(res ** 2))))
    if best is None:
        return None
    i, j, c, rms = best
    t = np.array(ts[i:j])
    # horizontal motion should be straight and steady in free flight
    xz = np.array([[X[0], X[2]] for X in Xs[i:j]])
    vx = np.polyfit(t, xz, 1)[0]
    return {"frames": [i, j], "duration_s": round(float(t[-1] - t[0]), 3), "g": round(float(-2 * c[0]), 3),
            "height_fit_rms_m": round(rms, 4), "horizontal_speed_mps": round(float(np.linalg.norm(vx)), 3)}


if __name__ == "__main__":
    cams = cameras()
    print(f"{len(cams)} calibrated cameras:", ", ".join(cams))
    cand = {}
    for n, c in cams.items():
        ims = frames(n)
        cand[n] = candidates(ims, c["K"], c["dist"])
        print(f"  {n}: {len(ims)} frames, {sum(len(x) for x in cand[n]) / max(len(ims), 1):.1f} candidates a frame")
    nf = min(len(v) for v in cand.values())
    track = []
    for f in range(nf):
        b = ball_at(cams, cand, f)
        if b:
            track.append({"frame": f, "t": round(T0 + f / FPS, 4), **{k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in b.items()}})
    print(f"ball found in {len(track)} of {nf} frames;",
          f"median re-projection {np.median([b['reproj_px'] for b in track]):.2f} px," if track else "",
          f"median views {np.median([len(b['views']) for b in track]):.0f}" if track else "")
    flight = fit_flight([b["t"] for b in track], [np.array(b["X"]) for b in track]) if track else None
    print("free flight:", flight)
    OUT.write_text(json.dumps({"sequence": "161029_sports1", "clip_start_s": T0, "cameras": list(cams), "track": track,
                               "flight": flight, "g_reference": 9.81}, indent=1))
    print("wrote", OUT)
