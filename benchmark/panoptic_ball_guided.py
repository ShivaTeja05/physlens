"""Second pass for the football: SAM 3.1 on the whole frame misses it where it is small (far cameras), so the flight
from the frames where 2+ views did find it (a first ballistic fit: straight across, parabola in height) predicts
where the ball must be in every view, and SAM 3.1 runs again on a 360 px crop there. Then: triangulation with the
cameras' calibration, leave-worst-out until every view re-projects within 8 px, and the gravity check.
Crops cached in datasets/panoptic/161029_sports1/sam_ball_crops.json (re-runs make no calls; ~$0.0025 per crop).
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/panoptic_ball_guided.py
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv  # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from panoptic_ball import D, FPS, T0, cameras, fit_flight, reproject, triangulate  # noqa: E402
from panoptic_ball_sam import FRAMES, frame  # noqa: E402

FULL = json.loads((D / "sam_ball.json").read_text())
CACHE = D / "sam_ball_crops.json"
OUT = ROOT / "benchmark" / "panoptic_ball.json"
HALF = 180


def undist(c, xy):
    return cv2.undistortPoints(np.array([[xy]], float), c["K"], c["dist"], P=c["K"]).reshape(2)


def distort(c, uv):
    """Undistorted pixel -> the raw image pixel, so crops are cut where the ball really is in the video."""
    K, dist = c["K"], c["dist"]
    x = np.linalg.inv(K) @ np.array([uv[0], uv[1], 1.0])
    pts, _ = cv2.projectPoints(np.array([[x[0], x[1], 1.0]]), np.zeros(3), np.zeros(3), K, dist)
    return pts.reshape(2)


def seed_track(cams):
    pts = []
    for i in FRAMES:
        obs = {n: undist(cams[n], max(d, key=lambda e: e["conf"] or 0)["c"]) for n in cams if (d := FULL.get(f"{n}:{i}"))}
        if len(obs) >= 2:
            X = triangulate([cams[n]["P"] for n in obs], list(obs.values()))
            pts.append((i / FPS, X))
    t = np.array([p[0] for p in pts])
    X = np.array([p[1] for p in pts])
    # ballistic model in the world frame: x, z linear in time, y (down) quadratic
    cx, cz = np.polyfit(t, X[:, 0], 1), np.polyfit(t, X[:, 2], 1)
    cy = np.polyfit(t, X[:, 1], 2) if len(t) >= 3 else np.append(0.0, np.polyfit(t, X[:, 1], 1))
    print(f"seed: {len(pts)} frames with 2+ views; first-guess g = {2 * cy[0]:.2f} m/s²")
    return lambda tt: np.array([np.polyval(cx, tt), np.polyval(cy, tt), np.polyval(cz, tt)])


def crop_detect(job):
    from physlens.perception.sam_image import segment
    name, i, cx, cy = job
    im = frame(name, i)
    h, w = im.shape[:2]
    x0, y0 = int(np.clip(cx - HALF, 0, w - 2 * HALF)), int(np.clip(cy - HALF, 0, h - 2 * HALF))
    crop = im[y0:y0 + 2 * HALF, x0:x0 + 2 * HALF]
    out = []
    for inst in segment(crop, "football", what=f"panoptic ball crop {name} f{i}"):
        ys, xs = np.nonzero(inst.mask)
        if 30 <= len(xs) <= 60000:
            out.append({"c": [float(xs.mean() + x0), float(ys.mean() + y0)], "area": int(len(xs)), "conf": inst.confidence})
    return name, i, out


if __name__ == "__main__":
    cams = cameras()
    predict = seed_track(cams)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = []
    for i in FRAMES:
        X = predict(i / FPS)
        for n, c in cams.items():
            if f"{n}:{i}" in cache:
                continue
            raw = distort(c, reproject(c["P"], X))
            if 0 <= raw[0] < 1920 and 0 <= raw[1] < 1080 and (c["R"] @ X + c["t"].ravel())[2] > 0:
                jobs.append((n, i, float(raw[0]), float(raw[1])))
    print(f"{len(jobs)} guided crops to run ({len(cache)} cached)")
    with ThreadPoolExecutor(6) as pool:
        for name, i, dets in pool.map(crop_detect, jobs):
            cache[f"{name}:{i}"] = dets
            CACHE.write_text(json.dumps(cache))

    track = []
    for i in FRAMES:
        Xp = predict(i / FPS)
        obs = {}
        for n, c in cams.items():
            dets = cache.get(f"{n}:{i}", []) + FULL.get(f"{n}:{i}", [])
            if not dets:
                continue
            # the detection nearest to where the flight says the ball is
            p = reproject(c["P"], Xp)
            uv = [undist(c, d["c"]) for d in dets]
            k = int(np.argmin([np.linalg.norm(u - p) for u in uv]))
            if np.linalg.norm(uv[k] - p) < 250:
                obs[n] = uv[k]
        names = list(obs)
        if len(names) < 3:
            continue
        while True:
            X = triangulate([cams[n]["P"] for n in names], [obs[n] for n in names])
            err = {n: float(np.linalg.norm(reproject(cams[n]["P"], X) - obs[n])) for n in names}
            worst = max(err, key=err.get)
            if err[worst] < 8 or len(names) <= 3:
                break
            names.remove(worst)
        rms = float(np.sqrt(np.mean([err[n] ** 2 for n in names])))
        if rms < 8:
            track.append({"frame": i, "t": round(T0 + i / FPS, 4), "X": X.tolist(), "views": names, "reproj_px": round(rms, 2)})
    for b in track:
        print(f"  frame {b['frame']:3d}  height {-b['X'][1]:.3f} m  views {len(b['views'])}  re-projection {b['reproj_px']} px")
    flight = fit_flight([b["t"] for b in track], [np.array(b["X"]) for b in track], min_len=6) if len(track) >= 6 else None
    print("free flight:", flight, "(gravity 9.81)")
    OUT.write_text(json.dumps({"sequence": "161029_sports1", "clip_start_s": T0, "frames": FRAMES,
                               "detector": "SAM 3.1, whole frame then crops guided by a first ballistic fit",
                               "cameras": list(cams), "track": track, "flight": flight, "g_reference": 9.81}, indent=1))
