"""The football's first flight in CMU Panoptic 161029_sports1 (clip frames 16-46, every 2nd), found by SAM 3.1 in each of
10 calibrated views, triangulated with the cameras' own calibration, and checked against gravity.
Colour can't separate it (skin matches, the ball reads grey-brown under the dome lights); motion picks up hands and
heads. SAM 3.1 recognises the object. Cost: about $0.0025 per image (160 images); masks cached in
datasets/panoptic/161029_sports1/sam_ball.json so a re-run makes no calls.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/panoptic_ball_sam.py
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

FRAMES = list(range(16, 47, 2))
CACHE = D / "sam_ball.json"
OUT = ROOT / "benchmark" / "panoptic_ball.json"


def frame(name, i):
    cap = cv2.VideoCapture(str(D / "hd" / f"hd_{name}.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, im = cap.read()
    return im if ok else None


def detect(job):
    from physlens.perception.sam_image import segment
    name, i = job
    im = frame(name, i)
    if im is None:
        return name, i, []
    out = []
    for inst in segment(im, "american football", what=f"panoptic ball {name} f{i}"):
        ys, xs = np.nonzero(inst.mask)
        area = len(xs)
        if 80 <= area <= 40000:                        # a ball, not a person or the floor
            out.append({"c": [float(xs.mean()), float(ys.mean())], "area": int(area), "conf": inst.confidence})
    return name, i, out


if __name__ == "__main__":
    cams = cameras()
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = [(n, i) for n in cams for i in FRAMES if f"{n}:{i}" not in cache]
    print(f"{len(jobs)} SAM 3.1 images to run ({len(cache)} cached)")
    with ThreadPoolExecutor(6) as pool:
        for name, i, dets in pool.map(detect, jobs):
            cache[f"{name}:{i}"] = dets
            CACHE.write_text(json.dumps(cache))
    track = []
    for i in FRAMES:
        obs = {}
        for n, c in cams.items():
            dets = cache.get(f"{n}:{i}", [])
            if dets:
                best = max(dets, key=lambda d: d["conf"] or 0)
                obs[n] = cv2.undistortPoints(np.array([[best["c"]]], float), c["K"], c["dist"], P=c["K"]).reshape(2)
        if len(obs) < 2:
            continue
        # leave-worst-out until every view re-projects within 8 px (or only 2 remain)
        names = list(obs)
        while True:
            X = triangulate([cams[n]["P"] for n in names], [obs[n] for n in names])
            err = {n: float(np.linalg.norm(reproject(cams[n]["P"], X) - obs[n])) for n in names}
            worst = max(err, key=err.get)
            if err[worst] < 8 or len(names) <= 2:
                break
            names.remove(worst)
        if len(names) >= 3:
            track.append({"frame": i, "t": round(T0 + i / FPS, 4), "X": X.tolist(), "views": names,
                          "reproj_px": round(float(np.sqrt(np.mean([err[n] ** 2 for n in names]))), 2)})
    for b in track:
        print(f"  frame {b['frame']:3d}  height {-b['X'][1]:.3f} m  views {len(b['views'])}  re-projection {b['reproj_px']} px")
    flight = fit_flight([b["t"] for b in track], [np.array(b["X"]) for b in track], min_len=6) if len(track) >= 6 else None
    print("free flight:", flight, "(gravity 9.81)")
    OUT.write_text(json.dumps({"sequence": "161029_sports1", "clip_start_s": T0, "frames": FRAMES, "detector": "SAM 3.1",
                               "cameras": list(cams), "track": track, "flight": flight, "g_reference": 9.81}, indent=1))
