"""Build the caches the two-body crash scripts read (records/real_cache/<name>/): the de-duplicated frames, MoGe-3
depth on a few frames (GPU worker) and the vehicles' outlines (SAM 3.1, or colour for the orange barrier cart).
Needs the videos (scripts/fetch_datasets.py) and both keys in backend/.env. Each call is cached: re-running skips
what exists.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/prepare_cache.py 8555 | 7371 | 10072
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")

NH = ROOT / "datasets" / "nhtsa"


def read(video):
    cap = cv2.VideoCapture(str(video))
    out = []
    while True:
        ok, x = cap.read()
        if not ok:
            break
        out.append(x)
    if not out:
        sys.exit(f"no frames in {video}: fetch it first (scripts/fetch_datasets.py)")
    return out


def depth(cache, U, ks):
    from physlens.geometry.client import sharp_depth
    todo = [k for k in ks if not (cache / f"moge_{k}.npz").exists()]
    for k, s in zip(todo, sharp_depth([U[k] for k in todo]) if todo else []):
        np.savez_compressed(cache / f"moge_{k}.npz", d=s.depth.astype(np.float32), m=s.mask, K=s.intrinsics)
        print("depth", k)


def largest(instances, shape):
    return max(instances, key=lambda i: i.mask.sum()).mask if instances else np.zeros(shape, bool)


def orange_cart(img):
    """The OMDB's orange truss: orange pixels joined up, the largest blob's convex hull."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    o = ((hsv[..., 0] >= 3) & (hsv[..., 0] <= 18) & (hsv[..., 1] > 150) & (hsv[..., 2] > 90)).astype(np.uint8)
    o = cv2.morphologyEx(o, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nl, lab, st, _ = cv2.connectedComponentsWithStats(o)
    m = np.isin(lab, [b for b in range(1, nl) if st[b, 4] > 300]).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(m)
    if cnts:
        cv2.drawContours(out, [cv2.convexHull(max(cnts, key=cv2.contourArea))], -1, 1, -1)
    return out > 0


def main(name):
    from physlens.perception.sam_image import segment
    if name == "8555":
        import rear8555
        cache = rear8555.CACHE
        cache.mkdir(parents=True, exist_ok=True)
        U = rear8555.unique_frames()                                     # one frame per clock reading
        depth(cache, U, [0, 185, 260, 350])
        for k in (0, 185, 350):
            if not (cache / f"masks_{k}.npz").exists():
                np.savez_compressed(cache / f"masks_{k}.npz", jeep=largest(segment(U[k], "silver suv"), U[k].shape[:2]),
                                    truck=largest(segment(U[k], "red pickup truck"), U[k].shape[:2]))
        if not (cache / "masks_260.npz").exists():                       # SAM gives both vehicles as one mask here
            both = largest(segment(U[260], "car"), U[260].shape[:2])
            np.savez_compressed(cache / "masks_260.npz", jeep=np.zeros_like(both), truck=both)
    elif name == "7371":
        cache = ROOT / "records" / "real_cache" / "yaris7371"
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / "frames_u8.npy"
        if not f.exists():
            np.save(f, np.array(read(NH / "7371" / "cam09_overhead_wide.mp4")[3:]))   # frames 0–2 repeat −18.62 ms
        U = np.load(f)
        ks = [0, 57, 100, 147, 200, 250]
        depth(cache, U, ks)
        for k in ks:
            if not (cache / f"masks_{k}.npz").exists():
                np.savez_compressed(cache / f"masks_{k}.npz", *[i.mask for i in segment(U[k], "car")])
    elif name == "10072":
        cache = ROOT / "records" / "real_cache" / "omdb10072"
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / "frames_u8.npy"
        if not f.exists():
            np.save(f, np.array(read(NH / "10072" / "cam07_overhead_omdb.mp4")[1:]))  # frame 0 repeats −50 ms
        U = np.load(f)
        ks = [0, 50, 100, 200]
        depth(cache, U, ks)
        for k in ks:
            if not (cache / f"masks_{k}.npz").exists():
                np.savez_compressed(cache / f"masks_{k}.npz", car=largest(segment(U[k], "car"), U[k].shape[:2]),
                                    cart=orange_cart(U[k]))
    else:
        sys.exit("usage: prepare_cache.py 8555 | 7371 | 10072")
    print("cache ready:", cache)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "")
