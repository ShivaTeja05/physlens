"""Solid bodies for CMU Panoptic 161029_sports1 by space carving (the visual hull): the capture volume is cut into
cubes, coarse (4 cm) then fine (1 cm) around what survives, and a cube is kept when it lies inside the moving
outline (people, ball) in nearly every calibrated HD camera that sees it. With 30 cameras all round, what survives is
each body as one solid, consistent from every side by construction. Its surface cubes are coloured from the camera
that faces each surface most squarely. Camera 00_15 is held out of the carving and checks the result: the solid's
outline projected into it against its own outline (IoU), and the stereo points' distance to the solid's surface.
Usage: backend/.venv/bin/python benchmark/panoptic_hull.py [frame ...]   (CPU; default frame 30)
Writes datasets/panoptic/161029_sports1/hull/f{frame}.npz (surface points, Panoptic world frame, metres, y down).
"""
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from panoptic_ball import D, cameras  # noqa: E402
from panoptic_consistency import HOLDOUT, SCALE, video  # noqa: E402

_bgc: dict = {}


def outline(name, colour_frames, i, thresh=16):
    """A person's outline for carving: colour change against the clip's median frame, closed, holes filled (a grey
    shirt on the grey dome leaves holes inside the outline; a body has none)."""
    if name not in _bgc:
        _bgc[name] = np.median(np.stack(colour_frames[name][::6]), 0).astype(np.int16)
    lab = cv2.cvtColor(colour_frames[name][i], cv2.COLOR_BGR2LAB).astype(np.int16)
    bgl = cv2.cvtColor(_bgc[name].astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.int16)
    m = (np.abs(lab - bgl).max(-1) > thresh).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    m = ndimage.binary_fill_holes(m)
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0


def colour_video(name):
    cap = cv2.VideoCapture(str(D / "hd" / f"hd_{name}.mp4"))
    out = []
    while True:
        ok, im = cap.read()
        if not ok:
            break
        out.append(cv2.resize(im, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA))
    return out

OUT = D / "hull"
OUT.mkdir(exist_ok=True)
MISSES = 4          # a cube may fall outside the outline in this many cameras (outlines can still miss a grey shirt on the grey dome)


def project(c, P):
    Xc = (c["R"] @ P.T + c["t"]).T
    px, _ = cv2.projectPoints(P.astype(np.float64), cv2.Rodrigues(c["R"])[0], c["t"], c["K"], c["dist"])
    return (px.reshape(-1, 2) * SCALE).round().astype(int), Xc[:, 2] > 0.2


def carve(cams, masks, names, P):
    """For cube centres P: number of cameras that see them, and number in which they fall inside the outline."""
    seen = np.zeros(len(P), np.int16)
    inside = np.zeros(len(P), np.int16)
    for n in names:
        px, front = project(cams[n], P)
        h, w = masks[n].shape
        ok = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h)
        seen += ok
        hit = np.zeros(len(P), bool)
        hit[ok] = masks[n][px[ok, 1], px[ok, 0]]
        inside += hit
    return seen, inside


def grid(lo, hi, step):
    axes = [np.arange(a + step / 2, b, step) for a, b in zip(lo, hi)]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), -1)
    return g.reshape(-1, 3), [len(a) for a in axes]


def solid(cams, masks, names):
    # coarse pass over the capture volume: x, z in [-2.2, 2.2], height 0-2.5 m (y points down)
    lo, hi = np.array([-2.2, -2.5, -2.2]), np.array([2.2, 0.0, 2.2])
    P, shape = grid(lo, hi, 0.04)
    seen, inside = carve(cams, masks, names, P)
    occ = (seen >= 8) & (inside >= seen - MISSES)
    occ = ndimage.binary_dilation(occ.reshape(shape), iterations=1).reshape(-1)      # margin for the fine pass
    # fine pass: 1 cm cubes inside the surviving coarse cubes
    fine = []
    offs = (np.stack(np.meshgrid(*[np.arange(4)] * 3, indexing="ij"), -1).reshape(-1, 3) - 1.5) * 0.01
    for chunk in np.array_split(P[occ], max(1, int(occ.sum()) // 20000 + 1)):
        Q = (chunk[:, None, :] + offs[None]).reshape(-1, 3)
        s, i = carve(cams, masks, names, Q)
        fine.append(Q[(s >= 8) & (i >= s - MISSES)])
    Q = np.concatenate(fine) if fine else np.zeros((0, 3))
    if len(Q) == 0:
        return Q, Q
    # voxel indices on the 1 cm lattice; surface = occupied with an empty 6-neighbour
    idx = np.round((Q - lo) / 0.01 - 0.5).astype(int)
    key = {tuple(v): k for k, v in enumerate(idx)}
    surf = []
    normals = []
    for k, v in enumerate(idx):
        empty = [d for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
                 if (v[0] + d[0], v[1] + d[1], v[2] + d[2]) not in key]
        if empty:
            surf.append(k)
            nvec = np.sum(empty, 0).astype(float)
            normals.append(nvec / (np.linalg.norm(nvec) + 1e-9))
    return Q[surf], np.array(normals)


def colour(cams, frames_bgr, names, S, N):
    """Each surface point coloured by the camera whose view is most head-on to the surface there."""
    best = np.full(len(S), -1.0)
    rgb = np.zeros((len(S), 3), np.uint8)
    for n in names:
        c = cams[n]
        to_cam = c["C"][None] - S
        to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
        facing = np.einsum("ij,ij->i", N, to_cam)
        px, front = project(c, S)
        im = frames_bgr[n]
        h, w = im.shape[:2]
        ok = front & (px[:, 0] >= 0) & (px[:, 0] < w) & (px[:, 1] >= 0) & (px[:, 1] < h) & (facing > best)
        best[ok] = facing[ok]
        rgb[ok] = im[px[ok, 1], px[ok, 0]][:, ::-1]
    return rgb


if __name__ == "__main__":
    cams = cameras()
    names = [n for n in cams if n != HOLDOUT]
    frames = {n: colour_video(n) for n in cams}
    for i in [int(x) for x in sys.argv[1:]] or [30]:
        masks = {n: outline(n, frames, i) for n in cams}
        S, N = solid(cams, masks, names)
        bgr = {n: frames[n][i] for n in names}
        rgb = colour(cams, bgr, names, S, N)
        # check 1: the solid's outline in the held-out camera vs that camera's own outline
        px, front = project(cams[HOLDOUT], S)
        hm = masks[HOLDOUT]
        sil = np.zeros_like(hm, np.uint8)
        ok = front & (px[:, 0] >= 0) & (px[:, 0] < hm.shape[1]) & (px[:, 1] >= 0) & (px[:, 1] < hm.shape[0])
        sil[px[ok, 1], px[ok, 0]] = 1
        sil = cv2.morphologyEx(sil, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) > 0
        fgm = hm & (cv2.dilate(sil.astype(np.uint8), np.ones((41, 41), np.uint8)) > 0)    # the outline near the bodies
        iou = (sil & fgm).sum() / max((sil | fgm).sum(), 1)
        # check 2: stereo points (kept by the all-camera vote) against the solid's surface
        st = D / "stereo" / f"f{i:03d}_kept.npz"
        line = ""
        if st.exists() and len(S):
            sp = np.load(st)["pts"].astype(np.float64)
            d, _ = cKDTree(S).query(sp)
            line = f"; stereo points to the solid's surface: median {np.median(d) * 100:.1f} cm, 90th pct {np.percentile(d, 90) * 100:.1f} cm"
        print(f"frame {i}: {len(S):,} surface points; held-out camera {HOLDOUT} outline IoU {iou:.2f}{line}")
        np.savez_compressed(OUT / f"f{i:03d}.npz", pts=S.astype(np.float32), rgb=rgb, normals=N.astype(np.float32))
