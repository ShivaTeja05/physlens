"""One 3D world from two cameras filming the same crash (NHTSA 9070: cam15 side-on, cam17 oblique, both real-time).

  1. Joint geometry: MapAnything reconstructs the empty lab from both cameras in one call, so it solves where each
     camera stands relative to the other and triangulates depth from two viewpoints instead of guessing it from one.
  2. Each camera's own stream (MapAnything per keyframe, plus MoGe-3 for cam15) is brought into that joint world:
     a scale fitted on the static lab (the joint call's depth vs the stream's, car excluded) and the joint pose.
  3. Sync: the cameras' recordings don't start together. The offset (sub-frame) is the one where the two cameras'
     3D positions of the car's front agree best: synchronised by geometry.
  4. Fusion: both cameras' car points per moment in one world; impact, crush and rebound from the fused track,
     checked against NHTSA's measurements.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/crash_multicam.py [--export]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
import crash3d as c15   # noqa: E402
from physlens import events3d as ev   # noqa: E402
from physlens.geometry.client import reconstruct   # noqa: E402
from physlens.moving_camera import floor_frame   # noqa: E402
from physlens.sharp import align, main_depth_cluster   # noqa: E402
from physlens.vehicle3d import geometry, read_at, video_masks   # noqa: E402

D = ROOT / "datasets" / "nhtsa" / "9070"
CACHE = c15.CACHE
IDX = c15.IDX
FPS = 30000 / 1001
CAMS = {"cam15": D / "cam15.mp4", "cam17": D / "cam17.mp4"}


def joint():
    """Both cameras' empty-lab frame in one MapAnything call (cached)."""
    f = CACHE / "joint_cam15_cam17.npz"
    imgs = [read_at(p, [20])[0][0] for p in CAMS.values()]
    return geometry(imgs, f)


def stream(cam):
    frames, _ = read_at(CAMS[cam], IDX)
    views = geometry(frames, CACHE / ("mapanything.npz" if cam == "cam15" else f"{cam}_mapanything.npz"))
    tag = CACHE / ("samvideo_car" if cam == "cam15" else f"{cam}_samvideo_car")
    ms = video_masks(frames, "car", tag)
    return frames, views, ms


def to_joint(view_stream0, view_joint, mask0):
    """Similarity taking a stream's camera frame to the joint world: scale from the static lab (depth ratio on
    pixels away from the car, both valid), then the joint call's camera pose."""
    zs, zj = view_stream0.depth, view_joint.depth
    if zs.shape != zj.shape:
        zj = cv2.resize(zj, zs.shape[::-1], interpolation=cv2.INTER_NEAREST)
    ok = np.isfinite(zs) & np.isfinite(zj) & (zs > 0.3) & (zj > 0.3)
    if mask0 is not None:
        ok &= ~c15.mask_at(mask0, zs.shape, dilate=15)
    r = zj[ok] / zs[ok]
    s = float(np.median(r))
    spread = float(np.subtract(*np.percentile(r, [75, 25])) / 2 / s)
    return s, view_joint.camera_pose, spread


def car_points(view, mask, s, pose, sharp=None):
    """The car's points in the joint world: from MoGe-3 aligned to this view when given, else MapAnything."""
    if sharp is not None:
        roi = c15.mask_at(mask, sharp.depth.shape, dilate=121)
        al = align(view, sharp, roi=roi)
        Pc = (al.points_world - view.camera_pose[:3, 3]) @ view.camera_pose[:3, :3]     # back to camera frame
        ok = al.valid_loose & c15.mask_at(mask, Pc.shape[:2], erode=3) & np.isfinite(Pc).all(-1)
    else:
        Pc = view.points_cam
        ok = c15.mask_at(mask, Pc.shape[:2], erode=2) & np.isfinite(Pc).all(-1) & (view.depth > 0.3)
    P = Pc[ok] * s
    if len(P) < 200:
        return None
    P = P[main_depth_cluster(P[:, 2])]
    return P @ pose[:3, :3].T + pose[:3, 3]


def sync(t, fa, fb, offsets=np.arange(-4, 4.01, 0.05)):
    """Offset (frames) to add to camera b's frame numbers so its front track matches camera a's best (median
    absolute difference over the approach, where both move)."""
    best = None
    for o in offsets:
        tb = t + o / FPS
        ok = np.isfinite(fa) & np.isfinite(fb)
        g = np.interp(t[ok], tb[ok], fb[ok])
        err = float(np.median(np.abs(fa[ok] - g)))
        if best is None or err < best[0]:
            best = (err, float(o))
    return best[1], best[0]


def main():
    J = joint()
    data = {}
    for cam in CAMS:
        frames, views, ms = stream(cam)
        k0 = 0                                   # keyframe 0 is the same empty-lab frame the joint call saw
        s, pose, spread = to_joint(views[k0], J[list(CAMS).index(cam)], ms[k0])
        sd = c15.sharp(views) if cam == "cam15" else None
        clouds = []
        for j in range(len(IDX)):
            clouds.append(car_points(views[j], ms[j], s, pose, sd[j] if sd else None) if ms[j] is not None else None)
        data[cam] = {"views": views, "ms": ms, "scale": s, "scale_spread": spread, "clouds": clouds, "frames": frames}
        print(cam, "scale to joint world", round(s, 4), "± (IQR/2)", round(spread, 3), "car seen in",
              sum(c is not None for c in clouds), "keyframes")
    # Ground frame of the joint world: the lab floor, from the joint call's two views.
    T, finfo = floor_frame(J, [v.camera_pose for v in J])
    for cam in CAMS:
        data[cam]["clouds_g"] = [None if cc is None else cc @ T[:3, :3].T + T[:3, 3] for cc in data[cam]["clouds"]]
    t = np.array(IDX) / FPS
    # Direction of travel: cam15's car before impact.
    a = data["cam15"]["clouds_g"]
    med = np.array([np.median(cc[:, [0, 2]], axis=0) if cc is not None else [np.nan, np.nan] for cc in a])
    i0, i1 = IDX.index(38), IDX.index(43)
    d = med[i1] - med[i0]
    d /= np.linalg.norm(d)
    fronts = {cam: np.array([np.percentile(cc[:, [0, 2]] @ d, 99) if cc is not None else np.nan
                             for cc in data[cam]["clouds_g"]]) for cam in CAMS}
    rears = {cam: np.array([np.percentile(cc[:, [0, 2]] @ d, 1) if cc is not None else np.nan
                            for cc in data[cam]["clouds_g"]]) for cam in CAMS}
    # Sync on the approach (frames 38–44 in cam15), where the front moves 0.5 m a frame.
    appr = (np.array(IDX) >= 36) & (np.array(IDX) <= 46)
    fa = np.where(appr, fronts["cam15"], np.nan)
    off, err = sync(t, fa, fronts["cam17"])
    print("cam17 offset", off, "frames; front disagreement after sync", round(err, 3), "m")
    out = {"scale_to_joint": {c: round(data[c]["scale"], 4) for c in CAMS},
           "scale_spread": {c: round(data[c]["scale_spread"], 3) for c in CAMS},
           "cam17_offset_frames": off, "front_disagreement_m": round(err, 3), "direction": d.round(4).tolist(),
           "t": t.round(4).tolist(), "frames": IDX,
           "front": {c: np.round(fronts[c], 3).tolist() for c in CAMS}, "rear": {c: np.round(rears[c], 3).tolist() for c in CAMS}}
    # Speed from each camera alone (in the joint world) and from both, over the approach.
    def approach_speed(tt, ff):
        ok = np.isfinite(ff)
        k = int(np.nanargmax(ff))
        sel = [i for i in range(len(ff)) if ok[i] and IDX[i] >= 38 and i < k and ff[i] < ff[k] - 0.25]
        if len(sel) < 3:
            return None, None, sel
        A = np.vstack([tt[sel], np.ones(len(sel))]).T
        cf, *_ = np.linalg.lstsq(A, ff[sel], rcond=None)
        r = ff[sel] - A @ cf
        e = float(np.sqrt((r @ r) / max(len(sel) - 2, 1) / np.sum((tt[sel] - tt[sel].mean()) ** 2)))
        return float(cf[0]), e, sel
    tb = t + off / FPS
    for cam, tt in (("cam15", t), ("cam17", tb)):
        v, e, sel = approach_speed(tt, fronts[cam])
        out[f"approach_kmh_{cam}"] = round(v * 3.6, 2) if v else None
        out[f"approach_err_kmh_{cam}"] = round(e * 3.6, 2) if e else None
        print(cam, "approach", out[f"approach_kmh_{cam}"], "±", out[f"approach_err_kmh_{cam}"], "km/h  (NHTSA 56.55)")
    # Both cameras' fronts on one time axis.
    tt = np.concatenate([t, tb])
    ff = np.concatenate([fronts["cam15"], fronts["cam17"]])
    order = np.argsort(tt)
    tt, ff = tt[order], ff[order]
    ok = np.isfinite(ff)
    k = int(np.nanargmax(np.where(tt < 1.6, ff, np.nan)))
    sel = [i for i in range(len(ff)) if ok[i] and tt[i] >= 38 / FPS and tt[i] < tt[k] and ff[i] < ff[k] - 0.25]
    A = np.vstack([tt[sel], np.ones(len(sel))]).T
    cf, *_ = np.linalg.lstsq(A, ff[sel], rcond=None)
    r = ff[sel] - A @ cf
    out["approach_kmh_fused"] = round(float(cf[0]) * 3.6, 2)
    out["approach_err_kmh_fused"] = round(float(np.sqrt((r @ r) / max(len(sel) - 2, 1) / np.sum((tt[sel] - tt[sel].mean()) ** 2))) * 3.6, 2)
    # Car length from each camera (whole car in view, before impact).
    for cam in CAMS:
        L = [fronts[cam][i] - rears[cam][i] for i in range(len(IDX)) if 41 <= IDX[i] <= 43 and np.isfinite(fronts[cam][i])]
        out[f"length_m_{cam}"] = round(float(np.median(L)), 3) if L else None
    print("fused approach", out["approach_kmh_fused"], "±", out["approach_err_kmh_fused"], "| lengths", out["length_m_cam15"], out["length_m_cam17"], "(4.63)")
    (ROOT / "benchmark" / "crash_multicam.json").write_text(json.dumps(out, indent=1, default=float))
    return data, out, J, T


if __name__ == "__main__":
    main()
