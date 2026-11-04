"""Object sizes from the sharp point clouds (MoGe-3 at 1280 px, aligned to MapAnything), fused across keyframes.

  truck  box truck vs the 11foot8 bridge: fused size; roof height and beam clearance each measured from the road
         right beneath them (one plane for the whole street sits off where the road dips under the bridge)
  rally  Toyota Yaris WRC: fused size vs the published 4.085 × 1.875 × 1.19 m
Compared with the same measurement on MapAnything's own 518-px clouds. Needs the caches from real_vehicles.py and
export_pointworlds.py --sharp. Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/sharp_measure.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from physlens.geometry.client import SharpDepth   # noqa: E402
from physlens.objfuse import fuse, local_ground, local_plane, size, size_on_plane   # noqa: E402
from physlens.sharp import align, main_depth_cluster   # noqa: E402
from physlens.vehicle3d import points_in, rescale, track   # noqa: E402

CLIPS, CACHE = ROOT / "datasets" / "commons", ROOT / "records" / "real_cache"
RES = json.loads((ROOT / "benchmark" / "real_vehicles.json").read_text())
LOCAL = True     # fit MoGe's scale around the object (vs over the whole picture)
W = 1280


def sharp(sub, n):
    z = np.load(CACHE / sub / f"moge3_{W}.npz")
    return [SharpDepth(z[f"d{j}"].astype(np.float32), z[f"m{j}"], z[f"n{j}"].astype(np.float32) / 127, z[f"K{j}"]) for j in range(n)]


def mask_at(m, shape, erode=0, dilate=0):
    r = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erode:
        r = cv2.erode(r, np.ones((erode, erode), np.uint8))
    if dilate:
        r = cv2.dilate(r, np.ones((dilate, dilate), np.uint8))
    return r > 0


def clouds_for(tr, views, sd, masks, idx, use_sharp, local_scale=True):
    """Per keyframe: object points and the world around it (ground frame, metres)."""
    T, s = tr.T_ground, tr.scale
    obj, stat = [], []
    for f in tr.frames:
        j = idx.index(f)
        if use_sharp:
            roi = mask_at(masks[j], sd[j].depth.shape, dilate=121) if local_scale else None
            al = align(views[j], sd[j], roi=roi)
            P = (al.points_world @ T[:3, :3].T + T[:3, 3]) * s
            ok = al.valid & np.isfinite(P).all(-1)
            o = al.valid_loose & np.isfinite(P).all(-1) & mask_at(masks[j], ok.shape, erode=3)
            # Depth from the camera, to cut off background caught inside the mask.
            cam = views[j].camera_pose
            zc = (al.points_world[o] - cam[:3, 3]) @ cam[:3, 2]
            keep = main_depth_cluster(zc * s)
            st = ok & ~mask_at(masks[j], ok.shape, dilate=15)
            obj.append(P[o][keep]); stat.append(P[st][::4])
        else:
            obj.append(tr.clouds[tr.frames.index(f)])
            v = views[j]
            h, w = v.depth.shape
            st = ~mask_at(masks[j], (h, w), dilate=7)
            Pw = v.points_world[st]
            Pw = Pw[np.isfinite(Pw).all(1)]
            stat.append(((Pw @ T[:3, :3].T + T[:3, 3]) * s)[::2])
    return obj, stat


def truck(use_sharp):
    idx = list(range(100, 248, 4))
    tr, frames, views = track(CLIPS / "truck-11foot8" / "clip.mp4", idx, "box truck", CACHE / "truck2", sam_video=True)
    rescale(tr, RES["truck"]["scale"], "bridge underside")
    masks = [cv2.imread(str(CACHE / "truck2" / "samvideo_box_truck" / f"{j:03d}.png"), 0) > 0 for j in range(len(idx))]
    sd = sharp("truck2", len(idx)) if use_sharp else None
    obj, stat = clouds_for(tr, views, sd, masks, idx, use_sharp)
    whole = [not (m[:3].any() or m[-3:].any() or m[:, :3].any() or m[:, -3:].any()) for m in masks]
    pre = [k for k, f in enumerate(tr.frames) if whole[idx.index(f)] and f <= 196]
    d = tr.p[pre[-1], [0, 2]] - tr.p[pre[0], [0, 2]]
    ref = pre[len(pre) // 2]
    # Along the road, each keyframe's position comes from the truck's rear face (seen in every frame); ICP then only
    # corrects across the road and the heading (it would slide frames along the flat box sides).
    dn = d / np.linalg.norm(d)
    rear = np.array([np.percentile(obj[k][:, [0, 2]] @ dn, 3) for k in pre])
    cen = tr.p[pre].copy()
    cen[:, [0, 2]] += ((rear - cen[:, [0, 2]] @ dn)[:, None]) * dn
    merged, rep = fuse([obj[k] for k in pre], cen, len(pre) // 2, lock_dir=dn)
    g_truck = local_ground(stat[ref], obj[ref])
    sz = size(merged, d, ground_y=g_truck)
    plane = local_plane(stat[ref], obj[ref])
    sz_plane = size_on_plane(merged, d, plane) if plane is not None else None
    # The beam: lower edge of the bridge mask over the road, last keyframe (the truck is under it, but its lower
    # edge where the truck isn't still shows), and the road right beneath the beam.
    bridge = cv2.imread(str(CACHE / "truck2" / "bridge.png"), 0) > 0
    j = len(idx) - 1
    if use_sharp:
        al = align(views[j], sd[j])
        P = (al.points_world @ T_(tr) + tr.T_ground[:3, 3]) * tr.scale
        valid = al.valid
    else:
        v = views[j]
        P = (v.points_world @ T_(tr) + tr.T_ground[:3, 3]) * tr.scale
        valid = np.isfinite(P).all(-1)
    bm = mask_at(bridge, valid.shape)
    tm = mask_at(masks[j], valid.shape, dilate=9)
    edge = []
    for x in range(valid.shape[1]):
        ys = np.nonzero(bm[:, x])[0]
        if len(ys) < 6:
            continue
        y = ys.max() - 2
        if valid[y, x] and not tm[y, x]:
            edge.append(P[y, x])
    edge = np.array(edge)
    edge = edge[edge[:, 1] > 1.5]
    road = P[valid & ~bm & ~tm]
    # Road under the beam: low points within 3 m (horizontally) of the beam edge's middle part.
    c = np.median(edge[:, [0, 2]], axis=0)
    near = road[np.linalg.norm(road[:, [0, 2]] - c, axis=1) < 3.0]
    g_beam = float(np.median(near[near[:, 1] < np.percentile(near[:, 1], 40)][:, 1])) if len(near) > 50 else 0.0
    clearance = float(np.median(edge[:, 1]) - g_beam)
    return {"depth": "MoGe-3 1280 px" if use_sharp else "MapAnything 518 px", "keyframes_fused": sum(r["used"] for r in rep),
            "of": len(rep), "icp_rms_m": round(float(np.median([r["rms"] for r in rep if r["used"] and r["rms"] > 0])), 3),
            "size_m": sz, "road_under_truck_y": round(g_truck, 3), "road_under_beam_y": round(g_beam, 3),
            "beam_clearance_m": round(clearance, 3), "beam_clearance_true_m": 3.556,
            "roof_minus_clearance_m": round(sz["top"] - g_truck - clearance, 3), "size_on_local_road_m": sz_plane}


def T_(tr):
    return tr.T_ground[:3, :3].T


def rally(use_sharp):
    idx = list(range(164, 186, 2))
    tr, frames, views = track(CLIPS / "rally-latvala" / "clip.mp4", idx, "rally car", CACHE / "rally", sam_video=True)
    rescale(tr, RES["rally"]["scale"], "car length")
    masks = [cv2.imread(str(CACHE / "rally" / "samvideo_rally_car" / f"{j:03d}.png"), 0) > 0 for j in range(len(idx))]
    sd = sharp("rally", len(idx)) if use_sharp else None
    obj, stat = clouds_for(tr, views, sd, masks, idx, use_sharp, local_scale=LOCAL)
    H = tr.p[:, [0, 2]]
    dv = np.gradient(H, axis=0)
    heads = -np.arctan2(dv[:, 1], dv[:, 0])
    ref = tr.frames.index(176)
    merged, rep = fuse(obj, tr.p, ref, cut=0.35, headings=heads)
    # The road under the car: the car's wheels stand on it, so its lowest points mark the road better than a ring
    # (here the ring mixes the road with the spectators' bank).
    base = float(np.percentile(merged[:, 1], 0.5))
    sz_scene = size(merged, dv[ref], ground_y=base)
    plane = local_plane(np.vstack([stat[k] for k in range(len(stat))]), obj[ref])
    # The fused cloud lives in the reference keyframe's position, so the road plane comes from around it.
    sz = size_on_plane(merged, dv[ref], plane) if plane is not None else sz_scene
    return {"size_scene_ground_m": sz_scene,"depth": "MoGe-3 1280 px" if use_sharp else "MapAnything 518 px", "keyframes_fused": sum(r["used"] for r in rep),
            "of": len(rep), "icp_rms_m": round(float(np.median([r["rms"] for r in rep if r["used"] and r["rms"] > 0])), 3),
            "size_m": sz, "true_m": {"length": 4.085, "width": 1.875, "height": 1.19}}


if __name__ == "__main__":
    out = {"truck": [truck(False), truck(True)], "rally": [rally(False), rally(True)]}
    LOCAL = False
    out["rally_global_scale"] = [rally(True)]
    (ROOT / "benchmark" / "sharp_measure.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    for k, rows in out.items():
        for r in rows:
            print(k, json.dumps(r, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
