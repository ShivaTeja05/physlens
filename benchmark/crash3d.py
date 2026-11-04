"""NHTSA test 9070 (2015 Lexus NX 200t into a rigid barrier, 56.55 km/h) as a 3D world, with its events found from
the measured points and checked against NHTSA's own measurements.

The real-time side camera (cam15, 30 fps). MapAnything gives depth and the camera; SAM 3.1 video tracking gives the
car in every keyframe and the barrier once; MoGe-3 gives full-resolution depth aligned to MapAnything. In the
ground frame the car's front and rear are followed along its path:
  impact     the front reaches the barrier's face and stops there
  crush      the body stops being rigid: front stopped, rear still coming (events3d.deformation)
  speeds     approach from the rear face before impact; rebound after the crush
Scale: MapAnything's own metric scale, checked against the car's published length (4.63 m) only afterwards.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/crash3d.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens import events3d as ev   # noqa: E402
from physlens.geometry.client import SharpDepth, sharp_depth   # noqa: E402
from physlens.sharp import align, fov_x, main_depth_cluster   # noqa: E402
from physlens.vehicle3d import masks as sam_masks, read_at, track   # noqa: E402

VIDEO = ROOT / "samples" / "crash-frontal.mp4"
CACHE = ROOT / "records" / "real_cache" / "crash9070"
IDX = [20, 28] + list(range(34, 81))       # empty scene, then every frame from the car's arrival to after rebound
SHARP_W = 1280
TRUTH = {"impact_kmh": 56.55, "car_length_m": 4.63, "crush_m": 0.71, "decel_g": 21.7, "rebound_ms": [1.6, 1.7],
         "source": "NHTSA test 9070: speed trap; crush and deceleration from the rear-floorpan accelerometers"}


def sharp(views):
    f = CACHE / f"moge3_{SHARP_W}.npz"
    if f.exists():
        z = np.load(f)
        return [SharpDepth(z[f"d{j}"].astype(np.float32), z[f"m{j}"], z[f"n{j}"].astype(np.float32) / 127, z[f"K{j}"])
                for j in range(len(IDX))]
    frames, _ = read_at(VIDEO, IDX, SHARP_W)
    sd = sharp_depth(frames, fov_x=[fov_x(v) for v in views])
    np.savez_compressed(f, **{k: v for j, s in enumerate(sd) for k, v in {
        f"d{j}": s.depth.astype(np.float16), f"m{j}": s.mask, f"n{j}": np.round(s.normal * 127).astype(np.int8),
        f"K{j}": s.intrinsics}.items()})
    return sd


def fix_camera(views) -> float:
    """The camera is bolted to the lab: one pose for every keyframe (the mean rotation, the median position).
    MapAnything estimates each view's pose on its own, and a large object crossing the view moves it by a few
    centimetres and tenths of a degree, which would show up as the car's motion. Returns the spread removed (m)."""
    R = np.stack([v.camera_pose[:3, :3] for v in views])
    U, _, Vt = np.linalg.svd(R.sum(0))
    Rm = U @ Vt
    t = np.stack([v.camera_pose[:3, 3] for v in views])
    tm = np.median(t, axis=0)
    for v in views:
        P = np.eye(4)
        P[:3, :3], P[:3, 3] = Rm, tm
        v.camera_pose = P
        v.points_world = v.points_cam @ Rm.T + tm
    return float(np.linalg.norm(t - tm, axis=1).max())


def setup():
    """Geometry, SAM masks (car per keyframe, barrier once) and MoGe-3 depth, all cached under CACHE."""
    CACHE.mkdir(parents=True, exist_ok=True)
    tr, frames, views = track(VIDEO, IDX, "car", CACHE, sam_video=True)
    # SAM 3.1 finds the barrier as "wall" (the largest one: it fills the left of the frame); "barrier" finds nothing.
    barrier = sam_masks([frames[0]], "wall", CACHE / "sam_barrier")[0]
    sd = sharp(views)
    tr.info["camera_fixed_spread_removed_m"] = round(fix_camera(views), 3)
    tag = "samvideo_car"
    ms = [cv2.imread(str(CACHE / tag / f"{j:03d}.png"), 0) for j in range(len(IDX))]
    ms = [m > 0 if m is not None else None for m in ms]
    return tr, frames, views, sd, ms, barrier


def mask_at(m, shape, erode=0, dilate=0):
    r = cv2.resize(m.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    if erode:
        r = cv2.erode(r, np.ones((erode, erode), np.uint8))
    if dilate:
        r = cv2.dilate(r, np.ones((dilate, dilate), np.uint8))
    return r > 0


def clouds(tr, views, sd, ms):
    """Per keyframe: the car's points (MoGe-3, scale fitted around the car, main depth cluster in the mask) and the
    keyframe's aligned view, in the ground frame (metres)."""
    T, s = tr.T_ground, tr.scale
    obj, als = [], []
    for j, f in enumerate(IDX):
        m = ms[j]
        roi = mask_at(m, sd[j].depth.shape, dilate=121) if m is not None else None
        al = align(views[j], sd[j], roi=roi)
        als.append(al)
        if m is None or f not in tr.frames:
            obj.append(None)
            continue
        P = (al.points_world @ T[:3, :3].T + T[:3, 3]) * s
        o = al.valid_loose & np.isfinite(P).all(-1) & mask_at(m, P.shape[:2], erode=3)
        cam = views[j].camera_pose
        zc = (al.points_world[o] - cam[:3, 3]) @ cam[:3, 2]
        obj.append(P[o][main_depth_cluster(zc * s)])
    return obj, als


def touches_edge(m, px=3):
    return bool(m[:px].any() or m[-px:].any() or m[:, :px].any() or m[:, -px:].any())


def analyse(tr, frames, views, sd, ms, barrier):
    fps = 30000 / 1001
    obj, als = clouds(tr, views, sd, ms)
    ks = [j for j, o in enumerate(obj) if o is not None and len(o) > 500]
    t = np.array([IDX[j] / fps for j in ks])
    P = [obj[j] for j in ks]
    # Direction of travel: from the car's first to its last position before it slows (its median point).
    med = np.array([np.median(o[:, [0, 2]], axis=0) for o in P])
    d = med[min(6, len(med) - 1)] - med[0]
    d /= np.linalg.norm(d)
    side = np.array([-d[1], d[0]])
    rear, front = ev.ends(P, d, lo=1, hi=99)
    whole = [i for i, j in enumerate(ks) if not touches_edge(ms[j])]
    # The barrier's face: SAM's barrier in the empty first keyframe, its points nearest the oncoming car within
    # the car's height and lateral band.
    Tg, s = tr.T_ground, tr.scale
    Pb = (als[0].points_world @ Tg[:3, :3].T + Tg[:3, 3]) * s
    bm = mask_at(barrier, Pb.shape[:2]) & als[0].valid
    B = Pb[bm]
    lat = np.concatenate([o[:, [0, 2]] @ side for o in P])
    l0, l1 = np.percentile(lat, [5, 95])
    hb = np.concatenate([o[:, 1] for o in P])
    y0, y1 = np.percentile(hb, [5, 60])
    Bl = B[((B[:, [0, 2]] @ side) > l0 - 0.5) & ((B[:, [0, 2]] @ side) < l1 + 0.5) & (B[:, 1] > y0) & (B[:, 1] < y1)]
    face = float(np.percentile(Bl[:, [0, 2]] @ d, 3))
    face_err = float(np.subtract(*np.percentile(Bl[:, [0, 2]] @ d, [10, 1])) / 2)
    # The front's resting place once it has stopped against the barrier, and the approach before it.
    k_stop = int(np.argmax(front))
    f_max = float(front[k_stop])
    steps = np.diff(front)
    step0 = float(np.median(steps[: max(k_stop - 1, 1)]))
    # Approach: frames before the front stops, from the first with the front well inside the picture (a car
    # entering at the image edge sits where the lens distorts most).
    first = next(i for i, j in enumerate(ks) if IDX[j] >= 38)
    pre = [i for i in range(first, k_stop + 1) if front[i] < f_max - 0.5 * step0]
    A = np.vstack([t[pre], np.ones(len(pre))]).T
    coef, *_ = np.linalg.lstsq(A, front[pre], rcond=None)
    v_app = float(coef[0])
    resid = front[pre] - A @ coef
    v_err = float(np.sqrt((resid @ resid) / max(len(pre) - 2, 1) / np.sum((t[pre] - t[pre].mean()) ** 2)))
    pre_r = [i for i in pre if i in whole]
    v_rear = float(np.polyfit(t[pre_r], rear[pre_r], 1)[0]) if len(pre_r) >= 2 else None
    # Impact: where the straight approach meets the front's resting place (between two frames at 30 fps).
    t_imp = float((f_max - coef[1]) / v_app)
    t_imp_err = float(np.hypot(v_err / v_app * (t_imp - t[pre].mean()), np.sqrt((resid @ resid) / max(len(pre) - 2, 1)) / v_app))
    hit = {"t": t_imp, "t_earliest": t_imp - t_imp_err, "t_latest": t_imp + t_imp_err,
           "front_rest_m": round(f_max, 3), "barrier_face_m": round(face, 3), "gap_front_to_barrier_m": round(face - f_max, 3)}
    # Rigid length: the whole car in view, before impact (same percentiles as the crush below).
    pre_whole = [i for i in whole if t[i] < t_imp]
    L = float(np.median((front - rear)[pre_whole])) if pre_whole else float(np.max((front - rear)[whole]))
    body = ev.RigidBody(L, float(np.ptp((front - rear)[pre_whole]) / 2) if len(pre_whole) > 1 else 0.0, L, 0.0,
                        float(np.median([np.percentile(P[i][:, 1], 98) for i in pre_whole])) if pre_whole else 0.0,
                        [IDX[ks[i]] for i in pre_whole])
    # Crush: the body stops being rigid (front held at the barrier, rear still coming).
    crush = ev.deformation(t[whole], rear[whole], front[whole], L, shrink=0.05)
    # Rebound: the rear's speed once it has turned back (after the largest shortening).
    reb = None
    if crush:
        after = [i for i in whole if t[i] > crush["t_max_crush"] + 0.03 and t[i] < crush["t_max_crush"] + 0.3]
        if len(after) >= 4:
            reb = float(-np.polyfit(t[after], rear[after], 1)[0])
    events = [{"t": float(t[pre[0]]), "kind": "cruise", "label": f"Approaching at {v_app * 3.6:.1f} km/h",
               "detail": f"front's positions before impact, straight-line fit (± {v_err * 3.6:.1f} km/h)",
               "truth": f"NHTSA speed trap: {TRUTH['impact_kmh']} km/h"}]
    if t_imp is not None:
        events.append({"t": t_imp, "kind": "impact", "label": "Front reaches the barrier",
                       "detail": f"the approach line meets the front's resting place {f_max:.2f} m along the track "
                                 f"(window {hit['t_earliest']:.3f}–{hit['t_latest']:.3f} s); the barrier face measured {face - f_max:+.2f} m from it",
                       "window": [hit["t_earliest"], hit["t_latest"]], "frame": round(t_imp * fps, 1)})
    if crush:
        dt = max(crush["t_max_crush"] - (t_imp or crush["t"]), 1 / fps)
        events.append({"t": crush["t"], "kind": "crush", "label": "Body stops being rigid: the front crushes",
                       "detail": f"front held at the barrier while the rear keeps coming; largest shortening {crush['max_crush_m']:.2f} m at {crush['t_max_crush']:.3f} s",
                       "truth": f"NHTSA accelerometers: {TRUTH['crush_m']} m dynamic crush"})
        events.append({"t": crush["t_max_crush"], "kind": "stop", "label": f"Most crushed: {crush['max_crush_m']:.2f} m shorter",
                       "detail": f"mean deceleration from impact ≥ {v_app / dt / 9.81:.0f} g (30 fps: a lower bound for the peak)",
                       "truth": f"NHTSA: {TRUTH['decel_g']} g average"})
    if reb is not None:
        events.append({"t": crush["t_max_crush"] + 0.1, "kind": "rebound", "label": f"Rebounds at {reb:.1f} m/s",
                       "truth": f"NHTSA: {TRUTH['rebound_ms'][0]}–{TRUTH['rebound_ms'][1]} m/s"})
    events.sort(key=lambda e: e["t"])
    # The whole car as a rigid box at every keyframe from its rear (dashed outline in the viewer), while rigid.
    rigid = []
    if body:
        for i, j in enumerate(ks):
            a0 = rear[i]
            ca = a0 + body.length / 2
            cl = float(np.median(P[i][:, [0, 2]] @ side))
            cx, cz = ca * d + cl * side
            base = float(np.percentile(P[i][:, 1], 1))
            rigid.append({"frame": IDX[j], "t": round(float(t[i]), 4), "centre": [round(float(cx), 3), round((body.top + base) / 2, 3), round(float(cz), 3)],
                          "size": [round(body.length, 3), round(body.top - base, 3), round(float(l1 - l0), 3)],
                          "yaw": round(float(np.arctan2(d[1], d[0])), 4)})
    a, b = face * d + (l0 - 0.3) * side, face * d + (l1 + 0.3) * side
    yb = float(np.percentile(Bl[:, 1], 50))
    structures = [{"name": "barrier", "label": "Barrier face", "from": [round(float(a[0]), 3), round(yb, 3), round(float(a[1]), 3)],
                   "to": [round(float(b[0]), 3), round(yb, 3), round(float(b[1]), 3)]}]
    out = {"direction": d.round(4).tolist(), "barrier_face_m": round(face, 3), "barrier_face_err_m": round(face_err, 3),
           "barrier_points": int(len(Bl)), "rigid_body": body.as_dict() if body else None,
           "approach_frames": [IDX[ks[i]] for i in pre], "camera_fixed_spread_removed_m": tr.info.get("camera_fixed_spread_removed_m"),
           "approach_kmh": round(v_app * 3.6, 2), "approach_err_kmh": round(v_err * 3.6, 2),
           "approach_from_rear_kmh": round(v_rear * 3.6, 2) if v_rear else None,
           "impact": hit, "crush": crush, "rebound_ms": round(reb, 2) if reb else None,
           "t": t.round(4).tolist(), "frames": [IDX[j] for j in ks], "rear_m": rear.round(3).tolist(),
           "front_m": front.round(3).tolist(), "whole": [IDX[ks[i]] for i in whole], "events": events, "truth": TRUTH,
           "scale_from": tr.info.get("scale_from", "MapAnything (metric, no reference used)")}
    # MapAnything's own metric scale makes the car too long; scaled by the car's published length instead (the
    # reference the 2D crash lab uses), the speed is a test of the motion alone.
    k_len = TRUTH["car_length_m"] / L
    out["approach_kmh_car_length_scale"] = round(v_app * 3.6 * k_len, 2)
    truth_table = [["Impact speed (AI metric scale)", f"{v_app * 3.6:.1f} ± {v_err * 3.6:.1f} km/h", f"{TRUTH['impact_kmh']} km/h (speed trap)"],
                   ["Impact speed (scaled by car length)", f"{v_app * 3.6 * k_len:.1f} km/h", f"{TRUTH['impact_kmh']} km/h"],
                   ["Car length (AI metric scale)", f"{L:.2f} m", f"{TRUTH['car_length_m']} m (published)"]]
    if crush:
        truth_table.append(["Largest crush (scaled by car length)", f"{crush['max_crush_m'] * k_len:.2f} m", f"{TRUTH['crush_m']} m (accelerometers)"])
    if reb is not None:
        truth_table.append(["Rebound (scaled by car length)", f"{reb * k_len:.1f} m/s", f"{TRUTH['rebound_ms'][0]}–{TRUTH['rebound_ms'][1]} m/s"])
    return out, {"events": events, "rigid": rigid, "structures": structures, "rigid_body": body.as_dict() if body else None,
                 "truth_table": truth_table, "truth": TRUTH, "speeds_kmh": [], "credit": "Video: NHTSA crash test 9070, camera 15 (public)."}


if __name__ == "__main__" and "--setup" not in sys.argv:
    from physlens.pointworld import export
    tr, frames, views, sd, ms, barrier = setup()
    out, notes = analyse(tr, frames, views, sd, ms, barrier)
    (ROOT / "benchmark" / "crash3d.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps({k: v for k, v in out.items() if k not in ("t", "frames", "rear_m", "front_m", "events", "whole")}, indent=1, default=float))
    for e in out["events"]:
        print(f"  {e['t']:.3f} s  {e['kind']:8s} {e['label']}  | {e.get('detail') or ''}  | {e.get('truth') or ''}")
    if "--export" in sys.argv:
        als = [align(v, s) for v, s in zip(views, sd)]
        obj_al = []
        for v, s, m in zip(views, sd, ms):
            roi = mask_at(m, s.depth.shape, dilate=121) if m is not None else None
            obj_al.append(align(v, s, roi=roi))
        h = export("crash-9070", "NHTSA crash test 9070: Lexus NX into a rigid barrier (fixed camera) · sharp (MoGe-3)",
                   VIDEO, tr, views, ms, IDX, ROOT / "media" / "pointworld", False, upsample=1, notes=notes,
                   object_label="Lexus NX 200t", sharp=als, sharp_object=obj_al, width=SHARP_W)
        print("exported", h["points"], "points")


if __name__ == "__main__" and "--setup" in sys.argv:
    tr, frames, views, sd, ms, barrier = setup()
    print("car seen in", len(tr.frames), "of", len(IDX), "keyframes:", tr.frames[:3], "…", tr.frames[-3:])
    print("barrier mask px", None if barrier is None else int(barrier.sum()), "scale_from", tr.info.get("scale_from"))
