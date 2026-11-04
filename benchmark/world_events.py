"""Events found in the 3D worlds, from the measured points (physlens/events3d.py), written into the point worlds.

  truck  11foot8: the truck's full rigid body placed at every keyframe from its rear face (the camera behind it
         never sees the box front that hits); when that front reaches the beam measured in the same world; the
         speed curve's braking and stop; the height check against the beam.
Needs the caches from real_vehicles.py and export_pointworlds.py --sharp. Writes benchmark/world_events.json and
the "events" / "rigid" / "structures" notes of media/pointworld/truck-sharp.json and truck.json.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/world_events.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
import sharp_measure as sm   # noqa: E402
from physlens import events3d as ev   # noqa: E402
from physlens.objfuse import local_ground   # noqa: E402
from physlens.sharp import align   # noqa: E402
from physlens.vehicle3d import rescale, track   # noqa: E402

PW = ROOT / "media" / "pointworld"
FPS_TRUCK = 29.97
KMH = 3.6
SIGNED_CLEARANCE_M = 3.556   # 11 ft 8 in, the sign on the beam (2019, before the bridge was raised)


def truck_clouds():
    """Per-keyframe sharp truck clouds and world points (ground frame, metres), cached."""
    idx = list(range(100, 248, 4))
    cache = sm.CACHE / "truck2" / "sharp_clouds.npz"
    tr, frames, views = track(sm.CLIPS / "truck-11foot8" / "clip.mp4", idx, "box truck", sm.CACHE / "truck2", sam_video=True)
    rescale(tr, sm.RES["truck"]["scale"], "bridge underside")
    masks = [cv2.imread(str(sm.CACHE / "truck2" / "samvideo_box_truck" / f"{j:03d}.png"), 0) > 0 for j in range(len(idx))]
    sd = sm.sharp("truck2", len(idx))
    if cache.exists():
        z = np.load(cache)
        obj = [z[f"o{k}"] for k in range(len(tr.frames))]
        stat = [z[f"s{k}"] for k in range(len(tr.frames))]
    else:
        obj, stat = sm.clouds_for(tr, views, sd, masks, idx, True)
        np.savez_compressed(cache, **{f"o{k}": o.astype(np.float32) for k, o in enumerate(obj)},
                            **{f"s{k}": s[::4].astype(np.float32) for k, s in enumerate(stat)})
    return tr, views, sd, masks, idx, obj, stat


def beam_edge(tr, views, sd, masks, j):
    """The beam's lower edge in 3D from keyframe j: the lower edge of SAM's bridge mask where the truck isn't,
    MoGe-3 depth aligned to MapAnything. Also returns that keyframe's road points (ground frame)."""
    bridge = cv2.imread(str(sm.CACHE / "truck2" / "bridge.png"), 0) > 0
    al = align(views[j], sd[j])
    P = (al.points_world @ tr.T_ground[:3, :3].T + tr.T_ground[:3, 3]) * tr.scale
    bm = sm.mask_at(bridge, al.valid.shape)
    tm = sm.mask_at(masks[j], al.valid.shape, dilate=9)
    edge = []
    for x in range(bm.shape[1]):
        ys = np.nonzero(bm[:, x])[0]
        if len(ys) < 6:
            continue
        for y in range(ys.max() - 6, ys.max() - 1):
            if al.valid[y, x] and not tm[y, x]:
                edge.append(P[y, x])
    edge = np.array(edge)
    road = P[al.valid & ~bm & ~tm]
    return edge[edge[:, 1] > 1.5], road[::3]


def truck() -> dict:
    tr, views, sd, masks, idx, obj, stat = truck_clouds()
    F = tr.frames
    t = np.array(F) / FPS_TRUCK
    # Direction of travel from the track while the truck is fully in view and before the bridge.
    pre = [k for k, f in enumerate(F) if 108 <= f <= 176]
    d = tr.p[pre[-1], [0, 2]] - tr.p[pre[0], [0, 2]]
    d /= np.linalg.norm(d)
    side = np.array([-d[1], d[0]])
    rear, front = ev.ends(obj, d)
    # Frames cut by the image edge don't show the whole body: left out of the rigid-body length.
    whole = [k for k, f in enumerate(F) if not (masks[idx.index(f)][:3].any() or masks[idx.index(f)][-3:].any()
                                                 or masks[idx.index(f)][:, :3].any() or masks[idx.index(f)][:, -3:].any())]
    body = ev.rigid_body([obj[k] for k in whole], [F[k] for k in whole], d)
    # Recursive completion: place the body in every rigid, straight-moving frame (after the turn, up to the hit),
    # rebuild it from every frame before the bridge, repeat until nothing moves. The rear positions it settles on
    # replace the per-frame rear face (frames that see little of the rear are placed by what they do see).
    use = [k for k, f in enumerate(F) if 116 <= f <= 196]
    seed = [use.index(k) for k in whole if F[k] in body.frames_used and k in use]
    ref = ev.refine_rigid([obj[k] for k in use], d, rear[use], seed, build=[i for i, k in enumerate(use) if F[k] <= 180])
    rear_refined = rear.copy()
    rear_refined[use] = ref.rear
    rear_frame = rear.copy()
    rear = rear_refined
    lat = np.concatenate([obj[k][:, [0, 2]] @ side for k in pre])
    l0, l1 = np.percentile(lat, [2, 98])
    # The beam over the truck's lane. Its position and the road under it from the first keyframe (the truck is far
    # away); its height also from the last (truck under it, the beam beside it still in view): the two give the
    # height's ± (27 m from the camera, depth wanders a little). The last keyframe's road there is the truck.
    beams = []
    for j in (0, len(idx) - 1):
        edge, road = beam_edge(tr, views, sd, masks, j)
        lat_e = edge[:, [0, 2]] @ side
        beams.append(edge[(lat_e > l0) & (lat_e < l1)])
        if j == 0:
            road0 = road
    over = beams[0]
    plane = float(np.median(over[:, [0, 2]] @ d))
    plane_err = float(np.subtract(*np.percentile(over[:, [0, 2]] @ d, [75, 25])) / 2)
    a_r, l_r = road0[:, [0, 2]] @ d, road0[:, [0, 2]] @ side
    near = road0[(np.abs(a_r - plane) < 2.5) & (l_r > l0) & (l_r < l1)]
    g_beam = float(np.median(near[near[:, 1] < np.percentile(near[:, 1], 40)][:, 1]))
    beam_ys = [float(np.median(b[:, 1])) for b in beams]
    beam_y = beam_ys[0]
    clear_err = float(np.ptp(beam_ys) / 2)
    hit = ev.contact(t, rear, body.section_front, body.section_front_err, plane, plane_err)
    # Speed curve (one fit over every keyframe, real_vehicles.py) and the events in it.
    sc = sm.RES["truck"]["speed_curve"]
    st, sv = np.array(sc["t"]), np.array(sc["kmh"]) / KMH
    motion = ev.motion_events(st, sv)
    t_c = hit["t"]
    v_c = float(np.interp(t_c, st, sv)) if t_c is not None else None
    stop = next((m["t"] for m in motion if m["kind"] == "stop"), None)
    r_mono = np.maximum.accumulate(rear)
    after = float(np.interp(stop, t, r_mono) - np.interp(t_c, t, r_mono)) if (stop and t_c) else None
    k_ref = F.index(176)
    g_truck = local_ground(stat[k_ref], obj[k_ref])
    # Was it a hit? The front reached the beam's plane, and then the truck stopped within a few metres: a truck
    # that clears the beam drives on under the bridge.
    stopped_under = after is not None and after < 3.0
    # Height: the top of the truck's tallest section vs the road under it; ± from the top's spread across frames
    # and the clearance's.
    heights = ev.height_check(body.top, g_truck, beam_y, g_beam, hit=bool(t_c and stopped_under),
                              err=float(np.hypot(clear_err, 0.05)), signed=SIGNED_CLEARANCE_M)
    heights["beam_y_per_keyframe"] = [round(y, 3) for y in beam_ys]
    heights["road_under_beam_y"] = round(g_beam, 3)
    events = []
    t_brake = next((m["t"] for m in motion if m["kind"] == "braking"), st[-1])
    cruise = float(np.mean(sv[st < t_brake])) * KMH
    for m in motion:
        if m["kind"] == "peak_speed":
            events.append({"t": float(st[0]), "kind": "cruise", "label": f"Cruising {cruise:.1f} km/h",
                           "detail": f"average before braking (peak {m['speed_ms'] * KMH:.1f} km/h)"})
        elif m["kind"] == "braking":
            events.append({"t": m["t"], "kind": "braking", "label": f"Braking starts at {m['speed_ms'] * KMH:.1f} km/h",
                           "detail": f"deceleration about {m['decel_ms2']:.1f} m/s²"})
    if t_c is not None:
        events.append({"t": t_c, "kind": "contact", "label": f"Box front reaches the beam at {v_c * KMH:.1f} km/h",
                       "detail": f"rigid body: rear face + {body.section_front:.2f} m reaches the beam "
                                 f"{plane:.2f} m along the road (window {hit['t_earliest']:.2f}–{hit['t_latest']:.2f} s)",
                       "window": [hit["t_earliest"], hit["t_latest"]], "frame": round(t_c * FPS_TRUCK)})
    if stop is not None:
        events.append({"t": stop, "kind": "stop", "label": "Stopped under the beam" if stopped_under else "Stopped",
                       "detail": f"{after:.2f} m after reaching the beam" if after is not None else None})
    events.sort(key=lambda e: e["t"])
    # The whole rigid body at every keyframe, placed from its rear face: the hidden front included.
    rigid = []
    for k, f in enumerate(F):
        a0 = max(r_mono[k], rear[k])
        c_al = a0 + body.length / 2
        c_lat = float(np.median(obj[k][:, [0, 2]] @ side))
        cx, cz = c_al * d + c_lat * side
        rigid.append({"frame": f, "t": round(float(t[k]), 4), "centre": [round(float(cx), 3), round((body.top + g_truck) / 2, 3), round(float(cz), 3)],
                      "size": [round(body.length, 3), round(body.top - g_truck, 3), round(float(l1 - l0), 3)],
                      "yaw": round(float(np.arctan2(d[1], d[0])), 4),
                      "front_section": round(body.section_front, 3)})
    a, b = plane * d + l0 * side, plane * d + l1 * side
    structures = [{"name": "beam", "label": f"Beam {beam_y - g_beam:.2f} m above the road",
                   "from": [round(float(a[0]), 3), round(beam_y, 3), round(float(a[1]), 3)],
                   "to": [round(float(b[0]), 3), round(beam_y, 3), round(float(b[1]), 3)]}]
    out = {"rigid_body": body.as_dict(), "beam": {"along_m": round(plane, 3), "along_err_m": round(plane_err, 3),
                                                  "y": round(beam_y, 3), "clearance_m": round(beam_y - g_beam, 3),
                                                  "clearance_err_m": round(clear_err, 3), "points": int(len(over))},
           "contact": hit, "speed_at_contact_kmh": round(v_c * KMH, 1) if v_c else None,
           "hand_label_frames": [196, 200], "detected_frame": round(t_c * FPS_TRUCK, 1) if t_c else None,
           "stopped_after_contact_m": round(after, 2) if after is not None else None,
           "height": heights, "events": events, "rear_m": rear.round(3).tolist(), "rear_face_m": rear_frame.round(3).tolist(),
           "front_visible_m": front.round(3).tolist(),
           "refinement": {"passes": ref.iterations, "points_in_body": int(len(ref.template)), "history": ref.history,
                          "largest_correction_m": round(float(np.max(np.abs(ref.rear - rear_frame[use]))), 3),
                          "frames": [F[k] for k in use]},
           "t": t.round(3).tolist()}
    for name in ("truck-sharp", "truck"):
        f = PW / f"{name}.json"
        if f.exists():
            h = json.loads(f.read_text())
            h["notes"]["events"] = events
            h["notes"]["rigid"] = rigid
            h["notes"]["rigid_body"] = body.as_dict()
            h["notes"]["structures"] = structures
            h["notes"]["height_check"] = heights
            h["notes"]["refinement"] = out["refinement"] | {"history": ref.history[-1:]}
            h["notes"].pop("contact_frames", None)
            h["notes"]["speed_at_contact_kmh"] = out["speed_at_contact_kmh"]
            h["notes"]["cruise_kmh"] = round(cruise, 1)
            h["notes"]["distance_after_contact_m"] = out["stopped_after_contact_m"]
            h["notes"]["contact_detected"] = {"t": t_c, "window": [hit["t_earliest"], hit["t_latest"]],
                                              "frame": out["detected_frame"], "hand_label_frames": [196, 200]}
            f.write_text(json.dumps(h))
    return out


if __name__ == "__main__":
    res = {"truck": truck()}
    (ROOT / "benchmark" / "world_events.json").write_text(json.dumps(res, indent=1, default=float))
    r = res["truck"]
    print("rigid body", r["rigid_body"])
    print("beam", r["beam"])
    print("contact", r["contact"], "detected frame", r["detected_frame"], "hand label", r["hand_label_frames"],
          "speed", r["speed_at_contact_kmh"], "km/h; stopped", r["stopped_after_contact_m"], "m later")
    print("height", r["height"])
    for e in r["events"]:
        print(f"  {e['t']:.2f} s  {e['kind']:8s} {e['label']}  {e.get('detail') or ''}")
