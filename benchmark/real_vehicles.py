"""Real footage, real vehicles: speed and impact from openly licensed Commons clips (datasets/commons/SOURCE.md).

  truck   fixed pole camera, a box truck hits the 11foot8 bridge (Durham NC, 2019-09-13, clearance 11′8″) and its box
          roof is peeled open. Scale: the bridge's underside (SAM 3.1 "bridge" mask, bottom edge over the road) is
          3.556 m above the road. Check: box width vs a rental box truck (~2.44 m).
  rally   handheld spectator panning with Latvala's Toyota Yaris WRC (Monte-Carlo 2019). Scale: the car's length
          (4.085 m) measured side-on. Check: speed in car-lengths per second needs no scale at all.

No official speed exists for either moment, so the checks are internal (a second, independent size; agreement
between frames). Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/real_vehicles.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from physlens.vehicle3d import read_at, rescale, track   # noqa: E402

CLIPS = ROOT / "datasets" / "commons"
CACHE = ROOT / "records" / "real_cache"
CLEARANCE_M = 3.556          # 11 ft 8 in
BOX_WIDTH_M = 2.44           # typical rental box truck body, ~8 ft (check, not used)
YARIS_WRC_LENGTH_M = 4.085   # 2019 Toyota Yaris WRC
KMH = 3.6


def extents(cloud: np.ndarray, heading: np.ndarray) -> tuple[float, float]:
    h = heading / np.linalg.norm(heading)
    side = np.array([-h[1], h[0]])
    H = cloud[:, [0, 2]]
    a, b = H @ h, H @ side
    return float(np.percentile(a, 99.5) - np.percentile(a, 0.5)), float(np.percentile(b, 99.5) - np.percentile(b, 0.5))


def headings(P: np.ndarray) -> np.ndarray:
    H = P[:, [0, 2]]
    d = np.gradient(H, axis=0)
    return d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)


def speed_series(tr, k: int = 1):
    H = tr.p[:, [0, 2]]
    d = np.linalg.norm(H[k:] - H[:-k], axis=1)
    dt = tr.t[k:] - tr.t[:-k]
    return (tr.t[k:] + tr.t[:-k]) / 2, d / dt


def underside(view, mask: np.ndarray, T: np.ndarray) -> tuple[float, float, float]:
    """Height of a bridge's lower edge: in every image column the mask's bottom pixel (2 px inside, edge pixels blend
    with what's behind) lifted to 3D; columns where that edge is on the ground (a pillar) are left out."""
    h, w = view.depth.shape
    mm = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    H = view.points_world @ T[1, :3] + T[1, 3]
    edge = [H[np.nonzero(mm[:, x])[0].max() - 2, x] for x in range(w) if mm[:, x].sum() >= 6]
    edge = np.array([e for e in edge if e > 1.5])
    return float(np.median(edge)), float(np.percentile(edge, 25)), float(np.percentile(edge, 75))


def truck() -> dict:
    tr, frames, views = track(CLIPS / "truck-11foot8" / "clip.mp4", list(range(100, 248, 4)), "box truck", CACHE / "truck2")
    bridge = cv2.imread(str(CACHE / "truck2" / "bridge.png"), cv2.IMREAD_GRAYSCALE) > 0     # SAM 3.1, prompt "bridge"
    u_med, u_lo, u_hi = underside(views[-1], bridge, tr.T_ground)
    tm, v = speed_series(tr)
    # The stop: the first keyframe after which the truck moves less than 0.1 (scene units) per keyframe for good.
    steps = np.linalg.norm(np.diff(tr.p[:, [0, 2]], axis=0), axis=1)
    moving = steps > 0.12
    stop = int(np.nonzero(moving)[0][-1]) + 1 if moving.any() else len(tr.p) - 1
    roof_scene = float(np.percentile(tr.clouds[stop][:, 1], 98))
    rescale(tr, CLEARANCE_M / u_med, f"bridge underside = {CLEARANCE_M} m")
    hd = headings(tr.p)
    widths = [extents(c, h)[1] for c, h in zip(tr.clouds[:stop], hd[:stop])]
    cam_h = float(tr.T_ground[1, :3] @ views[0].camera_pose[:3, 3] + tr.T_ground[1, 3]) * tr.scale
    # Approach speed: straight-line fit of distance travelled vs time over the approach (before the last 3 keyframes
    # of motion, where it brakes and hits).
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tr.p[:, [0, 2]], axis=0), axis=1))]
    a = slice(0, max(stop - 3, 4))
    A = np.stack([tr.t[a], np.ones(len(tr.t[a]))], 1)
    coef, res, *_ = np.linalg.lstsq(A, s[a], rcond=None)
    sig = np.sqrt(res[0] / max(len(s[a]) - 2, 1)) if len(res) else 0.0
    v_err = sig / np.sqrt(((tr.t[a] - tr.t[a].mean()) ** 2).sum())
    v_last = float(np.linalg.norm(tr.p[stop, [0, 2]] - tr.p[stop - 2, [0, 2]]) / (tr.t[stop] - tr.t[stop - 2]))
    return {"frames": tr.frames, "t": tr.t.round(3).tolist(), "p": tr.p.round(3).tolist(),
            "speed_t": tm.round(3).tolist(), "speed_ms": (v * tr.scale).round(2).tolist(),
            "last_frame_before_impact": tr.frames[stop], "impact_between_frames": [tr.frames[stop], tr.frames[stop] + 4],
            "scale": round(tr.scale, 4), "underside_scene_units": [round(u_med, 3), round(u_lo, 3), round(u_hi, 3)],
            "check_roof_m": round(roof_scene * tr.scale, 2), "check_roof_expected": f"> {CLEARANCE_M} m (it hit the beam)",
            "camera_height_m": round(cam_h, 2),
            "check_box_width_m": round(float(np.median(widths)), 2), "check_box_width_expected_m": BOX_WIDTH_M,
            "approach_speed_ms": round(float(coef[0]), 2), "approach_speed_err_ms": round(float(v_err), 2),
            "approach_speed_kmh": round(float(coef[0]) * KMH, 1),
            "speed_last_8_frames_before_stop_kmh": round(v_last * KMH, 1), "info": tr.info}


def rally() -> dict:
    tr, frames, views = track(CLIPS / "rally-latvala" / "clip.mp4", list(range(164, 186, 2)), "rally car", CACHE / "rally")
    hd = headings(tr.p)
    # Side-on frames: the car's heading is at right angles to the line from the camera to the car.
    cams = np.array([tr.T_ground[:3, :3] @ views[[164 + 2 * j for j in range(11)].index(f)].camera_pose[:3, 3]
                     + tr.T_ground[:3, 3] for f in tr.frames])
    look = tr.p[:, [0, 2]] - cams[:, [0, 2]]
    look /= np.linalg.norm(look, axis=1, keepdims=True)
    side_on = np.abs((hd * look).sum(1)) < 0.35
    lengths = [extents(c, h)[0] for c, h, so in zip(tr.clouds, hd, side_on) if so]
    L_scene = float(np.median(lengths))
    tm, v_scene = speed_series(tr)
    # Speed in the car's own lengths per second: independent of MapAnything's scale.
    lengths_per_s = v_scene / L_scene
    rescale(tr, YARIS_WRC_LENGTH_M / L_scene, f"car length side-on = {YARIS_WRC_LENGTH_M} m")
    v = v_scene * tr.scale
    pre = slice(0, len(v))
    return {"frames": tr.frames, "t": tr.t.round(3).tolist(), "p": tr.p.round(3).tolist(),
            "speed_t": tm.round(3).tolist(), "speed_ms": v.round(2).tolist(),
            "side_on_frames": [f for f, so in zip(tr.frames, side_on) if so],
            "length_scene_units": [round(x, 3) for x in lengths], "scale": round(tr.scale, 4),
            "speed_kmh_median": round(float(np.median(v[pre])) * KMH, 1),
            "speed_kmh_range": [round(float(np.percentile(v, 10)) * KMH, 1), round(float(np.percentile(v, 90)) * KMH, 1)],
            "car_lengths_per_s_median": round(float(np.median(lengths_per_s)), 2),
            "camera_path_m": round(float(np.linalg.norm(np.diff(cams[:, [0, 2]], axis=0), axis=1).sum() * tr.scale), 2),
            "info": tr.info}


def overlay(name: str, res: dict, idx_all: list[int], width: int = 960):
    """Each keyframe with the vehicle's measured path drawn into it (projected through that keyframe's camera)."""
    S = ROOT / "media"
    vid = {"truck": "truck-11foot8", "rally": "rally-latvala"}[name]
    frames, _ = read_at(CLIPS / vid / "clip.mp4", res["frames"], width)
    out = []
    for f, img, sp in zip(res["frames"], frames, [None] + res["speed_ms"]):
        img = img.copy()
        if sp is not None:
            cv2.putText(img, f"{sp * KMH:5.1f} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 6, cv2.LINE_AA)
            cv2.putText(img, f"{sp * KMH:5.1f} km/h", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (60, 230, 255), 2, cv2.LINE_AA)
        out.append(cv2.resize(img, (480, 270)))
    pick = out[:: max(1, len(out) // 8)][:8]
    while len(pick) < 8:
        pick.append(np.zeros_like(out[0]))
    cv2.imwrite(str(S / f"{name}_speed.jpg"), np.vstack([np.hstack(pick[:4]), np.hstack(pick[4:])]))


def main():
    res = {"truck": truck(), "rally": rally()}
    (ROOT / "benchmark" / "real_vehicles.json").write_text(json.dumps(res, indent=1))
    for k in res:
        overlay(k, res[k], [])
        print(k, json.dumps({x: y for x, y in res[k].items() if x not in ("t", "p", "speed_t", "speed_ms", "frames")}, indent=1))
        print("  speeds km/h:", [round(s * KMH) for s in res[k]["speed_ms"]])


if __name__ == "__main__":
    main()
