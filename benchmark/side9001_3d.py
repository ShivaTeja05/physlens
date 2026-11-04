"""A second event: NHTSA test 9001 (a 1,362 kg barrier cart, crabbed 27°, into a standing 2015 Toyota Camry at
62.19 km/h), real-time rear-view camera (cam10, fixed, 30 fps), in 3D.

  lens    MapAnything and MoGe-3 disagree on this camera's field of view (65° vs 56°), and neither is reliably
          right (on 9070's high-speed camera MoGe-3 was; here the Camry measures closer with MapAnything's). Both
          are run; results are reported for each, as a range.
  scale   the parked Camry: Muse Spark identifies it (priors.py) and its published length scales the world
  test    the Camry's size from its points (static, frames 20-36); the cart's speed before impact from its body's
          poses (NHTSA 62.19 km/h); the two vehicles' speed sliding together after impact (NHTSA accelerometers)
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/side9001_3d.py
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
from physlens.geometry.client import sharp_depth   # noqa: E402
from physlens.moving_camera import _plane_ransac   # noqa: E402
from physlens.objsim import build_body, clean   # noqa: E402
from physlens.sharp import main_depth_cluster   # noqa: E402
from physlens.vehicle3d import read_at, video_masks   # noqa: E402

VIDEO = ROOT / "datasets" / "nhtsa" / "9001" / "cam10.mp4"
CACHE = ROOT / "records" / "real_cache" / "side9001_cam10"
IDX = [20, 28] + list(range(36, 83))
FPS = 30000 / 1001
W = 1280
LENSES = {"MoGe-3": 56.4, "MapAnything": 65.3}
TRUTH = {"cart_kmh": 62.19, "camry_length_m": 4.85, "camry_width_m": 1.816, "camry_height_m": 1.47,
         "after_impact_kmh": "car 23-24 km/h, barrier 22-24 km/h at +60 to +100 ms (accelerometers; CRASH.md)",
         "source": "NHTSA test 9001"}


def masks(frames, prompt):
    return video_masks(frames, prompt, CACHE / f"samvideo_{prompt}")


def world(fov, frames_hi, ms_all):
    """MoGe-3 depth for every keyframe with the given lens; each frame's depth fitted to the first frame's on the
    unmoving lab; the floor (lowest level plane below the camera) → ground frame. Returns per-frame lifted points
    (camera frame), validity, the ground transform, and info."""
    f = CACHE / f"moge_{int(round(fov))}.npz"
    if f.exists():
        z = np.load(f)
        D = [z[f"d{j}"].astype(np.float32) for j in range(len(IDX))]
        M = [z[f"m{j}"] for j in range(len(IDX))]
        K = z["K0"]
    else:
        sd = sharp_depth(frames_hi, fov_x=[fov] * len(IDX))
        D, M, K = [s.depth for s in sd], [s.mask for s in sd], sd[0].intrinsics
        np.savez_compressed(f, K0=K, **{f"d{j}": d.astype(np.float16) for j, d in enumerate(D)}, **{f"m{j}": m for j, m in enumerate(M)})
    h, w = D[0].shape
    static = np.ones((h, w), bool)
    for ms in ms_all:
        for m in ms:
            if m is not None:
                static &= ~(cv2.dilate(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST), np.ones((25, 25), np.uint8)) > 0)
    for j in range(len(IDX)):
        ok = static & M[j] & M[0]
        D[j] = D[j] * float(np.median(D[0][ok] / D[j][ok]))
    ys, xs = np.mgrid[0:h, 0:w]
    lift = lambda d: np.dstack([(xs - K[0, 2]) * d / K[0, 0], (ys - K[1, 2]) * d / K[1, 1], d])
    P0 = lift(D[0])
    S = P0[static & M[0] & (ys > h * 0.45)][::5]
    up = np.array([0.0, -1.0, 0.0])
    rest, planes = S, []
    for _ in range(5):
        if len(rest) < 500:
            break
        n, c, inl = _plane_ransac(rest, 0.02 * float(np.median(D[0])))
        n = n if n @ up > 0 else -n
        planes.append((n, c, int(inl.sum()), float(-c @ n)))
        rest = rest[~inl]
    level = [p for p in planes if p[0] @ up > np.cos(np.radians(35)) and p[3] > 0.2]
    big = max(p[2] for p in level)
    n, c, _, cam_h = max([p for p in level if p[2] >= 0.2 * big], key=lambda p: p[3])
    x = np.array([1.0, 0, 0]) - n[0] * n
    x /= np.linalg.norm(x)
    T = np.eye(4)
    T[:3, :3] = np.stack([x, n, np.cross(x, n)])
    T[:3, 3] = -T[:3, :3] @ c
    return [lift(d) for d in D], M, T, {"fov_x_deg": fov, "camera_height_m": round(cam_h, 3)}


def object_points(Ps, M, T, ms, rgb):
    out = []
    for j in range(len(IDX)):
        m = ms[j]
        if m is None:
            out.append(None)
            continue
        h, w = Ps[j].shape[:2]
        mm = cv2.erode(cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST), np.ones((3, 3), np.uint8)) > 0
        ok = mm & M[j] & np.isfinite(Ps[j]).all(-1)
        if ok.sum() < 300:
            out.append(None)
            continue
        Pc = Ps[j][ok]
        keep = main_depth_cluster(Pc[:, 2])
        col = cv2.resize(rgb[j], (w, h), interpolation=cv2.INTER_AREA)[ok][:, ::-1]
        out.append(((Pc @ T[:3, :3].T + T[:3, 3])[keep], col[keep]))
    return out


def footprint(P):
    """Length, width, height of points standing on the ground (y up): principal axes of the ground footprint."""
    H = P[:, [0, 2]] - P[:, [0, 2]].mean(0)
    u = np.linalg.svd(H, full_matrices=False)[2]
    a, b = H @ u[0], H @ u[1]
    return (float(np.percentile(a, 99) - np.percentile(a, 1)), float(np.percentile(b, 99) - np.percentile(b, 1)),
            float(np.percentile(P[:, 1], 99) - max(np.percentile(P[:, 1], 0.5), 0.0)), u[0])


def run_lens(name, fov, frames_hi, ms_car, ms_cart, prior_len):
    Ps, M, T, info = world(fov, frames_hi, [ms_car, ms_cart])
    cam = T[:3, 3].copy()
    car = object_points(Ps, M, T, ms_car, frames_hi)
    cart = object_points(Ps, M, T, ms_cart, frames_hi)
    t = np.array(IDX) / FPS
    # The parked Camry: frames before the cart arrives (20-36), its points merged (it doesn't move).
    still = [j for j, f in enumerate(IDX) if f <= 36 and car[j] is not None]
    Pc = np.vstack([clean(car[j][0], car[j][1], cam, radius=3.5)[0] for j in still])
    L_ai, W_ai, H_ai, axis = footprint(Pc)
    k = prior_len / L_ai                                      # world → metres, by the identified model's length
    out = {"lens": name, "fov_x_deg": fov, "camry_ai_m": [round(L_ai, 3), round(W_ai, 3), round(H_ai, 3)],
           "scale_from_identified_length": round(k, 4), "camry_height_scaled_m": round(H_ai * k, 3)}
    # The Camry over time: its parked body (merged from the still frames) registered into each frame (turn about
    # the vertical + shift, ICP): whatever part is still visible places the whole car, so the cart passing in front
    # of it doesn't move it.
    from scipy.spatial import cKDTree
    from physlens.objfuse import register
    tmpl = Pc[np.linspace(0, len(Pc) - 1, min(len(Pc), 40000)).astype(int)]
    tree = cKDTree(tmpl)
    c0 = tmpl.mean(0)
    disp = np.full(len(IDX), np.nan)
    for j in range(len(IDX)):
        if car[j] is None:
            continue
        src = clean(car[j][0], car[j][1], cam, radius=3.5)[0]
        if len(src) < 300:
            continue
        src = src[np.linspace(0, len(src) - 1, min(len(src), 8000)).astype(int)]
        Tj, rms, ov = register(src, tree, tmpl, np.zeros(2), cut=0.3, pivot=src.mean(0))
        if ov < 0.3:
            continue
        # The car's centre in this frame: the template centre moved by the inverse of frame → template.
        Ti = np.linalg.inv(Tj)
        disp[j] = float(np.linalg.norm((Ti[:3, :3] @ c0 + Ti[:3, 3] - c0)[[0, 2]])) * k
    out["camry_displacement_m"] = [None if not np.isfinite(x) else round(float(x), 3) for x in disp]
    j_imp = next((j for j in range(len(IDX)) if IDX[j] > 36 and np.isfinite(disp[j]) and disp[j] > 0.08
                  and all(np.isfinite(disp[i]) and disp[i] > 0.08 for i in range(j, min(j + 3, len(IDX))))), None)
    out["camry_moves_from_frame"] = IDX[j_imp] if j_imp is not None else None
    if j_imp is not None:
        after = [j for j in range(j_imp + 1, len(IDX)) if np.isfinite(disp[j]) and t[j] - t[j_imp] < 0.35]
        if len(after) >= 4:
            v = float(np.polyfit(t[after], disp[after], 1)[0])
            out["camry_after_impact_kmh"] = round(v * 3.6, 1)
            out["camry_after_frames"] = [IDX[j] for j in after]
    # The cart before impact: its body from the frames it's in view and not yet touching the car.
    pre = [j for j in range(len(IDX)) if cart[j] is not None and (j_imp is None or IDX[j] < IDX[j_imp])]
    out["cart_frames_before_impact"] = [IDX[j] for j in pre]
    if len(pre) >= 3:
        cl = [clean(cart[j][0], cart[j][1], cam, radius=3.0) for j in pre]
        cen = np.array([np.median(p[0], 0) for p in cl])
        dv = cen[-1, [0, 2]] - cen[0, [0, 2]]
        heads = np.full(len(pre), np.arctan2(dv[1], dv[0]))
        try:
            b = build_body([p[0] for p in cl], [p[1] for p in cl], [IDX[j] for j in pre], [float(t[j]) for j in pre],
                           len(pre) // 2, cen, heads, cut=0.35)
            used = [p for p in b.poses if p["used"]]
            if len(used) >= 3:
                xz = np.array([[p["x"], p["z"]] for p in used])
                tt = np.array([p["t"] for p in used])
                d = xz[-1] - xz[0]
                d /= np.linalg.norm(d)
                out["cart_kmh_body"] = round(float(np.polyfit(tt, xz @ d, 1)[0]) * 3.6 * k, 1)
                out["cart_frames_used"] = [p["frame"] for p in used]
        except Exception as e:                               # noqa: BLE001
            out["cart_body_error"] = str(e)[:120]
        # Also from the cart's median point (no body): a cross-check.
        out["cart_kmh_median"] = round(float(np.polyfit(t[pre], (cen[:, [0, 2]] @ (dv / np.linalg.norm(dv))), 1)[0]) * 3.6 * k, 1)
    out["camera_height_m_scaled"] = round(info["camera_height_m"] * k, 2)
    return out


def main():
    from physlens.priors import identify
    CACHE.mkdir(parents=True, exist_ok=True)
    frames, _ = read_at(VIDEO, IDX, 960)
    hi, _ = read_at(VIDEO, IDX, W)
    ms_car = masks(frames, "car")
    ms_cart = masks(frames, "cart")
    pf = CACHE / "prior.json"
    if not pf.exists():
        f20 = read_at(VIDEO, [20], 1920)[0][0]
        m = cv2.resize(ms_car[0].astype(np.uint8), (f20.shape[1], f20.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
        pf.write_text(json.dumps(identify(f20, m), indent=1))
    prior = json.loads(pf.read_text())
    print("Muse Spark:", prior.get("make"), prior.get("model"), prior.get("years"), prior.get("length_m"), "x", prior.get("width_m"),
          "x", prior.get("height_m"), prior.get("confidence"), "| text:", prior.get("text_read"))
    print("SAM: car in", sum(m is not None for m in ms_car), "| cart in", sum(m is not None for m in ms_cart), "of", len(IDX), "keyframes")
    res = {"prior": prior, "truth": TRUTH, "by_lens": [run_lens(n, f, hi, ms_car, ms_cart, prior["length_m"]) for n, f in LENSES.items()]}
    for r in res["by_lens"]:
        print(json.dumps(r))
    (ROOT / "benchmark" / "side9001_3d.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
