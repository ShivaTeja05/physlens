"""Moving-camera pipeline vs ground truth on the synthetic handheld drop (samples/moving-camera-drop.mp4).

Three analyses of the same clip, same physics code:
  world   the moving-camera pipeline (MapAnything poses + floor plane + apparent size)
  pixels  what a fixed-camera lab would do: image height × a fixed metres-per-pixel scale
  truth   the renderer's own ball path
Writes benchmark/moving_camera.json, a scene (records/moving_camera_scene.json) and a chart (media/moving_camera.png).
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/moving_camera_eval.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
load_dotenv(ROOT / "backend" / ".env")
from physlens.moving_camera import drop_physics, run, to_scene   # noqa: E402

VIDEO = ROOT / "samples" / "moving-camera-drop.mp4"


def align_xz(est: np.ndarray, true: np.ndarray):
    """Rotation about the vertical + horizontal shift that best maps est onto true (no scale: scale is measured)."""
    A, B = est[:, [0, 2]], true[:, [0, 2]]
    ca, cb = A.mean(0), B.mean(0)
    U, _, Vt = np.linalg.svd((A - ca).T @ (B - cb))
    R = (U @ Vt).T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = (U @ Vt).T
    def apply(P):
        Q = P.copy()
        Q[:, [0, 2]] = (P[:, [0, 2]] - ca) @ R.T + cb
        return Q
    return apply


def chart(t, y_true, y_world, y_pix, out: Path):
    W, H, m = 900, 420, 50
    img = np.full((H, W, 3), 255, np.uint8)
    ymax = 1.7
    X = lambda s: int(m + (W - 2 * m) * s / t[-1])
    Y = lambda v: int(H - m - (H - 2 * m) * v / ymax)
    cv2.line(img, (m, Y(0)), (W - m, Y(0)), (180, 180, 180), 1)
    for v in (0.5, 1.0, 1.5):
        cv2.putText(img, f"{v} m", (5, Y(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1)
    for ys, col, name, k in ((y_true, (60, 60, 60), "truth", 0), (y_world, (20, 120, 240), "moving-camera pipeline", 1),
                             (y_pix, (200, 80, 160), "pixels (camera motion ignored)", 2)):
        pts = [(X(a), Y(b)) for a, b in zip(t, ys) if np.isfinite(b)]
        cv2.polylines(img, [np.array(pts, np.int32)], False, col, 2 if k else 3, cv2.LINE_AA)
        cv2.putText(img, name, (W - 320, 30 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    cv2.putText(img, "ball height (m) vs time, handheld camera", (m, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


def main():
    truth = json.loads(VIDEO.with_suffix(".truth.json").read_text())
    fps = truth["fps"]
    # --size: the ball's real size is given (as a user would: "it's a 12 cm ball").
    # --intrinsics: the lens is given to MapAnything (as phone metadata or glasses calibration would).
    K = np.array(truth["intrinsics"]) if "--intrinsics" in sys.argv else None
    size = truth["ball_radius_m"] if "--size" in sys.argv else None
    wt, sightings = run(VIDEO, step=10, cache=ROOT / "records" / "moving_cache", known_radius_m=size, intrinsics=K)
    tb = np.array(truth["ball_world"])
    t, P_true = tb[:, 0], tb[:, 1:]
    cam_true = np.array([np.array(T)[:3, 3] for T in truth["camera_poses_cam2world"]])

    world = drop_physics(wt.t, wt.p[:, 1], wt.radius_m, fps)
    true = drop_physics(t, P_true[:, 1], truth["ball_radius_m"], fps)
    # Pixels: image "up" with one metres-per-pixel scale, from the object's measured size (what a fixed camera does).
    v_px = np.array([s[1] if s else np.nan for s in sightings])
    r_px = np.array([s[2] if s else np.nan for s in sightings])
    scale = wt.radius_m / np.nanmedian(r_px)
    y_pix = (np.nanmax(v_px) - v_px) * scale + wt.radius_m
    pixels = drop_physics(t, y_pix, wt.radius_m, fps)

    apply = align_xz(wt.cam, cam_true)
    cam_err = np.linalg.norm(apply(wt.cam) - cam_true, axis=1)
    ok = np.isfinite(wt.p).all(1)
    ball_err = np.linalg.norm(apply(wt.p[ok]) - P_true[ok], axis=1)
    h_err = np.abs(wt.p[ok, 1] - P_true[ok, 1])
    travel_est = np.linalg.norm(np.diff(wt.cam, axis=0), axis=1).sum()
    travel_true = np.linalg.norm(np.diff(cam_true, axis=0), axis=1).sum()

    res = {"clip": str(VIDEO.relative_to(ROOT)), "frames": len(t), "fps": fps, "pipeline": wt.info,
           "truth": {"g": truth["g"], "restitution": truth["e"], "radius_m": truth["ball_radius_m"],
                     "focal_px": truth["intrinsics"][0][0], "physics_on_true_path": true},
           "world": world, "pixels": pixels,
           "errors": {"ball_rms_m": round(float(np.sqrt(np.mean(ball_err ** 2))), 4),
                      "ball_height_rms_m": round(float(np.sqrt(np.mean(h_err ** 2))), 4),
                      "camera_rms_m": round(float(np.sqrt(np.mean(cam_err ** 2))), 4),
                      "scale_ratio": round(float(travel_est / travel_true), 3),
                      "g_rel": round(abs(world["g"] - truth["g"]) / truth["g"], 4) if world.get("g") else None,
                      "g_rel_pixels": round(abs(pixels["g"] - truth["g"]) / truth["g"], 4) if pixels.get("g") else None}}
    scene = to_scene(wt, "orange ball", world, str(VIDEO.relative_to(ROOT)))
    try:
        from physlens.sim_mujoco import fit
        mj = fit(scene)
        res["mujoco"] = {"params": mj["params"], "rms_m": round(mj["rms_m"], 4), "rms_unfitted_m": round(mj["rms_unfitted_m"], 4)}
        scene["mujoco"] = {k: mj[k] for k in ("params", "rms_m", "track")}
    except Exception as e:  # noqa: BLE001
        res["mujoco"] = {"error": str(e)}
    (ROOT / "records" / "moving_camera_scene.json").write_text(json.dumps(scene))
    mode = "+".join(a[2:] for a in sys.argv[1:] if a.startswith("--")) or "blind"
    res["mode"] = mode
    (ROOT / "benchmark" / f"moving_camera_{mode}.json").write_text(json.dumps(res, indent=2))
    (ROOT / "benchmark" / "moving_camera.json").write_text(json.dumps(res, indent=2))
    chart(t, P_true[:, 1], wt.p[:, 1], y_pix, ROOT / "media" / "moving_camera.png")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
