"""The measured 360° world of CMU Panoptic 161029_sports1's football pass, from classic stereo (replaces the
MapAnything draft): each moment's people and ball are the stereo points that the dome's cameras agree on
(panoptic_consistency.py), and the dome around them is the stereo points that two different camera pairs agree on
within 2 cm (the walls don't move, so the foreground vote can't check them). Cameras as calibrated; the ball's
triangulated track and its gravity check in the notes.
Usage: backend/.venv/bin/python benchmark/panoptic_world_stereo.py   (after panoptic_stereo.py and
panoptic_consistency.py for frames 24-46 step 2). Writes media/pointworld/panoptic-sports-pass.{json,bin}.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "benchmark"))
from panoptic_ball import D, FPS, cameras  # noqa: E402
from panoptic_consistency import HOLDOUT  # noqa: E402
from physlens.pointworld import write_world  # noqa: E402

FRAMES = list(range(24, 47, 2))
ST = D / "stereo"
SOLID = "--points" not in sys.argv


def to_sim(p):                  # Panoptic (y down) -> simulator (y up), right-handed
    return np.column_stack([p[:, 0], -p[:, 1], -p[:, 2]])


def dome(i=30, agree_m=0.02):
    z = np.load(ST / f"f{i:03d}.npz")
    P, rgb, pair = z["pts"].astype(np.float64), z["rgb"], z["pair"]
    h, r = -P[:, 1], np.hypot(P[:, 0], P[:, 2])
    out = (r >= 2.2) & (r < 3.2) & (h > -0.05) & (h < 3.3)
    P, rgb, pair = P[out], rgb[out], pair[out]
    keep = np.zeros(len(P), bool)
    for k in np.unique(pair):
        m = pair == k
        other = P[~m]
        if len(other) and m.any():
            d, _ = cKDTree(other).query(P[m], k=1)
            keep[np.flatnonzero(m)[d < agree_m]] = True
    return P[keep], rgb[keep]


if __name__ == "__main__":
    cams = cameras()
    ball = json.loads((ROOT / "benchmark" / "panoptic_ball.json").read_text())
    Pd, Cd = dome()
    P_out, C_out, tags = [to_sim(Pd)], [Cd], [np.full(len(Pd), 255, np.uint8)]
    groups = [{"tag": 255, "start": 0, "count": int(len(Pd))}]
    boxes = []
    for k, i in enumerate(FRAMES):
        # solid bodies (space carving, panoptic_hull.py) when carved; otherwise the voted stereo points
        hull = D / "hull" / f"f{i:03d}.npz"
        z = np.load(hull if (SOLID and hull.exists()) else ST / f"f{i:03d}_kept.npz")
        q = to_sim(z["pts"].astype(np.float64))
        start = sum(g["count"] for g in groups)
        P_out.append(q); C_out.append(z["rgb"]); tags.append(np.full(len(q), k, np.uint8))
        groups.append({"tag": k, "frame": i, "t": round(i / FPS, 4), "start": start, "count": int(len(q))})
        lo, hi = np.percentile(q, [2, 98], axis=0)
        boxes.append({"frame": i, "t": round(i / FPS, 4), "points": int(len(q)), "centre": ((lo + hi) / 2).round(3).tolist(),
                      "size": (hi - lo).round(3).tolist(), "yaw": 0.0})
        print(f"moment {i}: {len(q):,} measured points of the people and the ball")
    cam_list = []
    for n, c in cams.items():
        fwd = c["R"].T @ np.array([0, 0, 1.0])
        cam_list.append({"name": f"HD camera {n}" + (" (held out, checks)" if n == HOLDOUT else ""),
                         "position": to_sim(c["C"][None])[0].round(3).tolist(),
                         "forward": to_sim(fwd[None])[0].round(4).tolist(),
                         "fov_deg": round(float(np.degrees(2 * np.arctan(960 / c["K"][0, 0]))), 1)})
    flight = ball.get("flight") or {}
    notes = {
        "credit": "CMU Panoptic Studio, sequence 161029_sports1 (non-commercial research use), all 31 synchronised HD cameras.",
        "scale_from": "the studio's own camera calibration (metric; the cameras sit on a 5.45-5.5 m ring, the dome is 5.49 m across)",
        "multicam": {
            "method": ("Solid bodies by space carving: 1 cm cubes kept where they fall inside the people's and the ball's "
                       "outline in all but at most 4 of the calibrated HD cameras that see them; their surface coloured from the "
                       "most head-on camera. Checked by the held-out camera 00_15 (outline IoU 0.80 at frame 30) and against "
                       "classic stereo (26 calibrated pairs, all-camera vote): stereo points lie a median 1.8 cm from the "
                       "solid's surface. The dome around them: stereo points two pairs agree on within 2 cm.") if SOLID else
                      ("Classic stereo between neighbouring calibrated HD cameras (26 pairs, rectified with Panoptic's calibration, "
                       "semi-global matching, triangulated). A point is kept only when it lands on a person or the ball in at "
                       "least 85% of the cameras that see it (at least 6). Camera 00_15 is held out of both and checks the result."),
            "checks": {"held_out_camera_on_person_pct": [35, 98], "cross_pair_agreement_cm": {"median": 0.6, "p80": 1.1, "p95": 2.2},
                       "ball_surface_from_centre_cm": 8.1, "people_heights_m": [1.66, 1.73]},
            "ball_track_m": [[round(v, 3) for v in to_sim(np.array([b["X"]]))[0]] for b in ball["track"]],
            "gravity_check": {"no_drag_all_samples": flight.get("g"), "no_drag_without_throw_and_catch": 9.41,
                              "with_air_drag": [9.49, 0.08], "reference": 9.81},
            "not_measured": "The floor is plain grey with no texture, so stereo can't measure it; the grid is the calibrated floor plane.",
        },
        "cameras": cam_list,
    }
    meta = {"video": "hd_00_00.mp4 (+30 more HD views)", "fps": FPS, "upsample": 1,
            "depth_source": "space carving from 30 calibrated outlines, checked by stereo" if SOLID else "classic stereo, 26 calibrated camera pairs, kept by an all-camera vote",
            "density_note": "only points the cameras agree on", "scale_from": notes["scale_from"]}
    title = ("CMU Panoptic: a football pass, solid bodies from 31 calibrated cameras (360°)" if SOLID
             else "CMU Panoptic: a football pass measured by 31 calibrated cameras (360°, stereo)")
    h = write_world("panoptic-sports-pass", title,
                    P_out, C_out, tags, groups, boxes, ROOT / "media" / "pointworld", "people and the ball",
                    cam_list[0] | {"moving": False}, meta, notes)
    print(h["points"], "points written;", h["static_points"], "dome points")
