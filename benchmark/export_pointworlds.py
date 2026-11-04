"""Export the real clips as point worlds (media/pointworld/). Needs the caches from benchmark/real_vehicles.py.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/export_pointworlds.py [upsample] [--sharp]
  --sharp  MoGe-3 depth at 1280 px (one GPU call per clip, cached), aligned to MapAnything; saved as <clip>-sharp."""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import numpy as np   # noqa: E402
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens.pointworld import export   # noqa: E402
from physlens.sharp import align, fov_x   # noqa: E402
import cv2 as _cv2   # noqa: E402
from physlens.vehicle3d import read_at, rescale, track   # noqa: E402

CLIPS, CACHE, OUT = ROOT / "datasets" / "commons", ROOT / "records" / "real_cache", ROOT / "media" / "pointworld"
res = json.loads((ROOT / "benchmark" / "real_vehicles.json").read_text())
args = [a for a in sys.argv[1:] if not a.startswith("--")]
up = int(args[0]) if args else 2
SHARP = "--sharp" in sys.argv
SHARP_W = 1280


def sharp_views(name, video, idx, views):
    """MoGe-3 per keyframe at SHARP_W px (cached), each aligned to its MapAnything view."""
    from physlens.geometry.client import SharpDepth, sharp_depth
    f = CACHE / name / f"moge3_{SHARP_W}.npz"
    if f.exists():
        z = np.load(f)
        sd = [SharpDepth(z[f"d{j}"].astype(np.float32), z[f"m{j}"], z[f"n{j}"].astype(np.float32) / 127, z[f"K{j}"]) for j in range(len(idx))]
    else:
        frames, _ = read_at(video, idx, SHARP_W)
        sd = sharp_depth(frames, fov_x=[fov_x(v) for v in views])
        np.savez_compressed(f, **{k: v for j, s in enumerate(sd) for k, v in {
            f"d{j}": s.depth.astype(np.float16), f"m{j}": s.mask, f"n{j}": np.round(s.normal * 127).astype(np.int8),
            f"K{j}": s.intrinsics}.items()})
    al = [align(v, s) for v, s in zip(views, sd)]
    print(name, "MoGe→MapAnything scale per frame:", np.round([a.scale for a in al], 3).tolist()[:6], "…",
          "median disagreement after alignment:", round(float(np.median([a.agree for a in al])) * 100, 1), "%")
    return al, sd


def local_views(views, sd, ms):
    """Each view aligned with its scale fitted around the object (object + ~60 px around it)."""
    out = []
    for v, s, m in zip(views, sd, ms):
        roi = None
        if m is not None:
            roi = _cv2.dilate(_cv2.resize(m.astype(np.uint8), s.depth.shape[::-1], interpolation=_cv2.INTER_NEAREST),
                              np.ones((121, 121), np.uint8)) > 0
        out.append(align(v, s, roi=roi))
    return out


def masks_of(folder: Path, n: int):
    out = []
    for j in range(n):
        m = cv2.imread(str(folder / f"{j:03d}.png"), cv2.IMREAD_GRAYSCALE)
        out.append(m > 0 if m is not None and (m > 0).any() else None)
    return out


for name, vid, idx, prompt, sub, moving, title, label in (
        ("truck", "truck-11foot8", list(range(100, 248, 4)), "box truck", "truck2", False,
         "Box truck hits the 11foot8 bridge (fixed camera)", "box truck"),
        ("rally", "rally-latvala", list(range(164, 186, 2)), "rally car", "rally", True,
         "Rally car past a handheld camera (moving camera)", "Toyota Yaris WRC")):
    tr, frames, views = track(CLIPS / vid / "clip.mp4", idx, prompt, CACHE / sub, sam_video=True)
    r = res[name]
    rescale(tr, r["scale"], r["info"]["scale_from"])
    ms = masks_of(CACHE / sub / f"samvideo_{prompt.replace(' ', '_')}", len(idx))
    notes = {"speeds_kmh": [round(v * 3.6, 1) for v in r["speed_ms"]], "speed_t": r["speed_t"]}
    if name == "truck":
        notes.update({"cruise_kmh": r["cruise_kmh"], "braking_decel_ms2": r["braking_decel_ms2"],
                      "speed_at_contact_kmh": r["speed_at_contact_kmh"], "speed_at_contact_err_kmh": r["speed_at_contact_err_kmh"],
                      "contact_frames": r["contact_frames"], "distance_after_contact_m": r["distance_after_contact_m"],
                      "speed_curve": r["speed_curve"], "reference": "bridge underside 3.556 m (11 ft 8 in)",
                      "check": f"box width {r['check_box_width_m']} m vs ~2.44 m for a rental box truck"})
    else:
        notes.update({"speed_kmh_median": r["speed_kmh_median"], "speed_kmh_range": r["speed_kmh_range"],
                      "reference": "car length 4.085 m (2019 Toyota Yaris WRC)",
                      "check": f"MapAnything's own scale was {r['scale']:.3f}× off the known length"})
    sh, sh_obj = None, None
    if SHARP:
        sh, sd = sharp_views(sub, CLIPS / vid / "clip.mp4", idx, views)
        sh_obj = local_views(views, sd, ms)
        meas = json.loads((ROOT / "benchmark" / "sharp_measure.json").read_text())
        row = meas[name][1]
        notes["fused_size_m"] = row["size_m"]
        notes["fused_from"] = f"{row['keyframes_fused']} moments merged (ICP, {row['icp_rms_m'] * 100:.1f} cm)"
        if name == "rally":
            notes["true_size_m"] = row["true_m"]
        else:
            notes["beam_clearance_m"] = row["beam_clearance_m"]
    out_name = f"{name}-sharp" if SHARP else name
    h = export(out_name, title + (" · sharp (MoGe-3)" if SHARP else ""), CLIPS / vid / "clip.mp4", tr, views, ms, idx, OUT, moving,
               upsample=up, notes=notes, object_label=label, sharp=sh, sharp_object=sh_obj, width=SHARP_W if SHARP else 960)
    size = (OUT / f"{out_name}.bin").stat().st_size / 1e6
    print(out_name, f"{h['points']:,} points ({h['static_points']:,} world, {h['object_points']:,} {label}), {size:.1f} MB,",
          "object size median (L×H×W m):", h["object_size_median_m"])
