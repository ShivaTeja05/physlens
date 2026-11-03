"""Export the real clips as point worlds (media/pointworld/). Needs the caches from benchmark/real_vehicles.py.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/export_pointworlds.py [upsample]"""
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from physlens.pointworld import export   # noqa: E402
from physlens.vehicle3d import read_at, rescale, track   # noqa: E402

CLIPS, CACHE, OUT = ROOT / "datasets" / "commons", ROOT / "records" / "real_cache", ROOT / "media" / "pointworld"
res = json.loads((ROOT / "benchmark" / "real_vehicles.json").read_text())
up = int(sys.argv[1]) if len(sys.argv) > 1 else 2


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
    tr, frames, views = track(CLIPS / vid / "clip.mp4", idx, prompt, CACHE / sub)
    r = res[name]
    rescale(tr, r["scale"], r["info"]["scale_from"])
    ms = masks_of(CACHE / sub / f"sam_{prompt.replace(' ', '_')}", len(idx))
    notes = {"speeds_kmh": [round(v * 3.6, 1) for v in r["speed_ms"]], "speed_t": r["speed_t"]}
    if name == "truck":
        notes.update({"approach_speed_kmh": r["approach_speed_kmh"], "approach_speed_err_kmh": round(r["approach_speed_err_ms"] * 3.6, 1),
                      "impact_between_frames": r["impact_between_frames"], "reference": "bridge underside 3.556 m (11 ft 8 in)",
                      "check": f"box width {r['check_box_width_m']} m vs ~2.44 m for a rental box truck"})
    else:
        notes.update({"speed_kmh_median": r["speed_kmh_median"], "speed_kmh_range": r["speed_kmh_range"],
                      "reference": "car length 4.085 m (2019 Toyota Yaris WRC)",
                      "check": f"MapAnything's own scale was {r['scale']:.3f}× off the known length"})
    h = export(name, title, CLIPS / vid / "clip.mp4", tr, views, ms, idx, OUT, moving, upsample=up, notes=notes, object_label=label)
    size = (OUT / f"{name}.bin").stat().st_size / 1e6
    print(name, f"{h['points']:,} points ({h['static_points']:,} world, {h['object_points']:,} {label}), {size:.1f} MB,",
          "object size median (L×H×W m):", h["object_size_median_m"])
