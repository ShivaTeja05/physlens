# Sharper points, steadier segmentation, more accurate sizes

Goal: turn real video into clear 3D points (every pixel a small point), keep each object's points together through
the whole clip, and read accurate measurements off them. Clips: `datasets/commons/` (CC BY, see SOURCE.md).
Scripts: `real_vehicles.py` (speeds), `sharp_measure.py` (sizes), `export_pointworlds.py --sharp` (the app's
**3D points** worlds marked "sharp (MoGe-3)").

## What changed, and why

| Problem | Fix | Where |
|---|---|---|
| Coarse points: MapAnything measures 518 × 294 depth per frame; upsampling only interpolated | **MoGe-3** (Microsoft, MIT code and weights, Aug 2026) on the GPU worker gives depth + surface normals at the frame's own resolution (1280 × 720 here: 921,600 real samples per frame, 6×). Its depth is fitted to MapAnything's (one robust scale per frame; 2.5–3.1 % median disagreement after) and lifted with MapAnything's camera poses and lens (MoGe is given MapAnything's field of view, so both use the same lens). | `gpu_worker/` (task `"moge"`), `geometry/client.py: sharp_depth`, `sharp.py: align` |
| Edges smear into tails ("flying pixels") | Pixels on a depth jump are dropped unless the surface is smooth there; for objects, only the main depth cluster inside the mask is kept (background caught by the mask sits metres behind, past a gap) | `sharp.py` |
| A per-frame text prompt lost the truck once its roof was torn open (25 of 37 keyframes) | **SAM 3.1 video tracking** over the keyframes (one identity through the clip): **37 of 37** | `vehicle3d.py: video_masks` |
| One frame shows only part of a vehicle | **Per-object fusion**: each keyframe's points registered onto the others (turn about the vertical + shift, ICP with robust cut-off, turns checked against the track, along-road shift pinned to the rear face for boxy vehicles whose flat sides let ICP slide) | `objfuse.py: fuse` |
| "Up" taken from one scene plane (the rally camera's plane is the spectators' bank) leaks width into height | Each object is measured against the **local road plane** fitted around it (RANSAC) | `objfuse.py: local_plane, size_on_plane` |
| Monocular depth bends slightly with distance; a whole-picture scale sits off at a nearby object | MoGe's scale is fitted **around the object** (object + ~60 px), carrying MapAnything's validated scale onto it | `sharp.py: align(roi=...)` |
| Frame-to-frame speeds jump (truck: 3–32 km/h) | One physical fit over all frames: speed as a smooth piecewise-linear curve, ± from the fit | `real_vehicles.py: speed_curve` |

## Results against known values

**Rally car** (2019 Toyota Yaris WRC, published 4.085 × 1.875 × 1.19 m), fused over the keyframes:

| depth | length | width | height | mean error |
|---|---|---|---|---|
| MapAnything 518 px | 4.29 (+5 %) | 1.71 (−9 %) | 1.04 (−13 %) | 9 % |
| MoGe-3, scale fitted on the whole picture | 4.46 (+9 %) | 1.78 (−5 %) | 1.34 (+13 %) | 9 % |
| **MoGe-3, scale fitted around the car** | **4.27 (+4.5 %)** | **1.73 (−8 %)** | **1.13 (−5 %)** | **6 %** |

**11foot8 bridge**: clearance measured from the road right beneath the beam: **3.63 m** with MoGe-3 vs the signed
3.556 m (+2 %); 3.41 m (−4 %) with MapAnything alone. The road dips ~17 cm under the bridge, so the street's one
plane can't be used for this.

**Box truck**: width 2.26–2.37 m vs ~2.44 m for a box truck (−3 to −7 %). The crash, from one fit over all 37
keyframes (rear face, 12.6 cm residual): cruising **21.2 km/h**, braking at **2.6 m/s²** from ~5.6 s, hits the beam
at **11.3 ± 0.4 km/h** (frames 196–200, the roof intact at 196 and being peeled at 200), moves **1.74 m** further
while the roof is peeled (mean 2.8 m/s²) and stops at 7.64 s.

## Not solved
- **Truck roof height** reads 3.25–3.41 m, but the truck hit a 3.556 m beam, so it must be taller: the roof is seen
  almost edge-on from the pole camera. Unverified: the truck's length (8–10 m from the points; the model isn't
  known).
- Rally width is short by ~8 %: the far side is never seen. SAM 3D Objects (Meta; full shape from one view,
  32 GB GPU) is the next step for unseen sides.
- MoGe-3's own metric scale ran 7–11 % larger than MapAnything's on these clips; MapAnything stays the scale
  reference (it matched the bridge and the car within ~1 %).

## Cost
MoGe-3 worker build: two builds (the first failed: FlexGEMM asks the GPU for its name on import, and RunPod's
build machines have none; the import check was removed from the build). MoGe-3 jobs: 48 keyframes, about 7 minutes
of GPU time including one cold start. The endpoint stays at min 0 workers / 60 s idle timeout.
