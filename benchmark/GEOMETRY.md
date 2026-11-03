# M1: measuring static things in metres (MapAnything on RunPod)

**Setup:** one frame of the real classroom clip (720×576, old phone video, unknown camera, black bars baked in).
SAM 3.1 outlines objects; MapAnything (Meta, Apache-2.0 weights) on the RunPod endpoint `physlens-geometry-gh`
gives depth per pixel and camera intrinsics. Ground truth is the teacher's 1.000 m tape on the wall.
Script: `geometry_m1.py`. Each run costs about $0.06 on RunPod plus about $0.005 for SAM.

## Results (2026-11-03)
| Run | Tape (true 1.000 m) | Wall distance | Person (arm raised) | Door |
|---|---|---|---|---|
| Full frame, outline-edge points | 0.879 m (−12%) | 3.17 m | 3.06 m ✗ (edge pixels picked up the wall behind) | 0.62 m ✗ |
| Black bars cropped, robust interior depth | 0.825 m (−17.5%) | 2.68 m | 1.59 m | 0.82 m ✗ (door cut off by the frame) |

## What it shows
- **SAM's outlines are right** (person including the raised arm, door). Checked visually.
- **Proportions are consistent; absolute scale from one frame is not.** Single-frame metric scale was off by 12–18% on
  this low-resolution, letterboxed clip. Scaled by the tape, the person comes out at 1.59 / 0.825 ≈ 1.9 m with the arm
  up, which is plausible.
- **Never measure an object the frame cuts off.** The door's top and bottom are outside the picture, so its "height"
  is just the visible strip. PhysLens must detect outlines that touch the frame edge and say "not measurable".
- **Outline edges are unreliable for depth** (they pick up the background), so measure from the eroded interior.

## Next
1. **Anchor the scale:** use a known-size object when one is visible (A4 sheet, credit card, door when fully in view,
   a ruler) or, on a phone or glasses, camera motion from the IMU, or two viewpoints.
2. **Refuse truncated objects** (outline touching the frame edge).
3. **Test on a modern phone photo with tape-measured truths** (higher resolution, known camera from EXIF).

---

# M1 on Objectron: real objects with known sizes (2026-11-03)

**Data:** 10 test-split clips from Google's Objectron (2 each: cup, bottle, laptop, chair, book; handheld phone video
1440×1920), each object with a metric 3D box and per-frame AR camera data. Licence C-UDA-1.0. Script:
`objectron_m1.py [mono|focal|motion]`; results in `objectron_m1_<mode>.json`; 3D results cached in `cache/`.

**Method:** middle frame. SAM 3.1 outlines the object by name (best match to the annotated box). MapAnything turns the
outline into 3D points in metres. Depth range comes from the outline's interior; every outline pixel within that
depth range is kept, and extents are measured along the object's true box axes. Only sides facing the camera are
scored (one view can't see hidden depth).

| Mode | What MapAnything gets | Median size error | Sides within 10% | Within 20% | Median scale |
|---|---|---|---|---|---|
| mono | one photo | 50.6% | 1/22 | 4/22 | ×1.76 |
| focal | one photo + phone focal length | 34.9% | 1/21 | 3/21 | ×1.34 |
| frames | 3 frames 1 s apart, **no device data** (plain webcam) | 44.6% | 3/21 | 4/21 | ×2.40 |
| **motion** | **3 frames 1 s apart + phone AR poses (metres)** | **13.8%** | **7/21** | **14/21** | **×0.96** |

**Takeaways**
- **Camera motion fixes the scale.** From one photo, close-ups (books and cups at 0.24–0.30 m) were placed 4–8× too
  far away, so sizes came out 4–8× too big. With three frames and the phone's AR poses (5–7 cm of movement), distances
  are nearly exact (0.24/0.24, 0.30/0.30, 0.40/0.40 m) and the median size error drops to 13.8%.
- **Focal length alone barely helps, and neither do extra frames without the phone's poses** ("frames": close-ups still
  placed 4–6× too far away). The metric scale comes from the device's motion tracking, not from more pictures.
- **So with a plain webcam, PhysLens needs a known-size reference** (the ball's diameter, an A4 sheet, a ruler or
  ChArUco board), as the drop experiment already uses. With a phone or glasses, use their AR/IMU poses.
- The remaining errors are mostly outline quality: bottles had low SAM overlap (0.43–0.46); SAM often caught part of
  the bottle.
- For PhysLens this confirms the plan: on a phone or Meta glasses, use the device's motion tracking (ARKit/ARCore, or
  the glasses' IMU) as poses. A single photo on its own isn't trustworthy for absolute size.

Cost of all Objectron runs: about $0.10 on SAM and well under $1 on RunPod.

## Trying to cut the 13.8% (2026-11-03)
Same 10 clips, all with phone AR poses. `objectron_m1.py motion_robust | fused | fused_median` (fused runs 7 views over
±3 s of the phone's orbit; about $0.15 of SAM for the extra views and a few cents of RunPod).

| Variant | Median size error | Within 10% | Within 20% | Median scale |
|---|---|---|---|---|
| motion: 3 views, 1 s apart (baseline) | **13.8%** | 7/21 | 14/21 | ×0.96 |
| + outlier removal (nearest-neighbour + biggest blob) | 15.9% | 6/21 | 11/21 | ×0.80 |
| fused: 7 views, every view's points merged | 19.9% | 2/21 | 11/21 | ×1.15 |
| fused_median: 7 views, each measured alone, median per side over views that see it face-on | 15.9% | **9/21** | **15/21** | **×1.04** |

**What it shows**
- **Outlier removal hurts.** Most sides were already measured short (outlines miss part of the object), so trimming
  points shrinks them further. It did fix the one 2.4× outlier.
- **Merging views smears.** Each view's depth is off by a few centimetres, which is nothing for a chair and a disaster
  for a 7 cm cup (cups came out 2–3× too big).
- **Measuring per view and taking the median works better** than merging (cups 0.074/0.070 and 0.099/0.099 m; book
  0.190/0.180 m), with more sides inside 10% and the scale bias gone (×1.04), but the median error doesn't beat the
  baseline.
- **What's left is shape, not scale.** Bottles (×0.6) and chairs (×0.87) dominate the error: SAM's outline covers
  under half of the annotated box (thin, glossy or open shapes), and merging SAM's mask pieces barely helps.

**So the next real gains are:** a known-size reference in the shot (the drop experiment already gets ~1% this way), a
tap on the object to seed SAM with a box instead of a word, or phone LiDAR depth. More views alone don't fix it.

## Telling SAM which object: taps, boxes, and the phone's own motion (2026-11-03)
Same 10 clips, 3-view AR-pose geometry (cached, no GPU). A simulated person taps the middle of the object (the
annotated box centre projected into the picture) or drags a rough box around it. `objectron_m1.py tap | tapname | box
| tapname_tri`. About $0.08 of SAM in all.

| How SAM is told which object | Median size error | Within 10% | Within 20% | Median scale |
|---|---|---|---|---|
| Name, best match to the **annotated box** (baseline; quietly uses the answer to choose) | 13.8% | 7/21 | 14/21 | ×0.96 |
| One tap only | 15.2% | 4/21 | 12/21 | ×0.90 |
| **Name finds candidates, the tap picks one** (no annotation used) | **13.8%** | 7/21 | 14/21 | ×0.96 |
| Box only | 29.6% | 1/21 | 7/21 | ×1.25 |
| Box + tap | 14.7% | 6/21 | 13/21 | ×1.01 |
| Name + tap, distance triangulated from the phone's motion (SIFT matches, AR poses) | 17.1% | 6/21 | 12/21 | ×0.93 |

**What it shows**
- **Name + tap is the honest version of the baseline**, with the same 13.8%. The original choice of SAM mask used the
  annotated box; a tap does that job instead, as a user would.
- **A tap alone often picks a part**: a patch of the book cover (overlap 0.02), the laptop's keyboard (0.03), part
  of a chair. One point can't say "part or whole".
- **A box alone picks the surface**: for the cup, SAM outlined the table inside the box. Box + tap fixes that.
- **Every working prompt gives about the same outline** (book 0.87–0.89, cup 0.66–0.67, bottles 0.43–0.47), so the
  outline isn't what limits us now. Bottles score low against their box partly because a cylinder fills only part of
  its box's outline.
- **What's left is MapAnything's depth.** The big misses are wrong distances (book 0.19 vs 0.26 m, bottle 0.23 vs
  0.31 m). Triangulating the distance from the phone's own motion fixes some (book 0.26/0.26, laptop 0.58/0.58), but
  sizes don't follow (the laptop then reads ×1.34), so the depth inside the object is distorted too, not just its scale.

**Limits of this test:** 10 clips and 21 scored sides. Differences of 1–3 points between variants are within noise.
The next step that would tell us anything firmer is a bigger set (about 100 Objectron clips across all 9 categories,
roughly $3–6 of SAM and RunPod).

## 99 clips: the real numbers, and a correction that holds on unseen objects (2026-11-03)
**Data:** the first 11 test-split clips of each of Objectron's 9 categories (bike, book, bottle, camera, cereal box,
chair, cup, laptop, shoe): 99 clips, 1.9 GB. A few clips fail (SAM finds no match): 94–99 measured per variant.
Intervals are 95% bootstraps over clips (`compare_objectron.py`, `correct_objectron.py`).

| Variant (3 views + phone AR poses) | Median size error | Within 20% | Median scale |
|---|---|---|---|
| Name + tap (honest baseline) | 22.8% (20.5–26.7%) | 41% | ×0.82 |
| Name, mask picked with the annotated box (old baseline) | 22.0% (20.0–26.4%) | 42% | ×0.81 |
| Box + tap | 23.9% (20.2–29.3%) | 41% | ×0.88 |
| Distance triangulated from the phone's motion | 19.4% (16.3–21.9%) | 52% | ×0.88 |

**The 10-clip 13.8% was lucky.** On 99 clips the honest baseline is about 22–23%, and everything comes out small.

**The shrink depends on distance.** Name + tap measures objects within 35 cm at ×0.75, 35–50 cm at ×0.85, 50–80 cm
at ×0.90 and beyond 80 cm at ×0.97. So close-ups (books, cups, cereal boxes) are where MapAnything's depth is short.

**Corrections, scored only on clips they weren't fitted on.** Leave one category out (fit on 8 categories, score
the 9th) is the honest test for a new kind of object:

| Held out by category | Median error | Within 20% | Paired vs uncorrected (per clip) |
|---|---|---|---|
| Name + tap, no correction | 22.8% | 41% | (reference) |
| Name + tap × one factor (×1.19–1.27) | 22.4% | 45% | not computed (barely moves) |
| Name + tap, distance-aware factor | 19.0% (15.2–22.5%) | 51% | −6.6 points (−10.5 to −0.3), better on 64% |
| **Triangulation × one factor (×1.10–1.17)** | **17.7% (13.3–22.6%)** | **53%** | **−8.4 points (−11.8 to −4.1), better on 66%** |
| Triangulation, distance-aware factor | 18.3% (14.7–22.3%) | 55% | −6.3 points (−10.8 to −2.2), better on 68% |
| Near/far rule (triangulate if nearer than 0.4–0.6 m) | 18.0% (13.8–22.5%) | 54% | −7.7 points (−10.8 to −4.0), better on 62% |

**Recommendation:** with a phone's AR poses, triangulate the object's distance and multiply by about ×1.13. Expect
**about 18% median error on a kind of object the system has never seen**, about 1 in 2 sides within 20%. Without
phone motion, use the distance-aware factor on MapAnything alone (about 19%).

**It isn't uniform.** Per category (held out), triangulation × factor: book 5.8%, cereal box 10%, cup 17.6%, shoe
19.2%, camera 20.1%, bottle 20.4%, bike 23.1%, chair 26.1%, laptop 34.6%. Any single correction hurts chairs and
laptops, which were already right uncorrected (6.9% and 13.5%). The distance-aware factor keeps chairs right (7.6%)
but not laptops. A per-object choice would need more than distance to make.

**Still true:** for a measurement you'd trust, a known-size reference in the shot beats all of this.

**7 views, finished (99 clips).** Each of 7 views over ±3 s measured on its own, median per side over the views that
see it face-on (`fused_median`): **18.0% (15.6–19.9%), 58% of sides within 20%, scale ×0.90, with no fitted
correction**, against 22.8% for one view (paired −4.4 points, −8.0 to −1.5). Merging the views' points into one
cloud (`fused`) is 19.4%, no better (the cross-view smear). A held-out correction on top doesn't help any further
(18.2% leave one category out). So two different routes, 7-view medians with no learned factor, or phone-motion
triangulation × ~1.13, both land at **about 18% on unseen object types**. That looks like this stack's floor without
a known-size reference. Per category (7-view median): book 9.6%, chair 11.6%, cereal box 18.3%, bike 21.0%, camera
21.5%, bottle 22.3% (and laptop, cup, shoe in `compare_objectron.py` output).
