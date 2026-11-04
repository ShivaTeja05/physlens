# Use cases

Status key: **Works** (shown on real footage against official data) · **Prototype** (runs, not yet validated on
real footage) · **Next** (the pieces exist; not built as a feature yet).

## 1. Crash reconstruction from video · Works
**Who:** crash investigators, insurers, safety researchers, journalists checking a claim.
**What it does:** from a fixed camera's video, the impact speed, when contact happened, how far the front crushed and
over what time, the rebound, how each car moved and turned after.
**Evidence:** four NHTSA crash tests, approach speeds within 0.0–0.6% of the speed trap; a crush curve within 1.5 cm
of the accelerometers; a two-car momentum balance within 6% ([benchmark/EVENTS.md](../benchmark/EVENTS.md)).
**Needs:** a camera that doesn't move during the event, one known length (a car's width or wheelbase, a marked
target), and the object large enough in the picture. Fast objects close to a 30 fps camera don't work (see limits).

## 2. Clearance and "will it fit?" checks · Works
**Who:** fleet managers, bridge and tunnel operators, delivery planning.
**What it does:** measures a structure's clearance in the 3D world and checks a vehicle against it; flags it before
contact. **Evidence:** the 11foot8 bridge: clearance 3.63 m (sign 3.556 m), impact found at frame 196.6 (hand label
196–200), the truck flagged at least 8% too tall.

## 3. Physics lab from a phone video · Works (synthetic + one real clip)
**Who:** students and teachers.
**What it does:** films a drop, a roll off a table, two cans racing down a slope, weights on springs; measures g,
bounciness, the hidden mass distribution or an unknown weight; **locks predictions before they happen** and checks
them. It also finds the equation of motion itself. **Evidence:** real classroom drop: g = 9.73 ± 0.21, next bounce
called within 1 mm. Stage experiments proven on rendered clips with known physics ([FEATURES.md](FEATURES.md)).

## 4. Fake-video / slow-motion detector · Works (benchmark)
**Who:** fact-checkers, platforms.
**What it does:** measures gravity in a clip; a clip that's slowed, sped up or generated shows the wrong g.
**Evidence:** 28/28 checks on trick clips vs 19/28 for Muse Spark alone ([benchmark/README.md](../benchmark/README.md)).

## 5. Scan: measure what's in a photo · Works
**Who:** anyone with a phone: builders, students, hobby astronomers.
**What it does:** names the scene, outlines its parts, and measures them: counts and rows (floors of a building), lean
from upright, contact, a Moon's lit share, metres from one known size; flags where the AI's words disagree with the
measurement ("gibbous" vs 46.5% lit).

## 6. Pull out objects (faces, plates, products) · Next
**Who:** privacy (blur faces and number plates before sharing a video), shops (cut out products), inventory.
**Pieces that exist:** SAM 3.1 outlines anything you name, in a photo or through a video; Muse Spark names objects and
gives published sizes; the 3D step gives each object its own points and measurements.
**Not built yet:** one "extract everything" call that returns each object's outline, name, size and cut-out.
Face *recognition* (who someone is) is deliberately out of scope.

## 7. Multi-camera and moving-camera scenes · Prototype
**What it does now:** worlds from a moving camera (someone walking with a phone), a fixed camera with stated
assumptions, unseen sides predicted by symmetry and labelled. **Not reliable yet:** merging two cameras of the same
event through AI depth (tried on NHTSA 9070; failed, see EVENTS.md).

## What doesn't work (so demos avoid it)
- Per-frame AI depth for fast motion (it jitters ~10 cm per frame).
- Objects too close and blurred (a cart 1–3 m from a 30 fps camera at 17 m/s: 25–33 km/h measured vs 62).
- Objects a few pixels across (Apollo 15 hammer and feather).
- Lenses that can't be calibrated from the scene (NHTSA 7371's near-fisheye overhead: in progress).
