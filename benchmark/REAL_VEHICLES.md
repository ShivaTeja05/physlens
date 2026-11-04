# Real footage: vehicles in 3D, and point worlds

Clips: openly licensed, from Wikimedia Commons (`datasets/commons/SOURCE.md`). Scripts: `real_vehicles.py` (speeds,
sizes), `export_pointworlds.py` (point worlds for the app's **3D points** mode). Code: `backend/physlens/vehicle3d.py`,
`backend/physlens/pointworld.py`.

Method: keyframes → MapAnything (depth + camera poses, one shared frame) + SAM 3.1 mask per keyframe → the object's
3D points (pixels on depth jumps and low-confidence pixels removed) → its position (median point) → ground frame →
one known size checks or fixes the scale → speeds, sizes, the moment of impact.

## Box truck vs the 11foot8 bridge (fixed pole camera, 30 fps)

| | value | how |
|---|---|---|
| scale | MapAnything's own was 0.9955× the bridge's | bridge underside (SAM "bridge", bottom edge over the road) = 3.556 m; spread of that edge ±15% |
| approach speed | **18.3 ± 0.4 km/h** | one constant-speed fit over 21 keyframes |
| last quarter-second | ~14 km/h | frames 188–196 |
| impact | between frames 196 and 200 (6.5–6.7 s) | the box roof is peeled open from frame 200; SAM no longer calls it a "box truck" |
| check: box width | **2.45 m** vs ~2.44 m (8 ft rental box) | from the truck's points |
| check: roof height | 3.14 m, **wrong** (it hit a 3.556 m beam, so it must be taller) | the roof is a thin surface seen at a grazing angle; its edge is trimmed |
| camera height | 4.5 m | |

Per-keyframe speeds jump between 3 and 32 km/h (every 4 frames, median-point jitter); a motion model fitted to all of
them is what gives ±0.4 km/h. That's the case for simulating rather than differencing frames.

## Rally car past a handheld camera (Monte-Carlo 2019, 30 fps, camera panning)

| | value | how |
|---|---|---|
| scale | MapAnything's own was **1.012×** the known length | car length side-on (frames 174–178) vs 4.085 m (2019 Toyota Yaris WRC) |
| speed | **124 km/h** median, 115–144 across frames | 10 keyframes over 0.6 s; no official speed at this point |
| speed in car lengths | 8.4 lengths/s | needs no scale at all |
| spectator's camera | moved 0.5 m while panning | removed |
| car size from points | 4.03 × 2.13 × 1.40 m (true 4.085 × 1.875 × 1.19) | width/height read high in the three-quarter views |

## Point worlds

| | points | of which the vehicle | file |
|---|---|---|---|
| truck | 1.42 M | 1.04 M over 25 moments | 14 MB |
| rally | 1.74 M | 0.19 M over 10 moments | 17 MB |

Every pixel MapAnything measured is a point (2× upsampled: the in-between points are interpolated, and the app says
so). Small points, coloured from the video, metres, ground at y = 0. Each object's points stay grouped per moment,
so the app draws its size box (length × width × height) and its trail.

Known weak spots: thin surfaces at grazing angles (truck roof); the plane found as "ground" can be a verge instead of
the road (rally: the spectators' bank, ~0.4 m higher); text burned into the video (the 11foot8.com watermark)
becomes part of the truck's colours; SAM loses an object once it's badly damaged.
