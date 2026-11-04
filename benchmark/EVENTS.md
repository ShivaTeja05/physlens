# Events measured from video, against ground truth (2026-11-04)

All footage is public (NHTSA, NASA, Wikimedia Commons; see each `datasets/*/SOURCE.md`). "Hybrid" is
`physlens/hybrid.py`: features followed in the image through every frame, made metric on a plane from the scene
(no per-frame AI depth), the lens's radial term from physics (a stretch of constant speed), one reference length.

| Event | Camera | Measured | Ground truth | Script |
|---|---|---|---|---|
| NHTSA 9070: Lexus NX into a rigid barrier | "Right Overall", 1,000 fps | approach **56.81 km/h** | 56.55 km/h (speed trap) | `hybrid_9070.py` |
| | | largest crush **0.728 m at 75 ms** | 0.707 m at 74.1 ms (accelerometers) | |
| | | crush curve 0–300 ms within **1.5 cm** RMS | accelerometers, integrated | |
| | | rebound 1.73 m/s | 1.44 m/s (same window) | |
| | real-time left, 30 fps | impact at 1.473 s; speed from the body's poses **56.9 km/h** (true-length scale) | 2D lab: 1.470 s; 56.55 km/h | `crash3d.py`, `simulate_worlds.py` |
| NHTSA 15756: Nissan Kicks sideways into a pole | overhead, 1,000 fps | speed **31.29 km/h** | 31.09 km/h | `pole15756.py` |
| | | travel at **74.6°** to the car's axis | 75° (the test's impact angle) | |
| | | length 4.27 m (from the width scale) | 4.354 m | |
| | | 0.36 m further towards the pole, stops at 65 ms; turns 35° by 300 ms | (accelerometers not yet compared) | |
| | 3D world (`pole_world.py`) | the car's body moved by that measurement; at 300 ms it sits a median 0.21 m from the car's measured points (the struck side is crushed; the turn's pivot is approximate) | | |
| 11foot8 bridge: box truck under a 3.556 m beam | street pole camera, 30 fps | impact found at frame 196.6; beam clearance 3.63 m | hand label 196–200; sign 3.556 m | `world_events.py` |
| NHTSA 8555: Ford F-150 (2,674 kg) into the rear of a standing Jeep Liberty (1,927 kg) | overhead, 1,000 fps (`rear8555.py`) | approach **69.09 km/h** | 69.1 km/h (speed trap) | `rear8555.py` |
| | | Jeep's speed change **+47.4 km/h** (100–200 ms); its curve within ~1–4 km/h from 0 to 170 ms | +47.8 km/h (accelerometers) | |
| | | F-150's speed change −35.8 km/h; within ~2.5 km/h from 120 to 200 ms, low at 80–100 ms (cab at the picture's edge), up to 12% high after 220 ms | −34.0 km/h | |
| | | momentum: Jeep gained ÷ F-150 lost = **0.954** | 1.013 from the accelerometers | |
| NHTSA 7371: Yaris at 113 km/h into a standing Yaris, 15°, both turning (in progress) | overhead wide, 1,000 fps (`yaris7371.py`, rigid bodies that turn) | approach 99.8 km/h (lens from floor seams, k1 −0.12) / 104.0 (no lens fix); 17 frames at the picture's edge | 113.0 km/h | `yaris7371.py` |
| | | struck car's speed change in its own axes at 100 / 150 ms: X −51.6 / −46.6, Y +16.2 / +18.5 (k1 −0.12); X −59.2 / −57.4, Y +18.1 / +21.8 (no fix) | X −58.4 / −57.5, Y +22.9 / +19.9 (accelerometers) | |
| | | not yet reliable: the lens can't be calibrated from this floor's seams (line by line k1 −0.40 to +0.34) | | |
| NHTSA 9001: barrier cart into a parked Camry | real-time rear view, 30 fps | parked Camry 4.54–4.78 m (lens range); Muse Spark: Camry, 4.869 m | 4.85 m | `side9001_3d.py` |
| | | cart 25–33 km/h ✗ | 62.19 km/h | |

## What failed, and why
- **Per-frame AI depth for fast motion** (9001 cart, 9070 at 1,000 fps in 3D): each frame's depth jitters by
  ~10 cm, and both AI models misjudge unusual lenses (63°/65° vs 52°/56°, neither reliably right).
- **Objects too close and blurred** (9001 cart: 1–3 m from a 30 fps camera at 17 m/s).
- **Objects too small** (Apollo 15 hammer and feather drop, `datasets/nasa/apollo15`): a few pixels each in
  analogue TV; neither SAM 3.1 nor frame differencing finds them.
- **Merging two cameras through AI depth** (9070 left + right, and the edited web video mistaken for a camera).

- **Lens from rigidity** (`hybrid.lens_from_rigidity`: a body must keep its size on the floor plane as it crosses
  the picture; lens and floor tilt fitted together). On 8555, where the answers are known, it chose k1 +0.10 and
  moved the approach to 64.9 km/h (69.1) and the Jeep to 51.2 (47.9), worse than the floor lines' 69.13 / 48.5:
  bodies pitch and bounce during a crash, which changes their size in the picture too. Not used.
- **Yaris (7371) scale and lens**: MoGe-3's points of the struck car give widths of 1.45–1.78 m depending on the
  part and a length 9% short of its width scale; its field of view changes 71–78° between frames; the floor's seams
  give k1 −0.40 to +0.34 line by line. Neither the scale nor the lens can be pinned from this footage yet.

## What made the hybrid work
- **Lens from straight floor lines** when the motion covers only part of the picture (8555: the physics fit sat at
  its search edge); **each feature at its own height** when a tracked part spans several (a cab's roof and hood),
  and **no glass** (it shows the room's reflection).
- A **plane from the scene**, not from AI depth: the car's side from its wheels' apparent sizes (9070; the AI plane
  was 52° off), the roof seen from overhead (15756).
- **Rigid parts only**: above the wheels (rolling tyres read slow), the side away from the impact.
- **Physics as calibration**: constant speed before impact gives the lens's radial term; a fit whose parameters sit
  at the edge of the search is under-constrained (two parameters trading off) and isn't trusted.
