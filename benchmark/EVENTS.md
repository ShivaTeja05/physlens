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
| 11foot8 bridge: box truck under a 3.556 m beam | street pole camera, 30 fps | impact found at frame 196.6; beam clearance 3.63 m | hand label 196–200; sign 3.556 m | `world_events.py` |
| NHTSA 9001: barrier cart into a parked Camry | real-time rear view, 30 fps | parked Camry 4.54–4.78 m (lens range); Muse Spark: Camry, 4.869 m | 4.85 m | `side9001_3d.py` |
| | | cart 25–33 km/h ✗ | 62.19 km/h | |

## What failed, and why
- **Per-frame AI depth for fast motion** (9001 cart, 9070 at 1,000 fps in 3D): each frame's depth jitters by
  ~10 cm, and both AI models misjudge unusual lenses (63°/65° vs 52°/56°, neither reliably right).
- **Objects too close and blurred** (9001 cart: 1–3 m from a 30 fps camera at 17 m/s).
- **Objects too small** (Apollo 15 hammer and feather drop, `datasets/nasa/apollo15`): a few pixels each in
  analogue TV; neither SAM 3.1 nor frame differencing finds them.
- **Merging two cameras through AI depth** (9070 left + right, and the edited web video mistaken for a camera).

## What made the hybrid work
- A **plane from the scene**, not from AI depth: the car's side from its wheels' apparent sizes (9070; the AI plane
  was 52° off), the roof seen from overhead (15756).
- **Rigid parts only**: above the wheels (rolling tyres read slow), the side away from the impact.
- **Physics as calibration**: constant speed before impact gives the lens's radial term; a fit whose parameters sit
  at the edge of the search is under-constrained (two parameters trading off) and isn't trusted.
