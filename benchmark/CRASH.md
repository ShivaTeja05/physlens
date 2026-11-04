# Crash reconstruction against NHTSA ground truth (2026-11-03)

Footage and sensor data: NHTSA Vehicle Crash Test Database (public; see `datasets/nhtsa/SOURCE.md`).

## Test 9070, frontal: 2015 Lexus NX 200t into a rigid barrier (real-time side camera, 30 fps)
In the app (`samples/crash-frontal.mp4`, `physlens/labs/crash.py`); offline tests in `backend/tests/test_crash.py`.

| | Video | NHTSA |
|---|---|---|
| Impact speed | **57.7 km/h (90%: 56.3–59.1)** | 56.55 km/h ✓ |
| Max crush | at least 0.53 m | 0.71 m (rear-floorpan accelerometers) |
| Average deceleration | about 17 g | 21.7 g |
| Rebound | about 2.5 m/s | 1.6–1.7 m/s |

## Test 9001, side: crabbed 1,362 kg barrier into a standing 1,666 kg Camry (overhead camera, 1,000 fps)
`benchmark/side_impact_9001.py`.
- **Scale:** the barrier face's standard 1,676 mm width.
- **Motion:** median optical flow of features followed frame to frame, with a forward-backward check.
- **Ground truth:** the barrier's and car's centre-of-gravity accelerometers.

| | Video | NHTSA |
|---|---|---|
| Barrier speed | **62.3 km/h** | 62.19 km/h |
| Crab angle (its wheels are angled) | **27.2°** | 27.0° |
| Straight in / along the car | 55.4 / 28.5 km/h | 55.41 / 28.23 km/h |
| +20 ms: barrier / car (straight in) | 45.0 / 8.4 | 47.0 / 10.4 |
| +40 ms | 33.0 / 18.2 | 34.1 / 19.5 |
| +60 to +100 ms | barrier 15–17 ✗, car ≈ 25–26 | barrier 22–24, car 23–24 |
| Momentum in the two bodies at +100 ms | | 92% (a perfectly shared speed would be 24.9 km/h) |

**What it shows**
- **Before impact, video alone recovers the barrier's speed and its crab angle** to within 0.2 km/h and 0.2°.
- **The first 40 ms of the crash match the sensors** for both bodies.
- **After about 60 ms the barrier reading drops below the truth.** Its tracked features sit on the honeycomb face, which crushes and deforms, so they stop following the rigid cart (it may also pitch towards the camera).
- **The car's centre is read as the mean of its front and rear features,** because the car spins (front about 13 km/h, rear about 40 km/h). That lands 7–9% high.
- **Momentum works as physics says:** 92% stays in the two bodies; the rest goes to tyre friction and spin.
- **Data quirk:** for this test, NHTSA's per-channel ASCII files all return the same file (same ETag). The binary UDS files are correct: 512-byte blocks, each opening with an 8-byte marker.

**Next**
- **Better barrier tracking after impact:** follow its rigid frame (the blue cart), not the face, and correct for its height above the floor.
- **Into the app:** a two-body reconstruction (barrier and car, with spin) beside the overhead video.
