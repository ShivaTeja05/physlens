# Moving camera: physics when the camera itself moves

Clip: `samples/moving-camera-drop.mp4`, rendered by `backend/physlens/synth_moving.py`. A ball is dropped from 1.5 m
(g = 9.81 m/s², bounciness 0.78, radius 6 cm) and filmed at 60 fps by a camera that slides 1.2 m sideways, bobs, and
pans to follow the ball, with hand shake. The true camera pose and ball position are saved for every frame
(`.truth.json`). Run: `PYTHONPATH=backend backend/.venv/bin/python benchmark/moving_camera_eval.py` (results in
`moving_camera.json`, chart `media/moving_camera.png`, scene `records/moving_camera_scene.json`).

Pipeline (`backend/physlens/moving_camera.py`):
1. 17 keyframes (every 10th frame) → MapAnything on RunPod, given nothing: it estimates focal length, poses, depth.
2. Every other frame gets its own pose: background features matched to the nearest keyframe (whose depth makes each
   match a 3D point) → PnP. Leaving it to interpolation flattened the camera's swings and biased gravity.
3. Floor = the plane in the point cloud whose normal points up for the camera → gravity direction, height 0.
4. Ball centre + apparent radius per frame; distance = focal × radius ÷ pixel radius. The radius itself comes from
   the floor: at every bounce the centre must sit one radius above the floor plane (MapAnything's depth at a 9-px
   ball blends with the wall behind it and gave 9.7 cm).
5. Same physics code as the fixed-camera lab (bounce split, one-g free-fall fit, contacts from meeting arcs), then
   atoms → MuJoCo.

| | gravity (m/s²) | bounciness | ball radius |
|---|---|---|---|
| truth | 9.81 | 0.78 | 6.0 cm |
| pixels, camera motion ignored | 3.78 (−62%) | 0.79, 0.77, 0.77 | (given) |
| keyframes only, depth radius | 13.3 (+36%) | 0.73, 0.76 | 9.7 cm |
| + floor-contact radius | 8.52 (−13%) | 0.73, 0.77 | 6.0 cm |
| + per-frame PnP poses (blind: nothing given) | 8.94 ± 0.03 (−9%) | 0.767, 0.774, 0.783 | 5.7 cm |
| + lens given to MapAnything | 8.41 ± 0.03 (−14%) | 0.767, 0.774, 0.787 | 5.4 cm |
| + ball size given (`--size`) | **9.35 ± 0.03 (−4.7%)** | 0.767, 0.774, 0.783 | (given) |
| + lens and size given | **9.38 ± 0.03 (−4.4%)** | 0.767, 0.774, 0.787 | (given) |
| control: true camera poses, same lifting code | 9.79 (−0.2%) | 0.777, 0.780, 0.782 | 6.1 cm |

Height error 5 cm RMS blind, 2.0 cm with the size given; MuJoCo replay of the fitted scene 2.5–5 cm RMS.
The control row shows the lifting and physics code is right; what's left (4–5% with a known size) comes from
MapAnything's poses and depth, and the scene's overall scale wanders by ±10% depending on what it's given.

Bounciness is a ratio of speeds, so the scene's scale cancels and it is right to about 1%. Gravity is in m/s², so it
carries MapAnything's metric-scale error. One known size (a ball, a door, a lane marking, a car's wheelbase) is
the fix that works: `run(..., known_radius_m=...)` rescales every length, camera path included. Giving the lens
(`intrinsics=`) makes MapAnything's focal right (555 vs 560 px) but doesn't improve gravity. Next lever, not done:
more keyframes or a joint refinement of poses and the object path (bundle adjustment).

Not yet tested: real phone footage (needs SAM 3.1 masks instead of the colour mask), objects that aren't round
(distance from apparent size needs a shape; CoTracker3 points on the object + depth would replace it), fast pans
with motion blur.
