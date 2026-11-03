# Source and licence

**"Tracker Video Analysis: Bouncing Ball"** by Loo Kang Wee, ComPADRE Open Source Physics collection.
Licence: Creative Commons Attribution 3.0 (CC BY 3.0).
Item page: https://www.compadre.org/OSP/items/detail.cfm?ID=11705
Downloaded 2026-11-03 as `package.trz` (Tracker package, 692 KB), extracted into `extracted/`.

Used here as a real-footage test. The video is copied to `../../samples/real-classroom-drop.mp4`.

## Ground truth taken from the teacher's Tracker file
- Scale: a 1.000 m tape on the wall spans **161.143 px** (vertical), so `px_per_m = 161.143`.
- 25 fps, 720×576, 70 frames.
- The teacher hand-clicked the ball in all 70 frames (`mass A`). A free-fall fit to those clicks gives
  g = 10.09 ± 0.08 m/s². That's about 3% above 9.81, probably because the ball falls slightly in front of
  the wall (nearer the camera than the tape), so it looks bigger than the tape's scale assumes.

## PhysLens result (prototype v0, OpenCV motion tracker, ruler scale 161.143 px/m)
- Ball found in 56/70 frames. Median distance from the teacher's clicks is 2.1 px, worst 5.9 px. The missed
  frames are the ball in the hand and motion-blurred frames at the first floor contact, which is detected
  as a contact hidden in a tracking gap.
- g = 9.86 ± 0.19 m/s² (Earth 9.81; the teacher's clicks give 10.09 ± 0.08).
- Bounciness is not constant: e = 0.62 at a 6.4 m/s impact, 0.69 at 4.0 m/s, 0.73 at 2.8 m/s. The teacher's
  clicks show the same thing (0.63, 0.69, 0.73). PhysLens rejects the constant-bounciness law and uses a
  speed-dependent one.
- Locked after the third contact, using earlier frames only: next peak called at 20.5 cm, actual 20.4 cm
  (hit); landing time hit. With the constant-bounciness law the same prediction missed (31.8 cm called,
  39.0 cm actual), which is why the law check exists.
