# Memory vs eyes benchmark

**Question:** does an AI answer physics questions from what it *sees* in a video, or from what it *remembers*?

Every clip is built so that answering from memory ("gravity is 9.81", "things fall down", "bounces lose energy",
"videos play in real time") gives the wrong answer. Two systems answer the same questions with the same scale information:

- **Plain Muse Spark** (`muse-spark-1.3`, watching the video): asked for gravity, bounciness, playback and orientation.
- **PhysLens** (SAM 3.1 tracking → physics fits with error bars → Muse Spark explanation checked against the numbers).

Results are in [RESULTS.md](RESULTS.md) and [results.json](results.json) (2026-11-03).

## Headline
**PhysLens passed 28 of 28 checks. Plain Muse Spark passed 19 of 28.**

| Clip | Truth | Plain Muse Spark | PhysLens |
|---|---|---|---|
| Real classroom drop | 9.81 | 9.80 ✓ | 9.73 ± 0.21 ✓ |
| Slow motion 2× | 2.45 | **0.30** ✗ | 2.48 ± 0.05 ✓ |
| Sped up 2× | 39.2 | **9.80** ✗ (and "real-time") | 39.50 ± 0.82 ✓, sped up ✓ |
| Played backwards | bounces gain speed | **0.59**, "slow-motion" ✗ | flagged reversed ✓ (no prediction made) |
| Upside down | gravity points up | 9.80 ✓, upside down ✓ | 9.81 ± 0.19 ✓, upside down ✓ |
| 12 fps | 9.81 | 9.80 ✓ | 10.33 ± 0.19 ✓ (see caveats) |
| Mars gravity, real time | 3.71 | **0.60** ✗ | 3.70 ± 0.04 ✓ |
| Bouncy twin (e = 0.85) | 9.81, 0.85 | **0.40**, "slow-motion" ✗; e = 0.80 ✓ | 9.79 ✓, e = 0.85 ✓ |
| Dead twin (e = 0.55) | 9.81, 0.55 | 9.80 ✓; **e = 0.30** ✗ | 9.79 ✓, e = 0.55 ✓ |

## What it shows
1. **Every gravity answer plain Muse Spark got right was exactly 9.80, the textbook value.** On every clip where the true
   value wasn't 9.81 (slow motion, sped up, Mars) it was wrong. That's the memory pattern.
2. **When it did try to measure, it was badly off:** 0.30, 0.40, 0.59 and 0.60 m/s², 6× to 16× too small. Its own
   reasoning on the slow-motion clip said a ~1.9 m fall "takes ~4.9 s"; the real fall takes about 1.2 s of video. It sees
   the video too sparsely to time motion.
3. It did notice the upside-down picture, and it called real slow motion "slow-motion" (with the wrong factor, about 8× instead of 2×).
4. PhysLens never answers from memory: SAM 3.1 tracks the ball in every frame, the physics is fitted by code, and Muse
   Spark only explains numbers it is handed. Its explanation is rejected if it uses a number that isn't in the measurements.

## Caveats (read before quoting)
- Small set: 9 clips, six derived from **one** real recording (time-scaled, reversed, flipped, frame-dropped) and three
  rendered. More real footage is the next step.
- 12 fps: PhysLens's 10.33 passes the 10% threshold but is **outside its own error bar** (true 9.81). At low frame rates,
  motion blur biases the centre and the error bar is too optimistic. This is a known limit (an expected-failure test in
  `backend/tests/test_trick_clips.py`).
- One clip can't separate slow motion from weaker gravity: both stretch a fall the same way. PhysLens says "either… or…"
  and is scored on the gravity number; Mars isn't scored on playback.
- Plain Muse Spark was asked once per clip, with no retries or prompt tuning beyond giving it the same scale as PhysLens.
- Cost of this run: about $1, metered in `backend/.spend.jsonl` against a $50 cap (`physlens/budget.py`).

## Re-running
```bash
cd backend
.venv/bin/python ../benchmark/run_benchmark.py          # all clips (~$1)
.venv/bin/python ../benchmark/run_benchmark.py real     # one clip, keeps the rest
```
Clips live in `clips/`. The real-footage variants are made from `samples/real-classroom-drop.mp4` with ffmpeg
(`setpts` for speed, `reverse`, `vflip`, every-other-frame select). The others are rendered by `physlens/synth.py`.
