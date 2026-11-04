# PhysLens features in detail (Lab, Scan, records, crash reconstruction, 3D world, MuJoCo)

> **Versions.** This branch (`simulator`, tag `v2.0-simulator`) has everything: the core Lab (also on `main`, tag
> `v1.0-normal`) plus the simulator work: what-if replays, crash reconstruction from NHTSA crash tests, the 3D world
> (Rapier) and MuJoCo fits, physics from a moving camera, and 3D point worlds of real footage
> (`benchmark/MOVING_CAMERA.md`, `benchmark/REAL_VEHICLES.md`), now with full-resolution MoGe-3 depth, SAM 3.1 video
> tracking and per-object fusion for sizes (`benchmark/SHARP_POINTS.md`).

The Lab started as a vertical slice built around **drop & bounce** plus three stage experiments (*Catch the cup*, *Mystery Cans*, *Weigh it by watching*). You upload a clip (or pick a test clip), the engine tracks the ball, measures gravity and bounciness with error bars, locks predictions for the next bounce using only earlier frames, checks them, and explains the result. The stage display plays it all back in sync with the video, keynote style.

It runs on a laptop with **no GPU and no Meta API key**. SAM 3.1 and Muse Spark have slots ready to switch on.

## Run it
Setup and commands: [SETUP.md](SETUP.md). Demo scripts: [DEMO.md](DEMO.md).

**Scale for your own clips:** in the app, next to *Upload clip*, choose *Ball ⌀ cm* (the ball's diameter) or *Ruler px/m* (pixels between two marks 1 m apart in the shot). CLI: `--diameter-cm 6.7` or `--px-per-m 161.1`.

## Stage experiments
Each one tracks with SAM 3.1 (two objects at once for the cans and the springs), locks a prediction from early frames only, then checks it against what happens. Results on the sample clips, tracked through SAM 3.1:

| Experiment | What it measures | Locked call | Reality |
|---|---|---|---|
| **Catch the cup** (`cup-catch`) | Roll speed off a 75 cm table, time in the air | Lands 43.0 cm past the edge (band 42.0–43.9) | 43.0 cm ✓, touchdown time ✓ |
| **Mystery Cans** (`mystery-cans`) | Each can's acceleration, then k = I/(m r²): where its weight hides | Red wins by 0.19 s; red 1.170 s, blue 1.364 s | 0.19 s, 1.170 s, 1.363 s: 3/3 ✓ (k 0.40 / 0.90; truth 0.40 / 0.90) |
| **Weigh it by watching** (`weigh-springs`) | Bob periods of a known 100 g weight and a mystery weight on identical springs | Mystery weighs 250 g (band 245–255); 3rd next crest at 3.796 s | Scale 250 g ✓ (stretch check agrees: 249.8 g), crest 3.79 s ✓ |

Uploads: pick the experiment next to *Upload clip* and fill in the tape-measure values it asks for (table height, course length, known weight…). The sample clips are rendered with known physics (`physlens/synth_labs.py`); real footage of these three is next.

## Law discovery: finding the equation, not picking it
Every run also searches for the equation of motion on its own (`physlens/physics/discover.py`): 12 combinations of
four force terms (constant pull, spring pull towards a centre, drag ∝ v, drag ∝ v²). A quick derivative fit shortlists
the best few. Each is then fitted as a simulated trajectory to the raw positions (no smoothing bias), and the
simplest one whose every term is clearly non-zero wins (BIC). No neural network and no memory: just a readable formula.

| Clip | Found | Truth |
|---|---|---|
| Real classroom drop (SAM 3.1) | a = −9.73 ± 0.21 (gravity only, no drag) | 9.81 |
| Slow-mo 2× / mystery clip | a = −2.45 / a = −1.81 | 2.45 / 1.81 |
| Catch the cup, in the air | sideways a = 0, vertical a = −9.92 | no force, 9.81 |
| Mystery Cans, along the slope | a = 1.46 and 1.07 (constant push) | 1.46, 1.08 |
| Weigh it by watching | a = −200·(y − y₀) − 0.50·v and −80.07·(y − y₀) − 0.50·v | ω² 200 and 80 (k = 20 N/m) |

Blind tests (`tests/test_discover.py`) also recover linear drag, speed-squared air drag, upside-down gravity, weak
damping and "no force", at 25 fps as well as 60–120 fps.

## Scan mode: point it at anything
Switch **Lab → Scan** in the header. Pick a sample, upload a photo, or use the camera and press **Scan**
(`physlens/scan.py`, `POST /api/scan`, about 45–65 s and $0.04–0.07 per new image; results are cached per image).

1. **Look:** Muse Spark (vision) names the scene (structure, astronomy, motion, activity, machine, nature), the parts
   worth outlining, how they act on each other (load, light, energy…), and how the thing works.
2. **Find:** SAM 3.1 outlines every instance of each part.
3. **Measure** (deterministic): counts, rows and columns of windows or balconies (floors), lean from upright, how
   high a sub-part sits, whether related parts touch, a round body's lit share (disc fitted to its sunlit edge,
   ± where the shadow line falls), and metres from one known size you type in.
4. **Explain:** every "what the camera saw" sentence must use only measured numbers (evidence-or-reject); "how it
   works" is shown as background knowledge, not measurement; where the model's words make a testable claim (a
   Moon's phase name) it is checked against the measurement ("Words vs eyes").
5. **Draw:** scanning sweep over an edge wireframe, glowing outlines, margin call-outs, floor lines, lean
   protractors, flowing relation arrows with contact checks, and the narrator lighting up each step's parts.

Samples (in `samples/scan/`, credits in `settings.json`): apartment block (17 windows and 15 balconies in 3 rows,
6 pillars), first-quarter Moon (46.5% lit, 45–50%; Muse Spark said "gibbous", flagged), motorbike (1.4° lean,
contact with the floor), classroom (free-fall set-up, suggests the drop experiment), basketball, and Objectron
frames. **Auto-detect** in Lab mode does the same look on three moments of an uploaded video to pick the experiment
and name what to track (the classroom clip: "Drop & bounce, tracking the orange ball").

## Ask PhysLens (voice)
Under the stage in Lab and Scan: **hold 🎙 (or the space bar) and ask**, or type. Speech goes live to Meta's
**Muse Voice** (`muse-voice-transcribe-1.0`, realtime-only: `wss://api.meta.ai/v1/asr/realtime`, relayed by the
engine at `/api/voice`), so words appear while you speak. The question goes to `POST /api/ask` with the scan or
run it's about: Muse Spark answers **only what was asked**, in 1–3 spoken sentences, from that scan's or run's
measurements, and **remembers the conversation** so follow-ups ("and the other one?") stay consistent. Every number
it says is checked against the measurements (one retry, else flagged); things that weren't measured are said to be
unmeasured, then explained from general physics without numbers. Answers are spoken back; in Scan the parts they're
about light up. About 4–7 s per answer (`reasoning_effort: low`).

## Trajectories
Lab videos draw each object's whole path: the line, a dot every 0.1 s (like a strobe photo: wider spacing = faster),
and markers at each bounce (◆) and peak (▲) with their times. **Path** in the transport switches between *so far*,
*whole run* and *off*. In Scan photos, motion can only be inferred: where Muse Spark says something is moving
toward something, the scan draws the arc a thrown object follows and labels it "likely path, inferred".

## Records: replay and what-if
Every run is written to a **black-box record** in `records/physlens.db` (SQLite, one file; `physlens/recorder.py`).
Old runs are imported from `replays/`:

| Table | What it holds |
|---|---|
| `capture` | video fingerprint (SHA-256), frame rate, scale and its uncertainty, experiment, models used |
| `object` | each tracked thing |
| `state` | every sampled instant: position and velocity in metres, ± position |
| `event` | key moments: start, each contact, each peak, end |
| `law` | the equations found from the motion, ± errors |
| `prediction` | locked calls with their fingerprints, and what really happened |

Contacts are found between frames: the falling and rising arcs either side are fitted to the positions, and the
contact is where they meet, with the speed into and out of it. On the rendered drop that recovers the true
bounciness 0.78 at every bounce; on the real classroom clip, 0.62 → 0.69 → 0.72 → 0.75 as impacts get gentler.

**Replay & what-if** (`physlens/replay.py`, Lab sidebar): a record is re-simulated from its stored start state with
the law found for it and each bounce's measured bounciness, then compared with what happened. The replay follows
the real classroom drop to within **1.9 cm** RMS, and the rendered drop to within 0.5 cm. Change gravity,
bounciness, launch speed or drag, and the what-if path is drawn over the real video next to the replay, with a plain
summary ("Peak 1: 94 cm at 2.26 s instead of 82 cm at 1.08 s"). The **Records** card reopens any past run.
`GET /api/records`, `GET /api/records/{id}`, `POST /api/records/{id}/whatif`. Supported so far: vertical motion
(drops, bounces, flights, springs).

## Crash reconstruction (real crash-test footage)
Sample **Real crash test (frontal)**: NHTSA test 9070, a 2015 Lexus NX 200t driven into a rigid barrier, filmed by a
fixed real-time side camera (30 fps); footage and sensor data from NHTSA's public Vehicle Crash Test Database
(`datasets/nhtsa/`, `SOURCE.md`). `physlens/labs/crash.py`:
- **Edges, not centre:** a background-subtraction box tracker (`perception/box_tracker.py`; free, frame-exact) gives the car's front and rear edges in every frame.
- **The approach calibrates the camera:** the car coasts at constant speed and is rigid. Its known length (4.63 m) gives the scale, and a perspective correction is tested and kept only if it clearly helps (here it doesn't).
- **The speed is locked before impact,** with an error bar that includes a ±2% edge-definition systematic.
- **Then impact, crush, deceleration and rebound,** with key moments saved to the recorder and an animated to-scale reconstruction under the video.
- **What-if:** "same car at another speed", from a labelled spring model of the car's front.

| | From the 30 fps video | NHTSA's own measurement |
|---|---|---|
| Impact speed | **57.7 km/h (90%: 56.3–59.1)** | 56.55 km/h, inside the band ✓ |
| Law of the approach | a = 0 (coasting, no force) | released before the barrier |
| Max crush | at least 0.53 m | 0.71 m (rear-floorpan accelerometers, integrated) |
| Average deceleration | about 17 g | 21.7 g (peak 57–59 g at 38 ms) |
| Rebound | about 2.5 m/s (the crushed front springs back) | 1.6–1.7 m/s |

Speed from video is solid. Crush and deceleration from 30 fps are coarse, because the crush lasts about 74 ms (2–3
frames). The test's 1,000 fps high-speed cameras would resolve them. Next: those cameras, two-body impacts (the
side test 9001, where momentum is checkable), and dashcam footage, which has no ground truth.

## Atomic scenes and the 3D world
Every record exports as an **atomic scene** (`physlens/scene.py`, `GET /api/records/{id}/scene`, format
`physlens-scene/1`):
- **World:** gravity (measured when a law was found), the ground, and fixed obstacles such as a crash barrier.
- **Objects:** shape, size, mass, bounciness, friction and post-impact drag, each with its source (measured / given / assumed).
- **Atoms:** one object at one instant: position, velocity, orientation, ± error, and which fields were measured. One camera sees no depth, so z and orientation are always marked assumed.
- **Moments:** contacts, impacts, crush.

**3D world** (Lab → *3D world*, three.js + react-three-fiber + Rapier):
- **Solid:** the measured atoms, replayed exactly. A car box shortens with its measured crush.
- **Magenta ghost:** Rapier re-simulating from the first atom with the measured parameters. Sliders change gravity, bounciness and speed.
- **Drift readout:** how far physics alone departs from reality.

On the real classroom drop the re-simulation stays within 7.4 cm RMS, using each bounce's measured bounciness. On
the NHTSA crash it stays within 28 cm, using the measured restitution and post-impact drag. The remaining gap there is
the crush, which rigid bodies can't model, and the panel says so.

## MuJoCo, fitted to the record (system identification)
**Fit MuJoCo** in the 3D world (`physlens/sim_mujoco.py`, `POST /api/records/{id}/mujoco`) exports the atomic
scene as a MuJoCo model (MJCF) and fits its unknown physics until the simulation reproduces the measured atoms. The
fitted model downloads as MJCF (`/api/records/{id}/mujoco.xml`), with the measured start as a keyframe, so the event
replays in any MuJoCo viewer.

| Record | Fitted | Simulation vs measured atoms |
|---|---|---|
| Rendered drop (bounciness 0.78) | bounciness ≈ 0.765 (gravity held at the measured 9.81) | **1.1 cm** RMS (3.8 unfitted) |
| Real classroom drop | speed-dependent bounciness e0 − e1·v (gravity held at the measured 9.73) | **6.8 cm** RMS (26.6 unfitted) |
| NHTSA frontal crash | crumple zone as MuJoCo's soft contact: crush 0.60 m (accelerometer 0.71), restitution 0.15, impact 58.7 km/h (official 56.55), drag after rebound | **10.3 cm** RMS (529.5 unfitted) |

Lessons from building it:
- **Bouncy contacts:** MuJoCo's default soft contact pumps energy into bouncy collisions (a target bounciness of 0.9 came back as 20). The direct spring-damper form keeps it within a few percent.
- **Gravity stays at the engine's measurement:** a whole-clip fit bends it to absorb contact details. Fitted freely, it drifted to 11.5 m/s².
- **The optimiser:** a derivative-free optimiser (Powell) works where finite-difference gradients stall on a stepped simulation.

## Speed
A real classroom run went from about 3 minutes to **24 s**:
- **SAM only gets the moving stretch:** a quick local tracker (about 1 s) finds where the object moves, and only that stretch is sent.
- **Small, fast objects skip the name prompt:** they never match by name, so they go straight to a box prompt.
- **Several objects are tracked in parallel.**
- **Muse Spark uses `reasoning_effort: low`:** about half the time.
- **Uploads are transcoded as H.264 Baseline without B-frames:** with ffmpeg's defaults, SAM billed a 70-frame clip as 1,680 frames.

## Memory vs eyes benchmark
On 9 trick clips (slowed, sped up, reversed, upside down, Mars gravity, identical-looking balls with different
bounciness), **PhysLens passed 28/28 checks and plain Muse Spark 19/28**. Every gravity value plain Muse Spark got right
was exactly the textbook 9.80; it failed every clip where gravity wasn't 9.81. Details and caveats:
[benchmark/README.md](../benchmark/README.md).

**Spending cap:** every Meta API call is priced from its reported usage and logged to `backend/.spend.jsonl`. Calls are
refused once the total reaches $50 (`PHYSLENS_BUDGET_USD`). Current total: `GET /api/budget`.

## Test clips
| Clip | True gravity in the video | What it shows |
|---|---|---|
| `real-classroom-drop` | 9.81 (real footage, 25 fps) | Real ball against a wall with a 1 m tape. Measures 9.86 ± 0.19, finds bounciness falls with impact speed, calls the next peak (20.5 cm called, 20.4 cm actual). See [real_clips/compadre-bouncing-ball/SOURCE.md](../real_clips/compadre-bouncing-ball/SOURCE.md) |
| `drop-real` | 9.81 | Matches real-world gravity |
| `drop-slowmo-2x` | 2.45 (9.81 at half speed) | **Memory vs eyes:** a model "remembers" 9.81; PhysLens measures 2.47 and reports 2× slow motion |
| `drop-low-gravity` | 1.81 (like AI-generated video) | Physics lie detector: physically inconsistent with Earth |

## What's where
```
contract/events.schema.json     the event stream every part agrees on
backend/physlens/
  physics/                      deterministic maths, no AI: calibration, bounce detection, law discovery,
                                3 candidate laws fitted jointly (AIC race), bounce-law check (constant vs
                                speed-dependent bounciness), predictions with 90% bands
  perception/                   tracker interface; SAM 3.1 slot; OpenCV fallbacks: motion tracker (default:
                                background subtraction, shadow rejection, best-path linking) and colour tracker
  agent/                        scientist agent (Muse Spark slot, template fallback) + evidence-or-reject verifier
  ledger.py                     locks predictions with a SHA-256 hash + timestamp before the outcome
  pipeline.py                   track → calibrate → fit → lock → verify → explain, as timed events
  api.py                        FastAPI orchestrator: runs, uploads (auto-converted to H.264), WebSocket stream
  labs/                         stage experiments: projectile (cup), race (cans), springs (weigh), shared helpers
  experiments.py                experiment cards: objects to track, tape-measure params, narration
  synth.py, synth_labs.py       test clips with known physics
web/src/
  App.tsx                       stage, transport, presenter beats (pause + speak each narration line), keynote
                                moments (title card, prediction-locked flash, HIT/MISS stamp), experiment picker
  overlay.ts                    canvas: speed-coloured trail, glowing mask, velocity + gravity arrows,
                                prediction ghost path + apex band / landing target with floor tape / finish line + ghost ball,
                                one colour per tracked object
  panels.tsx                    gravity readout or measurements, found law, memory vs eyes, lock cards, hypothesis races, why,
                                scoreboard
```

## Meta's models (one key for all of them)
Put your Meta Model API key in `backend/.env` (created for you, git-ignored, readable only by you):
```
META_API_KEY=...
MUSE_MODEL=muse-spark-1.3
SAM_MODEL=sam-3.1
```
The engine loads it on start (`physlens/config.py`). The one key lists 9 models, including `sam-3.1`, `muse-spark-1.3`,
`muse-voice-transcribe-1.0` (speech to text) and `muse-image-1.0`.

- **SAM 3.1** (`perception/sam_tracker.py`): uploads the clip once (`POST /v1/files`, purpose `user_data`, under 20 MiB),
  then `POST /v1/responses` with an `input_video` part, parsed with Meta's `meta-sam-parser`. It tries the text prompt
  first ("orange ball"). Small, fast objects often get no text match, so it then seeds a **box prompt** from the motion
  tracker and lets SAM track that object both ways. SAM's masks of a tiny blurred ball are loose, so each centre is
  re-measured sub-pixel inside SAM's region, and frames that can't be measured cleanly are dropped.
  Real classroom clip: SAM masks alone gave g = 9.35 ± 0.48 and missed the bounce law; SAM + sub-pixel centres gave
  9.73 ± 0.21, found the speed-dependent bounce law, and hit both predictions. Each call bills per video frame
  (`video_frames_processed`); a text-then-box run segments the clip twice.
- **Muse Spark** (`agent/agent.py`): writes the explanation from the measurements (about 15 s per call). A glossary in the
  prompt says what each number means, and the verifier rejects any number that isn't in the evidence.
- `PHYSLENS_TRACKER=motion` forces the offline tracker (no API calls). Tests never load `.env`, so they stay offline and free.

## Honest limits of v0
- Fixed camera, motion roughly parallel to the image. The stage experiments are proven on rendered clips with
  known physics, not yet on real footage.
- Objects that stay in one spot (weights on springs) can't be refined by motion, so their centres come from the
  colour blob inside SAM's region. SAM's own mask takes in a bit of the spring and once made the stretch check
  read 215 g instead of 250 g. Real footage needs weights with a distinct colour, or a different refiner.
- Scale comes from the ball's known diameter or a ruler in the shot. If the ball moves nearer the camera than
  the ruler, measurements read a few percent high (perspective).
- The motion tracker needs a still camera and assumes the ball is the main small moving thing.
- Analysis runs after upload rather than live; the stage replays events in sync with the video.
- A handheld camera still needs SAM 3.1 plus camera motion (see GEOMETRY.md in benchmark/).
- Deviations from architecture.md, to move fast: Vite instead of Next.js, and one Python package instead of separate services.
