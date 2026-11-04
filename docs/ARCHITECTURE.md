# How PhysLens works

PhysLens has three ways in (a **video** for the 3D simulator, a **clip** for the Lab, a **photo** for Scan) and one
rule throughout: **AI sees, deterministic code measures, official data only checks.** Meta's models find and name
things; the numbers come from geometry and physics we can inspect; ground truth (speed traps, accelerometers) is
never an input, only the thing we compare with at the end.

The diagrams below are Mermaid (GitHub draws them). An illustrated version: [architecture.html](architecture.html).

## 1. Where everything runs
```mermaid
flowchart LR
  B[Browser] -->|:5173| W[Web app<br/>Vite + React + three.js<br/>Lab · Scan · 3D world · sim.html]
  W -->|/api, /media proxied| E[PhysLens engine<br/>FastAPI :8000<br/>holds the keys and the $ cap]
  E -->|SAM 3.1 · Muse Spark · Muse Voice| META[Meta Model API<br/>api.meta.ai]
  E -->|MapAnything · MoGe-3| GPU[RunPod serverless GPU<br/>scales to zero]
  GH[GitHub repo<br/>physlens-gpu-worker] -->|builds the image| GPU
  E --> DB[(records/physlens.db<br/>SQLite black box)]
  E --> MED[(media/pointworld<br/>3D worlds)]
  MED -->|web_copy.py| SH[Static shared copy<br/>sim.html + data<br/>no engine, no GPU]
```
- Keys live only in `backend/.env` (git-ignored). The browser never sees them.
- Every Meta call is priced and logged (`backend/.spend.jsonl`); calls stop at `PHYSLENS_BUDGET_USD` (default $50).
- The GPU worker runs only when called (min 0 workers, 60 s idle), so it costs nothing while idle.

## 2. From video to an explained 3D world (end to end)
```mermaid
flowchart TD
  IN[Video: fixed or moving camera, 30–1,000 fps<br/>one or several cameras] --> SEE
  subgraph SEE [SEE · Meta AI]
    S1[SAM 3.1 video: every object's mask in every keyframe]
    S2[Muse Spark: what each object is + its published size, labelled 'predicted']
  end
  SEE --> G3
  subgraph G3 [3D · GPU worker + engine]
    G1[MapAnything: camera pose, metric depth per frame]
    G2[MoGe-3: sharp depth and lens per frame]
    G4[Engine: one camera pose for a fixed camera,<br/>floor plane, edge filtering]
  end
  G3 --> MEAS
  subgraph MEAS [MEASURE · deterministic]
    M1[Object body: extract → combine ⟲ → analyse → predict<br/>objsim.py]
    M2[Hybrid motion: image features every frame,<br/>made metric on a plane from the scene<br/>hybrid.py]
  end
  MEAS --> UND
  subgraph UND [UNDERSTAND]
    U1[Events: contact, crossing, crush, rebound, height check<br/>events3d.py]
    U2[Muse Spark explains; every number it says<br/>must be one we measured]
    U3[Check vs official data:<br/>speed trap, accelerometers]
  end
  UND --> OUT[SHOW: point world file → 3D simulator,<br/>timeline, crush chart, coverage, ask Muse, shared link]
```

## 3. The simulator core: one solid body, moving over time
Two lanes from the same video. Keyframes give **shape**; every frame gives **motion**.
```mermaid
flowchart LR
  subgraph KF [Keyframes · shape]
    A1[extract<br/>each frame's object points,<br/>cleaned of background] --> A2[combine<br/>one body in its own coordinates<br/>+ a pose per frame]
    A2 -->|refine until no pose<br/>moves more than 1 cm| A2
    A2 --> A3[analyse<br/>which sides no frame ever saw]
    A3 --> A4[predict<br/>mirror the seen side, violet,<br/>labelled predicted]
  end
  subgraph EF [Every frame · motion]
    B1[features on a rigid part,<br/>tracked forward-backward] --> B2[each feature's ray meets<br/>a plane from the scene,<br/>at its own height]
    B2 --> B3[rigid move per frame<br/>slide + turn]
    B3 --> B4[lens: straight lines,<br/>only if they agree]
    B4 --> B5[scale: one known length<br/>wheelbase, width, targets]
  end
  A4 --> SIM[Simulator plays the body on that motion]
  B5 --> SIM
```
Why two lanes: per-frame AI depth jitters by about 10 cm, more than a crash moves between 1 ms frames. Image
tracking lands within a fraction of a pixel, so motion comes from the image and only the plane, lens and one
length come from 3D. (Lessons behind each choice: [benchmark/EVENTS.md](../benchmark/EVENTS.md).)

## 4. How a new crash video is measured (the method, as a decision flow)
```mermaid
flowchart TD
  V[New video] --> C{Clock burned in?}
  C -->|yes| C1[one frame per clock reading<br/>NHTSA transcodes may repeat frames]
  C -->|no| C2[use the file's frame rate]
  C1 --> D[MoGe-3 depth on a few frames<br/>SAM 3.1 outlines]
  C2 --> D
  D --> F[floor plane from the floor's points]
  F --> R{Known length in view?}
  R -->|targets on the object| R1[photo-target spacing<br/>e.g. 450 mm on a barrier face]
  R -->|car| R2[published width or wheelbase<br/>not length if a spare tyre / mirrors interfere]
  R1 --> L
  R2 --> L
  L{Straight floor lines agree?} -->|middle half within 0.1| L1[lens from lines]
  L -->|no| L2[no lens correction]
  L1 --> T[track rigid parts only:<br/>away from the crush, no glass,<br/>one part seeded once and carried]
  L2 --> T
  T --> P{Physics check<br/>without the answer}
  P -->|straight approach, no turn before impact,<br/>bodies the same size| OK[compare with the official numbers]
  P -->|fails| T
```

## 5. Lab: a clip becomes locked predictions
```mermaid
sequenceDiagram
  participant U as User
  participant W as Web app
  participant E as Engine
  participant S as SAM 3.1
  participant M as Muse Spark
  U->>W: upload clip / pick a sample
  W->>E: POST /api/runs
  E->>S: track the object (text, then box prompt)
  S-->>E: masks per frame
  E->>E: sub-pixel centres → calibrate (ball size or ruler)
  E->>E: discover the law (12 force combinations, BIC)
  E->>E: lock predictions from early frames (SHA-256 + time)
  E->>E: check against the later frames
  E->>M: explain from the measurements
  M-->>E: text, and the verifier rejects any number not measured
  E-->>W: event stream (WebSocket /api/runs/{id}/stream)
  W-->>U: stage replay: trail, arrows, locked calls, HIT/MISS
```

## 6. Scan: a photo becomes measurements
```mermaid
flowchart LR
  P[Photo] --> L[Look<br/>Muse Spark: scene, parts,<br/>how they act on each other]
  L --> F[Find<br/>SAM 3.1 outlines every part]
  F --> ME[Measure<br/>counts, rows, lean, contact,<br/>lit share, metres from one size]
  ME --> X[Explain<br/>evidence-or-reject;<br/>words vs eyes checks]
  X --> D[Draw<br/>outlines, call-outs, arrows]
```

## 7. Code map
| Piece | Files |
|---|---|
| API and orchestration | `backend/physlens/api.py`, `pipeline.py`, `events.py` |
| Meta models | `perception/sam_tracker.py` (video), `perception/sam_image.py` (photo), `agent/agent.py`, `agent/verifier.py`, `ask.py`, `voice.py`, `priors.py` |
| 3D from the GPU | `geometry/client.py` (MapAnything, MoGe-3 on RunPod), `gpu_worker/` |
| Moving / fixed camera geometry | `moving_camera.py`, `sharp.py`, `objfuse.py`, `vehicle3d.py` |
| Point worlds and bodies | `pointworld.py` (file format `physlens-pointworld/1`), `objsim.py` (extract → combine → analyse), `events3d.py` |
| Hybrid motion | `hybrid.py`: `track_part`, `track_rigid`, `lens_from_lines`, `fit_distortion`, `plane_point` |
| Lab physics | `physics/` (calibration, bounces, law discovery), `labs/` (cup, cans, springs, crash), `ledger.py` |
| Records, replay, sims | `recorder.py`, `replay.py`, `scene.py`, `sim_mujoco.py` |
| Scan | `scan.py` |
| Web | `web/src/App.tsx` (Lab/Scan), `PointWorld.tsx` (3D simulator), `sim.tsx` + `web/sim.html` (standalone) |
| Real test cases | `benchmark/*.py`, one per event; results in `benchmark/*.json`; write-up in `benchmark/EVENTS.md` |

## 8. The point world file (`physlens-pointworld/1`)
A JSON header plus a binary of int16 positions, RGB colours and a group tag per point:
- tag 255: the static scene; per-moment tags: an object at each keyframe
- 254: the object's combined body (real colours); 253: the same body coloured by how many frames saw each spot
- 252: predicted points (the side no camera saw), drawn violet
- `notes`: events, rigid body poses, structures, coverage, priors, truth table, crush curve, explanation, cameras
