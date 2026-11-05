# Demo guide

Three demos, from no setup to the full pipeline. Start the engine and the web app first ([README](../README.md),
"Try it in 5 minutes"). Everything marked **offline** needs no keys and costs nothing.

## What each part needs
| Part | Keys | Cost per use | Data |
|---|---|---|---|
| Lab on the shipped samples | none (offline motion tracker) | free | `samples/` (in the repo) |
| Lab with SAM 3.1 tracking + Muse Spark explanations | `META_API_KEY` | SAM bills per video frame; priced and logged per call | `samples/` or your clip |
| Scan a photo | `META_API_KEY` | $0.04–0.07 per new photo (cached after) | `samples/scan/` or your photo |
| Voice and "Ask" | `META_API_KEY` | priced and logged per call | — |
| 3D simulator (viewing worlds) | none | free | `media/pointworld/` (generate it, or use the shared link) |
| Building 3D worlds, measuring crash tests | `META_API_KEY` + RunPod | RunPod bills GPU-seconds (a few MoGe-3 calls per test) + SAM | `scripts/fetch_datasets.py` |

The spend cap stops Meta calls at `PHYSLENS_BUDGET_USD` (default $50); check it at `GET /api/budget`.

## A. The 5-minute judges' demo
1. **The hook (30 s).** Open http://localhost:5173/app.html (the Lab; the landing page is at http://localhost:5173). Pick sample **Slow-mo 2×**, press play. PhysLens measures
   g ≈ 2.47 m/s² and says the video runs at half speed. *Point:* a language model "remembers" 9.81; PhysLens measures
   what's in the video. (**Memory vs eyes** panel.)
2. **Predictions locked before they happen (1 min).** Pick **Real classroom drop**. Watch the trail, the law it finds
   (gravity only: 9.73 ± 0.21 with SAM 3.1, 9.86 ± 0.19 with the offline tracker), and the next-bounce prediction
   stamped **locked** with a hash, then **HIT**.
3. **The 3D simulator (2 min).** Open http://localhost:5173/sim.html (or the shared simulator link).
   - World **NHTSA crash test 9070**: switch **Simulation** / **Raw frames**; scrub the timeline through impact;
     open the crush chart (video vs accelerometers within 1.5 cm); the **Ground truth** table.
   - World **NHTSA side pole test 15756**: the car as one body moving at 74.6° into the pole, 31.29 km/h vs 31.09.
   - World **Box truck hits the 11foot8 bridge**: the impact event at the beam, clearance 3.63 m, "too tall".
   - **The body, and what the camera never saw**: violet points are predicted (mirror symmetry), not measured.
   - **Explain what happened** / **Ask**: Muse Spark answers from the world's measurements only.
4. **The proof table (1 min).** Show the results table in the README: four crash tests, impact speeds within
   0.0–0.6% of NHTSA's speed traps, and the failures written down.
5. **Close (30 s).** Scan (needs the Meta key): switch **Lab → Scan**, pick the Moon sample: 46.5% lit, and Muse
   Spark said "gibbous" (flagged by "Words vs eyes").

## B. Lab, step by step
- Samples: **Real drop**, **Slow-mo 2×**, **Mystery clip**, **Real classroom drop**, **Catch the cup**,
  **Mystery Cans**, **Weigh it by watching**, **Real crash test (frontal)**.
- Your own clip: next to *Upload clip*, choose the experiment and its tape-measure values; give the scale as the ball's
  diameter (*Ball ⌀ cm*) or a ruler (*Ruler px/m*). **Auto-detect** asks Muse Spark which experiment it is.
- **3D world** (toggle) re-simulates the record with Rapier; **Fit MuJoCo** fits a MuJoCo model to it and downloads MJCF.
- **Records** reopens any past run; **what-if** changes gravity, bounciness, speed or drag and redraws the path.
- **Voice**: hold the mic button (or space) and ask; answers come only from the run's measurements.

## C. Measuring a crash test yourself (the full pipeline)
```bash
python3 scripts/fetch_datasets.py --list                 # what can be downloaded, with sizes
python3 scripts/fetch_datasets.py --yes 8555             # one NHTSA test (video + accelerometers)
PYTHONPATH=backend backend/.venv/bin/python benchmark/prepare_cache.py 8555   # MoGe-3 depth + SAM outlines (GPU + Meta key)
PYTHONPATH=backend backend/.venv/bin/python benchmark/rear8555.py
```
`prepare_cache.py` writes the depth and outline caches into `records/real_cache/` (it calls the GPU worker and
SAM 3.1, so it needs both keys; see [SETUP.md](SETUP.md)). The same goes for `7371` (`yaris7371.py`) and `10072`
(`omdb10072.py`). Each benchmark script's docstring says what it needs and how to run it; results are written to
`benchmark/<name>.json` and summarised in [benchmark/EVENTS.md](../benchmark/EVENTS.md).

To rebuild the 3D worlds for the simulator:
```bash
PYTHONPATH=backend backend/.venv/bin/python benchmark/export_pointworlds.py --sharp   # truck, rally
PYTHONPATH=backend backend/.venv/bin/python benchmark/crash3d.py --export             # crash-9070
PYTHONPATH=backend backend/.venv/bin/python benchmark/pole_world.py                   # pole-15756
PYTHONPATH=backend backend/.venv/bin/python benchmark/world_events.py
PYTHONPATH=backend backend/.venv/bin/python benchmark/simulate_worlds.py
PYTHONPATH=backend backend/.venv/bin/python benchmark/explain_worlds.py
```
To make a shareable copy that opens without the engine: `cd web && pnpm build`, then
`PYTHONPATH=backend backend/.venv/bin/python benchmark/web_copy.py <out_dir>`.

## Troubleshooting
| Symptom | Fix |
|---|---|
| Lab runs but says "template explanation" | No `META_API_KEY` in `backend/.env` (fine offline) |
| 3D simulator shows no worlds | `media/pointworld/` is empty: build worlds (C) or use the shared link |
| SAM finds nothing for an object | Try a plainer prompt ("car", "wall"); some objects only work by colour or a box prompt |
| GPU call takes ~1–8 minutes the first time | The RunPod worker is cold-starting; later calls are fast |
| `refused: budget` | The $ cap was reached; raise `PHYSLENS_BUDGET_USD` deliberately |
