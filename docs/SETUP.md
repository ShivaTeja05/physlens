# Setup

## Requirements
- Python 3.11 and [uv](https://docs.astral.sh/uv/) (or plain `venv` + `pip`)
- Node 20+ and [pnpm](https://pnpm.io/)
- ffmpeg is bundled through `imageio-ffmpeg`; nothing else to install
- Optional: a Meta Model API key; a RunPod account for the GPU worker

## 1. Engine
```bash
cd backend
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env              # then fill in what you have (see below)
.venv/bin/python -m physlens.cli samples      # renders the synthetic test clips into ../samples
.venv/bin/uvicorn physlens.api:app --port 8000 --reload
```
Health check: http://localhost:8000/api/health. Spend so far: http://localhost:8000/api/budget.

## 2. Web app
```bash
cd web
pnpm install
pnpm dev            # http://localhost:5173 (Lab, Scan), http://localhost:5173/sim.html (3D simulator)
pnpm build          # production build into web/dist (index.html + sim.html)
```
The dev server proxies `/api` and `/media` to the engine on :8000.

## 3. Keys (`backend/.env`, never committed)
| Variable | What for |
|---|---|
| `META_API_KEY` | SAM 3.1, Muse Spark, Muse Voice (one key for all) |
| `MUSE_MODEL`, `SAM_MODEL`, `TRANSCRIBE_MODEL` | model names (defaults in `.env.example`) |
| `PHYSLENS_BUDGET_USD` | spending cap for Meta calls (default 50) |
| `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID` | the GPU worker for MapAnything and MoGe-3 |
| `PHYSLENS_TRACKER=motion` | force the offline tracker |
Each teammate uses their own key. Don't paste keys into code, issues, chat or commits.

## 4. GPU worker (only for building 3D worlds)
The worker image is built by RunPod from its own GitHub repo; the code is mirrored in `gpu_worker/`. Follow
[gpu_worker/README.md](../gpu_worker/README.md): serverless endpoint, 24 GB GPU, min 0 / max 1 workers, 60 s idle,
30 GB container disk. Put the endpoint ID in `.env`. With min 0 workers it costs nothing while idle. Use serverless
only, and make sure no worker is left running after you finish.

## 5. Datasets (not in git)
```bash
python3 scripts/fetch_datasets.py --list          # every file, its source and size
python3 scripts/fetch_datasets.py --yes 9070 8555 # download chosen sets into datasets/
```
Sources: NHTSA's Vehicle Crash Test Database (public; credit NHTSA; don't edit or re-host their videos), NASA
(public domain), Wikimedia Commons (CC BY; credit the author). Each `datasets/*/SOURCE.md` records what was
downloaded, from where, and the licence.

## 6. Tests
```bash
cd backend && .venv/bin/python -m pytest -q       # about 3–4 min, offline, no keys, free
```
Expect one `xfailed`: the 12 fps test documents a known limit. Real-footage tests skip if their clips aren't present.

## 7. Adding a new real test case (the checklist)
1. **Pick footage with official numbers** (speed trap, accelerometers, published dimensions) and a public licence.
   Add it to `scripts/fetch_datasets.py` and write a `SOURCE.md`.
2. **Write `benchmark/<name>.py`** with a docstring saying: the event, the camera, and how each of floor, scale,
   lens and tracking is done. Look at `benchmark/rear8555.py` (two bodies) or `benchmark/omdb10072.py` (one turning
   body) as templates.
3. **Decide the method before looking at the answer.** Scale from a published length or marked targets; lens from
   straight lines only if they agree; geometry chosen by physics (no turn before impact, constant approach speed).
   Use the official numbers only in the final comparison.
4. **Record honestly** in `benchmark/EVENTS.md`, including what's off and why, and commit the `.json` results.
