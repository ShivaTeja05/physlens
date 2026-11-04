# Working on PhysLens together

## Branches
- `main`: the current project, everything (Lab, Scan, 3D simulator, crash measurements, docs). Work from here.
- `simulator`: kept level with `main` for links that point at it; don't start new work on it.
- Tags mark earlier versions: `v1.0-normal` (the Lab alone), `v2.0-simulator` (the first simulator release).
- New work: a branch off `main` named for the change (`scan-extract-objects`, `lens-grid-calibration`), then a pull
  request into `main`. Keep pull requests to one change.

## Before you push
```bash
cd backend && .venv/bin/python -m pytest -q     # all pass (one expected xfail)
cd web && pnpm build                            # type-checks and builds
```

## Commits
One change per commit; the first line says what changed and, for measurements, the result
("Moving barrier into Altima (NHTSA 10072): approach 90.72 km/h vs 90.33"). Explain *why* in the body.

## Rules that keep the project trustworthy
- **Keys** stay in `backend/.env` (git-ignored). Never commit, print or paste them.
- **Data** stays out of git: `datasets/`, `records/`, `media/` are ignored. Commit scripts that fetch or rebuild
  them, and a `SOURCE.md` with the licence.
- **Claims** come with evidence. A number in the README or a demo must trace to a script and its `.json` result.
  Ground truth is only for checking, never an input. If a method was chosen after seeing the answer, say so.
- **Failures** are written down in `benchmark/EVENTS.md`, not deleted.
- **Money**: the Meta spend cap stays on; GPU workers run serverless and scale to zero. Stop anything you start.

## Where to start
| You want to | Start in |
|---|---|
| Improve the 3D simulator page | `web/src/PointWorld.tsx`, `web/src/sim.tsx` |
| Add a Lab experiment | `backend/physlens/labs/`, `experiments.py`, `synth_labs.py` for a test clip |
| Extract objects from photos/videos (next feature) | `perception/sam_image.py`, `scan.py`, `priors.py` |
| Measure a new crash video | `docs/SETUP.md` §7, `benchmark/rear8555.py` as a template |
| Understand the flow | `docs/ARCHITECTURE.md` |
