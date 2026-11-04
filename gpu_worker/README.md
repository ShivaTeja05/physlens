# PhysLens GPU worker (RunPod Serverless)

Runs Meta's [MapAnything](https://github.com/facebookresearch/map-anything) (Apache-2.0 checkpoint
`facebook/map-anything-apache`, pinned commit) to turn photos or video frames into metric 3D: depth per pixel, camera
pose and scale. PhysLens calls it to measure real things in metres (heights, distances, motion with a moving camera).

Everything is installed when the image is **built**, and the weights are baked in, so a worker only loads the model
when it starts (an install at startup doesn't work on RunPod, which restarts containers after a few minutes).

## Deploy on RunPod (from this GitHub repo)
1. RunPod console → **Settings → Connections → GitHub**: connect your GitHub account and allow this repo.
2. **Serverless → New Endpoint → GitHub Repo**: pick this repo, branch `main`, Dockerfile path `Dockerfile`.
3. GPU: 24 GB cards (RTX 4090 / A5000 / 3090 / L4). Workers: **min 0, max 1**. Idle timeout: 60 s.
   Container disk: **30 GB**. No network volume needed. With min 0 it costs nothing while idle.
4. Wait for the build to finish (the Builds/Releases tab shows logs). PhysLens finds the endpoint by name
   (`physlens-geometry-gh`) through the API, or put its ID in `backend/.env` as `RUNPOD_ENDPOINT_ID=...`.

## API
`POST https://api.runpod.ai/v2/<endpoint>/runsync`
```json
{"input": {"images": ["<base64 png>", "..."], "size": [518, 392]}}
```
Returns one compressed `npz` per view: `depth_z` (m), `conf`, `mask`, `intrinsics`, `camera_pose`, `metric_scale`.

**MoGe-3** (Microsoft, MIT code and weights `Ruicheng/moge-3-vitl`): sharp metric depth per image at its own resolution.
```json
{"input": {"task": "moge", "images": ["<base64 jpg>"], "fov_x": 62.5}}
```
Returns per image: `depth` (m), `mask`, `normal` (int8 ×127), `intrinsics` (pixels). PhysLens aligns it to
MapAnything's depth and lifts it with MapAnything's camera poses, for full-resolution point clouds.
