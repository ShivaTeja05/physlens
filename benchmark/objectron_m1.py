"""M1 on Objectron: measure real objects in metres from ONE phone frame, scored against Google's 3D boxes.

For each clip (middle frame): SAM 3.1 outlines the object by name; MapAnything (RunPod) turns the outline's pixels
into 3D points in metres, with no camera intrinsics given (realistic). The points are measured along the
object's true box axes, and only the sides that face the camera are scored (one photo can't see an object's hidden
depth). Prints a table and writes objectron_m1.json.

  cd backend && .venv/bin/python ../benchmark/objectron_m1.py
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "backend"))
from physlens.config import load_env  # noqa: E402

load_env()
from physlens import budget  # noqa: E402
from physlens.datasets.objectron import load_clip  # noqa: E402
from physlens.geometry.client import reconstruct  # noqa: E402

DATA = HERE.parent / "datasets/objectron"
WORK_WIDTH = 720  # frames are downscaled for SAM and MapAnything (MapAnything works at ~518 px anyway)
# ARKit camera frame (x right, y up, z backward) → OpenCV/MapAnything (x right, y down, z forward).
ARKIT_TO_CV = np.diag([1.0, -1.0, -1.0])


def sam_masks(img: np.ndarray, prompt: str) -> list[np.ndarray]:
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["META_API_KEY"], base_url="https://api.meta.ai/v1")
    ok, png = cv2.imencode(".png", img)
    budget.check(budget.SAM_PER_IMAGE)
    r = client.responses.create(model="sam-3.1", input=[{"type": "message", "role": "user", "content": [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}]}],
        metadata={"mask_encoding": "one_bit"})
    budget.record("sam-3.1", budget.SAM_PER_IMAGE, what=f"objectron image: {prompt}")
    parser = image_segmentation_format().create_parser()
    for _ in parser.push(r.output_text or ""):
        pass
    out = []
    for rec in parser.finish(CompletedOutcome()).result.records:
        if rec.kind != "mask":
            continue
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
        full = np.zeros(img.shape[:2], bool)
        b = rec.bounds
        full[b.top:b.top + raster.shape[0], b.left:b.left + raster.shape[1]] = raster > 0
        out.append(full)
    return out


def sam_masks_cached(img: np.ndarray, prompt: str, cache: Path, name: str = "sam_masks") -> list[np.ndarray]:
    f = cache / f"{name}.npz"
    if f.exists():
        z = np.load(f)
        return [z[k] for k in sorted(z.files)]
    masks = sam_masks(img, prompt)
    np.savez_compressed(f, **{f"m{i:02d}": m for i, m in enumerate(masks)})
    return masks


MODE = "mono"  # mono | focal | frames | motion | motion_robust | fused  (set from the command line)
# frames = several frames with NO device data (plain webcam); motion = several frames + phone focal length + AR poses
# motion_robust = motion's cached 3D, measured with outlier removal (no GPU); fused = 7 views over 6 s of the phone's
# orbit, every view's outline lifted to 3D and merged into one point cloud (fused_robust: plus outlier removal;
# fused_median: each view measured on its own, median per side over the views that see it face-on)
MOTION_STEP = 30  # frames between the views used in "motion" mode (1 s at 30 fps)
TAP = False  # tap: one tap on the object instead of its name; tapname: name finds candidates, the tap picks one;
# box: a rough box dragged around the object, plus a tap in it. tapname_tri: tapname, with the object's distance
# triangulated from the phone's motion and MapAnything's points rescaled to it. *_median: the same on 7 views, median per side
FUSED_OFFSETS = (-90, -60, -30, 0, 30, 60, 90)   # frames around the middle one (±3 s at 30 fps)


def device_knowledge(clip, fids: list[int], R: np.ndarray, work_size: tuple[int, int]):
    """What a phone or glasses would know, from Objectron's AR data: focal length (intrinsics in the portrait
    work image) and camera-to-world poses in metres (OpenCV axes)."""
    ww, hw = work_size
    Ks, poses = [], []
    for fid in fids:
        f = clip.frames[fid]
        scale = ww / f.image_size[1]  # sensor is landscape; its height is the portrait width
        focal = 0.5 * (f.intrinsics[0, 0] + f.intrinsics[1, 1]) * scale
        Ks.append(np.array([[focal, 0, ww / 2], [0, focal, hw / 2], [0, 0, 1]]))
        T = np.eye(4)
        T[:3, :3] = R.T                      # image-aligned OpenCV camera → ARKit camera
        poses.append(f.transform @ T)        # → world (metres)
    return Ks, poses


def reconstruct_cached(img: np.ndarray, cache: Path, clip=None, fid: int = 0, R=None):
    """MapAnything result for this frame (and its neighbours in motion mode), saved so re-scoring never pays
    for the GPU twice. Returns the view for frame `fid`."""
    from physlens.geometry.client import ViewGeometry
    f = cache / f"geometry_{'motion' if MODE in ('motion_robust', 'tap', 'tapname', 'box', 'tapname_tri') else MODE}.npz"
    if f.exists():
        z = np.load(f, allow_pickle=False)
        return ViewGeometry(z["depth"], z["points_cam"], z["points_world"], None, None, z["intrinsics"],
                            z["camera_pose"], float(z["scale_x"]), float(z["scale_y"]))
    if MODE == "mono":
        v = reconstruct([img], timeout_s=900)[0]
    else:
        n = len(clip.frames)
        multi = MODE in ("motion", "frames")
        fids = [fid] + ([max(fid - MOTION_STEP, 0), min(fid + MOTION_STEP, n - 1)] if multi else [])
        imgs = [img] + [cv2.resize(clip.read_frame(i), img.shape[1::-1], interpolation=cv2.INTER_AREA) for i in fids[1:]]
        if MODE == "frames":
            v = reconstruct(imgs, timeout_s=900)[0]
        else:
            Ks, poses = device_knowledge(clip, fids, R, img.shape[1::-1])
            v = reconstruct(imgs, timeout_s=900, intrinsics=Ks, camera_poses=poses if MODE == "motion" else None)[0]
    np.savez_compressed(f, depth=v.depth, points_cam=v.points_cam, points_world=v.points_world,
                        intrinsics=v.intrinsics, camera_pose=v.camera_pose, scale_x=v.scale_x, scale_y=v.scale_y)
    return v


def clean_points(pts: np.ndarray, k: int = 12, n_sigma: float = 2.0) -> np.ndarray:
    """Statistical outlier removal: drop points whose mean distance to their k nearest neighbours is far above the
    typical one (stray background pixels and depth spikes at the outline), then keep the biggest blob."""
    from scipy.spatial import cKDTree
    if len(pts) < k + 2:
        return pts
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    keep = mean_d < np.median(mean_d) + n_sigma * 1.4826 * np.median(np.abs(mean_d - np.median(mean_d)))
    pts = pts[keep]
    # Largest connected blob at a few times the typical spacing: a background patch is a separate blob.
    radius = 4 * float(np.median(mean_d))
    pairs = cKDTree(pts).query_pairs(radius, output_type="ndarray")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(pts)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n)) if len(pairs) else None
    if g is None:
        return pts
    _, labels = connected_components(g, directed=False)
    return pts[labels == np.bincount(labels).argmax()]


def object_points(view, mask: np.ndarray, world: bool = False) -> np.ndarray:
    """3D points of an outline. Depth range from the eroded interior (reliable), then every outline pixel inside
    that range (edges included, background rejected)."""
    interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    ys, xs = np.nonzero(interior if interior.sum() > 100 else mask)
    z_in = np.array([view.point_at(x, y)[2] for x, y in zip(xs[::3], ys[::3])])
    z_med = float(np.median(z_in))
    band = max(4 * float(np.median(np.abs(z_in - z_med))), 0.15 * z_med)
    ys, xs = np.nonzero(mask)
    cam = np.array([view.point_at(x, y) for x, y in zip(xs[::2], ys[::2])])
    ok = np.abs(cam[:, 2] - z_med) < band
    if not world:
        return cam[ok]
    return np.array([view.point_at(x, y, world=True) for x, y in zip(xs[::2], ys[::2])])[ok]


def tap_point(frame, size) -> tuple[int, int]:
    """Where a person would tap: the middle of the object (the annotated box centre, projected into the picture)."""
    w, h = size
    c = frame.keypoints_2d[0][0]
    return int(np.clip(round(c[0] * w), 0, w - 1)), int(np.clip(round(c[1] * h), 0, h - 1))


def object_mask(img: np.ndarray, frame, category: str, cache: Path) -> tuple[np.ndarray | None, float]:
    """The object's outline. Name mode: SAM finds it by its name, best match to the annotated box. Tap mode: SAM is
    given one tap on the object, and the largest mask under the tap is used (no annotation involved)."""
    size = img.shape[1::-1]
    hull = box_hull_mask(frame.keypoints_2d[0], size)
    iou = lambda m: float((m & hull).sum() / max((m | hull).sum(), 1))
    if MODE.startswith("tapname"):
        # SAM finds candidates by name; the tap only says which one (instead of the annotated box choosing).
        masks = sam_masks_cached(img, category, cache)
        if not masks:
            return None, 0.0
        x, y = tap_point(frame, size)
        under = [m for m in masks if m[y, x]]
        if not under:   # tap fell just outside every outline: take the outline nearest to it
            d = [np.hypot(*(np.argwhere(m).mean(axis=0) - (y, x))) if m.any() else np.inf for m in masks]
            under = [masks[int(np.argmin(d))]]
        m = max(under, key=lambda m: int(m.sum()))
        return m, iou(m)
    if MODE.startswith("box"):
        # The person drags a rough box around the object (the annotated corners' bounding box, padded 5%).
        from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt
        w, h = size
        k = frame.keypoints_2d[0][1:]
        xs, ys = np.clip(k[:, 0] * w, 0, w - 1), np.clip(k[:, 1] * h, 0, h - 1)
        px, py = 0.05 * (xs.max() - xs.min()), 0.05 * (ys.max() - ys.min())
        box = PromptBox(left=int(max(xs.min() - px, 0)), top=int(max(ys.min() - py, 0)),
                        right=int(min(xs.max() + px, w - 1)) + 1, bottom=int(min(ys.max() + py, h - 1)) + 1)
        # A box alone is ambiguous (SAM outlined the table under a cup), so the tap in its middle goes with it,
        # as in the video tracker's seed prompt.
        x, y = tap_point(frame, size)
        prompt = build_object_prompt(size=size, objects=[PromptObject(id=1, prompts=[
            FramePrompt(box=box, points=[PromptPoint(x, y)])])])
        masks = sam_masks_cached(img, prompt, cache, "sam_masks_boxtap")
        if not masks:
            return None, 0.0
        # The mask that best fills the box the person drew (not the biggest: that's often the table inside it).
        rect = np.zeros(img.shape[:2], bool)
        rect[box.top:box.bottom, box.left:box.right] = True
        m = max(masks, key=lambda m: (m & rect).sum() / max((m | rect).sum(), 1))
        return m, iou(m)
    if TAP:
        from meta_sam_parser import FramePrompt, PromptObject, PromptPoint, build_object_prompt
        x, y = tap_point(frame, size)
        prompt = build_object_prompt(size=size, objects=[PromptObject(id=1, prompts=[FramePrompt(points=[PromptPoint(x, y)])])])
        masks = sam_masks_cached(img, prompt, cache, "sam_masks_tap")
        if not masks:
            return None, 0.0
        under = [m for m in masks if m[y, x]] or masks
        m = max(under, key=lambda m: int(m.sum()))
        return m, iou(m)
    masks = sam_masks_cached(img, category, cache)
    if not masks:
        return None, 0.0
    m, best = pick_mask(masks, frame, size)
    return m, best


def pick_mask(masks: list[np.ndarray], frame, size) -> tuple[np.ndarray, float]:
    hull = box_hull_mask(frame.keypoints_2d[0], size)
    iou = [float((m & hull).sum() / max((m | hull).sum(), 1)) for m in masks]
    return masks[int(np.argmax(iou))], max(iou)


def fused_points(clip, fid: int, work: np.ndarray, R: np.ndarray, cache: Path, category: str):
    """Every view's outline lifted to 3D in the AR world frame and brought into the middle view's camera frame."""
    from physlens.geometry.client import ViewGeometry
    n = len(clip.frames)
    fids = sorted({min(max(fid + o, 0), n - 1) for o in FUSED_OFFSETS}, key=lambda i: (i != fid, i))
    size = work.shape[1::-1]
    imgs = [work if i == fid else cv2.resize(clip.read_frame(i), size, interpolation=cv2.INTER_AREA) for i in fids]
    f = cache / "geometry_fused.npz"
    if f.exists():
        z = np.load(f)
        views = [ViewGeometry(z[f"depth{j}"], z[f"cam{j}"], z[f"world{j}"], None, None, z[f"K{j}"], z[f"pose{j}"],
                              float(z[f"sx{j}"]), float(z[f"sy{j}"])) for j in range(len(fids))]
    else:
        Ks, poses = device_knowledge(clip, fids, R, size)
        views = reconstruct(imgs, timeout_s=900, intrinsics=Ks, camera_poses=poses)
        np.savez_compressed(f, **{k: v for j, vw in enumerate(views) for k, v in {
            f"depth{j}": vw.depth, f"cam{j}": vw.points_cam, f"world{j}": vw.points_world, f"K{j}": vw.intrinsics,
            f"pose{j}": vw.camera_pose, f"sx{j}": vw.scale_x, f"sy{j}": vw.scale_y}.items()})
    world, centres = [], []
    for j, (i, img, vw) in enumerate(zip(fids, imgs, views)):
        sub = cache / f"view_{i}"
        sub.mkdir(exist_ok=True)
        mask, _ = object_mask(img, clip.frames[i], category, sub if i != fid else cache)
        if mask is None:
            continue
        world.append(object_points(vw, mask, world=True))
        centres.append(vw.camera_pose[:3, 3])
    T = np.linalg.inv(views[0].camera_pose)     # world → middle view's camera
    to_ref = lambda P: (T[:3, :3] @ np.asarray(P).T).T + T[:3, 3]
    per_view = [(to_ref(P), to_ref([c])[0]) for P, c in zip(world, centres)]
    return to_ref(np.concatenate(world)), len(world), per_view


TRI_OFFSETS = (-60, -30, 30, 60)   # frames whose views are matched against the middle one (up to 2 s of motion)


def triangulated_depth(clip, fid: int, work: np.ndarray, mask: np.ndarray, R: np.ndarray) -> tuple[float | None, int]:
    """The object's distance from the phone's own motion: SIFT features on the object in the middle frame are matched
    in neighbouring frames and triangulated with the AR camera poses (metres) and focal length. Returns the median
    depth in the middle camera and how many points agreed."""
    size = work.shape[1::-1]
    n = len(clip.frames)
    others = [min(max(fid + o, 0), n - 1) for o in TRI_OFFSETS]
    Ks, poses = device_knowledge(clip, [fid] + others, R, size)
    sift = cv2.SIFT_create(4000)
    g0 = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    obj = cv2.dilate(mask.astype(np.uint8), np.ones((9, 9), np.uint8))
    kp0, d0 = sift.detectAndCompute(g0, obj)
    if d0 is None or len(kp0) < 8:
        return None, 0
    P0 = Ks[0] @ np.linalg.inv(poses[0])[:3]
    w2c0 = np.linalg.inv(poses[0])
    depths = []
    for j, i in enumerate(others, start=1):
        img = cv2.resize(clip.read_frame(i), size, interpolation=cv2.INTER_AREA)
        kp1, d1 = sift.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
        if d1 is None:
            continue
        pairs = cv2.BFMatcher().knnMatch(d0, d1, k=2)
        good = [a for a, b in (p for p in pairs if len(p) == 2) if a.distance < 0.75 * b.distance]
        if len(good) < 8:
            continue
        x0 = np.float32([kp0[m.queryIdx].pt for m in good]).T
        x1 = np.float32([kp1[m.trainIdx].pt for m in good]).T
        P1 = Ks[j] @ np.linalg.inv(poses[j])[:3]
        X = cv2.triangulatePoints(P0, P1, x0, x1)
        X = (X[:3] / X[3]).T
        cam0 = (w2c0[:3, :3] @ X.T).T + w2c0[:3, 3]
        # Keep points in front of both cameras that reproject within 2 px in both views.
        ok = cam0[:, 2] > 0.05
        for P, x in ((P0, x0), (P1, x1)):
            pr = P @ np.c_[X, np.ones(len(X))].T
            ok &= (pr[2] > 0) & (np.linalg.norm(pr[:2] / pr[2] - x, axis=0) < 2.0)
        depths.extend(cam0[ok, 2].tolist())
    if len(depths) < 10:
        return None, len(depths)
    return float(np.median(depths)), len(depths)


def box_hull_mask(k2: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    pts = np.stack([k2[1:, 0] * w, k2[1:, 1] * h], axis=1).astype(np.int32)
    m = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m, cv2.convexHull(pts), 1)
    return m > 0


def _rot_z(deg: int) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def camera_alignment(k3: np.ndarray, k2: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Objectron's camera frame follows the phone sensor (landscape) while the video is stored portrait, so the
    two differ by a quarter turn. Pick the rotation that makes the annotated 3D corners project onto the annotated
    2D corners in the portrait frame."""
    w, h = size
    uv = np.stack([k2[1:, 0] * w, k2[1:, 1] * h], axis=1)
    best, best_err = None, np.inf
    for deg in (0, 90, 180, 270):
        R = _rot_z(deg) @ ARKIT_TO_CV
        c = (R @ k3[1:].T).T
        if np.any(c[:, 2] <= 0):
            continue
        p = c[:, :2] / c[:, 2:3]
        # uv ≈ f·p + centre with one shared focal length: least squares in both axes together.
        A = np.zeros((16, 3)); A[0::2, 0], A[0::2, 1] = p[:, 0], 1; A[1::2, 0], A[1::2, 2] = p[:, 1], 1
        sol, *_ = np.linalg.lstsq(A, uv.reshape(-1), rcond=None)
        err = np.linalg.norm(A @ sol - uv.reshape(-1))
        if sol[0] > 0 and err < best_err:
            best, best_err = R, err
    if best is None:
        raise ValueError("couldn't align the annotation camera with the video")
    return best


def box_axes_cv(k3: np.ndarray, R: np.ndarray = ARKIT_TO_CV) -> list[tuple[np.ndarray, float]]:
    """The box's three edge directions in the image-aligned OpenCV camera frame, with their true lengths (metres)."""
    c = (R @ k3.T).T
    out = []
    for a, b in [(1, 5), (1, 3), (1, 2)]:
        v = c[b] - c[a]
        out.append((v / np.linalg.norm(v), float(np.linalg.norm(v))))
    return out


def measure_clip(folder: Path) -> dict:
    clip = load_clip(folder)
    category = folder.name.rsplit("_batch-", 1)[0].replace("_", " ")   # "cereal_box_batch-1_2" → "cereal box"
    fid = len(clip.frames) // 2
    frame = clip.frames[fid]
    full = clip.read_frame(fid)
    h0, w0 = full.shape[:2]
    work = cv2.resize(full, (WORK_WIDTH, round(h0 * WORK_WIDTH / w0)), interpolation=cv2.INTER_AREA)
    hw, ww = work.shape[:2]

    truth_hull = box_hull_mask(frame.keypoints_2d[0], (ww, hw))
    R = camera_alignment(frame.keypoints_3d[0], frame.keypoints_2d[0], (ww, hw))
    cache = HERE / "cache" / folder.name
    cache.mkdir(parents=True, exist_ok=True)
    mask, best_iou = object_mask(work, frame, category, cache)
    if mask is None:
        return {"clip": folder.name, "category": category, "error": "SAM found no match"}
    iou = [best_iou]

    n_views = 1
    per_view = None
    if MODE.startswith("fused") or MODE.endswith("_median"):
        pts, n_views, per_view = fused_points(clip, fid, work, R, cache, category)
        if MODE == "fused_robust":
            pts = clean_points(pts)
    else:
        view = reconstruct_cached(work, cache, clip, fid, R)
        pts = object_points(view, mask)
        if MODE.endswith("_tri"):
            # Rescale MapAnything's points so the object sits at the distance the phone's motion triangulates.
            z_tri, n_tri = triangulated_depth(clip, fid, work, mask, R)
            if z_tri:
                pts = pts * (z_tri / float(np.median(pts[:, 2])))
        if MODE == "motion_robust":
            pts = clean_points(pts)

    view_dir = pts.mean(axis=0) / np.linalg.norm(pts.mean(axis=0))
    sides = []
    for axis, true_len in box_axes_cv(frame.keypoints_3d[0], R):
        facing = abs(float(axis @ view_dir)) < 0.7      # side mostly across the view, not along it
        if MODE.endswith("_median"):
            # Each view measures the object on its own (no cross-view smear); a side counts only from views that
            # see it face-on; the median over those views is the answer.
            got = []
            for P, cam in per_view:
                d = P.mean(axis=0) - cam
                if abs(float(axis @ (d / np.linalg.norm(d)))) < 0.7:
                    pr = P @ axis
                    got.append(float(np.percentile(pr, 99.5) - np.percentile(pr, 0.5)))
            proj = pts @ axis
            measured = float(np.median(got)) if got else float(np.percentile(proj, 99.5) - np.percentile(proj, 0.5))
        else:
            proj = pts @ axis
            measured = float(np.percentile(proj, 99.5) - np.percentile(proj, 0.5))
        sides.append({"true_m": round(true_len, 4), "measured_m": round(measured, 4), "scored": facing,
                      "ratio": round(measured / true_len, 3)})
    true_center = R @ frame.keypoints_3d[0][0]
    all_ratio = [s["ratio"] for s in sides]
    scored = [s for s in sides if s["scored"]]
    return {"clip": folder.name, "category": category, "frame": fid, "sam_iou": round(max(iou), 3),
            "sides": sides, "median_ratio": round(float(np.median([s["ratio"] for s in scored])), 3) if scored else None,
            "true_distance_m": round(float(true_center[2]), 3), "measured_distance_m": round(float(np.median(pts[:, 2])), 3),
            "views": n_views, "all_sides_ratio": all_ratio}


def main() -> None:
    rows = []
    for folder in sorted(DATA.glob("*_batch-*")):
        try:
            row = measure_clip(folder)
        except Exception as exc:
            row = {"clip": folder.name, "error": str(exc)[:200]}
        rows.append(row)
        if "error" in row:
            print(f"{row['clip']:22} ERROR {row['error']}", flush=True)
            continue
        sides = "  ".join(f"{s['measured_m']:.3f}/{s['true_m']:.3f}{'' if s['scored'] else '*'}" for s in row["sides"])
        print(f"{row['clip']:22} SAM IoU {row['sam_iou']:.2f} | measured/true m: {sides} | "
              f"scale ×{row['median_ratio']} | distance {row['measured_distance_m']:.2f}/{row['true_distance_m']:.2f} m",
              flush=True)
    ok = [r for r in rows if r.get("median_ratio")]
    ratios = np.array([r["median_ratio"] for r in ok])
    errs = np.abs(np.array([s["ratio"] for r in ok for s in r["sides"] if s["scored"]]) - 1)
    summary = {"clips": len(rows), "measured": len(ok),
               "median_abs_size_error_pct": round(float(np.median(errs) * 100), 1) if len(errs) else None,
               "sides_within_10pct": f"{int((errs <= 0.10).sum())}/{len(errs)}",
               "sides_within_20pct": f"{int((errs <= 0.20).sum())}/{len(errs)}",
               "median_scale": round(float(np.median(ratios)), 3) if len(ratios) else None}
    every = np.abs(np.array([x for r in ok for x in r["all_sides_ratio"]]) - 1)
    summary["all_sides_median_error_pct"] = round(float(np.median(every) * 100), 1) if len(every) else None
    print("\n(* = side mostly along the view direction: not scored)")
    print("SUMMARY", json.dumps(summary))
    print("spend:", budget.summary())
    summary["mode"] = MODE
    (HERE / f"objectron_m1_{MODE}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        MODE = sys.argv[1]
    assert MODE in ("mono", "focal", "frames", "motion", "motion_robust", "fused", "fused_robust", "fused_median", "tap",
                    "tap_median", "tapname", "tapname_median", "box", "box_median", "tapname_tri"), MODE
    TAP = MODE.startswith("tap")
    print("mode:", MODE)
    main()
