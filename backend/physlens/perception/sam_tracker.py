"""SAM 3.1 tracker on the Meta Model API.

The clip is uploaded once (`POST /v1/files`, purpose `user_data`), then segmented over
`POST /v1/responses` with model `sam-3.1` and the file as an `input_video` part. The reply is SAM's
special-token text, one line per frame, which Meta's `meta-sam-parser` turns into box and mask records with
object identities that stay stable across frames.

Two ways to say what to track:
  1. A text prompt ("orange ball"). Works for objects that are reasonably large in the frame.
  2. A box prompt on one frame, then SAM tracks that object forwards and backwards. Small, fast objects
     (a 12 px ball) often get no text match, so if the text prompt finds nothing we seed a box from the
     OpenCV motion tracker's most confident detection and let SAM do the tracking.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import Detection, TrackResult

API_BASE = "https://api.meta.ai/v1"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # the documented playground limit; keep clips under it


class SamTracker:
    name = "sam-3.1"

    def __init__(self, color: Optional[str] = None):
        key = os.environ.get("META_API_KEY")
        if not key:
            raise RuntimeError("SAM 3.1 not configured: set META_API_KEY in backend/.env")
        from openai import OpenAI
        self.client = OpenAI(api_key=key, base_url=os.environ.get("META_API_BASE", API_BASE))
        self.model = os.environ.get("SAM_MODEL", "sam-3.1")
        self.color = color
        self.last_mode = ""

    # --- API calls -------------------------------------------------------------------------------------

    def _upload(self, video: Path) -> str:
        if video.stat().st_size > MAX_UPLOAD_BYTES:
            raise RuntimeError(f"{video.name} is over 20 MiB; trim or compress it before segmenting")
        with video.open("rb") as f:
            return self.client.files.create(file=f, purpose="user_data").id

    def _segment(self, file_id: str, prompt_text: str, n_frames: int = 0) -> str:
        from .. import budget
        budget.check(budget.sam_video_estimate(n_frames))
        stream = self.client.responses.create(
            model=self.model,
            input=[{"type": "message", "role": "user", "content": [
                {"type": "input_text", "text": prompt_text},
                {"type": "input_video", "file_id": file_id},
            ]}],
            stream=True,
            metadata={"mask_encoding": "one_bit", "include_confidence": "true"},
        )
        deltas, done_text, usage = [], "", None
        for event in stream:
            if event.type == "response.output_text.delta":
                deltas.append(event.delta)
            elif event.type == "response.output_text.done":
                done_text = event.text or ""
            elif event.type == "response.completed":
                usage = getattr(event.response, "usage", None)
            elif event.type in ("response.failed", "error"):
                raise RuntimeError(f"SAM request failed: {event}")
        budget.record_sam(self.model, usage, what="segment")
        return "".join(deltas) or done_text

    # --- parsing ---------------------------------------------------------------------------------------

    @staticmethod
    def _detections(text: str, n_frames: int) -> list[Optional[Detection]]:
        """One detection per frame for the object SAM followed longest."""
        from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, video_segmentation_format

        parser = video_segmentation_format().create_parser()
        for _ in parser.push(text):
            pass
        records = [r for r in parser.finish(CompletedOutcome()).result.records if r.kind == "mask"]
        if not records:
            return [None] * n_frames
        counts: dict[str, int] = {}
        for r in records:
            counts[r.object_id] = counts.get(r.object_id, 0) + 1
        target = max(counts, key=counts.get)

        frames: list[Optional[Detection]] = [None] * n_frames
        for r in records:
            i = r.frame.frame_index
            if r.object_id != target or not 0 <= i < n_frames:
                continue
            w, h = r.mask.width, r.mask.height
            raster = np.frombuffer(decode_mask_to_raster(r.mask), dtype=np.uint8).reshape(h, w)
            area = int(raster.sum())
            if area == 0:
                continue
            ys, xs = np.nonzero(raster)
            cx, cy = r.bounds.left + xs.mean(), r.bounds.top + ys.mean()
            contours, _ = cv2.findContours(raster.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            outline = []
            if contours:
                c = max(contours, key=cv2.contourArea)
                pts = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
                outline = [(int(x + r.bounds.left), int(y + r.bounds.top)) for x, y in pts[:48]]
            frames[i] = Detection(float(cx), float(cy), float(np.sqrt(area / np.pi)), outline)
        return frames

    # --- seeding ---------------------------------------------------------------------------------------

    def _local_sightings(self, video: Path, prompt: str) -> list[Optional[Detection]]:
        """A quick local look (seconds, free): the motion tracker, or for objects that never leave their spot (a
        weight bobbing on a spring is part of the median background) the colour tracker."""
        from .motion_tracker import MotionBlobTracker
        found = MotionBlobTracker(color=self.color).track(video, prompt).frames
        if not any(d is not None for d in found) and self.color:
            from .opencv_tracker import ColorBlobTracker
            found = ColorBlobTracker(color=self.color).track(video, prompt).frames
        return found

    def _seed_box(self, found: list[Optional[Detection]], width: int, height: int) -> Optional[str]:
        """Box prompt around the local tracker's clearest sighting: typical size, near mid-clip."""
        from meta_sam_parser import FramePrompt, PromptBox, PromptObject, PromptPoint, build_object_prompt

        sightings = [(i, d) for i, d in enumerate(found) if d is not None]
        if not sightings:
            return None
        radii = np.array([d.r_px for _, d in sightings])
        typical = float(np.median(radii))
        mid = len(found) / 2
        i, d = min(sightings, key=lambda s: abs(s[1].r_px - typical) / typical + 0.5 * abs(s[0] - mid) / len(found))
        pad = max(2.5 * d.r_px, 6.0)
        box = PromptBox(left=int(max(d.x_px - pad, 0)), top=int(max(d.y_px - pad, 0)),
                        right=int(min(d.x_px + pad, width - 1)) + 1, bottom=int(min(d.y_px + pad, height - 1)) + 1)
        point = PromptPoint(x=int(round(min(max(d.x_px, 0), width - 1))), y=int(round(min(max(d.y_px, 0), height - 1))))
        return build_object_prompt(size=(width, height), objects=[
            PromptObject(id=1, prompts=[FramePrompt(frame=i, box=box, points=[point])])])

    # --- refinement ------------------------------------------------------------------------------------

    def _refine(self, video: Path, frames: list[Optional[Detection]]) -> tuple[list[Optional[Detection]], int]:
        """SAM says where the object is; the centre is then measured with sub-pixel background-difference
        weighting inside SAM's region. SAM's masks of small, blurred objects include some blur and shadow,
        which moves the centroid by a few pixels; that is enough to bias a gravity fit. Frames where no
        clean blob is found (motion blur at impact, the object in a hand) are dropped, because a loose centre
        there spoils the fit more than a gap does; the bounce logic bridges gaps. If refinement fails on most
        frames, SAM's own centres are kept everywhere. Returns (frames, number refined)."""
        from .motion_tracker import MotionBlobTracker, _read_background

        keep_raw: set[int] = set()
        helper = MotionBlobTracker(color=self.color)
        cap = cv2.VideoCapture(str(video))
        background = _read_background(video, len(frames))
        refined, out = 0, list(frames)
        for i, det in enumerate(frames):
            ok, frame = cap.read()
            if not ok:
                break
            if det is None:
                continue
            pad = int(max(3 * det.r_px, 12))
            x0, y0 = max(int(det.x_px) - pad, 0), max(int(det.y_px) - pad, 0)
            x1, y1 = min(int(det.x_px) + pad + 1, frame.shape[1]), min(int(det.y_px) + pad + 1, frame.shape[0])
            crop, bg = frame[y0:y1, x0:x1], background[y0:y1, x0:x1]
            helper.max_area_frac = 0.9  # the crop is already only the object's neighbourhood
            best = None
            for _, (cx, cy, area), _, _ in helper._candidates(crop, bg):
                gx, gy = cx + x0, cy + y0
                dist = np.hypot(gx - det.x_px, gy - det.y_px)
                if dist <= max(det.r_px, 4) and (best is None or dist < best[0]):
                    best = (dist, gx, gy, area)
            if best is not None:
                _, gx, gy, area = best
                out[i] = Detection(gx, gy, float(np.sqrt(area / np.pi)), det.outline)
                refined += 1
            else:
                out[i] = None if i not in keep_raw else det
        cap.release()
        if refined < 0.5 * sum(d is not None for d in frames):
            # Motion can't refine an object that never leaves its spot (it is part of the background). Its colour
            # blob inside SAM's region still can: SAM's mask of a weight on a spring takes in a bit of the spring,
            # which pulls the centre up.
            by_colour = self._refine_by_colour(video, frames)
            if by_colour is not None:
                return by_colour
            # Refinement mostly failed (e.g. a moving camera), so SAM's own centres are the best we have.
            return list(frames), 0
        return out, refined

    def _refine_by_colour(self, video: Path, frames: list[Optional[Detection]]):
        if not self.color:
            return None
        from .opencv_tracker import ColorBlobTracker
        blobs = ColorBlobTracker(color=self.color).track(video, "").frames
        out, refined = list(frames), 0
        for i, det in enumerate(frames):
            blob = blobs[i] if i < len(blobs) else None
            if det is None or blob is None:
                continue
            if np.hypot(blob.x_px - det.x_px, blob.y_px - det.y_px) <= max(det.r_px, 4):
                out[i] = blob
                refined += 1
        if refined < 0.5 * sum(d is not None for d in frames):
            return None
        return out, refined

    # --- public ----------------------------------------------------------------------------------------

    @staticmethod
    def _active_window(found: list[Optional[Detection]], fps: float) -> Optional[tuple[int, int]]:
        """Frames where the local tracker saw the object, padded by half a second each side."""
        idx = [i for i, d in enumerate(found) if d is not None]
        if len(idx) < 5:
            return None
        pad = int(round(0.5 * fps))
        return max(idx[0] - pad, 0), min(idx[-1] + pad, len(found) - 1)

    @staticmethod
    def _cut(video: Path, a: int, b: int, fps: float) -> Optional[Path]:
        """Frames a..b as their own clip (exact frames, so detections map straight back)."""
        import subprocess
        import tempfile

        from ..synth import ffmpeg_exe
        out = Path(tempfile.mkdtemp()) / f"{video.stem}-{a}-{b}.mp4"
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(video),
                        "-vf", f"select='between(n\\,{a}\\,{b})',setpts=N/FRAME_RATE/TB", "-an",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "16", "-pix_fmt", "yuv420p",
                        # Baseline, no B-frames, 90 kHz: other encodings made SAM bill ~24× the frames (see api.py).
                        "-profile:v", "baseline", "-bf", "0", "-video_track_timescale", "90000",
                        "-r", f"{fps:.6f}", str(out)], check=True)
        cap = cv2.VideoCapture(str(out))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return out if n == b - a + 1 else None

    def track(self, video: Path, prompt: str) -> TrackResult:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Speed: look locally first. SAM only needs the stretch where the object moves (a 67 s classroom clip has a
        # 4 s drop in it), and a small fast object never matches by name, so it goes straight to a box prompt.
        found = self._local_sightings(video, prompt)
        window = self._active_window(found, fps)
        clip, a, b = video, 0, n_frames - 1
        if window and (window[1] - window[0] + 1) < 0.7 * n_frames:
            cut = self._cut(video, *window, fps)
            if cut is not None:
                clip, (a, b) = cut, window
        n_clip = b - a + 1
        local = found[a:b + 1]
        seen = [d.r_px for d in local if d is not None]
        small = bool(seen) and float(np.median(seen)) < 12

        file_id = self._upload(clip)
        part: list[Optional[Detection]] = [None] * n_clip
        self.last_mode = ""
        if not small:
            part = self._detections(self._segment(file_id, prompt, n_clip), n_clip)
            self.last_mode = f'text prompt "{prompt}"'
        if sum(d is not None for d in part) < 0.3 * n_clip:
            seed = self._seed_box(local, width, height)
            if seed is not None:
                part = self._detections(self._segment(file_id, seed, n_clip), n_clip)
                self.last_mode = "box prompt seeded by the local tracker"
        if clip is not video:
            self.last_mode += f"; frames {a}–{b} of {n_frames} sent to SAM"
            clip.unlink(missing_ok=True)
        frames: list[Optional[Detection]] = [None] * n_frames
        frames[a:b + 1] = part
        frames, refined = self._refine(video, frames)
        self.last_mode += f"; sub-pixel centres in {refined}/{sum(d is not None for d in frames)} frames"
        return TrackResult(fps=fps, width=width, height=height, label=prompt, frames=frames)
