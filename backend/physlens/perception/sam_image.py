"""SAM 3.1 on one image: every instance of a named thing (or the thing under a tap), as full-size boolean masks."""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass

import cv2
import numpy as np

API_BASE = "https://api.meta.ai/v1"


@dataclass
class Instance:
    mask: np.ndarray          # bool, image size
    confidence: float | None


def segment(img: np.ndarray, prompt: str, what: str = "image") -> list[Instance]:
    """`prompt` is plain text ("window") or an object prompt built with meta_sam_parser.build_object_prompt."""
    from meta_sam_parser import CompletedOutcome, decode_mask_to_raster, image_segmentation_format
    from openai import OpenAI

    from .. import budget

    key = os.environ.get("META_API_KEY")
    if not key:
        raise RuntimeError("SAM 3.1 not configured: set META_API_KEY in backend/.env")
    client = OpenAI(api_key=key, base_url=os.environ.get("META_API_BASE", API_BASE))
    ok, png = cv2.imencode(".png", img)
    budget.check(budget.SAM_PER_IMAGE)
    r = client.responses.create(
        model=os.environ.get("SAM_MODEL", "sam-3.1"),
        input=[{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode()}]}],
        metadata={"mask_encoding": "one_bit", "include_confidence": "true"})
    budget.record_sam("sam-3.1", getattr(r, "usage", None), what=f"{what}: {prompt[:40]}")
    parser = image_segmentation_format().create_parser()
    for _ in parser.push(r.output_text or ""):
        pass
    out = []
    h, w = img.shape[:2]
    for rec in parser.finish(CompletedOutcome()).result.records:
        if rec.kind != "mask":
            continue
        raster = np.frombuffer(decode_mask_to_raster(rec.mask), np.uint8).reshape(rec.mask.height, rec.mask.width)
        full = np.zeros((h, w), bool)
        b = rec.bounds
        top, left = int(b.top), int(b.left)
        hh, ww = min(raster.shape[0], h - top), min(raster.shape[1], w - left)
        full[top:top + hh, left:left + ww] = raster[:hh, :ww] > 0
        if full.any():
            out.append(Instance(full, rec.confidence))
    return out


def outline(mask: np.ndarray, max_points: int = 64) -> list[list[int]]:
    """Simplified outer contour of the biggest blob, for drawing."""
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []
    c = max(cnts, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(c, True)
    while True:
        poly = cv2.approxPolyDP(c, eps, True)
        if len(poly) <= max_points:
            break
        eps *= 1.5
    return [[int(x), int(y)] for x, y in poly[:, 0]]
