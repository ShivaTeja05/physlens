"""What an object is, and its published size, from Muse Spark looking at it: knowledge, not measurement.

The camera measures shape and motion; it can't know a car's true width when it sees one side, or the world's true
scale. A recognised object brings both: "a 2015 Lexus NX 200t" has a published length and width. These numbers
are labelled as predicted from the model's knowledge everywhere they're used (scale, the mirror plane of the
unseen side), never as measured.
"""
from __future__ import annotations

import numpy as np

from .scan import _client, _json_reply, _jpeg_url

PROMPT = """You see one object cut out from a video frame. First read any text on it (badges, placards, signs): text
naming the model beats a judgement from the shape. Identify it as precisely as the image allows and give its
published exterior dimensions (manufacturer specification, metres), not a guess from the picture's pixels.
Reply with ONE JSON object:
{"kind": "car|truck|bus|motorcycle|person|other", "make": "...", "model": "...", "years": "...",
 "length_m": number, "width_m": number (body, without mirrors), "height_m": number,
 "text_read": "any text you can read on it", "confidence": "high|medium|low",
 "why": "what in the image identifies it (text, badges, lights, shape)"}
If you can't identify the exact model, give the closest class and typical dimensions with confidence "low"."""


def identify(img: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Muse Spark's identification of the object in `img` (BGR), cropped to `mask`'s box when given."""
    client, model = _client()
    if client is None:
        raise RuntimeError("Needs Muse Spark: set META_API_KEY in backend/.env")
    if mask is not None and mask.any():
        ys, xs = np.nonzero(mask)
        pad = 20
        y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad, img.shape[0])
        x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad, img.shape[1])
        img = img[y0:y1, x0:x1]
    out = _json_reply(client, model, [{"role": "user", "content": [
        {"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": _jpeg_url(img, 1600)}}]}], "priors: identify")
    for k in ("length_m", "width_m", "height_m"):
        try:
            out[k] = float(out[k])
        except (KeyError, TypeError, ValueError):
            out[k] = None
    out["source"] = f"Muse Spark ({model}), published dimensions: predicted from knowledge, not measured"
    return out
