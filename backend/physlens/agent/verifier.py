"""Evidence-or-reject: every number in an explanation must match a measured value."""
from __future__ import annotations

import re

NUMBER = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?")


def numbers_in(text: str) -> list[str]:
    return [m.group().replace("−", "-") for m in NUMBER.finditer(text)]


def check(text: str, evidence: dict[str, float]) -> tuple[bool, list[str]]:
    """A printed number is supported if some evidence value rounds to it at the printed precision.
    Returns (all supported, unsupported numbers)."""
    unsupported = []
    values = [float(v) for v in evidence.values() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    for token in numbers_in(text):
        decimals = len(token.split(".")[1]) if "." in token else 0
        value = float(token)
        if not any(round(v, decimals) == value or round(abs(v), decimals) == abs(value) for v in values):
            unsupported.append(token)
    return not unsupported, unsupported
