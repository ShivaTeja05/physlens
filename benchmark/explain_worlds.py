"""Muse Spark explains each 3D world's event from its measurements (physlens/ask.py: every number it says must be one
the world measured, checked by the verifier). Saved as the world's "explanation" note, so the shared static copy
carries it too.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/explain_worlds.py [world ...]
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from dotenv import load_dotenv   # noqa: E402
load_dotenv(ROOT / "backend" / ".env")
from physlens import ask   # noqa: E402

PW = ROOT / "media" / "pointworld"
QUESTION = ("What happened in this video, physically? Explain the event in order and why it happened, and say where "
            "the measurement disagrees with known values.")

if __name__ == "__main__":
    names = sys.argv[1:] or [w["name"] for w in json.loads((PW / "index.json").read_text())]
    for name in names:
        f = PW / f"{name}.json"
        h = json.loads(f.read_text())
        if not h["notes"].get("events"):
            print(name, "no events: skipped")
            continue
        ask.reset(f"world:{name}")
        out = ask.ask(f"world:{name}", ask.world_facts(h), QUESTION)
        h["notes"]["explanation"] = out["answer"]
        h["notes"]["explanation_verified"] = out.get("verified", True)
        f.write_text(json.dumps(h))
        print(name, "→", out["answer"], "| verified:", out.get("verified"))
