"""Command line: render test clips, or analyse a video and write a replay log.

  python -m physlens.cli samples
  python -m physlens.cli analyze ../samples/drop-slowmo-2x.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_env
from .events import EventLog
from .experiments import CARDS
from .ledger import Ledger
from .pipeline import run_pipeline
from .synth import render_samples

ROOT = Path(__file__).resolve().parents[2]
load_env()


def main() -> None:
    parser = argparse.ArgumentParser(prog="physlens")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("samples", help="render the synthetic test clips into ../samples")
    a = sub.add_parser("analyze", help="run one experiment on a video")
    a.add_argument("video", type=Path)
    a.add_argument("--experiment", default="drop", choices=sorted(CARDS))
    a.add_argument("--px-per-m", type=float, help="scale from a ruler in the shot: pixels per metre")
    a.add_argument("--diameter-cm", type=float, help="known diameter of the tracked ball, in cm")
    args = parser.parse_args()

    if args.cmd == "samples":
        for p in render_samples(ROOT / "samples"):
            print(p)
        return

    name = args.video.stem
    log = EventLog(name, ROOT / "replays" / f"{name}.jsonl")
    summary = run_pipeline(args.video, CARDS[args.experiment], log, Ledger(ROOT / "replays" / f"{name}.ledger.jsonl"),
                           video_url=f"/samples/{args.video.name}", px_per_m=args.px_per_m,
                           diameter_m=args.diameter_cm / 100 if args.diameter_cm else None)
    print(json.dumps(summary, indent=2, default=float))
    print(f"\n{len(log.events)} events → {log.path}")


if __name__ == "__main__":
    main()
