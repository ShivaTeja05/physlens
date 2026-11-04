"""A static copy of the point worlds for sharing: the simulator page (web/dist/sim.html) plus each world thinned to
at most MAX_POINTS (a random subset, kept in each group: the world and every moment of the object), so the copy
opens anywhere without the engine. Headers keep every note (events, rigid body, checks); "decimated_from" records
the full count.
Usage: PYTHONPATH=backend backend/.venv/bin/python benchmark/web_copy.py <out_dir> [world ...]
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PW = ROOT / "media" / "pointworld"
MAX_POINTS = 1_300_000          # ~13 MB per world (10 bytes a point): under the 15 MB per-file limit for sharing


def thin(name: str, out: Path) -> dict:
    h = json.loads((PW / f"{name}.json").read_text())
    n = h["points"]
    raw = np.fromfile(PW / f"{name}.bin", dtype=np.uint8)
    q = raw[: n * 6].view(np.int16).reshape(n, 3)
    rgb = raw[n * 6: n * 9].reshape(n, 3)
    step = max(int(np.ceil(n / MAX_POINTS)), 1)
    rng = np.random.default_rng(0)
    keep_q, keep_c, groups, start = [], [], [], 0
    for g in h["groups"]:
        sl = slice(g["start"], g["start"] + g["count"])
        # The object's moments keep a denser share than the static world (they are what is measured and played).
        s = max(step - 1, 1) if g["tag"] != 255 else step
        # A random (seeded) subset, not every n-th point: points come in pixel order, and a regular stride
        # leaves stripes.
        pick = np.sort(rng.choice(g["count"], g["count"] // s, replace=False)) if s > 1 else np.arange(g["count"])
        qq, cc = q[sl][pick], rgb[sl][pick]
        keep_q.append(qq)
        keep_c.append(cc)
        groups.append({**g, "start": start, "count": len(qq)})
        start += len(qq)
    Q, C = np.concatenate(keep_q), np.concatenate(keep_c)
    h2 = {**h, "points": int(len(Q)), "groups": groups, "decimated_from": n,
          "static_points": groups[0]["count"] if groups and groups[0]["tag"] == 255 else 0}
    h2["object_points"] = h2["points"] - h2["static_points"]
    (out / f"{name}.json").write_text(json.dumps(h2))
    (out / f"{name}.bin").write_bytes(Q.astype(np.int16).tobytes() + C.astype(np.uint8).tobytes())
    return {"name": name, "title": h["title"], "points": h2["points"]}


if __name__ == "__main__":
    out = Path(sys.argv[1])
    names = sys.argv[2:] or [w["name"] for w in json.loads((PW / "index.json").read_text())]
    dist = ROOT / "web" / "dist"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(dist / "sim.html", out / "index.html")
    # Only the files the simulator page loads (the main app's bundle isn't needed), following imports one level.
    (out / "assets").mkdir(exist_ok=True)
    todo, seen = re.findall(r"\./assets/([\w.-]+)", (dist / "sim.html").read_text()), set()
    while todo:
        f = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        shutil.copy(dist / "assets" / f, out / "assets" / f)
        if f.endswith(".js"):
            todo += re.findall(r"\./([\w.-]+\.(?:js|css))", (dist / "assets" / f).read_text())
    data = out / "data"
    data.mkdir(exist_ok=True)
    index = [thin(nm, data) for nm in names]
    (data / "index.json").write_text(json.dumps(index))
    for w in index:
        print(w["name"], f"{w['points']:,} points,", f"{(data / (w['name'] + '.bin')).stat().st_size / 1e6:.1f} MB")
