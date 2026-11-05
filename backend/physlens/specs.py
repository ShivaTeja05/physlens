"""Published specifications of a measured object, looked up by a Muse Spark agent with tools: sources, not memory.

priors.py asks Muse Spark what an object is and its size from the model's knowledge, which can be wrong (it once
called a Lexus NX an RX). Here Muse Spark plans and calls lookup tools instead: NHTSA's VIN decoder and crash-test
vehicle records, and Wikipedia's infobox for the identified model. Every number it reports must appear in what a tool
returned (evidence or reject, as in agent/verifier.py), two sources must agree before a value is trusted, and the
trusted values are then checked against what the video itself measured, so a wrong trim or a bad page is caught
instead of quietly scaling the measurement.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx

UA = {"User-Agent": "PhysLens/1.0 (research; github.com/ShivaTeja05/physlens)", "Accept": "application/json"}
VPIC = "https://vpic.nhtsa.dot.gov/api/vehicles/DecodeVinValues/{vin}?format=json"
NHTSA_TEST = "https://nrd.api.nhtsa.dot.gov/nhtsa/vehicle/api/v1/vehicle-database-test-results/metadata/{test}"
WIKI_API = "https://en.wikipedia.org/w/api.php"
SPEC_FIELDS = ("wheelbase", "length", "width", "height", "track", "tires", "weight")


# --- tools ------------------------------------------------------------------------------------------------------

def decode_vin(vin: str) -> dict:
    """NHTSA vPIC: what a VIN says (make, model, year, series, body)."""
    r = httpx.get(VPIC.format(vin=vin.strip()), headers=UA, timeout=20, follow_redirects=True).json()["Results"][0]
    keep = ("Make", "Model", "ModelYear", "Series", "Trim", "BodyClass", "Doors", "DriveType", "ErrorText")
    return {"source": "NHTSA vPIC VIN decoder", "url": VPIC.format(vin=vin.strip()),
            **{k: r.get(k) for k in keep if r.get(k)}}


def nhtsa_test_vehicles(test: int) -> dict:
    """NHTSA crash-test record: each vehicle in the test with its VIN and measured length, width and wheelbase (mm)."""
    url = NHTSA_TEST.format(test=int(test))
    m = httpx.get(url, headers=UA, timeout=30, follow_redirects=True).json()
    if isinstance(m, dict) and "results" in m:
        m = m["results"][0] if m["results"] else {}
    vehicles = [{"vehicle": v.get("VEHNO"), "make": v.get("MAKED"), "model": v.get("MODELD"), "year": v.get("YEAR"),
                 "vin": v.get("VIN"), "length_mm": v.get("VEHLEN"), "width_mm": v.get("VEHWID"),
                 "wheelbase_mm": v.get("WHLBAS"), "test_weight_kg": v.get("VEHTWT")} for v in m.get("VEHICLE", [])]
    return {"source": f"NHTSA crash test {int(test)} data sheet", "url": url,
            "title": (m.get("TEST") or {}).get("TITLE"), "vehicles": vehicles}


def wikipedia_search(query: str) -> dict:
    r = httpx.get(WIKI_API, params={"action": "opensearch", "search": query, "limit": 6, "namespace": 0, "format": "json"},
                  headers=UA, timeout=20, follow_redirects=True).json()
    return {"source": "Wikipedia search", "titles": r[1], "urls": r[3]}


def _mm(text: str) -> list[float]:
    """Millimetre values in an infobox field: {{convert|2775|mm|in}} templates, then bare 'NNNN mm'."""
    text = re.sub(r"(\d),(\d{3})\b", r"\1\2", text)          # 1,816 -> 1816
    vals = [float(a) * {"mm": 1, "cm": 10, "m": 1000, "in": 25.4}.get(u, 0)
            for a, u in re.findall(r"\{\{convert\|([\d.]+)\|(mm|cm|m|in)\b", text)]
    vals += [float(a) for a in re.findall(r"(?<![\d.])(\d{3,4}(?:\.\d)?)\s*mm", text)]
    return [round(v, 1) for v in vals if v > 0]


def _fields(wt: str) -> dict:
    fields = {}
    for name in SPEC_FIELDS + ("name", "model_code", "production", "model_years"):
        # a field runs until the next "| name =" line
        m = re.search(rf"^\s*\|\s*{name}\s*=(.*?)(?=^\s*\|\s*\w[\w ]*=|\Z)", wt, re.S | re.M)
        if m:
            raw = re.sub(r"<ref.*?(</ref>|/>)", "", m.group(1), flags=re.S).strip()
            fields[name] = {"raw": raw[:300], "mm": _mm(raw)} if name in SPEC_FIELDS else raw[:120]
    return fields


def wikipedia_infobox(title: str) -> dict:
    """The automobile infoboxes of one Wikipedia article (one per generation when the article covers several):
    wheelbase, length, width, height, track, tires, weight (mm) with each box's name, model code and years."""
    r = httpx.get(WIKI_API, params={"action": "parse", "page": title, "prop": "wikitext", "format": "json",
                                    "formatversion": 2, "redirects": 1}, headers=UA, timeout=20, follow_redirects=True).json()
    if "parse" not in r:
        return {"source": "Wikipedia", "title": title, "error": r.get("error", {}).get("info", "not found")}
    wt = r["parse"]["wikitext"]
    page = r["parse"]["title"]
    chunks = re.split(r"\{\{\s*Infobox automobile", wt, flags=re.I)[1:]
    boxes = [f for f in (_fields(c[:8000]) for c in chunks) if any(k in f for k in SPEC_FIELDS)]
    return {"source": f"Wikipedia: {page}", "url": "https://en.wikipedia.org/wiki/" + page.replace(" ", "_"),
            "infoboxes": boxes[:6]}


TOOLS = {"decode_vin": decode_vin, "nhtsa_test_vehicles": nhtsa_test_vehicles,
         "wikipedia_search": wikipedia_search, "wikipedia_infobox": wikipedia_infobox}
TOOL_SPECS = [
    {"type": "function", "function": {"name": "nhtsa_test_vehicles", "description": "Vehicles in an NHTSA crash test: VIN, make, model, year and the test lab's measured length, width, wheelbase in mm.",
     "parameters": {"type": "object", "properties": {"test": {"type": "integer", "description": "NHTSA test number"}}, "required": ["test"]}}},
    {"type": "function", "function": {"name": "decode_vin", "description": "Decode a VIN with NHTSA vPIC: make, model, model year, series, body.",
     "parameters": {"type": "object", "properties": {"vin": {"type": "string"}}, "required": ["vin"]}}},
    {"type": "function", "function": {"name": "wikipedia_search", "description": "Find Wikipedia article titles, e.g. the model generation article ('Toyota Camry (XV50)').",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "wikipedia_infobox", "description": "Spec fields (wheelbase, length, width, height, track, tires, weight; values in mm) of one Wikipedia article: one infobox per generation, each with its name, model code and years. Pick the box matching the identified generation.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}}, "required": ["title"]}}},
]

PROMPT = """You find the published specifications of the object PhysLens measured, using only the tools. You never
state a number from memory: every value you report must be copied from a tool result.
Steps: identify the exact object (for an NHTSA test, start with nhtsa_test_vehicles, then decode the VIN of the
vehicle that was measured), find the article for its generation (model code from the series, e.g. XV50), read its
infobox, and collect wheelbase, length, width, height (and track and tyre size if given).
Reply at the end with ONE JSON object:
{"object": "make model year (generation)", "identified_by": "how (VIN decoded, test record, badge text)",
 "specs": [{"name": "wheelbase|length|width|height|track|tires", "value_mm": number, "source": "source name",
            "url": "url from the tool result"}]}
List each value once per source (the same quantity from two sources is two entries), and include EVERY source you
read: for an NHTSA test, the data sheet's length, width and wheelbase of the measured vehicle are always entries."""


# --- the agent ---------------------------------------------------------------------------------------------------

def _evidence_numbers(results: list[dict]) -> list[float]:
    nums = []
    for r in results:
        for tok in re.findall(r"(?<![\w.])\d+(?:\.\d+)?", json.dumps(r)):
            nums.append(float(tok))
    return nums


def verify(specs: list[dict], results: list[dict]) -> list[dict]:
    """Evidence or reject: a value stands only if a tool returned it (in mm, or in m x 1000)."""
    ev = _evidence_numbers(results)
    out = []
    for s in specs:
        v = s.get("value_mm")
        ok = isinstance(v, (int, float)) and any(abs(e - v) <= 0.5 or abs(e * 1000 - v) <= 0.5 for e in ev)
        out.append({**s, "verified": bool(ok)})
    return out


def agree(specs: list[dict], tol: float = 0.01) -> dict:
    """Per quantity: the verified values, whether at least two sources agree within `tol`, and their mean."""
    by: dict[str, list[dict]] = {}
    for s in specs:
        if s.get("verified"):
            by.setdefault(s["name"], []).append(s)
    out = {}
    for name, ss in by.items():
        vals = [float(s["value_mm"]) for s in ss]
        agreed = len({s["source"] for s in ss}) >= 2 and (max(vals) - min(vals)) <= tol * max(vals)
        out[name] = {"values_mm": vals, "sources": [{"source": s["source"], "url": s.get("url")} for s in ss],
                     "agreed": agreed, "value_m": round(sum(vals) / len(vals) / 1000, 4)}
    return out


def lookup(context: str, max_steps: int = 10) -> dict:
    """Run the agent on a text description of what was measured (world title, object label, credits)."""
    from . import budget
    from .scan import _client

    client, model = _client()
    if client is None:
        raise RuntimeError("The specs agent needs Muse Spark: set META_API_KEY in backend/.env")
    messages = [{"role": "system", "content": PROMPT}, {"role": "user", "content": context}]
    results, steps = [], []
    for _ in range(max_steps):
        budget.check(0.02)
        r = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SPECS, max_tokens=3000,
                                           reasoning_effort="low")
        budget.record_tokens(model, r.usage, what="specs agent")
        msg = r.choices[0].message
        if not msg.tool_calls:
            raw = msg.content or ""
            try:
                final = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
            except ValueError:
                final = {"object": None, "specs": [], "error": "no JSON in the final reply"}
            specs = verify(final.get("specs", []), results)
            return {"object": final.get("object"), "identified_by": final.get("identified_by"), "steps": steps,
                    "specs": specs, "agreed": agree(specs)}
        messages.append({"role": "assistant", "content": msg.content or "",
                         "tool_calls": [{"id": t.id, "type": "function",
                                         "function": {"name": t.function.name, "arguments": t.function.arguments}}
                                        for t in msg.tool_calls]})
        for t in msg.tool_calls:
            try:
                args = json.loads(t.function.arguments or "{}")
                out = TOOLS[t.function.name](**args)
            except Exception as exc:          # a failed lookup is reported back to the agent, not raised
                out = {"error": str(exc)[:200]}
            results.append(out)
            steps.append({"tool": t.function.name, "args": args if isinstance(args, dict) else {},
                          "source": out.get("source"), "url": out.get("url"), "error": out.get("error")})
            messages.append({"role": "tool", "tool_call_id": t.id, "content": json.dumps(out)[:6000]})
    return {"object": None, "steps": steps, "specs": [], "agreed": {}, "error": "stopped after too many steps"}


def check_against(agreed: dict, measured: dict) -> list[dict]:
    """Published (agreed) vs what the video measured, for each quantity both have."""
    rows = []
    for name, m in measured.items():
        if name in agreed and m:
            p = agreed[name]["value_m"]
            rows.append({"name": name, "measured_m": round(m, 3), "published_m": p,
                         "off_pct": round((m - p) / p * 100, 2), "published_agreed": agreed[name]["agreed"]})
    return rows


def world_context(header: dict) -> tuple[str, dict]:
    """The agent's brief and the measured sizes for one simulator world (media/pointworld/<name>.json)."""
    notes = header.get("notes", {})
    measured = {}
    # Only whole-body measurements count: a moment's box covers just the points that camera saw of the object.
    checks = (notes.get("multicam") or {}).get("checks") or {}
    rb = notes.get("rigid_body") or {}
    if isinstance(checks.get("length_m"), list):
        measured["length"] = checks["length_m"][0]
    elif rb.get("length"):
        measured["length"] = rb["length"]
    context = (f"World: {header.get('title')}\nObject measured: {header.get('object')}\n"
               f"Footage: {notes.get('credit', '')}\nFind this object's published specifications.")
    return context, measured


def for_world(pw_dir: Path, name: str, force: bool = False) -> dict:
    """Cached per world (the lookup costs Muse Spark calls): media/pointworld/<name>.specs.json."""
    cache = pw_dir / f"{name}.specs.json"
    if cache.exists() and not force:
        return json.loads(cache.read_text())
    header = json.loads((pw_dir / f"{name}.json").read_text())
    context, measured = world_context(header)
    out = lookup(context)
    out["checks"] = check_against(out.get("agreed", {}), measured)
    cache.write_text(json.dumps(out, indent=1))
    return out
