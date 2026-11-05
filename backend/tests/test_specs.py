"""Specs agent, offline: parsing, evidence-or-reject, cross-source agreement, the video check, and the endpoints
reading a cached lookup (no network, no model calls)."""
import json

from fastapi.testclient import TestClient

from physlens import api, specs


def test_mm_reads_convert_templates_and_bare_values():
    assert specs._mm("{{convert|2775|mm|in|1|abbr=on}}") == [2775.0]
    assert specs._mm("{{unbulleted list|{{convert|4865|mm|in}} (US)|{{convert|4820|mm|in}} (JP)}}") == [4865.0, 4820.0]
    assert specs._mm("1,816 mm or 1816 mm") == [1816.0, 1816.0]


def test_verify_rejects_numbers_no_tool_returned():
    results = [{"vehicles": [{"wheelbase_mm": 2773.0, "length_mm": 4850.0}]}, {"fields": {"width": {"mm": [1820.0]}}}]
    out = specs.verify([{"name": "wheelbase", "value_mm": 2773, "source": "NHTSA"},
                        {"name": "width", "value_mm": 1820, "source": "Wikipedia"},
                        {"name": "height", "value_mm": 1470, "source": "memory"}], results)
    assert [s["verified"] for s in out] == [True, True, False]


def test_agree_needs_two_sources_within_one_percent():
    a = specs.agree([{"name": "wheelbase", "value_mm": 2773, "source": "NHTSA", "verified": True},
                     {"name": "wheelbase", "value_mm": 2775, "source": "Wikipedia", "verified": True},
                     {"name": "width", "value_mm": 1814, "source": "NHTSA", "verified": True},
                     {"name": "width", "value_mm": 1845, "source": "Wikipedia", "verified": True},
                     {"name": "height", "value_mm": 1470, "source": "Wikipedia", "verified": True},
                     {"name": "length", "value_mm": 9999, "source": "Wikipedia", "verified": False}])
    assert a["wheelbase"]["agreed"] and abs(a["wheelbase"]["value_m"] - 2.774) < 1e-9
    assert not a["width"]["agreed"]          # 1.7% apart
    assert not a["height"]["agreed"]         # one source
    assert "length" not in a                 # unverified values never count


def test_check_against_video_and_whole_body_only():
    agreed = {"length": {"value_m": 4.845, "agreed": True}}
    assert specs.check_against(agreed, {"length": 4.863}) == [
        {"name": "length", "measured_m": 4.863, "published_m": 4.845, "off_pct": 0.37, "published_agreed": True}]
    # a moment's box (visible points only) is not a whole-body length
    _, measured = specs.world_context({"title": "t", "object": "car", "boxes": [{"size": [2.1, 1.0, 1.0]}], "notes": {}})
    assert measured == {}
    _, measured = specs.world_context({"notes": {"rigid_body": {"length": 4.76}}})
    assert measured == {"length": 4.76}


def test_endpoints_read_the_cached_lookup(tmp_path, monkeypatch):
    pw = tmp_path / "pointworld"
    pw.mkdir()
    (pw / "w.json").write_text(json.dumps({"title": "t", "object": "car", "notes": {}}))
    cached = {"object": "Car", "steps": [], "specs": [], "agreed": {}, "checks": []}
    monkeypatch.setattr(api, "MEDIA", tmp_path)
    client = TestClient(api.app)
    assert client.get("/api/worlds/w/specs").status_code == 404          # no lookup yet
    (pw / "w.specs.json").write_text(json.dumps(cached))
    assert client.get("/api/worlds/w/specs").json() == cached
    assert client.post("/api/worlds/w/specs", data={"refresh": "false"}).json() == cached   # cached: no model call
    assert client.post("/api/worlds/nope/specs").status_code == 404
