"""The brief names the ohsome API as the OSM source: Overpass must never be used unless switched on."""
from datetime import date

import pytest

from floodsight.fetch import osm


def test_overpass_off_by_default(monkeypatch):
    calls = []
    monkeypatch.delenv("OHSOME_API_KEY", raising=False)
    monkeypatch.setattr(osm, "_ohsome_v1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("v1 down")))
    monkeypatch.setattr(osm, "_overpass", lambda *a, **k: calls.append(1))
    with pytest.raises(osm.OSMUnavailable):
        osm.fetch_layer("roads", None, date(2026, 7, 27), {"ohsome_url": "x"})
    assert calls == []


def test_overpass_only_when_switched_on(monkeypatch):
    calls = []
    monkeypatch.delenv("OHSOME_API_KEY", raising=False)
    monkeypatch.setattr(osm, "_ohsome_v1", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("v1 down")))
    monkeypatch.setattr(osm, "_overpass", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError("x")))
    with pytest.raises(osm.OSMUnavailable):
        osm.fetch_layer("roads", None, date(2026, 7, 27), {"ohsome_url": "x", "overpass_fallback": True})
    assert calls == [1]


def _ctx(tmp_path):
    from datetime import date
    from floodsight.core import AOI, RunContext, load_config
    return RunContext(AOI(85.30, 27.95, 85.40, 28.03), date(2026, 8, 26), load_config(), runs_dir=tmp_path, name="osm_t")


def test_optional_layer_failing_keeps_the_rest(monkeypatch, tmp_path):
    import geopandas as gpd
    from shapely.geometry import Point
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(osm, "latest_available", lambda *a, **k: "2026-08-01")
    one = gpd.GeoDataFrame({"highway": ["primary"]}, geometry=[Point(85.35, 28.0)], crs=4326)

    def fake(key, *a, **k):
        if key == "power_lines":
            raise osm.OSMUnavailable("ohsome v2: 429; ohsome v1: 403")
        return one, "ohsome v2"
    monkeypatch.setattr(osm, "fetch_layer", fake)
    out = osm.fetch_osm(ctx)
    assert len(out["roads"]) == 1 and len(out["buildings"]) == 1 and len(out["power_lines"]) == 0
    assert any("power lines could not be downloaded" in n for n in ctx.notes)
    assert not ctx.path("inputs", "osm_power_lines_2026-08-01.geojson").exists()      # retried next run


def test_required_layer_failing_still_raises(monkeypatch, tmp_path):
    ctx = _ctx(tmp_path)
    monkeypatch.setattr(osm, "latest_available", lambda *a, **k: "2026-08-01")

    def fake(key, *a, **k):
        raise osm.OSMUnavailable("down")
    monkeypatch.setattr(osm, "fetch_layer", fake)
    with pytest.raises(osm.OSMUnavailable):
        osm.fetch_osm(ctx)


def test_ohsome_v2_waits_and_retries_on_429(monkeypatch):
    from datetime import date
    from floodsight.core import AOI

    class R:
        def __init__(self, code):
            self.status_code, self.headers, self.content = code, {}, b""
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)
    seq = [R(429), R(429), R(200)]
    monkeypatch.setattr(osm.requests, "request", lambda *a, **k: seq.pop(0))
    slept = []
    monkeypatch.setattr(osm.time, "sleep", lambda s: slept.append(s))
    monkeypatch.setattr(osm.gpd if hasattr(osm, "gpd") else __import__("geopandas"), "read_parquet",
                        lambda b: __import__("geopandas").GeoDataFrame(geometry=[], crs=4326))
    osm._ohsome_v2(AOI(85.30, 27.95, 85.40, 28.03), date(2026, 8, 1), "highway=*", "key")
    assert slept == [5, 10] and not seq
