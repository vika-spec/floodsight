"""Sentinel Hub route, tested against a fake server (no network):
pixel alignment with the analysis grid, tiling of large areas, token reuse, rate-limit retries, raw-only
evalscript, and backend fallback / same-backend rule in fetch_pair."""
import io
import json
from datetime import date

import numpy as np
import pytest


class FakeResp:
    def __init__(self, status, content=b"", headers=None, js=None):
        self.status_code, self.content, self.headers, self._js = status, content, headers or {}, js
        self.text = content.decode(errors="ignore") if isinstance(content, bytes) else str(content)

    def json(self):
        return self._js


def _tiff(body):
    """GeoTIFF the way Sentinel Hub returns it: band values encode pixel-centre coordinates."""
    import rasterio
    from rasterio.transform import from_bounds

    x0, y0, x1, y1 = body["input"]["bounds"]["bbox"]
    w, h = body["output"]["width"], body["output"]["height"]
    tf = from_bounds(x0, y0, x1, y1, w, h)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    xs, ys = tf * (cols, rows)
    nb = 3 if "VV" in body["evalscript"] else 4
    data = np.stack([xs / 1e6, ys / 1e6, np.ones_like(xs)] + ([np.ones_like(xs)] if nb == 4 else [])).astype("float32")
    buf = io.BytesIO()
    with rasterio.MemoryFile() as mf:
        with mf.open(driver="GTiff", width=w, height=h, count=nb, dtype="float32", transform=tf,
                     crs=body["input"]["bounds"]["properties"]["crs"].split("/")[-1].join(["EPSG:", ""])) as dst:
            dst.write(data)
        buf.write(mf.read())
    return buf.getvalue()


class FakeSession:
    def __init__(self, fail_first=0):
        self.headers, self.calls, self.tokens, self.fail_first, self.bodies = {}, 0, 0, fail_first, []

    def post(self, url, data=None, json=None, timeout=None, headers=None):
        if "openid-connect/token" in url:
            self.tokens += 1
            return FakeResp(200, js={"access_token": "tok", "expires_in": 600})
        self.calls += 1
        if self.calls <= self.fail_first:
            return FakeResp(429, b"slow down", {"Retry-After": "10"})
        assert headers["Authorization"] == "Bearer tok"
        self.bodies.append(json)
        return FakeResp(200, _tiff(json), {"x-processingunits-spent": "12.5"})


@pytest.fixture
def shub(monkeypatch):
    from floodsight.fetch import shub as m
    monkeypatch.setenv("SH_CLIENT_ID", "id")
    monkeypatch.setenv("SH_CLIENT_SECRET", "secret")
    m._TOKEN.update(value=None, expires=0)
    m.USAGE.update(pu=0.0, requests=0)
    monkeypatch.setattr(m.time, "sleep", lambda s: None)
    return m


def _ctx(tmp_path, cfg, bbox="85.30,27.95,85.33,27.98", res=20):
    from floodsight.core import AOI, RunContext, deep_merge
    return RunContext(AOI.parse(bbox), date(2026, 8, 26), deep_merge(cfg, {"grid": {"resolution_m": res}}),
                      runs_dir=tmp_path, name="t")


def _acq(day="2026-08-28", direction="ascending"):
    from floodsight.fetch.catalog import Acquisition
    return Acquisition(day=date.fromisoformat(day), rel_orbit=85, direction=direction,
                       scenes=["S1A_IW_GRDH_1SDV_20260828T001010_20260828T001035_000000_000000_ABCD.SAFE"],
                       coverage=1.0)


def test_s1_is_pixel_aligned_and_raw_only(tmp_path, cfg, shub, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(shub, "_session", lambda: fake)
    ctx = _ctx(tmp_path, cfg)
    out = shub.s1_acquisition(ctx, _acq(), ctx.path("inputs", "s1_test_shub.tif"))
    from floodsight.core import read_to_grid
    arr = read_to_grid(out, ctx.grid)
    g = ctx.grid
    r, c = 7, 11
    x, y = g.transform * (c + 0.5, r + 0.5)
    assert arr.shape == (2, g.height, g.width)
    assert abs(arr[0, r, c] * 1e6 - x) < 1 and abs(arr[1, r, c] * 1e6 - y) < 1      # same pixel centres
    body = fake.bodies[0]
    p = body["input"]["data"][0]
    assert p["type"] == "sentinel-1-grd" and p["processing"]["orthorectify"] is True
    assert p["processing"]["backCoeff"] == "SIGMA0_ELLIPSOID" and p["processing"]["demInstance"] == "COPERNICUS_30"
    assert p["dataFilter"]["orbitDirection"] == "ASCENDING"
    assert p["dataFilter"]["timeRange"]["from"] == "2026-08-28T00:08:10Z"            # this pass only
    es = body["evalscript"]
    assert "return [s.VV, s.VH, s.dataMask]" in es                                   # raw backscatter only
    for derived in ("NDWI", "water", "flood", "index"):
        assert derived.lower() not in es.lower()
    assert shub.USAGE["pu"] == 12.5


def test_large_grid_is_tiled_and_token_reused(tmp_path, cfg, shub, monkeypatch):
    fake = FakeSession(fail_first=2)          # two rate-limit answers first
    monkeypatch.setattr(shub, "_session", lambda: fake)
    ctx = _ctx(tmp_path, cfg, bbox="85.00,27.80,85.60,28.30", res=20)       # ~59 x 55 km -> > 2500 px wide
    assert max(ctx.grid.width, ctx.grid.height) > shub.MAX_PX
    out = shub.s1_acquisition(ctx, _acq(), ctx.path("inputs", "s1_big_shub.tif"))
    from floodsight.core import read_to_grid
    arr = read_to_grid(out, ctx.grid)
    assert np.isfinite(arr).all()                                                   # tiles cover everything
    g = ctx.grid
    for r, c in ((0, 0), (g.height - 1, g.width - 1), (2001, 1999)):
        x, y = g.transform * (c + 0.5, r + 0.5)
        assert abs(arr[0, r, c] * 1e6 - x) < 1 and abs(arr[1, r, c] * 1e6 - y) < 1
    assert all(b["output"]["width"] <= shub.MAX_PX and b["output"]["height"] <= shub.MAX_PX for b in fake.bodies)
    assert fake.tokens == 1


def test_s2_evalscript_masks_clouds(shub):
    es = shub.s2_evalscript("median")
    assert 'mosaicking: "ORBIT"' in es and "[0, 1, 2, 3, 8, 9, 10, 11]" in es   # clouds, shadows, dark areas, snow
    assert "clear[clear.length - 1]" in shub.s2_evalscript("first")      # earliest clear pass after the event


def test_s2_requested_as_uint16_and_nodata_restored(shub):
    """Float output costs double processing units; integers lose nothing for Sentinel-2 digital numbers."""
    import numpy as np
    es = shub.s2_evalscript("median")
    assert 'sampleType: "UINT16"' in es and "FLOAT32" not in es and "[0, 0, 0, 0]" in es
    raw = np.array([[[0, 812]], [[0, 640]], [[0, 2310]], [[0, 1500]]], np.uint16)   # pixel 0: no clear view
    out = shub.s2_nodata_to_nan(raw)
    assert np.isnan(out[:, 0, 0]).all() and out[:, 0, 1].tolist() == [812, 640, 2310, 1500]


def test_fallback_to_openeo_and_same_backend_rule(tmp_path, cfg, shub, monkeypatch):
    from floodsight.core import write_raster
    from floodsight.fetch import s1
    from floodsight.fetch.catalog import Pair

    ctx = _ctx(tmp_path, cfg)
    pair = Pair(pre=_acq("2026-08-16"), post=_acq("2026-08-28"), score=1, incidence_deg=38, heading_deg=-12)

    def broken(*a, **k):
        raise shub.ShubUnavailable("quota exhausted")

    def fake_openeo(ctx_, acq, out):
        write_raster(out, np.full((2,) + ctx_.grid.shape, 0.05, np.float32), ctx_.grid, nodata=np.nan)
        return out

    monkeypatch.setattr(shub, "s1_acquisition", broken)
    monkeypatch.setattr(s1, "_openeo_acquisition", fake_openeo)
    out = s1.fetch_pair(ctx, pair)
    assert out["backend"] == "openeo"
    assert any("sentinelhub" in n for n in ctx.notes)
    assert np.allclose(out["pre"][0], 10 * np.log10(0.05), atol=0.01)

    # next run: both openEO files are cached, so the pair is reused as is (never mixed with Sentinel Hub)
    monkeypatch.setattr(s1, "_openeo_acquisition", lambda *a: (_ for _ in ()).throw(AssertionError("refetched")))
    assert s1.fetch_pair(ctx, pair)["backend"] == "openeo"


def test_auto_without_key_skips_sentinel_hub(monkeypatch):
    from floodsight.fetch import s1
    monkeypatch.delenv("SH_CLIENT_ID", raising=False)
    monkeypatch.delenv("SH_CLIENT_SECRET", raising=False)
    assert s1._backend_order("auto") == ["openeo", "local"]
    monkeypatch.setenv("SH_CLIENT_ID", "a")
    monkeypatch.setenv("SH_CLIENT_SECRET", "b")
    assert s1._backend_order("auto") == ["sentinelhub", "openeo", "local"]
    assert json.dumps(s1._backend_order("local")) == '["local"]'


def test_parallel_requests_report_their_own_units(tmp_path, cfg, shub, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    fake = FakeSession()
    monkeypatch.setattr(shub, "_session", lambda: fake)
    ctx = _ctx(tmp_path, cfg)
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda i: shub._request_grid(ctx.grid, shub.s1_request_data(_acq(), cfg),
                                                       shub.S1_EVALSCRIPT, 3, f"r{i}")[1], range(4)))
    assert res == [12.5] * 4                     # each request: its own cost, not the others'
    assert shub.USAGE["pu"] == 50.0


def test_cloudy_after_image_skips_optical(tmp_path, cfg, shub, monkeypatch):
    from floodsight.fetch import s2
    ctx = _ctx(tmp_path, cfg)
    monkeypatch.setattr(shub, "s2_clear_fraction", lambda *a, **k: (0.02, 0.4))
    monkeypatch.setattr(shub, "s2_composite", lambda *a, **k: (_ for _ in ()).throw(AssertionError("fetched")))
    assert s2.fetch_s2(ctx) is None
    assert any("Sentinel-2 skipped" in n for n in ctx.notes)


def test_cache_name_depends_on_calibration(cfg):
    from floodsight.core import deep_merge
    from floodsight.fetch import shub
    s0 = shub.cache_tag(cfg)
    g0 = shub.cache_tag(deep_merge(cfg, {"sentinel1": {"shub_coefficient": "GAMMA0_TERRAIN"}}))
    assert s0 == "_shub-s0" and g0 == "_shub-g0t"     # gamma0 images are never reused as sigma0
