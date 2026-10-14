"""Sentinel Hub Process API on the Copernicus Data Space Ecosystem (sh.dataspace.copernicus.eu).

Same raw input as openEO (Level-1 GRD for Sentinel-1, L2A for Sentinel-2, from dataspace.copernicus.eu), but
processed by Sentinel Hub's low-latency engine instead of openEO's batch engine: on CDSE the same request
typically returns in seconds instead of minutes.

Sentinel-1 processing chain requested here (all standard SAR preprocessing, chosen by FloodSight):
  thermal-noise removal (always on) -> calibration (sigma0 by default, config sentinel1.shub_coefficient) ->
  orthorectification with the Copernicus 30 m DEM -> Lee speckle filter -> resampled onto our analysis grid.
sigma0 is the default because it matches openEO (so a fallback does not change results) and is the closest
to the Kuro Siwo training data (scripts/check_domain.py).
The evalscript returns ONLY raw backscatter (VV, VH linear) and a data mask: no water index, flood product or
any derived layer, so every detection is still made by our own code (brief: run from raw satellite data;
published flood/damage maps are not inputs).

Credentials: an OAuth client from the CDSE Sentinel Hub dashboard, in .env as SH_CLIENT_ID / SH_CLIENT_SECRET.
Free tier: 10 000 processing units (PU) and 10 000 requests per month, 300 of each per minute.
"""
from __future__ import annotations

import io
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from ..core import Grid, RunContext, write_raster
from .. import activity

log = logging.getLogger("floodsight")


class _QuietSentinelHubTiff(logging.Filter):
    """Sentinel Hub's float GeoTIFFs make GDAL print a harmless 'Photometric ... ExtraSamples' note."""

    def filter(self, record):
        return "ExtraSamples" not in record.getMessage()


logging.getLogger("rasterio._env").addFilter(_QuietSentinelHubTiff())

TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
PROCESS_URL = "https://sh.dataspace.copernicus.eu/api/v1/process"
MAX_PX = 2500            # Process API output limit per request (width and height)
TILE_PX = 2000           # tile size used for larger grids
SLOTS = threading.BoundedSemaphore(4)
WAITS = (5, 15, 30, 60)

_TOKEN = {"value": None, "expires": 0.0}
_TOKEN_LOCK = threading.Lock()
USAGE = {"pu": 0.0, "requests": 0}     # processing units spent in this process (logged per run)


class ShubUnavailable(RuntimeError):
    pass


class ShubQuotaExceeded(ShubUnavailable):
    """The Copernicus account behind these requests has used up its processing-unit or request quota."""


# Keys a user typed in the web app for ONE run (their own Copernicus account). Held in memory only while that
# run is processed; never written to disk, logs or results. Runs are processed one at a time (server/jobs.py).
_OVERRIDE: dict = {"cred": None}


def override_active() -> bool:
    return _OVERRIDE["cred"] is not None


class use_credentials:
    """with use_credentials((client_id, secret)): ... -> every Sentinel Hub request inside uses these keys."""

    def __init__(self, cred: tuple[str, str] | None):
        self.cred = cred

    def __enter__(self):
        with _TOKEN_LOCK:
            _OVERRIDE["cred"] = self.cred
            _TOKEN["value"] = None                     # never reuse a token from another account
        return self

    def __exit__(self, *exc):
        with _TOKEN_LOCK:
            _OVERRIDE["cred"] = None
            _TOKEN["value"] = None
        return False


def credentials() -> tuple[str, str] | None:
    if _OVERRIDE["cred"]:
        return _OVERRIDE["cred"]
    cid, sec = os.environ.get("SH_CLIENT_ID", "").strip(), os.environ.get("SH_CLIENT_SECRET", "").strip()
    return (cid, sec) if cid and sec else None


QUOTA_RE = re.compile(r"quota|processing units? (limit|exhausted|depleted)|insufficient", re.I)
RATE_RE = re.compile(r"per (minute|second)|rate", re.I)


def _is_quota(status: int, text: str) -> bool:
    """Account quota used up (stop and say so), as opposed to a per-minute rate limit (wait and retry)."""
    t = text or ""
    return status in (402, 403, 429) and bool(QUOTA_RE.search(t)) and not RATE_RE.search(t)


def available() -> bool:
    return credentials() is not None


def _token(session) -> str:
    with _TOKEN_LOCK:
        if _TOKEN["value"] and time.time() < _TOKEN["expires"] - 60:
            return _TOKEN["value"]
        cred = credentials()
        if not cred:
            raise ShubUnavailable("SH_CLIENT_ID / SH_CLIENT_SECRET are not set in .env")
        r = session.post(TOKEN_URL, data={"grant_type": "client_credentials", "client_id": cred[0],
                                          "client_secret": cred[1]}, timeout=30)
        if r.status_code != 200:
            who = "the keys you entered" if _OVERRIDE["cred"] else "the server's keys"
            raise ShubUnavailable(f"Sentinel Hub login failed with {who} ({r.status_code}); check the client ID and secret")
        js = r.json()
        _TOKEN["value"] = js["access_token"]
        _TOKEN["expires"] = time.time() + float(js.get("expires_in", 600))
        return _TOKEN["value"]


def _session():
    import requests

    s = requests.Session()
    s.headers["User-Agent"] = "FloodSight/1.0 (hackathon prototype)"
    return s


def _post(session, body: dict, what: str) -> tuple[bytes, float]:
    """One Process API request with retries for rate limits and temporary server errors."""
    for attempt in range(len(WAITS) + 1):
        with SLOTS:
            tok = _token(session)
            try:
                r = session.post(PROCESS_URL, json=body, timeout=300,
                                 headers={"Authorization": f"Bearer {tok}", "Accept": "image/tiff"})
            except Exception as e:  # noqa: BLE001 - network hiccup
                r, err = None, e
        if r is not None and r.status_code == 200:
            pu = float(r.headers.get("x-processingunits-spent", 0) or 0)
            with _TOKEN_LOCK:
                USAGE["pu"] += pu
                USAGE["requests"] += 1
            activity.add(requests=1, mb=len(r.content) / 1e6, pu=pu)
            return r.content, pu
        if r is not None and r.status_code == 401:
            _TOKEN["value"] = None                       # token expired: log in again
        if r is not None and _is_quota(r.status_code, r.text):
            raise ShubQuotaExceeded("The Copernicus account used for this run has reached its processing limit "
                                    f"(Sentinel Hub {r.status_code}).")
        transient = r is None or r.status_code in (401, 429, 500, 502, 503, 504)
        if not transient or attempt == len(WAITS):
            detail = f"{r.status_code}: {r.text[:300]}" if r is not None else str(err)
            raise ShubUnavailable(f"Sentinel Hub request for {what} failed ({detail})")
        wait = WAITS[attempt]
        ra = r.headers.get("Retry-After") if r is not None else None
        if ra:
            try:
                v = float(ra)
                wait = max(wait, v / 1000.0 if v > 100 else v)   # Sentinel Hub sends milliseconds
            except ValueError:
                pass
        log.info("Sentinel Hub busy for %s (%s); retrying in %.0fs", what,
                 r.status_code if r is not None else type(err).__name__, wait)
        activity.wait(wait, f"Sentinel Hub busy ({r.status_code if r is not None else type(err).__name__})")
        time.sleep(wait)
    raise ShubUnavailable(what)


def _tiles(grid: Grid):
    """Pixel windows (r0, r1, c0, c1) covering the grid, each within the Process API size limit."""
    H, W = grid.height, grid.width
    step_r = H if H <= MAX_PX else TILE_PX
    step_c = W if W <= MAX_PX else TILE_PX
    for r0 in range(0, H, step_r):
        for c0 in range(0, W, step_c):
            yield r0, min(r0 + step_r, H), c0, min(c0 + step_c, W)


def _request_grid(grid: Grid, data: list, evalscript: str, nbands: int, what: str) -> tuple[np.ndarray, float]:
    """Run the evalscript over the whole analysis grid (tiled if large); returns (nbands, H, W) float32,
    pixel-aligned with the grid (the request asks for exactly the grid's bounds and pixel count), and the
    processing units this request used (its own, not other requests running at the same time)."""
    import rasterio
    from concurrent.futures import ThreadPoolExecutor

    epsg = int(str(grid.crs).split(":")[1])
    t = grid.transform
    out = np.full((nbands, grid.height, grid.width), np.nan, np.float32)
    session = _session()

    def one(win):
        r0, r1, c0, c1 = win
        x0, y1 = t * (c0, r0)
        x1, y0 = t * (c1, r1)
        body = {
            "input": {"bounds": {"bbox": [x0, y0, x1, y1],
                                 "properties": {"crs": f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"}},
                      "data": data},
            "output": {"width": c1 - c0, "height": r1 - r0,
                       "responses": [{"identifier": "default", "format": {"type": "image/tiff"}}]},
            "evalscript": evalscript,
        }
        raw, pu = _post(session, body, what)
        spent.append(pu)
        with rasterio.MemoryFile(io.BytesIO(raw)) as mf, mf.open() as src:
            arr = src.read().astype(np.float32)
        out[:, r0:r1, c0:c1] = arr[:nbands, : r1 - r0, : c1 - c0]

    spent = []
    wins = list(_tiles(grid))
    with ThreadPoolExecutor(min(4, len(wins))) as ex:
        list(ex.map(activity.bind(one), wins))
    return out, float(sum(spent))


# ------------------------------------------------------------------------------------------------ Sentinel-1
S1_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["VV", "VH", "dataMask"]}],
          output: {bands: 3, sampleType: "FLOAT32"}};
}
function evaluatePixel(s) { return [s.VV, s.VH, s.dataMask]; }
"""

_SCENE_TIME = re.compile(r"_(\d{8}T\d{6})_(\d{8}T\d{6})_")


def _time_range(acq) -> tuple[str, str]:
    """The pass's own sensing window (from the scene names) or the whole UTC day as a fallback."""
    starts, stops = [], []
    for s in acq.scenes or []:
        m = _SCENE_TIME.search(str(s))
        if m:
            starts.append(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
            stops.append(datetime.strptime(m.group(2), "%Y%m%dT%H%M%S"))
    if starts:
        a, b = min(starts) - timedelta(minutes=2), max(stops) + timedelta(minutes=2)
    else:
        a = datetime.combine(acq.day, datetime.min.time())
        b = a + timedelta(days=1)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return a.replace(tzinfo=timezone.utc).strftime(fmt), b.replace(tzinfo=timezone.utc).strftime(fmt)


def coefficient(cfg: dict) -> str:
    return cfg["sentinel1"].get("shub_coefficient", "SIGMA0_ELLIPSOID")


def cache_tag(cfg: dict) -> str:
    """Cache file tag per calibration, so images made with one calibration are never reused for another."""
    return {"SIGMA0_ELLIPSOID": "_shub-s0", "GAMMA0_ELLIPSOID": "_shub-g0", "GAMMA0_TERRAIN": "_shub-g0t",
            "BETA0": "_shub-b0"}.get(coefficient(cfg), "_shub-" + coefficient(cfg).lower())


def s1_request_data(acq, cfg: dict) -> list:
    s1 = cfg["sentinel1"]
    t0, t1 = _time_range(acq)
    k = int(s1.get("speckle_filter_size", 5)) | 1
    processing = {"backCoeff": coefficient(cfg), "orthorectify": True,
                  "demInstance": "COPERNICUS_30"}
    if k > 1:
        processing["speckleFilter"] = {"type": "LEE", "windowSizeX": k, "windowSizeY": k}
    return [{"type": "sentinel-1-grd",
             "dataFilter": {"timeRange": {"from": t0, "to": t1}, "acquisitionMode": "IW", "polarization": "DV",
                            "orbitDirection": acq.direction.upper(), "resolution": "HIGH"},
             "processing": processing}]


def s1_acquisition(ctx: RunContext, acq, out):
    """Linear VV/VH backscatter of one pass on the analysis grid, saved as a GeoTIFF (NaN = no data)."""
    if out.exists():
        return out
    t0 = time.time()
    log.info("Sentinel Hub: processing %s %s...", acq.track, acq.day)
    arr, pu = _request_grid(ctx.grid, s1_request_data(acq, ctx.cfg), S1_EVALSCRIPT, 3, f"{acq.track} {acq.day}")
    valid = arr[2] > 0
    if valid.mean() < 0.01:
        raise ShubUnavailable(f"Sentinel Hub returned no data for {acq.track} {acq.day}")
    vv = np.where(valid, arr[0], np.nan)
    vh = np.where(valid, arr[1], np.nan)
    tmp = out.with_suffix(".part.tif")
    write_raster(tmp, np.stack([vv, vh]).astype(np.float32), ctx.grid, nodata=np.nan)
    tmp.replace(out)
    log.info("Sentinel Hub: %s %s ready (%.0fs, %.0f PU, %.0f%% of area covered)", acq.track, acq.day,
             time.time() - t0, pu, 100 * valid.mean())
    return out


# ------------------------------------------------------------------------------------------------ Sentinel-2
def s2_evalscript(reducer: str) -> str:
    """Cloud/shadow pixels removed with the scene classification layer (SCL), then per pixel either the
    median of clear observations ('median') or the earliest clear observation ('first'). Reflectance x 10000,
    the same convention as the openEO route.

    Output is UINT16 (digital numbers are integers 0-10000+, so nothing is lost; a median of two values is
    rounded to the nearest DN = 0.00005 reflectance). Sentinel Hub charges 32-bit float output double
    ("multiplication factor of 2", CDSE Processing Unit definition), so this halves the cost of every
    Sentinel-2 request. 0 in all bands = no clear observation, turned back into NaN in s2_composite."""
    pick = ("var v = []; for (var b = 0; b < 4; b++) { var col = clear.map(function (s) { return s[B[b]]; })"
            ".sort(function (a, c) { return a - c; }); var m = Math.floor(col.length / 2); "
            "v.push(col.length % 2 ? col[m] : (col[m - 1] + col[m]) / 2); }"
            if reducer == "median" else
            "var s0 = clear[clear.length - 1]; var v = B.map(function (b) { return s0[b]; });")
    return f"""//VERSION=3
var B = ["B03", "B04", "B08", "B11"];
function setup() {{
  return {{input: [{{bands: ["B03", "B04", "B08", "B11", "SCL", "dataMask"], units: "DN"}}],
          output: {{bands: 4, sampleType: "UINT16"}}, mosaicking: "ORBIT"}};
}}
function evaluatePixel(samples) {{
  var bad = [0, 1, 2, 3, 8, 9, 10, 11];
  var clear = samples.filter(function (s) {{ return s.dataMask === 1 && bad.indexOf(s.SCL) < 0; }});
  if (clear.length === 0) {{ return [0, 0, 0, 0]; }}
  {pick}
  return v;
}}
"""


def s2_nodata_to_nan(arr: np.ndarray) -> np.ndarray:
    """UINT16 response -> float with NaN where no clear observation (all four bands 0) or outside the data."""
    arr = arr.astype(np.float32)
    arr[:, np.all(arr == 0, axis=0)] = np.nan
    return arr


def s2_composite(ctx: RunContext, start, end, reducer: str, out):
    if out.exists():
        return out
    s2 = ctx.cfg["sentinel2"]
    data = [{"type": "sentinel-2-l2a",
             "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                            "maxCloudCoverage": int(s2["max_cloud_pct"]),
                            "mosaickingOrder": "mostRecent"}}]
    t0 = time.time()
    arr, pu = _request_grid(ctx.grid, data, s2_evalscript(reducer), 4, f"Sentinel-2 {out.stem}")
    arr = s2_nodata_to_nan(arr)
    tmp = out.with_suffix(".part.tif")
    write_raster(tmp, arr, ctx.grid, nodata=np.nan)
    tmp.replace(out)
    log.info("Sentinel Hub: Sentinel-2 %s ready (%.0fs, %.0f PU)", out.stem, time.time() - t0, pu)
    return out


S2_PROBE_EVALSCRIPT = """//VERSION=3
function setup() {
  return {input: [{bands: ["SCL", "dataMask"]}], output: {bands: 1, sampleType: "UINT8"}, mosaicking: "ORBIT"};
}
function evaluatePixel(samples) {
  var bad = [0, 1, 2, 3, 8, 9, 10, 11];
  for (var i = 0; i < samples.length; i++) {
    if (samples[i].dataMask === 1 && bad.indexOf(samples[i].SCL) < 0) { return [1]; }
  }
  return [0];
}
"""


def s2_clear_fraction(ctx: RunContext, start, end, res_m: float = 200.0) -> tuple[float, float]:
    """Cheap check before the full optical request: share of the area with at least one cloud-free
    Sentinel-2 view between start and end, from the scene classification layer on a coarse 200 m grid
    (about 1/100 of the full request's cost). Returns (clear fraction, processing units)."""
    s2 = ctx.cfg["sentinel2"]
    coarse = Grid.for_aoi(ctx.aoi, res_m)
    data = [{"type": "sentinel-2-l2a",
             "dataFilter": {"timeRange": {"from": f"{start}T00:00:00Z", "to": f"{end}T23:59:59Z"},
                            "maxCloudCoverage": int(s2["max_cloud_pct"])}}]
    arr, pu = _request_grid(coarse, data, S2_PROBE_EVALSCRIPT, 1, "Sentinel-2 cloud check")
    return float(np.nanmean(arr[0] > 0)), pu
