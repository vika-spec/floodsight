"""OpenStreetMap as it was BEFORE the event (full-history OSM, queried at a past date).

Snapshot date = min(day before the flood, latest date ohsome serves). Post-event edits (which often come
from mappers tracing damage) can therefore never leak in.

Sources, tried in order until one works (all return OSM *as of the snapshot date*):
  1. ohsome API v2  – needs a free key (OHSOME_API_KEY in .env, sign up at account.heigit.org). Returns GeoParquet.
  2. ohsome API v1  – no key; being retired (shut down 30 Nov 2026) and may refuse requests (HTTP 403).
  3. Overpass API   – OFF by default (osm.overpass_fallback: false). The hackathon brief names the ohsome API as
     the OSM source, so only ohsome is used unless this is switched on deliberately (e.g. for offline testing).
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
from datetime import date

import requests

from ..core import AOI, RunContext, event_start, osm_snapshot_date
from .. import activity

log = logging.getLogger("floodsight")

UA = {"User-Agent": "FloodSight/1.0 (hackathon flood-mapping prototype; contact via GitHub)"}
OHSOME_V2 = "https://api.heigit.org/ohsome-api/v2-rc"
OVERPASS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]

QUERIES = {
    "buildings": "building=* and building!=no and geometry:polygon",
    "roads": "highway=* and type:way",
    "places": "place in (city, town, village, hamlet) and (type:node or type:way or type:relation)",
    "health": ("(amenity in (hospital, clinic, doctors) or healthcare in (hospital, centre, clinic)) "
               "and (type:node or type:way)"),
    # used by the flood-path tracer along the valley
    "bridges": "highway=* and bridge=yes and type:way",
    "rivers": "waterway=river and type:way",
    # helicopter access for cut-off settlements (analysis/landing.py)
    "helipads": "aeroway in (helipad, heliport)",
    # overhead lines: candidate landing ground is kept away from them
    "power_lines": "power in (line, minor_line) and type:way",
}
OVERPASS_QL = {
    "buildings": ['way["building"]["building"!="no"]'],
    "roads": ['way["highway"]'],
    "places": ['node["place"~"^(city|town|village|hamlet)$"]', 'way["place"~"^(city|town|village|hamlet)$"]'],
    "health": ['node["amenity"~"^(hospital|clinic|doctors)$"]', 'way["amenity"~"^(hospital|clinic|doctors)$"]',
               'node["healthcare"~"^(hospital|centre|clinic)$"]', 'way["healthcare"~"^(hospital|centre|clinic)$"]'],
    "bridges": ['way["highway"]["bridge"="yes"]'],
    "rivers": ['way["waterway"="river"]'],
    "helipads": ['node["aeroway"~"^(helipad|heliport)$"]', 'way["aeroway"~"^(helipad|heliport)$"]'],
    "power_lines": ['way["power"~"^(line|minor_line)$"]'],
}
KEEP_TAGS = ["name", "name:en", "ref", "highway", "bridge", "building", "place", "amenity", "healthcare", "waterway", "aeroway", "power"]


class OSMUnavailable(RuntimeError):
    pass


def latest_available(base_url: str, configured: str) -> str:
    try:
        r = requests.get(f"{base_url}/metadata", timeout=20, headers=UA)
        r.raise_for_status()
        ts = r.json()["extractRegion"]["temporalExtent"]["toTimestamp"][:10]
        return min(ts, configured) if configured else ts
    except Exception:  # noqa: BLE001 - fall back to the configured value from the brief
        return configured


# ---------------------------------------------------------------------------------------------- normalise

def _normalise(gdf):
    """Same columns whatever the source: geometry + KEEP_TAGS (strings or None), EPSG:4326."""
    import geopandas as gpd
    import pandas as pd

    if gdf is None or len(gdf) == 0:
        return gpd.GeoDataFrame({k: [] for k in KEEP_TAGS}, geometry=[], crs=4326)
    gdf = gdf.copy()
    if not isinstance(gdf, gpd.GeoDataFrame) or gdf.geometry.name != "geometry":
        gdf = _with_geometry_column(gdf)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    # v2 may deliver tags as one map/JSON column
    for col in ("tags", "properties"):
        if col in gdf.columns:
            def get(d, k):
                if isinstance(d, str):
                    try:
                        d = json.loads(d)
                    except ValueError:
                        return None
                if isinstance(d, dict):
                    return d.get(k)
                if isinstance(d, (list, tuple)):          # parquet map -> list of (key, value)
                    return dict(d).get(k)
                return None
            for k in KEEP_TAGS:
                if k not in gdf.columns:
                    gdf[k] = gdf[col].map(lambda d, k=k: get(d, k))
    for k in KEEP_TAGS:
        if k not in gdf.columns:
            gdf[k] = None
    out = gdf[KEEP_TAGS + ["geometry"]].copy()
    out = out[out.geometry.notna() & ~out.geometry.is_empty]
    if "name" in out and "name:en" in out:
        out["name"] = out["name"].where(out["name"].notna(), out["name:en"])
    for k in KEEP_TAGS:
        out[k] = out[k].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return gpd.GeoDataFrame(out, geometry="geometry", crs=4326).reset_index(drop=True)


def _with_geometry_column(df):
    """Whatever the source calls its geometry column (geometry, geom, wkb_geometry, ...), rename it to
    'geometry'. Handles both real GeoDataFrames and plain tables with WKB/WKT geometry."""
    import geopandas as gpd
    import shapely

    if isinstance(df, gpd.GeoDataFrame):
        try:
            name = df.geometry.name
            return df.rename_geometry("geometry") if name != "geometry" else df
        except AttributeError:          # no active geometry set
            pass
    for col in ("geometry", "geom", "wkb_geometry", "the_geom", "wkt", "shape"):
        if col in df.columns:
            vals = df[col]
            first = next((v for v in vals if v is not None), None)
            if isinstance(first, (bytes, bytearray)):
                geoms = shapely.from_wkb(vals)
            elif isinstance(first, str):
                geoms = shapely.from_wkt(vals)
            else:
                geoms = vals
            out = gpd.GeoDataFrame(df.drop(columns=[col]), geometry=list(geoms),
                                   crs=getattr(df, "crs", None) or 4326)
            return out
    raise ValueError(f"no geometry column found (columns: {list(df.columns)[:15]})")


# ---------------------------------------------------------------------------------------------- sources

def _ohsome_v2(aoi: AOI, snap: date, flt: str, key: str):
    import geopandas as gpd

    hdr = {**UA, "Authorization": key}
    body = {"filter": flt, "aoi": [round(v, 6) for v in aoi.as_list()], "time": snap.isoformat()}
    attempts = [
        ("POST", f"{OHSOME_V2}/extraction/features", {"json": body}),
        ("POST", f"{OHSOME_V2}/extraction/features.parquet", {"json": body}),
        ("GET", f"{OHSOME_V2}/extraction/features.parquet",
         {"params": {"filter": flt, "aoi": ",".join(str(v) for v in body["aoi"]), "time": body["time"]}}),
    ]
    last = None
    for method, url, kw in attempts:
        r = requests.request(method, url, headers=hdr, timeout=300, **kw)
        if r.status_code in (404, 405):
            last = f"{r.status_code} at {url}"
            continue
        for wait in OHSOME_WAITS:                   # rate limit of the free service: wait and try again
            if r.status_code != 429:
                break
            try:
                wait = max(wait, float(r.headers.get("Retry-After", 0)))
            except ValueError:
                pass
            activity.wait(wait, "OpenStreetMap service busy (429)")
            time.sleep(wait)
            r = requests.request(method, url, headers=hdr, timeout=300, **kw)
        r.raise_for_status()
        buf = io.BytesIO(r.content)
        try:
            gdf = gpd.read_parquet(buf)
        except Exception:  # noqa: BLE001 - parquet without GeoParquet metadata
            import pandas as pd
            buf.seek(0)
            gdf = pd.read_parquet(buf)
        log.info("ohsome v2: %d rows, columns %s", len(gdf), list(gdf.columns)[:12])
        return gdf
    raise RuntimeError(f"ohsome v2 extraction endpoint not found ({last})")


def _ohsome_v1(base_url: str, aoi: AOI, snap: date, flt: str):
    import geopandas as gpd

    data = {"bboxes": ",".join(f"{v:.6f}" for v in aoi.as_list()), "time": snap.isoformat(),
            "filter": flt, "properties": "tags"}
    last = None
    for method in ("POST", "GET"):
        kw = {"data": data} if method == "POST" else {"params": data}
        r = requests.request(method, f"{base_url}/elements/geometry", headers=UA, timeout=300, **kw)
        if r.ok:
            fc = r.json()
            return gpd.GeoDataFrame.from_features(fc.get("features", []), crs=4326) if fc.get("features") \
                else gpd.GeoDataFrame(geometry=[], crs=4326)
        last = r.status_code
    raise RuntimeError(f"ohsome v1 refused the request (HTTP {last})")


def _overpass(aoi: AOI, snap: date, key: str):
    import geopandas as gpd
    from shapely.geometry import LineString, Point, Polygon

    w, s, e, n = aoi.as_list()
    bb = f"({s:.6f},{w:.6f},{n:.6f},{e:.6f})"
    q = (f'[out:json][timeout:240][date:"{snap.isoformat()}T00:00:00Z"];('
         + "".join(f"{x}{bb};" for x in OVERPASS_QL[key]) + ");out tags geom;")
    last = None
    for url in OVERPASS:
        try:
            r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
            if r.status_code == 429:
                time.sleep(10)
                r = requests.post(url, data={"data": q}, headers=UA, timeout=300)
            r.raise_for_status()
            els = r.json().get("elements", [])
            break
        except Exception as ex:  # noqa: BLE001
            last = ex
    else:
        raise RuntimeError(f"Overpass failed: {last}")
    rows, geoms = [], []
    for el in els:
        tags = el.get("tags", {})
        if el["type"] == "node":
            if key == "buildings":
                continue
            g = Point(el["lon"], el["lat"])
        elif el["type"] == "way" and el.get("geometry"):
            pts = [(p["lon"], p["lat"]) for p in el["geometry"]]
            closed = len(pts) >= 4 and pts[0] == pts[-1]
            if key == "buildings" or (closed and key in ("places", "health")):
                if not closed:
                    continue
                g = Polygon(pts)
            else:
                if len(pts) < 2:
                    continue
                g = LineString(pts)
        else:
            continue
        rows.append({k: tags.get(k) for k in KEEP_TAGS})
        geoms.append(g)
    return gpd.GeoDataFrame(rows, geometry=geoms, crs=4326) if rows else gpd.GeoDataFrame(geometry=[], crs=4326)


OSM_PARALLEL = 2      # the free ohsome service rate-limits (HTTP 429) at three parallel requests
OHSOME_WAITS = (5, 10, 20, 30)
# needed for damage and cut-off; the others only add detail, so one failing never removes the rest
OSM_REQUIRED = ("buildings", "roads", "places", "health")
OSM_LABEL = {"buildings": "Buildings", "roads": "Roads and bridges", "places": "Settlements",
             "health": "Hospitals and clinics", "bridges": "Bridges", "helipads": "Helipads", "power_lines": "Power lines",
             "rivers": "Rivers"}


def fetch_layer(key: str, aoi: AOI, snap: date, cfg_osm: dict):
    """Try each source in turn; returns (GeoDataFrame, source name)."""
    errors = []
    api_key = os.environ.get("OHSOME_API_KEY", "").strip()
    sources = []
    if api_key:
        sources.append(("ohsome v2", lambda: _ohsome_v2(aoi, snap, QUERIES[key], api_key)))
    sources.append(("ohsome v1", lambda: _ohsome_v1(cfg_osm["ohsome_url"], aoi, snap, QUERIES[key])))
    if cfg_osm.get("overpass_fallback", False):
        sources.append(("Overpass (attic)", lambda: _overpass(aoi, snap, key)))
    for name, fn in sources:
        try:
            return _normalise(fn()), name
        except Exception as e:  # noqa: BLE001
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            log.warning("OSM %s via %s failed: %s", key, name, str(e)[:160])
    raise OSMUnavailable("; ".join(errors))


def fetch_osm(ctx: RunContext) -> dict:
    """Returns GeoDataFrames (EPSG:4326): buildings (AOI), roads/places/health (AOI + buffer)."""
    ocfg = ctx.cfg["osm"]
    snap = osm_snapshot_date(event_start(ctx.cfg, ctx.flood_date), latest_available(ocfg["ohsome_url"], ocfg["latest_snapshot"]))
    big = ctx.aoi.buffered(ocfg["network_buffer_km"])
    out = {"snapshot": snap.isoformat()}
    used = set()

    def one(key):
        path = ctx.path("inputs", f"osm_{key}_{snap}.geojson")
        what = f"{OSM_LABEL.get(key, key)} (map of {snap})"
        if path.exists():
            activity.cached("OpenStreetMap", what)
            return key, load_osm_layer(path), None
        try:
            with activity.task("OpenStreetMap (ohsome)", what) as t:
                gdf, src = fetch_layer(key, ctx.aoi if key == "buildings" else big, snap, ocfg)
                t.source = f"OpenStreetMap ({src})"
                t.detail = f"{len(gdf):,} features"
        except OSMUnavailable:
            if key in OSM_REQUIRED:
                raise
            ctx.note(f"OpenStreetMap {OSM_LABEL.get(key, key).lower()} could not be downloaded; that detail is "
                     "missing from this run (the rest of the analysis is complete).")
            return key, _normalise(None), None   # not cached: the next run tries again
        if len(gdf):
            gdf.to_file(path, driver="GeoJSON")
        else:
            path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
        log.info("OSM %s: %d features (%s)", key, len(gdf), src)
        return key, gdf, src

    # Layers are independent requests: two at a time (polite to the public ohsome service). Same data
    # as one after another, only faster.
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(OSM_PARALLEL) as ex:
        for key, gdf, src in ex.map(one, list(QUERIES)):
            out[key] = gdf
            if src:
                used.add(src)
    out["source"] = ", ".join(sorted(used)) or "cache"
    ctx.note(f"OpenStreetMap snapshot of {snap} (before the event).")
    return out


def empty_osm(snapshot=None) -> dict:
    e = _normalise(None)
    return {"snapshot": snapshot, "buildings": e, "roads": e, "places": e, "health": e, "source": None}


def load_osm_layer(path):
    import geopandas as gpd

    try:
        gdf = gpd.read_file(path)
    except Exception:  # noqa: BLE001 - empty FeatureCollection
        gdf = gpd.GeoDataFrame(geometry=[], crs=4326)
    return _normalise(gdf)
