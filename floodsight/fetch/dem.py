"""Copernicus DEM GLO-30 for the AOI.

Primary source: the public Cloud-Optimised GeoTIFF tiles of the Copernicus DEM (no login, 1°x1° tiles,
read directly over HTTP). Fallback: the COPERNICUS_30 collection on openEO.
"""
from __future__ import annotations

import logging
import math
import time

import numpy as np

from ..core import RunContext, write_raster
from contextlib import nullcontext as _nullctx

from .. import activity

log = logging.getLogger("floodsight")

COG_URL = ("https://copernicus-dem-30m.s3.amazonaws.com/Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM/"
           "Copernicus_DSM_COG_10_{lat}_00_{lon}_00_DEM.tif")


def tile_names(west, south, east, north):
    for la in range(math.floor(south), math.ceil(north)):
        for lo in range(math.floor(west), math.ceil(east)):
            lat = f"{'N' if la >= 0 else 'S'}{abs(la):02d}"
            lon = f"{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            yield COG_URL.format(lat=lat, lon=lon)


def _read_tile(url: str, grid, bounds_ll, retries: int = 3) -> np.ndarray | None:
    """Read only the window of one 1x1 degree COG tile that covers the AOI, then reproject it."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject
    from rasterio.windows import from_bounds

    w, s, e, n = bounds_ll
    for attempt in range(retries):
        try:
            with rasterio.open(url) as src:
                tb = src.bounds
                win_b = (max(w, tb.left), max(s, tb.bottom), min(e, tb.right), min(n, tb.top))
                if win_b[0] >= win_b[2] or win_b[1] >= win_b[3]:
                    return None
                win = from_bounds(*win_b, transform=src.transform).round_offsets().round_lengths()
                arr = src.read(1, window=win, boundless=True, fill_value=np.nan, out_dtype="float32")
                tr = src.window_transform(win)
                nod = src.nodata
                crs = src.crs
            if nod is not None:
                arr[arr == nod] = np.nan
            out = np.full(grid.shape, np.nan, np.float32)
            reproject(arr, out, src_transform=tr, src_crs=crs, src_nodata=np.nan,
                      dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                      resampling=Resampling.bilinear)
            return out
        except rasterio.errors.RasterioIOError as ex:
            if "No such file" in str(ex) or "404" in str(ex) or "does not exist" in str(ex):
                return None                                     # ocean / no tile there
            err = ex
        except Exception as ex:  # noqa: BLE001
            err = ex
        log.info("DEM tile %s: attempt %d failed (%s)", url.rsplit("/", 1)[-1], attempt + 1, type(err).__name__)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"DEM tile unreadable after {retries} tries: {url.rsplit('/', 1)[-1]} ({err})")


def dem_on_grid(grid, bounds_ll) -> np.ndarray:
    """Copernicus DEM GLO-30 resampled onto any grid (public COG tiles; NaN where no tile exists)."""
    import rasterio

    dem = np.full(grid.shape, np.nan, np.float32)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
               GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        for url in tile_names(*bounds_ll):
            tile = _read_tile(url, grid, bounds_ll)
            if tile is not None:
                dem = np.where(np.isfinite(dem), dem, tile)
    return fill_seams(dem)


def fill_seams(dem: np.ndarray, max_px: int = 3) -> np.ndarray:
    """Tiles are resampled one by one, which leaves thin no-data lines along 1-degree tile seams. Fill gaps
    up to max_px wide from the nearest valid pixel; real voids (sea, outside the data) stay NaN."""
    from scipy import ndimage

    bad = ~np.isfinite(dem)
    if not bad.any() or bad.all():
        return dem
    dist, (ir, ic) = ndimage.distance_transform_edt(bad, return_indices=True)
    fill = bad & (dist <= max_px)
    out = dem.copy()
    out[fill] = dem[ir[fill], ic[fill]]
    return out


DEM_SOURCE = "Copernicus DEM (AWS open data)"


def fetch_dem(ctx: RunContext) -> np.ndarray:
    with activity.task(DEM_SOURCE, "Elevation (Copernicus DEM GLO-30)") if not ctx.path("inputs", "dem.tif").exists() \
            else _nullctx():
        return _fetch_dem(ctx)


def _fetch_dem(ctx: RunContext) -> np.ndarray:
    out = ctx.path("inputs", "dem.tif")
    if out.exists():
        from ..core import read_to_grid
        cached = read_to_grid(out, ctx.grid, band=1)   # 2-D (rows, cols), same as a fresh download
        if np.mean(~np.isfinite(cached)) <= 0.001:
            activity.cached(DEM_SOURCE, "Elevation (Copernicus DEM GLO-30)")
            return cached
        log.info("Cached DEM has gaps; downloading it again")

    import rasterio

    grid = ctx.grid
    dem = np.full(grid.shape, np.nan, np.float32)
    a = ctx.aoi.buffered(1.0)
    t0 = time.time()
    failed = []
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
               GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES", VSI_CACHE="TRUE",
               CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        for url in tile_names(*a.as_list()):
            try:
                tile = _read_tile(url, grid, a.as_list())
            except RuntimeError as ex:
                failed.append(str(ex))
                continue
            if tile is not None:
                dem = np.where(np.isfinite(dem), dem, tile)
    dem = fill_seams(dem)
    missing = float(np.mean(~np.isfinite(dem)))
    if failed or missing > 0.001:
        log.warning("Public DEM tiles incomplete (%.1f%% missing%s); trying openEO DEM",
                    100 * missing, "; " + failed[0] if failed else "")
        try:
            alt = _dem_openeo(ctx)
            dem = np.where(np.isfinite(dem), dem, alt)
        except Exception as ex:  # noqa: BLE001
            log.warning("openEO DEM failed too: %s", str(ex)[:120])
    missing = float(np.mean(~np.isfinite(dem)))
    if missing >= 0.999:
        raise RuntimeError("Could not obtain the Copernicus DEM for this area.")
    if missing > 0.001:
        ctx.note(f"Elevation data missing for {missing:.0%} of the area (gaps filled from neighbours).")
    log.info("DEM ready (%.0fs)", time.time() - t0)
    write_raster(out, dem, grid, nodata=np.nan)
    return dem


def _dem_openeo(ctx: RunContext) -> np.ndarray:
    from ..core import read_to_grid
    from .s1 import openeo_connection

    con = openeo_connection()
    a = ctx.aoi
    cube = con.load_collection("COPERNICUS_30", spatial_extent={"west": a.west, "south": a.south,
                                                                "east": a.east, "north": a.north})
    cube = cube.reduce_dimension(dimension="t", reducer="max")
    from .s1 import run_openeo

    tmp = ctx.path("inputs", "dem_openeo.tif")
    run_openeo(lambda: cube.download(str(tmp), format="GTiff"), "DEM")
    return read_to_grid(tmp, ctx.grid, band=1)
