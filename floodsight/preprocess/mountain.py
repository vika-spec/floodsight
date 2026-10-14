"""Is this mountain terrain? (UNEP-WCMC mountain classes, Kapos et al. 2000)

Debris flows and mud deposits from flash floods need steep catchments; on plains a flood shows up as water.
So the debris class is only allowed in mountain terrain. To avoid inventing a threshold, the standard
global definition is used (UNEP-WCMC / Kapos, Rhind, Edwards, Price & Ravilious 2000, "Developing a map of
the world's mountain forests"; also used by FAO and the Mountain Partnership):

  class 1-3   elevation >= 2500 m
  class 4     1500-2500 m and slope >= 2 deg
  class 5     1000-1500 m and (slope >= 5 deg or local elevation range > 300 m)
  class 6      300-1000 m and local elevation range > 300 m
  (class 7, isolated inner basins < 25 km2 enclosed by mountains, is not computed)

Local elevation range (LER) = max - min elevation within a 7 km radius. Elevation and slope are taken from
the DEM averaged to ~1 km, as in the original (1 km GTOPO30). The DEM context is read with an 8 km margin
around the area so LER is not cut off at the box edge; if that read fails the run's own DEM is used and a
note says so.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger("floodsight")

COARSE_M = 250.0
LER_RADIUS_M = 7000.0
MARGIN_KM = 8.0


def kapos_classes(dem: np.ndarray, px_m: float) -> np.ndarray:
    """Mountain class 0 (not mountain) .. 6 for a DEM on a metric grid of pixel size px_m."""
    z = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0.0).astype(np.float64)
    k = max(1, int(round(1000.0 / px_m)))                 # ~1 km averaging window
    z1 = ndimage.uniform_filter(z, k) if k > 1 else z
    gy, gx = np.gradient(z1, px_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    r = max(1, int(round(LER_RADIUS_M / px_m)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    disk = (xx * xx + yy * yy) <= r * r
    ler = ndimage.maximum_filter(z, footprint=disk) - ndimage.minimum_filter(z, footprint=disk)
    e = z1
    cls = np.zeros(z.shape, np.uint8)
    cls[(e >= 300) & (e < 1000) & (ler > 300)] = 6
    cls[(e >= 1000) & (e < 1500) & ((slope >= 5) | (ler > 300))] = 5
    cls[(e >= 1500) & (e < 2500) & (slope >= 2)] = 4
    cls[(e >= 2500) & (e < 3500)] = 3
    cls[(e >= 3500) & (e < 4500)] = 2
    cls[e >= 4500] = 1
    return cls


def kapos_on_grid(dem: np.ndarray, px_m: float) -> np.ndarray:
    """Kapos classes for a fine DEM: computed at ~250 m (as the definition intends) and mapped back."""
    f = max(1, int(round(COARSE_M / px_m)))
    if f == 1:
        return kapos_classes(dem, px_m)
    H, W = dem.shape
    z = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0.0)
    h, w = -(-H // f), -(-W // f)
    zp = np.pad(z, ((0, h * f - H), (0, w * f - W)), mode="edge")
    coarse = zp.reshape(h, f, w, f).mean(axis=(1, 3))
    cls = kapos_classes(coarse, px_m * f)
    return np.repeat(np.repeat(cls, f, 0), f, 1)[:H, :W]


def mountain_mask(ctx, dem_fine: np.ndarray) -> tuple[np.ndarray, dict]:
    """Boolean mask on the run grid (True = mountain terrain) + info for results.json."""
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from ..core import Grid

    source = "DEM with 8 km margin (250 m)"
    try:
        big = ctx.aoi.buffered(MARGIN_KM)
        cgrid = Grid.for_aoi(big, COARSE_M)
        path = ctx.path("inputs", "dem_context_250m.tif")
        if path.exists():
            from ..core import read_to_grid
            coarse = read_to_grid(path, cgrid, band=1)
        else:
            coarse = _read_coarse(cgrid, big)
            from ..core import write_raster
            write_raster(path, coarse, cgrid, nodata=np.nan)
        if np.mean(~np.isfinite(coarse)) > 0.2:
            raise RuntimeError("context DEM incomplete")
        ccls = kapos_classes(coarse, COARSE_M)
        out = np.zeros(ctx.grid.shape, np.uint8)
        reproject(ccls, out, src_transform=cgrid.transform, src_crs=cgrid.crs, dst_transform=ctx.grid.transform,
                  dst_crs=ctx.grid.crs, resampling=Resampling.nearest)
    except Exception as e:  # noqa: BLE001 - fall back to the run's own DEM (LER may be underestimated at edges)
        log.info("Mountain context DEM unavailable (%s); using the run DEM", e)
        source = "run DEM only (no margin: local relief may be underestimated near the edges)"
        out = kapos_on_grid(dem_fine, ctx.grid.res)
    mask = out > 0
    info = {"definition": "UNEP-WCMC mountain classes (Kapos et al. 2000)", "source": source,
            "mountain_fraction": round(float(mask.mean()), 3)}
    return mask, info


def _read_coarse(cgrid, aoi) -> np.ndarray:
    import rasterio

    from ..fetch.dem import _read_tile, tile_names

    dem = np.full(cgrid.shape, np.nan, np.float32)
    env = dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="2",
               VSI_CACHE="TRUE", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif")
    with rasterio.Env(**env):
        for url in tile_names(*aoi.as_list()):
            tile = _read_tile(url, cgrid, aoi.as_list())
            if tile is not None:
                dem = np.where(np.isfinite(dem), dem, tile)
    return dem
