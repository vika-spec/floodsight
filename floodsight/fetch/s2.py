"""Sentinel-2 L2A optical evidence (used when the sky is clear).

Cloud, cloud-shadow, dark/terrain-shadow and snow pixels are removed with the scene classification layer (SCL)
before compositing (SCL 0,1,2,3,8,9,10,11: shadows look like water and snow has a water-like index):
'before' = median of clear pixels in the pre window, 'after' = earliest clear pixel after the event.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np

from ..core import RunContext, event_start, read_to_grid
from .. import activity

log = logging.getLogger("floodsight")

BANDS = ["B03", "B04", "B08", "B11"]   # green, red, NIR, SWIR1


def _composite(ctx: RunContext, start, end, reducer: str, out):
    what = f"Sentinel-2 {'before' if 'pre' in out.stem else 'after'} image ({'median' if reducer == 'median' else 'first clear view'}, {start} to {end})"
    if out.exists():
        activity.cached("Sentinel-2", what)
        return out
    from . import shub
    if shub.available() and ctx.cfg["sentinel2"].get("source", "sentinelhub") == "sentinelhub" and \
            ctx.cfg["sentinel1"].get("backend", "auto") in ("auto", "sentinelhub"):
        try:
            with activity.task("Sentinel Hub (Copernicus)", what):
                return shub.s2_composite(ctx, start, end, reducer, out)
        except Exception as e:  # noqa: BLE001 - fall back to openEO
            if shub.override_active():
                raise                  # own keys: never fall back to the server's own openEO account
            log.warning("Sentinel Hub Sentinel-2 failed (%s); trying openEO", str(e)[:120])
    with activity.task("openEO (Copernicus)", what):
        return _composite_openeo(ctx, start, end, reducer, out)


def _composite_openeo(ctx: RunContext, start, end, reducer: str, out):
    from .s1 import openeo_connection

    con = openeo_connection()
    a = ctx.aoi
    s2 = ctx.cfg["sentinel2"]
    cube = con.load_collection(
        "SENTINEL2_L2A",
        spatial_extent={"west": a.west, "south": a.south, "east": a.east, "north": a.north},
        temporal_extent=[str(start), str(end)], bands=BANDS + ["SCL"], max_cloud_cover=s2["max_cloud_pct"],
    )
    scl = cube.band("SCL")
    bad = (scl == 0) | (scl == 1) | (scl == 2) | (scl == 3) | (scl == 8) | (scl == 9) | (scl == 10) | (scl == 11)
    cube = cube.filter_bands(BANDS).mask(bad)
    cube = cube.reduce_dimension(dimension="t", reducer=reducer)
    cube = cube.resample_spatial(resolution=ctx.grid.res, projection=int(ctx.grid.crs.split(":")[1]))
    from .s1 import run_openeo

    tmp = out.with_suffix(".part.tif")
    try:
        if ctx.aoi.area_km2() > ctx.cfg["sentinel1"].get("sync_max_km2", 400):
            raise RuntimeError("large area: using a batch job")
        run_openeo(lambda: cube.download(str(tmp), format="GTiff"), f"Sentinel-2 {out.stem}",
                   retry_timeouts=False)
    except Exception as e:  # noqa: BLE001 - large/slow requests: retry as a batch job
        log.info("Sentinel-2 sync request failed (%s); retrying as batch job", str(e)[:80])
        run_openeo(lambda: cube.execute_batch(outputfile=str(tmp), out_format="GTiff",
                                              title=f"floodsight {out.stem}"), f"Sentinel-2 {out.stem}")
    tmp.replace(out)
    return out


def fetch_s2(ctx: RunContext) -> dict | None:
    s2 = ctx.cfg["sentinel2"]
    if not s2.get("enabled", True):
        return None
    d = ctx.flood_date
    post_start, post_end = d, d + timedelta(days=s2["post_window_days"])
    from . import shub
    if shub.available() and s2.get("source", "sentinelhub") == "sentinelhub" and \
            not ctx.path("inputs", "s2_post_m2.tif").exists():
        # Cheap cloud check first. Only clearly cloudy cases are skipped (below half the share the analysis
        # needs later); borderline cases still get the full images and the usual decision in the pipeline.
        try:
            with activity.task("Sentinel Hub (Copernicus)", f"Sentinel-2 cloud check after the flood ({post_start} to {post_end})") as t:
                clear, pu = shub.s2_clear_fraction(ctx, post_start, post_end)
                t.detail = f"{clear:.0%} of the area seen without cloud"
            log.info("Sentinel-2 cloud check: %.0f%% of the area has a clear view after the flood (%.1f PU)",
                     100 * clear, pu)
            if clear < 0.5 * s2["min_clear_fraction"]:
                ctx.note(f"Sentinel-2 skipped: only {clear:.0%} of the area had a cloud-free view after the "
                         "flood. Results are radar-only.")
                return None
        except Exception as e:  # noqa: BLE001 - the check is only an optimisation
            log.info("Sentinel-2 cloud check failed (%s); fetching the images anyway", str(e)[:80])
    try:
        e = event_start(ctx.cfg, d)            # before-composite ends before the event began (see --before-date)
        pre_p = _composite(ctx, e - timedelta(days=s2["pre_window_days"]), e - timedelta(days=1), "median",
                           ctx.path("inputs", "s2_pre_m2.tif"))
        post_p = _composite(ctx, post_start, post_end, "first",
                            ctx.path("inputs", "s2_post_m2.tif"))
    except Exception as e:  # noqa: BLE001 - optical is optional evidence
        ctx.note(f"Sentinel-2 not used ({type(e).__name__}: {str(e)[:80]}). Results are radar-only.")
        return None
    pre = read_to_grid(pre_p, ctx.grid) / 10000.0
    post = read_to_grid(post_p, ctx.grid) / 10000.0
    return {"pre": pre, "post": post}


def optical_evidence(pre: np.ndarray, post: np.ndarray) -> dict:
    """Water and fresh-debris evidence from before/after reflectance composites (bands G, R, NIR, SWIR)."""
    eps = 1e-6

    def idx(img):
        g, r, n, s = img
        mndwi = (g - s) / (g + s + eps)
        ndvi = (n - r) / (n + r + eps)
        return mndwi, ndvi

    m0, v0 = idx(pre)
    m1, v1 = idx(post)
    valid_post = np.isfinite(post).all(0)
    valid_both = valid_post & np.isfinite(pre).all(0)
    water_post = valid_post & (m1 > 0.0)
    new_water = valid_both & water_post & (m0 <= 0.0)
    # fresh mud/rock: vegetation stripped and bright bare surface appears
    debris = valid_both & ~water_post & (v0 - v1 > 0.25) & (v1 < 0.25)
    return {"valid": valid_post, "valid_both": valid_both, "water_post": water_post,
            "new_water": new_water, "debris": debris, "mndwi_post": m1, "ndvi_pre": v0, "ndvi_post": v1}
