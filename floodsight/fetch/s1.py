"""Sentinel-1 backscatter for the AOI.

Three backends produce the same output (backscatter VV/VH in dB on the analysis grid):
  * sentinelhub – Level-1 GRD processed by Sentinel Hub on Copernicus Data Space (floodsight.fetch.shub):
              terrain-flattened gamma0, Copernicus 30 m DEM, Lee filter. Seconds per image. Needs
              SH_CLIENT_ID / SH_CLIENT_SECRET in .env. First choice when configured.
  * openeo  – Level-1 GRD processed on Copernicus Data Space servers with the processing chain defined
              here (radiometric calibration to sigma0, orthorectification with the Copernicus 30 m DEM),
              cropped to the AOI. Fast: only the valley is transferred. CDSE currently offers only the
              'sigma0-ellipsoid' coefficient; this matches how the Kuro Siwo training data was processed.
  * local   – Level-1 GRD downloaded and processed on your machine (calibration, geocoding,
              terrain normalisation) by floodsight.fetch.s1_local. Slow but independent of openEO.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np

from ..core import RunContext, read_to_grid
from .catalog import Acquisition, Pair
from .. import activity

log = logging.getLogger("floodsight")

OPENEO_URL = "openeo.dataspace.copernicus.eu"
_CONN = None
_CONN_LOCK = threading.Lock()
# CDSE answers "429 Too Many Requests" above ~2 concurrent requests per user: queue ours instead.
OPENEO_SLOTS = threading.BoundedSemaphore(2)


TRANSIENT = ("429", "502", "503", "504", "no available server", "Gateway Timeout", "Too Many Requests",
             "IncompleteRead", "Connection broken", "Connection aborted", "ConnectionError", "Read timed out",
             "RemoteDisconnected")
WAITS = (15, 45, 90, 180)        # ~5.5 minutes of patience for a busy server before giving up


def _is_transient(e: Exception, timeouts_too: bool = True) -> bool:
    msg = f"{type(e).__name__} {e}"
    if not timeouts_too and ("504" in msg or "Gateway Timeout" in msg):
        return False
    return any(t in msg for t in TRANSIENT)


def run_openeo(fn, what: str, retry_timeouts: bool = True):
    """Run one openEO request inside a concurrency slot. Temporary server problems (overload, outages,
    dropped connections) are retried with growing waits instead of failing the whole run.
    retry_timeouts=False for synchronous requests: a 504 there means 'too big for sync', so the caller
    should switch to a batch job rather than wait and repeat the same request."""
    for attempt in range(len(WAITS) + 1):
        with OPENEO_SLOTS:
            try:
                return fn()
            except Exception as e:  # noqa: BLE001
                if attempt == len(WAITS) or not _is_transient(e, retry_timeouts):
                    raise
                wait = WAITS[attempt]
                log.info("openEO temporarily unavailable for %s (%s); retrying in %ds",
                         what, str(e).strip()[:60], wait)
        time.sleep(wait)


def openeo_connection():
    """One authenticated connection per process. First use opens a browser/device login; the refresh
    token is cached by the openeo client so later runs (and judging day) need no interaction."""
    global _CONN
    with _CONN_LOCK:
        return _connect()


def _connect():
    global _CONN
    if _CONN is None:
        import openeo

        try:
            con = openeo.connect(OPENEO_URL, auto_validate=False)   # skip the extra pre-check request
        except TypeError:
            con = openeo.connect(OPENEO_URL)
        from .shub import credentials
        cred = credentials()
        try:
            if cred:   # the CDSE OAuth client works for openEO too: no browser login needed
                con.authenticate_oidc_client_credentials(client_id=cred[0], client_secret=cred[1], provider_id="CDSE")
            else:
                con.authenticate_oidc_client_credentials()   # if OPENEO_AUTH_* env vars are set
        except Exception:
            con.authenticate_oidc()
        _CONN = con
    return _CONN


def _to_db(lin: np.ndarray) -> np.ndarray:
    lin = np.where(np.isfinite(lin) & (lin > 0), lin, np.nan)
    return (10.0 * np.log10(np.clip(lin, 1e-5, None))).astype(np.float32)


def _openeo_acquisition(ctx: RunContext, acq: Acquisition, out: Path) -> Path:
    if out.exists():
        return out
    con = openeo_connection()
    aoi = ctx.aoi
    extent = {"west": aoi.west, "south": aoi.south, "east": aoi.east, "north": aoi.north}
    temporal = [str(acq.day), str(acq.day + timedelta(days=1))]
    orbit_state = acq.direction.upper()          # CDSE uses "ASCENDING" / "DESCENDING"

    def build(with_props: bool, coefficient: str):
        kw = {}
        if with_props:
            kw["properties"] = {"sat:orbit_state": lambda v: v == orbit_state}
        cube = con.load_collection("SENTINEL1_GRD", spatial_extent=extent, temporal_extent=temporal,
                                   bands=ctx.cfg["sentinel1"]["polarisations"], **kw)
        cube = cube.sar_backscatter(coefficient=coefficient, elevation_model="COPERNICUS_30")
        cube = cube.reduce_dimension(dimension="t", reducer="mean")
        return cube.resample_spatial(resolution=ctx.grid.res, projection=int(ctx.grid.crs.split(":")[1]))

    last = None
    coef = ctx.cfg["sentinel1"].get("coefficient", "sigma0-ellipsoid")
    # Synchronous requests are cut off after ~5 min by the server; large areas go straight to batch jobs.
    if ctx.aoi.area_km2() > ctx.cfg["sentinel1"].get("sync_max_km2", 400):
        attempts = [(True, coef, "batch"), (False, coef, "batch")]
    else:
        attempts = [(True, coef, "sync"), (True, coef, "batch"), (False, coef, "batch")]
    for with_props, coefficient, mode in attempts:
        try:
            cube = build(with_props, coefficient)
            tmp = out.with_suffix(".part.tif")
            t0 = time.time()
            log.info("openEO: processing %s %s (%s)...", acq.track, acq.day, mode)
            if mode == "sync":
                run_openeo(lambda: cube.download(str(tmp), format="GTiff"), f"{acq.day}", retry_timeouts=False)
            else:
                run_openeo(lambda: cube.execute_batch(outputfile=str(tmp), out_format="GTiff",
                                                      title=f"floodsight {out.stem}"), f"{acq.day}")
            tmp.replace(out)
            log.info("openEO: %s %s ready (%.0fs)", acq.track, acq.day, time.time() - t0)
            return out
        except Exception as e:  # noqa: BLE001 - backend errors are heterogeneous
            last = e
            log.warning("openEO %s %s (props=%s) failed for %s: %s", mode, coefficient, with_props, acq.day, e)
    raise RuntimeError(f"openEO could not process {acq.day}: {last}")


def _backend_order(requested: str) -> list[str]:
    from . import shub

    if requested == "auto":
        return (["sentinelhub"] if shub.available() else []) + ["openeo", "local"]
    return [requested]


SUFFIX = {"openeo": "", "local": "_local"}     # Sentinel Hub: tag depends on its calibration (shub.cache_tag)


def stack_reference(lin_list: list, min_obs: int = 2) -> np.ndarray:
    """Multi-image 'before' reference in dB: per pixel the temporal MEDIAN backscatter of all before-images of
    the same track, as in Handwerger et al. (2022, NHESS 22, 753) for Sentinel-1 landslide mapping. Combining N
    images lowers the radar's grain (speckle), so a real change after the flood stands out more clearly; the
    median (rather than the mean) is not pulled by one unusual image (e.g. a rain-soaked day). Where fewer than
    min_obs images see a pixel, the latest before-image (lin_list[0]) is used unchanged."""
    arr = np.stack([np.asarray(a, np.float64) for a in lin_list])          # (n, bands, H, W), linear
    n_ok = np.isfinite(arr).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.nanmedian(np.where(np.isfinite(arr), arr, np.nan), axis=0)
    ref = np.where(n_ok >= min_obs, med, arr[0])
    return np.stack([_to_db(b) for b in ref]).astype(np.float32)


BACKEND_NAME = {"sentinelhub": "Sentinel Hub (Copernicus)", "openeo": "openEO (Copernicus)",
                "local": "Copernicus raw GRD + own processing"}


def fetch_pair(ctx: RunContext, pair: Pair, dem_ready=None) -> dict:
    """Returns {'pre': (2,H,W) dB, 'post': (2,H,W) dB, 'backend': str} on the analysis grid.
    dem_ready: optional future; the local fallback waits for the DEM, the server backends do not need it.
    The two images of a pair always come from the SAME backend: backends differ slightly in radiometry and
    geometry, so mixing them would show up as false change. Each backend has its own cache file names."""
    track = pair.pre.track
    order = _backend_order(ctx.cfg["sentinel1"].get("backend", "auto"))
    from .shub import override_active
    if override_active():            # a user's own keys: only their Sentinel Hub account is used for this run
        order = ["sentinelhub"]

    def paths_for(b):
        from .shub import cache_tag
        if b == "sentinelhub":
            tag = cache_tag(ctx.cfg)
        elif b == "local":                     # cache name tied to the calibration, like Sentinel Hub
            tag = "_local" if ctx.cfg["sentinel1"].get("local_calibration", "sigma0") == "gamma0" else "_local-s0"
            if ctx.cfg["sentinel1"].get("local_thermal_noise", True):
                tag += "n"                     # noise-removed images are a different product: own cache files
        else:
            tag = SUFFIX[b]
        return {k: ctx.path("inputs", f"s1_{track}_{k}_{getattr(pair, k).day}{tag}.tif") for k in ("pre", "post")}

    cached = [b for b in order if all(p.exists() for p in paths_for(b).values())]
    if cached:                                   # an earlier run already has both images from one backend
        order = [cached[0]] + [b for b in order if b != cached[0]]

    def label(k, acq=None):
        acq = acq or getattr(pair, k)
        return f"Sentinel-1 {'before' if k == 'pre' else 'after'} image, track {track}, {acq.day}"

    def tracked(b, fn, k):
        def run(*a):
            with activity.task(BACKEND_NAME.get(b, b), label(k)):
                return fn(*a)
        return run

    used, errors = None, []
    for b in order:
        paths = paths_for(b)
        if all(p.exists() for p in paths.values()):
            used = b
            for k in paths:
                activity.cached(BACKEND_NAME.get(b, b), label(k))
            break
        try:
            if b == "sentinelhub":
                from .shub import s1_acquisition
                with ThreadPoolExecutor(2) as ex:
                    for f in [ex.submit(tracked(b, s1_acquisition, k), ctx, getattr(pair, k), paths[k]) for k in paths]:
                        f.result()
            elif b == "openeo":
                with ThreadPoolExecutor(2) as ex:   # pre and post processed concurrently on the server
                    for f in [ex.submit(tracked(b, _openeo_acquisition, k), ctx, getattr(pair, k), paths[k]) for k in paths]:
                        f.result()
            else:
                from .s1_local import process_acquisition
                if dem_ready is not None:
                    dem_ready.result()
                for k in paths:
                    if not paths[k].exists():
                        tracked(b, process_acquisition, k)(ctx, getattr(pair, k), paths[k])
            used = b
            break
        except Exception as e:  # noqa: BLE001 - try the next backend
            from .shub import ShubQuotaExceeded
            if isinstance(e, ShubQuotaExceeded) and len(order) == 1:
                raise
            errors.append(f"{b}: {str(e)[:160]}")
            log.warning("%s failed for track %s: %s", b, track, str(e)[:200])
            if len(order) == 1:
                raise
    if used is None:
        raise RuntimeError("No backend could process track " + track + " (" + "; ".join(errors) + ")")
    if errors:
        ctx.note(f"Track {track}: {', '.join(e.split(':')[0] for e in errors)} unavailable; "
                 f"both images of that track processed with {used}.")

    out = {"backend": used}
    lins = {}
    for k, p in paths.items():
        lin = read_to_grid(p, ctx.grid)
        lins[k] = lin[:2]
        out[k] = np.stack([_to_db(b) for b in lin[:2]])
    out["pre_stack_dates"] = [str(pair.pre.day)]
    extras = list(getattr(pair, "extra_pres", []) or [])
    if extras and used != "sentinelhub":
        ctx.note(f"Track {track}: multi-image before-reference needs Sentinel Hub; the single latest "
                 "before-image is used.")
    elif extras:
        from .shub import cache_tag, s1_acquisition
        tag = cache_tag(ctx.cfg)
        more = []
        def one(acq):
            pth = ctx.path("inputs", f"s1_{track}_pre_{acq.day}{tag}.tif")
            if pth.exists():
                activity.cached(BACKEND_NAME["sentinelhub"], label("pre", acq) + " (extra before-image)")
            else:
                with activity.task(BACKEND_NAME["sentinelhub"], label("pre", acq) + " (extra before-image)"):
                    s1_acquisition(ctx, acq, pth)
            return acq.day, read_to_grid(pth, ctx.grid)[:2]
        with ThreadPoolExecutor(3) as ex:
            for f in [ex.submit(one, a) for a in extras]:
                try:
                    more.append(f.result())
                except Exception as e:  # noqa: BLE001 - fewer images is still fine
                    log.warning("Extra before-image skipped on %s: %s", track, str(e)[:160])
        if more:
            out["pre_ref"] = stack_reference([lins["pre"]] + [m[1] for m in more])
            out["pre_stack_dates"] += [str(m[0]) for m in more]
            log.info("Track %s: before-reference = median of %d images (%s)", track, len(out["pre_stack_dates"]),
                     ", ".join(out["pre_stack_dates"]))
    return out
