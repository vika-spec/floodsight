"""Local Sentinel-1 GRD processing (fallback backend; also proves the chain starts from Level-1 data).

Chain:  download only annotation, calibration and measurement files for the frame
     -> read the measurement window covering the AOI (not the whole 250 km swath)
     -> thermal-noise removal with the product's own noise LUTs (range x azimuth, ESA IPF >= 2.9 format; older
        products: range only), as in ESA SNAP "Thermal Noise Removal": DN^2 - noise, negatives set to 0
     -> radiometric calibration to sigma0 using the product's own sigmaNought LUT
     -> geocode with the product's geolocation grid (thin-plate-spline on GCPs)
     -> terrain parallax correction with the Copernicus DEM (shift by (h - h_ref)/tan(theta) toward sensor)
     -> optional terrain normalisation to gamma0 (config sentinel1.local_calibration; default sigma0, the same
        calibration as the Sentinel Hub and openEO routes, so a fallback does not change the model's input)
Not done (documented limitation): precise-orbit range-Doppler geocoding (GCP grid + DEM parallax instead).
"""
from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import requests

from ..core import RunContext, write_raster
from ..preprocess.geometry import local_incidence, sensor_ground_vector
from .catalog import Acquisition, cdse_token

log = logging.getLogger("floodsight")

DL = "https://download.dataspace.copernicus.eu/odata/v1/Products"


# ---------------------------------------------------------------------------
# Download only what we need from the SAFE
# ---------------------------------------------------------------------------

def _nodes(session, url):
    r = session.get(url + "/Nodes", timeout=60)
    r.raise_for_status()
    js = r.json()
    return [n["Name"] for n in js.get("result", js.get("value", []))]


def _download(session, url, dest: Path):
    if dest.exists():
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    with session.get(url + "/$value", stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    tmp.replace(dest)
    return dest


def download_frame(scene, cache: Path, pols=("vv", "vh")) -> dict:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {cdse_token()}"
    root = f"{DL}({scene.id})/Nodes({scene.name})"
    out = {}
    folder = cache / scene.name
    folder.mkdir(parents=True, exist_ok=True)
    meas = _nodes(s, root + "/Nodes(measurement)")
    ann = [n for n in _nodes(s, root + "/Nodes(annotation)") if n.endswith(".xml")]
    cal = _nodes(s, root + "/Nodes(annotation)/Nodes(calibration)")    # calibration-* and noise-* files
    for pol in pols:
        m = next((n for n in meas if f"-{pol}-" in n), None)
        a = next((n for n in ann if f"-{pol}-" in n), None)
        c = next((n for n in cal if n.startswith("calibration") and f"-{pol}-" in n), None)
        nz = next((n for n in cal if n.startswith("noise") and f"-{pol}-" in n), None)
        if not (m and a and c):
            continue
        log.info("Downloading %s %s (large file, local fallback)", scene.name[:32], pol.upper())
        out[pol] = {
            "measurement": _download(s, f"{root}/Nodes(measurement)/Nodes({m})", folder / m),
            "annotation": _download(s, f"{root}/Nodes(annotation)/Nodes({a})", folder / a),
            "calibration": _download(s, f"{root}/Nodes(annotation)/Nodes(calibration)/Nodes({c})", folder / c),
        }
        if nz:
            out[pol]["noise"] = _download(s, f"{root}/Nodes(annotation)/Nodes(calibration)/Nodes({nz})", folder / nz)
    return out


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def parse_annotation(path: Path) -> dict:
    root = ET.parse(path).getroot()
    pts = []
    for g in root.iter("geolocationGridPoint"):
        pts.append([float(g.findtext(k)) for k in
                    ("line", "pixel", "latitude", "longitude", "height", "incidenceAngle")])
    heading = float(root.findtext(".//productInformation/platformHeading"))
    return {"gcps": np.array(pts), "heading": heading}


def parse_calibration(path: Path):
    """sigmaNought LUT as (lines, pixels, values). Vectors are put on one common pixel axis and
    sorted/de-duplicated so interpolation never sees repeated or unordered coordinates."""
    root = ET.parse(path).getroot()
    rows = []
    for v in root.iter("calibrationVector"):
        px = np.array(v.findtext("pixel").split(), dtype=float)
        sg = np.array(v.findtext("sigmaNought").split(), dtype=float)
        n = min(len(px), len(sg))
        px, sg = px[:n], sg[:n]
        px, idx = np.unique(px, return_index=True)
        rows.append((int(v.findtext("line")), px, sg[idx]))
    if not rows:
        raise ValueError(f"no calibration vectors in {path.name}")
    pixels = np.unique(np.concatenate([r[1] for r in rows]))
    by_line = {}
    for line, px, sg in rows:                    # later duplicates of the same line replace earlier ones
        by_line[line] = np.interp(pixels, px, sg)
    lines = np.array(sorted(by_line), float)
    vals = np.vstack([by_line[int(l)] for l in lines])
    return lines, pixels, vals


def parse_noise(path: Path) -> dict:
    """Thermal-noise LUTs. Range: (lines, pixels, values) like the calibration LUT, from noiseRangeVector
    (IPF >= 2.9) or noiseVector (older). Azimuth (IPF >= 2.9 only): list of blocks, each with its line/pixel
    extent and a LUT along the lines; outside every block the azimuth factor is 1."""
    root = ET.parse(path).getroot()
    rows = []
    for tag, lut in (("noiseRangeVector", "noiseRangeLut"), ("noiseVector", "noiseLut")):
        for v in root.iter(tag):
            px = np.array(v.findtext("pixel").split(), dtype=float)
            nv = np.array(v.findtext(lut).split(), dtype=float)
            n = min(len(px), len(nv))
            px, idx = np.unique(px[:n], return_index=True)
            rows.append((int(v.findtext("line")), px, nv[:n][idx]))
        if rows:
            break
    rng = None
    if rows:
        pixels = np.unique(np.concatenate([r[1] for r in rows]))
        by_line = {line: np.interp(pixels, px, nv) for line, px, nv in rows}
        lines = np.array(sorted(by_line), float)
        rng = (lines, pixels, np.vstack([by_line[int(l)] for l in lines]))
    az = []
    for v in root.iter("noiseAzimuthVector"):
        ln = np.array(v.findtext("line").split(), dtype=float)
        lut = np.array(v.findtext("noiseAzimuthLut").split(), dtype=float)
        n = min(len(ln), len(lut))
        az.append({"l0": int(v.findtext("firstAzimuthLine")), "l1": int(v.findtext("lastAzimuthLine")),
                   "p0": int(v.findtext("firstRangeSample")), "p1": int(v.findtext("lastRangeSample")),
                   "line": ln[:n], "lut": lut[:n]})
    return {"range": rng, "azimuth": az}


def noise_lut(noise: dict, r0, r1, c0, c1) -> np.ndarray | None:
    """Noise power for every pixel of the window: range LUT x azimuth LUT (blocks)."""
    if noise is None or noise.get("range") is None:
        return None
    nl, npx, nv = noise["range"]
    N = calibration_lut(nl, npx, nv, r0, r1, c0, c1)
    for b in noise.get("azimuth", []):
        a0, a1 = max(b["l0"], r0), min(b["l1"], r1 - 1)
        q0, q1 = max(b["p0"], c0), min(b["p1"], c1 - 1)
        if a0 > a1 or q0 > q1 or len(b["line"]) == 0:
            continue
        f = np.interp(np.arange(a0, a1 + 1, dtype=float), b["line"], b["lut"]).astype(np.float32)
        N[a0 - r0:a1 - r0 + 1, q0 - c0:q1 - c0 + 1] *= f[:, None]
    return N


def calibration_lut(cl, cp, cv, r0, r1, c0, c1) -> np.ndarray:
    """Separable linear interpolation of the LUT onto the window (fast, low memory, extrapolates)."""
    cols = np.arange(c0, c1, dtype=float)
    rows = np.arange(r0, r1, dtype=float)
    along_px = np.vstack([np.interp(cols, cp, v) for v in cv])            # (n_lines, ncols)
    if len(cl) == 1:
        return np.repeat(along_px, len(rows), axis=0).astype(np.float32)
    i = np.clip(np.searchsorted(cl, rows) - 1, 0, len(cl) - 2)
    t = ((rows - cl[i]) / (cl[i + 1] - cl[i]))[:, None]
    return (along_px[i] * (1 - t) + along_px[i + 1] * t).astype(np.float32)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _aoi_window(gcps: np.ndarray, ctx: RunContext, shape, margin=300):
    from scipy.interpolate import LinearNDInterpolator

    li = LinearNDInterpolator(gcps[:, [3, 2]], gcps[:, [0, 1]])
    a = ctx.aoi
    lon = np.linspace(a.west, a.east, 25)
    lat = np.linspace(a.south, a.north, 25)
    LON, LAT = np.meshgrid(lon, lat)
    lp = li(LON.ravel(), LAT.ravel())
    lp = lp[np.isfinite(lp).all(1)]
    if len(lp) == 0:
        return None
    r0 = int(max(lp[:, 0].min() - margin, 0))
    r1 = int(min(lp[:, 0].max() + margin, shape[0]))
    c0 = int(max(lp[:, 1].min() - margin, 0))
    c1 = int(min(lp[:, 1].max() + margin, shape[1]))
    return r0, r1, c0, c1


def process_frame(ctx: RunContext, files: dict, dem: np.ndarray) -> np.ndarray | None:
    """Returns (2, H, W) linear backscatter (sigma0, or gamma0 if configured) on the analysis grid (NaN = no data)."""
    import rasterio
    from rasterio.control import GroundControlPoint
    from rasterio.warp import Resampling, reproject
    from rasterio.windows import Window
    from scipy.interpolate import griddata
    from scipy.ndimage import map_coordinates

    grid = ctx.grid
    bands = []
    geo = None
    for pol in ("vv", "vh"):
        if pol not in files:
            bands.append(np.full(grid.shape, np.nan, np.float32))
            continue
        f = files[pol]
        ann = parse_annotation(f["annotation"])
        g = ann["gcps"]
        with rasterio.open(f["measurement"]) as src:
            win = _aoi_window(g, ctx, (src.height, src.width))
            if win is None:
                return None
            r0, r1, c0, c1 = win
            dn = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)

        # 1. thermal-noise removal and calibration: sigma0 = max(DN^2 - N, 0) / A_sigma^2
        power = dn ** 2
        if ctx.cfg["sentinel1"].get("local_thermal_noise", True) and f.get("noise"):
            N = noise_lut(parse_noise(f["noise"]), r0, r1, c0, c1)
            if N is not None:
                power = np.maximum(power - N, 0.0)
                del N
        cl, cp, cv = parse_calibration(f["calibration"])
        A = calibration_lut(cl, cp, cv, r0, r1, c0, c1)
        sigma0 = np.where(dn > 0, power / np.maximum(A, 1e-6) ** 2, np.nan).astype(np.float32)
        del A, power

        # 2. geocode using the product's GCP grid (ellipsoid + reference height)
        sel = (g[:, 0] >= r0 - 2000) & (g[:, 0] <= r1 + 2000) & (g[:, 1] >= c0 - 3000) & (g[:, 1] <= c1 + 3000)
        if sel.sum() < 6:
            sel = np.ones(len(g), bool)
        gcps = [GroundControlPoint(row=p[0] - r0, col=p[1] - c0, x=p[3], y=p[2], z=p[4]) for p in g[sel]]
        ell = np.full(grid.shape, np.nan, np.float32)
        reproject(sigma0, ell, gcps=gcps, src_crs="EPSG:4326", dst_transform=grid.transform,
                  dst_crs=grid.crs, resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan,
                  SRC_METHOD="GCP_TPS")

        if geo is None:  # geometry fields shared by both polarisations
            xs, ys = grid.lonlat_to_xy(g[sel, 3], g[sel, 2])
            rows, cols = grid.xy_to_rowcol(np.array(xs), np.array(ys))
            R, C = np.mgrid[0:grid.height, 0:grid.width]
            h_ref = griddata((rows, cols), g[sel, 4], (R, C), method="linear")
            inc = griddata((rows, cols), g[sel, 5], (R, C), method="linear")
            h_ref = np.where(np.isfinite(h_ref), h_ref, np.nanmean(g[sel, 4]))
            inc = np.where(np.isfinite(inc), inc, np.nanmean(g[sel, 5]))
            geo = {"h_ref": h_ref, "inc": inc, "heading": ann["heading"]}

        # 3. terrain parallax: a point at height h appears (h - h_ref)/tan(theta) closer to the sensor
        ux, uy = sensor_ground_vector(geo["heading"])          # horizontal unit vector toward sensor (E, N)
        shift_m = (np.nan_to_num(dem, nan=0.0) - geo["h_ref"]) / np.tan(np.radians(geo["inc"]))
        R, C = np.mgrid[0:grid.height, 0:grid.width].astype(np.float32)
        src_c = C + shift_m * ux / grid.res
        src_r = R - shift_m * uy / grid.res                    # rows grow southward
        corr = map_coordinates(ell, [src_r, src_c], order=1, cval=np.nan)

        # 4. optional terrain normalisation to gamma0 with the local incidence angle
        if ctx.cfg["sentinel1"].get("local_calibration", "sigma0") == "gamma0":
            cos_loc = np.cos(np.radians(local_incidence(dem, grid.res, geo["inc"], geo["heading"])))
            corr = corr / np.clip(cos_loc, 0.15, None)
        bands.append(corr.astype(np.float32))
    return np.stack(bands)


def process_acquisition(ctx: RunContext, acq: Acquisition, out: Path) -> Path:
    import rasterio  # noqa: F401  (fail early if missing)

    from ..core import read_to_grid

    dem = read_to_grid(ctx.path("inputs", "dem.tif"), ctx.grid, band=1)
    cache = ctx.runs_dir / "_grd_cache"
    mosaic = None
    for scene in acq.scenes:
        files = download_frame(scene, cache)
        arr = process_frame(ctx, files, dem)
        if arr is None:
            continue
        mosaic = arr if mosaic is None else np.where(np.isfinite(mosaic), mosaic, arr)
    if mosaic is None:
        raise RuntimeError(f"No Sentinel-1 frame of {acq.day} overlaps the AOI")
    write_raster(out, mosaic, ctx.grid, nodata=np.nan)
    return out
