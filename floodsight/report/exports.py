"""Exports for the dashboard and the report: GeoTIFFs, web overlays (PNG in WGS84), GeoJSON."""
from __future__ import annotations

import numpy as np

from ..core import Grid, write_raster
from ..detect.baseline import DEBRIS, FLOOD, NODATA, PERMANENT

# RGBA per class; confidence modulates opacity for flood/debris
CLASS_RGBA = {FLOOD: (30, 110, 230), DEBRIS: (150, 95, 40), PERMANENT: (120, 170, 220), NODATA: (128, 128, 128)}


def to_wgs84(arr: np.ndarray, grid: Grid, nearest=True, max_px=2400):
    """Reproject a grid array to EPSG:4326 for Leaflet overlays. Returns (array, [[s, w], [n, e]])."""
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    w, s, e, n = grid.bounds
    tr, W, H = calculate_default_transform(grid.crs, "EPSG:4326", grid.width, grid.height, w, s, e, n)
    f = min(1.0, max_px / max(W, H))
    if f < 1:
        from affine import Affine
        W, H = int(W * f), int(H * f)
        tr = Affine(tr.a / f, tr.b, tr.c, tr.d, tr.e / f, tr.f)
    fill = 0 if arr.dtype == np.uint8 else np.nan
    out = np.full((H, W), fill, dtype=arr.dtype)
    reproject(arr, out, src_transform=grid.transform, src_crs=grid.crs, dst_transform=tr, dst_crs="EPSG:4326",
              resampling=Resampling.nearest if nearest else Resampling.bilinear,
              src_nodata=None if arr.dtype == np.uint8 else np.nan, dst_nodata=fill)
    west, north = tr.c, tr.f
    east, south = west + tr.a * W, north + tr.e * H
    return out, [[south, west], [north, east]]


def class_png(cls: np.ndarray, conf: np.ndarray, grid: Grid, path, show_nodata=True):
    from PIL import Image

    c4, bounds = to_wgs84(cls, grid)
    k4, _ = to_wgs84(conf, grid)
    rgba = np.zeros(c4.shape + (4,), np.uint8)
    valid_px = np.zeros(c4.shape, bool)
    for k, col in CLASS_RGBA.items():
        m = c4 == k
        if k == NODATA and not show_nodata:
            continue
        rgba[m, :3] = col
        if k in (FLOOD, DEBRIS):
            rgba[m, 3] = np.array([0, 140, 190, 235], np.uint8)[np.clip(k4[m], 0, 3)]
        elif k == NODATA:
            # hatched grey so "unknown" is never confused with "safe"
            yy, xx = np.nonzero(m)
            rgba[yy, xx, 3] = np.where(((yy + xx) // 3) % 2 == 0, 110, 35)
        else:
            rgba[m, 3] = 150
        valid_px |= m
    # outside the analysed rectangle after reprojection: transparent (already 0)
    Image.fromarray(rgba, "RGBA").save(path, optimize=True)
    return bounds


def shared_stretch(*dbs, pct=(2, 98)):
    """One dB range for all dates, from the data itself: the same grey means the same radar return in every
    image, so a difference you can see is a difference in the data (fixed -25..0 dB saturated mountain slopes)."""
    v = np.concatenate([d[np.isfinite(d)].ravel()[::7] for d in dbs if d is not None])
    if v.size == 0:
        return -25.0, 0.0
    lo, hi = np.percentile(v, pct)
    return float(lo), float(max(hi, lo + 1))


def _grey(db, lo, hi):
    return np.clip((np.nan_to_num(db, nan=lo) - lo) / (hi - lo), 0, 1)


def quicklook_png(db: np.ndarray, grid: Grid, path, lo=-25.0, hi=0.0):
    from PIL import Image

    a, bounds = to_wgs84(db.astype(np.float32), grid, nearest=False)
    g = (_grey(a, lo, hi) * 255).astype(np.uint8)
    alpha = np.where(np.isfinite(a), 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([g, g, g, alpha]), "RGBA").save(path, optimize=True)
    return bounds


def prob_png(prob: np.ndarray, grid: Grid, path, floor: float = 0.2):
    """AI model flood probability for the map: transparent below `floor`, then light to dark blue. Shows what the
    model alone sees, before it is combined with the radar rules."""
    from PIL import Image

    a, bounds = to_wgs84(prob.astype(np.float32), grid, nearest=False)
    v = np.nan_to_num(a, nan=0.0)
    t = np.clip((v - floor) / (1 - floor), 0, 1)
    light, dark = np.array([170, 210, 255]), np.array([10, 40, 160])
    rgb = (light[None, None] * (1 - t[..., None]) + dark[None, None] * t[..., None]).astype(np.uint8)
    alpha = np.where(v >= floor, (90 + 150 * t).astype(np.uint8), 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path, optimize=True)
    return bounds


CHANGE_DB = 2.0     # display only: |after - before| below this is shown as grey (speckle-level change)
CHANGE_FULL = 6.0   # change at which the colour is fully saturated


def change_rgb(pre, post, lo, hi):
    """Radar change picture: grey terrain, RED where the ground got darker (water, smoothing),
    CYAN where it got brighter (debris, rubble, roughening). Display of the raw data, not the classification."""
    base = 0.25 + 0.6 * _grey((pre + post) / 2, lo, hi)
    d = np.nan_to_num(post - pre, nan=0.0)
    t = np.clip((np.abs(d) - CHANGE_DB) / (CHANGE_FULL - CHANGE_DB), 0, 1)
    rgb = np.dstack([base, base, base])
    red, cyan = np.array([0.93, 0.16, 0.12]), np.array([0.05, 0.85, 0.95])
    for mask, col in ((d < 0, red), (d > 0, cyan)):
        k = (t * mask)[..., None]
        rgb = rgb * (1 - k) + col * k
    return (rgb * 255).astype(np.uint8)


def change_png(pre, post, grid: Grid, path, lo, hi):
    from PIL import Image

    a0, bounds = to_wgs84(pre.astype(np.float32), grid, nearest=False)
    a1, _ = to_wgs84(post.astype(np.float32), grid, nearest=False)
    rgb = change_rgb(a0, a1, lo, hi)
    alpha = np.where(np.isfinite(a0) & np.isfinite(a1), 255, 0).astype(np.uint8)
    Image.fromarray(np.dstack([rgb, alpha]), "RGBA").save(path, optimize=True)
    return bounds


def zoom_window(cls, min_px=200, pad_px=40):
    """Window around the largest connected flood/debris patch (rows, cols slices), or None."""
    from scipy import ndimage

    hit = (cls == FLOOD) | (cls == DEBRIS)
    if not hit.any():
        return None
    lab, n = ndimage.label(hit, structure=np.ones((3, 3)))
    sizes = ndimage.sum(hit, lab, range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    rr, cc = np.nonzero(lab == k)
    r0, r1, c0, c1 = rr.min() - pad_px, rr.max() + pad_px, cc.min() - pad_px, cc.max() + pad_px
    def fit(a, b, n_):
        if b - a < min_px:
            m = (a + b) // 2
            a, b = m - min_px // 2, m + min_px // 2
        return max(a, 0), min(b, n_)
    r0, r1 = fit(r0, r1, cls.shape[0])
    c0, c1 = fit(c0, c1, cls.shape[1])
    return np.s_[r0:r1, c0:c1]


def _outline(mask: np.ndarray, width: int) -> np.ndarray:
    from scipy import ndimage
    return mask & ~ndimage.binary_erosion(mask, iterations=max(1, width))


def _optical_rgb(img4: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Sentinel-2 false colour R = SWIR (B11), G = NIR (B08), B = red (B04): vegetation green, bare soil, mud and
    debris pink-brown, water near black. Same stretch for both dates. No clear view: grey hatching."""
    rgb = np.stack([img4[3], img4[2], img4[1]], axis=-1)
    out = np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1)
    bad = ~np.isfinite(rgb).all(axis=-1)
    out = np.nan_to_num(out, nan=0.0)
    yy, xx = np.nonzero(bad)
    out[yy, xx] = np.where((((yy + xx) // 4) % 2 == 0)[:, None], 0.55, 0.75)
    return (out * 255).astype(np.uint8)


def zoom_pngs(pre, post, cls, grid: Grid, folder, lo, hi, scale_to=700, s2=None) -> dict | None:
    """Close-up of the largest detected patch at full 20 m detail, north up: radar before / after / change, and
    Sentinel-2 before / after where the sky was clear in both. Made to be READ by a person:
      * the grey scale is stretched on this close-up only (same scale for both dates, so equal grey = equal
        radar return), which shows differences that a whole-area stretch flattens;
      * our detection is outlined in yellow on every picture, so the eye knows where to look;
      * plain numbers: how much darker or brighter the ground got inside the outline vs. outside it."""
    from PIL import Image

    win = zoom_window(cls)
    if win is None:
        return None
    p0, p1, c = pre[win], post[win], cls[win]
    f = max(1, int(round(scale_to / max(c.shape))))
    up = lambda a: np.kron(a, np.ones((f, f), dtype=a.dtype)) if f > 1 else a
    det = (c == FLOOD) | (c == DEBRIS)
    from scipy import ndimage
    # outline of the detected area, smoothed (closed, holes filled, tiny specks dropped) so it reads as one
    # clean line around the patch instead of a dotted fringe; the statistics below use the exact detection
    area = ndimage.binary_fill_holes(ndimage.binary_closing(det, iterations=2))
    area = ndimage.binary_opening(area, iterations=1)
    edge = _outline(up(area.astype(np.uint8)).astype(bool), max(2, f // 3))
    zlo, zhi = shared_stretch(p0, p1)                  # local stretch, shared by both dates
    out = {}

    def save(name, rgb):
        big = np.dstack([up(rgb[..., i]) for i in range(3)])
        big[edge] = (255, 220, 0)
        Image.fromarray(big, "RGB").save(folder / f"zoom_{name}.png", optimize=True)
        out[name] = f"layers/zoom_{name}.png"

    save("before", np.dstack([(_grey(p0, zlo, zhi) * 255).astype(np.uint8)] * 3))
    save("after", np.dstack([(_grey(p1, zlo, zhi) * 255).astype(np.uint8)] * 3))
    save("change", change_rgb(p0, p1, zlo, zhi))
    out["stretch_db"] = [round(zlo, 1), round(zhi, 1)]

    # what changed, in numbers (radar, dB): inside our detection vs. the rest of the close-up
    d = p1 - p0
    ok = np.isfinite(d)
    def med(m):
        v = d[m & ok]
        return round(float(np.median(v)), 1) if v.size else None
    stats = {"inside_change_db": med(det), "outside_change_db": med(~det)}
    for k, m in (("flood", c == FLOOD), ("debris", c == DEBRIS)):
        if (m & ok).sum() >= 10:
            stats[f"{k}_before_db"] = round(float(np.median(p0[m & ok])), 1)
            stats[f"{k}_after_db"] = round(float(np.median(p1[m & ok])), 1)
            stats[f"{k}_px"] = int((m & ok).sum())
    out["stats"] = stats

    # Sentinel-2 close-up, only where both dates had a clear view of most of it
    if s2 is not None:
        o0, o1 = s2["pre"][(slice(None),) + win], s2["post"][(slice(None),) + win]
        clear = float((np.isfinite(o0).all(axis=0) & np.isfinite(o1).all(axis=0)).mean())
        out["optical_clear_fraction"] = round(clear, 2)
        if clear >= 0.5:
            both = np.concatenate([o0[[3, 2, 1]].reshape(3, -1), o1[[3, 2, 1]].reshape(3, -1)], axis=1)
            lo3 = np.nanpercentile(both, 2, axis=1)
            hi3 = np.nanpercentile(both, 98, axis=1)
            save("opt_before", _optical_rgb(o0, lo3, hi3))
            save("opt_after", _optical_rgb(o1, lo3, hi3))

    px = abs(grid.transform.a)
    out["size_km"] = [round(c.shape[1] * px / 1000, 1), round(c.shape[0] * px / 1000, 1)]
    x0, y0 = grid.transform * (win[1].start, win[0].start)
    x1, y1 = grid.transform * (win[1].stop, win[0].stop)
    from rasterio.warp import transform as warp
    lons, lats = warp(grid.crs, "EPSG:4326", [x0, x1], [y0, y1])
    out["bbox"] = [round(min(lons), 4), round(min(lats), 4), round(max(lons), 4), round(max(lats), 4)]
    return out


def save_layers(ctx, cls, conf, extra: dict | None = None):
    write_raster(ctx.path("layers", "flood_class.tif"), cls, ctx.grid, nodata=None)
    write_raster(ctx.path("layers", "confidence.tif"), conf, ctx.grid, nodata=None)
    for name, arr in (extra or {}).items():
        write_raster(ctx.path("layers", f"{name}.tif"), arr.astype(np.float32), ctx.grid, nodata=np.nan)


def to_geojson(gdf, path, keep=None):
    g = gdf.to_crs(4326) if gdf.crs else gdf
    if keep is not None and not g.empty:
        g = g[[c for c in keep if c in g.columns] + ["geometry"]]
    for c in g.columns:
        if c != "geometry" and g[c].dtype == object:
            g[c] = g[c].astype(str)
    if g.empty:
        path.write_text('{"type": "FeatureCollection", "features": []}', encoding="utf-8")
    else:
        g.to_file(path, driver="GeoJSON")
    return path
