"""Turn every GeoTIFF of a run into a PNG you can open anywhere (VS Code, Explorer, slides).

GeoTIFFs store measurements, not pictures: radar as small decimals (0.001-0.5), elevation in metres,
classes as 0/1/2/3/255. Image viewers expect brightness 0-255, so they show black or nothing.
This script colours each file the right way and adds a title, a colour bar or legend, and lon/lat axes.

    python scripts/preview_run.py runs\\trishuli_case
Output: runs/<run>/previews/<file>.png  (originals are not changed)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CLASS_NAMES = {0: ("No change", "#f2f2f2"), 1: ("New flood water", "#1e6ee6"), 2: ("Debris / mud", "#965f28"),
               3: ("Permanent water", "#78aadc"), 255: ("Radar blind / no image", "#808080")}
CONF_NAMES = {0: ("-", "#f2f2f2"), 1: ("Low", "#fdd49e"), 2: ("Medium", "#fc8d59"), 3: ("High", "#d7301f")}
S2_BANDS = ["B03 green", "B04 red", "B08 NIR", "B11 SWIR"]


def _read(path, max_px=1800):
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        f = max(1, int(np.ceil(max(src.width, src.height) / max_px)))
        shape = (src.count, src.height // f, src.width // f)
        cat = "class" in path.name or "confidence" in path.name
        a = src.read(out_shape=shape, resampling=Resampling.nearest if cat else Resampling.average).astype(np.float32)
        if src.nodata is not None and not cat:
            a[a == src.nodata] = np.nan
        b = src.bounds
        if src.crs and not src.crs.is_geographic:
            from rasterio.warp import transform_bounds
            b = transform_bounds(src.crs, "EPSG:4326", *b)
        return a, (b[0], b[2], b[1], b[3]), src.crs


def _stretch(a, pct=(2, 98)):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return np.zeros_like(a), 0, 1
    lo, hi = np.percentile(v, pct)
    hi = max(hi, lo + 1e-6)
    return np.clip((a - lo) / (hi - lo), 0, 1), lo, hi


def _db(lin):
    with np.errstate(divide="ignore", invalid="ignore"):
        return 10 * np.log10(np.where(lin > 0, lin, np.nan))


def render(path: Path, out: Path) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    a, ext, crs = _read(path)
    name = path.stem
    kw = dict(extent=ext, interpolation="nearest")

    def finish(fig, ax_list, title):
        for ax in ax_list:
            ax.set_xlabel("longitude", fontsize=7)
            ax.set_ylabel("latitude", fontsize=7)
            ax.tick_params(labelsize=6)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out, dpi=130)
        plt.close(fig)

    if name.startswith("s1_"):                      # radar: linear backscatter -> dB, one panel per polarisation
        n = min(a.shape[0], 2)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 6.5))
        axes = np.atleast_1d(axes)
        for i, ax in enumerate(axes):
            band = a[i]   # openEO files hold linear backscatter; a file already in dB has negative values
            db = band if np.nanmin(band) < 0 else _db(band)
            _, lo, hi = _stretch(db)
            im = ax.imshow(db, cmap="gray", vmin=lo, vmax=hi, **kw)
            ax.set_title(["VV", "VH"][i] + " backscatter (dB)", fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.6, label="dB")
        parts = name.split("_")      # s1_<track>_<pre|post>_<day>
        what = f"Sentinel-1 radar, track {parts[1]}, {'BEFORE' if 'pre' in parts else 'AFTER'} image"
        finish(fig, axes, f"{what}\nbright = rough surfaces / slopes facing the satellite, dark = smooth (water)")
        return what

    if name.startswith("s2_"):                      # optical: false colour SWIR-NIR-red
        if a.shape[0] >= 4:
            rgb = np.dstack([_stretch(a[3])[0], _stretch(a[2])[0], _stretch(a[1])[0]])
            rgb[~np.isfinite(a[1])] = 1
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.imshow(np.nan_to_num(rgb, nan=1.0), **kw)
            cloud = np.mean(~np.isfinite(a[1]))
            finish(fig, [ax], f"Sentinel-2 {'BEFORE' if 'pre' in name else 'AFTER'} (false colour SWIR/NIR/red)\n"
                              f"green = vegetation, dark blue = water, pink/brown = bare soil or debris, "
                              f"white = cloud masked ({100 * cloud:.0f}%)")
            return "Sentinel-2 false colour"

    if name.startswith("dem"):                      # elevation with hillshade
        z = a[0]
        gy, gx = np.gradient(np.nan_to_num(z, nan=np.nanmean(z)))
        shade = np.clip(0.5 + (gx - gy) / (np.nanpercentile(np.hypot(gx, gy), 95) * 2 + 1e-6), 0, 1)
        fig, ax = plt.subplots(figsize=(7, 7))
        im = ax.imshow(z, cmap="terrain", **kw)
        ax.imshow(shade, cmap="gray", alpha=0.35, **kw)
        fig.colorbar(im, ax=ax, shrink=0.6, label="metres above sea level")
        finish(fig, [ax], "Copernicus DEM GLO-30 (elevation)")
        return "elevation"

    if name in ("flood_class", "confidence"):       # categories with a legend
        names = CLASS_NAMES if name == "flood_class" else CONF_NAMES
        codes = sorted(names)
        idx = np.zeros(a[0].shape, int)
        for i, c in enumerate(codes):
            idx[a[0] == c] = i
        fig, ax = plt.subplots(figsize=(7, 7.5))
        ax.imshow(idx, cmap=ListedColormap([names[c][1] for c in codes]), vmin=0, vmax=len(codes) - 1, **kw)
        ax.legend(handles=[Patch(color=names[c][1], label=f"{c}: {names[c][0]}") for c in codes],
                  loc="lower right", fontsize=7)
        title = ("FloodSight result: what changed" if name == "flood_class"
                 else "FloodSight confidence for flood / debris pixels")
        finish(fig, [ax], title + "\n(tiny areas: zoom in on the PNG, or open the .tif in QGIS)")
        return title

    # anything else: each band grey, stretched
    fig, axes = plt.subplots(1, min(a.shape[0], 4), figsize=(6 * min(a.shape[0], 4), 6.5))
    axes = np.atleast_1d(axes)
    for i, ax in enumerate(axes):
        _, lo, hi = _stretch(a[i])
        im = ax.imshow(a[i], cmap="gray", vmin=lo, vmax=hi, **kw)
        ax.set_title(f"band {i + 1}", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.6)
    finish(fig, axes, name)
    return name


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="run folder, e.g. runs\\trishuli_case")
    a = ap.parse_args(argv)
    run = Path(a.run)
    out_dir = run / "previews"
    out_dir.mkdir(exist_ok=True)
    tifs = sorted(p for p in run.rglob("*.tif") if ".part" not in p.name and "previews" not in p.parts)
    if not tifs:
        print(f"No .tif files under {run.resolve()}")
        return
    for p in tifs:
        out = out_dir / f"{p.stem}.png"
        try:
            what = render(p, out)
            print(f"  {p.relative_to(run)}  ->  previews/{out.name}   ({what})")
        except Exception as e:  # noqa: BLE001
            print(f"  ! {p.name}: {e}")
    print(f"\nOpen the PNGs in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
