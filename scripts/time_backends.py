"""Measure Sentinel Hub against openEO on the same radar images, and check the two agree.

    python scripts/time_backends.py                                   # Trishuli box and date
    python scripts/time_backends.py --compare runs\\trishuli_case       # also compare with openEO images
    python scripts/time_backends.py --bbox 85.5,27.8,85.65,28.05 --date 2021-06-15

What it does (nothing here touches reference maps; it only downloads raw Sentinel-1):
  1. finds the same before/after pairs the pipeline would use,
  2. downloads each image through Sentinel Hub, timing each one and reading the processing units spent,
  3. with --compare: for every image that the earlier openEO run also has, reports how similar they are
     (correlation of dB values, median difference on flat ground and on steep slopes). Sentinel Hub gives
     terrain-flattened gamma0, openEO gives sigma0, so flat ground should agree and steep slopes should
     differ: that difference is exactly the terrain brightening Sentinel Hub removes.
Results go to runs/_timing_shub/ (a separate cache, so it never mixes with real runs).
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default="85.08,27.85,85.40,28.29")
    ap.add_argument("--date", default="2026-08-26")
    ap.add_argument("--compare", help="earlier run folder with openEO images, e.g. runs\\trishuli_case")
    ap.add_argument("--fresh", action="store_true", help="delete the timing cache first (re-measure)")
    a = ap.parse_args(argv)

    from dotenv import load_dotenv
    load_dotenv()
    from floodsight.core import AOI, RunContext, load_config, parse_date, read_to_grid, setup_logging
    from floodsight.fetch import shub
    from floodsight.fetch.catalog import search_s1, select_pairs
    from floodsight.preprocess.geometry import slope_deg

    setup_logging()
    if not shub.available():
        print("SH_CLIENT_ID / SH_CLIENT_SECRET are not set in .env (in the folder with run.py).")
        return 1
    cfg = load_config()
    aoi, d = AOI.parse(a.bbox), parse_date(a.date)
    ctx = RunContext(aoi, d, cfg, name="_timing_shub")
    if a.fresh:
        for p in ctx.dir.joinpath("inputs").glob("s1_*_shub.tif"):
            p.unlink()

    print(f"\nArea {aoi.area_km2():.0f} km², grid {ctx.grid.width} x {ctx.grid.height} px at {ctx.grid.res:g} m")
    t = time.time()
    s1 = cfg["sentinel1"]
    pairs = select_pairs(search_s1(aoi, d - timedelta(days=s1["pre_window_days"]),
                                   d + timedelta(days=s1["post_window_days"])), aoi, d, cfg)
    print(f"Catalogue: {len(pairs)} pair(s) in {time.time() - t:.1f}s")

    rows = []
    for p in pairs:
        for k in ("pre", "post"):
            acq = getattr(p, k)
            out = ctx.path("inputs", f"s1_{acq.track}_{k}_{acq.day}_shub.tif")     # same names as the pipeline
            old = ctx.path("inputs", f"s1_{acq.track}_{k}_{acq.day.day}_shub.tif")  # first version of this script
            if old.exists() and not out.exists():
                old.rename(out)                       # reuse what was already downloaded (no extra quota)
            cached = out.exists()
            pu0, t0 = shub.USAGE["pu"], time.time()
            try:
                shub.s1_acquisition(ctx, acq, out)
                rows.append((acq.track, k, str(acq.day), "cached" if cached else f"{time.time() - t0:.1f}s",
                             shub.USAGE["pu"] - pu0, out))
            except Exception as e:  # noqa: BLE001
                rows.append((acq.track, k, str(acq.day), f"FAILED: {str(e)[:120]}", 0, None))

    print(f"\n{'track':9s} {'image':5s} {'date':11s} {'time':>10s} {'PU':>7s}")
    for tr, k, day, tm, pu, _ in rows:
        print(f"{tr:9s} {k:5s} {day:11s} {tm:>10s} {pu:7.1f}")
    print(f"Total processing units this run: {shub.USAGE['pu']:.1f} (free tier: 10 000 per month)")

    if a.compare:
        from scipy.ndimage import uniform_filter
        from skimage.registration import phase_cross_correlation

        run = Path(a.compare)
        dem_p = run / "inputs" / "dem.tif"
        slope = slope_deg(read_to_grid(dem_p, ctx.grid, band=1), ctx.grid.res) if dem_p.exists() else None

        def smooth(x, k=5):                      # 5 x 5 pixels = 100 m: averages speckle away
            ok = np.isfinite(x)
            num = uniform_filter(np.where(ok, x, 0.0), k)
            den = uniform_filter(ok.astype(float), k)
            return np.where(den > 0.5, num / np.maximum(den, 1e-6), np.nan)

        def corr(x, y, m):
            m = m & np.isfinite(x) & np.isfinite(y)
            return np.corrcoef(x[m], y[m])[0, 1] if m.sum() > 100 else float("nan")

        print(f"\nAgreement with openEO images in {run} (VV, dB):")
        print(f"{'track':9s} {'image':5s} {'corr raw':>8s} {'corr 100m':>9s} {'shift px (row,col)':>19s} "
              f"{'flat diff':>9s} {'steep diff':>10s}")
        figs, store = [], {}
        for tr, k, day, _, _, out in rows:
            if out is None:
                continue
            oe = run / "inputs" / out.name.replace("_shub", "")
            if not oe.exists():
                print(f"{tr:9s} {k:5s} no openEO file ({oe.name})")
                continue
            a_db = 10 * np.log10(np.clip(read_to_grid(out, ctx.grid)[0], 1e-5, None))
            b_db = 10 * np.log10(np.clip(read_to_grid(oe, ctx.grid)[0], 1e-5, None))
            ok = np.isfinite(a_db) & np.isfinite(b_db)
            sa, sb = smooth(a_db), smooth(b_db)
            # misregistration check on a central 1024 px window (sub-pixel phase correlation)
            h, w = a_db.shape
            r0, c0 = max(0, h // 2 - 512), max(0, w // 2 - 512)
            wa = np.nan_to_num(sa[r0:r0 + 1024, c0:c0 + 1024], nan=np.nanmean(sa))
            wb = np.nan_to_num(sb[r0:r0 + 1024, c0:c0 + 1024], nan=np.nanmean(sb))
            shift, _, _ = phase_cross_correlation(wa - wa.mean(), wb - wb.mean(), upsample_factor=4,
                                                  normalization=None)   # plain cross-correlation: robust to speckle
            diff = a_db - b_db
            flat = ok & (slope < 5) if slope is not None else ok
            steep = ok & (slope > 25) if slope is not None else ok
            c100 = corr(sa, sb, ok)
            # a shift is only measurable when the two images look alike; otherwise the number is meaningless
            shift_txt = f"({shift[0]:+.1f}, {shift[1]:+.1f})" if c100 >= 0.6 else "not measurable"
            print(f"{tr:9s} {k:5s} {corr(a_db, b_db, ok):8.3f} {c100:9.3f} "
                  f"{shift_txt:>19s} {np.nanmedian(diff[flat]):9.2f} {np.nanmedian(diff[steep]):10.2f}")
            figs.append((f"{tr} {k} {day}", sa, sb, r0, c0))
            store[(tr, k)] = (sa, sb, flat)
            print(f"{'':15s} flat-ground only: corr 100m {corr(sa, sb, flat):.3f}")
        tracks = sorted({t for t, _ in store})
        if tracks:
            print("\nChange after minus before (what detection uses; terrain brightening cancels here):")
            print(f"{'track':9s} {'corr 100m':>9s} {'corr where |change| > 3 dB':>27s}")
            for tr in tracks:
                if (tr, "pre") in store and (tr, "post") in store:
                    a_pre, b_pre, flat = store[(tr, "pre")]
                    a_post, b_post, _ = store[(tr, "post")]
                    da, db_ = a_post - a_pre, b_post - b_pre
                    m = np.isfinite(da) & np.isfinite(db_)
                    big = m & ((np.abs(da) > 3) | (np.abs(db_) > 3))
                    print(f"{tr:9s} {corr(da, db_, m):9.3f} {corr(da, db_, big):27.3f}")
            print("If the change correlates well (above ~0.6) the two services see the same change; the low "
                  "single-image correlation is then the terrain brightening that gamma0-terrain removes.")
        print("Reading it: 'corr 100m' above ~0.8 and a shift near (0, 0) mean both services show the same scene "
              "in the same place; a low raw correlation alone is just speckle (openEO images are unfiltered).")
        if figs:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(figs), 2, figsize=(10, 5 * len(figs)))
            axes = np.atleast_2d(axes)
            for i, (title, sa, sb, r0, c0) in enumerate(figs):
                for j, (img, name) in enumerate(((sa, "Sentinel Hub gamma0-terrain"), (sb, "openEO sigma0"))):
                    crop = img[r0:r0 + 600, c0:c0 + 600]
                    lo, hi = np.nanpercentile(crop, [2, 98])
                    axes[i, j].imshow(crop, cmap="gray", vmin=lo, vmax=hi)
                    axes[i, j].set_title(f"{title}: {name}", fontsize=8)
                    axes[i, j].set_axis_off()
            fig.tight_layout()
            png = ctx.dir / "compare_backends.png"
            fig.savefig(png, dpi=110)
            print(f"Side-by-side picture (same 12 km window, both smoothed): {png}")
    print(f"\nImages saved in {ctx.dir / 'inputs'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())