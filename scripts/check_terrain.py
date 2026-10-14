"""How much steep terrain is in each Kuro Siwo event, compared with the Trishuli case study? (read-only)

    python scripts/check_terrain.py --root D:\\data\\KuroSiwo\\GRD\\data --run runs\\trishuli_shub

For every event folder it samples tiles and reports: tile count, location (if the tiles carry coordinates),
elevation, slope statistics, share of steep ground, and how much of the tile is labelled flood / permanent
water. Slope is computed after resampling every DEM to the same 30 m pixel size, so Kuro Siwo (10 m tiles)
and Trishuli (20 m grid) are compared fairly. Nothing is trained, tuned or written except a CSV + PNG.
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

COMMON_RES = 30.0


def slope_at(dem: np.ndarray, px_m: float, res: float = COMMON_RES) -> np.ndarray:
    from scipy.ndimage import zoom

    z = np.where(np.isfinite(dem), dem, np.nanmean(dem) if np.isfinite(dem).any() else 0.0)
    f = px_m / res
    if abs(f - 1) > 1e-3:
        z = zoom(z, f, order=1)
    gy, gx = np.gradient(z, res)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def stats(slopes: list[np.ndarray]) -> dict:
    s = np.concatenate([x.ravel() for x in slopes]) if slopes else np.zeros(1)
    return {"slope_mean": float(np.mean(s)), "slope_p90": float(np.percentile(s, 90)),
            "pct_gt10": 100 * float((s > 10).mean()), "pct_gt20": 100 * float((s > 20).mean()),
            "pct_gt30": 100 * float((s > 30).mean())}


def tile_center(path):
    import rasterio
    from rasterio.warp import transform

    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        cx, cy = src.xy(src.height // 2, src.width // 2)
        lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
        return float(lon[0]), float(lat[0])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Kuro Siwo folder, e.g. D:\\data\\KuroSiwo\\GRD\\data")
    ap.add_argument("--run", help="a FloodSight run folder to compare with, e.g. runs\\trishuli_shub")
    ap.add_argument("--per-event", type=int, default=150, help="tiles sampled per event (speed)")
    a = ap.parse_args(argv)

    import rasterio

    from floodsight.model.dataset import _read, find_samples

    print("Scanning tiles (read-only)...")
    samples = find_samples(a.root)
    by_ev = defaultdict(list)
    for s in samples:
        by_ev[s["event"]].append(s)
    rng = random.Random(0)
    rows = []
    for ev in sorted(by_ev, key=lambda e: (len(e), e)):
        tiles = by_ev[ev]
        pick = rng.sample(tiles, min(a.per_event, len(tiles)))
        slopes, elev, flood, perm, centers = [], [], [], [], []
        for s in pick:
            if s.get("dem"):
                with rasterio.open(s["dem"]) as src:
                    px = abs(src.transform.a) if src.crs is not None and not src.crs.is_geographic else 10.0
                dem = _read(s["dem"])
                slopes.append(slope_at(dem, px))
                elev.append(np.nanmedian(dem))
            lab = _read(s["label"])
            valid = _read(s["valid"]) == 1
            n = max(valid.sum(), 1)
            flood.append(((lab == 2) & valid).sum() / n)
            perm.append(((lab == 1) & valid).sum() / n)
            if len(centers) < 5:
                c = tile_center(s["post_vv"])
                if c:
                    centers.append(c)
        st = stats(slopes)
        lon, lat = (np.mean([c[0] for c in centers]), np.mean([c[1] for c in centers])) if centers else (None, None)
        rows.append({"event": ev, "tiles": len(tiles), "sampled": len(pick),
                     "lon": None if lon is None else round(lon, 2), "lat": None if lat is None else round(lat, 2),
                     "elev_median_m": round(float(np.median(elev)), 0) if elev else None,
                     **{k: round(v, 1) for k, v in st.items()},
                     "flood_pct": round(100 * float(np.mean(flood)), 1), "perm_water_pct": round(100 * float(np.mean(perm)), 1)})

    if a.run:
        from floodsight.core import read_to_grid  # noqa: F401
        run = Path(a.run)
        with rasterio.open(run / "inputs" / "dem.tif") as src:
            dem = src.read(1).astype(np.float32)
            px = abs(src.transform.a)
        sl = slope_at(dem, px)
        # valley floor = lowest 25% of the area by height above the valley bottom (no labels involved)
        from scipy.ndimage import minimum_filter, zoom
        z = zoom(np.where(np.isfinite(dem), dem, np.nanmean(dem)), px / COMMON_RES, order=1)
        rel = z - minimum_filter(z, size=int(1000 / COMMON_RES))
        floor = rel <= np.percentile(rel, 25)
        st_all, st_floor = stats([sl]), stats([sl[floor[: sl.shape[0], : sl.shape[1]]]])
        for name, st in (("TRISHULI (whole area)", st_all), ("TRISHULI (valley floor)", st_floor)):
            rows.append({"event": name, "tiles": None, "sampled": None, "lon": None, "lat": None,
                         "elev_median_m": round(float(np.nanmedian(dem)), 0), **{k: round(v, 1) for k, v in st.items()},
                         "flood_pct": None, "perm_water_pct": None})

    cols = ["event", "tiles", "lon", "lat", "elev_median_m", "slope_mean", "slope_p90", "pct_gt10", "pct_gt20",
            "pct_gt30", "flood_pct", "perm_water_pct"]
    print(f"\nSlope computed at a common {COMMON_RES:g} m pixel size.  pct_gtN = % of pixels steeper than N degrees.")
    print(" ".join(f"{c:>12s}" for c in cols))
    for r in rows:
        print(" ".join(f"{('-' if r[c] is None else r[c])!s:>12s}" for c in cols))

    out = Path(a.run or ".") / "terrain_check.csv" if a.run else Path("terrain_check.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
