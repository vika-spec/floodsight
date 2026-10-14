"""Case-study check against a reference map (e.g. Copernicus EMS EMSR927).

THIS MODULE IS THE ONLY PLACE REFERENCE DAMAGE MAPS ARE OPENED. The pipeline never imports it
(tests/test_rules.py enforces this), so reference data cannot leak into the system's inputs.

    python -m floodsight.validate.compare --run runs/<run_id> \
        --reference validation_data/EMSR927/..._observedEventA_v1.shp \
        [--reference-aoi validation_data/EMSR927/..._areaOfInterestA_v1.shp]

Credit when publishing: "European Union, Copernicus Emergency Management Service data".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..detect.baseline import DEBRIS, FLOOD, NODATA
from ..model.metrics import binary_scores


def rasterize_reference(paths, grid_tf, shape, crs, field=None, values=None):
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize

    gdf = pd.concat([gpd.read_file(p) for p in paths], ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, crs=gdf.crs or 4326)
    if field and values:
        gdf = gdf[gdf[field].astype(str).isin(values)]
    gdf = gdf.to_crs(crs)
    gdf = gdf[gdf.geometry.notna()]
    if gdf.empty:
        return np.zeros(shape, bool), gdf
    geoms = [g.buffer(0) if g.geom_type in ("Polygon", "MultiPolygon") else g.buffer(10) for g in gdf.geometry]
    return rasterize(((g, 1) for g in geoms), out_shape=shape, transform=grid_tf, fill=0, dtype="uint8").astype(bool), gdf


def compare(run: Path, reference: list, reference_aoi: list | None = None, field=None, values=None) -> dict:
    import rasterio

    run = Path(run)
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        cls = src.read(1)
        tf, crs, shape = src.transform, src.crs, cls.shape
    ref, gdf = rasterize_reference(reference, tf, shape, crs, field, values)
    if reference_aoi:
        aoi_mask, _ = rasterize_reference(reference_aoi, tf, shape, crs)
    else:
        aoi_mask = np.ones(shape, bool)
    observed = cls != NODATA
    valid = aoi_mask & observed
    pred = (cls == FLOOD) | (cls == DEBRIS)
    px_km2 = abs(tf.a * tf.e) / 1e6
    m = binary_scores(pred, ref, valid)
    out = {
        "reference": ", ".join(Path(p).name for p in reference),
        "note": "Reference used only for checking. Pixels the radar could not observe are excluded "
                "and reported separately.",
        "flood_extent": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "area_km2": {"ours": round(float((pred & aoi_mask).sum() * px_km2), 3),
                     "reference": round(float((ref & aoi_mask).sum() * px_km2), 3),
                     "reference_in_radar_blind_spots": round(float((ref & aoi_mask & ~observed).sum() * px_km2), 3)},
        "compared_area_km2": round(float(valid.sum() * px_km2), 1),
    }
    _figure(run, cls, pred, ref, valid)
    (run / "validation").mkdir(exist_ok=True)
    (run / "validation" / "validation.json").write_text(json.dumps(out, indent=1))
    return out


def _figure(run: Path, cls, pred, ref, valid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    img = np.ones(cls.shape + (3,)) * 0.94
    img[~valid] = (0.8, 0.8, 0.8)
    img[valid & pred & ref] = (0.17, 0.63, 0.17)
    img[valid & pred & ~ref] = (0.84, 0.15, 0.16)
    img[valid & ~pred & ref] = (0.12, 0.47, 0.71)
    (run / "validation").mkdir(exist_ok=True)
    h = min(6, 9 * cls.shape[0] / cls.shape[1])
    fig, ax = plt.subplots(figsize=(9, h), dpi=150)
    ax.imshow(img, interpolation="nearest")
    ax.set_axis_off()
    ax.legend(handles=[Patch(color=(0.17, 0.63, 0.17), label="Both (agree)"),
                       Patch(color=(0.84, 0.15, 0.16), label="Only ours"),
                       Patch(color=(0.12, 0.47, 0.71), label="Only reference"),
                       Patch(color=(0.8, 0.8, 0.8), label="Not compared (radar blind / outside)")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(run / "validation" / "comparison.png")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--reference", nargs="+", required=True, help="reference flood-extent vector file(s)")
    ap.add_argument("--reference-aoi", nargs="*", help="reference area-of-interest polygon(s)")
    ap.add_argument("--field", help="attribute to filter reference features")
    ap.add_argument("--values", nargs="*", help="attribute values to keep")
    a = ap.parse_args(argv)
    print(json.dumps(compare(Path(a.run), a.reference, a.reference_aoi, a.field, a.values), indent=1))


if __name__ == "__main__":
    main()
