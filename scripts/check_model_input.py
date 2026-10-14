"""Does our image preparation cost the model accuracy? Compare model inputs on a finished run. (read-only)

    python scripts/check_model_input.py --run runs\\ks_205_05 --root D:\\data\\KuroSiwo\\GRD\\data --event 205

Uses only files already in the run folder (no satellite processing, no processing units) and the Kuro Siwo
tiles of that event. It answers:
  1. How grainy is our after-image, before and after the pipeline's speckle matching, compared with Kuro
     Siwo's after-image of the SAME place and date (same measure the matching uses: median 5x5 local std of
     VV in dB on flat ground)?
  2. How does the model score against the Kuro Siwo labels on our images with:
       none     no speckle matching
       matched  smoothed to model.input_speckle_db (the pipeline did this until 4 Oct; now off)
       Lee k    a fixed Lee window k for both dates (3, 5, 7)
Nothing is changed. Use it on a training event (e.g. 1111009) to choose a setting, and treat any event you
look at with it as a DIAGNOSIS event afterwards (docs/tuning_log.md, data roles).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _pair_files(run: Path, res: dict, ks_dates):
    pairs = [p for p in res["inputs"]["sentinel1_pairs"] if "track" in p]
    same = [p for p in pairs if p["post_date"] in set(ks_dates)]
    p = (same or pairs)[0]
    pre = sorted((run / "inputs").glob(f"s1_{p['track']}_pre_{p['pre_date']}*.tif"))
    post = sorted((run / "inputs").glob(f"s1_{p['track']}_post_{p['post_date']}*.tif"))
    if not pre or not post:
        raise SystemExit(f"Cached images of track {p['track']} not found in {run / 'inputs'}")
    return p, pre[0], post[0]


def _ks_post_on_grid(samples, shape, tf, crs):
    import rasterio
    from rasterio.warp import Resampling, reproject

    from floodsight.model.dataset import _db

    out = np.full(shape, np.nan, np.float32)
    for s in samples:
        with rasterio.open(s["post_vv"]) as src:
            db = _db(src.read(1).astype(np.float32))
            dst = np.full(shape, np.nan, np.float32)
            reproject(db, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tf, dst_crs=crs,
                      src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest)
        put = np.isfinite(dst) & ~np.isfinite(out)
        out[put] = dst[put]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--event", required=True)
    ap.add_argument("--weights", help="default: config model.weights")
    a = ap.parse_args(argv)

    import rasterio

    from floodsight.core import Grid, load_config, read_to_grid
    from floodsight.detect.baseline import lee_filter_db
    from floodsight.model.dataset import find_samples
    from floodsight.model.infer import predict, speckle_level
    from floodsight.model.metrics import binary_scores
    from floodsight.preprocess.geometry import slope_deg
    from floodsight.validate.kurosiwo import _dates_near, _tiles_in_box, labels_on_grid, model_role

    cfg = load_config()
    weights = a.weights or cfg["model"]["weights"]
    thr = cfg["model"].get("flood_prob", 0.8)
    run = Path(a.run)
    res = json.loads((run / "results.json").read_text())
    grid = Grid.from_dict(res["grid"])
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        shape, tf, crs = (src.height, src.width), src.transform, src.crs
    samples = [s for s in find_samples(a.root) if str(s["event"]) == str(a.event)]
    truth, used = labels_on_grid(samples, shape, tf, crs)
    known = truth != 255
    if not known.any():
        raise SystemExit("No Kuro Siwo labels inside this run.")
    pair, f_pre, f_post = _pair_files(run, res, _dates_near(samples))

    def db(path):
        lin = read_to_grid(path, grid)[:2]
        lin = np.where(np.isfinite(lin) & (lin > 0), lin, np.nan)
        return (10 * np.log10(np.clip(lin, 1e-5, None))).astype(np.float32)

    pre, post = db(f_pre), db(f_post)
    dem = read_to_grid(run / "inputs" / "dem.tif", grid)[0]
    slope = slope_deg(dem, grid.res)
    target = cfg["model"].get("input_speckle_db", 0.4)

    ks = _ks_post_on_grid(_tiles_in_box(samples, tf, crs, shape), shape, tf, crs)
    both = np.isfinite(ks) & np.isfinite(post[0])
    print(f"Run {run.name}, track {pair['track']} {pair['pre_date']} -> {pair['post_date']}; event {a.event} "
          f"(model role: {model_role(weights, a.event)}); {used} label tiles\n")
    print("Graininess, median 5x5 std of VV on flat ground (dB), same pixels:")
    print(f"  Kuro Siwo after-image            {speckle_level(np.where(both, ks, np.nan), slope):.2f}")
    print(f"  ours, raw                        {speckle_level(np.where(both, post[0], np.nan), slope):.2f}")
    for k in (3, 5, 7):
        print(f"  ours, Lee {k}x{k}                    "
              f"{speckle_level(np.where(both, lee_filter_db(post[0], k), np.nan), slope):.2f}")
    print(f"  training-data target (config)    {target:.2f}\n")

    variants = {"none": dict(speckle_target=None), f"matched to {target} dB": dict(speckle_target=target)}
    rows = []
    for name, kw in variants.items():
        r = predict(pre, post, slope, grid.res, weights, cfg["model"]["tile"], cfg["model"]["overlap"], **kw)
        rows.append((name, r["p_flood"], r.get("speckle_matching")))
    for k in (3, 5, 7):
        f = lambda img: np.stack([lee_filter_db(b, k) for b in img])
        r = predict(f(pre), f(post), slope, grid.res, weights, cfg["model"]["tile"], cfg["model"]["overlap"],
                    speckle_target=None)
        rows.append((f"Lee {k}x{k}", r["p_flood"], None))

    print(f"Model alone (p >= {thr}) vs Kuro Siwo labels:")
    print(f"  {'input':28s} {'IoU':>6s} {'prec':>6s} {'recall':>7s}  matching")
    out = {}
    for name, p, info in rows:
        sc = binary_scores(np.nan_to_num(p, nan=0) >= thr, truth == 2, known & np.isfinite(p))
        out[name] = {k: round(float(v), 3) for k, v in sc.items() if isinstance(v, (int, float))}
        print(f"  {name:28s} {sc['iou']:6.3f} {sc['precision']:6.3f} {sc['recall']:7.3f}  {info or ''}")
    (run / "validation").mkdir(exist_ok=True)
    (run / "validation" / f"model_input_{a.event}.json").write_text(json.dumps(out, indent=1))
    print("\nDecide on a TRAINING event; events inspected here become diagnosis events.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
