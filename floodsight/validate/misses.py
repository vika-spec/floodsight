"""WHY did we miss what EMS mapped? (and how much of the gap is just pixel position?)

    python -m floodsight.validate.misses --run runs\\slo_meza --ems validation_data\\EMSR680 --only AOI15

Checking only: opened after a run, never imported by the pipeline (tests/test_rules.py enforces this).
It changes nothing and tunes nothing; it explains a score.

For the EMS-mapped area inside the run (latest map per AOI), every pixel lands in exactly one bucket, in order:
  found                      our flood/debris map covers it
  radar blind spot           layover/shadow or no image: unknown, not missed
  called permanent water     we saw water before AND after (river channel)
  no radar change            |change| < 1.5 dB in VV and VH on every track: the radar saw (almost) nothing
                             here between the two dates (damage too small/subtle, or already gone by the after-image)
  weak change                1.5 - 3 dB: below the debris rule's 3 dB
  removed: slope             >= 3 dB but steeper than the debris rule allows
  removed: height above river  >= 3 dB but higher above the stream than the rule allows (HAND)
  removed: other filters     >= 3 dB but dropped by the 2-of-3 vote, minimum size or river-corridor link
It also reports
  * position tolerance: recall/precision when a hit within 40 m (2 pixels at 20 m) counts, which separates
    "wrong place by a pixel or two" from "not seen at all";
  * size: share of the EMS area in features smaller than our minimum object size (0.5 ha at 20 m);
  * dates: the before/after images of each track and how many days after the event the after-image is.
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np

from ..detect.baseline import DEBRIS, FLOOD, NODATA, PERMANENT, lee_filter_db
from .ems import POLYS, LINES, _only, _raster, _read, discover, latest_per_aoi

BUCKETS = ["found", "radar blind spot", "called permanent water", "no radar change", "weak change (1.5-3 dB)",
           "removed: slope", "removed: height above river", "removed: other filters"]


def _track_change(run: Path, R: dict, grid, cfg):
    """Max |dVV| and |dVH| over tracks, after the same Lee filter the detector uses (dB)."""
    from ..core import read_to_grid

    size = cfg["sentinel1"].get("speckle_filter_size", 5)
    best_vv = best_vh = None
    tracks = []
    for p in R["inputs"]["sentinel1_pairs"]:
        f = {}
        for k in ("pre", "post"):
            hits = sorted((run / "inputs").glob(f"s1_{p['track']}_{k}_{p[k + '_date']}*.tif"))
            f[k] = hits[0] if hits else None
        if not f["pre"] or not f["post"]:
            continue
        ch = []
        for b in (0, 1):
            pre = read_to_grid(f["pre"], grid, band=b + 1)
            post = read_to_grid(f["post"], grid, band=b + 1)
            db = lambda x: 10 * np.log10(np.clip(np.nan_to_num(x, nan=0.0), 1e-5, None))
            d = np.abs(lee_filter_db(db(post), size) - lee_filter_db(db(pre), size))
            d[~(np.isfinite(pre) & np.isfinite(post))] = 0
            ch.append(np.nan_to_num(d))
        best_vv = ch[0] if best_vv is None else np.maximum(best_vv, ch[0])
        best_vh = ch[1] if best_vh is None else np.maximum(best_vh, ch[1])
        tracks.append(p)
    return best_vv, best_vh, tracks


def breakdown(run, folder, only=None, event_date=None, tol_m=40.0) -> dict:
    import rasterio
    from scipy import ndimage

    from ..core import Grid, load_config
    from ..preprocess.geometry import slope_deg
    from ..preprocess.terrain import hydro_layers

    run = Path(run)
    cfg = load_config()
    d = cfg["detection"]
    R = json.loads((run / "results.json").read_text())
    grid = Grid.from_dict(R["grid"])
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        cls, tf, crs = src.read(1), src.transform, src.crs
    res = abs(tf.a)

    prods = discover(folder)
    if only:
        prods = {k: v for k, v in prods.items() if any(o.lower() in k.lower() for o in only)}
    keys = latest_per_aoi(prods)
    ref = np.zeros(cls.shape, bool)
    in_aoi = np.zeros(cls.shape, bool)
    small = np.zeros(cls.shape, bool)
    from ..detect.baseline import min_object_px
    min_area = min_object_px(d, res) * res * res          # the rule's minimum object, in m²
    for k in keys:
        aoi = _read(prods[k].get("aoi", []))
        ev = _read(prods[k].get("event", []))
        if len(aoi):
            in_aoi |= _raster(aoi, tf, cls.shape, crs)
        if len(ev):
            ev = _only(ev, POLYS + LINES)
            ref |= _raster(ev, tf, cls.shape, crs)
            pol = _only(ev, POLYS)
            if len(pol):
                m = pol.to_crs(crs)
                sm = m[m.geometry.area < min_area]
                if len(sm):
                    small |= _raster(sm, tf, cls.shape, crs)
    ref &= in_aoi
    if not ref.any():
        raise SystemExit("No EMS-mapped area falls inside this run. Check --only and the run's box.")

    pred = (cls == FLOOD) | (cls == DEBRIS)
    dvv, dvh, tracks = _track_change(run, R, grid, cfg)
    hydro = np.load(run / "layers" / "hydro.npz")
    dem = hydro["dem"]
    slope = slope_deg(dem, res)
    hand = np.nan_to_num(hydro_layers(dem, grid)["hand"], nan=9999.0)
    have_change = dvv is not None
    chg = np.maximum(dvv, dvh) if have_change else np.full(cls.shape, np.inf)   # unknown -> falls to "other"

    bucket = np.full(cls.shape, -1, np.int8)
    rest = ref.copy()

    def put(i, m):
        nonlocal rest
        sel = rest & m
        bucket[sel] = i
        rest &= ~sel

    put(0, pred)
    put(1, cls == NODATA)
    put(2, cls == PERMANENT)
    if have_change:
        put(3, chg < 1.5)
        put(4, chg < d["debris_change_db"])
        put(5, slope > d["debris_max_slope_deg"])
        put(6, hand > d["debris_max_hand_m"])
    put(7, np.ones(cls.shape, bool))

    px_km2 = res * res / 1e6
    total = float(ref.sum())
    out = {"run": run.name, "products": keys, "ems_km2": round(total * px_km2, 3), "buckets": {},
           "radar_change_available": have_change}
    names = BUCKETS
    for i, name in enumerate(names):
        n = float((bucket == i).sum())
        out["buckets"][name] = {"km2": round(n * px_km2, 3), "pct": round(100 * n / total, 1)}

    # terrain of what EMS mapped vs what we found
    out["ems_on_slope_gt20_pct"] = round(100 * float((ref & (slope > 20)).sum()) / total, 1)

    # position tolerance
    r = max(1, int(round(tol_m / res)))
    st = ndimage.generate_binary_structure(2, 1)
    pred_d = ndimage.binary_dilation(pred, st, iterations=r)
    ref_d = ndimage.binary_dilation(ref, st, iterations=r)
    seen = in_aoi & (cls != NODATA)
    strict_p = float((pred & ref & seen).sum()) / max(float((pred & seen & in_aoi).sum()), 1)
    strict_r = float((pred & ref & seen).sum()) / max(float((ref & seen).sum()), 1)
    tol_r = float((pred_d & ref & seen).sum()) / max(float((ref & seen).sum()), 1)
    tol_p = float((ref_d & pred & seen & in_aoi).sum()) / max(float((pred & seen & in_aoi).sum()), 1)
    out["position_tolerance_m"] = tol_m
    out["strict"] = {"precision": round(strict_p, 3), "recall": round(strict_r, 3)}
    out["with_tolerance"] = {"precision": round(tol_p, 3), "recall": round(tol_r, 3)}
    out["ems_in_features_below_min_size_pct"] = round(100 * float((ref & small).sum()) / total, 1)
    out["ours_km2_in_aoi"] = round(float((pred & in_aoi).sum()) * px_km2, 3)

    # what the segmentation model said about the EMS-mapped area (explains "model alone" scores)
    ptif = run / "layers" / "model_p_flood.tif"
    if ptif.exists():
        import rasterio
        with rasterio.open(ptif) as src:
            pf = src.read(1)
        thr = cfg["model"].get("flood_prob", 0.8)
        m = ref & seen & np.isfinite(pf)
        if m.any():
            v = pf[m]
            found = m & pred
            out["model_on_ems_area"] = {
                "median_p_flood": round(float(np.median(v)), 3),
                f"share_p_ge_{thr}": round(float((v >= thr).mean()), 3),
                "share_p_ge_0.5": round(float((v >= 0.5).mean()), 3),
                "median_p_where_rules_found_it": round(float(np.median(pf[found])), 3) if found.any() else None,
                "km2_p_ge_thr_outside_ems": round(float(((pf >= thr) & ~ref & seen & in_aoi).sum()) * px_km2, 3)}

    ev_day = date.fromisoformat(event_date) if event_date else None
    out["tracks"] = [{"track": p["track"], "before": p["pre_date"], "after": p["post_date"],
                      "after_days_after_event": (date.fromisoformat(p["post_date"]) - ev_day).days if ev_day else None}
                     for p in (tracks or R["inputs"]["sentinel1_pairs"])]
    (run / "validation").mkdir(exist_ok=True)
    (run / "validation" / "misses.json").write_text(json.dumps(out, indent=1))
    _figure(run, bucket, ref, pred, in_aoi, names)
    return out


def _figure(run, bucket, ref, pred, in_aoi, names=BUCKETS):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        rows, cols = np.where(in_aoi)
        if not rows.size:
            return
        sl = (slice(rows.min(), rows.max() + 1), slice(cols.min(), cols.max() + 1))
        img = np.where(ref, bucket, -1)[sl].astype(float)
        img[(pred & ~ref)[sl]] = 8                       # our detections outside EMS
        img[img < 0] = np.nan
        colors = ["#1E6EE6", "#9a9a9a", "#4fc3f7", "#f2c14e", "#f28e2b", "#c03ad6", "#7b3f00", "#e15759", "#2ca02c"]
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_facecolor("#f4f4f4")
        ax.imshow(img, cmap=ListedColormap(colors), vmin=0, vmax=8, interpolation="nearest")
        labels = list(names) + ["ours, not in EMS"]
        for c, lab in zip(colors, labels):
            ax.plot([], [], "s", color=c, label=lab)
        ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
        ax.set_title(f"{run.name}: where the EMS-mapped area went", fontsize=10)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(run / "validation" / "misses.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")


def print_summary(o):
    print(f"\nRun {o['run']}  EMS products: {', '.join(o['products'])}")
    for t in o["tracks"]:
        late = f"  ({t['after_days_after_event']} days after the event)" if t["after_days_after_event"] is not None else ""
        print(f"  track {t['track']}: before {t['before']}  after {t['after']}{late}")
    print(f"\nEMS-mapped area inside the run: {o['ems_km2']} km²   ours inside the EMS areas: {o['ours_km2_in_aoi']} km²")
    print(f"EMS area on slopes > 20 deg: {o['ems_on_slope_gt20_pct']}%   "
          f"in features smaller than our minimum object: {o['ems_in_features_below_min_size_pct']}%\n")
    print(f"{'where the EMS-mapped area went':34s} {'km2':>8s} {'share':>7s}")
    for k, v in o["buckets"].items():
        print(f"{k:34s} {v['km2']:8.3f} {v['pct']:6.1f}%")
    s, t = o["strict"], o["with_tolerance"]
    print(f"\nexact pixels:          precision {s['precision']:.3f}  recall {s['recall']:.3f}")
    print(f"within {o['position_tolerance_m']:.0f} m:           precision {t['precision']:.3f}  recall {t['recall']:.3f}"
          "   (big jump = position/width mismatch, not missed events)")
    m = o.get("model_on_ems_area")
    if m:
        k = next(x for x in m if x.startswith("share_p_ge_") and x != "share_p_ge_0.5")
        print(f"\nsegmentation model on the EMS-mapped area: median p(flood) {m['median_p_flood']}, "
              f"p >= {k[11:]} on {100 * m[k]:.1f}%, p >= 0.5 on {100 * m['share_p_ge_0.5']:.1f}%; "
              f"median p where the rules found flood: {m['median_p_where_rules_found_it']}; "
              f"model flood (p >= {k[11:]}) outside EMS: {m['km2_p_ge_thr_outside_ems']} km²")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ems", required=True)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--event-date", help="YYYY-MM-DD, to show how late the after-image was")
    a = ap.parse_args(argv)
    o = breakdown(a.run, a.ems, a.only, a.event_date)
    print_summary(o)
    print(f"\nSaved {Path(a.run) / 'validation' / 'misses.json'} and misses.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
