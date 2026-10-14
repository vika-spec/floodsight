"""Score a FloodSight run against Kuro Siwo flood labels (an allowed, hand-checked reference).

    python -m floodsight.validate.kurosiwo --run runs\\ks_427_xxx --root D:\\data\\KuroSiwo\\GRD\\data --event 427

Like ems.py this is checking only: the pipeline never imports it (tests/test_rules.py enforces this).

What it does
  * collects every Kuro Siwo tile of that event that falls inside the run's area and puts its labels on the
    run's grid (10 m -> run grid with the majority value; pixels the label marks invalid are ignored);
  * scores flood (label 2 = flood water) against:
      - the FINAL map (flood water class),
      - the final map counting debris too,
      - the segmentation model on its own (model_p_flood >= the configured threshold), if the run used it;
  * reports pixels that are flood in the labels but inside our radar blind mask separately (not scored);
  * says whether the model saw this event during training (read from the weights file), and compares the
    run's after-image date with the Kuro Siwo image dates, because flood extent changes from day to day.
Outputs: validation/kurosiwo_<event>.json and kurosiwo_<event>.png in the run folder.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..detect.baseline import DEBRIS, FLOOD, NODATA, PERMANENT
from ..model.metrics import binary_scores

UNKNOWN = 255


def model_role(weights, event):
    try:
        import torch
        ck = torch.load(weights, map_location="cpu", weights_only=False)
        for role, evs in (ck.get("split") or {}).items():
            if str(event) in {str(e) for e in evs}:
                return role
        return "not used" if ck.get("split") else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def labels_on_grid(samples, shape, tf, crs):
    """Mosaic Kuro Siwo labels (0 dry, 1 permanent water, 2 flood; invalid -> UNKNOWN) onto the run grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject, transform_bounds

    truth = np.full(shape, UNKNOWN, np.uint8)
    h, w = shape
    x0, y0 = tf * (0, 0)
    x1, y1 = tf * (w, h)
    run_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    used = 0
    for s in samples:
        with rasterio.open(s["label"]) as src:
            if src.crs is None:
                continue
            b = transform_bounds(src.crs, crs, *src.bounds)
            if b[2] < run_box[0] or b[0] > run_box[2] or b[3] < run_box[1] or b[1] > run_box[3]:
                continue
            lab = src.read(1).astype(np.uint8)
            stf, scrs = src.transform, src.crs
        with rasterio.open(s["valid"]) as v:
            valid = v.read(1) == 1
        lab = np.where(valid & (lab <= 2), lab, UNKNOWN).astype(np.uint8)
        dst = np.full(shape, UNKNOWN, np.uint8)
        reproject(lab, dst, src_transform=stf, src_crs=scrs, dst_transform=tf, dst_crs=crs,
                  src_nodata=UNKNOWN, dst_nodata=UNKNOWN, resampling=Resampling.mode)
        put = (dst != UNKNOWN) & (truth == UNKNOWN)
        truth[put] = dst[put]
        used += 1
    return truth, used


def _dates_near(samples, n=40, prefix="MS1"):
    found = set()
    for s in samples[:n]:
        for f in Path(s["folder"]).iterdir():
            if f.name.startswith(prefix):
                m = re.search(r"(20\d\d)[-_]?([01]\d)[-_]?([0-3]\d)", f.name)
                if m:
                    found.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
    return sorted(found)


def score(run, root, event, weights=None, ks_post_date=None, prob=None):
    import rasterio

    from ..core import load_config
    from ..model.dataset import find_samples

    run = Path(run)
    res = json.loads((run / "results.json").read_text())
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        cls, tf, crs = src.read(1), src.transform, src.crs
    samples = [s for s in find_samples(root) if str(s["event"]) == str(event)]
    if not samples:
        raise SystemExit(f"No Kuro Siwo tiles found for event {event} under {root}")
    truth, used = labels_on_grid(samples, cls.shape, tf, crs)
    if used == 0 or (truth != UNKNOWN).sum() == 0:
        raise SystemExit("No labelled Kuro Siwo pixels fall inside this run's area. Check the bbox.")

    labelled = truth != UNKNOWN
    seen = labelled & (cls != NODATA)
    t_flood = truth == 2
    out = {"run": run.name, "event": str(event), "tiles_used": used,
           "labelled_km2": round(float(labelled.sum()) * abs(tf.a * tf.e) / 1e6, 2),
           "label_flood_km2": round(float((labelled & t_flood).sum()) * abs(tf.a * tf.e) / 1e6, 3),
           "label_flood_in_blind_mask_pct": round(100 * float((t_flood & labelled & (cls == NODATA)).sum())
                                                  / max(float((t_flood & labelled).sum()), 1), 1),
           "scores": {}}
    variants = {"final map: flood water": cls == FLOOD,
                "final map: flood water + debris": (cls == FLOOD) | (cls == DEBRIS)}
    ptif = run / "layers" / "model_p_flood.tif"
    thr = prob if prob is not None else load_config()["model"].get("flood_prob", 0.8)
    if ptif.exists():
        with rasterio.open(ptif) as src:
            p = src.read(1)
        variants[f"segmentation model alone (p >= {thr})"] = np.nan_to_num(p, nan=0) >= thr
    for name, pred in variants.items():
        sc = binary_scores(pred, t_flood, seen)
        out["scores"][name] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in sc.items()}
    out["permanent_water_agreement"] = binary_scores(cls == PERMANENT, truth == 1, seen)["iou"]
    # Where did the labelled flood go in our map? If most of it is our "permanent water", our before-image
    # was already flooded (the flood had started before it), not a detection failure.
    lf = t_flood & seen
    n = max(float(lf.sum()), 1.0)
    out["label_flood_in_our_classes_pct"] = {
        "flood or debris": round(100 * float((lf & ((cls == FLOOD) | (cls == DEBRIS))).sum()) / n, 1),
        "permanent water (water already in our before-image)": round(100 * float((lf & (cls == PERMANENT)).sum()) / n, 1),
        "dry / no change": round(100 * float((lf & ~np.isin(cls, [FLOOD, DEBRIS, PERMANENT])).sum()) / n, 1)}

    pairs = res.get("inputs", {}).get("sentinel1_pairs", [])
    out["run_after_dates"] = sorted({p.get("post_date") for p in pairs if p.get("post_date")})
    out["run_before_dates"] = sorted({p.get("pre_date") for p in pairs if p.get("pre_date")})
    out["kurosiwo_post_dates"] = [ks_post_date] if ks_post_date else _dates_near(samples)
    out["kurosiwo_before_dates"] = _dates_near(samples, prefix="SL1")
    out["model_role_for_event"] = model_role(weights or load_config()["model"]["weights"], event)
    out["model_used_in_run"] = ptif.exists()

    # Diagnostics on the Kuro Siwo tiles that overlap this run (same tiles as the labels above)
    near = _tiles_in_box(samples, tf, crs, cls.shape)
    out["diagnostic_tiles"] = len(near)
    out["diagnostics_skipped"] = []
    w = weights or load_config()["model"]["weights"]
    if not near:
        out["diagnostics_skipped"].append("no Kuro Siwo tile overlaps the run area")
    elif not Path(w).exists():
        out["diagnostics_skipped"].append(f"model weights not found: {w}")
    else:
        out["model_on_kurosiwo_images_of_this_area"] = _ks_direct(near, w, thr)
    if near:
        b, why = _compare_backscatter(run, res, near, tf, crs, cls.shape, out["kurosiwo_post_dates"])
        out["backscatter_ours_vs_kurosiwo"] = b
        if why:
            out["diagnostics_skipped"].append(why)

    (run / "validation").mkdir(exist_ok=True)
    (run / "validation" / f"kurosiwo_{event}.json").write_text(json.dumps(out, indent=1))
    _figure(run, event, cls, truth, seen)
    return out


def _tiles_in_box(samples, tf, crs, shape):
    """Kuro Siwo tiles with at least half their area inside the run's grid."""
    import rasterio
    from rasterio.warp import transform_bounds

    h, w = shape
    x0, y0 = tf * (0, 0)
    x1, y1 = tf * (w, h)
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    out = []
    for s in samples:
        with rasterio.open(s["label"]) as src:
            if src.crs is None:
                continue
            b = transform_bounds(src.crs, crs, *src.bounds)
        ix = max(0.0, min(b[2], box[2]) - max(b[0], box[0]))
        iy = max(0.0, min(b[3], box[3]) - max(b[1], box[1]))
        area = max((b[2] - b[0]) * (b[3] - b[1]), 1e-9)
        if ix * iy / area >= 0.5:                      # at least half of the tile inside the run area
            out.append(s)
    return out


def _ks_direct(samples, weights, thr):
    """The same model on Kuro Siwo's OWN images of the tiles inside this run (training-style input)."""
    import torch

    from ..model.dataset import load_sample
    from ..model.evaluate import probabilities
    from ..model.infer import load_model

    model, info = load_model(weights)
    tp = fp = fn = 0
    for s in samples:
        pre, post, lab = load_sample(s)
        with torch.no_grad():
            t = lambda x: torch.from_numpy(x)[None].to(info["device"])
            p = probabilities(model, t(pre), t(post))[0, 2].float().cpu().numpy()
        ok = lab != 255
        pr, tr = (p >= thr) & ok, (lab == 2) & ok
        tp += int((pr & tr).sum()); fp += int((pr & ~tr).sum()); fn += int((~pr & tr).sum())
    return {"tiles": len(samples), "iou": round(tp / max(tp + fp + fn, 1), 3),
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}


def _compare_backscatter(run, res, samples, tf, crs, shape, ks_dates=()):
    """Our after-image vs Kuro Siwo's after-image of the same place (dB): level, agreement and graininess.
    Big differences mean our image preparation differs from the training data's.
    Returns (result or None, reason it was skipped or None)."""
    from rasterio.warp import Resampling, reproject
    import rasterio

    from ..core import Grid, read_to_grid
    from ..detect.baseline import local_std
    from ..model.dataset import _db

    pairs = [p for p in res.get("inputs", {}).get("sentinel1_pairs", []) if "track" in p and "post_date" in p]
    if not pairs:
        return None, "run has no Sentinel-1 pair information"
    # prefer the track whose after-image is on the Kuro Siwo image date (the same acquisition)
    same = [p for p in pairs if p["post_date"] in set(ks_dates)]
    p0 = (same or pairs)[0]
    hits = sorted((run / "inputs").glob(f"s1_{p0['track']}_post_{p0['post_date']}*.tif"))
    if not hits:
        return None, f"after-image file s1_{p0['track']}_post_{p0['post_date']}*.tif not found in {run / 'inputs'}"
    grid = Grid.from_dict(res["grid"])
    ours = _db(read_to_grid(hits[0], grid, band=1))
    ks = np.full(shape, np.nan, np.float32)
    ks_std = np.full(shape, np.nan, np.float32)
    for s in samples:
        with rasterio.open(s["post_vv"]) as src:
            lin = src.read(1).astype(np.float32)
            db = _db(lin)
            for arr, outarr in ((db, ks), (local_std(db, 5), ks_std)):
                dst = np.full(shape, np.nan, np.float32)
                reproject(arr, dst, src_transform=src.transform, src_crs=src.crs, dst_transform=tf, dst_crs=crs,
                          src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.average)
                put = np.isfinite(dst) & ~np.isfinite(outarr)
                outarr[put] = dst[put]
    ok = np.isfinite(ks) & np.isfinite(ours)
    if ok.sum() < 1000:
        return None, f"only {int(ok.sum())} pixels where both after-images have values"
    a, b = ours[ok], ks[ok]
    return {"track": p0["track"], "our_after_date": p0["post_date"], "pixels": int(ok.sum()), "median_ours_db": round(float(np.median(a)), 2),
            "median_kurosiwo_db": round(float(np.median(b)), 2),
            "median_difference_db": round(float(np.median(a - b)), 2),
            "correlation": round(float(np.corrcoef(a, b)[0, 1]), 3),
            "graininess_ours_db": round(float(np.nanmedian(local_std(np.nan_to_num(ours, nan=-15), 5)[ok])), 2),
            "graininess_kurosiwo_db_at_10m": round(float(np.nanmedian(ks_std[ok])), 2),
            "note": "ours at the run's grid; Kuro Siwo averaged onto it (graininess of Kuro Siwo measured at its 10 m)"}, None


def _figure(run, event, cls, truth, seen):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        rows, cols = np.where(truth != UNKNOWN)
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        pred = ((cls == FLOOD) | (cls == DEBRIS))[r0:r1, c0:c1]
        t = (truth == 2)[r0:r1, c0:c1]
        sv = seen[r0:r1, c0:c1]
        img = np.full(t.shape + (3,), 235, np.uint8)
        img[~sv] = (190, 190, 190)
        img[sv & t & pred] = (30, 110, 230)      # both
        img[sv & t & ~pred] = (240, 160, 0)      # missed
        img[sv & ~t & pred] = (220, 40, 40)      # extra
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, interpolation="nearest")
        ax.set_title(f"{run.name} vs Kuro Siwo {event}\nblue = both, orange = missed, red = extra, grey = no label / blind",
                     fontsize=9)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(run / "validation" / f"kurosiwo_{event}.png", dpi=130)
        plt.close(fig)
    except Exception as e:  # noqa: BLE001
        print(f"figure skipped: {e}")


def print_summary(o):
    print(f"\nRun {o['run']} vs Kuro Siwo event {o['event']}  ({o['tiles_used']} tiles, {o['labelled_km2']} km² labelled, "
          f"{o['label_flood_km2']} km² flood in the labels)")
    role = o["model_role_for_event"]
    flag = ("" if role in ("test", "not used") else
            "   <-- could not read the training split from the weights file" if role == "unknown" else
            "   <-- NOT an unseen event for the model; model scores are optimistic")
    print(f"Model's role for this event during training: {role}{flag}")
    print(f"Before-image date(s):  run {', '.join(o.get('run_before_dates') or []) or '?'}   "
          f"Kuro Siwo {', '.join(o.get('kurosiwo_before_dates') or []) or '?'}")
    print(f"After-image date(s):   run {', '.join(o['run_after_dates']) or '?'}   "
          f"Kuro Siwo {', '.join(o['kurosiwo_post_dates']) or 'not found in file names'}")
    print(f"Label flood inside our radar blind mask (not scored): {o['label_flood_in_blind_mask_pct']}%\n")
    print(f"{'what is scored':44s} {'IoU':>6s} {'F1':>6s} {'prec':>6s} {'recall':>7s}")
    for name, s in o["scores"].items():
        print(f"{name:44s} {s['iou']:6.3f} {s['f1']:6.3f} {s['precision']:6.3f} {s['recall']:7.3f}")
    print(f"\npermanent water IoU (our class vs label 1): {o['permanent_water_agreement']:.3f}")
    c = o.get("label_flood_in_our_classes_pct")
    if c:
        print("Where the labelled flood ends up in our map: " + ", ".join(f"{k} {v}%" for k, v in c.items()))
        if c["permanent water (water already in our before-image)"] >= 50:
            print("  -> most of the labelled flood was ALREADY water in our before-image: the flood had started "
                  "before it. Re-run with --before-date set to a day when the area was still dry.")
    d = o.get("model_on_kurosiwo_images_of_this_area")
    if d:
        print(f"\nSame model on Kuro Siwo's own images of the {d['tiles']} tiles in this area: IoU {d['iou']} "
              f"(prec {d['precision']}, recall {d['recall']})  <- compare with 'segmentation model alone' above")
    b = o.get("backscatter_ours_vs_kurosiwo")
    if b:
        print(f"After-image VV ({b['track']} {b['our_after_date']}), ours vs Kuro Siwo: median {b['median_ours_db']} "
              f"vs {b['median_kurosiwo_db']} dB (difference {b['median_difference_db']:+} dB), correlation "
              f"{b['correlation']}, graininess {b['graininess_ours_db']} vs {b['graininess_kurosiwo_db_at_10m']} dB")
    for why in o.get("diagnostics_skipped") or []:
        print(f"diagnostic skipped: {why}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--root", required=True, help="Kuro Siwo folder")
    ap.add_argument("--event", required=True)
    ap.add_argument("--weights", help="model weights (default: config model.weights) to report the event's role")
    ap.add_argument("--ks-post-date", help="Kuro Siwo flood-image date, if it is not in the file names")
    ap.add_argument("--prob", type=float, help="model threshold (default: config model.flood_prob)")
    a = ap.parse_args(argv)
    o = score(a.run, a.root, a.event, a.weights, a.ks_post_date, a.prob)
    print_summary(o)
    print(f"\nSaved {Path(a.run) / 'validation'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
