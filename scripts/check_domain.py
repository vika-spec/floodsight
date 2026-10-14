"""Does the model see the same kind of radar data live as in training? (read-only)

    python scripts/check_domain.py --root D:\\data\\KuroSiwo\\GRD\\data --run runs\\trishuli_shub
    python scripts/check_domain.py --root ... --run runs\\trishuli_case          # the openEO version
    add --model models\\siamese_unet.pt to also compare the model's flood probabilities

Compares Kuro Siwo training tiles with the radar images of a FloodSight run, in the units the model uses
(dB, before normalisation):
  * brightness of VV / VH on flat ground (slope < 5 deg) and on steep ground (> 20 deg)
  * brightness of water: Kuro Siwo flood/permanent-water labels vs the run's rule-detected water
  * speckle level: local standard deviation of VV in 5x5 windows on flat ground (shows filtering differences)
  * optional: the model's flood probability on Kuro Siwo flood vs non-flood pixels, and on the run's
    rule-detected water vs unchanged ground
Nothing is trained or tuned. Output: a table and domain_check.png in the run folder.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def local_std(x, k=5):
    from scipy.ndimage import uniform_filter
    m = uniform_filter(x, k)
    return np.sqrt(np.maximum(uniform_filter(x * x, k) - m * m, 0))


def summ(v):
    v = v[np.isfinite(v)]
    if v.size < 50:
        return None
    return {"n": int(v.size), "p10": float(np.percentile(v, 10)), "median": float(np.median(v)),
            "p90": float(np.percentile(v, 90))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--events", nargs="*", help="Kuro Siwo events to use (default: all)")
    ap.add_argument("--tiles", type=int, default=300, help="Kuro Siwo tiles sampled in total")
    ap.add_argument("--model", help="optional model weights to compare predicted probabilities")
    a = ap.parse_args(argv)

    import rasterio

    from floodsight.core import Grid, read_to_grid
    from floodsight.model.dataset import _db, _read, _slope, find_samples

    run = Path(a.run)
    R = json.loads((run / "results.json").read_text())
    grid = Grid.from_dict(R["grid"])

    # ---------------- Kuro Siwo sample
    samples = find_samples(a.root)
    if a.events:
        samples = [s for s in samples if s["event"] in set(a.events)]
    rng = random.Random(0)
    pick = rng.sample(samples, min(a.tiles, len(samples)))
    ks = {k: [] for k in ("vv_flat", "vh_flat", "vv_steep", "vv_water", "vh_water", "vv_flood", "speck_flat")}
    for s in pick:
        vv, vh = _db(_read(s["post_vv"])), _db(_read(s["post_vh"]))
        lab, valid = _read(s["label"]), _read(s["valid"]) == 1
        sl = _slope(_read(s["dem"]), 10.0) if s.get("dem") else np.zeros_like(vv)
        dry = valid & (lab == 0)
        ks["vv_flat"].append(vv[dry & (sl < 5)]); ks["vh_flat"].append(vh[dry & (sl < 5)])
        ks["vv_steep"].append(vv[dry & (sl > 20)])
        ks["vv_water"].append(vv[valid & (lab == 1)]); ks["vh_water"].append(vh[valid & (lab == 1)])
        ks["vv_flood"].append(vv[valid & (lab == 2)])
        ks["speck_flat"].append(local_std(vv)[dry & (sl < 5)])
    ks = {k: np.concatenate(v) if v else np.zeros(0) for k, v in ks.items()}

    # ---------------- the run (same units: dB of the backscatter the model receives)
    backend = R["inputs"].get("sentinel1_backend", "")
    calib = R["inputs"].get("sentinel1_calibration", "")
    if backend == "sentinelhub":     # older runs (before the calibration tag) used "_shub" = gamma0-terrain
        from floodsight.fetch.shub import cache_tag
        suffix = cache_tag({"sentinel1": {"shub_coefficient": calib}}) if calib else "_shub"
    elif backend == "local":
        suffix = "_local" if "gamma0" in calib or not calib else "_local-s0"
    else:
        suffix = ""
    p0 = R["inputs"]["sentinel1_pairs"][0]
    post_f = run / "inputs" / f"s1_{p0['track']}_post_{p0['post_date']}{suffix}.tif"
    lin = read_to_grid(post_f, grid)
    vv, vh = _db(lin[0]), _db(lin[1])
    dem = read_to_grid(run / "inputs" / "dem.tif", grid, band=1)
    sl = _slope(dem, grid.res)
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        cls = src.read(1)
    ok = np.isfinite(vv) & (cls != 255)
    rn = {"vv_flat": vv[ok & (cls == 0) & (sl < 5)], "vh_flat": vh[ok & (cls == 0) & (sl < 5)],
          "vv_steep": vv[ok & (cls == 0) & (sl > 20)],
          "vv_water": vv[ok & (cls == 3)], "vh_water": vh[ok & (cls == 3)], "vv_flood": vv[ok & (cls == 1)],
          "speck_flat": local_std(np.nan_to_num(vv, nan=-20))[ok & (cls == 0) & (sl < 5)]}

    # graininess of what the MODEL receives: run image resampled to 10 m, then matched to the training level
    from floodsight.core import load_config
    from floodsight.model.infer import match_speckle, speckle_level
    from scipy.ndimage import zoom
    f = grid.res / 10.0
    pre_f = run / "inputs" / f"s1_{p0['track']}_pre_{p0['pre_date']}{suffix}.tif"
    pre_lin = read_to_grid(pre_f, grid)
    up = lambda x: zoom(np.nan_to_num(x, nan=np.nanmedian(x)), f, order=1)
    pre10 = np.stack([up(_db(pre_lin[0])), up(_db(pre_lin[1]))])
    post10 = np.stack([up(vv), up(vh)])
    sl10 = up(sl)
    tgt = load_config()["model"].get("input_speckle_db", 0.4)
    _, post_m, sp = match_speckle(pre10, post10, sl10, tgt)
    print(f"Model input graininess (10 m, flat ground): raw {speckle_level(post10[0], sl10):.2f} dB -> after matching "
          f"{sp['after_db']:.2f} dB (Lee window {sp['window']}; training target {tgt} dB)\n")

    labels = {"vv_flat": "VV dB, flat dry ground", "vh_flat": "VH dB, flat dry ground",
              "vv_steep": "VV dB, steep dry ground (>20 deg)", "vv_water": "VV dB, permanent water",
              "vh_water": "VH dB, permanent water", "vv_flood": "VV dB, flood water",
              "speck_flat": "speckle: local std of VV (dB)"}
    print(f"\nKuro Siwo: {len(pick)} tiles ({'events ' + ', '.join(a.events) if a.events else 'all events'}), 10 m.")
    print(f"Run: {run.name}, track {p0['track']} after-image, backend '{backend or 'openeo'}', {grid.res:g} m.")
    print("Run water = the rule-based map's classes (no labels exist for the run).\n")
    print(f"{'quantity':36s} {'Kuro Siwo p10 / median / p90':>30s} {'Run p10 / median / p90':>28s} {'median shift':>13s}")
    out = {}
    for k, name in labels.items():
        a_, b_ = summ(ks[k]), summ(rn[k])
        fa = f"{a_['p10']:7.1f} {a_['median']:7.1f} {a_['p90']:7.1f}" if a_ else "too few pixels"
        fb = f"{b_['p10']:7.1f} {b_['median']:7.1f} {b_['p90']:7.1f}" if b_ else "too few pixels"
        sh = f"{b_['median'] - a_['median']:+.1f}" if a_ and b_ else "-"
        print(f"{name:36s} {fa:>30s} {fb:>28s} {sh:>13s}")
        out[k] = {"kurosiwo": a_, "run": b_}

    if a.model:
        try:
            import torch  # noqa: F401
            from floodsight.model.evaluate import probabilities
            from floodsight.model.infer import load_model, predict
            from floodsight.model.net import normalise
            model, info = load_model(a.model)
            pf, pn = [], []
            for s in pick[:120]:
                pre = normalise(_db(_read(s["pre_vv"])), _db(_read(s["pre_vh"])), _slope(_read(s["dem"]), 10.0) if s.get("dem") else 0)
                post = normalise(_db(_read(s["post_vv"])), _db(_read(s["post_vh"])), _slope(_read(s["dem"]), 10.0) if s.get("dem") else 0)
                t = lambda x: torch.from_numpy(np.ascontiguousarray(x, np.float32))[None].to(info["device"])
                with torch.no_grad():
                    p = probabilities(model, t(pre), t(post), False)[0, 2].float().cpu().numpy()
                lab, valid = _read(s["label"]), _read(s["valid"]) == 1
                pf.append(p[valid & (lab == 2)]); pn.append(p[valid & (lab == 0)])
            ptif = run / "layers" / "model_p_flood.tif"
            if ptif.exists():
                with rasterio.open(ptif) as src:
                    pr = src.read(1)
                rf, rnc = pr[(cls == 1) & np.isfinite(pr)], pr[(cls == 0) & np.isfinite(pr)]
            else:
                rf = rnc = np.zeros(0)
            print("\nModel flood probability (median, share above 0.8):")
            for name, v in (("Kuro Siwo, labelled flood", np.concatenate(pf)), ("Kuro Siwo, labelled dry", np.concatenate(pn)),
                            ("Run, rule-detected new water", rf), ("Run, unchanged ground", rnc)):
                if v.size:
                    print(f"  {name:32s} median {np.median(v):.2f}   >0.8: {100 * (v > 0.8).mean():5.1f}%   (n={v.size})")
                else:
                    print(f"  {name:32s} no pixels (for the run: rerun it with the model enabled)")
        except Exception as e:  # noqa: BLE001
            print(f"\nModel comparison skipped: {type(e).__name__}: {e}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        keys = ["vv_flat", "vv_steep", "vv_water", "speck_flat"]
        fig, axes = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 3.4))
        for ax, k in zip(axes, keys):
            for v, lab, c in ((ks[k], "Kuro Siwo", "#1E6EE6"), (rn[k], run.name, "#965F28")):
                v = v[np.isfinite(v)]
                if v.size > 50:
                    ax.hist(v, bins=80, range=(np.percentile(v, 0.5), np.percentile(v, 99.5)), density=True,
                            histtype="step", lw=1.6, color=c, label=lab)
            ax.set_title(labels[k], fontsize=9)
            ax.legend(fontsize=7)
        fig.tight_layout()
        png = run / "domain_check.png"
        fig.savefig(png, dpi=120)
        print(f"\nSaved {png}")
    except Exception as e:  # noqa: BLE001
        print(f"Plot skipped: {e}")
    (run / "domain_check.json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
