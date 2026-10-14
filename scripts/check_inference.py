"""Why does the model find almost no flood in real scenes? Test the inference path on labelled tiles. (read-only)

    python scripts/check_inference.py --root D:\\data\\KuroSiwo\\GRD\\data --weights models\\siamese_unet.pt

On Kuro Siwo TEST events (never used for training or model selection; default 427 and 1111006) it scores the
same flood tiles four ways, against the Kuro Siwo labels:
  A  direct           the training-style input (what evaluate.py measures)
  B  pipeline, 10 m   the same tiles through infer.predict, the code the pipeline runs on real scenes
  C  pipeline, 20 m   tiles first degraded to 20 m (2x2 block average of linear backscatter, DEM averaged),
                      then predict at 20 m with speckle matching on: what the model meets in a 20 m run
  D  rules, 20 m      the rule-based detector on the same 20 m tiles (water threshold + 3 dB drop)
Reading the result:
  A >> B  -> a bug in the pipeline's inference path (B must equal A up to tiling effects)
  B >> C  -> the 20 m resolution is what hurts the model on real scenes -> retrain with 20 m augmentation
  C ~ B   -> resolution is not the problem; the difference lies in the scenes (terrain, flood type)
Nothing is trained or changed.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _block(a, f=2, how="mean"):
    H, W = (a.shape[-2] // f) * f, (a.shape[-1] // f) * f
    a = a[..., :H, :W]
    s = a.reshape(*a.shape[:-2], H // f, f, W // f, f)
    if how == "mode":                                    # labels: majority of the 2x2 block
        flat = s.transpose(*range(s.ndim - 4), -4, -2, -3, -1).reshape(*a.shape[:-2], H // f, W // f, f * f)
        from scipy import stats
        return stats.mode(flat, axis=-1, keepdims=False).mode
    return np.nanmean(s, axis=(-3, -1))


def score(p, lab, thr):
    ok = lab != 255
    pred, t = (p >= thr) & ok, (lab == 2) & ok
    tp, fp, fn = (pred & t).sum(), (pred & ~t).sum(), (~pred & t).sum()
    return tp, fp, fn


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", default="models/siamese_unet.pt")
    ap.add_argument("--events", nargs="+", default=["427", "1111006"])
    ap.add_argument("--tiles", type=int, default=80, help="flood tiles sampled (with >= 5% flood)")
    ap.add_argument("--prob", type=float, help="flood threshold (default: config model.flood_prob)")
    a = ap.parse_args(argv)

    import torch

    from floodsight.core import load_config
    from floodsight.detect.baseline import FLOOD, detect_change
    from floodsight.model.dataset import _db, _read, _slope, find_samples, load_sample
    from floodsight.model.evaluate import probabilities
    from floodsight.model.infer import load_model, predict

    cfg = load_config()
    thr = a.prob if a.prob is not None else cfg["model"].get("flood_prob", 0.8)
    target = cfg["model"].get("input_speckle_db", 0.4)
    model, info = load_model(a.weights)
    dev = info["device"]
    samples = [s for s in find_samples(a.root) if str(s["event"]) in set(a.events)]
    rng = random.Random(0)
    rng.shuffle(samples)
    pick = []
    for s in samples:
        lab = _read(s["label"])
        if ((lab == 2) & (_read(s["valid"]) == 1)).mean() >= 0.05:
            pick.append(s)
        if len(pick) >= a.tiles:
            break
    print(f"{len(pick)} flood tiles from events {', '.join(a.events)}; threshold {thr}\n")
    tot = {k: np.zeros(3) for k in "ABCD"}
    mean_p = {k: [] for k in "ABC"}
    for s in pick:
        pre, post, lab = load_sample(s)
        with torch.no_grad():
            t = lambda x: torch.from_numpy(x)[None].to(dev)
            pA = probabilities(model, t(pre), t(post))[0, 2].float().cpu().numpy()
        tot["A"] += score(pA, lab, thr)
        dem = _read(s["dem"]) if s.get("dem") else np.zeros(lab.shape, np.float32)
        lin = {k: _read(s[k]) for k in ("pre_vv", "pre_vh", "post_vv", "post_vh")}
        pre_db = np.stack([_db(lin["pre_vv"]), _db(lin["pre_vh"])])
        post_db = np.stack([_db(lin["post_vv"]), _db(lin["post_vh"])])
        slope10 = _slope(dem, 10.0)
        pB = predict(pre_db, post_db, slope10, 10.0, a.weights, speckle_target=None)["p_flood"]
        tot["B"] += score(np.nan_to_num(pB), lab, thr)
        # degrade to 20 m like a 20 m run: average linear power over 2x2, then dB
        pre20 = np.stack([_db(_block(lin["pre_vv"])), _db(_block(lin["pre_vh"]))])
        post20 = np.stack([_db(_block(lin["post_vv"])), _db(_block(lin["post_vh"]))])
        dem20 = _block(dem)
        slope20 = _slope(dem20, 20.0)
        lab20 = _block(np.where(lab == 255, 255, lab), how="mode")
        pC = predict(pre20, post20, slope20, 20.0, a.weights, speckle_target=target)["p_flood"]
        tot["C"] += score(np.nan_to_num(pC), lab20, thr)
        # rules on the same 20 m tile (no HAND on a single tile: water zone limited by slope only)
        hand = np.zeros(slope20.shape, np.float32)
        r = detect_change(pre20, post20, np.zeros(slope20.shape, bool), hand, slope20,
                          np.zeros(slope20.shape, bool), 20.0, cfg)
        tot["D"] += score((r["cls"] == FLOOD).astype(float), lab20, 0.5)
        for k, p, L in (("A", pA, lab), ("B", pB, lab), ("C", pC, lab20)):
            m = (L == 2)
            if m.any():
                mean_p[k].append(float(np.nanmean(p[m])))

    names = {"A": "A direct (as in training)", "B": "B pipeline path, 10 m", "C": "C pipeline path, 20 m",
             "D": "D rules, 20 m"}
    print(f"{'':28s} {'flood IoU':>9s} {'precision':>9s} {'recall':>7s} {'mean p(flood) on flood':>24s}")
    for k in "ABCD":
        tp, fp, fn = tot[k]
        iou = tp / max(tp + fp + fn, 1)
        pr = tp / max(tp + fp, 1)
        rc = tp / max(tp + fn, 1)
        mp = f"{np.mean(mean_p[k]):.2f}" if mean_p.get(k) else "-"
        print(f"{names[k]:28s} {iou:9.3f} {pr:9.3f} {rc:7.3f} {mp:>24s}")
    print("\nA >> B: inference bug.  B >> C: 20 m resolution hurts -> retrain with 20 m augmentation.  "
          "C ~ B: the gap on real scenes comes from the scenes themselves.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
