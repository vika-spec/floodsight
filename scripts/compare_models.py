"""Which flood model works best? A fair comparison on the same Kuro Siwo tiles. (read-only for the models)

    # 1) train the comparison models with the SAME budget (see docs/model_comparison.md for the commands)
    # 2) train the classical baseline and compare everything:
    python scripts/compare_models.py --root D:\\data\\KuroSiwo\\GRD\\data ^
        --weights models\\cmp_siamese.pt models\\cmp_early_fusion.pt models\\cmp_post_only.pt ^
        --gbm models\\cmp_gbm.joblib --train-gbm --extra-events 205

Methods compared (all see exactly the same tiles):
  deep models   every checkpoint given with --weights (siamese / early_fusion / post_only)
  gbm           gradient-boosted trees on per-pixel features (before, after, change, slope, 5x5 context);
                a classical machine-learning baseline (scikit-learn HistGradientBoostingClassifier)
  rules         FloodSight's physical rules alone (water threshold + 3 dB darkening), no learning

Fairness rules (also printed with the results):
  * one event split for everything: the train/val/test events stored in the FIRST checkpoint; a checkpoint
    trained on a different split is refused;
  * each method's flood threshold is chosen on the VALIDATION events only (best mean flood IoU on a fixed
    grid), then applied unchanged to the test events;
  * test events are scored once; --extra-events adds events outside the split (e.g. 205, never used in
    training or selection) and labels them as such;
  * --max-tiles samples the same tiles (fixed seed) for every method.
Output: printed table + models/model_comparison.json (copy the table into the report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

GRID = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ----------------------------------------------------------------------------------------------- features
def pixel_features(pre, post):
    """(F, H, W) features for the classical baseline from the normalised network inputs (VV, VH, slope)."""
    from scipy.ndimage import uniform_filter

    d = post[:2] - pre[:2]
    m = lambda x: uniform_filter(x, 5)
    sd = np.sqrt(np.maximum(m(post[0] ** 2) - m(post[0]) ** 2, 0))
    return np.stack([pre[0], pre[1], post[0], post[1], d[0], d[1], post[2],
                     m(post[0]), m(post[1]), m(d[0]), m(d[1]), sd]).astype(np.float32)


def train_gbm(train_tiles, out, px, n_px=2000, seed=0):
    from joblib import dump
    from sklearn.ensemble import HistGradientBoostingClassifier

    from floodsight.model.dataset import IGNORE, load_sample

    rng = np.random.default_rng(seed)
    X, y = [], []
    for s in train_tiles:
        pre, post, lab = load_sample(s, px)
        ok = np.flatnonzero(lab.ravel() != IGNORE)
        if not ok.size:
            continue
        pick = rng.choice(ok, min(n_px, ok.size), replace=False)
        f = pixel_features(pre, post).reshape(12, -1)[:, pick].T
        X.append(f)
        y.append((lab.ravel()[pick] == 2).astype(np.uint8))
    X, y = np.concatenate(X), np.concatenate(y)
    w = np.where(y == 1, 0.5 / max(y.mean(), 1e-6), 0.5 / max(1 - y.mean(), 1e-6))   # balanced classes
    t0 = time.time()
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=63, random_state=seed)
    clf.fit(X, y, sample_weight=w)
    dump({"model": clf, "features": 12, "train_pixels": int(len(y)), "train_tiles": len(train_tiles),
          "train_minutes": round((time.time() - t0) / 60, 2)}, out)
    print(f"GBM trained on {len(y):,} pixels from {len(train_tiles)} tiles ({(time.time() - t0) / 60:.1f} min) -> {out}")


# ----------------------------------------------------------------------------------------------- scoring
class Tally:
    def __init__(self, grid):
        self.T = np.asarray(grid, float)
        self.c = defaultdict(lambda: np.zeros((3, len(self.T))))

    def add(self, ev, p, lab):
        from floodsight.model.dataset import IGNORE
        v = lab != IGNORE
        truth = (lab == 2) & v
        for k, t in enumerate(self.T):
            pred = (p >= t) & v
            self.c[ev][:, k] += [(pred & truth).sum(), (pred & ~truth).sum(), (~pred & truth).sum()]

    def score(self, evs, k):
        tp, fp, fn = sum(self.c[e][:, k] for e in evs)
        return {"iou": tp / max(tp + fp + fn, 1), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1)}


def _round(x):
    if isinstance(x, dict):
        return {k: _round(v) for k, v in x.items()}
    if isinstance(x, (float, np.floating)):
        return round(float(x), 3)
    return x


def run_method(name, prob_fn, tiles_by_ev, grid):
    tally, n, sec = Tally(grid), 0, 0.0
    for ev, tiles in tiles_by_ev.items():
        for s in tiles:
            t0 = time.perf_counter()
            p, lab = prob_fn(s)
            sec += time.perf_counter() - t0
            n += 1
            tally.add(ev, p, lab)
        print(f"  {name}: event {ev} done ({len(tiles)} tiles)", flush=True)
    return tally, 1000 * sec / max(n, 1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--gbm", help="classical baseline model file (.joblib)")
    ap.add_argument("--train-gbm", action="store_true", help="(re)train the classical baseline first")
    ap.add_argument("--no-rules", action="store_true")
    ap.add_argument("--extra-events", nargs="*", default=[], help="events outside the split (never trained on)")
    ap.add_argument("--max-tiles", type=int, default=600, help="tiles per event (same for every method; 0 = all)")
    ap.add_argument("--gbm-train-tiles", type=int, default=400)
    ap.add_argument("--out", default="models/model_comparison.json")
    a = ap.parse_args(argv)
    if not a.weights:
        raise SystemExit("Give at least one checkpoint with --weights (its stored split defines the events).")

    import torch

    from floodsight.core import load_config
    from floodsight.detect.baseline import FLOOD, detect_change
    from floodsight.model.dataset import _db, _read, _slope, find_samples, load_sample
    from floodsight.model.evaluate import probabilities
    from floodsight.model.infer import load_model

    ck0 = torch.load(a.weights[0], map_location="cpu", weights_only=False)
    split = ck0.get("split") or {}
    if not split.get("val") or not split.get("test"):
        raise SystemExit(f"{a.weights[0]} has no stored train/val/test split.")
    for w in a.weights[1:]:
        sp = torch.load(w, map_location="cpu", weights_only=False).get("split") or {}
        if {k: sorted(map(str, v)) for k, v in sp.items()} != {k: sorted(map(str, v)) for k, v in split.items()}:
            raise SystemExit(f"{w} was trained on a different event split than {a.weights[0]}: not comparable.")
    roles = {str(e): r for r, evs in split.items() for e in evs}
    for e in a.extra_events:
        if str(e) in roles:
            raise SystemExit(f"--extra-events {e} is already a '{roles[str(e)]}' event of the split.")
    samples = find_samples(a.root)
    by_ev = defaultdict(list)
    for s in samples:
        by_ev[str(s["event"])].append(s)
    rng = np.random.default_rng(0)

    def pick(ev):
        lst = sorted(by_ev.get(str(ev), []), key=lambda s: s["folder"])
        if a.max_tiles and len(lst) > a.max_tiles:
            lst = [lst[i] for i in sorted(rng.choice(len(lst), a.max_tiles, replace=False))]
        return lst

    val = {str(e): pick(e) for e in split["val"]}
    test = {str(e): pick(e) for e in split["test"]}
    extra = {str(e): pick(e) for e in a.extra_events}
    evals = {**val, **test, **extra}
    print("Event split (from the first checkpoint):", {k: sorted(map(str, v)) for k, v in split.items()})
    print("Tiles per event:", {e: len(t) for e, t in evals.items()}, "\n")
    px = ck0.get("px_m", 10.0)
    cfg = load_config()

    methods = {}
    for w in a.weights:
        model, info = load_model(w)
        ck = torch.load(w, map_location="cpu", weights_only=False)
        dev = info["device"]

        def fn(s, model=model, dev=dev):
            pre, post, lab = load_sample(s, px)
            with torch.no_grad():
                t = lambda x: torch.from_numpy(x)[None].to(dev)
                p = probabilities(model, t(pre), t(post))[0, 2].float().cpu().numpy()
            return p, lab
        name = f"{info.get('arch', 'siamese')}{' + terrain loss' if ck.get('terrain_loss') else ''} ({Path(w).stem})"
        methods[name] = {"fn": fn, "kind": "deep", "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                         "epoch": ck.get("epoch"), "budget": ck.get("budget")}

    if a.gbm:
        if a.train_gbm or not Path(a.gbm).exists():
            tr = [s for e in split["train"] for s in by_ev.get(str(e), [])]
            tr = [tr[i] for i in np.random.default_rng(1).choice(len(tr), min(a.gbm_train_tiles, len(tr)), replace=False)]
            train_gbm(tr, a.gbm, px)
        from joblib import load
        g = load(a.gbm)

        def fn(s, clf=g["model"]):
            pre, post, lab = load_sample(s, px)
            f = pixel_features(pre, post)
            return clf.predict_proba(f.reshape(12, -1).T)[:, 1].reshape(lab.shape), lab
        methods["gradient-boosted trees (pixel features)"] = {"fn": fn, "kind": "classical ML",
                                                              "budget": {k: g[k] for k in ("train_pixels", "train_tiles", "train_minutes")}}

    if not a.no_rules:
        def fn(s):
            _, _, lab = load_sample(s, px)
            dem = _read(s["dem"]) if s.get("dem") else np.zeros(lab.shape, np.float32)
            sl = _slope(dem, px)
            pre_db = np.stack([_db(_read(s["pre_vv"])), _db(_read(s["pre_vh"]))])
            post_db = np.stack([_db(_read(s["post_vv"])), _db(_read(s["post_vh"]))])
            z = np.zeros(lab.shape, bool)
            r = detect_change(pre_db, post_db, z, np.zeros(lab.shape, np.float32), sl, z, px, cfg)
            return (r["cls"] == FLOOD).astype(np.float32), lab
        methods["physical rules (no learning)"] = {"fn": fn, "kind": "rules"}

    results = {}
    for name, m in methods.items():
        print(f"Scoring {name} ...", flush=True)
        tally, ms = run_method(name, m["fn"], evals, GRID if m["kind"] != "rules" else [0.5])
        if m["kind"] == "rules":
            k = 0
        else:   # threshold chosen on the validation events only
            k = int(np.argmax([np.mean([tally.score([e], j)["iou"] for e in val]) for j in range(len(GRID))]))
        thr = None if m["kind"] == "rules" else GRID[k]
        res = {"kind": m["kind"], "threshold_from_val": thr, "ms_per_tile": round(ms, 1),
               "params_m": m.get("params_m"), "epoch": m.get("epoch"), "budget": m.get("budget"),
               "val": tally.score(list(val), k), "test": {e: tally.score([e], k) for e in test},
               "test_combined": tally.score(list(test), k), "extra": {e: tally.score([e], k) for e in extra}}
        results[name] = _round(res)

    def f(s):
        return f"{s['iou']:.3f} ({s['precision']:.2f}/{s['recall']:.2f})"
    tev, xev = list(test), list(extra)
    print("\nFlood IoU (precision/recall). Threshold chosen on validation events " + ", ".join(val)
          + "; test events " + ", ".join(tev) + (f"; extra unseen events {', '.join(xev)}" if xev else ""))
    head = ["method", "val"] + [f"test {e}" for e in tev] + ["test all"] + [f"unseen {e}" for e in xev] + ["ms/tile"]
    print(" | ".join(head))
    for name, r in results.items():
        row = [name, f(r["val"])] + [f(r["test"][e]) for e in tev] + [f(r["test_combined"])] + \
              [f(r["extra"][e]) for e in xev] + [str(r["ms_per_tile"])]
        print(" | ".join(row))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"split": {k: sorted(map(str, v)) for k, v in split.items()},
                                       "extra_events": xev, "tiles_per_event": {e: len(t) for e, t in evals.items()},
                                       "grid": GRID, "results": results}, indent=1, default=float))
    print(f"\nSaved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
