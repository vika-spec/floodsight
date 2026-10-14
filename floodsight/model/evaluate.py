"""Score a trained model on chosen Kuro Siwo events, one row per event plus the combined total.

    python -m floodsight.model.evaluate --root D:/data/KuroSiwo --weights models/siamese_unet.pt \
        --events 427 1111006
    # threshold sweep (choose the threshold on the VALIDATION event, then apply it to the test events):
    python -m floodsight.model.evaluate --root D:/data/KuroSiwo --events 1111013 --sweep 0.3 0.4 0.5 0.6 0.7 0.8
    # add --tta to average predictions over flipped copies of each tile (slower, usually smoother)
Writes <weights>.eval.json / <weights>.sweep.json. Also used by train.py for the held-out test events.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .dataset import IGNORE, KuroSiwo, find_samples
from .metrics import confusion, scores

CLASS_NAMES = ["no water", "permanent water", "flood"]


def probabilities(model, pre, post, tta=False):
    """Softmax probabilities (B, 3, H, W); with tta, averaged over the 4 flip combinations."""
    import torch

    if not tta:
        return torch.softmax(model(pre, post).float(), 1)
    acc = 0
    for dims in ((), (-1,), (-2,), (-2, -1)):
        f = (lambda x: torch.flip(x, dims)) if dims else (lambda x: x)
        acc = acc + f(torch.softmax(model(f(pre), f(post)).float(), 1))
    return acc / 4


def sweep(model, samples, device, thresholds, batch=8, px=10.0, workers=0, tta=False) -> dict:
    """Flood IoU / precision / recall per event when 'flood' means p(flood) >= t."""
    import torch
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    by_ev = defaultdict(list)
    for s in samples:
        by_ev[s["event"]].append(s)
    T = np.asarray(thresholds, float)
    out = {}
    model.eval()
    for ev, lst in sorted(by_ev.items()):
        tp = np.zeros(len(T)); fp = np.zeros(len(T)); fn = np.zeros(len(T))
        dl = DataLoader(KuroSiwo(lst, px_m=px), batch_size=batch, num_workers=workers)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                             enabled=device.type == "cuda"):
            for pre, post, lab in tqdm(dl, desc=f"sweep event {ev}", unit="batch", dynamic_ncols=True, leave=False):
                p = probabilities(model, pre.to(device), post.to(device), tta)[:, 2].cpu().numpy()
                lab = lab.numpy()
                v = lab != IGNORE
                truth = (lab == 2) & v
                for k, t in enumerate(T):
                    pred = (p >= t) & v
                    tp[k] += (pred & truth).sum(); fp[k] += (pred & ~truth).sum(); fn[k] += (~pred & truth).sum()
        out[ev] = [{"threshold": float(t), "iou": float(tp[k] / max(tp[k] + fp[k] + fn[k], 1)),
                    "precision": float(tp[k] / max(tp[k] + fp[k], 1)), "recall": float(tp[k] / max(tp[k] + fn[k], 1))}
                   for k, t in enumerate(T)]
    return out


def print_sweep(res: dict) -> float:
    evs = sorted(res)
    T = [r["threshold"] for r in res[evs[0]]]
    print(f"{'threshold':>9} " + " ".join(f"{('IoU ' + e):>14}" for e in evs) + f" {'mean IoU':>9}"
          + "   (precision/recall per event)")
    best_t, best = T[0], -1
    for k, t in enumerate(T):
        ious = [res[e][k]["iou"] for e in evs]
        m = float(np.mean(ious))
        pr = "  ".join(f"{res[e][k]['precision']:.2f}/{res[e][k]['recall']:.2f}" for e in evs)
        print(f"{t:>9.2f} " + " ".join(f"{x:>14.3f}" for x in ious) + f" {m:>9.3f}   {pr}")
        if m > best:
            best, best_t = m, t
    print(f"Best mean flood IoU {best:.3f} at threshold {best_t:.2f}")
    return best_t


def _cm(model, loader, device, n=3, desc="test"):
    import torch
    from tqdm import tqdm

    model.eval()
    cm = np.zeros((n, n), np.int64)
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
        for pre, post, lab in tqdm(loader, desc=desc, unit="batch", dynamic_ncols=True, leave=False):
            pred = model(pre.to(device), post.to(device)).argmax(1).cpu().numpy()
            cm += confusion(pred, lab.numpy(), n, IGNORE)
    return cm


def evaluate_events(model, samples, device, batch=8, px=10.0, dl_kwargs=None) -> dict:
    from torch.utils.data import DataLoader

    dl_kwargs = dict(dl_kwargs or {})
    dl_kwargs.pop("persistent_workers", None)
    by_ev = defaultdict(list)
    for s in samples:
        by_ev[s["event"]].append(s)
    per_event, total = {}, np.zeros((3, 3), np.int64)
    for ev, lst in sorted(by_ev.items()):
        cm = _cm(model, DataLoader(KuroSiwo(lst, px_m=px), batch_size=batch, **dl_kwargs), device,
                 desc=f"test event {ev}")
        total += cm
        sc = scores(cm, CLASS_NAMES)
        sc["tiles"] = len(lst)
        sc["flood_px"] = int(cm[2].sum())
        per_event[ev] = sc
    comb = scores(total, CLASS_NAMES)
    comb["tiles"] = len(samples)
    return {"per_event": per_event, "combined": comb}


def print_table(res: dict) -> None:
    print(f"{'event':>10} {'tiles':>6} {'flood px':>10} {'flood IoU':>9} {'F1':>6} {'prec':>6} {'recall':>6} {'mIoU':>6}")
    rows = list(res["per_event"].items()) + [("ALL", res["combined"])]
    for ev, sc in rows:
        f = sc["per_class"][2]
        print(f"{ev:>10} {sc['tiles']:>6} {sc.get('flood_px', f['support_px']):>10} {f['iou']:>9.3f} {f['f1']:>6.3f} "
              f"{f['precision']:>6.3f} {f['recall']:>6.3f} {sc['miou']:>6.3f}")
    print("(events with very few flood pixels give unstable scores; judge those with care)")


def main(argv=None):
    import torch

    from .infer import load_model

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", default="models/siamese_unet.pt")
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--sweep", type=float, nargs="*", help="flood probability thresholds to compare")
    ap.add_argument("--tta", action="store_true", help="average over flipped copies (with --sweep)")
    a = ap.parse_args(argv)
    model, info = load_model(a.weights)
    samples = [s for s in find_samples(a.root) if s["event"] in set(a.events)]
    if not samples:
        raise SystemExit("No tiles found for those events.")
    if a.sweep:
        res = sweep(model, samples, info["device"], a.sweep, a.batch, info["px_m"], a.workers, a.tta)
        best = print_sweep(res)
        out = Path(a.weights).with_suffix(".sweep.json")
        out.write_text(json.dumps({"tta": a.tta, "best_threshold": best, "events": res}, indent=1))
        print(f"Saved {out}.  Pick the threshold on the VALIDATION event, then set model.flood_prob in "
              "config.yaml (and confirm on the test events with the same --sweep value).")
        return
    res = evaluate_events(model, samples, info["device"], a.batch, info["px_m"], {"num_workers": a.workers})
    print_table(res)
    out = Path(a.weights).with_suffix(".eval.json")
    out.write_text(json.dumps(res, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
