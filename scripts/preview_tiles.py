"""Make Kuro Siwo tiles (and model predictions) visible.

The .tif files look black in normal image viewers because they store radar backscatter as small decimal
numbers (about 0.001-0.5), while viewers expect brightness 0-255. This script converts to decibels,
stretches the contrast, and draws each tile as:  before | after | change | label (| prediction | errors)

Which tiles to show (--select):
  flood   tiles with at least 5% flood in the label (default)
  random  any tiles, including ones with no flood at all
  fp      tiles where the model raises the MOST FALSE ALARMS   (needs --model)
  fn      tiles where the model MISSES the most flood          (needs --model)

    python scripts/preview_tiles.py --root D:\\data\\KuroSiwo --n 8
    python scripts/preview_tiles.py --root D:\\data\\KuroSiwo --n 8 --event 427 --model models/siamese_unet.pt --select fp
Output: previews/tiles_<event>_<select>[_pred].png
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floodsight.model.dataset import IGNORE, _db, _read, find_samples, load_sample  # noqa: E402

LABEL_COLORS = np.array([[235, 235, 225], [120, 170, 220], [30, 110, 230], [180, 180, 180]], np.uint8)
FLOOD = 2


def stretch(db, lo=-25, hi=0):
    return np.clip((np.nan_to_num(db, nan=lo) - lo) / (hi - lo), 0, 1)


def predict(model, info, s):
    import torch

    xp, xq, lab = load_sample(s)
    with torch.no_grad():
        logits = model(torch.from_numpy(xp)[None].to(info["device"]), torch.from_numpy(xq)[None].to(info["device"]))
    return logits.argmax(1)[0].cpu().numpy(), lab


def error_rgb(pred, lab):
    """green = flood found, red = false alarm, orange = missed flood, white = correct non-flood."""
    img = np.full(pred.shape + (3,), 245, np.uint8)
    valid = lab != IGNORE
    img[valid & (pred == FLOOD) & (lab == FLOOD)] = (40, 160, 60)
    img[valid & (pred == FLOOD) & (lab != FLOOD)] = (215, 40, 40)
    img[valid & (pred != FLOOD) & (lab == FLOOD)] = (245, 160, 30)
    img[~valid] = (190, 190, 190)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--event")
    ap.add_argument("--select", choices=["flood", "random", "fp", "fn"], default="flood")
    ap.add_argument("--candidates", type=int, default=400, help="tiles scanned to find the worst (fp/fn)")
    ap.add_argument("--model", help="optional trained weights to show predictions")
    ap.add_argument("--out", default="previews")
    a = ap.parse_args()
    if a.select in ("fp", "fn") and not a.model:
        ap.error("--select fp/fn needs --model")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    samples = find_samples(a.root)
    if a.event:
        samples = [s for s in samples if s["event"] == str(a.event)]
    if not samples:
        raise SystemExit("No tiles found.")
    random.Random(0).shuffle(samples)

    model = info = None
    if a.model:
        from floodsight.model.infer import load_model
        model, info = load_model(a.model)

    preds = {}
    if a.select == "random":
        picked = samples[:a.n]
    elif a.select == "flood":
        picked = [s for s in samples[: a.n * 20] if (_read(s["label"]) == FLOOD).mean() > 0.05][:a.n]
        picked = picked or samples[:a.n]
    else:
        from tqdm import tqdm
        scored = []
        for s in tqdm(samples[:a.candidates], desc=f"scanning for worst {a.select}", unit="tile"):
            pr, lab = predict(model, info, s)
            preds[s["folder"]] = pr
            valid = lab != IGNORE
            if a.select == "fp":
                bad = valid & (pr == FLOOD) & (lab != FLOOD)
            else:
                bad = valid & (pr != FLOOD) & (lab == FLOOD)
            scored.append((int(bad.sum()), s))
        scored.sort(key=lambda x: -x[0])
        picked = [s for _, s in scored[:a.n]]

    cols = 6 if model is not None else 4
    fig, axes = plt.subplots(len(picked), cols, figsize=(cols * 2.5, len(picked) * 2.5), squeeze=False)
    titles = ["Before (radar VV)", "After (radar VV)", "Change (after - before)", "Label"] + \
             (["Model prediction", "Errors"] if model is not None else [])
    for i, s in enumerate(picked):
        pre = _db(_read(s["pre_vv"]))
        post = _db(_read(s["post_vv"]))
        lab = _read(s["label"]).astype(int)
        panels = [(stretch(pre), "gray", None), (stretch(post), "gray", None),
                  (np.clip(post - pre, -10, 10), "RdBu_r", (-10, 10)), (LABEL_COLORS[np.clip(lab, 0, 3)], None, None)]
        info_txt = f"flood {100 * (lab == FLOOD).mean():.0f}%"
        if model is not None:
            pr = preds.get(s["folder"])
            if pr is None:
                pr, _ = predict(model, info, s)
            lab_m = load_sample(s)[2]
            panels += [(LABEL_COLORS[pr], None, None), (error_rgb(pr, lab_m), None, None)]
            v = lab_m != IGNORE
            fp = (v & (pr == FLOOD) & (lab_m != FLOOD)).mean() * 100
            fn = (v & (pr != FLOOD) & (lab_m == FLOOD)).mean() * 100
            info_txt += f"\nfalse alarm {fp:.0f}%  missed {fn:.0f}%"
        for j, (img, cmap, lim) in enumerate(panels):
            ax = axes[i, j]
            if cmap:
                ax.imshow(img, cmap=cmap, vmin=lim[0] if lim else None, vmax=lim[1] if lim else None)
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(titles[j], fontsize=9)
        axes[i, 0].set_ylabel(f"event {s['event']}\n{info_txt}", fontsize=7)

    handles = [Patch(color=LABEL_COLORS[k] / 255, label=t) for k, t in
               enumerate(["No water", "Permanent water", "Flood", "Other/ignored"])]
    if model is not None:
        handles += [Patch(color=np.array(c) / 255, label=t) for c, t in
                    (((40, 160, 60), "Flood found"), ((215, 40, 40), "False alarm"), ((245, 160, 30), "Missed flood"))]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
    fig.text(0.5, 0.045, "Change: blue = got darker (typical of new water), red = got brighter", ha="center",
             fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    p = out / f"tiles_{a.event or 'mixed'}_{a.select}{'_pred' if model is not None else ''}.png"
    fig.savefig(p, dpi=110)
    print(f"Saved {p}")


if __name__ == "__main__":
    main()
