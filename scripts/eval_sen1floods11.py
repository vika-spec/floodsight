"""Score a FloodSight model on Sen1Floods11 (Bonafilia et al. 2020), a second listed dataset it was NOT trained on.

Sen1Floods11 has ONE Sentinel-1 image per chip (no before-image) and hand labels of all surface water. So only a
model that needs no before-image can be scored here: the after-only U-Net from the model comparison
(`--arch post_only`). The two-date Siamese U-Net cannot be scored on this dataset, which the report states.

    python scripts/eval_sen1floods11.py --root D:\\data\\Sen1Floods11 --weights models\\cmp_post_only.pt

Expected layout under --root (from the dataset's v1.1 bucket):
    S1Hand/<chip>_S1Hand.tif        VV, VH backscatter in dB, 512 x 512 at 10 m
    LabelHand/<chip>_LabelHand.tif  1 water, 0 not water, -1 no data
    flood_test_data.csv, flood_bolivia_data.csv   (split files; first column = S1 chip file name)
Any other folder names can be given with --s1 / --labels.

Water = the model's 'permanent water' or 'flood' class (argmax, no threshold chosen here). Terrain slope is not
in Sen1Floods11, so it is set to 0 (the model's slope channel is then uninformative; stated in the output).
Also printed: the dataset's own Otsu-threshold labels (S1OtsuLabelHand) scored the same way, if present, as a
simple reference. Nothing is tuned on these results.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _read(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def _chips(split_csv: Path) -> list[str]:
    """Chip names from a split file (rows like 'Bolivia_103757_S1Hand.tif,Bolivia_103757_LabelHand.tif')."""
    out = []
    with open(split_csv, newline="") as f:
        for row in csv.reader(f):
            if row and row[0].strip().endswith(".tif"):
                out.append(row[0].strip())
    return out


def _chip_id(name: str) -> str:
    return name.replace("_S1Hand.tif", "").replace("_LabelHand.tif", "").replace(".tif", "")


def score(pred: np.ndarray, lab: np.ndarray) -> dict:
    ok = lab >= 0
    p, t = pred & ok, (lab == 1) & ok
    tp, fp, fn = float((p & t).sum()), float((p & ~t).sum()), float((~p & t & ok).sum())
    return {"tp": tp, "fp": fp, "fn": fn}


def summary(acc: dict) -> dict:
    tp, fp, fn = acc["tp"], acc["fp"], acc["fn"]
    return {"iou": round(tp / max(tp + fp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "recall": round(tp / max(tp + fn, 1), 3)}


def main(argv=None):
    import torch

    from floodsight.model.net import build_model, normalise

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", required=True, help="an after-only model (train.py --arch post_only)")
    ap.add_argument("--s1", default="S1Hand")
    ap.add_argument("--labels", default="LabelHand")
    ap.add_argument("--otsu", default="S1OtsuLabelHand")
    ap.add_argument("--splits", nargs="*", default=["flood_test_data.csv", "flood_bolivia_data.csv"])
    ap.add_argument("--out", default="models/sen1floods11_eval.json")
    a = ap.parse_args(argv)
    root = Path(a.root)
    ck = torch.load(a.weights, map_location="cpu", weights_only=False)
    arch = ck.get("arch", "siamese")
    if arch != "post_only":
        print(f"This checkpoint is '{arch}', which needs a before-image; Sen1Floods11 has none. "
              "Use the after-only model (train.py --arch post_only).")
        return 2
    m = build_model(arch, ck.get("encoder", "resnet34"))
    m.load_state_dict(ck["state_dict"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(dev).eval()
    print(f"Model: {Path(a.weights).name} ({arch}, {ck.get('encoder')}), trained on Kuro Siwo only; "
          "Sen1Floods11 never used for training or selection. Slope channel set to 0 (no DEM in Sen1Floods11).")
    results = {}
    for sp in a.splits:
        f = next(iter(root.rglob(sp)), None)
        if f is None:
            print(f"  split file {sp} not found under {root}; skipped")
            continue
        chips = [_chip_id(c) for c in _chips(f)]
        acc_m = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        acc_o = {"tp": 0.0, "fp": 0.0, "fn": 0.0}
        n, n_otsu = 0, 0
        for cid in chips:
            s1 = next(iter(root.rglob(f"{cid}_{a.s1}.tif")), None)
            lb = next(iter(root.rglob(f"{cid}_{a.labels}.tif")), None)
            if s1 is None or lb is None:
                continue
            img, lab = _read(s1), _read(lb)[0]
            x = normalise(img[0], img[1], np.zeros_like(img[0]))
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(x))[None].to(dev)
                cls = m(t, t).argmax(1)[0].cpu().numpy()
            water = (cls == 1) | (cls == 2)
            water &= np.isfinite(img[0])
            for k, v in score(water, lab).items():
                acc_m[k] += v
            n += 1
            ot = next(iter(root.rglob(f"{cid}_{a.otsu}.tif")), None)
            if ot is not None:
                for k, v in score(_read(ot)[0] == 1, lab).items():
                    acc_o[k] += v
                n_otsu += 1
        results[sp] = {"chips": n, "model": summary(acc_m), "otsu_reference": summary(acc_o) if n_otsu else None}
        r = results[sp]
        line = f"{sp:28s} {n:4d} chips  model IoU {r['model']['iou']:.3f} (P {r['model']['precision']:.3f}, R {r['model']['recall']:.3f})"
        if r["otsu_reference"]:
            o = r["otsu_reference"]
            line += f"   dataset's Otsu labels IoU {o['iou']:.3f}"
        print(line)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
