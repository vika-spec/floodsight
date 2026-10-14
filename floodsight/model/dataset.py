"""Kuro Siwo loader (Bountos et al., NeurIPS 2024 Datasets & Benchmarks).

Each Kuro Siwo sample folder holds Sentinel-1 GRD patches named by prefix:
  MS1_IVV / MS1_IVH  – image during the flood ("post")
  SL1_IVV / SL1_IVH  – pre-event image 1 ("pre")      (SL2_* = second pre-event image, unused here)
  MK0_DEM            – elevation
  MK0_MLU            – label (water classes)          MK0_MNA – valid-pixel mask (1 = valid)
SAR values are linear backscatter; we convert to dB so training matches what the pipeline produces.

BEFORE TRAINING run `python -m floodsight.model.dataset --inspect <root>` once. It prints the label values
actually present so you can confirm LABEL_MAP (default: 0 no water, 1 permanent water, 2 flood).

Splits are by EVENT (activation id), never by random tile, so the test score is honest:
the model has never seen any tile from a test event.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PREFIX = {"post_vv": "MS1_IVV", "post_vh": "MS1_IVH", "pre_vv": "SL1_IVV", "pre_vh": "SL1_IVH",
          "dem": "MK0_DEM", "label": "MK0_MLU", "valid": "MK0_MNA"}
LABEL_MAP = {0: 0, 1: 1, 2: 2}      # documented: 0 no water, 1 permanent water, 2 flood; others -> ignore
IGNORE = 255


def _event_from_json(folder: Path) -> str | None:
    """Some Kuro Siwo layouts keep per-tile metadata ({"actid": ..., "aoiid": ...}) next to the images."""
    for j in folder.glob("*.json"):
        try:
            info = json.loads(j.read_text())
        except Exception:  # noqa: BLE001
            continue
        info = info.get("info", info) if isinstance(info, dict) else {}
        if "actid" in info:
            return str(info["actid"])
    return None


def find_samples(root: str | Path, event_level: int | None = None) -> list[dict]:
    """Walk the dataset once; a sample = a folder containing all required prefixes.

    The flood event (activation) of each tile comes from per-tile JSON metadata if present, otherwise from
    the folder tree: by default the first folder level below `root` where tiles actually split into
    several groups (e.g. root/GRD/<activation>/<aoi>/<tile> -> the <activation> level). Override with
    event_level (0 = first folder below root).
    """
    root = Path(root)
    samples = []
    for f in root.rglob("MS1_IVV*"):
        folder = f.parent
        files = {p.name: p for p in folder.iterdir()}
        rec = {}
        for k, pre in PREFIX.items():
            hit = next((p for n, p in files.items() if n.startswith(pre)), None)
            if hit is None and k not in ("dem",):
                rec = None
                break
            rec[k] = str(hit) if hit else None
        if rec:
            rec["folder"] = str(folder)
            rec["_parts"] = folder.relative_to(root).parts
            rec["event"] = _event_from_json(folder)
            samples.append(rec)

    if samples and any(s["event"] is None for s in samples):
        level = event_level
        if level is None:
            depth = min(len(s["_parts"]) for s in samples)
            level = 0
            for d in range(max(depth - 1, 0)):        # never use the tile folder itself
                level = d
                if len({s["_parts"][d] for s in samples}) > 1:
                    break
        for s in samples:
            if s["event"] is None:
                s["event"] = s["_parts"][level] if len(s["_parts"]) > level else "unknown"
    for s in samples:
        s.pop("_parts", None)
    return samples


def split_by_event(samples, val_events=None, test_events=None, seed=0, val_frac=0.12):
    events = sorted({s["event"] for s in samples})
    test_events = set(test_events or [])
    rng = random.Random(seed)
    rest = [e for e in events if e not in test_events]
    if val_events:
        val = set(val_events)
    else:
        rng.shuffle(rest)
        val = set(rest[: max(1, int(len(rest) * val_frac))])
    out = defaultdict(list)
    for s in samples:
        k = "test" if s["event"] in test_events else "val" if s["event"] in val else "train"
        out[k].append(s)
    return out


def _read(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float32)


def _db(lin):
    return 10 * np.log10(np.clip(np.nan_to_num(lin, nan=0.0), 1e-5, None))


def _slope(dem, px_m):
    gy, gx = np.gradient(np.nan_to_num(dem, nan=float(np.nanmean(dem)) if np.isfinite(dem).any() else 0.0), px_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def load_sample(s: dict, px_m: float = 10.0):
    from .net import normalise

    pre_vv = _read(s["pre_vv"])
    dem = _read(s["dem"]) if s.get("dem") else None
    slope = _slope(dem, px_m) if dem is not None else np.zeros(pre_vv.shape, np.float32)
    pre = normalise(_db(pre_vv), _db(_read(s["pre_vh"])), slope)
    post = normalise(_db(_read(s["post_vv"])), _db(_read(s["post_vh"])), slope)
    raw = _read(s["label"]).astype(np.int64)
    valid = _read(s["valid"]) == 1
    lab = np.full(raw.shape, IGNORE, np.int64)
    for k, v in LABEL_MAP.items():
        lab[raw == k] = v
    lab[~valid] = IGNORE
    return pre.astype(np.float32), post.astype(np.float32), lab


class KuroSiwo:
    """torch Dataset (duck-typed so this module imports without torch)."""

    def __init__(self, samples, augment=False, px_m=10.0):
        self.samples, self.augment, self.px_m = samples, augment, px_m

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        import torch

        pre, post, lab = load_sample(self.samples[i], self.px_m)
        if self.augment:
            k = random.randint(0, 3)
            pre, post, lab = (np.rot90(a, k, axes=(-2, -1)) for a in (pre, post, lab))
            if random.random() < 0.5:
                pre, post, lab = (a[..., ::-1] for a in (pre, post, lab))
            # small radiometric jitter: calibration differs slightly between processors
            off = np.random.normal(0, 0.1, (2, 1, 1)).astype(np.float32)
            pre = pre.copy(); post = post.copy()
            pre[:2] += off; post[:2] += off
        return (torch.from_numpy(np.ascontiguousarray(pre)), torch.from_numpy(np.ascontiguousarray(post)),
                torch.from_numpy(np.ascontiguousarray(lab)))


def inspect(root: str, n: int = 200, event_level: int | None = None) -> None:
    samples = find_samples(root, event_level)
    if not samples:
        print("No tiles found (looked for folders containing MS1_IVV* files).")
        return
    print(f"Found {len(samples)} samples in {len(set(s['event'] for s in samples))} events")
    ev = Counter(s["event"] for s in samples)
    items = sorted(ev.items())
    print("Samples per event:", dict(items[:40]), "..." if len(items) > 40 else "")
    print("\nExample tile folders (relative to root):")
    for s in random.Random(0).sample(samples, min(3, len(samples))):
        print("  ", Path(s["folder"]).relative_to(root), "-> event", s["event"])
    one = Path(samples[0]["folder"])
    print("Files in one tile folder:", sorted(p.name for p in one.iterdir())[:12])
    vals, lin = Counter(), []
    for s in random.sample(samples, min(n, len(samples))):
        u, c = np.unique(_read(s["label"]), return_counts=True)
        vals.update(dict(zip(u.tolist(), c.tolist())))
        lin.append(np.nanmedian(_read(s["post_vv"])))
    print("Label values (pixel counts):", dict(vals))
    extra = sorted(set(int(v) for v in vals) - set(LABEL_MAP))
    if extra:
        print(f"Values {extra} are not documented classes and are IGNORED in training/evaluation.")
    print(f"Median post VV value: {np.median(lin):.4f} (linear ~0.01-0.2 expected; negative means already dB)")
    print("Check LABEL_MAP in floodsight/model/dataset.py matches the dataset documentation.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", required=True, help="Kuro Siwo root folder")
    ap.add_argument("--event-level", type=int, help="folder depth below root that identifies the flood event")
    a = ap.parse_args()
    inspect(a.inspect, event_level=a.event_level)
