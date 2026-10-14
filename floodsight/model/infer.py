"""Sliding-window inference of the Siamese U-Net on any AOI.

The model was trained at Kuro Siwo's pixel size; inputs on the analysis grid are resampled to that size,
predicted in overlapping tiles blended with a smooth window (no seams), and resampled back.
Returns probabilities for permanent water and flood; the pipeline fuses them with the rule-based map.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("floodsight")
_MODEL = None


def load_model(weights: str | Path):
    global _MODEL
    import torch

    from .net import build_model

    if _MODEL is not None and _MODEL[0] == str(weights):
        return _MODEL[1], _MODEL[2]
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    m = build_model(ck.get("arch", "siamese"), ck.get("encoder", "resnet34"), pretrained=False)
    m.load_state_dict(ck["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m.to(device).eval()
    _MODEL = (str(weights), m, {"px_m": ck.get("px_m", 10.0), "device": device, "epoch": ck.get("epoch"),
                                "val": ck.get("val"), "arch": ck.get("arch", "siamese")})
    return m, _MODEL[2]


def speckle_level(vv_db: np.ndarray, slope: np.ndarray) -> float:
    """Median local standard deviation (5x5) of VV in dB on flat ground: how grainy the image is."""
    from ..detect.baseline import local_std

    valid = np.isfinite(vv_db)
    flat = valid & (np.nan_to_num(slope, nan=90) < 5)
    sel = flat if flat.sum() >= 2000 else valid
    return float(np.median(local_std(vv_db, 5)[sel])) if sel.any() else float("nan")


def match_speckle(pre_db: np.ndarray, post_db: np.ndarray, slope: np.ndarray, target: float) -> tuple:
    """Bring the image graininess down to the level of the training data (Kuro Siwo), so the model sees the
    kind of image it learned from. The target comes from the TRAINING tiles (scripts/check_domain.py), never
    from a test or case-study scene; the smallest Lee window that reaches it is chosen per image, the same
    window for both dates. Images already as smooth as the target are left untouched."""
    from ..detect.baseline import lee_filter_db

    before = speckle_level(post_db[0], slope)
    info = {"target_db": target, "before_db": round(before, 3), "window": 0, "after_db": round(before, 3)}
    if not np.isfinite(before) or before <= target * 1.1:
        return pre_db, post_db, info
    chosen, level = 11, before
    for k in (3, 5, 7, 9, 11):
        level = speckle_level(lee_filter_db(post_db[0], k), slope)
        if level <= target * 1.1:
            chosen = k
            break
    f = lambda img: np.stack([lee_filter_db(b, chosen) for b in img])
    info.update(window=chosen, after_db=round(level, 3))
    return f(pre_db), f(post_db), info


def predict(pre_db: np.ndarray, post_db: np.ndarray, slope: np.ndarray, grid_res: float, weights,
            tile: int = 256, overlap: int = 64, tta: bool = False, speckle_target: float | None = None) -> dict:
    import torch
    import torch.nn.functional as F

    from .net import normalise

    model, info = load_model(weights)
    dev = info["device"]
    H, W = slope.shape
    scale = grid_res / info["px_m"]

    def to_model_res(a):            # (C, H, W) numpy -> model pixel size, bilinear
        t = torch.from_numpy(np.ascontiguousarray(a, np.float32))[None]
        if abs(scale - 1) > 0.05:
            size = (max(int(round(H * scale)), tile), max(int(round(W * scale)), tile))
            t = F.interpolate(t, size=size, mode="bilinear", align_corners=False)
        return t[0].numpy()

    # NaN-safe resampling: interpolate filled values, then put the gaps back
    def resample_db(img):
        ok = to_model_res(np.isfinite(img).astype(np.float32)) > 0.5
        fill = np.where(np.isfinite(img), img, np.nanmedian(img, axis=(1, 2), keepdims=True))
        out = to_model_res(fill)
        out[~ok] = np.nan
        return out

    pre_m, post_m = resample_db(pre_db[:2]), resample_db(post_db[:2])
    slope_m = to_model_res(np.nan_to_num(slope, nan=0.0)[None])[0]
    speckle = None
    if speckle_target:
        pre_m, post_m, speckle = match_speckle(pre_m, post_m, slope_m, speckle_target)
    x_pre = torch.from_numpy(normalise(pre_m[0], pre_m[1], slope_m).astype(np.float32))[None]
    x_post = torch.from_numpy(normalise(post_m[0], post_m[1], slope_m).astype(np.float32))[None]
    h, w = x_pre.shape[-2:]
    ph, pw = max(tile - h, 0), max(tile - w, 0)
    if ph or pw:
        x_pre = F.pad(x_pre, (0, pw, 0, ph), mode="reflect")
        x_post = F.pad(x_post, (0, pw, 0, ph), mode="reflect")
    Hp, Wp = x_pre.shape[-2:]

    win1 = np.hanning(tile + 2)[1:-1]
    win = torch.from_numpy(np.outer(win1, win1).astype(np.float32)).clamp_min(1e-3)
    acc = torch.zeros(3, Hp, Wp)
    wsum = torch.zeros(Hp, Wp)
    stride = tile - overlap
    rows = list(range(0, max(Hp - tile, 0) + 1, stride))
    cols = list(range(0, max(Wp - tile, 0) + 1, stride))
    if rows[-1] != Hp - tile:
        rows.append(Hp - tile)
    if cols[-1] != Wp - tile:
        cols.append(Wp - tile)
    coords = [(r, c) for r in rows for c in cols]
    bs = 8
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=dev.type == "cuda"):
        for i in range(0, len(coords), bs):
            chunk = coords[i:i + bs]
            a = torch.cat([x_pre[..., r:r + tile, c:c + tile] for r, c in chunk]).to(dev)
            b = torch.cat([x_post[..., r:r + tile, c:c + tile] for r, c in chunk]).to(dev)
            from .evaluate import probabilities
            p = probabilities(model, a, b, tta).cpu()
            for k, (r, c) in enumerate(chunk):
                acc[:, r:r + tile, c:c + tile] += p[k] * win
                wsum[r:r + tile, c:c + tile] += win
    prob = (acc / wsum.clamp_min(1e-6))[:, :h, :w][None]
    if (h, w) != (H, W):
        prob = F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)
    prob = prob[0].numpy()
    invalid = ~(np.isfinite(pre_db[0]) & np.isfinite(post_db[0]))
    prob[:, invalid] = np.nan
    return {"p_permanent": prob[1].astype(np.float32), "p_flood": prob[2].astype(np.float32),
            "model_epoch": info.get("epoch"), "speckle_matching": speckle}
