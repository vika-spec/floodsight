"""Rule-based change detection: new flood water and debris/mud from a same-track Sentinel-1 pair.

Every threshold that depends on the image (the water level in dB) is estimated from the scene itself,
so the same code works for any valley. Terrain rules (HAND, slope, radar blind spots) remove the
classic mountain false alarms.

Output classes:  0 = no change, 1 = new flood water, 2 = debris / mud, 3 = permanent water, 255 = no data
Confidence:      0 none, 1 low, 2 medium, 3 high  (for classes 1 and 2)
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger("floodsight")

NO_CHANGE, FLOOD, DEBRIS, PERMANENT, NODATA = 0, 1, 2, 3, 255
CLASS_NAMES = {NO_CHANGE: "no change", FLOOD: "flood water", DEBRIS: "debris / mud",
               PERMANENT: "permanent water", NODATA: "no data (radar blind spot / no image)"}


def lee_filter_db(img_db: np.ndarray, size: int = 5, enl: float = 4.4) -> np.ndarray:
    """Lee speckle filter applied in linear power, returned in dB. NaNs are preserved."""
    valid = np.isfinite(img_db)
    lin = np.where(valid, 10 ** (img_db / 10), 0.0)
    w = ndimage.uniform_filter(valid.astype(np.float64), size)
    m = ndimage.uniform_filter(lin, size) / np.maximum(w, 1e-6)
    m2 = ndimage.uniform_filter(lin * lin, size) / np.maximum(w, 1e-6)
    var = np.maximum(m2 - m * m, 1e-12)
    cu2 = 1.0 / enl
    k = np.clip(1 - (cu2 * m * m) / var, 0, 1)
    out = m + k * (lin - m)
    return np.where(valid, 10 * np.log10(np.clip(out, 1e-6, None)), np.nan).astype(np.float32)


def local_std(img_db: np.ndarray, size: int = 5) -> np.ndarray:
    x = np.nan_to_num(img_db, nan=np.nanmean(img_db))
    m = ndimage.uniform_filter(x, size)
    m2 = ndimage.uniform_filter(x * x, size)
    return np.sqrt(np.maximum(m2 - m * m, 0)).astype(np.float32)


def water_threshold(vv_db: np.ndarray, candidate: np.ndarray, lo: float, hi: float,
                    tile: int = 64) -> tuple[float, str]:
    """Scene-adaptive water threshold.

    Water is usually a tiny fraction of a mountain scene, so a global histogram is not bimodal. Following
    the split-based approach (Martinis et al.; Chini et al. 2017), we pick tiles whose histogram IS clearly
    bimodal (Ashman's D > 2, both classes 10-90%), pool them and run Otsu. Result is clamped to
    physically plausible values; if no tile qualifies we fall back to a clamped global estimate.
    """
    from skimage.filters import threshold_otsu

    ok = candidate & np.isfinite(vv_db)
    H, W = vv_db.shape
    pooled = []
    for r in range(0, H - tile + 1, tile // 2):
        for c in range(0, W - tile + 1, tile // 2):
            m = ok[r:r + tile, c:c + tile]
            if m.sum() < tile * tile * 0.3:
                continue
            v = np.clip(vv_db[r:r + tile, c:c + tile][m], -35, 5)
            try:
                t = threshold_otsu(v)
            except ValueError:
                continue
            a, b = v[v < t], v[v >= t]
            if a.size < 20 or b.size < 20:
                continue
            p = a.size / v.size
            dA = np.sqrt(2) * abs(b.mean() - a.mean()) / np.sqrt(a.var() + b.var() + 1e-6)
            if dA > 2.0 and 0.1 < p < 0.9 and a.mean() < hi:
                pooled.append(v)
    if len(pooled) >= 3:
        t = float(threshold_otsu(np.concatenate(pooled)))
        return float(np.clip(t, lo, hi)), f"bimodal tiles ({len(pooled)}) + Otsu"
    vals = vv_db[ok]
    if vals.size < 500:
        return (lo + hi) / 2, "default (too few valley-floor pixels)"
    t = float(threshold_otsu(np.clip(vals, -35, 5)))
    return float(np.clip(t, lo, hi)), "global Otsu (clamped)"


def min_object_px(d: dict, res: float) -> int:
    """Minimum object size in pixels for this resolution (config gives it in m²; old configs in px)."""
    if "min_object_m2" in d:
        return max(1, int(round(d["min_object_m2"] / (res * res))))
    return int(d.get("min_object_px", 12))


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, index=np.arange(1, n + 1))
    keep = np.zeros(n + 1, bool)
    keep[1:] = sizes >= min_px
    return keep[lab]


def detect_change(pre_db: np.ndarray, post_db: np.ndarray, blind: np.ndarray, hand: np.ndarray,
                  slope: np.ndarray, streams: np.ndarray, res: float, cfg: dict,
                  mountain: np.ndarray | None = None) -> dict:
    """mountain: optional boolean mask of mountain terrain (preprocess.mountain). Debris flows and mud
    deposits are a mountain process; where the mask is False only flood water is mapped."""
    d = cfg["detection"]
    size = cfg["sentinel1"].get("speckle_filter_size", 5)
    pre = np.stack([lee_filter_db(b, size) for b in pre_db])
    post = np.stack([lee_filter_db(b, size) for b in post_db])
    vv0, vh0 = pre[0], pre[1]
    vv1, vh1 = post[0], post[1]
    has_vh = np.isfinite(vh0).any() and np.isfinite(vh1).any()

    valid = np.isfinite(vv0) & np.isfinite(vv1) & ~blind
    hand_ = np.nan_to_num(hand, nan=9999.0)

    # --- water -----------------------------------------------------------------------------------
    water_zone = valid & (hand_ <= d["water_max_hand_m"]) & (slope <= d["water_max_slope_deg"])
    t_water, t_how = water_threshold(vv1, water_zone, d["water_db_min"], d["water_db_max"])
    water_post = water_zone & (vv1 < t_water)
    water_pre = water_zone & (vv0 < t_water)
    dvv = vv1 - vv0
    new_water = water_post & ~water_pre & (dvv < -d["water_drop_db"])
    new_water = remove_small(new_water, min_object_px(d, res))
    permanent = water_post & water_pre

    # --- debris / mud ----------------------------------------------------------------------------
    dvh = (vh1 - vh0) if has_vh else np.zeros_like(dvv)
    tex = local_std(vv1) / np.maximum(local_std(vv0), 0.3)
    debris_zone = valid & (hand_ <= d["debris_max_hand_m"]) & (slope <= d["debris_max_slope_deg"]) & ~water_post
    if mountain is not None:
        debris_zone &= mountain
    ind_vv = np.abs(dvv) > d["debris_change_db"]
    ind_vh = np.abs(dvh) > d["debris_change_db"] if has_vh else np.zeros_like(ind_vv)
    ind_tex = (tex > d["debris_texture_ratio"]) & (np.abs(dvv) > d["debris_change_db"] / 2)
    votes = ind_vv.astype(np.uint8) + ind_vh.astype(np.uint8) + ind_tex.astype(np.uint8)
    debris = debris_zone & (votes >= (2 if has_vh else 1))
    debris = remove_small(debris, min_object_px(d, res))

    # debris must connect to the river corridor (streams or water); isolated hillside change = landslide
    # or farming, not the flood
    corridor_seed = streams | water_post
    if corridor_seed.any():
        dist = ndimage.distance_transform_edt(~corridor_seed) * res
        lab, n = ndimage.label(debris, structure=np.ones((3, 3)))
        if n:
            near = ndimage.minimum(dist, lab, index=np.arange(1, n + 1))
            ok = np.zeros(n + 1, bool)
            ok[1:] = near <= d["corridor_link_m"]
            debris = ok[lab]

    # --- assemble ----------------------------------------------------------------------------------
    cls = np.full(vv1.shape, NO_CHANGE, np.uint8)
    cls[permanent] = PERMANENT
    cls[debris] = DEBRIS
    cls[new_water] = FLOOD
    cls[~(np.isfinite(vv0) & np.isfinite(vv1)) | blind] = NODATA

    conf = np.zeros(vv1.shape, np.uint8)
    margin = t_water - vv1
    strong_w = (margin > 2.0) & (dvv < -2 * d["water_drop_db"])
    vh_agree = (dvh < -d["water_drop_db"]) if has_vh else np.zeros_like(strong_w)
    conf[new_water] = 1
    conf[new_water & (strong_w | vh_agree)] = 2
    conf[new_water & strong_w & vh_agree] = 3
    # One radar image alone is low confidence for debris: its VV, VH and texture signs all come from the same
    # acquisition and can all be caused by rain-soaked soil and vegetation. Independent confirmation (the other
    # orbit track, Sentinel-2) raises it later (fuse.merge_tracks, fuse.apply_optical).
    conf[debris] = 1

    stats = {"water_threshold_db": round(t_water, 2), "water_threshold_method": t_how,
             "valid_fraction": round(float(valid.mean()), 4), "blind_fraction": round(float(blind.mean()), 4),
             "has_vh": bool(has_vh)}
    return {"cls": cls, "conf": conf, "stats": stats, "dvv": dvv.astype(np.float32),
            "vv_pre": vv0, "vv_post": vv1}
