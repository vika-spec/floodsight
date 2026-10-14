"""Fuse all evidence into one flood/debris map with a per-pixel confidence level.

Evidence sources (each optional except the first):
  * one or more Sentinel-1 track results (ascending + descending fill each other's blind spots)
  * the segmentation model's flood probability
  * Sentinel-2 optical evidence: confirms/contradicts radar, fills radar blind spots and, where the sky was
    clear, adds what only the optical sensor sees (low confidence, same physical limits)
"""
from __future__ import annotations

import numpy as np

from .baseline import DEBRIS, FLOOD, NO_CHANGE, NODATA, PERMANENT


def merge_tracks(results: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Per pixel: keep the most confident valid observation across tracks."""
    cls = np.full(results[0]["cls"].shape, NODATA, np.uint8)
    conf = np.zeros_like(results[0]["conf"])
    best = np.full(cls.shape, -1, np.int16)
    for r in results:
        c, k = r["cls"], r["conf"]
        valid = c != NODATA
        # rank: detections by confidence (1..3 -> 11..13), then plain valid observations (10)
        rank = np.where(valid, 10 + np.where((c == FLOOD) | (c == DEBRIS), k, 0), -1).astype(np.int16)
        take = rank > best
        cls[take], conf[take], best[take] = c[take], k[take], rank[take]
    return cls, conf


def combine_track_probs(probs: list, track_cls: list) -> np.ndarray:
    """Model flood probability from several orbit tracks -> one map: per pixel the maximum over the tracks
    that see the pixel (a track's layover/shadow/no-image pixels, NODATA in its class map, are ignored).
    NaN where no track sees the pixel. Same principle as merge_tracks for the rule-based map."""
    out = None
    for p, c in zip(probs, track_cls):
        p = np.where(c == NODATA, np.nan, p).astype(np.float32)
        out = p if out is None else np.fmax(out, p)
    return out


def apply_model(cls, conf, prob_flood: np.ndarray, water_zone: np.ndarray, thr: float):
    cls, conf = cls.copy(), conf.copy()
    valid = cls != NODATA
    m = np.nan_to_num(prob_flood, nan=0.0)
    agree = (cls == FLOOD) & (m >= thr)
    conf[agree] = np.minimum(conf[agree] + 1, 3)
    disagree = (cls == FLOOD) & (m < 0.2)
    conf[disagree] = np.maximum(conf[disagree].astype(int) - 1, 1).astype(np.uint8)
    add = valid & (cls == NO_CHANGE) & water_zone & (m >= max(thr, 0.7))
    cls[add], conf[add] = FLOOD, 1
    return cls, conf


def apply_optical(cls, conf, opt: dict, water_zone: np.ndarray, debris_zone: np.ndarray | None = None,
                  flood_zone: np.ndarray | None = None, min_px: int = 0, detect: bool = False):
    """Sentinel-2 as a second sensor (only where the sky was clear before AND after).

    * agreement with a radar detection raises its confidence by one level;
    * clearly vegetated land after the event lowers a radar detection's confidence by one level;
    * radar blind spots: optical new water / fresh debris fills them (low confidence);
    * detect=True: optical new water / fresh debris is ALSO added where the radar saw no change, at low
      confidence, under the same physical limits as the radar rules: water only where water can pond
      (flood_zone: low height above the river, gentle slope), debris only where debris is possible
      (debris_zone: mountain terrain, near the river corridor). Two independent sensors that see different
      things (radar: roughness/water; optical: vegetation loss, mud, water colour) recover damage the
      other misses, e.g. flood traces that left no radar change.
    Returns (cls, conf, stats in pixels)."""
    from .baseline import remove_small

    cls, conf = cls.copy(), conf.copy()
    vb = opt["valid_both"]
    wz = water_zone if flood_zone is None else flood_zone
    dz = water_zone if debris_zone is None else water_zone & debris_zone
    stats = {}
    for target, ev in ((FLOOD, opt["new_water"]), (DEBRIS, opt["debris"])):
        agree = vb & (cls == target) & ev
        conf[agree] = np.minimum(conf[agree] + 1, 3)
        stats[f"confirmed_{'flood' if target == FLOOD else 'debris'}_px"] = int(agree.sum())
    # clearly vegetated land after the event contradicts a radar detection
    green = vb & (np.nan_to_num(opt["ndvi_post"]) > 0.55)
    bad = green & ((cls == FLOOD) | (cls == DEBRIS))
    conf[bad] = np.maximum(conf[bad].astype(int) - 1, 1).astype(np.uint8)
    stats["contradicted_px"] = int(bad.sum())
    # optical fills radar blind spots where it sees new water / debris in the valley
    fill_w = (cls == NODATA) & opt["new_water"] & wz
    fill_d = (cls == NODATA) & opt["debris"] & dz
    cls[fill_w], conf[fill_w] = FLOOD, 1
    cls[fill_d], conf[fill_d] = DEBRIS, 1
    blind_seen = (cls == NODATA) & opt["valid_both"]
    cls[blind_seen & ~(fill_w | fill_d)] = NO_CHANGE
    stats["blind_filled_px"] = int(fill_w.sum() + fill_d.sum())
    if detect:
        add_w = (cls == NO_CHANGE) & vb & opt["new_water"] & wz
        add_d = (cls == NO_CHANGE) & vb & opt["debris"] & dz & ~add_w
        if min_px:
            add_w, add_d = remove_small(add_w, min_px), remove_small(add_d, min_px)
        cls[add_w], conf[add_w] = FLOOD, 1
        cls[add_d], conf[add_d] = DEBRIS, 1
        stats["added_flood_px"], stats["added_debris_px"] = int(add_w.sum()), int(add_d.sum())
    return cls, conf, stats


def area_summary(cls, conf, res: float) -> dict:
    px = res * res / 1e6
    out = {}
    for name, c in (("flood", FLOOD), ("debris", DEBRIS), ("permanent_water", PERMANENT)):
        m = cls == c
        out[f"{name}_km2"] = round(float(m.sum() * px), 3)
        if c in (FLOOD, DEBRIS):
            out[f"{name}_high_conf_km2"] = round(float((m & (conf >= 2)).sum() * px), 3)
    out["nodata_km2"] = round(float((cls == NODATA).sum() * px), 3)
    out["aoi_km2"] = round(float(cls.size * px), 3)
    out["nodata_fraction"] = round(float((cls == NODATA).mean()), 4)
    return out