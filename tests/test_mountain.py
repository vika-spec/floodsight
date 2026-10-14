"""Debris only in mountain terrain (UNEP-WCMC classes): plains -> water only, mountains unchanged."""
import numpy as np

from floodsight.preprocess.mountain import kapos_classes


def test_kapos_classes():
    px = 250.0
    n = 120                                               # 30 km square
    plain = np.full((n, n), 180.0) + np.random.default_rng(0).normal(0, 2, (n, n))
    assert kapos_classes(plain, px).max() == 0            # Mura-like plain: never mountain
    x = np.arange(n) * px
    valley = 600 + 0.25 * np.abs(x - x.mean())[None, :].repeat(n, 0)   # 600 m floor, walls to ~2100 m
    cls = kapos_classes(valley, px)
    assert (cls[:, n // 2] > 0).all()                     # valley floor counts (relief > 300 m within 7 km)
    high = np.full((n, n), 2700.0)
    assert (kapos_classes(high, px) == 3).all()           # high plateau: class 3 by elevation alone
    flat_mid = np.full((n, n), 1200.0)
    assert kapos_classes(flat_mid, px).max() == 0         # 1200 m, flat, no relief: not mountain


def test_debris_gated_off_on_plains(cfg, scene):
    from floodsight.detect.baseline import DEBRIS, detect_change
    from floodsight.preprocess.geometry import slope_deg
    from floodsight.preprocess.terrain import hydro_layers

    g = scene["grid"]
    slope = slope_deg(scene["dem"], g.res)
    hydro = hydro_layers(scene["dem"], g)
    blind = np.zeros(g.shape, bool)
    on = detect_change(scene["pre"], scene["post"], blind, hydro["hand"], slope, hydro["streams"], g.res, cfg,
                       mountain=np.ones(g.shape, bool))
    off = detect_change(scene["pre"], scene["post"], blind, hydro["hand"], slope, hydro["streams"], g.res, cfg,
                        mountain=np.zeros(g.shape, bool))
    assert (on["cls"] == DEBRIS).sum() > 100
    assert (off["cls"] == DEBRIS).sum() == 0
