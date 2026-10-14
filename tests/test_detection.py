import numpy as np

from floodsight.detect.baseline import DEBRIS, FLOOD, detect_change, water_threshold
from floodsight.model.metrics import binary_scores


def test_flood_and_debris_found(scene, detection):
    cls, truth = detection["cls"], scene["truth"]
    assert binary_scores(cls == FLOOD, truth == 1)["iou"] > 0.9
    assert binary_scores(cls == DEBRIS, truth == 2)["iou"] > 0.7


def test_no_flood_means_no_detection(cfg, terrain):
    from floodsight.synthetic import make_scene
    s = make_scene(flood=False, seed=3)
    h = terrain["hydro"]
    r = detect_change(s["pre"], s["post"], terrain["vis"]["blind"], h["hand"], terrain["slope"], h["streams"],
                      s["grid"].res, cfg)
    changed_km2 = ((r["cls"] == FLOOD) | (r["cls"] == DEBRIS)).sum() * 400 / 1e6
    assert changed_km2 < 0.05


def test_water_threshold_is_clamped():
    vv = np.full((256, 256), -8.0)
    t, how = water_threshold(vv, np.ones_like(vv, bool), -24, -15)
    assert -24 <= t <= -15
