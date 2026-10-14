"""Kuro Siwo label check: labels land on the run grid in the right place and the scores are right."""
import json

import numpy as np
import rasterio
from rasterio.transform import from_origin

from floodsight.detect.baseline import FLOOD, NODATA
from floodsight.validate.kurosiwo import labels_on_grid, score

CRS = "EPSG:32645"


def _tif(path, arr, x0, y0, res):
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype=arr.dtype, crs=CRS, transform=from_origin(x0, y0, res, res)) as dst:
        dst.write(arr, 1)


def _make(tmp_path):
    root = tmp_path / "ks"
    # two 10 m tiles of 40x40 px side by side; left tile flooded in its left half
    # a second event far away (must be ignored), so events are told apart by folder like the real dataset
    for i, (ev, x0) in enumerate((("427", 300000), ("427", 300400), ("999", 500000))):
        t = root / ev / "aoiA" / f"tile{i}"
        t.mkdir(parents=True)
        lab = np.zeros((40, 40), np.uint8)
        if i == 0:
            lab[:, :20] = 2
        valid = np.ones((40, 40), np.uint8)
        valid[:5, :] = 0                                    # top rows invalid -> ignored
        for name, arr in (("MS1_IVV", np.ones((40, 40), np.float32)), ("MS1_IVH", np.ones((40, 40), np.float32)),
                          ("SL1_IVV", np.ones((40, 40), np.float32)), ("SL1_IVH", np.ones((40, 40), np.float32)),
                          ("MK0_DEM", np.zeros((40, 40), np.float32)), ("MK0_MLU", lab), ("MK0_MNA", valid)):
            _tif(t / f"{name}_x.tif", arr, x0, 3100000, 10)
    run = tmp_path / "run"
    (run / "layers").mkdir(parents=True)
    cls = np.zeros((20, 40), np.uint8)                     # 20 m grid covering both tiles
    cls[:, :10] = FLOOD                                     # exactly the labelled flood
    cls[:, 10:12] = FLOOD                                   # 2 extra columns
    cls[:, 35:] = NODATA                                    # blind strip on the right
    _tif(run / "layers" / "flood_class.tif", cls, 300000, 3100000, 20)
    (run / "results.json").write_text(json.dumps({"inputs": {"sentinel1_pairs": [{"post_date": "2020-01-02"}]}}))
    return root, run


def test_labels_on_grid_positions(tmp_path):
    root, run = _make(tmp_path)
    from floodsight.model.dataset import find_samples
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        truth, used = labels_on_grid(find_samples(root), (20, 40), src.transform, src.crs)
    assert used == 2
    assert (truth[3:, :10] == 2).all() and (truth[3:, 10:] == 0).all()
    assert (truth[:2] == 255).all()                         # invalid rows stay unknown


def test_scores(tmp_path):
    root, run = _make(tmp_path)
    o = score(run, root, "427", weights="missing.pt", ks_post_date="2020-01-02")
    s = o["scores"]["final map: flood water"]
    assert s["recall"] == 1.0
    assert abs(s["precision"] - 10 / 12) < 0.01            # 2 extra columns of 12 predicted
    assert o["label_flood_in_blind_mask_pct"] == 0.0
    assert (run / "validation" / "kurosiwo_427.json").exists()


def test_label_flood_breakdown_flags_already_flooded_before_image(tmp_path):
    """Labelled flood that our map calls permanent water = water already in our before-image."""
    from floodsight.detect.baseline import PERMANENT
    root, run = _make(tmp_path)
    with rasterio.open(run / "layers" / "flood_class.tif", "r+") as dst:
        cls = dst.read(1)
        cls[:, :10] = PERMANENT                             # the labelled flood was water before
        dst.write(cls, 1)
    o = score(run, root, "427", weights="missing.pt", ks_post_date="2020-01-02")
    c = o["label_flood_in_our_classes_pct"]
    assert c["permanent water (water already in our before-image)"] == 100.0
    assert c["flood or debris"] == 0.0
    assert o["diagnostic_tiles"] == 2                       # both tiles overlap the run area
    assert any("weights not found" in w for w in o["diagnostics_skipped"])


def test_backscatter_uses_the_after_image_of_the_kurosiwo_date(tmp_path):
    root, run = _make(tmp_path)
    grid = {"crs": CRS, "transform": [20, 0, 300000, 0, -20, 3100000], "width": 40, "height": 20}
    pairs = [{"track": "asc-001", "pre_date": "2019-12-20", "post_date": "2020-01-05"},
             {"track": "des-002", "pre_date": "2019-12-21", "post_date": "2020-01-02"}]
    (run / "results.json").write_text(json.dumps({"grid": grid, "inputs": {"sentinel1_pairs": pairs}}))
    (run / "inputs").mkdir()
    _tif(run / "inputs" / "s1_des-002_post_2020-01-02_s0.tif", np.ones((2, 20, 40), np.float32)[0], 300000, 3100000, 20)
    o = score(run, root, "427", weights="missing.pt", ks_post_date="2020-01-02")
    assert o["run_before_dates"] == ["2019-12-20", "2019-12-21"]
    # found the des-002 file (the Kuro Siwo date); this tiny area is below the pixel minimum, so it says so
    assert any("pixels where both after-images" in w for w in o["diagnostics_skipped"])
