"""What the AI did on a run, measured without a reference map: agreement with the radar rules, what each found
alone, and how the model changed the final map. Also: a run with a model writes the AI layer for the map."""
import numpy as np

from floodsight.detect.baseline import FLOOD, NO_CHANGE, NODATA
from floodsight.detect.fuse import apply_model, model_vs_rules


def test_counts_match_the_masks():
    cls0 = np.full((10, 10), NO_CHANGE, np.uint8)
    cls0[0:4, :] = FLOOD                        # rules: 40 px of water
    cls0[9, :] = NODATA
    conf0 = np.where(cls0 == FLOOD, 2, 0).astype(np.uint8)
    p = np.zeros((10, 10), np.float32)
    p[2:6, :] = 0.95                            # model: rows 2-5 (40 px), 20 px shared with the rules
    zone = np.ones((10, 10), bool)
    cls1, conf1 = apply_model(cls0, conf0, p, zone, 0.8)
    v = model_vs_rules(cls0, conf0, cls1, conf1, p, 0.8, 20.0)
    px = 20 * 20 / 1e6
    assert v["both_km2"] == round(20 * px, 3) and v["model_only_km2"] == round(20 * px, 3)
    assert v["rules_only_km2"] == round(20 * px, 3)
    assert v["added_to_map_km2"] == round(20 * px, 3)          # rows 4-5 added at low confidence
    assert v["confidence_raised_km2"] == round(20 * px, 3)     # rows 2-3 agreed
    assert v["confidence_lowered_km2"] == round(20 * px, 3)    # rows 0-1: model clearly says no
    assert abs(v["agreement_iou"] - 20 / 60) < 1e-3


def test_run_with_model_writes_ai_layer(tmp_path, cfg):
    import copy

    import torch

    from floodsight.model.net import build_model
    from floodsight.pipeline import run
    torch.manual_seed(0)
    w = tmp_path / "m.pt"
    torch.save({"arch": "siamese", "encoder": "resnet34", "state_dict": build_model("siamese").state_dict()}, w)
    c = copy.deepcopy(cfg)
    c["model"]["weights"] = str(w)
    c["model"]["enabled"] = True
    r = run(None, "2026-08-26", c, demo=True, runs_dir=tmp_path / "runs")
    m = r["inputs"]["model"]
    assert m["used"] and "vs_rules" in m and m["threshold"] == c["model"]["flood_prob"]
    assert r["web"]["model"]["png"] == "layers/model_web.png"
    assert (tmp_path / "runs" / r["run_id"] / "layers" / "model_web.png").exists()
