"""scripts/check_model_input.py runs end to end on a tiny run folder with untrained weights (smoke test)."""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tests.test_kurosiwo_validate import CRS, _make, _tif  # noqa: E402


def test_runs_and_scores(tmp_path, capsys):
    from floodsight.model.net import SiameseUNet
    root, run = _make(tmp_path)
    w = tmp_path / "w.pt"
    torch.save({"state_dict": SiameseUNet(3, 3, "resnet34", pretrained=False).state_dict(), "px_m": 10.0,
                "split": {"train": ["427"]}}, w)
    grid = {"crs": CRS, "transform": [20, 0, 300000, 0, -20, 3100000], "width": 40, "height": 20}
    pairs = [{"track": "des-002", "pre_date": "2019-12-21", "post_date": "2020-01-02"}]
    (run / "results.json").write_text(json.dumps({"grid": grid, "inputs": {"sentinel1_pairs": pairs}}))
    (run / "inputs").mkdir()
    rng = np.random.default_rng(0)
    for k in ("pre", "post"):
        arr = rng.gamma(4, 0.05 / 4, (2, 20, 40)).astype(np.float32)
        with rasterio.open(run / "inputs" / f"s1_des-002_{k}_{pairs[0][k + '_date']}_s0.tif", "w", driver="GTiff",
                           height=20, width=40, count=2, dtype="float32", crs=CRS,
                           transform=from_origin(300000, 3100000, 20, 20)) as dst:
            dst.write(arr)
    _tif(run / "inputs" / "dem.tif", np.zeros((20, 40), np.float32), 300000, 3100000, 20)
    import check_model_input
    assert check_model_input.main(["--run", str(run), "--root", str(root), "--event", "427",
                                   "--weights", str(w)]) == 0
    out = capsys.readouterr().out
    assert "Lee 5x5" in out and "model role: train" in out
    scores = json.loads((run / "validation" / "model_input_427.json").read_text())
    assert set(scores) >= {"none", "Lee 3x3", "Lee 7x7"}
