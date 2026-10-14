"""Sen1Floods11 scorer runs on a tiny fake copy of the dataset layout and refuses two-date models."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _tif(path, arr):
    import rasterio
    from rasterio.transform import from_origin
    arr = np.atleast_3d(arr).transpose(2, 0, 1) if arr.ndim == 2 else arr
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0],
                       dtype="float32", crs="EPSG:4326", transform=from_origin(0, 1, 1e-4, 1e-4)) as dst:
        dst.write(arr.astype(np.float32))


def test_scores_post_only_and_refuses_siamese(tmp_path):
    from eval_sen1floods11 import main

    from floodsight.model.net import build_model
    (tmp_path / "S1Hand").mkdir()
    (tmp_path / "LabelHand").mkdir()
    s1 = np.stack([np.full((64, 64), -10.0), np.full((64, 64), -17.0)])
    s1[:, :20] = -25.0
    lab = np.zeros((64, 64))
    lab[:, :20] = 1
    _tif(tmp_path / "S1Hand" / "Bolivia_1_S1Hand.tif", s1)
    _tif(tmp_path / "LabelHand" / "Bolivia_1_LabelHand.tif", lab)
    (tmp_path / "flood_bolivia_data.csv").write_text("Bolivia_1_S1Hand.tif,Bolivia_1_LabelHand.tif\n")
    w = tmp_path / "p.pt"
    torch.save({"arch": "post_only", "encoder": "resnet18", "state_dict": build_model("post_only", "resnet18").state_dict()}, w)
    out = tmp_path / "r.json"
    assert main(["--root", str(tmp_path), "--weights", str(w), "--splits", "flood_bolivia_data.csv", "--out", str(out)]) == 0
    r = json.loads(out.read_text())["flood_bolivia_data.csv"]
    assert r["chips"] == 1 and 0.0 <= r["model"]["iou"] <= 1.0
    w2 = tmp_path / "s.pt"
    torch.save({"arch": "siamese", "encoder": "resnet18", "state_dict": build_model("siamese", "resnet18").state_dict()}, w2)
    assert main(["--root", str(tmp_path), "--weights", str(w2), "--out", str(out)]) == 2
