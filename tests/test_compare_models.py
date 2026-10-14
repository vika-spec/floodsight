"""scripts/compare_models.py: same tiles and split for every method, thresholds from validation only,
refuses checkpoints trained on another split. Smoke test on tiny synthetic Kuro Siwo-like tiles."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from tests.test_kurosiwo_validate import _tif

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _tiles(root, size=48):
    rng = np.random.default_rng(0)
    for ev in ("411", "1111013", "427", "205"):
        for i in range(3):
            t = root / ev / "01" / f"t{i}"
            t.mkdir(parents=True)
            lab = np.zeros((size, size), np.uint8)
            lab[:, :20] = 2
            pre = rng.gamma(4, 0.1 / 4, (size, size)).astype(np.float32)
            post = pre.copy()
            post[:, :20] *= 0.05                                   # flood: much darker after
            for name, arr in (("MS1_IVV", post), ("MS1_IVH", post * 0.3), ("SL1_IVV", pre), ("SL1_IVH", pre * 0.3),
                              ("MK0_DEM", np.zeros((size, size), np.float32)), ("MK0_MLU", lab),
                              ("MK0_MNA", np.ones((size, size), np.uint8))):
                _tif(t / f"{name}_x.tif", arr, 300000 + 1000 * i, 3100000, 10)


def _ckpt(path, arch, split):
    from floodsight.model.net import build_model
    torch.save({"state_dict": build_model(arch).state_dict(), "arch": arch, "px_m": 10.0, "epoch": 1,
                "split": split}, path)


def test_compare_runs_all_methods(tmp_path, capsys):
    pytest.importorskip("sklearn")                 # optional package, only for the model comparison
    import compare_models
    root = tmp_path / "ks"
    _tiles(root)
    split = {"train": ["411"], "val": ["1111013"], "test": ["427"]}
    w1, w2 = tmp_path / "s.pt", tmp_path / "p.pt"
    _ckpt(w1, "siamese", split)
    _ckpt(w2, "post_only", split)
    out = tmp_path / "cmp.json"
    assert compare_models.main(["--root", str(root), "--weights", str(w1), str(w2), "--gbm", str(tmp_path / "g.joblib"),
                                "--train-gbm", "--extra-events", "205", "--out", str(out), "--gbm-train-tiles", "3"]) == 0
    r = json.loads(out.read_text())
    names = list(r["results"])
    assert any(n.startswith("siamese") for n in names) and any(n.startswith("post_only") for n in names)
    assert "gradient-boosted trees (pixel features)" in names and "physical rules (no learning)" in names
    gbm = r["results"]["gradient-boosted trees (pixel features)"]
    assert gbm["threshold_from_val"] in r["grid"] and gbm["test"]["427"]["iou"] > 0.9   # easy synthetic flood
    assert "205" in gbm["extra"]
    assert r["tiles_per_event"] == {"1111013": 3, "427": 3, "205": 3}


def test_refuses_different_split(tmp_path):
    import compare_models
    root = tmp_path / "ks"
    _tiles(root)
    _ckpt(tmp_path / "a.pt", "siamese", {"train": ["411"], "val": ["1111013"], "test": ["427"]})
    _ckpt(tmp_path / "b.pt", "early_fusion", {"train": ["411", "427"], "val": ["1111013"], "test": ["205"]})
    with pytest.raises(SystemExit, match="different event split"):
        compare_models.main(["--root", str(root), "--weights", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"),
                             "--no-rules"])
    with pytest.raises(SystemExit, match="already a 'test' event"):
        compare_models.main(["--root", str(root), "--weights", str(tmp_path / "a.pt"), "--extra-events", "427"])
