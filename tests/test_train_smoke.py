"""train.py runs end to end (1 epoch, CPU, tiny tiles) for a library model with the terrain loss, and the
checkpoint records architecture, budget and split."""
import pytest
import torch

from tests.test_compare_models import _tiles


@pytest.mark.parametrize("arch", ["siamese", "siamese_segformer"])
def test_train_one_epoch(tmp_path, arch):
    if arch != "siamese":
        pytest.importorskip("segmentation_models_pytorch")
    from floodsight.model.train import main
    root = tmp_path / "ks"
    _tiles(root, size=64)
    out = tmp_path / f"{arch}.pt"
    main(["--root", str(root), "--out", str(out), "--epochs", "1", "--batch", "2", "--workers", "0",
          "--arch", arch, "--terrain-loss", "0.5", "--val-events", "1111013", "--test-events", "427",
          "--exclude-events", "205"])
    ck = torch.load(out, map_location="cpu", weights_only=False)
    assert ck["arch"] == arch and ck["terrain_loss"] == 0.5
    assert ck["budget"]["epochs"] == 1 and ck["split"]["test"] == ["427"]
    assert "205" not in {str(e) for v in ck["split"].values() for e in v}      # excluded event never seen
