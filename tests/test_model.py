import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")


def test_forward_shape_and_inference(scene):
    from floodsight.model.net import SiameseUNet
    m = SiameseUNet().eval()
    x = torch.zeros(2, 3, 224, 224)
    assert m(x, x).shape == (2, 3, 224, 224)


def test_metrics():
    from floodsight.model.metrics import confusion, scores
    cm = confusion(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), 3)
    s = scores(cm)
    assert s["per_class"][0]["iou"] == 1.0 and s["per_class"][2]["precision"] == 0.5


def test_speckle_matching_reaches_training_level_and_leaves_smooth_images_alone():
    """Model input is filtered to the training data's graininess; already-smooth input is untouched."""
    import numpy as np
    from scipy.ndimage import gaussian_filter, zoom

    from floodsight.model.infer import match_speckle, speckle_level

    rng = np.random.default_rng(0)
    base = -9 + gaussian_filter(rng.normal(size=(300, 300)), 6) * 20
    noisy = lambda: zoom(10 * np.log10(10 ** (base / 10) * rng.gamma(4.4, 1 / 4.4, base.shape)), 2, order=1)
    pre, post = np.stack([noisy(), noisy() - 7]), np.stack([noisy(), noisy() - 7])
    slope = np.zeros(pre.shape[1:])
    _, post_m, info = match_speckle(pre, post, slope, 0.4)
    assert info["window"] > 0 and speckle_level(post_m[0], slope) <= 0.44
    assert abs(np.nanmedian(post_m[0]) - np.nanmedian(post[0])) < 1.0      # brightness level kept
    smooth = np.stack([zoom(base, 2, order=1)] * 2)
    _, same, info2 = match_speckle(smooth, smooth, slope, 0.4)
    assert info2["window"] == 0 and np.array_equal(same, smooth)


@pytest.mark.parametrize("arch", ["siamese", "early_fusion", "post_only"])
def test_all_architectures_same_interface(arch, tmp_path):
    """Comparison models share the input/output contract, and a saved checkpoint reloads as its own arch."""
    from floodsight.model.infer import load_model
    from floodsight.model.net import build_model
    m = build_model(arch).eval()
    pre, post = torch.zeros(1, 3, 128, 128), torch.ones(1, 3, 128, 128)
    assert m(pre, post).shape == (1, 3, 128, 128)
    if arch == "post_only":                          # truly ignores the before image
        assert torch.allclose(m(pre, post), m(post, post))
    w = tmp_path / f"{arch}.pt"
    torch.save({"state_dict": m.state_dict(), "arch": arch, "px_m": 10.0}, w)
    m2, info = load_model(w)
    assert info["arch"] == arch and type(m2) is type(m)


@pytest.mark.parametrize("arch", ["siamese_unetpp", "siamese_attn_unet", "siamese_deeplabv3p", "siamese_segformer",
                                  "segformer_early"])
def test_library_architectures_as_change_models(arch):
    pytest.importorskip("segmentation_models_pytorch")
    from floodsight.model.net import build_model
    m = build_model(arch).eval()
    pre, post = torch.zeros(1, 3, 224, 224), torch.ones(1, 3, 224, 224)
    with torch.no_grad():
        y = m(pre, post)
        assert y.shape == (1, 3, 224, 224)
        if arch.startswith("siamese"):                  # the before image matters
            assert not torch.allclose(y, m(post, post))


def test_terrain_loss_penalises_flood_on_steep_ground():
    from floodsight.model.train import terrain_loss
    post = torch.zeros(1, 3, 4, 4)
    post[0, 2, :, :2] = 30 / 30.0                       # left half 30 degrees, right half flat
    target = torch.zeros(1, 4, 4, dtype=torch.long)
    flood_left = torch.zeros(1, 3, 4, 4); flood_left[0, 2, :, :2] = 10.0
    flood_right = torch.zeros(1, 3, 4, 4); flood_right[0, 2, :, 2:] = 10.0
    assert terrain_loss(flood_left, post, target, 12.0) > 0.9
    assert terrain_loss(flood_right, post, target, 12.0) < 0.4
    flat = torch.zeros(1, 3, 4, 4)
    assert terrain_loss(flood_left, flat, target, 12.0) == 0
