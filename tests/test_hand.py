"""HAND must exist everywhere on flat plains (paths that leave the area before a stream used to get none,
which silently removed them from the water zone) and must equal the classic definition elsewhere."""
import numpy as np

from floodsight.preprocess.terrain import D8_OFFSETS, height_above_drainage


def _fdir_towards(H, W, code):
    return np.full((H, W), code, np.int32)


def test_path_leaving_area_is_measured_to_exit_cell():
    H, W = 5, 6
    z = np.tile(np.arange(W, 0, -1, dtype=float), (H, 1))      # falls to the east, no stream inside
    east = [k for k, v in D8_OFFSETS.items() if v == (0, 1)][0]
    hand, edge = height_above_drainage(_fdir_towards(H, W, east), z, np.zeros((H, W), bool))
    assert np.isfinite(hand).all()
    assert np.allclose(hand[:, 0], z[:, 0] - z[:, -1])          # measured to the last cell inside
    assert edge.all()


def test_stream_is_reference_and_hollow_not_negative():
    H, W = 3, 5
    z = np.array([[5, 4, 1, 3, 2]] * H, float)                  # cell 2 is a hollow below the stream at col 4
    east = [k for k, v in D8_OFFSETS.items() if v == (0, 1)][0]
    streams = np.zeros((H, W), bool)
    streams[:, 4] = True
    hand, edge = height_above_drainage(_fdir_towards(H, W, east), z, streams)
    assert np.allclose(hand[:, 0], 3.0) and np.allclose(hand[:, 1], 2.0)
    assert (hand[:, 2] == 0).all()                              # 1 - 2 < 0 -> 0
    assert not edge.any()


def test_flat_plain_has_hand_everywhere(cfg):
    from floodsight.core import AOI, Grid
    from floodsight.preprocess.terrain import hydro_layers
    from scipy import ndimage as ndi
    g = Grid.for_aoi(AOI(68.12, 27.74, 68.15, 27.77), 20)
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:g.shape[0], 0:g.shape[1]]
    dem = (40 + 0.002 * (yy + xx) + ndi.gaussian_filter(rng.normal(0, 1, g.shape), 3) * 3).astype(np.float32)
    h = hydro_layers(dem, g)
    assert np.isfinite(h["hand"]).all()
    assert (h["hand"] <= cfg["detection"]["water_max_hand_m"]).all()
