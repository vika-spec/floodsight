"""The model must use every orbit track, like the rules. Found on Slovenia (Mura): the model ran only on the
first track, whose after-image was taken before the river rose, and said 'no flood' where the second track
(one day later) clearly showed it."""
import numpy as np

from floodsight.detect.baseline import FLOOD, NO_CHANGE, NODATA
from floodsight.detect.fuse import combine_track_probs


def test_max_over_tracks_that_see_the_pixel():
    early = np.array([[0.0, 0.1, 0.9, 0.2]], np.float32)       # first pass: river not up yet
    late = np.array([[0.95, 0.0, 0.3, 0.99]], np.float32)      # second pass, a day later
    cls_early = np.array([[NO_CHANGE, NO_CHANGE, FLOOD, NO_CHANGE]], np.uint8)
    cls_late = np.array([[FLOOD, NO_CHANGE, NO_CHANGE, NODATA]], np.uint8)   # last pixel in its shadow
    p = combine_track_probs([early, late], [cls_early, cls_late])
    assert np.allclose(p, [[0.95, 0.1, 0.9, 0.2]])             # shadowed 0.99 ignored, early 0.2 kept


def test_pixel_no_track_sees_is_nan():
    p = combine_track_probs([np.ones((1, 2), np.float32)], [np.array([[NODATA, NO_CHANGE]], np.uint8)])
    assert np.isnan(p[0, 0]) and p[0, 1] == 1
