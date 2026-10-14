"""Debris is confident only when an independent observation agrees: the other orbit track (within the
geolocation tolerance) or Sentinel-2. One radar image alone, however strong, stays low confidence."""
import numpy as np

from floodsight.detect.baseline import DEBRIS, FLOOD, NO_CHANGE, NODATA
from floodsight.detect.fuse import apply_optical, merge_tracks


def _track(debris_cols, shape=(6, 12), nodata_cols=()):
    cls = np.full(shape, NO_CHANGE, np.uint8)
    conf = np.zeros(shape, np.uint8)
    for c in debris_cols:
        cls[:, c], conf[:, c] = DEBRIS, 1
    for c in nodata_cols:
        cls[:, c] = NODATA
    return {"cls": cls, "conf": conf}


def test_two_tracks_agree_within_tolerance():
    a = _track([2, 3, 8])
    b = _track([4, 11])                                      # col 4 is 1 px from col 3; col 11 is 3 px from 8
    cls, conf = merge_tracks([a, b], corroboration_px=2)
    assert (cls[:, [2, 3, 4, 8, 11]] == DEBRIS).all()       # map extent = union, unchanged
    assert (conf[:, 3] == 2).all() and (conf[:, 4] == 2).all() and (conf[:, 2] == 2).all()
    assert (conf[:, 8] == 1).all() and (conf[:, 11] == 1).all()   # too far apart: not corroborated


def test_single_track_stays_low():
    cls, conf = merge_tracks([_track([5])], corroboration_px=2)
    assert (cls[:, 5] == DEBRIS).all() and (conf[:, 5] == 1).all()


def test_flood_water_confidence_untouched():
    a = _track([])
    a["cls"][:, 0], a["conf"][:, 0] = FLOOD, 3
    cls, conf = merge_tracks([a, _track([])])
    assert (conf[:, 0] == 3).all()


def test_sentinel2_confirmation_raises_single_track_debris():
    cls, conf = merge_tracks([_track([5])])
    sh = cls.shape
    opt = {"valid_both": np.ones(sh, bool), "new_water": np.zeros(sh, bool), "debris": np.zeros(sh, bool),
           "ndvi_post": np.zeros(sh, np.float32)}
    opt["debris"][:, 5] = True
    cls2, conf2, _ = apply_optical(cls, conf, opt, np.ones(sh, bool))
    assert (conf2[:, 5] == 2).all()


def test_a_track_that_cannot_see_never_overwrites_one_that_can():
    """Regression (10 Oct 2026): uint8 wrap-around made 'no data' rank highest, so a later track's blind or
    uncovered pixels replaced an earlier track's valid observations."""
    import numpy as np

    from floodsight.detect.baseline import FLOOD, NO_CHANGE, NODATA
    from floodsight.detect.fuse import merge_tracks

    def res(c):
        return {"cls": c, "conf": np.where(c == FLOOD, 2, 0).astype(np.uint8)}
    seen = np.full((4, 4), NO_CHANGE, np.uint8)
    seen[0, 0] = FLOOD
    partial = np.full((4, 4), NODATA, np.uint8)
    partial[:2] = NO_CHANGE
    for order in ([seen, partial], [partial, seen], [seen, seen, partial]):
        cls, conf = merge_tracks([res(c) for c in order], 2)
        assert (cls == NODATA).sum() == 0
        assert cls[0, 0] == FLOOD and conf[0, 0] == 2
    cls, _ = merge_tracks([res(partial), res(partial)], 2)
    assert (cls == NODATA).sum() == 8                     # nobody sees the lower half: stays unknown
