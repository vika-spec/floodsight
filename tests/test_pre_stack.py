"""Multi-image before-reference: mean of same-track before-images (linear), single image where too few see a
pixel; the catalogue offers earlier passes of the same track, newest first, only when asked for."""
from datetime import date

import numpy as np

from floodsight.fetch.s1 import stack_reference


def test_median_ignores_one_odd_image_and_falls_back():
    a, b, c = (np.full((2, 3, 3), v) for v in (0.1, 0.12, 0.9))   # c: one unusual (e.g. rain-soaked) image
    b[:, 0, 0] = c[:, 0, 0] = np.nan                                # only one image sees this pixel
    ref = stack_reference([a, b, c])
    assert np.allclose(ref[:, 1, 1], 10 * np.log10(0.12), atol=1e-4)   # median, not pulled up by 0.9
    assert np.allclose(ref[:, 0, 0], 10 * np.log10(0.1), atol=1e-4)


def test_speckle_lower_with_more_images():
    rng = np.random.default_rng(0)
    imgs = [rng.gamma(4.4, 0.1 / 4.4, (2, 200, 200)) for _ in range(4)]
    one = stack_reference(imgs[:1], min_obs=1)
    four = stack_reference(imgs)
    assert np.std(four) < 0.6 * np.std(one)     # about 1/sqrt(4) of the grain


def test_catalogue_offers_same_track_extra_pres(cfg):
    import copy

    from floodsight.core import AOI
    from floodsight.fetch import catalog

    aoi = AOI(85.0, 28.0, 85.1, 28.1)
    acq = lambda d, orb=85: catalog.Acquisition(d, orb, "ASCENDING", [], 1.0)
    acqs = [acq(date(2026, 7, 11)), acq(date(2026, 7, 23)), acq(date(2026, 8, 4)), acq(date(2026, 8, 16)),
            acq(date(2026, 8, 28)), acq(date(2026, 8, 10), orb=19)]
    catalog_group = catalog.group_acquisitions
    catalog.group_acquisitions = lambda scenes, a: acqs
    try:
        c = copy.deepcopy(cfg)
        c["sentinel1"]["pre_stack_images"] = 3
        p = [x for x in catalog.select_pairs([], aoi, date(2026, 8, 26), c) if x.pre.rel_orbit == 85][0]
        assert p.pre.day == date(2026, 8, 16) and [a.day for a in p.extra_pres] == [date(2026, 8, 4), date(2026, 7, 23)]
        c["sentinel1"]["pre_stack_images"] = 1
        p = [x for x in catalog.select_pairs([], aoi, date(2026, 8, 26), c) if x.pre.rel_orbit == 85][0]
        assert p.extra_pres == []
    finally:
        catalog.group_acquisitions = catalog_group
