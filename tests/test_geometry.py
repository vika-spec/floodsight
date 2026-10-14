import numpy as np

from floodsight.preprocess.geometry import local_incidence, visibility_mask


def ramp(slope_deg, facing_east=True, n=120, res=20.0):
    x = np.arange(n) * res
    z = np.tan(np.radians(slope_deg)) * x * (-1 if facing_east else 1)
    return np.tile(z, (n, 1)).astype(np.float32)


def test_steep_slope_facing_sensor_is_layover():
    # ascending pass looks east; a slope rising toward the east faces the sensor (west)
    dem = ramp(50, facing_east=False)
    v = visibility_mask(dem, 20.0, 38.0, -12.0)
    assert v["layover"][10:-10, 10:-10].mean() > 0.9


def test_steep_slope_facing_away_is_shadow():
    dem = ramp(60, facing_east=True)
    v = visibility_mask(dem, 20.0, 38.0, -12.0)
    assert v["shadow"][10:-10, 10:-10].mean() > 0.9


def test_flat_ground_visible_and_incidence_matches():
    dem = np.zeros((50, 50), np.float32)
    v = visibility_mask(dem, 20.0, 38.0, -12.0)
    assert not v["blind"].any()
    assert np.allclose(local_incidence(dem, 20.0, 38.0, -12.0), 38.0, atol=0.1)
