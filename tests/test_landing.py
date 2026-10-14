"""Helicopter access: candidate landing ground must be flat, clear, dry, visible and away from buildings;
mapped helipads report what the flood map shows at the pad; cut-off settlements get the nearest of each."""
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from floodsight.analysis.landing import candidate_zones, helicopter_access
from floodsight.core import AOI, Grid
from floodsight.detect.baseline import DEBRIS, FLOOD, NO_CHANGE, NODATA

CFG = {"landing": {"max_slope_deg": 7, "clear_diameter_m": 60, "flood_clearance_m": 50,
                   "building_clearance_m": 15, "max_distance_km": 5}}


def _scene():
    g = Grid.for_aoi(AOI(85.30, 27.90, 85.32, 27.92), 20)     # ~ 2 x 2 km, 20 m pixels
    H, W = g.shape
    cls = np.full((H, W), NO_CHANGE, np.uint8)
    slope = np.full((H, W), 25.0, np.float32)                   # steep everywhere ...
    slope[10:30, 10:30] = 2.0                                   # ... except a flat 400 x 400 m field (A)
    slope[60:80, 60:80] = 2.0                                   # and a second flat field (B) ...
    cls[60:80, 60:80] = FLOOD                                   # ... that is flooded
    slope[10:30, 60:80] = 2.0                                   # third flat field (C) in a radar blind spot
    cls[10:30, 60:80] = NODATA
    return g, cls, slope


def _xy(g, r, c):
    return g.transform * (c + 0.5, r + 0.5)


def test_zone_only_on_flat_dry_visible_ground():
    g, cls, slope = _scene()
    z = candidate_zones(cls, slope, None, g, CFG)
    assert len(z) == 1
    r, c = g.xy_to_rowcol(np.array([z[0]["x"]]), np.array([z[0]["y"]]))
    assert 10 <= r[0] < 30 and 10 <= c[0] < 30 and z[0]["slope_deg"] == 2.0


def test_buildings_and_flood_clearance_remove_ground():
    g, cls, slope = _scene()
    x0, y0 = _xy(g, 10, 10)
    x1, y1 = _xy(g, 29, 29)
    houses = gpd.GeoDataFrame(geometry=[box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))], crs=g.crs)
    assert candidate_zones(cls, slope, houses, g, CFG) == []      # field covered by a building
    cls2 = cls.copy()
    cls2[31:33, 10:30] = DEBRIS                                   # debris right next to the field
    z = candidate_zones(cls2, slope, None, g, CFG)
    assert all(_rc_ok(g, zz, max_row=31 - 3) for zz in z)         # centre kept >= 50 m (2.5 px) away


def _rc_ok(g, z, max_row):
    r, _ = g.xy_to_rowcol(np.array([z["x"]]), np.array([z["y"]]))
    return r[0] <= max_row


def test_access_for_cut_off_settlement():
    g, cls, slope = _scene()
    sx, sy = _xy(g, 50, 20)                                      # village south of field A
    px, py = _xy(g, 70, 70)                                      # mapped helipad inside flooded field B
    qx, qy = _xy(g, 45, 45)                                      # second helipad, dry
    settle = gpd.GeoDataFrame([{"name": "Village X", "status": "cut_off", "priority": 1, "geometry": Point(sx, sy)},
                               {"name": "Village Y", "status": "reachable", "priority": None, "geometry": Point(qx, qy)}],
                              crs=g.crs)
    pads = gpd.GeoDataFrame([{"name": "Flooded pad", "aeroway": "helipad", "geometry": Point(px, py)},
                             {"name": "Dry pad", "aeroway": "helipad", "geometry": Point(qx, qy)}], crs=g.crs).to_crs(4326)
    out = helicopter_access(cls, slope, settle, pads, None, g, CFG)
    states = {h["name"]: h["state"] for h in out["helipads"]}
    assert states["Flooded pad"].startswith("flood") and states["Dry pad"] == "no damage detected"
    assert [p["name"] for p in out["per_settlement"]] == ["Village X"]          # only settlements needing air access
    p = out["per_settlement"][0]
    assert p["helipad"]["name"] == "Dry pad"                                     # flooded pad never suggested
    assert p["zone"] is not None and p["zone"]["km"] < 1.0


def test_power_line_removes_ground_and_text_is_honest():
    from shapely.geometry import LineString
    g, cls, slope = _scene()
    x0, y0 = _xy(g, 20, 5)
    x1, y1 = _xy(g, 20, 35)
    line = gpd.GeoDataFrame(geometry=[LineString([(x0, y0), (x1, y1)])], crs=g.crs)   # wire across field A
    cfg = {"landing": {**CFG["landing"], "power_line_clearance_m": 30}}
    z = candidate_zones(cls, slope, None, g, cfg, power_lines=line)
    rows = [g.xy_to_rowcol(np.array([zz["x"]]), np.array([zz["y"]]))[0][0] for zz in z]
    assert z and all(abs(r - 20) * 20 >= 30 + 30 for r in rows)    # centre: wire buffer + clear radius away
    out = helicopter_access(cls, slope, None, None, None, g, cfg, power_lines=line)
    assert "missing from OpenStreetMap" in out["not_checked"] and "our map missed" in out["not_checked"]
    assert "FM 3-21.38" in out["criteria"]["slope_source"] and out["criteria"]["power_lines_mapped"] == 1


def test_no_cut_off_settlement_skips_ground_search_but_keeps_helipads():
    g, cls, slope = _scene()
    qx, qy = _xy(g, 45, 45)
    settle = gpd.GeoDataFrame([{"name": "Village Y", "status": "reachable", "priority": None, "geometry": Point(qx, qy)}],
                              crs=g.crs)
    pads = gpd.GeoDataFrame([{"name": "Dry pad", "aeroway": "helipad", "geometry": Point(qx, qy)}], crs=g.crs).to_crs(4326)
    out = helicopter_access(cls, slope, settle, pads, None, g, CFG)
    assert out["zones_searched"] is False and out["zones_total"] == 0 and out["per_settlement"] == []
    assert [h["name"] for h in out["helipads"]] == ["Dry pad"]
    assert candidate_zones(cls, slope, None, g, CFG)                     # the ground exists; it was just not needed


def test_building_clearance_identical_to_one_by_one_buffers():
    from rasterio.features import rasterize
    from shapely.geometry import box
    from floodsight.analysis.landing import _occupied
    g, _, _ = _scene()
    x0, y0 = _xy(g, 10, 10)
    blds = gpd.GeoDataFrame(geometry=[box(x0 + 37 * i, y0 - 23 * i, x0 + 37 * i + 12, y0 - 23 * i + 9) for i in range(30)],
                            crs=g.crs)
    old = rasterize(((geom.buffer(25), 1) for geom in blds.geometry), out_shape=g.shape, transform=g.transform,
                    fill=0, all_touched=True, dtype="uint8").astype(bool)
    assert np.array_equal(_occupied(blds, 25, g), old)
