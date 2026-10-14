"""Cut-off certainty levels: confident closure -> cut_off; only low-confidence -> likely_cut_off;
route only through radar blind spots -> unknown. Plus travel time and building counts."""
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point, box

from floodsight.analysis.damage import hit_layers
from floodsight.analysis.network import analyse_access
from floodsight.core import AOI, Grid
from floodsight.detect.baseline import DEBRIS, NODATA, NO_CHANGE


def _setup():
    g = Grid.for_aoi(AOI.parse("85.0,28.0,85.05,28.03"), 20.0)
    x0, y0, x1, y1 = g.bounds
    ym = (y0 + y1) / 2
    road = LineString([(x0 + 200, ym), (x1 - 200, ym)])            # one road: village (west) -> hospital (east)
    roads = gpd.GeoDataFrame({"highway": ["secondary"], "bridge": ["no"]}, geometry=[road], crs=g.crs)
    places = gpd.GeoDataFrame({"name": ["V"], "place": ["village"]}, geometry=[Point(x0 + 200, ym)], crs=g.crs)
    health = gpd.GeoDataFrame({"name": ["H"], "amenity": ["hospital"]}, geometry=[Point(x1 - 200, ym)], crs=g.crs)
    bld = gpd.GeoDataFrame(geometry=[box(x0 + 150 + i * 20, ym + 30, x0 + 160 + i * 20, ym + 40) for i in range(7)],
                           crs=g.crs)
    mid = g.width // 2
    return g, roads, places, health, bld, mid


def _status(cls, conf, g, roads, places, health, bld, cfg):
    res = analyse_access(roads, places, health, hit_layers(cls, conf, g.res), g, cfg, buildings=bld)
    return res["settlements"].iloc[0], res["summary"]


def test_certainty_levels_time_and_buildings(cfg):
    g, roads, places, health, bld, mid = _setup()
    cls = np.full(g.shape, NO_CHANGE, np.uint8)
    conf = np.zeros(g.shape, np.uint8)

    s, _ = _status(cls, conf, g, roads, places, health, bld, cfg)
    assert s["status"] == "reachable" and s["buildings"] == 7
    # secondary road at 55 km/h: minutes = km / 55 * 60
    assert abs(s["min_before"] - s["km_before"] / 55 * 60) <= 1

    cls[:, mid - 2:mid + 2] = DEBRIS
    conf[:, mid - 2:mid + 2] = 1                                     # low confidence only
    s, summ = _status(cls, conf, g, roads, places, health, bld, cfg)
    assert s["status"] == "likely_cut_off" and s["priority"] == 1
    assert summ["settlements_likely_cut_off"] == 1

    conf[:, mid - 2:mid + 2] = 3                                     # confident
    s, summ = _status(cls, conf, g, roads, places, health, bld, cfg)
    assert s["status"] == "cut_off" and summ["buildings_in_cut_off_settlements"] == 7

    cls[:] = NO_CHANGE
    conf[:] = 0
    cls[:, mid - 2:mid + 2] = NODATA                                 # radar blind strip across the only road
    s, _ = _status(cls, conf, g, roads, places, health, bld, cfg)
    assert s["status"] == "unknown"


def test_routes_and_blocking_points(cfg):
    g, roads, places, health, bld, mid = _setup()
    cls = np.full(g.shape, NO_CHANGE, np.uint8)
    conf = np.zeros(g.shape, np.uint8)
    cls[:, mid - 2:mid + 2] = DEBRIS
    conf[:, mid - 2:mid + 2] = 3
    res = analyse_access(roads, places, health, hit_layers(cls, conf, g.res), g, cfg, buildings=bld)
    r = res["routes"]
    assert list(r["route"]) == ["before"]                            # cut off: no route now
    assert r.geometry.iloc[0].length > 0.8 * roads.geometry.iloc[0].length
    b = res["route_blocks"]
    assert len(b) >= 1
    x = b.geometry.iloc[0].x
    x0 = g.bounds[0]
    assert abs((x - x0) / g.res - mid) < 6                           # the block is where the debris crosses the road
    assert "_node" not in res["settlements"].columns
