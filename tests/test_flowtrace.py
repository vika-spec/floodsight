"""Flood path from any point: continues across windows (not limited to an analysed box), lists places on
the valley floor with km downstream and height above the river, ignores places high above it, measures how
well the path follows the mapped river, and works offline from a demo run."""
import geopandas as gpd
import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString, Point

from floodsight.analysis.flowtrace import flow_trace

LON0, LAT0 = 85.0, 28.0
EPSG = 32645
to_xy = Transformer.from_crs(4326, EPSG, always_xy=True)
to_ll = Transformer.from_crs(EPSG, 4326, always_xy=True)
X0, Y0 = to_xy.transform(LON0, LAT0)


def valley_dem(grid, aoi):
    """A valley running east from the start, falling 20 m per km, walls rising 0.2 m per m."""
    t = Transformer.from_crs(grid.crs, EPSG, always_xy=True)
    r, c = np.mgrid[0:grid.height, 0:grid.width]
    x, y = grid.rowcol_to_xy(r, c)
    X, Y = t.transform(x, y)
    return (4000 - 0.02 * (X - X0) + 0.2 * np.abs(Y - Y0)).astype(np.float32)


def ll(dx_km, dy_m=0):
    return to_ll.transform(X0 + dx_km * 1000, Y0 + dy_m)


def osm(key, bbox):
    if key == "places":
        pts = [("Near village", ll(12, 60)), ("Far village", ll(45, -100)), ("High village", ll(20, 900)),
               ("Terrace village", ll(30, 400))]
        return gpd.GeoDataFrame({"name": [n for n, _ in pts], "place": ["village"] * len(pts)},
                                geometry=[Point(*p) for _, p in pts], crs=4326)
    if key == "rivers":
        return gpd.GeoDataFrame({"waterway": ["river"]}, geometry=[LineString([ll(-1), ll(80)])], crs=4326)
    if key == "bridges":
        return gpd.GeoDataFrame({"name": ["Bridge"], "highway": ["primary"]},
                                geometry=[LineString([ll(30, -50), ll(30, 50)])], crs=4326)
    return None


def test_trace_crosses_windows_and_lists_places(cfg):
    lon, lat = ll(0, 200)                         # click 200 m off the river: snaps onto it
    out = flow_trace(lon, lat, cfg, max_km=60, dem_reader=valley_dem, osm_reader=osm)
    assert 58 <= out["length_km"] <= 61 and out["windows"] >= 3            # well beyond one 30 km window
    assert out["stop_reason"] == "length limit reached"
    names = [s["name"] for s in out["settlements"]]
    assert names == ["Near village", "Terrace village", "Far village"]       # sorted; 180 m above: not listed
    terrace = out["settlements"][1]
    assert terrace["band"].startswith("> 50 m") and 60 <= terrace["m_above_river"] <= 100
    near = out["settlements"][0]
    assert abs(near["km_downstream"] - 12) < 1.5 and near["m_above_river"] <= 15
    assert out["path_on_mapped_river"] > 0.9
    assert out["bridges"] and abs(out["bridges"][0]["km_downstream"] - 30) < 2
    assert out["corridor"]["features"] and out["drop_m"] > 1000
    assert len(out["profile"]["km"]) == len(out["profile"]["elev_m"])


def test_demo_run_flow_offline(tmp_path, cfg):
    from floodsight.analysis.flowtrace import run_readers
    from floodsight.pipeline import run
    r = run(None, "2026-08-26", cfg, demo=True, runs_dir=tmp_path)
    d = tmp_path / r["run_id"]
    import json
    R = json.loads((d / "results.json").read_text())
    w, s, e, n = R["aoi"]["bbox"]
    out = flow_trace(w + 0.02, (s + n) / 2, cfg, max_km=30, run_dir=d, **run_readers(d))
    assert out["length_km"] > 3
    assert "no elevation data" in out["stop_reason"] or out["stop_reason"] == "length limit reached"
    assert all("damage_nearby" in x for x in out["settlements"])


def test_tile_seam_does_not_stop_the_path(cfg):
    from floodsight.fetch.dem import fill_seams

    def seam_dem(grid, aoi, fill=True):
        """A 1-px no-data line 20 km downstream, at a fixed place, like a 1-degree tile seam."""
        z = valley_dem(grid, aoi)
        t = Transformer.from_crs(grid.crs, EPSG, always_xy=True)
        r, c = np.mgrid[0:grid.height, 0:grid.width]
        X, _ = t.transform(*grid.rowcol_to_xy(r, c))
        z[np.abs(X - (X0 + 20000)) < grid.res / 2] = np.nan
        return fill_seams(z) if fill else z

    broken = flow_trace(*ll(0, 0), cfg, max_km=40, dem_reader=lambda g, a: seam_dem(g, a, False), osm_reader=osm)
    assert broken["length_km"] < 21                       # without the fix the seam ends the path (the bug)

    out = flow_trace(*ll(0, 0), cfg, max_km=40, dem_reader=seam_dem, osm_reader=osm)
    assert out["length_km"] > 39 and out["stop_reason"] == "length limit reached"


def test_windows_ahead_same_path_fewer_windows(cfg):
    """Windows placed ahead of the path: fewer windows, same path; river check per stretch."""
    lon, lat = ll(0, 200)
    out = flow_trace(lon, lat, cfg, max_km=80, dem_reader=valley_dem, osm_reader=osm)
    assert out["windows"] <= 5                       # centred windows needed ~7 for 80 km
    xs = [to_xy.transform(a, b)[1] - Y0 for a, b in out["path"]["geometry"]["coordinates"]]
    assert max(abs(v) for v in xs) < 100             # stays on the valley axis all the way
    assert out["rivers_followed"] and out["rivers_followed"][0]["name"] == "unnamed river"
    segs = out["path_check"]["features"]
    on = [f["properties"] for f in segs if f["properties"]["on_mapped_river"]]
    assert on and on[0]["from_km"] == 0 and on[0]["to_km"] >= 79      # the mapped river ends at 80 km
    assert out["length_km"] >= 79
