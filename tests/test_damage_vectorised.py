"""The vectorised road / building / road-network steps must give EXACTLY the results of the original
one-by-one loops (kept here as the reference). Speed only: no result may change."""
import numpy as np
import geopandas as gpd
import pandas as pd
import pytest
import shapely

from floodsight.analysis import damage, network
from floodsight.analysis.damage import _lookup, sample_points
from floodsight.core import AOI, Grid, load_config


# ----------------------------------------------------------------------------- reference (original code)
def ref_assess_roads(roads, layers, grid, cfg):
    r = roads.to_crs(grid.crs).reset_index(drop=True)
    r = r[r.geometry.notna() & r.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    d = cfg["damage"]
    gx0, gy0, gx1, gy1 = grid.bounds
    status, cut_m, is_bridge = [], [], []
    for _, row in r.iterrows():
        bridge = str(row.get("bridge") or "no").lower() not in ("no", "none", "nan", "")
        bx0, by0, bx1, by1 = row.geometry.bounds
        if bx1 < gx0 or bx0 > gx1 or by1 < gy0 or by0 > gy1:
            status.append("outside_aoi"); cut_m.append(0.0); is_bridge.append(bridge)
            continue
        buf = d["bridge_buffer_m"] if bridge else d["road_buffer_m"]
        pts = sample_points(row.geometry, step=grid.res / 2)
        dist, inside = _lookup(layers["dist_m"], grid, pts)
        if not inside.any():
            status.append("outside_aoi"); cut_m.append(0.0); is_bridge.append(bridge)
            continue
        near = (dist <= buf) & inside
        nod, _ = _lookup(layers["nodata"], grid, pts)
        step = row.geometry.length / max(len(pts) - 1, 1)
        cut_m.append(float(near.sum() * step))
        if near.any():
            status.append("likely_hit" if (bridge or near.sum() >= 3) else "possibly_hit")
        elif (nod & inside).mean() > 0.5:
            status.append("not_observed")
        else:
            status.append("not_hit")
        is_bridge.append(bridge)
    r["status"] = status
    r["affected_m"] = np.round(cut_m, 1)
    r["is_bridge"] = is_bridge
    return r


def ref_build_edges(roads, grid, cfg):
    ncfg = cfg["network"]
    allowed = set(ncfg["motorable"])
    if ncfg.get("include_paths"):
        allowed |= {"path", "footway", "steps", "bridleway", "track"}
    r = roads.to_crs(grid.crs)
    if "highway" in r:
        r = r[r["highway"].isin(allowed)]
    speeds = ncfg.get("speed_kmh", {})
    default_speed = float(speeds.get("default", 25))
    a, b, ridx, bridge, kmh = [], [], [], [], []
    for i, row in r.iterrows():
        v = float(speeds.get(str(row.get("highway")), default_speed))
        for ln in getattr(row.geometry, "geoms", [row.geometry]):
            if ln.geom_type != "LineString":
                continue
            xy = np.asarray(ln.coords)[:, :2]
            a.append(xy[:-1]); b.append(xy[1:])
            ridx.append(np.full(len(xy) - 1, i)); bridge.append(np.full(len(xy) - 1, network._is_bridge(row.get("bridge"))))
            kmh.append(np.full(len(xy) - 1, v))
    if not a:
        return None
    return {"a": np.vstack(a), "b": np.vstack(b), "road": np.concatenate(ridx), "bridge": np.concatenate(bridge),
            "kmh": np.concatenate(kmh)}


# ----------------------------------------------------------------------------- data
@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(7)
    cfg = load_config()
    grid = Grid.for_aoi(AOI(85.30, 27.95, 85.40, 28.03), 20)
    x0, y0, x1, y1 = grid.bounds
    geoms, hw, br = [], [], []
    kinds = ["residential", "primary", "track", "footway", "service", "secondary", None]
    bridges = ["yes", None, np.nan, "no", "", "viaduct", "No", True]
    for i in range(1500):
        nv = int(rng.integers(2, 8))
        sx, sy = rng.uniform(x0 - 1500, x1 + 1500), rng.uniform(y0 - 1500, y1 + 1500)
        pts = np.cumsum(rng.normal(0, 70, (nv, 2)), 0) + [sx, sy]
        g = shapely.linestrings(pts)
        if i % 9 == 0:
            g = shapely.multilinestrings([g, shapely.linestrings(pts + [150, -90])])
        geoms.append(g)
        hw.append(kinds[i % len(kinds)])
        br.append(bridges[i % len(bridges)])
    roads = gpd.GeoDataFrame({"highway": hw, "bridge": pd.Series(br, dtype=object)}, geometry=geoms, crs=grid.crs)
    roads.index = roads.index * 3 + 11                 # non-trivial index labels (edges keep them)
    roads = roads.to_crs(4326)
    cls = np.zeros(grid.shape, np.uint8)
    conf = np.zeros(grid.shape, np.uint8)
    for _ in range(40):
        r, c = rng.integers(0, grid.height), rng.integers(0, grid.width)
        cls[r:r + 25, c:c + 40] = rng.choice([2, 3])
        conf[r:r + 25, c:c + 40] = rng.integers(1, 4)
    for _ in range(10):                                 # radar-blind patches (nodata)
        r, c = rng.integers(0, grid.height), rng.integers(0, grid.width)
        cls[r:r + 30, c:c + 30] = 255
    layers = damage.hit_layers(cls, conf, grid.res)
    cx, cy = rng.uniform(x0, x1, 3000), rng.uniform(y0, y1, 3000)
    blds = gpd.GeoDataFrame(geometry=shapely.box(cx - 8, cy - 6, cx + 8, cy + 6), crs=grid.crs).to_crs(4326)
    return cfg, grid, roads, layers, blds


def test_roads_identical(scene):
    cfg, grid, roads, layers, _ = scene
    new = damage.assess_roads(roads, layers, grid, cfg)
    old = ref_assess_roads(roads, layers, grid, cfg)
    assert list(new["status"]) == list(old["status"])
    assert np.array_equal(new["affected_m"].to_numpy(), old["affected_m"].to_numpy())
    assert list(new["is_bridge"]) == list(old["is_bridge"])
    assert set(new["status"]) >= {"likely_hit", "not_hit", "outside_aoi"}          # the test covers the cases


def test_roads_without_bridge_column(scene):
    cfg, grid, roads, layers, _ = scene
    r2 = roads.drop(columns=["bridge"])
    assert list(damage.assess_roads(r2, layers, grid, cfg)["status"]) == list(ref_assess_roads(r2, layers, grid, cfg)["status"])


def test_edges_identical(scene):
    cfg, grid, roads, _, _ = scene
    new, old = network.build_edges(roads, grid, cfg), ref_build_edges(roads, grid, cfg)
    for k in ("a", "b", "road", "bridge", "kmh"):
        assert np.array_equal(new[k], old[k]), k


def test_edges_skip_non_lines(scene):
    cfg, grid, roads, _, _ = scene
    r2 = roads.copy()
    r2.loc[r2.index[:5], "geometry"] = r2.geometry.iloc[:5].buffer(0.0001)      # polygons are skipped
    new, old = network.build_edges(r2, grid, cfg), ref_build_edges(r2, grid, cfg)
    for k in ("a", "b", "road", "bridge", "kmh"):
        assert np.array_equal(new[k], old[k]), k


def test_buildings_same_as_loop_centroids(scene):
    cfg, grid, _, layers, blds = scene
    new = damage.assess_buildings(blds, layers, grid, cfg)
    b = blds.to_crs(grid.crs)
    cent = np.array([(g.centroid.x, g.centroid.y) for g in b.geometry])
    assert np.array_equal(shapely.get_coordinates(shapely.centroid(b.geometry.values)), cent)
    assert {"possibly_hit", "not_hit"} <= set(new["status"])
