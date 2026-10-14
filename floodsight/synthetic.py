"""Synthetic mountain valley with a flood and a debris flow, for tests, CI and the offline demo mode.

Layout (mirrors the brief's 'Who is cut off?' figure): a river runs west->east along a V-shaped valley.
The main road follows the north bank and dips to the valley floor in two places. The flood (water ponds)
cuts the first dip; the debris flow cuts the second. Hospital West and Hospital East sit at the two ends.
Expected answer: Village A and Village D keep a road to a hospital; Villages B and C are cut off;
Village E (across a destroyed bridge) is cut off.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from .core import AOI, Grid

DEFAULT_AOI = AOI(85.30, 27.95, 85.44, 28.04)


def _speckle(rng, shape, enl=4.4):
    return rng.gamma(enl, 1.0 / enl, size=shape)


def make_scene(aoi: AOI = DEFAULT_AOI, res: float = 20.0, flood: bool = True, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    grid = Grid.for_aoi(aoi, res)
    H, W = grid.shape
    R, C = np.mgrid[0:H, 0:W].astype(np.float32)
    xs = np.arange(W, dtype=np.float32)
    centre = H * 0.5 + 60 * np.sin(xs / 90.0)                        # river centre row per column
    dist_m = (R - centre[None, :]) * res                              # signed metres across the valley
    adist = np.abs(dist_m)

    noise = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 12) * 400
    # flat-ish valley floor (~220 m each side of the river) then 22-degree walls
    dem = 1400 - C * res * 0.01 + adist * 0.02 + np.maximum(adist - 220, 0) * 0.40 \
        + np.clip(noise, -150, 150) * (adist > 500)
    dem = dem.astype(np.float32)

    # --- truth masks --------------------------------------------------------------------------------
    fx = (C >= 0.33 * W) & (C <= 0.60 * W)
    dx = (C >= 0.62 * W) & (C <= 0.86 * W)
    river = adist < 25
    flood_m = fx & (adist < 160) & ~river if flood else np.zeros((H, W), bool)
    debris_m = dx & (adist < 210) & ~river if flood else np.zeros((H, W), bool)

    # --- radar (dB, gamma0-like) --------------------------------------------------------------------
    gy, gx = np.gradient(dem, res)
    terrain = -6.0 * gx                                               # slopes facing west-looking-east sensor brighter
    base_vv = -9.0 + np.clip(terrain, -4, 4) + ndimage.gaussian_filter(rng.normal(0, 1.2, (H, W)), 2)
    base_vh = base_vv - 7.0

    def to_db(level_db, s):
        return (10 * np.log10(10 ** (level_db / 10) * s)).astype(np.float32)

    vv0 = np.where(river, -21.0, base_vv)
    vh0 = np.where(river, -28.0, base_vh)
    vv1, vh1 = vv0.copy(), vh0.copy()
    vv1[flood_m], vh1[flood_m] = -21.5, -28.5
    vv1[debris_m] += 4.5
    vh1[debris_m] += 5.0
    rough = ndimage.gaussian_filter(rng.normal(0, 1, (H, W)), 1) * 3.0
    vv1[debris_m] += rough[debris_m]
    pre = np.stack([to_db(vv0, _speckle(rng, (H, W))), to_db(vh0, _speckle(rng, (H, W)))])
    post = np.stack([to_db(vv1, _speckle(rng, (H, W))), to_db(vh1, _speckle(rng, (H, W)))])

    truth = np.zeros((H, W), np.uint8)
    truth[flood_m] = 1
    truth[debris_m] = 2

    osm = _make_osm(grid, centre, res, W, H)
    return {"grid": grid, "aoi": aoi, "dem": dem, "pre": pre, "post": post, "truth": truth, "osm": osm,
            "heading": -12.0, "incidence": 38.0}


def _make_osm(grid: Grid, centre, res, W, H) -> dict:
    import geopandas as gpd
    from pyproj import Transformer
    from shapely.geometry import LineString, Point, Polygon

    to_ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
    t = grid.transform

    def ll(col, row):
        x, y = grid.rowcol_to_xy(row, col)
        return to_ll.transform(x, y)

    def road_row(c):
        # north bank, 260 m from river; dips to 40 m in two places (the fragile valley-floor sections)
        off = 260.0
        for a, b in ((0.40, 0.46), (0.70, 0.76)):
            if a * W <= c <= b * W:
                off = 40.0
        return centre[int(min(max(c, 0), W - 1))] - off / res

    villages = {"Village A": 0.18, "Village B": 0.50, "Village C": 0.58, "Village D": 0.93}
    cb = 0.52 * W
    # junction columns are included as exact vertices so side roads share a node with the highway (as in OSM)
    cols = np.unique(np.concatenate([np.linspace(0, W - 1, 220), [f * W for f in villages.values()], [cb]]))
    main = LineString([ll(c, road_row(c)) for c in cols])
    roads = [{"geometry": main, "highway": "primary", "name": "Valley Highway", "bridge": None}]

    places = []
    for name, f in villages.items():
        c = f * W
        r0 = road_row(c)
        r1 = r0 - 900 / res
        roads.append({"geometry": LineString([ll(c, r0), ll(c, r1)]), "highway": "unclassified",
                      "name": f"{name} road", "bridge": None})
        places.append({"geometry": Point(ll(c, r1)), "name": name, "place": "village"})

    # bridge across the river inside the flood zone to Village E on the south bank
    rs = centre[int(cb)] + 200 / res
    roads.append({"geometry": LineString([ll(cb, road_row(cb)), ll(cb, centre[int(cb)] - 40 / res)]),
                  "highway": "unclassified", "name": "Bridge approach N", "bridge": None})
    roads.append({"geometry": LineString([ll(cb, centre[int(cb)] - 40 / res), ll(cb, centre[int(cb)] + 40 / res)]),
                  "highway": "unclassified", "name": "Sano Bridge", "bridge": "yes"})
    roads.append({"geometry": LineString([ll(cb, centre[int(cb)] + 40 / res), ll(cb, rs)]),
                  "highway": "unclassified", "name": "Bridge approach S", "bridge": None})
    places.append({"geometry": Point(ll(cb, rs)), "name": "Village E", "place": "hamlet"})
    places.append({"geometry": Point(ll(0.43 * W, road_row(0.43 * W) + 3)), "name": "Riverside Bazar",
                   "place": "hamlet"})

    health = [
        {"geometry": Point(ll(2, road_row(2))), "name": "Hospital West", "amenity": "hospital"},
        {"geometry": Point(ll(W - 3, road_row(W - 3))), "name": "Hospital East", "amenity": "hospital"},
    ]

    # buildings: clusters at each village + a roadside hamlet on the valley floor inside the flood dip
    rng = np.random.default_rng(1)
    b = []
    sites = [(f * W, road_row(f * W) - 900 / res) for f in villages.values()]
    sites += [(cb, rs), (0.43 * W, road_row(0.43 * W) + 3), (0.73 * W, road_row(0.73 * W) + 3)]
    for (c0, r0) in sites:
        for _ in range(25):
            c = c0 + rng.normal(0, 4)
            r = r0 + rng.normal(0, 3)
            s = 0.5
            b.append({"geometry": Polygon([ll(c - s, r - s), ll(c + s, r - s), ll(c + s, r + s), ll(c - s, r + s)]),
                      "building": "yes"})
    return {
        "roads": gpd.GeoDataFrame(roads, crs=4326),
        "places": gpd.GeoDataFrame(places, crs=4326),
        "health": gpd.GeoDataFrame(health, crs=4326),
        "buildings": gpd.GeoDataFrame(b, crs=4326),
        # one mapped helipad near the eastern hospital (demo of the helicopter-access section)
        "helipads": gpd.GeoDataFrame([{"geometry": Point(ll(W - 6, road_row(W - 6) - 8)), "name": "District helipad",
                                       "aeroway": "helipad"}], crs=4326),
        "snapshot": "2026-07-27",
    }
