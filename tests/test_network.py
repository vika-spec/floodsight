from floodsight.analysis.damage import assess_roads, hit_layers, summarise, assess_buildings
from floodsight.analysis.network import analyse_access


def test_expected_cut_off_villages(scene, detection, cfg):
    o = scene["osm"]
    layers = hit_layers(detection["cls"], detection["conf"], scene["grid"].res)
    res = analyse_access(o["roads"], o["places"], o["health"], layers, scene["grid"], cfg)
    st = dict(zip(res["settlements"]["name"], res["settlements"]["status"]))
    assert st["Village A"] == "reachable"
    assert st["Village D"] == "reachable"
    # blocked by confident flood water (west of the bridge): cut off
    for v in ("Village E", "Riverside Bazar"):
        assert st[v] == "cut_off", v
    # blocked only by debris seen by the demo's single radar track: not independently confirmed -> likely
    for v in ("Village B", "Village C"):
        assert st[v] == "likely_cut_off", v


def test_bridge_hit(scene, detection, cfg):
    o = scene["osm"]
    layers = hit_layers(detection["cls"], detection["conf"], scene["grid"].res)
    rd = assess_roads(o["roads"], layers, scene["grid"], cfg)
    b = assess_buildings(o["buildings"], layers, scene["grid"], cfg)
    s = summarise(b, rd)
    assert s["bridges_hit"] == 1
    assert s["buildings_likely_hit"] > 0


def test_nothing_cut_without_flood(scene, cfg):
    import numpy as np
    o = scene["osm"]
    cls = np.zeros(scene["grid"].shape, np.uint8)
    layers = hit_layers(cls, cls, scene["grid"].res)
    res = analyse_access(o["roads"], o["places"], o["health"], layers, scene["grid"], cfg)
    assert set(res["settlements"]["status"]) == {"reachable"}


def test_empty_osm_is_handled(scene, detection, cfg):
    import geopandas as gpd
    from floodsight.analysis.damage import summarise
    empty = gpd.GeoDataFrame(geometry=[], crs=4326)
    layers = hit_layers(detection["cls"], detection["conf"], scene["grid"].res)
    notes = []
    res = analyse_access(empty, empty, empty, layers, scene["grid"], cfg, notes)
    assert res["settlements"].empty and notes
    o = scene["osm"]
    res = analyse_access(o["roads"], o["places"], empty, layers, scene["grid"], cfg, notes)   # no hospitals
    assert any("hospital" in n.lower() for n in notes)
    assert summarise(empty.assign(status=[]), assess_roads(o["roads"], layers, scene["grid"], cfg))["bridges_hit"] == 1


def test_fast_routing_matches_reference():
    """The compiled multi-source search must give exactly the distances of the original per-hospital
    networkx search (kept in network.build_graph as the reference), before and after damage."""
    import networkx as nx
    import numpy as np

    from floodsight.analysis.network import build_graph, merge_nodes, nearest_target, road_matrix

    rng = np.random.default_rng(3)
    # irregular grid-like road network with jittered junctions, duplicate segments and dead ends
    n = 60
    xs, ys = np.meshgrid(np.arange(n) * 100.0, np.arange(n) * 100.0)
    P = np.c_[xs.ravel(), ys.ravel()] + rng.normal(0, 15, (n * n, 2))
    idx = np.arange(n * n).reshape(n, n)
    pairs = np.r_[np.c_[idx[:, :-1].ravel(), idx[:, 1:].ravel()], np.c_[idx[:-1].ravel(), idx[1:].ravel()]]
    pairs = pairs[rng.random(len(pairs)) < 0.8]
    a, b = P[pairs[:, 0]], P[pairs[:, 1]] + rng.normal(0, 0.2, (len(pairs), 2))   # sub-metre noise at junctions
    a, b = np.r_[a, a[:50]], np.r_[b, b[:50]]                  # exact duplicate segments (OSM sometimes has them)
    pairs = np.r_[pairs, pairs[:50]]
    edges = {"a": a, "b": b}
    broken = rng.random(len(pairs)) < 0.15
    coords, ua, ub, nn = merge_nodes(edges)
    L = np.hypot(*(edges["b"] - edges["a"]).T)
    pick = rng.choice(len(pairs), 12, replace=False)            # hospitals at the start of 12 random roads

    for brk in (None, broken):
        G, meta = build_graph(edges, brk)
        ref = {}
        for node in {int(meta["ua"][e]) for e in pick}:
            if node in G:
                for k, v in nx.single_source_dijkstra_path_length(G, node, weight="length").items():
                    ref[k] = min(v, ref.get(k, np.inf))
        # same junction positions for every road end (node ids differ, positions must not)
        # (a merged junction may keep any of its member positions, all within the 1 m merge tolerance chain)
        assert np.abs(coords[ua] - meta["coords"][meta["ua"]]).max() < 3.0
        assert np.abs(coords[ub] - meta["coords"][meta["ub"]]).max() < 3.0
        M, present = road_matrix(ua, ub, L, np.ones(len(L), bool) if brk is None else ~brk, nn)
        d, _ = nearest_target(M, present, [(int(ua[e]), f"H{i}") for i, e in enumerate(pick)])
        # compare at every road end point (the two methods number junctions differently)
        for old_ids, new_ids in ((meta["ua"], ua), (meta["ub"], ub)):
            old = np.array([ref.get(int(i), np.inf) for i in old_ids])
            new = d[new_ids]
            assert np.array_equal(np.isfinite(old), np.isfinite(new))
            f = np.isfinite(old)
            assert f.sum() > 1000 and np.abs(old[f] - new[f]).max() < 1e-6


def _reference_access(roads, places, health, dist_m, grid, cfg):
    """Slow, independent reference: networkx graph from build_graph, one Dijkstra per hospital, villages
    snapped with the reference graph's own node coordinates. Same status rules as analyse_access."""
    import networkx as nx
    import numpy as np
    from scipy.spatial import cKDTree

    from floodsight.analysis.network import build_edges, build_graph, edge_damage

    n = cfg["network"]
    edges = build_edges(roads, grid, cfg)
    broken = edge_damage(edges, dist_m, grid, cfg)
    G0, meta = build_graph(edges)
    G1, _ = build_graph(edges, broken)
    ids = np.array(sorted(G0.nodes))
    tree = cKDTree(meta["coords"][ids])

    def snap(gdf):
        g = gdf.to_crs(grid.crs)
        d, i = tree.query(np.array([(p.x, p.y) for p in g.geometry]))
        return d, ids[i]

    td, tn = snap(health)
    targets = [int(t) for t, d in zip(tn, td) if d <= n["snap_max_m"]]

    def dist(G):
        best = {}
        for t in targets:
            if t in G:
                for k, v in nx.single_source_dijkstra_path_length(G, t, weight="length").items():
                    best[k] = min(v, best.get(k, np.inf))
        return best

    d0, d1 = dist(G0), dist(G1)
    sd, sn = snap(places)
    out = {}
    for name, node, snap_m in zip(places["name"], sn, sd):
        a, b = d0.get(int(node), np.inf), d1.get(int(node), np.inf)
        if snap_m > n["snap_max_m"]:
            st = "no_road_data"
        elif not np.isfinite(a):
            st = "isolated_before"
        elif not np.isfinite(b):
            st = "cut_off"
        elif b > a * n["detour_ratio"] and (b - a) / 1000 >= n["detour_min_extra_km"]:
            st = "detour"
        else:
            st = "reachable"
        out[name] = (st, round(a / 1000, 1) if np.isfinite(a) else None)
    return out


def test_full_access_matches_reference_on_realistic_roads(cfg):
    """Regression for a real bug: villages were snapped to the wrong junction after the fast routing
    renumbered nodes. OSM-like roads (polylines sharing vertices), random villages, hospitals and floods."""
    import geopandas as gpd
    import numpy as np
    from scipy import ndimage
    from shapely.geometry import LineString, Point

    from floodsight.core import AOI, Grid

    g = Grid.for_aoi(AOI.parse("85.30,27.95,85.40,28.05"), 20)
    x0, y0 = g.transform.c, g.transform.f + g.transform.e * g.height
    for seed in range(3):
        rng = np.random.default_rng(seed)
        verts, lines = [(x0 + 5000, y0 + 5000)], []
        for _ in range(300):
            pts = [verts[rng.integers(len(verts))]]
            ang = rng.uniform(0, 2 * np.pi)
            for _ in range(rng.integers(3, 15)):
                ang += rng.normal(0, 0.4)
                pts.append((pts[-1][0] + 150 * np.cos(ang), pts[-1][1] + 150 * np.sin(ang)))
            verts += pts[1:]
            lines.append(LineString(pts))
        roads = gpd.GeoDataFrame({"highway": ["primary"] * len(lines), "bridge": ["no"] * len(lines)},
                                 geometry=lines, crs=g.crs).to_crs(4326)
        vs = np.array(verts)
        pick = rng.choice(len(vs), 50, replace=False)
        places = gpd.GeoDataFrame({"name": [f"P{i}" for i in range(40)], "place": ["village"] * 40},
                                  geometry=[Point(vs[i] + rng.normal(0, 30, 2)) for i in pick[:40]], crs=g.crs).to_crs(4326)
        health = gpd.GeoDataFrame({"name": [f"H{i}" for i in range(10)], "amenity": ["hospital"] * 10},
                                  geometry=[Point(vs[i]) for i in pick[40:]], crs=g.crs).to_crs(4326)
        flood = np.zeros(g.shape, bool)
        for _ in range(25):
            r, c = rng.integers(0, g.height), rng.integers(0, g.width)
            flood[max(0, r - 15):r + 15, max(0, c - 15):c + 15] = True
        dist_m = ndimage.distance_transform_edt(~flood) * g.res
        ref = _reference_access(roads, places, health, dist_m, g, cfg)
        res = analyse_access(roads, places, health, {"dist_m": dist_m}, g, cfg)["settlements"]
        got = {r["name"]: (r["status"], r["km_before"]) for _, r in res.iterrows()}
        assert set(got) == set(ref)
        # statuses must match exactly; distances to within the 0.1 km rounding (summation order can move a
        # value like 1650.0000000001 m vs 1649.9999999995 m across the rounding boundary)
        bad = {k: (got[k], ref[k]) for k in ref if got[k][0] != ref[k][0] or
               (got[k][1] is None) != (ref[k][1] is None) or
               (got[k][1] is not None and abs(got[k][1] - ref[k][1]) > 0.1 + 1e-9)}
        assert not bad, f"seed {seed}: {bad}"
        assert len({s for s, _ in ref.values()}) >= 2          # the case is non-trivial (several statuses)
