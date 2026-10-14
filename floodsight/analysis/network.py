"""Who is cut off? Road-network reachability before vs after the flood.

Graph: OSM road ways split at their shared vertices (real junctions only; an overpass without a shared
node is not a junction). Edges touching the flood/debris map (or a hit bridge) are removed.
For each settlement in the AOI we compare the shortest road distance to the nearest hospital before and
after. Roads, hospitals and towns are fetched beyond the AOI (config: osm.network_buffer_km) so a village
is not reported cut off just because its escape route leaves the map.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree

log = logging.getLogger("floodsight")

STATUS_ORDER = ["cut_off", "likely_cut_off", "unknown", "detour", "reachable", "isolated_before", "no_road_data"]
STATUS_LABEL = {
    "cut_off": "Cut off (no road to a hospital; blocked by confident detections)",
    "likely_cut_off": "Likely cut off (only low-confidence detections block the last route)",
    "unknown": "Unknown (the only remaining route runs through roads the radar could not see)",
    "detour": "Reachable only by a long detour",
    "reachable": "Road access unchanged",
    "isolated_before": "No road to a hospital even before the flood (in OSM)",
    "no_road_data": "No mapped road nearby",
}


def _is_bridge(v) -> bool:
    return str(v or "no").lower() not in ("no", "none", "nan", "")


def build_edges(roads, grid, cfg: dict):
    """Vertex-to-vertex edges from road lines (projected). Returns arrays + per-edge road index."""
    ncfg = cfg["network"]
    allowed = set(ncfg["motorable"])
    if ncfg.get("include_paths"):
        allowed |= {"path", "footway", "steps", "bridleway", "track"}
    r = roads.to_crs(grid.crs)
    if "highway" in r:
        r = r[r["highway"].isin(allowed)]
    speeds = ncfg.get("speed_kmh", {})
    default_speed = float(speeds.get("default", 25))
    if r.empty:
        return None
    import shapely
    # Vectorised over all roads (same edges, same order as walking them one by one; tests/test_damage_vectorised.py)
    parts, row = shapely.get_parts(r.geometry.values, return_index=True)
    keep = shapely.get_type_id(parts) == 1                               # LineString parts only
    parts, row = parts[keep], row[keep]
    if not len(parts):
        return None
    xy, pidx = shapely.get_coordinates(parts, return_index=True)
    same = pidx[:-1] == pidx[1:]                                         # consecutive vertices of one part
    a, b = xy[:-1][same], xy[1:][same]
    rows = row[pidx[:-1][same]]
    labels = r.index.to_numpy()
    hw = r["highway"].astype(str).to_numpy() if "highway" in r else np.full(len(r), "None")
    v = np.array([float(speeds.get(h, default_speed)) for h in hw]) if len(r) else np.zeros(0)
    br = np.array([_is_bridge(x) for x in (r["bridge"] if "bridge" in r else [None] * len(r))], bool)
    return {"a": a, "b": b, "road": labels[rows], "bridge": br[rows], "kmh": v[rows]}


def edge_damage(edges: dict, dist_m: np.ndarray, grid, cfg: dict, buffers: tuple | None = None) -> np.ndarray:
    """An edge is broken if any point along it lies within the damage buffer of flood/debris.
    buffers=(road_m, bridge_m) overrides the configured buffers (0, 0 = the point itself is inside)."""
    d = cfg["damage"]
    A, B = edges["a"], edges["b"]
    L = np.hypot(*(B - A).T)
    step = grid.res / 2
    n = np.maximum(np.ceil(L / step).astype(int) + 1, 2)
    eid = np.repeat(np.arange(len(A)), n)
    start = np.repeat(np.cumsum(n) - n, n)
    t = (np.arange(len(eid)) - start) / (n[eid] - 1)          # 0..1 along each edge (vectorised linspace)
    pts = A[eid] + (B[eid] - A[eid]) * t[:, None]
    r, c = grid.xy_to_rowcol(pts[:, 0], pts[:, 1])
    inside = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    dist = np.full(len(pts), np.inf)
    dist[inside] = dist_m[r[inside], c[inside]]
    rb, bb = buffers if buffers is not None else (d["road_buffer_m"], d["bridge_buffer_m"])
    buf = np.where(edges["bridge"][eid], bb, rb)
    hitpt = dist <= buf
    broken = np.zeros(len(A), bool)
    np.logical_or.at(broken, eid, hitpt)
    return broken


def build_graph(edges: dict, broken: np.ndarray | None = None, tol_m: float = 1.0) -> tuple[nx.Graph, dict]:
    """Vertices closer than tol_m are merged into one node (absorbs floating-point noise at shared OSM
    nodes after reprojection; real distinct junctions are never 1 m apart)."""
    pts = np.vstack([edges["a"], edges["b"]])
    uniq, inv = np.unique(np.round(pts, 3), axis=0, return_inverse=True)
    parent = np.arange(len(uniq))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j in cKDTree(uniq).query_pairs(tol_m):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(len(uniq))])
    node = root[inv.ravel()]
    n = len(edges["a"])
    ua, ub = node[:n], node[n:]

    G = nx.Graph()
    L = np.hypot(*(edges["b"] - edges["a"]).T)
    for i in range(n):
        if broken is not None and broken[i]:
            continue
        if ua[i] != ub[i]:
            G.add_edge(int(ua[i]), int(ub[i]), length=float(L[i]), eid=i)
    return G, {"coords": uniq, "ua": ua, "ub": ub}


def merge_nodes(edges: dict, tol_m: float = 1.0):
    """Same vertex merging as build_graph, vectorised: vertices closer than tol_m become one node.
    Returns (coordinates of each node id, node id of each edge end a, node id of each edge end b, n_nodes)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    pts = np.vstack([edges["a"], edges["b"]])
    uniq, inv = np.unique(np.round(pts, 3), axis=0, return_inverse=True)
    pairs = cKDTree(uniq).query_pairs(tol_m, output_type="ndarray")
    m = len(uniq)
    if len(pairs):
        adj = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        _, label = connected_components(adj, directed=False)
    else:
        label = np.arange(m)
    node = label[inv.ravel()]
    n = len(edges["a"])
    n_nodes = int(label.max()) + 1 if m else 0
    # coordinates PER NODE ID (labels are renumbered, so uniq[i] is NOT the position of node i)
    coords = np.zeros((n_nodes, 2))
    coords[label] = uniq
    return coords, node[:n], node[n:], n_nodes


def road_matrix(ua, ub, length, keep, n_nodes):
    """Sparse undirected road graph; parallel edges keep their shortest length (as networkx does when an
    edge is re-added, the last one wins there; the shortest is the physically right choice and both give
    the same shortest-path distances)."""
    from scipy.sparse import csr_matrix

    ok = keep & (ua != ub)
    u, v, w = ua[ok], ub[ok], length[ok]
    lo, hi = np.minimum(u, v), np.maximum(u, v)
    order = np.lexsort((w, hi, lo))
    lo, hi, w = lo[order], hi[order], w[order]
    first = np.ones(len(lo), bool)
    first[1:] = (lo[1:] != lo[:-1]) | (hi[1:] != hi[:-1])
    lo, hi, w = lo[first], hi[first], np.maximum(w[first], 1e-6)   # zero-length edges must stay edges
    M = csr_matrix((np.r_[w, w], (np.r_[lo, hi], np.r_[hi, lo])), shape=(n_nodes, n_nodes))
    deg = np.bincount(np.r_[lo, hi], minlength=n_nodes)
    return M, deg > 0


def nearest_target(M, present, targets, with_predecessors: bool = False):
    """Road distance from every node to its nearest target node, and which target (one compiled
    multi-source Dijkstra instead of one search per hospital). with_predecessors=True also returns the
    predecessor of every node on its shortest path (to draw the route)."""
    from scipy.sparse.csgraph import dijkstra

    n = M.shape[0]
    dist = np.full(n, np.inf)
    src = np.full(n, -1)
    pred = np.full(n, -9999)
    nodes = sorted({t for t, _ in targets if present[t]})
    if not nodes:
        return (dist, src, pred) if with_predecessors else (dist, src)
    d, pr, s = dijkstra(M, directed=False, indices=nodes, min_only=True, return_predecessors=True)
    dist, src, pred = d, s, pr
    dist[~present] = np.inf                       # a node with no road left is not reachable
    return (dist, src, pred) if with_predecessors else (dist, src)


def _first_hit(a, b, dist_m, grid, cfg):
    """Point along segment a->b closest to the start that lies within the road damage buffer (the block)."""
    L = float(np.hypot(*(b - a)))
    n = max(int(L / (grid.res / 2)) + 1, 2)
    pts = a + (b - a) * np.linspace(0, 1, n)[:, None]
    r, c = grid.xy_to_rowcol(pts[:, 0], pts[:, 1])
    ok = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    d = np.full(n, np.inf)
    d[ok] = dist_m[r[ok], c[ok]]
    hit = np.flatnonzero(d <= cfg["damage"]["road_buffer_m"])
    return pts[hit[0]] if len(hit) else (a + b) / 2


def trace_route(pred, node, max_steps: int = 200000) -> list[int]:
    """Node ids from a settlement's node to its nearest target, following Dijkstra predecessors."""
    out = [int(node)]
    while pred[out[-1]] >= 0 and len(out) < max_steps:
        out.append(int(pred[out[-1]]))
    return out


def _points(gdf, grid):
    g = gdf.to_crs(grid.crs)
    return np.array([(geom.centroid.x, geom.centroid.y) for geom in g.geometry]), g


def _settlement_buildings(buildings, spts, grid, radius_m: float) -> np.ndarray:
    """Buildings whose nearest settlement point is this one (within radius_m): a size / population proxy."""
    if buildings is None or len(buildings) == 0 or len(spts) == 0:
        return np.zeros(len(spts), int)
    b = buildings.to_crs(grid.crs)
    b = b[b.geometry.notna() & ~b.geometry.is_empty]
    if b.empty:
        return np.zeros(len(spts), int)
    import shapely
    c = shapely.get_coordinates(b.geometry.centroid)
    d, i = cKDTree(spts).query(c, distance_upper_bound=radius_m)
    ok = np.isfinite(d)
    return np.bincount(i[ok], minlength=len(spts)).astype(int)


def analyse_access(roads, places, health, layers: dict, grid, cfg: dict, notes: list | None = None,
                   buildings=None) -> dict:
    """Statuses, from most to least severe (see STATUS_LABEL):
    cut_off         no route even if only high/medium-confidence detections close roads
    likely_cut_off  a route exists unless low-confidence detections also close roads
    unknown         a route exists, but only through roads the radar could not observe (blind spots)
    detour / reachable / isolated_before / no_road_data as before.
    Travel times use nominal free-flow speeds per OSM road class (config network.speed_kmh)."""
    import geopandas as gpd
    from shapely.geometry import LineString

    ncfg = cfg["network"]
    notes = notes if notes is not None else []
    empty = {"settlements": gpd.GeoDataFrame({"status": []}, geometry=[], crs=grid.crs),
             "broken_roads": gpd.GeoDataFrame(geometry=[], crs=grid.crs), "summary": {}}
    if places is None or places.empty or roads is None or roads.empty:
        notes.append("No settlements or roads in OpenStreetMap for this area; cut-off analysis not possible.")
        return empty
    edges = build_edges(roads, grid, cfg)
    if edges is None:
        notes.append("No motorable roads in OpenStreetMap for this area.")
        return empty

    broken = edge_damage(edges, layers["dist_m"], grid, cfg)
    # confident closures only (high/medium confidence); without that layer every closure counts as confident
    broken_strong = edge_damage(edges, layers["dist_strong_m"], grid, cfg) if "dist_strong_m" in layers else broken
    # roads crossing radar blind spots: state unknown (point inside a no-data pixel)
    unseen = (edge_damage(edges, layers["dist_nodata_m"], grid, cfg, buffers=(0.0, 0.0))
              if "dist_nodata_m" in layers else np.zeros(len(broken), bool))
    # all graphs use identical node ids (same vertex merging); post graphs simply lack some edges
    coords, ua, ub, n_nodes = merge_nodes(edges)
    L = np.hypot(*(edges["b"] - edges["a"]).T)
    T = L / (edges["kmh"] / 3.6)                      # seconds
    keep_all = np.ones(len(L), bool)
    M_pre, present_pre = road_matrix(ua, ub, L, keep_all, n_nodes)
    M_post, present_post = road_matrix(ua, ub, L, ~broken, n_nodes)
    M_strong, present_strong = road_matrix(ua, ub, L, ~broken_strong, n_nodes)
    M_pess, present_pess = road_matrix(ua, ub, L, ~(broken | unseen), n_nodes)
    Mt_pre, _ = road_matrix(ua, ub, T, keep_all, n_nodes)
    Mt_post, _ = road_matrix(ua, ub, T, ~broken, n_nodes)
    in_graph = np.flatnonzero(present_pre)
    tree = cKDTree(coords[in_graph])

    def snap(pts):
        d, i = tree.query(pts)
        return d, in_graph[i]

    # targets: hospitals first; towns/cities as a fallback
    target_kind = "hospital"
    hosp = health
    if hosp is not None and not hosp.empty:
        is_h = np.zeros(len(hosp), bool)
        for col in ("amenity", "healthcare"):
            if col in hosp:
                is_h |= hosp[col].astype(str).str.lower().eq("hospital").to_numpy()
        hosp = hosp[is_h]
    if hosp is None or hosp.empty:
        hosp = places[places["place"].isin(["town", "city"])] if "place" in places else places.iloc[0:0]
        target_kind = "town"
        notes.append("No hospital mapped in OpenStreetMap nearby; nearest town/city used as the target instead.")
    if hosp.empty:
        notes.append("No hospital or town in OpenStreetMap within reach; cut-off analysis not possible.")
        return empty

    tpts, tg = _points(hosp, grid)
    td, tn = snap(tpts)
    tnames = [str(n) if n and str(n) != "nan" else f"{target_kind} #{i + 1}"
              for i, n in enumerate(tg.get("name", [None] * len(tg)))]
    targets = [(int(n), tnames[i]) for i, (n, d) in enumerate(zip(tn, td)) if d <= ncfg["snap_max_m"]]

    node_name = {}
    for node, nm in targets:                     # several hospitals on one junction: keep the first name
        node_name.setdefault(node, nm)
    d_pre, s_pre, p_pre = nearest_target(M_pre, present_pre, targets, with_predecessors=True)
    d_post, s_post, p_post = nearest_target(M_post, present_post, targets, with_predecessors=True)
    d_strong, _ = nearest_target(M_strong, present_strong, targets)
    d_pess, _ = nearest_target(M_pess, present_pess, targets)
    t_pre, _ = nearest_target(Mt_pre, present_pre, targets)
    t_post, _ = nearest_target(Mt_post, present_post, targets)
    pre_d = {i: d_pre[i] for i in np.flatnonzero(np.isfinite(d_pre))}
    post_d = {i: d_post[i] for i in np.flatnonzero(np.isfinite(d_post))}
    pre_n = {i: node_name.get(int(s_pre[i]), "") for i in pre_d}
    post_n = {i: node_name.get(int(s_post[i]), "") for i in post_d}

    # settlements inside the AOI only
    spts, sg = _points(places, grid)
    r, c = grid.xy_to_rowcol(spts[:, 0], spts[:, 1])
    in_aoi = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    sg = sg[in_aoi].reset_index(drop=True)
    spts = spts[in_aoi]
    if sg.empty:
        notes.append("No mapped settlements inside the area.")
        return empty
    sd, sn = snap(spts)
    nb = _settlement_buildings(buildings, spts, grid, float(ncfg.get("settlement_radius_m", 1500)))

    def minutes(sec):
        return round(float(sec) / 60) if np.isfinite(sec) else None

    rows = []
    for i in range(len(sg)):
        node, snap = int(sn[i]), float(sd[i])
        dp, dq = pre_d.get(node, np.inf), post_d.get(node, np.inf)
        ds, dz = d_strong[node], d_pess[node]
        if snap > ncfg["snap_max_m"]:
            st = "no_road_data"
        elif not np.isfinite(dp):
            st = "isolated_before"
        elif not np.isfinite(ds):
            st = "cut_off"
        elif not np.isfinite(dq):
            st = "likely_cut_off"
        elif not np.isfinite(dz):
            st = "unknown"
        elif dq > dp * ncfg["detour_ratio"] and (dq - dp) / 1000 >= ncfg["detour_min_extra_km"]:
            st = "detour"
        else:
            st = "reachable"
        tb, ta = t_pre[node], t_post[node]
        rows.append({
            "name": str(sg.loc[i].get("name")) if str(sg.loc[i].get("name")) != "nan" else "(unnamed)",
            "name_en": str(sg.loc[i].get("name:en")) if str(sg.loc[i].get("name:en")) not in ("nan", "None") else "",
            "place": sg.loc[i].get("place", ""), "status": st,
            "target_before": pre_n.get(node, ""), "target_after": post_n.get(node, ""),
            "km_before": round(dp / 1000, 1) if np.isfinite(dp) else None,
            "km_after": round(dq / 1000, 1) if np.isfinite(dq) else None,
            "extra_km": round((dq - dp) / 1000, 1) if np.isfinite(dp) and np.isfinite(dq) else None,
            "min_before": minutes(tb), "min_after": minutes(ta),
            "extra_min": minutes(ta - tb) if np.isfinite(tb) and np.isfinite(ta) else None,
            "buildings": int(nb[i]),
            "snap_m": round(snap), "_node": node,
        })
    settle = gpd.GeoDataFrame(rows, geometry=list(sg.geometry.centroid), crs=grid.crs)
    settle["_o"] = settle["status"].map({s: i for i, s in enumerate(STATUS_ORDER)})
    settle["_x"] = settle["extra_min"].fillna(0)
    # response priority: most severe status first, then the largest settlement, then the longest delay
    settle = settle.sort_values(["_o", "buildings", "_x", "km_before"], ascending=[True, False, False, True])
    settle = settle.drop(columns=["_o", "_x"]).reset_index(drop=True)
    needs = settle["status"].isin(["cut_off", "likely_cut_off", "unknown", "detour"])
    settle["priority"] = np.where(needs, np.cumsum(needs), 0).astype(int)

    # routes for the settlements that need attention: usual route before the flood, route now (if any), and
    # the points where the usual route is blocked
    from shapely.geometry import Point
    key = {}
    for e in range(len(ua)):
        k = (min(ua[e], ub[e]), max(ua[e], ub[e]))
        key[k] = key.get(k, False) or bool(broken[e])
    r_rows, r_geom, b_rows, b_geom = [], [], [], []
    for i in range(len(settle)):
        st = settle.loc[i, "status"]
        if st not in ("cut_off", "likely_cut_off", "unknown", "detour"):
            continue
        nm = settle.loc[i, "name"]
        node = int(settle.loc[i, "_node"])
        before = trace_route(p_pre, node) if np.isfinite(d_pre[node]) else []
        if len(before) > 1:
            r_rows.append({"name": nm, "route": "before", "status": st})
            r_geom.append(LineString(coords[before]))
            for a_, b_ in zip(before[:-1], before[1:]):
                if key.get((min(a_, b_), max(a_, b_))):
                    b_rows.append({"name": nm})
                    b_geom.append(Point(_first_hit(coords[a_], coords[b_], layers["dist_m"], grid, cfg)))
        after = trace_route(p_post, node) if np.isfinite(d_post[node]) else []
        if len(after) > 1:
            r_rows.append({"name": nm, "route": "now", "status": st})
            r_geom.append(LineString(coords[after]))
    settle = settle.drop(columns="_node")
    routes = gpd.GeoDataFrame(r_rows, geometry=r_geom, crs=grid.crs) if r_rows else \
        gpd.GeoDataFrame({"name": [], "route": [], "status": []}, geometry=[], crs=grid.crs)
    blocks = gpd.GeoDataFrame(b_rows, geometry=b_geom, crs=grid.crs) if b_rows else \
        gpd.GeoDataFrame({"name": []}, geometry=[], crs=grid.crs)

    bl = [LineString([edges["a"][i], edges["b"][i]]) for i in np.flatnonzero(broken)]
    broken_gdf = gpd.GeoDataFrame({"bridge": edges["bridge"][broken]}, geometry=bl, crs=grid.crs)

    counts = settle["status"].value_counts().to_dict()
    summary = {f"settlements_{s}": int(counts.get(s, 0)) for s in STATUS_ORDER}
    summary["settlements_total"] = int(len(settle))
    summary["target_kind"] = target_kind
    summary["targets_used"] = len(targets)
    summary["broken_road_edges"] = int(broken.sum())
    summary["broken_road_edges_confident"] = int(broken_strong.sum())
    summary["unseen_road_edges"] = int(unseen.sum())
    summary["buildings_in_cut_off_settlements"] = int(settle.loc[settle["status"] == "cut_off", "buildings"].sum())
    summary["buildings_in_likely_cut_off_settlements"] = int(
        settle.loc[settle["status"] == "likely_cut_off", "buildings"].sum())
    summary["speeds"] = "nominal free-flow speeds per OSM road class (OSRM car-profile defaults)"
    return {"settlements": settle, "broken_roads": broken_gdf, "summary": summary, "routes": routes,
            "route_blocks": blocks}
