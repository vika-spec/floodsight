"""Which buildings, roads and bridges were hit, from the fused flood/debris map + pre-event OSM.

Status values:
  likely_hit    – substantial overlap with flood/debris (medium+ confidence)
  possibly_hit  – touching or very close to the affected area, or overlap only at low confidence
  not_observed  – mostly inside radar blind spots / no data: we cannot say
  not_hit
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..detect.baseline import DEBRIS, FLOOD, NODATA


def hit_layers(cls: np.ndarray, conf: np.ndarray, res: float) -> dict:
    hit = (cls == FLOOD) | (cls == DEBRIS)
    hit_strong = hit & (conf >= 2)
    nodata = cls == NODATA

    def dist(mask):
        return ndimage.distance_transform_edt(~mask) * res if mask.any() else np.full(cls.shape, np.inf)

    return {"hit": hit, "hit_strong": hit_strong, "dist_m": dist(hit), "dist_strong_m": dist(hit_strong),
            "dist_nodata_m": dist(nodata), "nodata": nodata}


def sample_points(geom, step: float) -> np.ndarray:
    """Points every `step` metres along a (Multi)LineString in projected coordinates (vectorised)."""
    import shapely

    parts = getattr(geom, "geoms", [geom])
    out = []
    for ln in parts:
        n = max(int(ln.length // step) + 1, 2)
        pts = shapely.line_interpolate_point(ln, np.linspace(0, ln.length, n))
        out.append(shapely.get_coordinates(pts))
    return np.vstack(out)


def _lookup(arr, grid, pts):
    r, c = grid.xy_to_rowcol(pts[:, 0], pts[:, 1])
    inside = (r >= 0) & (r < grid.height) & (c >= 0) & (c < grid.width)
    out = np.zeros(len(pts), dtype=arr.dtype) if arr.dtype == bool else np.full(len(pts), np.inf)
    out[inside] = arr[r[inside], c[inside]]
    return out, inside


def assess_buildings(buildings, layers: dict, grid, cfg: dict):
    import geopandas as gpd
    from rasterio.features import rasterize

    b = buildings.to_crs(grid.crs).reset_index(drop=True)
    b = b[b.geometry.notna() & ~b.geometry.is_empty].reset_index(drop=True)
    if b.empty:
        return gpd.GeoDataFrame({"status": []}, geometry=[], crs=grid.crs)
    ids = rasterize(((g, i + 1) for i, g in enumerate(b.geometry)), out_shape=grid.shape,
                    transform=grid.transform, fill=0, all_touched=True, dtype="int32")
    n = len(b) + 1
    tot = np.bincount(ids.ravel(), minlength=n)
    hit = np.bincount(ids.ravel(), weights=layers["hit"].ravel(), minlength=n)
    strong = np.bincount(ids.ravel(), weights=layers["hit_strong"].ravel(), minlength=n)
    nod = np.bincount(ids.ravel(), weights=layers["nodata"].ravel(), minlength=n)
    import shapely
    cent = shapely.get_coordinates(shapely.centroid(b.geometry.values))     # vectorised, same points
    dist, inside = _lookup(layers["dist_m"], grid, cent)

    d = cfg["damage"]
    frac = np.where(tot[1:] > 0, hit[1:] / np.maximum(tot[1:], 1), 0.0)
    frac_strong = np.where(tot[1:] > 0, strong[1:] / np.maximum(tot[1:], 1), 0.0)
    frac_nod = np.where(tot[1:] > 0, nod[1:] / np.maximum(tot[1:], 1), 1.0)
    status = np.full(len(b), "not_hit", dtype=object)
    status[(frac > 0) | (dist <= d["building_buffer_m"])] = "possibly_hit"
    status[frac_strong >= d["building_likely_frac"]] = "likely_hit"
    status[(status == "not_hit") & (frac_nod >= 0.5)] = "not_observed"
    status[~inside] = "outside_aoi"
    b["status"] = status
    b["overlap_pct"] = np.round(frac * 100, 1)
    return b


def _bridge_flags(r) -> np.ndarray:
    if "bridge" not in r:
        return np.zeros(len(r), bool)
    v = r["bridge"].astype(object).where(r["bridge"].notna(), "no").astype(str).str.lower()
    return ~v.isin(["no", "none", "nan", ""]).to_numpy()


def assess_roads(roads, layers: dict, grid, cfg: dict):
    """Status of every road segment. Vectorised over all segments at once (large city maps have hundreds of
    thousands); gives exactly the same result as checking them one by one (tests/test_damage_vectorised.py)."""
    import shapely

    r = roads.to_crs(grid.crs).reset_index(drop=True)
    r = r[r.geometry.notna() & r.geom_type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    d = cfg["damage"]
    n = len(r)
    is_bridge = _bridge_flags(r)
    status = np.full(n, "outside_aoi", dtype=object)
    cut_m = np.zeros(n)
    cut_conf_m = np.zeros(n)                    # part of affected_m next to medium/high-confidence detections
    if n:
        geoms = r.geometry.values
        gx0, gy0, gx1, gy1 = grid.bounds
        bb = shapely.bounds(geoms)
        cand = np.flatnonzero(~((bb[:, 2] < gx0) | (bb[:, 0] > gx1) | (bb[:, 3] < gy0) | (bb[:, 1] > gy1)))
        if len(cand):
            # sample points every res/2 metres along every part of every candidate road (as sample_points)
            parts, row_of_part = shapely.get_parts(geoms[cand], return_index=True)
            row_of_part = cand[row_of_part]
            plen = shapely.length(parts)
            step = grid.res / 2
            npts = np.maximum((plen // step).astype(np.int64) + 1, 2)
            pid = np.repeat(np.arange(len(parts)), npts)
            first = np.repeat(np.cumsum(npts) - npts, npts)
            k = np.arange(len(pid)) - first
            pos = k * (plen / (npts - 1))[pid]                  # exactly np.linspace(0, len, n): k * step ...
            last = np.cumsum(npts) - 1
            pos[last] = plen                                    # ... with the end point set to len
            pts = shapely.get_coordinates(shapely.line_interpolate_point(parts[pid], pos))
            rows = row_of_part[pid]
            dist, inside = _lookup(layers["dist_m"], grid, pts)
            nod, _ = _lookup(layers["nodata"], grid, pts)
            buf = np.where(is_bridge[rows], d["bridge_buffer_m"], d["road_buffer_m"])
            near = (dist <= buf) & inside
            if "dist_strong_m" in layers:                 # same test against medium/high-confidence detections only
                dist_s, _ = _lookup(layers["dist_strong_m"], grid, pts)
                near_s = (dist_s <= buf) & inside
            else:
                near_s = near
            cnt = np.bincount(rows, minlength=n)
            n_in = np.bincount(rows, weights=inside, minlength=n)
            n_near = np.bincount(rows, weights=near, minlength=n)
            n_nod = np.bincount(rows, weights=nod & inside, minlength=n)
            n_near_s = np.bincount(rows, weights=near_s, minlength=n)
            seg_step = shapely.length(geoms) / np.maximum(cnt - 1, 1)
            seen = (cnt > 0) & (n_in > 0)
            cut_m[seen] = n_near[seen] * seg_step[seen]
            cut_conf_m[seen] = n_near_s[seen] * seg_step[seen]
            hit = seen & (n_near > 0)
            status[seen] = "not_hit"
            status[seen & ~hit & (n_nod / np.maximum(cnt, 1) > 0.5)] = "not_observed"
            status[hit] = np.where(is_bridge[hit] | (n_near[hit] >= 3), "likely_hit", "possibly_hit")
    r["status"] = status
    r["affected_m"] = np.round(cut_m, 1)
    r["affected_confident_m"] = np.round(cut_conf_m, 1)
    r["is_bridge"] = is_bridge
    return r


ROAD_RANK = {"motorway": 0, "trunk": 1, "primary": 2, "secondary": 3, "tertiary": 4, "unclassified": 5,
             "residential": 6, "service": 7, "track": 8, "path": 9, "footway": 9}


def _road_label(row) -> str:
    """Name a responder recognises: English name, local name, road number (OSM ref), else the road type."""
    def ok(v):
        return v is not None and str(v).strip() not in ("", "nan", "None")
    for k in ("name:en", "name", "ref"):
        if k in row and ok(row[k]):
            return str(row[k]).strip()
    hw = str(row.get("highway") or "road").replace("_", " ")
    return f"unnamed {hw}"


def named_blockages(roads, max_roads: int = 6, max_bridges: int = 6) -> dict:
    """Blocked roads and hit bridges for the situation report, named as on the pre-event map.

    Roads: affected segments grouped by name (or road number), most important road class first, then the
    length affected; each with one location (lat/lon of the longest affected segment). Bridges: every bridge
    marked likely hit, with its name (or the road it carries) and location. Only 'likely_hit'/'possibly_hit'
    segments inside the analysed area are used; nothing is inferred beyond the flood map."""
    out = {"roads": [], "bridges": [], "roads_total": 0, "bridges_total": 0}
    if roads is None or roads.empty or "status" not in roads:
        return out
    hit = roads[roads["status"].isin(["likely_hit", "possibly_hit"])]
    if hit.empty:
        return out
    hit = hit.to_crs(4326) if hit.crs is not None else hit

    def loc(geom):
        pt = geom.interpolate(0.5, normalized=True)
        return [round(pt.y, 4), round(pt.x, 4)]

    groups = {}
    for _, row in hit[~hit["is_bridge"]].iterrows():
        lab = _road_label(row)
        g = groups.setdefault(lab, {"name": lab, "highway": row.get("highway"), "km": 0.0, "best": None,
                                    "rank": ROAD_RANK.get(str(row.get("highway")), 8), "certain": False})
        g["km"] += float(row.get("affected_m") or 0) / 1000
        g["rank"] = min(g["rank"], ROAD_RANK.get(str(row.get("highway")), 8))
        g["certain"] |= row["status"] == "likely_hit"
        if g["best"] is None or (row.get("affected_m") or 0) > g["best"][0]:
            g["best"] = (row.get("affected_m") or 0, loc(row.geometry))
    rows = sorted(groups.values(), key=lambda g: (g["rank"], g["name"].startswith("unnamed"), -g["km"]))
    out["roads_total"] = len(rows)
    out["roads"] = [{"name": g["name"], "highway": g["highway"], "km_affected": round(g["km"], 2),
                     "certain": bool(g["certain"]), "lat_lon": g["best"][1]} for g in rows[:max_roads]]

    br = hit[hit["is_bridge"] & (hit["status"] == "likely_hit")]
    seen, blist = set(), []
    for _, row in br.iterrows():
        lab = _road_label(row)
        lab = lab if not lab.startswith("unnamed") else f"bridge on {lab}"
        ll = loc(row.geometry)
        key = (lab, round(ll[0], 3), round(ll[1], 3))
        if key in seen:
            continue
        seen.add(key)
        blist.append({"name": lab, "highway": row.get("highway"), "lat_lon": ll,
                      "rank": ROAD_RANK.get(str(row.get("highway")), 8)})
    blist.sort(key=lambda b: (b["rank"], b["name"].startswith("bridge on unnamed")))
    out["bridges_total"] = len(blist)
    out["bridges"] = [{k: v for k, v in b.items() if k != "rank"} for b in blist[:max_bridges]]
    return out


def summarise(buildings, roads) -> dict:
    def count(df, s, mask=None):
        if df is None or df.empty:
            return 0
        m = df["status"] == s
        if mask is not None:
            m &= mask
        return int(m.sum())

    # bridges counted inside the analysed area only (roads are also fetched in a buffer around it)
    br = (roads["is_bridge"] & (roads["status"] != "outside_aoi")) if (roads is not None and not roads.empty) else None
    return {
        "buildings_total": int((buildings["status"] != "outside_aoi").sum()) if not buildings.empty else 0,
        "buildings_likely_hit": count(buildings, "likely_hit"),
        "buildings_possibly_hit": count(buildings, "possibly_hit"),
        "buildings_not_observed": count(buildings, "not_observed"),
        "road_km_affected": round(float(roads.loc[roads["status"].isin(["likely_hit", "possibly_hit"]),
                                                  "affected_m"].sum()) / 1000, 2) if not roads.empty else 0.0,
        "road_segments_hit": count(roads, "likely_hit") + count(roads, "possibly_hit"),
        "bridges_total": int(br.sum()) if br is not None else 0,
        "bridges_hit": count(roads, "likely_hit", br) + count(roads, "possibly_hit", br) if br is not None else 0,
        # confirmed = next to medium/high-confidence flood or debris (two sources agree); the rest is 'possible'
        "road_km_confirmed": round(float(roads.loc[roads["status"].isin(["likely_hit", "possibly_hit"]),
                                                   "affected_confident_m"].sum()) / 1000, 2)
        if (not roads.empty and "affected_confident_m" in roads) else 0.0,
        "bridges_confirmed": int((br & roads["status"].isin(["likely_hit", "possibly_hit"])
                                  & (roads["affected_confident_m"] > 0)).sum())
        if (br is not None and "affected_confident_m" in roads) else 0,
    }
