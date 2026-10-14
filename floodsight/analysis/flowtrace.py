"""Bonus: from ANY point upstream, trace the flood path down the valley and list what lies along it.

    python -m floodsight.analysis.flowtrace --lon 85.36 --lat 28.27 --km 80 [--date 2026-08-26] [--run runs/x]

Not limited to an analysed box: the path is traced window by window on the Copernicus DEM GLO-30 at its
native 30 m (steepest descent, D8, after filling pits and depressions, as in the main pipeline). Each window
is about 30 km wide; the path is only trusted until it comes within a few km of the window edge (edge
effects of depression filling). The next window is placed AHEAD of the path's last point, in the direction
the path was heading, so each window adds up to ~20 km instead of ~12 (the last point always stays well
inside the trusted part of the new window, whatever way the river turns). It stops at the length limit, at the
sea / a closed basin, or where the DEM has no data.

Along the path (OpenStreetMap as it was BEFORE the event, ohsome API, same snapshot rule as the pipeline):
  * settlements, hospitals/clinics and bridges near the river, with km downstream, distance from the river
    and height above it;
  * height bands above the river: scenario levels for how high a flood or debris surge reaches
    (<= 10 m: valley floor, <= 25 m: low terraces, <= 50 m: high terraces in gorges). These are exposure
    bands, not a prediction of this event's flood depth;
  * if a finished analysis (run) is given: whether flood water / debris was detected near each place;
  * a check of the path itself, independent of the DEM: which parts of it lie within 150 m of a river
    mapped in OpenStreetMap, and which named rivers it follows (km from-to). Shown on the map: solid where
    it matches a mapped river, dashed where it does not.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import date
from pathlib import Path

import numpy as np

log = logging.getLogger("floodsight")


def _ahead(lonlat_pts, km_back: float, shift_km: float):
    """Centre for the next window: shift_km beyond the last point, along the direction from the point
    ~km_back earlier to the last point (local flat-earth approximation, fine at these distances)."""
    lon1, lat1 = lonlat_pts[-1]
    kx = 111.32 * max(math.cos(math.radians(lat1)), 0.2)
    pts = np.asarray(lonlat_pts[-400:], float)
    dx = (lon1 - pts[:, 0]) * kx
    dy = (lat1 - pts[:, 1]) * 110.57
    d = np.hypot(dx, dy)
    far = np.nonzero(d >= km_back)[0]
    j = far[-1] if len(far) else 0
    if d[j] <= 0:
        return lon1, lat1
    ux, uy = dx[j] / d[j], dy[j] / d[j]
    return lon1 + ux * shift_km / kx, lat1 + uy * shift_km / 110.57


def _window(lon, lat, km):
    from ..core import AOI
    dlat = km / 2 / 110.57
    dlon = km / 2 / (111.32 * max(math.cos(math.radians(lat)), 0.2))
    return AOI(lon - dlon, lat - dlat, lon + dlon, lat + dlat)


def trace_path(lon: float, lat: float, cfg: dict, max_km: float, progress=None, dem_reader=None) -> dict:
    """Window-by-window D8 trace. Returns lon/lat path, elevation, km along, windows and the stop reason."""
    from scipy import ndimage

    from ..core import Grid
    from ..fetch.dem import dem_on_grid
    from ..preprocess.terrain import D8_OFFSETS, flow_layers

    fc = cfg["flowpath"]
    res = float(fc.get("dem_res_m", 30))
    win_km = float(fc.get("window_km", 30))
    margin_m = float(fc.get("edge_margin_km", 3)) * 1000
    snap_m = float(fc.get("snap_m", 300))
    read = dem_reader or (lambda grid, aoi: dem_on_grid(grid, aoi.as_list()))

    pts_ll, elev, windows = [], [], []
    cur_lon, cur_lat = lon, lat
    cen_lon, cen_lat = lon, lat
    # the new window's centre may lie this far ahead of the last point; the point then stays
    # (win/2 - shift) km from any edge, i.e. 1 km inside the trusted part
    shift_km = max(win_km / 2 - margin_m / 1000 - 1.0, 0.0)
    stop = "length limit reached"
    km_done, first = 0.0, True
    max_windows = int(max_km / max(win_km / 2 - margin_m / 1000, 1)) + 3     # worst case (path turns back)
    for w in range(max_windows):
        aoi = _window(cen_lon, cen_lat, win_km)
        grid = Grid.for_aoi(aoi, res)
        if progress:
            progress(f"Tracing the path: window {w + 1}", min(0.1 + 0.6 * km_done / max_km, 0.7))
        dem = read(grid, aoi)
        if not np.isfinite(dem).any():
            stop = "no elevation data here (sea or outside the DEM)" if not first else "no elevation data at the start point"
            break
        hy = flow_layers(dem, grid)
        fdir, acc = hy["fdir"], hy["acc"]
        x, y = grid.lonlat_to_xy(cur_lon, cur_lat)
        r, c = grid.xy_to_rowcol(np.array([x]), np.array([y]))
        r, c = int(r[0]), int(c[0])
        H, W = fdir.shape
        if first:                                     # a click near the river lands on the river
            k = int(snap_m / res)
            rs, cs = max(r - k, 0), max(c - k, 0)
            win = acc[rs:r + k + 1, cs:c + k + 1]
            rr, cc = np.unravel_index(np.nanargmax(win), win.shape)
            r, c = rs + rr, cs + cc
        edge_px = int(margin_m / res)
        path = [(r, c)]
        seen = {(r, c)}
        reached_edge = False
        while True:
            off = D8_OFFSETS.get(int(fdir[r, c]))
            if off is None:
                stop = "the path ends in a closed basin, lake or the sea"
                break
            r, c = r + off[0], c + off[1]
            if not (0 <= r < H and 0 <= c < W):
                reached_edge = True
                break
            if (r, c) in seen:                         # cannot happen after depression filling; never loop
                stop = "the path ends in a flat area"
                break
            if not np.isfinite(dem[r, c]):              # left the elevation data (sea, or a demo area's edge)
                stop = "no elevation data beyond this point (sea or edge of the data)"
                break
            path.append((r, c))
            seen.add((r, c))
            if min(r, c, H - 1 - r, W - 1 - c) < edge_px:   # near the edge: trust it no further
                reached_edge = True
                break
        pr = np.array([p[0] for p in path])
        pc = np.array([p[1] for p in path])
        xs, ys = grid.rowcol_to_xy(pr, pc)
        from pyproj import Transformer
        to_ll = Transformer.from_crs(grid.crs, 4326, always_xy=True)
        lons, lats = to_ll.transform(xs, ys)
        z = np.nan_to_num(dem, nan=float(np.nanmean(dem)))[pr, pc]
        # distances of the newly kept points (later windows start at the previous window's last point)
        step = np.hypot(np.diff(xs), np.diff(ys)) / 1000
        if first:
            d_new, keep_from = np.concatenate([[0.0], step]), 0
        else:
            d_new, keep_from = step, 1
        cum = km_done + np.cumsum(d_new)
        n = len(cum)
        if n and cum[-1] > max_km:
            n = max(int(np.searchsorted(cum, max_km, side="right")), 1)
            reached_edge = False
            stop = "length limit reached"
        if n == 0:
            break
        pts_ll.extend(zip(lons[keep_from:keep_from + n], lats[keep_from:keep_from + n]))
        elev.extend(list(z[keep_from:keep_from + n]))
        km_done = float(cum[n - 1])
        windows.append({"grid": grid, "dem": dem, "bounds": aoi.as_list()})
        first = False
        if not reached_edge:
            break
        cur_lon, cur_lat = pts_ll[-1]
        cen_lon, cen_lat = _ahead(pts_ll, 2.0, shift_km)
    return {"lonlat": np.array(pts_ll), "elev": np.array(elev, float), "windows": windows, "stop_reason": stop}


def _project(lonlat: np.ndarray, epsg: int):
    from pyproj import Transformer
    t = Transformer.from_crs(4326, epsg, always_xy=True)
    x, y = t.transform(lonlat[:, 0], lonlat[:, 1])
    return np.column_stack([x, y])


def _elev_at(lon, lat, windows) -> float | None:
    for w in windows:
        g = w["grid"]
        x, y = g.lonlat_to_xy(lon, lat)
        r, c = g.xy_to_rowcol(np.array([x]), np.array([y]))
        r, c = int(r[0]), int(c[0])
        if 0 <= r < g.height and 0 <= c < g.width and np.isfinite(w["dem"][r, c]):
            return float(w["dem"][r, c])
    return None


def flow_trace(lon: float, lat: float, cfg: dict, max_km: float | None = None, flood_date: date | None = None,
               run_dir: Path | None = None, progress=None, dem_reader=None, osm_reader=None) -> dict:
    """Full bonus answer for one start point. All outputs in WGS84 (lon/lat) for the dashboard."""
    from scipy.spatial import cKDTree

    from ..core import AOI, osm_snapshot_date

    t0 = time.perf_counter()
    fc = cfg["flowpath"]
    max_km = float(max_km or fc.get("max_length_km", 80))
    tr = trace_path(lon, lat, cfg, max_km, progress, dem_reader)
    ll = tr["lonlat"]
    if len(ll) < 2:
        raise ValueError("No downhill path could be traced from this point (flat ground or no elevation data).")
    epsg = AOI(lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01).utm_epsg()
    xy = _project(ll, epsg)
    along = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]) / 1000
    tree = cKDTree(xy)

    # pre-event OpenStreetMap along the path
    if progress:
        progress("Reading OpenStreetMap (before the event) along the path", 0.75)
    snap = osm_snapshot_date(flood_date or date.today(), cfg["osm"]["latest_snapshot"])
    pad = float(fc.get("max_distance_m", 1500)) / 1000 + 0.5
    bbox = AOI(float(ll[:, 0].min()), float(ll[:, 1].min()), float(ll[:, 0].max()), float(ll[:, 1].max())).buffered(pad)

    def read_osm(key):
        if osm_reader is not None:
            return osm_reader(key, bbox)
        from ..fetch.osm import fetch_layer
        try:
            return fetch_layer(key, bbox, snap, cfg["osm"])[0]
        except Exception as e:  # noqa: BLE001 - the path itself is still useful
            log.warning("OSM %s along the path unavailable: %s", key, str(e)[:120])
            return None

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(4) as ex:
        layers = dict(zip(("places", "health", "bridges", "rivers"),
                          ex.map(read_osm, ("places", "health", "bridges", "rivers"))))

    bands = fc.get("height_bands_m", [10, 25, 50])
    list_max_h = float(fc.get("list_max_height_m", 150))   # listed even above the top band, marked high ground
    max_d = float(fc.get("max_distance_m", 1500))
    damage = _damage_lookup(run_dir) if run_dir else None

    def along_path(gdf, kind):
        out = []
        if gdf is None or len(gdf) == 0:
            return out
        g = gdf.to_crs(4326)
        for _, f in g.iterrows():
            p = f.geometry.representative_point() if kind != "bridge" else f.geometry.interpolate(0.5, normalized=True)
            pxy = _project(np.array([[p.x, p.y]]), epsg)[0]
            d, i = tree.query(pxy)
            if d > max_d:
                continue
            z = _elev_at(p.x, p.y, tr["windows"])
            rel = None if z is None else z - float(tr["elev"][i])
            if kind != "bridge" and (rel is None or rel > list_max_h):
                continue
            band = next((f"<= {b} m" for b in bands if rel is not None and rel <= b),
                        f"> {bands[-1]} m (high ground)" if rel is not None else None)
            rec = {"name": str(f.get("name") or f.get("name:en") or "(unnamed)"),
                   "name_en": str(f.get("name:en") or "") if str(f.get("name:en")) not in ("None", "nan") else "",
                   "km_downstream": round(float(along[i]), 1), "m_from_river": round(float(d)),
                   "m_above_river": None if rel is None else round(rel), "band": band,
                   "lon": round(p.x, 5), "lat": round(p.y, 5)}
            if kind == "settlement":
                rec["place"] = str(f.get("place") or "")
            if kind == "bridge":
                rec["road"] = str(f.get("highway") or "")
            if damage is not None:
                rec["damage_nearby"] = damage(p.x, p.y)
            out.append(rec)
        return sorted(out, key=lambda r: r["km_downstream"])

    settlements = along_path(layers["places"], "settlement")
    facilities = along_path(layers["health"], "facility")
    bridges = [b for b in along_path(layers["bridges"], "bridge") if b["m_from_river"] <= 150]

    # does the DEM path follow the river people mapped?
    river_share, followed, near = None, [], None
    rv = layers["rivers"]
    if rv is not None and len(rv):
        import shapely
        rvp = rv.to_crs(epsg)
        lines = list(rvp.geometry)
        pts = shapely.points(xy)
        near = np.zeros(len(xy), bool)
        tree_l = shapely.STRtree(lines)
        idx = tree_l.query(pts, predicate="dwithin", distance=150)
        near[np.unique(idx[0])] = True
        river_share = round(float(near.mean()), 3)
        followed = _rivers_followed(rvp, tree_l, pts, near, along)

    corridor = _band_polygons(tr, bands, max_d)
    prof_i = np.unique(np.linspace(0, len(along) - 1, min(len(along), 400)).astype(int))
    out = {
        "start": [lon, lat], "length_km": round(float(along[-1]), 1), "stop_reason": tr["stop_reason"],
        "drop_m": round(float(tr["elev"][0] - tr["elev"][-1])),
        "path": {"type": "Feature", "properties": {"length_km": round(float(along[-1]), 1)},
                 "geometry": {"type": "LineString", "coordinates": [[round(a, 5), round(b, 5)] for a, b in ll]}},
        "profile": {"km": [round(float(v), 2) for v in along[prof_i]],
                    "elev_m": [round(float(v)) for v in tr["elev"][prof_i]]},
        "settlements": settlements, "facilities": facilities, "bridges": bridges,
        "corridor": corridor, "height_bands_m": bands, "max_distance_m": max_d,
        "path_on_mapped_river": river_share, "rivers_followed": followed,
        "path_check": _check_segments(ll, near, along), "osm_snapshot": snap.isoformat(),
        "windows": len(tr["windows"]), "dem": "Copernicus DEM GLO-30 (30 m)",
        "seconds": round(time.perf_counter() - t0, 1),
        "note": ("Steepest-descent path on the elevation model. Height bands are exposure levels above the river, "
                 "not a flood-depth prediction. A dam break or debris flow can spread wider or deflect at bends."),
    }
    return out


def _name(v) -> str:
    v = "" if v is None else str(v)
    return "" if v in ("nan", "None") else v


def _rivers_followed(rvp, tree_l, pts, near, along, min_km: float = 1.0) -> list[dict]:
    """Named OpenStreetMap rivers the path runs along, in order, with km from-to (stretches < min_km dropped)."""
    nearest = tree_l.query_nearest(pts, return_distance=False)       # (2, k): point index, line index
    line_of = np.full(len(pts), -1)
    line_of[nearest[0]] = nearest[1]
    names = []
    for i in range(len(pts)):
        if not near[i] or line_of[i] < 0:
            names.append("")
            continue
        row = rvp.iloc[int(line_of[i])]
        names.append(_name(row.get("name:en")) or _name(row.get("name")) or "unnamed river")
    out = []
    for i, nm in enumerate(names):
        if not nm:
            continue
        if out and out[-1]["name"] == nm and along[i] - out[-1]["to_km"] < 2.0:
            out[-1]["to_km"] = round(float(along[i]), 1)
        else:
            out.append({"name": nm, "from_km": round(float(along[i]), 1), "to_km": round(float(along[i]), 1)})
    return [r for r in out if r["to_km"] - r["from_km"] >= min_km]


def _check_segments(ll, near, along) -> dict:
    """The path split into stretches that do / do not lie within 150 m of a mapped river (for the map)."""
    feats = []
    if near is None:
        return {"type": "FeatureCollection", "features": feats}
    start = 0
    for i in range(1, len(ll) + 1):
        if i == len(ll) or near[i] != near[start]:
            seg = ll[max(start - 1, 0):i]
            if len(seg) >= 2:
                feats.append({"type": "Feature",
                              "properties": {"on_mapped_river": bool(near[start]),
                                             "from_km": round(float(along[start]), 1),
                                             "to_km": round(float(along[i - 1]), 1)},
                              "geometry": {"type": "LineString",
                                           "coordinates": [[round(a, 5), round(b, 5)] for a, b in seg]}})
            start = i
    return {"type": "FeatureCollection", "features": feats}


def _band_polygons(tr, bands, max_d) -> dict:
    """Valley-floor bands along the path: cells within max_d of the path and <= band m above it."""
    from rasterio.features import shapes
    from scipy import ndimage
    from pyproj import Transformer
    from shapely.geometry import mapping, shape
    from shapely.ops import transform, unary_union

    feats = {b: [] for b in bands}
    ll = tr["lonlat"]
    for w in tr["windows"]:
        g, dem = w["grid"], w["dem"]
        x, y = g.lonlat_to_xy(ll[:, 0], ll[:, 1])
        r, c = g.xy_to_rowcol(np.asarray(x), np.asarray(y))
        ok = (r >= 0) & (r < g.height) & (c >= 0) & (c < g.width)
        if not ok.any():
            continue
        on = np.zeros(g.shape, bool)
        on[r[ok], c[ok]] = True
        dist, (ir, ic) = ndimage.distance_transform_edt(~on, return_indices=True)
        dist *= g.res
        z = np.nan_to_num(dem, nan=float(np.nanmean(dem)))
        rel = z - z[ir, ic]
        to_ll = Transformer.from_crs(g.crs, 4326, always_xy=True).transform
        for b in bands:
            m = (dist <= max_d) & (rel <= b) & np.isfinite(dem)
            m = ndimage.binary_opening(m, iterations=1)
            for geom, v in shapes(m.astype(np.uint8), mask=m, transform=g.transform):
                if v == 1:
                    feats[b].append(transform(to_ll, shape(geom)))
    out = []
    for b in sorted(bands, reverse=True):                 # widest first so narrower bands draw on top
        if feats[b]:
            u = unary_union(feats[b]).simplify(0.0002)
            out.append({"type": "Feature", "properties": {"band_m": b}, "geometry": mapping(u)})
    return {"type": "FeatureCollection", "features": out}


def _damage_lookup(run_dir: Path):
    """Was flood water / debris detected within 200 m of a point in a finished run? 'not visible' = radar blind."""
    import rasterio

    from ..core import Grid
    from ..detect.baseline import DEBRIS, FLOOD, NODATA

    f = Path(run_dir) / "layers" / "flood_class.tif"
    if not f.exists():
        return None
    R = json.loads((Path(run_dir) / "results.json").read_text())
    g = Grid.from_dict(R["grid"])
    with rasterio.open(f) as src:
        cls = src.read(1)
    from scipy import ndimage

    hit = (cls == FLOOD) | (cls == DEBRIS)
    dist = ndimage.distance_transform_edt(~hit) * g.res if hit.any() else np.full(cls.shape, np.inf)
    k = max(1, int(round(100 / g.res)))

    def look(lon, lat):
        """Distance from the settlement point to the nearest detected flood water / debris."""
        x, y = g.lonlat_to_xy(lon, lat)
        r, c = g.xy_to_rowcol(np.array([x]), np.array([y]))
        r, c = int(r[0]), int(c[0])
        if not (0 <= r < g.height and 0 <= c < g.width):
            return "outside the analysed area"
        d = float(dist[r, c])
        if d <= g.res:
            return "flood/debris detected at the settlement"
        if d <= 500:
            return f"flood/debris detected {int(round(d, -1))} m away"
        win = cls[max(r - k, 0):r + k + 1, max(c - k, 0):c + k + 1]
        if (win == NODATA).mean() > 0.5:
            return "not visible to the radar"
        return "none detected within 500 m"
    return look


def main(argv=None):
    from ..core import load_config, parse_date, setup_logging
    ap = argparse.ArgumentParser(description="Trace a flood path downstream from any point")
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--km", type=float, help="maximum length (default: config flowpath.max_length_km)")
    ap.add_argument("--date", help="flood date: OpenStreetMap is read as it was before this day")
    ap.add_argument("--run", help="a finished run folder, to say where damage was detected along the path")
    ap.add_argument("--out", help="write the result as JSON here")
    a = ap.parse_args(argv)
    try:                                   # API keys (ohsome v2, Copernicus) live in .env, as for run.py
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    setup_logging()
    out = flow_trace(a.lon, a.lat, load_config(), a.km, parse_date(a.date) if a.date else None,
                     Path(a.run) if a.run else None)
    print(f"\nPath {out['length_km']} km, drop {out['drop_m']} m; stopped: {out['stop_reason']}; "
          f"on a mapped river for {100 * (out['path_on_mapped_river'] or 0):.0f}% of its length; "
          f"{out['windows']} windows, {out['seconds']} s")
    for r in out.get("rivers_followed", []):
        print(f"  follows {r['name']}: km {r['from_km']}-{r['to_km']}")
    for s in out["settlements"]:
        dmg = f", {s['damage_nearby']}" if "damage_nearby" in s else ""
        nm = s["name"] + (f" ({s['name_en']})" if s.get("name_en") and s["name_en"] != s["name"] else "")
        print(f"  {s['km_downstream']:6.1f} km  {nm:40s} {s['m_above_river']:>5} m above river, "
              f"{s['m_from_river']} m away ({s['band']}){dmg}")
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def run_readers(run_dir: Path) -> dict:
    """Offline readers from a finished run (used for the synthetic demo valley): the run's own DEM and its
    OpenStreetMap layers. The path then stops at the edge of the run's area."""
    import geopandas as gpd
    from rasterio.warp import Resampling, reproject

    from ..core import Grid

    run_dir = Path(run_dir)
    R = json.loads((run_dir / "results.json").read_text())
    g = Grid.from_dict(R["grid"])
    dem = np.load(run_dir / "layers" / "hydro.npz")["dem"].astype(np.float32)

    def dem_reader(grid, aoi):
        out = np.full(grid.shape, np.nan, np.float32)
        reproject(dem, out, src_transform=g.transform, src_crs=g.crs, dst_transform=grid.transform,
                  dst_crs=grid.crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
        return out

    vec = R.get("vectors", {})

    def osm_reader(key, bbox):
        rel = {"places": vec.get("places"), "health": vec.get("health")}.get(key)
        if rel and (run_dir / rel).exists():
            return gpd.read_file(run_dir / rel)
        return None

    return {"dem_reader": dem_reader, "osm_reader": osm_reader}
