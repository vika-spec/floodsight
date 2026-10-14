"""Compare a FloodSight run with Copernicus EMS rapid-mapping products (e.g. EMSR927), area by area.

VALIDATION ONLY: reference maps are opened here and nowhere else (tests/test_rules.py enforces that the
pipeline never imports this package), so they can never become inputs.

Put the downloaded zip files (or their extracted folders) in one folder, then:

    python -m floodsight.validate.ems --inspect validation_data/EMSR927          # what is inside?
    python -m floodsight.validate.ems --run runs/trishuli_case --ems validation_data/EMSR927
    python -m floodsight.validate.ems --ems validation_data/EMSR680 --suggest --event-date 2023-08-04   # new event

For every EMS product (AOI x product/monitoring version) it reports:
  * flood/debris extent:  IoU, F1, precision, recall inside the EMS area of interest, excluding pixels the
    radar could not observe (reported separately), plus area of each EMS event type
  * buildings: share of EMS-damaged buildings that we flagged (recall) and share of our flagged buildings
    that EMS marks damaged (precision), matched within a distance tolerance
  * roads: share of EMS-damaged road length that our cut-road analysis also marks
Outputs runs/<run>/validation/{validation.json, ems_<product>.png, comparison.png}

Credit when publishing: "European Union, Copernicus Emergency Management Service data (EMSR927)".
"""
from __future__ import annotations

import argparse
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..detect.baseline import DEBRIS, FLOOD, NODATA
from ..model.metrics import binary_scores

VECTOR_EXT = (".shp", ".gpkg", ".geojson", ".json", ".kml")
EXT_PRIORITY = {".gpkg": 0, ".geojson": 1, ".json": 2, ".shp": 3, ".kml": 4}   # same layer in several formats -> one
ROLES = {  # filename fragment (lower case) -> role
    "areaofinterest": "aoi", "observedevent": "event", "builtup": "buildings",
    "transportation": "roads", "facilities": "facilities", "hydrography": "hydro",
    "imagefootprint": "images", "sourceimage": "images",
}
DAMAGED_WORDS = ("destroy", "damaged", "affected", "flooded")
NOT_DAMAGED = ("no visible", "not affected", "no damage", "undamaged", "unaffected", "not analysed",
               "not analyzed", "unknown", "no_visible")
POSSIBLY = ("possibl",)


# ---------------------------------------------------------------------------------------------- discovery

def _extract_zips(folder: Path) -> None:
    for z in folder.rglob("*.zip"):
        dest = z.with_suffix("")
        if not dest.exists():
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)


def discover(folder: str | Path) -> dict:
    """{product_key: {role: [files]}}, product_key like 'AOI01_GRA_PRODUCT_v1'."""
    folder = Path(folder)
    _extract_zips(folder)
    products = defaultdict(lambda: defaultdict(list))
    best = {}   # EMS ships every layer as .json AND .shp: keep one file per layer name, or features double
    for f in folder.rglob("*"):
        if f.suffix.lower() not in VECTOR_EXT or "__MACOSX" in f.parts:
            continue
        stem = f.stem.lower()
        if stem not in best or EXT_PRIORITY[f.suffix.lower()] < EXT_PRIORITY[best[stem].suffix.lower()]:
            best[stem] = f
    for f in sorted(best.values()):
        name = f.name.lower()
        role = next((r for frag, r in ROLES.items() if frag in name), None)
        if role is None:
            continue
        aoi = re.search(r"aoi(\d+)", name)
        prod = re.search(r"(gra|del|fep)_(product|monit\d*)", name)
        ver = re.search(r"_v(\d+)", name)
        key = "_".join(x for x in (f"AOI{int(aoi.group(1)):02d}" if aoi else "AOI??",
                                   prod.group(0).upper() if prod else "PRODUCT",
                                   f"v{ver.group(1)}" if ver else "") if x)
        products[key][role].append(f)
    return {k: dict(v) for k, v in sorted(products.items())}


def _rank(key: str) -> tuple:
    """Later map of the same AOI ranks higher: PRODUCT < MONIT01 < MONIT02 ..., then version."""
    m = re.search(r"MONIT(\d*)", key)
    mon = int(m.group(1) or 1) if m else 0
    v = re.search(r"_v(\d+)$", key)
    return mon, int(v.group(1)) if v else 0


def latest_per_aoi(keys) -> list:
    """One product per AOI for the combined score, so an AOI mapped twice is not counted twice."""
    by_aoi = {}
    for k in keys:
        a = k.split("_")[0]
        if a not in by_aoi or _rank(k) > _rank(by_aoi[a]):
            by_aoi[a] = k
    return sorted(by_aoi.values())


def _only(gdf, kinds):
    if gdf is None or len(gdf) == 0:
        return gdf
    return gdf[gdf.geom_type.isin(kinds)]


LINES = ("LineString", "MultiLineString")
POLYS = ("Polygon", "MultiPolygon")


def _read(files):
    import geopandas as gpd
    import pandas as pd

    parts = []
    for f in files:
        try:
            g = gpd.read_file(f)
        except Exception as e:  # noqa: BLE001
            print(f"  ! could not read {f.name}: {e}")
            continue
        if g.crs is None:
            g = g.set_crs(4326)
        parts.append(g.to_crs(4326))
    if not parts:
        return gpd.GeoDataFrame(geometry=[], crs=4326)
    return gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=4326)


def _category_columns(gdf, max_unique=20):
    out = {}
    for c in gdf.columns:
        if c == "geometry" or gdf[c].dtype.kind in "fiu":
            continue
        vals = gdf[c].dropna().astype(str)
        u = vals.unique()
        if 1 <= len(u) <= max_unique:
            out[c] = {v: int((vals == v).sum()) for v in u}
    return out


def damage_column(gdf):
    for c in gdf.columns:
        lc = c.lower()
        if any(k in lc for k in ("dmg", "damage", "grading", "grade")) and c != "geometry":
            return c
    return None


def damage_state(value) -> str:
    v = str(value).lower()
    if any(w in v for w in NOT_DAMAGED):
        return "none"
    if any(w in v for w in POSSIBLY):
        return "possible"
    if any(w in v for w in DAMAGED_WORDS):
        return "damaged"
    return "none"


def event_type_column(gdf):
    for c in ("event_type", "obj_type", "notation", "type", "descr", "evnt_type", "det_method"):
        for col in gdf.columns:
            if col.lower() == c:
                return col
    cats = _category_columns(gdf)
    return next(iter(cats), None)


def inspect(folder) -> None:
    folder = Path(folder)
    if not folder.exists():
        print(f"Folder not found: {folder.resolve()}\n"
              "Run this from the folder that contains run.py, with the zips in validation_data\\EMSR927.")
        return
    prods = discover(folder)
    if not prods:
        zips = list(folder.rglob("*.zip"))
        print(f"No EMS vector layers found under {folder.resolve()} ({len(zips)} zip files there).")
        if zips:
            with zipfile.ZipFile(zips[0]) as zf:
                print(f"Contents of {zips[0].name}: {zf.namelist()[:15]}  <- send this line")
        return
    for key, roles in prods.items():
        print(f"\n=== {key} ===")
        for role, files in roles.items():
            g = _read(files)
            types = g.geom_type.value_counts().to_dict() if len(g) else {}
            print(f"  {role:10s} {len(g):6d} features {types}  files: {[f.name for f in files]}")
            print(f"             columns: {[c for c in g.columns if c != 'geometry'][:20]}")
            for c, vals in list(_category_columns(g).items())[:6]:
                print(f"             {c}: {vals}")
            if role in ("buildings", "roads") and len(g):
                dc = damage_column(g)
                if dc:
                    states = g[dc].map(damage_state).value_counts().to_dict()
                    print(f"             damage column '{dc}' -> {states}")
    print("\nSend this output to check that event types and damage classes are read correctly.")


# ---------------------------------------------------------------------------------------------- suggest

def _footprint_info(files):
    """Image dates and sensors EMS lists in its image-footprint layer (names of columns differ between packages)."""
    g = _read(files)
    if not len(g):
        return [], []
    dates, sensors = set(), set()
    for c in g.columns:
        lc = c.lower()
        if c == "geometry":
            continue
        vals = g[c].dropna().astype(str)
        if any(w in lc for w in ("date", "time", "acq")):
            for v in vals:
                m = re.search(r"(20\d\d)-?([01]\d)-?([0-3]\d)", v)
                if m:
                    dates.add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
        if any(w in lc for w in ("sensor", "satellite", "platform", "src_name", "source")):
            sensors.update(v for v in vals.unique()[:10])
    return sorted(dates), sorted(sensors)


def suggest(folder, event_date=None, max_km2=3000.0) -> list:
    """Per AOI: its box, the images EMS used, and a ready run.py command (the date is the EVENT date)."""
    import math

    prods = discover(folder)
    rows, seen_aoi = [], set()
    for key in latest_per_aoi(prods):
        roles = prods[key]
        aoi_id = key.split("_")[0]
        if aoi_id in seen_aoi or "aoi" not in roles:
            continue
        seen_aoi.add(aoi_id)
        g = _read(roles["aoi"]).to_crs(4326)
        w, s_, e, n = g.total_bounds
        km2 = (e - w) * 111.32 * math.cos(math.radians((s_ + n) / 2)) * (n - s_) * 110.57
        dates, sensors = _footprint_info(roles["images"]) if "images" in roles else ([], [])
        ev = _read(roles["event"]) if "event" in roles else None
        types = {}
        if ev is not None and len(ev):
            col = event_type_column(ev)
            if col:
                types = ev[col].astype(str).value_counts().head(4).to_dict()
        rows.append({"product": key, "bbox": [round(float(x), 4) for x in (w, s_, e, n)], "km2": round(km2),
                     "image_dates": dates, "sensors": sensors, "event_types": types})
    print(f"\nEMS areas in {folder} (latest map per AOI):\n")
    for r in rows:
        print(f"{r['product']:24s} {r['km2']:6d} km²  box {','.join(map(str, r['bbox']))}")
        print(f"    EMS images: {', '.join(r['image_dates']) or 'dates not listed'}   sensors: {', '.join(r['sensors']) or '?'}")
        if r["event_types"]:
            print(f"    mapped as: {r['event_types']}")
        if any("sentinel-1" in x.lower() or "s1" == x.lower() for x in r["sensors"]):
            print("    note: EMS used Sentinel-1 here too -> agreement partly reflects the same image; say so when reporting")
        d = event_date or "<EVENT date (when the flood happened), not the EMS image date>"
        if r["km2"] <= max_km2:
            print(f"    run: python run.py --bbox {','.join(map(str, r['bbox']))} --date {d} --name <name>_{r['product'][:5].lower()}")
        else:
            print(f"    too large for one run ({r['km2']} km² > {max_km2:.0f}); pick a part of it")
        print()
    return rows


# ---------------------------------------------------------------------------------------------- compare

def _raster(gdf, tf, shape, crs, line_buffer_m=10.0):
    from rasterio.features import rasterize

    if gdf is None or len(gdf) == 0:
        return np.zeros(shape, bool)
    g = gdf.to_crs(crs)
    geoms = []
    for geom in g.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms.append(geom.buffer(0) if geom.geom_type in ("Polygon", "MultiPolygon") else geom.buffer(line_buffer_m))
    if not geoms:
        return np.zeros(shape, bool)
    return rasterize(((x, 1) for x in geoms), out_shape=shape, transform=tf, fill=0, dtype="uint8").astype(bool)


def _load_run(run: Path):
    import rasterio

    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        cls = src.read(1)
        return cls, src.transform, src.crs


def _vec(run: Path, name: str):
    import geopandas as gpd

    res = json.loads((run / "results.json").read_text())
    rel = res.get("vectors", {}).get(name)
    if not rel or not (run / rel).exists():
        return None
    g = gpd.read_file(run / rel)
    return None if g.empty else g


def _match_points(a, b, crs, tol_m):
    """For each geometry in a: is any geometry of b within tol_m? (centroids, metric CRS)"""
    from scipy.spatial import cKDTree

    if a is None or b is None or len(a) == 0 or len(b) == 0:
        return np.zeros(0 if a is None else len(a), bool)
    pa = np.array([(p.x, p.y) for p in a.to_crs(crs).geometry.centroid])
    pb = np.array([(p.x, p.y) for p in b.to_crs(crs).geometry.centroid])
    d, _ = cKDTree(pb).query(pa)
    return d <= tol_m


def _layer(run: Path, name: str):
    import rasterio

    f = run / "layers" / name
    if not f.exists():
        return None
    with rasterio.open(f) as src:
        return src.read(1)


def _breakdowns(run, cls, pred, ref, valid, in_aoi, event, tcol, tf, crs, shape, px_km2) -> dict:
    """Is the confidence meaningful (precision per confidence level)? How do mountain vs plain parts score?
    Water only: rules vs the segmentation model alone, against EMS 'flood' areas."""
    out = {}
    conf = _layer(run, "confidence.tif")
    if conf is not None:
        lv = {}
        for name, m in (("high_medium", pred & (conf >= 2)), ("low", pred & (conf == 1))):
            m = m & valid
            n = int(m.sum())
            lv[name] = {"km2": round(n * px_km2, 3), "precision": round(float((m & ref).sum() / n), 3) if n else None}
        out["by_confidence"] = lv
    mt = _layer(run, "mountain.tif")
    if mt is not None:
        tr = {}
        for name, m in (("mountain", mt > 0), ("plain", mt == 0)):
            v = valid & m
            if v.sum() and (ref & v).sum():
                sc = binary_scores(pred, ref, v)
                tr[name] = {"iou": round(sc["iou"], 3), "precision": round(sc["precision"], 3),
                            "recall": round(sc["recall"], 3), "ems_km2": round(float((ref & v).sum() * px_km2), 3)}
        out["by_terrain"] = tr
    p = _layer(run, "model_p_flood.tif")
    if p is not None and tcol and len(event):
        fl = event[event[tcol].astype(str).str.lower().str.contains("flood")]
        if len(fl):
            ref_w = _raster(fl, tf, shape, crs) & in_aoi
            if (ref_w & valid).any():
                from ..core import load_config
                thr = load_config()["model"].get("flood_prob", 0.8)
                rules = binary_scores(cls == FLOOD, ref_w, valid)
                model = binary_scores(np.nan_to_num(p, nan=0) >= thr, ref_w, valid)
                out["water_rules_vs_model"] = {
                    "threshold": thr,
                    "rules": {k: round(rules[k], 3) for k in ("iou", "precision", "recall")},
                    "model_alone": {k: round(model[k], 3) for k in ("iou", "precision", "recall")}}
    return out


def compare_product(run: Path, key: str, roles: dict, cls, tf, crs, tol_m=25.0) -> dict:
    px_km2 = abs(tf.a * tf.e) / 1e6
    shape = cls.shape
    aoi = _read(roles.get("aoi", []))
    event = _read(roles.get("event", []))
    in_aoi = _raster(aoi, tf, shape, crs) if len(aoi) else np.ones(shape, bool)
    covered = float(in_aoi.sum() * px_km2)
    out = {"product": key, "aoi_km2_in_run": round(covered, 2)}
    if covered == 0:
        out["skipped"] = "EMS area of interest lies outside this run's area"
        return out

    observed = cls != NODATA
    valid = in_aoi & observed
    pred = (cls == FLOOD) | (cls == DEBRIS)
    # extent from areas (and buffered lines); point symbols have no footprint
    event = _only(event, POLYS + LINES) if len(event) else event
    ref = _raster(event, tf, shape, crs)
    m = binary_scores(pred, ref, valid)
    out["extent"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()}
    out["area_km2"] = {
        "ours": round(float((pred & in_aoi).sum() * px_km2), 3),
        "ems": round(float((ref & in_aoi).sum() * px_km2), 3),
        "ems_in_radar_blind_spots": round(float((ref & in_aoi & ~observed).sum() * px_km2), 3),
        "radar_blind_share_of_aoi": round(float((in_aoi & ~observed).sum() / max(in_aoi.sum(), 1)), 3),
    }
    tcol = event_type_column(event) if len(event) else None
    if tcol:
        by_type = {}
        for t, sub in event.groupby(event[tcol].astype(str)):
            r = _raster(sub, tf, shape, crs) & in_aoi
            hit = r & valid
            n = max(hit.sum(), 1)
            by_type[t] = {"km2": round(float(r.sum() * px_km2), 3),
                          "share_we_detected": round(float((hit & pred).sum() / n), 3),
                          "detected_as_debris": round(float((hit & (cls == DEBRIS)).sum() / n), 3),
                          "detected_as_water": round(float((hit & (cls == FLOOD)).sum() / n), 3)}
        out["ems_event_types"] = {"column": tcol, "types": by_type}

    out.update(_breakdowns(run, cls, pred, ref, valid, in_aoi, event, tcol, tf, crs, shape, px_km2))

    # buildings
    b_ref = _read(roles.get("buildings", []))
    ours_b = _vec(run, "buildings")
    if len(b_ref):
        dc = damage_column(b_ref)
        if dc:
            st = b_ref[dc].map(damage_state)
            dmg = b_ref[st == "damaged"]
            dmg_loose = b_ref[st.isin(["damaged", "possible"])]
            if ours_b is not None:
                aoi_u = aoi.union_all() if len(aoi) else None
                ours_in = ours_b[ours_b.within(aoi_u)] if aoi_u is not None else ours_b
            else:
                ours_in = None
            rec = _match_points(dmg, ours_b, crs, tol_m)
            rec_l = _match_points(dmg_loose, ours_b, crs, tol_m)
            prec = _match_points(ours_in, dmg_loose, crs, tol_m)
            out["buildings"] = {
                "damage_column": dc, "ems_damaged": int(len(dmg)), "ems_damaged_or_possible": int(len(dmg_loose)),
                "ours_flagged_in_aoi": int(0 if ours_in is None else len(ours_in)),
                "recall_damaged": round(float(rec.mean()), 3) if len(rec) else None,
                "recall_incl_possible": round(float(rec_l.mean()), 3) if len(rec_l) else None,
                "precision": round(float(prec.mean()), 3) if len(prec) else None,
                "match_tolerance_m": tol_m,
            }

    # roads
    r_ref = _read(roles.get("roads", []))
    broken = _vec(run, "broken_roads")
    roads_hit = _vec(run, "roads")
    if len(r_ref):
        dc = damage_column(r_ref)
        if dc:
            lines = _only(r_ref, LINES)   # transportation layers mix points, areas and lines
            dmg = lines[lines[dc].map(damage_state).isin(["damaged", "possible"])].to_crs(crs)
            total = float(dmg.length.sum())
            ours = [g for g in (broken, roads_hit) if g is not None]
            if total > 0 and ours:
                import pandas as pd
                import geopandas as gpd
                u = gpd.GeoDataFrame(pd.concat([g.to_crs(crs) for g in ours]), crs=crs).buffer(30).union_all()
                covered_len = float(dmg.intersection(u).length.sum())
                out["roads"] = {"damage_column": dc, "ems_damaged_km": round(total / 1000, 2),
                                "ems_damaged_segments": int(len(dmg)),
                                "share_we_flagged": round(covered_len / total, 3)}
                # precision of OUR road closures (the ones that drive the cut-off analysis): share of the road
                # length we closed inside the EMS area that EMS also shows as affected (damaged road within
                # 30 m, or inside an EMS-mapped flood / mass-movement area)
                if broken is not None and len(aoi):
                    ours_b = broken.to_crs(crs)
                    a_u = aoi.to_crs(crs).union_all()
                    ours_in = ours_b.intersection(a_u)
                    ours_len = float(ours_in.length.sum())
                    if ours_len > 0:
                        ok = dmg.buffer(30).union_all()
                        if len(event):
                            ok = ok.union(_only(event, POLYS).to_crs(crs).buffer(0).union_all()) if len(_only(event, POLYS)) else ok
                        conf_len = float(ours_in.intersection(ok).length.sum())
                        out["roads"]["our_closed_km_in_aoi"] = round(ours_len / 1000, 2)
                        out["roads"]["closure_precision"] = round(conf_len / ours_len, 3)

    # facilities (dams, hydropower, bridges ...): is each damaged one inside or next to our flood/debris map?
    f_ref = _read(roles.get("facilities", []))
    if len(f_ref):
        dc = damage_column(f_ref)
        if dc:
            from scipy.ndimage import binary_dilation
            st = f_ref[dc].map(damage_state)
            dmg = f_ref[st.isin(["damaged", "possible"])]
            if len(dmg):
                near = binary_dilation(pred, iterations=max(1, int(round(50 / abs(tf.a)))))
                g = dmg.to_crs(crs)
                rows, cols = [], []
                for p in g.geometry.representative_point():
                    col, row = ~tf * (p.x, p.y)
                    rows.append(int(np.floor(row)))
                    cols.append(int(np.floor(col)))
                rows, cols = np.array(rows), np.array(cols)
                ok = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
                hit = np.zeros(len(g), bool)
                hit[ok] = near[rows[ok], cols[ok]]
                tcol = next((c for c in g.columns if c.lower() in ("obj_type", "obj_desc", "type", "notation")), None)
                out["facilities"] = {
                    "ems_damaged": int(len(g)), "inside_run": int(ok.sum()),
                    "within_50m_of_our_map": int(hit[ok].sum()),
                    "items": [{"type": str(g[tcol].iloc[i]) if tcol else "", "state": str(g[dc].iloc[i]),
                               "we_flagged": bool(hit[i])} for i in range(len(g)) if ok[i]][:30],
                }
    _figure(run, key, cls, pred, ref, valid, in_aoi)
    return out


_EMS_KIND = {"GRA": "damage map", "DEL": "flood outline", "FEP": "first estimate"}


def plain_name(key: str) -> str:
    """Readable name for a product key: 'AOI01_GRA_MONIT01_v1' -> 'EMS area 1, damage map, update 1'."""
    m = re.search(r"AOI0*(\d+)_([A-Z]+)_(PRODUCT|MONIT0*(\d+))", str(key))
    if not m:
        return str(key).replace("_", " ")
    kind = _EMS_KIND.get(m.group(2), m.group(2))
    return f"EMS area {m.group(1)}, {kind}" + (f", update {m.group(4)}" if m.group(4) else ", first version")


def _figure(run, key, cls, pred, ref, valid, in_aoi):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rows, cols = np.nonzero(in_aoi)
    if len(rows) == 0:
        return
    pad = 15
    r0, r1 = max(rows.min() - pad, 0), min(rows.max() + pad, cls.shape[0])
    c0, c1 = max(cols.min() - pad, 0), min(cols.max() + pad, cls.shape[1])
    sl = np.s_[r0:r1, c0:c1]
    img = np.ones((r1 - r0, c1 - c0, 3)) * 0.97
    v, p, t, a = valid[sl], pred[sl], ref[sl], in_aoi[sl]
    img[~a] = (0.9, 0.9, 0.9)
    img[a & ~v] = (0.7, 0.7, 0.7)
    img[v & p & t] = (0.17, 0.63, 0.17)
    img[v & p & ~t] = (0.84, 0.15, 0.16)
    img[v & ~p & t] = (0.12, 0.47, 0.71)
    h = max(3, min(8, 7 * img.shape[0] / img.shape[1]))
    fig, ax = plt.subplots(figsize=(7, h), dpi=150)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(f"{plain_name(key)}: FloodSight vs Copernicus EMS", fontsize=9)
    ax.set_axis_off()
    ax.legend(handles=[Patch(color=(0.17, 0.63, 0.17), label="Found by both"),
                       Patch(color=(0.84, 0.15, 0.16), label="Only FloodSight (extra)"),
                       Patch(color=(0.12, 0.47, 0.71), label="Only EMS (we missed it)"),
                       Patch(color=(0.7, 0.7, 0.7), label="Radar could not see"),
                       Patch(color=(0.97, 0.97, 0.97), label="EMS area, no damage"),
                       Patch(color=(0.9, 0.9, 0.9), label="Outside EMS area")],
              loc="lower right", fontsize=6)
    fig.tight_layout()
    (run / "validation").mkdir(exist_ok=True)
    fig.savefig(run / "validation" / f"ems_{key}.png")
    plt.close(fig)


def compare_all(run, folder, only=None) -> dict:
    from PIL import Image

    run = Path(run)
    cls, tf, crs = _load_run(run)
    prods = discover(folder)
    if only:
        prods = {k: v for k, v in prods.items() if any(o.lower() in k.lower() for o in only)}
    results = [compare_product(run, k, v, cls, tf, crs) for k, v in prods.items()]
    done = [r for r in results if "extent" in r]
    used = latest_per_aoi([r["product"] for r in done])
    done = [r for r in done if r["product"] in used]
    tot = {k: sum(r["extent"][k] for r in done) for k in ("tp", "fp", "fn", "tn")} if done else None
    combined = None
    if tot:
        tp, fp, fn = tot["tp"], tot["fp"], tot["fn"]
        combined = {"iou": round(tp / max(tp + fp + fn, 1), 4), "precision": round(tp / max(tp + fp, 1), 4),
                    "recall": round(tp / max(tp + fn, 1), 4), "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
                    **tot}
    summary = {
        "reference": f"Copernicus EMS ({Path(folder).name}); used only for checking, never as input",
        "flood_extent": combined or {"iou": 0, "f1": 0, "precision": 0, "recall": 0},
        "combined_from": used,
        "what_ems_mapped": _what_ems_mapped(results),
        "products": results,
        "note": "Scores exclude pixels the radar could not observe. EMS maps come from very-high-resolution "
                "optical images; differences in date, resolution and mapping definitions are expected.",
    }
    out_dir = run / "validation"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "validation.json").write_text(json.dumps(summary, indent=1))
    figs = [out_dir / f"ems_{r['product']}.png" for r in done if (out_dir / f"ems_{r['product']}.png").exists()]
    if figs:   # stack per-AOI figures into one image for the dashboard
        ims = [Image.open(f).convert("RGB") for f in figs]
        w = max(i.width for i in ims)
        canvas = Image.new("RGB", (w, sum(i.height for i in ims)), "white")
        y = 0
        for i in ims:
            canvas.paste(i, (0, y))
            y += i.height
        canvas.save(out_dir / "comparison.png")
    return summary


def _what_ems_mapped(results) -> str:
    types = {t.lower() for r in results for t in r.get("ems_event_types", {}).get("types", {})}
    if types and all(("mass" in t or "landslide" in t or "debris" in t) for t in types):
        return ("EMS mapped the event as mass movement (debris flow / landslide), not open water, so it is compared "
                "with our flood + debris footprint; the share we classed as debris is reported per area")
    return "EMS observed-event extent compared with our flood + debris footprint"


def print_summary(s: dict) -> None:
    print(f"\n{'product':28s} {'AOI km2':>8s} {'IoU':>6s} {'F1':>6s} {'prec':>6s} {'recall':>6s} "
          f"{'ours km2':>9s} {'EMS km2':>8s} {'blind%':>7s}")
    for r in s["products"]:
        if "extent" not in r:
            print(f"{r['product']:28s} {r['aoi_km2_in_run']:8.2f}  -- {r.get('skipped', '')}")
            continue
        e, a = r["extent"], r["area_km2"]
        print(f"{r['product']:28s} {r['aoi_km2_in_run']:8.2f} {e['iou']:6.3f} {e['f1']:6.3f} {e['precision']:6.3f} "
              f"{e['recall']:6.3f} {a['ours']:9.3f} {a['ems']:8.3f} {100 * a['radar_blind_share_of_aoi']:6.1f}%")
        if r.get("ems_event_types"):
            for t, v in r["ems_event_types"]["types"].items():
                print(f"    EMS '{t}': {v['km2']} km2, we detected {100 * v['share_we_detected']:.0f}% "
                      f"(as debris {100 * v.get('detected_as_debris', 0):.0f}%, as water {100 * v.get('detected_as_water', 0):.0f}%)")
        if r.get("buildings"):
            b = r["buildings"]
            print(f"    buildings: EMS damaged {b['ems_damaged']} (+possible {b['ems_damaged_or_possible'] - b['ems_damaged']}),"
                  f" recall {b['recall_damaged']}, precision {b['precision']} (ours flagged {b['ours_flagged_in_aoi']})")
        if r.get("by_confidence"):
            bc = r["by_confidence"]
            print("    confidence: " + ", ".join(f"{k.replace('_', '/')} {v['km2']} km2 precision {v['precision']}"
                                            for k, v in bc.items() if v["km2"]))
        if r.get("by_terrain"):
            print("    terrain: " + ", ".join(f"{k} IoU {v['iou']} prec {v['precision']} recall {v['recall']}"
                                         for k, v in r["by_terrain"].items()))
        if r.get("water_rules_vs_model"):
            w = r["water_rules_vs_model"]
            print(f"    water only vs EMS flood: rules IoU {w['rules']['iou']} (prec {w['rules']['precision']}, "
                  f"recall {w['rules']['recall']}); model alone IoU {w['model_alone']['iou']} "
                  f"(prec {w['model_alone']['precision']}, recall {w['model_alone']['recall']})")
        if r.get("roads"):
            rr = r["roads"]
            extra = (f"; of the {rr['our_closed_km_in_aoi']} km we closed, {100 * rr['closure_precision']:.0f}% are "
                     "affected in EMS" if rr.get("closure_precision") is not None else "")
            print(f"    roads: EMS damaged {rr['ems_damaged_km']} km, we flagged {100 * rr['share_we_flagged']:.0f}%{extra}")
        if r.get("facilities"):
            f = r["facilities"]
            print(f"    facilities: EMS damaged {f['ems_damaged']} ({f['inside_run']} in our area), "
                  f"{f['within_50m_of_our_map']} within 50 m of our flood/debris map")
    c = s["flood_extent"]
    print(f"\n{s.get('what_ems_mapped', '')}")
    print(f"Combined uses the latest map per AOI: {', '.join(s.get('combined_from', []))}")
    print(f"ALL AOIs combined: IoU {c['iou']:.3f}  F1 {c['f1']:.3f}  precision {c['precision']:.3f}  recall {c['recall']:.3f}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ems", "--inspect", dest="ems", required=True, help="folder with the EMS zip files")
    ap.add_argument("--run", help="run folder to compare (omit to only inspect the EMS data)")
    ap.add_argument("--only", nargs="*", help="limit to products whose name contains these, e.g. AOI01 PRODUCT")
    ap.add_argument("--suggest", action="store_true", help="print each AOI's box, EMS image dates and a run command")
    ap.add_argument("--event-date", help="with --suggest: the date the flood happened (YYYY-MM-DD)")
    a = ap.parse_args(argv)
    if a.suggest:
        suggest(a.ems, a.event_date)
        return
    if not a.run:
        inspect(a.ems)
        return
    s = compare_all(a.run, a.ems, a.only)
    print_summary(s)
    print(f"\nSaved {Path(a.run) / 'validation'}")


if __name__ == "__main__":
    main()
