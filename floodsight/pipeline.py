"""End-to-end pipeline: area + flood date -> maps, damage, cut-off settlements, results.json, reports.

    python run.py --bbox 85.25,27.85,85.45,28.15 --date 2026-08-26
    python run.py --demo               # offline synthetic valley (no accounts, no internet)
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .core import AOI, RunContext, event_start, load_config, parse_date
from .detect.baseline import DEBRIS, FLOOD, NODATA, PERMANENT, detect_change, min_object_px
from .detect.fuse import apply_model, apply_optical, area_summary, combine_track_probs, merge_tracks
from . import activity

log = logging.getLogger("floodsight")

VERSION = "1.0.0"

ATTRIBUTIONS = [
    "Contains modified Copernicus Sentinel data 2026.",
    "Produced using Copernicus WorldDEM-30 © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH "
    "2014–2018 provided under COPERNICUS by the European Union and ESA; all rights reserved.",
    "© OpenStreetMap contributors.",
    "Model trained on Kuro Siwo: Bountos et al., 2024 (NeurIPS Datasets & Benchmarks).",
]

NO_CHANGE_KM2 = 0.05   # below this, the verdict is "no significant flood change detected"


class NoData(RuntimeError):
    """Raised when no usable satellite pair exists; carries a human-readable reason."""


def run(bbox: str | AOI, flood_date, cfg: dict | None = None, progress=None, demo: bool = False,
        runs_dir: str | Path | None = None, name: str | None = None) -> dict:
    t_start = time.perf_counter()
    cfg = cfg or load_config()
    flood_date = parse_date(flood_date)

    if demo:
        from .synthetic import DEFAULT_AOI
        aoi = DEFAULT_AOI
    else:
        aoi = bbox if isinstance(bbox, AOI) else AOI.parse(bbox)
    area = aoi.area_km2()
    if area > cfg["grid"]["max_area_km2"]:
        raise ValueError(f"Area is {area:.0f} km²; the limit is {cfg['grid']['max_area_km2']} km². "
                         "Draw a smaller box around the valley (or split it).")

    kw = {"runs_dir": Path(runs_dir)} if runs_dir else {}
    ctx = RunContext(aoi, flood_date, cfg, progress=progress, name=name or (f"demo_{flood_date}" if demo else None), **kw)
    log.info("Run %s  area %.0f km²  grid %s px at %s m", ctx.run_id, area, ctx.grid.shape, ctx.grid.res)

    # ------------------------------------------------------------------ 1. data
    if demo:
        from .synthetic import make_scene
        with ctx.stage("Loading demo valley", 0.05):
            sc = make_scene(aoi, cfg["grid"]["resolution_m"], flood=True)
            dem, osm, s2 = sc["dem"], sc["osm"], None
            pairs_data = [{"pair": {"track": "asc-085", "pre_date": str(flood_date - timedelta(days=4)),
                                    "post_date": str(flood_date + timedelta(days=8)), "gap_days": 12,
                                    "coverage": 1.0, "incidence_deg": sc["incidence"], "heading_deg": sc["heading"],
                                    "pre_scenes": ["synthetic"], "post_scenes": ["synthetic"]},
                           "pre": sc["pre"], "post": sc["post"], "backend": "synthetic",
                           "incidence": sc["incidence"], "heading": sc["heading"]}]
            ctx.note("DEMO MODE: synthetic valley, not real satellite data.")
    else:
        requests_log = {}
        def sink(d):
            requests_log[d["id"]] = d
        activity.add_sink(sink)
        t_wall = time.time()
        try:
            dem, osm, s2, pairs_data = _fetch_all(ctx)
        finally:
            activity.remove_sink(sink)
        ctx.data_requests = [{"source": d["source"], "what": d["what"], "status": d["status"],
                              "start_s": round(d["start"] - t_wall, 1),
                              "seconds": round((d["end"] or d["start"]) - d["start"], 1), "detail": d["detail"],
                              "requests": d["requests"], "mb": d["mb"], "pu": d["pu"], "waited_s": d["waited_s"]}
                             for d in sorted(requests_log.values(), key=lambda x: x["start"])]

    # ------------------------------------------------------------------ 2. terrain
    from .preprocess.geometry import slope_deg, visibility_mask
    from .preprocess.terrain import hydro_layers

    with ctx.stage("Terrain analysis (slope, drainage, HAND)", 0.45):
        slope = slope_deg(dem, ctx.grid.res)
        hydro = hydro_layers(dem, ctx.grid)
        mountain, mountain_info = None, {"applied": False}
        if cfg["detection"].get("debris_terrain", "mountain") == "mountain":
            from .preprocess.mountain import kapos_on_grid, mountain_mask
            if demo:
                mountain = kapos_on_grid(dem, ctx.grid.res) > 0
                mountain_info = {"definition": "UNEP-WCMC mountain classes (Kapos et al. 2000)",
                                 "source": "demo DEM", "mountain_fraction": round(float(mountain.mean()), 3)}
            else:
                mountain, mountain_info = mountain_mask(ctx, dem)
            mountain_info["applied"] = True
            from .core import write_raster
            write_raster(ctx.path("layers", "mountain.tif"), mountain.astype(np.uint8), ctx.grid, nodata=None)
            if mountain_info["mountain_fraction"] < 0.01:
                ctx.note("Plains terrain (UNEP-WCMC mountain definition): flooding is mapped as water; the "
                         "debris class applies only to mountain terrain.")
        # uncompressed (compressing 3 full-size layers took ~6 s on a large area); fdir codes fit in int16
        np.savez(ctx.path("layers", "hydro.npz"), fdir=hydro["fdir"].astype(np.int16), acc=hydro["acc"], dem=dem)

    # ------------------------------------------------------------------ 3. detection per track
    track_results = []
    with ctx.stage("Detecting flood water and debris", 0.55):
        for pdat in pairs_data:
            vis = visibility_mask(dem, ctx.grid.res, pdat["incidence"], pdat["heading"])
            ref = pdat.get("pre_ref")
            r = detect_change(pdat["pre"] if ref is None else ref, pdat["post"], vis["blind"], hydro["hand"], slope, hydro["streams"],
                              ctx.grid.res, cfg, mountain=mountain)
            r["track"] = pdat["pair"]["track"]
            r["stats"]["track"] = r["track"]
            track_results.append(r)
        corr_px = max(1, int(round(cfg["detection"].get("debris_corroboration_m", 40) / ctx.grid.res)))
        cls, conf = merge_tracks(track_results, corr_px)
        water_zone = (np.nan_to_num(hydro["hand"], nan=9e9) <= cfg["detection"]["debris_max_hand_m"]) & \
                     (slope <= cfg["detection"]["debris_max_slope_deg"])

    # ------------------------------------------------------------------ 4. model
    model_info = {"used": False}
    mcfg = cfg["model"]
    weights = Path(mcfg["weights"])
    if not weights.is_absolute():
        weights = Path(__file__).resolve().parent.parent / weights
    if mcfg.get("enabled") and weights.exists():
        try:
            with ctx.stage("Segmentation model (Siamese U-Net)", 0.65):
                from .model.infer import predict
                # Every track, like the rules: tracks see the flood on different days and from different sides.
                # Per pixel the highest flood probability among the tracks that actually see it (a track's
                # layover/shadow pixels are ignored for that track).
                probs, per_track = [], []
                for pdat, tr in zip(pairs_data, track_results):
                    pr = predict(pdat["pre"], pdat["post"], slope, ctx.grid.res, weights, mcfg["tile"],
                                 mcfg["overlap"], mcfg.get("tta", False),
                                 speckle_target=mcfg.get("input_speckle_db") if mcfg.get("match_speckle", True)
                                 else None)
                    p = np.where(tr["cls"] == NODATA, np.nan, pr["p_flood"]).astype(np.float32)
                    probs.append(p)
                    per_track.append({"track": pdat["pair"]["track"], "after": pdat["pair"]["post_date"],
                                      "flood_km2_p_ge_thr": round(float((np.nan_to_num(p) >= mcfg["flood_prob"]).sum())
                                                                  * ctx.grid.res ** 2 / 1e6, 3),
                                      "input_speckle_matching": pr.get("speckle_matching")})
                p_all = combine_track_probs(probs, [t["cls"] for t in track_results])
                cls0, conf0 = cls.copy(), conf.copy()
                cls, conf = apply_model(cls, conf, p_all, water_zone, mcfg["flood_prob"])
                from .detect.fuse import model_vs_rules
                vs_rules = model_vs_rules(cls0, conf0, cls, conf, p_all, mcfg["flood_prob"], ctx.grid.res)
                from .report.exports import class_png
                rules_bounds = class_png(cls0, conf0, ctx.grid, ctx.path("layers", "rules_web.png"), show_nodata=False)
                from .core import write_raster
                write_raster(ctx.path("layers", "model_p_flood.tif"), p_all, ctx.grid, nodata=np.nan)
                model_info = {"used": True, "weights": weights.name, "epoch": pr.get("model_epoch"),
                              "tracks": per_track, "combine": "per-pixel maximum over tracks that see the pixel",
                              "threshold": mcfg["flood_prob"], "vs_rules": vs_rules, "_rules_bounds": rules_bounds}
        except Exception as e:  # noqa: BLE001 - the rule-based map always stands on its own
            ctx.note(f"Segmentation model skipped ({type(e).__name__}: {str(e)[:80]}); rule-based map used.")
    elif mcfg.get("enabled"):
        ctx.note("No trained model weights found; flood map is from the rule-based detector only.")

    # ------------------------------------------------------------------ 5. optical
    optical_info = {"used": False}
    if s2 is not None:
        from .fetch.s2 import optical_evidence
        with ctx.stage("Adding Sentinel-2 optical evidence", 0.7):
            opt = optical_evidence(s2["pre"], s2["post"])
            clear = float(opt["valid_both"].mean())
            optical_info = {"used": clear >= cfg["sentinel2"]["min_clear_fraction"], "clear_fraction": round(clear, 3)}
            if optical_info["used"]:
                det = cfg["detection"]
                hand_ = np.nan_to_num(hydro["hand"], nan=9e9)
                flood_zone = (hand_ <= det["water_max_hand_m"]) & (slope <= det["water_max_slope_deg"])
                # debris only near the river corridor (same rule as the radar debris)
                from scipy import ndimage as ndi
                seed = hydro["streams"] | (cls == PERMANENT) | (cls == FLOOD)
                corridor = (ndi.distance_transform_edt(~seed) * ctx.grid.res <= det["corridor_link_m"]) if seed.any() \
                    else np.zeros(cls.shape, bool)
                dz = corridor if mountain is None else corridor & mountain
                cls, conf, ost = apply_optical(cls, conf, opt, water_zone, dz, flood_zone, min_object_px(det, ctx.grid.res),
                                               detect=cfg["sentinel2"].get("detect", True))
                px = ctx.grid.res ** 2 / 1e6
                optical_info.update({k.replace("_px", "_km2"): round(v * px, 3) for k, v in ost.items()})
            else:
                ctx.note(f"Sentinel-2 mostly cloudy ({clear:.0%} clear); radar-only result.")

    areas = area_summary(cls, conf, ctx.grid.res)
    changed = areas["flood_km2"] + areas["debris_km2"]
    # confirmed = medium/high confidence (two sources agree); low confidence = seen by one source only
    areas["confirmed_km2"] = round(areas["flood_high_conf_km2"] + areas["debris_high_conf_km2"], 3)
    areas["possible_km2"] = round(changed - areas["confirmed_km2"], 3)
    if areas["confirmed_km2"] >= NO_CHANGE_KM2:
        verdict = "flood_detected"
    elif changed >= NO_CHANGE_KM2:
        verdict = "possible_flood"          # only single-source detections: shown, but not called confirmed damage
        ctx.note("Only low-confidence changes were found (each seen by one source only): possible flooding, not "
                 "confirmed. Check on the ground or with another image before acting on it.")
    else:
        verdict = "no_significant_change"
    if verdict == "no_significant_change":
        ctx.note("No significant new flood water or debris detected between the two images.")
    if areas["nodata_fraction"] > 0.05:
        ctx.note(f"{areas['nodata_fraction']:.0%} of the area is hidden from the radar (steep slopes: "
                 "layover/shadow) or has no image; results there are unknown, not 'safe'.")

    # ------------------------------------------------------------------ 6. damage + access
    from .analysis.damage import assess_buildings, assess_roads, hit_layers, summarise
    from .analysis.network import analyse_access

    with ctx.stage("Assessing buildings, roads and bridges", 0.8):
        layers = hit_layers(cls, conf, ctx.grid.res)
        b = assess_buildings(osm["buildings"], layers, ctx.grid, cfg) if not osm["buildings"].empty else osm["buildings"]
        rd = assess_roads(osm["roads"], layers, ctx.grid, cfg) if not osm["roads"].empty else osm["roads"]
        damage = summarise(b, rd) if not (b.empty and rd.empty) else {}
    with ctx.stage("Finding cut-off settlements", 0.85):
        acc = analyse_access(osm["roads"], osm["places"], osm["health"], layers, ctx.grid, cfg, ctx.notes,
                             buildings=osm.get("buildings"))

    with ctx.stage("Helicopter landing options", 0.88):
        landing = _landing(cls, slope, acc, osm, ctx)

    # ------------------------------------------------------------------ 7. exports
    with ctx.stage("Writing maps and results", 0.9):
        results = _export(ctx, cls, conf, pairs_data, track_results, osm, b, rd, acc, areas, damage, verdict,
                          model_info, optical_info, mountain_info, s2=s2)
        results["helicopter_access"] = landing
        _landing_vectors(ctx, landing, results)
    with ctx.stage("Generating situation reports", 0.95):
        from .report.sitrep import build_reports
        results["reports"] = build_reports(results, ctx.dir)
    results["timings"] = dict(ctx.timings)
    results["timings"]["total"] = round(time.perf_counter() - t_start, 1)
    results["data_requests"] = getattr(ctx, "data_requests", [])
    (ctx.dir / "results.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    if progress:
        progress("Done", 1.0)
    log.info("Finished in %.1fs -> %s", results["timings"]["total"], ctx.dir)
    return results


# ---------------------------------------------------------------------------

def _fetch_all(ctx: RunContext):
    from .fetch.catalog import search_s1, select_pairs
    from .fetch.dem import fetch_dem
    from .fetch.osm import fetch_osm
    from .fetch.s1 import fetch_pair
    from .fetch.s2 import fetch_s2
    from .fetch.shub import USAGE

    USAGE.update(pu=0.0, requests=0)          # processing units are reported per run
    cfg, d = ctx.cfg, ctx.flood_date
    e = event_start(cfg, d)                   # before-images must be earlier than this day
    s1 = cfg["sentinel1"]
    if e < d:
        ctx.note(f"Before-images taken on or before {e - timedelta(days=1)} (set by the operator: the area was "
                 "still dry then). Change between that date and the after-image includes the whole build-up "
                 "of the flood, and also any other change in that period (crops, construction).")
    with ctx.stage("Searching the Sentinel-1 catalogue", 0.05):
        pairs = []
        for widen in (1, 2):
            back = max(s1["pre_window_days"], s1.get("pre_stack_window_days", 0) if s1.get("pre_stack_images", 1) > 1 else 0)
            a_, b_ = e - timedelta(days=back * widen), d + timedelta(days=s1["post_window_days"] * widen)
            with activity.task("Copernicus Data Space catalogue", f"Find Sentinel-1 passes, {a_} to {b_}") as t:
                scenes = search_s1(ctx.aoi, a_, b_)
                t.detail = f"{len(scenes)} image frames found"
            pairs = select_pairs(scenes, ctx.aoi, d, cfg, event_start=e)
            if pairs:
                break
        if not pairs:
            raise NoData("No Sentinel-1 before/after pair on the same orbit track covers this area within "
                         f"{s1['pre_window_days'] * 2} days before and {s1['post_window_days'] * 2} days after "
                         "the date. Try a later date or a different area.")
        if widen == 2:
            ctx.note("Search window was widened to find a same-track Sentinel-1 pair.")
        for p in pairs:
            log.info("Selected %s", p.describe())

    # Everything starts at once. openEO requests share 2 server slots (fetch.s1.OPENEO_SLOTS); Sentinel-1
    # is submitted first so it gets them first. Only the local GRD fallback has to wait for the DEM.
    with ctx.stage("Downloading satellite, elevation and map data", 0.1):
        with ThreadPoolExecutor(8) as ex:
            f_dem = ex.submit(fetch_dem, ctx)
            f_s1 = [ex.submit(fetch_pair, ctx, p, f_dem) for p in pairs]
            f_osm = ex.submit(fetch_osm, ctx)
            f_s2 = ex.submit(fetch_s2, ctx)
            dem = f_dem.result()
            pairs_data = []
            quota_err = None
            for p, f in zip(pairs, f_s1):
                try:
                    r = f.result()
                except Exception as e:  # noqa: BLE001 - one track failing should not kill the run
                    from .fetch.shub import ShubQuotaExceeded
                    if isinstance(e, ShubQuotaExceeded):
                        quota_err = e
                    import traceback
                    log.warning("Track %s failed:\n%s", p.pre.track, traceback.format_exc())
                    ctx.note(f"Track {p.pre.track} could not be processed ({type(e).__name__}: {str(e)[:120]}).")
                    continue
                pd_ = p.to_dict()
                pd_["pre_stack_dates"] = r.get("pre_stack_dates", [pd_["pre_date"]])
                pairs_data.append({"pair": pd_, "pre": r["pre"], "post": r["post"], "pre_ref": r.get("pre_ref"),
                                   "backend": r["backend"], "incidence": p.incidence_deg, "heading": p.heading_deg})
            try:
                osm = f_osm.result()
            except Exception as e:  # noqa: BLE001 - still deliver the flood map
                from .fetch.osm import empty_osm
                log.warning("OpenStreetMap unavailable: %s", str(e)[:300])
                ctx.note("OpenStreetMap could not be downloaded, so damage and cut-off analysis are missing "
                         "from this run; the flood map is complete. Re-run later to add them.")
                osm = empty_osm()
            s2 = f_s2.result()
        if not pairs_data:
            if quota_err is not None:
                raise quota_err
            raise NoData("Sentinel-1 images were found but none could be processed (see log).")
    return dem, osm, s2, pairs_data


AFFECTED = ("cut_off", "likely_cut_off", "unknown", "detour")


def _health_csv(health, settlements, path):
    """Every hospital / clinic from the pre-event map, with how many settlements depend on it and which of
    those lost or changed their road access. Opens in Excel; the report map shows only the facilities that
    affected settlements depend on, so the printed page stays readable."""
    import csv
    h = health.to_crs(4326)
    pts = h.geometry.representative_point()
    st = settlements if settlements is not None and not settlements.empty else None
    rows = []
    for i, (idx, rec) in enumerate(h.iterrows()):
        name = rec.get("name")
        name = str(name) if name is not None and str(name) not in ("", "nan", "None") else ""
        kind = next((str(rec.get(k)) for k in ("amenity", "healthcare") if rec.get(k) not in (None, "", "nan")
                     and str(rec.get(k)) != "nan"), "")
        users = aff = []
        if st is not None and name and "target_before" in st:
            users = st[st["target_before"].astype(str) == name]
            aff = users[users["status"].isin(AFFECTED)]
        rows.append({"name": name or f"unnamed {kind or 'facility'}", "type": kind,
                     "lat": round(pts.iloc[i].y, 5), "lon": round(pts.iloc[i].x, 5),
                     "settlements_nearest_before_flood": int(len(users)),
                     "of_them_lost_or_changed_access": int(len(aff)),
                     "affected_settlements": "; ".join(str(x) for x in (aff["name"].tolist() if len(aff) else []))})
    rows.sort(key=lambda r: (-r["of_them_lost_or_changed_access"], -r["settlements_nearest_before_flood"], r["name"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        w.writeheader()
        w.writerows(rows)
    return path


def _landing(cls, slope, acc, osm, ctx):
    from .analysis.landing import helicopter_access
    if "landing" not in ctx.cfg:
        return None
    try:
        return helicopter_access(cls, slope, acc.get("settlements"), osm.get("helipads"), osm.get("buildings"),
                                 ctx.grid, ctx.cfg, osm.get("power_lines"))
    except Exception as e:  # noqa: BLE001 - an extra must never fail the run
        log.warning("helicopter access skipped: %s", e)
        return None


def _landing_vectors(ctx, landing, results):
    """Landing zones and helipads as GeoJSON for the map; results.json is rewritten by the caller."""
    if not landing:
        return
    import json as _json
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [z["lat_lon"][1], z["lat_lon"][0]]},
              "properties": {"kind": "candidate landing zone", "clear_radius_m": z["clear_radius_m"],
                             "slope_deg": z["slope_deg"]}} for z in landing["zones"]]
    feats += [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [h["lat_lon"][1], h["lat_lon"][0]]},
               "properties": {"kind": "mapped helipad", "name": h["name"], "state": h["state"]}}
              for h in landing["helipads"]]
    path = ctx.path("vectors", "landing.geojson")
    path.write_text(_json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    results.setdefault("vectors", {})["landing"] = str(path.relative_to(ctx.dir))


def _blockages(rd):
    from .analysis.damage import named_blockages
    try:
        return named_blockages(rd)
    except Exception as e:  # noqa: BLE001 - a summary must never fail the run
        log.warning("named blockages skipped: %s", e)
        return {"roads": [], "bridges": [], "roads_total": 0, "bridges_total": 0}


def _shub_pu():
    from .fetch.shub import USAGE
    return round(USAGE["pu"], 1)


def _export(ctx, cls, conf, pairs_data, track_results, osm, b, rd, acc, areas, damage, verdict,
            model_info, optical_info, mountain_info=None, s2=None) -> dict:
    from .analysis.network import STATUS_LABEL
    from .report.exports import change_png, class_png, prob_png, quicklook_png, save_layers, shared_stretch, to_geojson, zoom_pngs

    save_layers(ctx, cls, conf)
    web = {}
    web["flood"] = {"png": "layers/flood_web.png", "bounds": class_png(cls, conf, ctx.grid, ctx.path("layers", "flood_web.png"))}
    p0 = pairs_data[0]
    pre, post = track_results[0]["vv_pre"], track_results[0]["vv_post"]
    lo, hi = shared_stretch(pre, post)       # same grey scale for both dates
    web["before"] = {"png": "layers/before_web.png", "bounds": quicklook_png(pre, ctx.grid, ctx.path("layers", "before_web.png"), lo, hi)}
    web["after"] = {"png": "layers/after_web.png", "bounds": quicklook_png(post, ctx.grid, ctx.path("layers", "after_web.png"), lo, hi)}
    web["change"] = {"png": "layers/change_web.png", "bounds": change_png(pre, post, ctx.grid, ctx.path("layers", "change_web.png"), lo, hi)}
    web["stretch_db"] = [round(lo, 1), round(hi, 1)]
    rules_bounds = model_info.pop("_rules_bounds", None)
    mp = ctx.path("layers", "model_p_flood.tif")
    if model_info.get("used") and mp.exists():
        try:
            import rasterio
            with rasterio.open(mp) as src:
                pm = src.read(1)
            web["model"] = {"png": "layers/model_web.png", "bounds": prob_png(pm, ctx.grid, ctx.path("layers", "model_web.png"))}
            if rules_bounds:
                web["rules"] = {"png": "layers/rules_web.png", "bounds": rules_bounds}
        except Exception as e:  # noqa: BLE001 - a picture must never fail the run
            log.warning("model layer picture skipped: %s", e)
    try:
        z = zoom_pngs(pre, post, cls, ctx.grid, ctx.path("layers", "zoom_before.png").parent, lo, hi, s2=s2)
        if z:
            web["zoom"] = z
    except Exception as e:  # noqa: BLE001 - a picture must never fail the run
        log.warning("zoom images skipped: %s", e)

    vec = {}
    if b is not None and not b.empty:
        hitb = b[b["status"].isin(["likely_hit", "possibly_hit"])]
        vec["buildings"] = str(to_geojson(hitb, ctx.path("vectors", "buildings_hit.geojson"),
                                          ["status", "overlap_pct"]).relative_to(ctx.dir))
    if rd is not None and not rd.empty:
        hitr = rd[rd["status"].isin(["likely_hit", "possibly_hit"])]
        vec["roads"] = str(to_geojson(hitr, ctx.path("vectors", "roads_hit.geojson"),
                                      [c for c in ("name", "highway", "status", "affected_m", "affected_confident_m", "is_bridge")
                                       if c in hitr.columns]).relative_to(ctx.dir))
    vec["settlements"] = str(to_geojson(acc["settlements"], ctx.path("vectors", "settlements.geojson")).relative_to(ctx.dir))
    vec["broken_roads"] = str(to_geojson(acc["broken_roads"], ctx.path("vectors", "broken_roads.geojson")).relative_to(ctx.dir))
    for k, fn in (("routes", "routes.geojson"), ("route_blocks", "route_blocks.geojson")):
        g = acc.get(k)
        if g is not None and not g.empty:
            vec[k] = str(to_geojson(g, ctx.path("vectors", fn)).relative_to(ctx.dir))
    if not acc["settlements"].empty:   # responders' priority list (also opens in Excel)
        pr = acc["settlements"].to_crs(4326)
        pr = pr.assign(lon=pr.geometry.x.round(5), lat=pr.geometry.y.round(5),
                       status_label=pr["status"].map(STATUS_LABEL)).drop(columns="geometry")
        csv_path = ctx.path("vectors", "priority_list.csv")
        pr.to_csv(csv_path, index=False, encoding="utf-8-sig")
        vec["priority_csv"] = str(csv_path.relative_to(ctx.dir))
    h = osm.get("health")
    health_total = 0
    if h is not None and not h.empty:
        keep = [c for c in ("name", "amenity", "healthcare") if c in h.columns]
        vec["health"] = str(to_geojson(h, ctx.path("vectors", "health.geojson"), keep).relative_to(ctx.dir))
        csv_h = _health_csv(h, acc["settlements"], ctx.path("report", "health_facilities.csv"))
        vec["health_csv"] = str(csv_h.relative_to(ctx.dir))
        health_total = int(len(h))
    pl = osm.get("places")
    if pl is not None and not pl.empty:
        keep = [c for c in ("name", "place") if c in pl.columns]
        vec["places"] = str(to_geojson(pl, ctx.path("vectors", "places.geojson"), keep).relative_to(ctx.dir))

    settle = acc["settlements"].drop(columns="geometry") if not acc["settlements"].empty else None
    table = []
    if settle is not None:
        for rec in settle.to_dict("records"):
            rec = {k: (None if isinstance(v, float) and not np.isfinite(v) else v) for k, v in rec.items()}
            rec["status_label"] = STATUS_LABEL.get(rec["status"], rec["status"])
            table.append(rec)

    lon, lat = ctx.aoi.center
    return {
        "floodsight_version": VERSION,
        "run_id": ctx.run_id,
        "generated_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "aoi": {"bbox": ctx.aoi.as_list(), "area_km2": round(ctx.aoi.area_km2(), 1), "center": [lon, lat]},
        "flood_date": ctx.flood_date.isoformat(),
        "before_images_until": (event_start(ctx.cfg, ctx.flood_date) - timedelta(days=1)).isoformat(),
        "grid": {**ctx.grid.to_dict(), "resolution_m": ctx.grid.res},
        "verdict": verdict,
        "health_facilities_total": health_total,
        "inputs": {
            "sentinel1_pairs": [p["pair"] for p in pairs_data],
            "sentinel1_backend": pairs_data[0]["backend"],
            "sentinel1_backend_per_track": {p["pair"]["track"]: p["backend"] for p in pairs_data},
            "sentinel1_calibration": {"sentinelhub": ctx.cfg["sentinel1"].get("shub_coefficient", "SIGMA0_ELLIPSOID"),
                                      "openeo": ctx.cfg["sentinel1"].get("coefficient", "sigma0-ellipsoid"),
                                      "local": ctx.cfg["sentinel1"].get("local_calibration", "sigma0") + " (local GRD processing)",
                                      "synthetic": "synthetic"}.get(
                                          pairs_data[0]["backend"], ""),
            "sentinelhub_processing_units": _shub_pu(),
            "sentinel2": optical_info,
            "osm_snapshot": osm.get("snapshot"),
            "model": model_info,
            "detection_stats": [t["stats"] for t in track_results],
            "debris_terrain": mountain_info or {"applied": False},
        },
        "areas": areas,
        "damage": damage,
        "access": acc["summary"],
        "blockages": _blockages(rd),
        "settlements": table,
        "notes": list(ctx.notes),
        "web": web,
        "vectors": vec,
        "attributions": ATTRIBUTIONS if model_info.get("used") else ATTRIBUTIONS[:3],
    }
