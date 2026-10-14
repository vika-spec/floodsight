"""EMS rapid-mapping comparison on a package laid out like the real EMSR927 download:
every layer shipped as both .json and .shp, mixed point/line road layers, an AOI mapped twice
(PRODUCT + MONIT01), and the event typed as mass movement."""


def _write(gdf, folder, stem):
    gdf.to_file(folder / f"{stem}.json", driver="GeoJSON")
    gdf.to_file(folder / f"{stem}.shp")


def _fake_ems(tmp_path, scene):
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import LineString, Point, box, shape

    g = scene["grid"]
    polys = [shape(s) for s, v in shapes((scene["truth"] > 0).astype("uint8"), transform=g.transform) if v == 1]
    x0, y0 = g.transform.c, g.transform.f + g.transform.e * g.height
    x1, y1 = g.transform.c + g.transform.a * g.width, g.transform.f
    folder = tmp_path / "EMSR999"
    for prod in ("PRODUCT", "MONIT01"):
        d = folder / f"EMSR999_AOI01_GRA_{prod}_VECTOR_v1"
        d.mkdir(parents=True)
        pre = f"EMSR999_AOI01_GRA_{prod}"
        _write(gpd.GeoDataFrame({"name": ["Test"]}, geometry=[box(x0, y0, x1, y1)], crs=g.crs), d,
               f"{pre}_areaOfInterestA_v1")
        _write(gpd.GeoDataFrame({"event_type": ["6-Mass Movement"] * len(polys), "obj_desc": ["Landslide"] * len(polys)},
                                geometry=polys, crs=g.crs), d, f"{pre}_observedEventA_v1")
        c = polys[0].representative_point()
        _write(gpd.GeoDataFrame({"damage_gra": ["Destroyed", "Possibly damaged"]},
                                geometry=[c, Point(c.x + 5, c.y)], crs=g.crs), d, f"{pre}_builtUpP_v1")
        _write(gpd.GeoDataFrame({"damage_gra": ["Damaged", "Damaged"]},
                                geometry=[LineString([(c.x - 300, c.y), (c.x + 300, c.y)]),
                                          LineString([(c.x, c.y - 300), (c.x, c.y + 300)])], crs=g.crs),
               d, f"{pre}_transportationL_v1")
        _write(gpd.GeoDataFrame({"damage_gra": ["Damaged"]}, geometry=[c], crs=g.crs), d, f"{pre}_transportationP_v1")
        _write(gpd.GeoDataFrame({"damage_gra": ["Destroyed"], "obj_type": ["Dam"]}, geometry=[c], crs=g.crs),
               d, f"{pre}_facilitiesP_v1")
        _write(gpd.GeoDataFrame({"sensor_name": ["Sentinel-1A"], "acquisition_time": ["2026-08-28T00:12:00"]},
                                geometry=[box(x0, y0, x1, y1)], crs=g.crs), d, f"{pre}_imageFootprintA_v1")
    return folder


def test_ems_dedupe_and_compare(tmp_path, cfg, scene):
    from floodsight.pipeline import run
    from floodsight.validate.ems import compare_all, discover

    folder = _fake_ems(tmp_path, scene)
    prods = discover(folder)
    assert set(prods) == {"AOI01_GRA_PRODUCT_v1", "AOI01_GRA_MONIT01_v1"}
    # one file per layer even though each is shipped as .json and .shp
    assert len(prods["AOI01_GRA_PRODUCT_v1"]["event"]) == 1
    assert len(prods["AOI01_GRA_PRODUCT_v1"]["roads"]) == 2       # L and P are different layers

    r = run(None, "2026-08-26", cfg, demo=True, runs_dir=tmp_path / "runs")
    s = compare_all(tmp_path / "runs" / r["run_id"], folder)
    assert s["combined_from"] == ["AOI01_GRA_MONIT01_v1"]          # AOI mapped twice is counted once
    one = next(p for p in s["products"] if p["product"] == "AOI01_GRA_MONIT01_v1")
    assert s["flood_extent"]["tp"] == one["extent"]["tp"]
    assert s["flood_extent"]["iou"] > 0.8
    assert "mass movement" in s["what_ems_mapped"]
    assert one["buildings"]["ems_damaged"] == 1                    # not doubled
    assert one["by_confidence"]["high_medium"]["precision"] > 0.8    # confident detections are mostly right
    assert "mountain" in one["by_terrain"]
    assert one["roads"]["closure_precision"] > 0.7                 # our closed roads lie in the EMS-mapped area
    assert one["roads"]["ems_damaged_segments"] == 2               # lines only, the point is ignored
    assert abs(one["roads"]["ems_damaged_km"] - 1.2) < 0.05
    assert one["facilities"]["ems_damaged"] == 1
    assert one["facilities"]["within_50m_of_our_map"] == 1
    t = one["ems_event_types"]["types"]["6-Mass Movement"]
    assert abs(t["detected_as_debris"] + t["detected_as_water"] - t["share_we_detected"]) < 0.01


def test_ems_suggest_lists_box_dates_and_sensor(tmp_path, scene, capsys):
    from floodsight.validate.ems import suggest

    rows = suggest(_fake_ems(tmp_path, scene), "2026-08-26")
    assert len(rows) == 1 and rows[0]["product"] == "AOI01_GRA_MONIT01_v1"
    assert rows[0]["image_dates"] == ["2026-08-28"] and rows[0]["sensors"] == ["Sentinel-1A"]
    w, s_, e, n = rows[0]["bbox"]
    assert -180 <= w < e <= 180 and -90 <= s_ < n <= 90
    out = capsys.readouterr().out
    assert "--date 2026-08-26" in out and "Sentinel-1 here too" in out


def test_miss_breakdown_runs_and_accounts_for_every_pixel(tmp_path, cfg, scene):
    from floodsight.pipeline import run
    from floodsight.validate.misses import breakdown

    folder = _fake_ems(tmp_path, scene)
    r = run(None, "2026-08-26", cfg, demo=True, runs_dir=tmp_path / "runs")
    o = breakdown(tmp_path / "runs" / r["run_id"], folder, event_date="2026-08-26")
    assert abs(sum(b["pct"] for b in o["buckets"].values()) - 100) < 0.5
    assert o["buckets"]["found"]["pct"] > 50                       # the demo detects most of its own truth
    assert o["with_tolerance"]["recall"] >= o["strict"]["recall"]
    assert o["tracks"] and o["tracks"][0]["after_days_after_event"] is not None
    assert (tmp_path / "runs" / r["run_id"] / "validation" / "misses.png").exists()
    # model statistics on the EMS area: write a model layer that is 0.9 wherever our map found flood/debris
    import numpy as np
    import rasterio
    rd = tmp_path / "runs" / r["run_id"]
    with rasterio.open(rd / "layers" / "flood_class.tif") as src:
        cls, prof = src.read(1), src.profile
    prof.update(dtype="float32", nodata=None)
    with rasterio.open(rd / "layers" / "model_p_flood.tif", "w", **prof) as dst:
        dst.write(np.where((cls == 1) | (cls == 2), 0.9, 0.1).astype(np.float32), 1)
    m = breakdown(rd, folder, event_date="2026-08-26")["model_on_ems_area"]
    assert m["median_p_where_rules_found_it"] == 0.9
    assert m["share_p_ge_0.8"] == m["share_p_ge_0.5"] > 0.5


def test_water_rules_vs_model_breakdown(tmp_path):
    import geopandas as gpd
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box

    from floodsight.detect.baseline import FLOOD, NO_CHANGE
    from floodsight.validate.ems import _breakdowns

    run = tmp_path / "r"
    (run / "layers").mkdir(parents=True)
    tf = from_origin(500000, 3100000, 20, 20)
    shape = (50, 50)
    p = np.zeros(shape, np.float32)
    p[10:30, 10:30] = 0.95                                     # model finds the whole flooded square
    with rasterio.open(run / "layers" / "model_p_flood.tif", "w", driver="GTiff", height=50, width=50, count=1,
                       dtype="float32", crs="EPSG:32645", transform=tf) as d:
        d.write(p, 1)
    cls = np.full(shape, NO_CHANGE, np.uint8)
    cls[10:20, 10:30] = FLOOD                                  # rules find half of it
    ev = gpd.GeoDataFrame({"event_type": ["5-Flood"]}, geometry=[box(500200, 3099400, 500600, 3099800)],
                          crs="EPSG:32645")
    ones = np.ones(shape, bool)
    from floodsight.validate.ems import _raster
    ref = _raster(ev, tf, shape, "EPSG:32645")
    out = _breakdowns(run, cls, cls == FLOOD, ref, ones, ones, ev, "event_type", tf, "EPSG:32645", shape, 0.0004)
    w = out["water_rules_vs_model"]
    assert w["model_alone"]["recall"] > 0.95 and abs(w["rules"]["recall"] - 0.5) < 0.05


def test_plain_name():
    from floodsight.validate.ems import plain_name
    assert plain_name("AOI01_GRA_PRODUCT_v1") == "EMS area 1, damage map, first version"
    assert plain_name("AOI03_GRA_MONIT01_v1") == "EMS area 3, damage map, update 1"
    assert plain_name("AOI10_DEL_MONIT02_v2") == "EMS area 10, flood outline, update 2"
    assert plain_name("something_else") == "something else"
