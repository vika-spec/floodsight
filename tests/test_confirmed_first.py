"""Results lead with confirmed numbers (two sources agree); Nepali report only for areas in Nepal; the report
map shows only hospitals that affected settlements depend on, the full list goes to a CSV."""
import csv
import json

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from floodsight.pipeline import _health_csv, run
from floodsight.report.sitrep import report_languages, touches_nepal


def test_nepal_check():
    assert touches_nepal([85.08, 27.85, 85.40, 28.29])          # Trishuli
    assert touches_nepal([85.30, 27.95, 85.44, 28.04])          # demo valley location
    assert not touches_nepal([76.84, 28.40, 77.35, 28.88])      # north India plains
    assert not touches_nepal([25.40, 35.14, 25.54, 35.22])      # Crete
    assert not touches_nepal([72.40, 34.40, 72.70, 34.60])      # Pakistan
    assert report_languages({"aoi": {"bbox": [76.84, 28.40, 77.35, 28.88]}}) == ["en"]
    assert report_languages({"aoi": {"bbox": [85.08, 27.85, 85.40, 28.29]}}) == ["en", "ne"]
    assert report_languages({}) == ["en"]                       # never lose the English report


def test_demo_run_confirmed_split_and_outputs(tmp_path):
    R = run(None, "2026-08-26", demo=True, runs_dir=tmp_path, name="demo_t")
    a, d = R["areas"], R["damage"]
    assert abs(a["confirmed_km2"] - (a["flood_high_conf_km2"] + a["debris_high_conf_km2"])) < 1e-6
    assert abs(a["confirmed_km2"] + a["possible_km2"] - (a["flood_km2"] + a["debris_km2"])) < 1e-3
    assert 0 <= d["road_km_confirmed"] <= d["road_km_affected"] and 0 <= d["bridges_confirmed"] <= d["bridges_hit"]
    assert R["verdict"] == "flood_detected"
    assert set(R["reports"]) >= {"en", "ne"}                    # demo valley sits in Nepal
    rows = list(csv.DictReader(open(tmp_path / "demo_t" / R["vectors"]["health_csv"], encoding="utf-8-sig")))
    assert len(rows) == R["health_facilities_total"] and rows[0]["lat"] and rows[0]["lon"]


def test_health_csv_counts_dependent_settlements(tmp_path):
    h = gpd.GeoDataFrame({"name": ["A Hospital", "B Clinic", None], "amenity": ["hospital", "clinic", "doctors"]},
                         geometry=[Point(85.30, 28.0), Point(85.32, 28.0), Point(85.34, 28.0)], crs=4326)
    st = gpd.GeoDataFrame({"name": ["V1", "V2", "V3"], "status": ["cut_off", "reachable", "detour"],
                           "target_before": ["A Hospital", "A Hospital", "B Clinic"]},
                          geometry=[Point(85.31, 28.01)] * 3, crs=4326)
    p = _health_csv(h, st, tmp_path / "h.csv")
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    first = rows[0]
    assert first["name"] == "A Hospital" and first["settlements_nearest_before_flood"] == "2"
    assert first["of_them_lost_or_changed_access"] == "1" and first["affected_settlements"] == "V1"
    assert any(r["name"].startswith("unnamed") for r in rows)
