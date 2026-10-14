"""Situation report 'Can we get there?': blocked roads and bridges named as on the pre-event map."""
import geopandas as gpd
from shapely.geometry import LineString

from floodsight.analysis.damage import named_blockages


def _roads():
    rows = [
        {"name": None, "name:en": "Pasang Lhamu Highway", "ref": "F21", "highway": "primary", "bridge": None,
         "status": "likely_hit", "affected_m": 900.0, "is_bridge": False, "geometry": LineString([(85.30, 28.10), (85.31, 28.10)])},
        {"name": None, "name:en": "Pasang Lhamu Highway", "ref": "F21", "highway": "primary", "bridge": None,
         "status": "possibly_hit", "affected_m": 300.0, "is_bridge": False, "geometry": LineString([(85.32, 28.10), (85.33, 28.10)])},
        {"name": None, "name:en": None, "ref": None, "highway": "track", "bridge": None,
         "status": "possibly_hit", "affected_m": 50.0, "is_bridge": False, "geometry": LineString([(85.40, 28.0), (85.41, 28.0)])},
        {"name": "Betrawati Bridge", "name:en": None, "ref": None, "highway": "primary", "bridge": "yes",
         "status": "likely_hit", "affected_m": 40.0, "is_bridge": True, "geometry": LineString([(85.18, 27.97), (85.181, 27.97)])},
        {"name": None, "name:en": None, "ref": None, "highway": "residential", "bridge": "yes",
         "status": "possibly_hit", "affected_m": 5.0, "is_bridge": True, "geometry": LineString([(85.2, 27.9), (85.201, 27.9)])},
        {"name": "Safe Road", "name:en": None, "ref": None, "highway": "secondary", "bridge": None,
         "status": "not_hit", "affected_m": 0.0, "is_bridge": False, "geometry": LineString([(85.5, 28.0), (85.51, 28.0)])},
    ]
    return gpd.GeoDataFrame(rows, crs=4326)


def test_named_roads_grouped_and_ranked():
    b = named_blockages(_roads())
    assert [r["name"] for r in b["roads"]] == ["Pasang Lhamu Highway", "unnamed track"]
    hw = b["roads"][0]
    assert hw["km_affected"] == 1.2 and hw["certain"] is True        # 900 m + 300 m, one segment certain
    assert b["roads"][1]["certain"] is False
    assert abs(hw["lat_lon"][0] - 28.10) < 1e-6 and 85.30 <= hw["lat_lon"][1] <= 85.31   # longest segment
    assert "Safe Road" not in [r["name"] for r in b["roads"]]


def test_only_likely_hit_bridges_listed():
    b = named_blockages(_roads())
    assert [x["name"] for x in b["bridges"]] == ["Betrawati Bridge"] and b["bridges_total"] == 1


def test_empty_input():
    assert named_blockages(None)["roads"] == []
