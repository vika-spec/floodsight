import re
from datetime import date
from pathlib import Path

import pytest

from floodsight.core import DataRuleViolation, assert_allowed_input, osm_snapshot_date

PKG = Path(__file__).resolve().parent.parent / "floodsight"


def test_osm_snapshot_is_before_event():
    assert osm_snapshot_date(date(2026, 8, 26), "2026-07-27") == date(2026, 7, 27)
    assert osm_snapshot_date(date(2023, 10, 4), "2026-07-27") == date(2023, 10, 3)


def test_reference_maps_refused_as_input():
    for p in ("validation_data/EMSR927/x.tif", "C:/data/UNOSAT_damage.tif"):
        with pytest.raises(DataRuleViolation):
            assert_allowed_input(p)
    assert_allowed_input("runs/x/inputs/dem.tif")


def test_pipeline_never_imports_validation_code():
    for f in PKG.rglob("*.py"):
        if "validate" in f.parts:
            continue
        src = f.read_text(encoding="utf-8")
        assert not re.search(r"(from|import)\s+[.\w]*validate", src), f"{f} imports validation code"
