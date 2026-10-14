from datetime import date, datetime

from floodsight.core import AOI
from floodsight.fetch.catalog import Scene, parse_s1_items, select_pairs

AOI_ = AOI(85.2, 27.9, 85.4, 28.1)
FP = [[84.0, 27.0], [86.5, 27.0], [86.5, 29.0], [84.0, 29.0], [84.0, 27.0]]


def sc(d, orbit, direction="ASCENDING"):
    return Scene(f"id{d}{orbit}", f"S1A_IW_GRDH_1SDV_{d.replace('-', '')}T000000", datetime.fromisoformat(d),
                 "S1A", orbit, direction, FP)


def test_same_track_latest_before_earliest_after(cfg):
    scenes = [sc("2026-08-10", 85), sc("2026-08-22", 85), sc("2026-09-03", 85), sc("2026-09-15", 85),
              sc("2026-08-24", 12, "DESCENDING"), sc("2026-08-30", 12, "DESCENDING")]
    pairs = select_pairs(scenes, AOI_, date(2026, 8, 26), cfg)
    tracks = {p.pre.track: p for p in pairs}
    assert tracks["asc-085"].pre.day == date(2026, 8, 22)
    assert tracks["asc-085"].post.day == date(2026, 9, 3)
    assert tracks["des-012"].post.day == date(2026, 8, 30)
    for p in pairs:
        assert p.pre.rel_orbit == p.post.rel_orbit and p.pre.direction == p.post.direction
    assert pairs[0].pre.track == "des-012"      # earlier post image ranks first


def test_no_pair_without_post_image(cfg):
    assert select_pairs([sc("2026-08-22", 85)], AOI_, date(2026, 8, 26), cfg) == []


def test_parse_dedupes_cog_and_reads_attributes():
    item = {"Id": "a", "Name": "S1A_IW_GRDH_1SDV_20260822T001122_20260822T001147_060123_07ABCD_1234.SAFE",
            "ContentDate": {"Start": "2026-08-22T00:11:22.000Z"},
            "Attributes": [{"Name": "relativeOrbitNumber", "Value": 85}, {"Name": "orbitDirection", "Value": "ASCENDING"}],
            "GeoFootprint": {"type": "Polygon", "coordinates": [FP]}}
    cog = dict(item, Id="b", Name=item["Name"].replace(".SAFE", "_COG.SAFE"))
    s = parse_s1_items([item, cog])
    assert len(s) == 1 and s[0].rel_orbit == 85 and s[0].direction == "ASCENDING"


def test_before_date_moves_the_before_image_back(cfg):
    """Flood that built up over weeks: with --before-date the before-image must be on/before that day,
    while the after-image stays the earliest one on/after the flood date."""
    import pytest

    from floodsight.core import event_start, osm_snapshot_date
    scenes = [sc(d, 85) for d in ("2026-06-07", "2026-06-19", "2026-07-01", "2026-08-30", "2026-09-11")]
    c = {**cfg, "event": {"before_date": "2026-06-19"}}
    e = event_start(c, date(2026, 9, 11))
    assert e == date(2026, 6, 20)
    p = select_pairs(scenes, AOI_, date(2026, 9, 11), c, event_start=e)[0]
    assert p.pre.day == date(2026, 6, 19) and p.post.day == date(2026, 9, 11)
    # default unchanged: latest image before the flood date
    assert select_pairs(scenes, AOI_, date(2026, 9, 11), cfg)[0].pre.day == date(2026, 8, 30)
    assert event_start(cfg, date(2026, 9, 11)) == date(2026, 9, 11)
    # OSM snapshot follows the event start (pre-event map)
    assert osm_snapshot_date(e, "2026-10-01") == date(2026, 6, 19)
    with pytest.raises(ValueError):
        event_start({"event": {"before_date": "2026-09-11"}}, date(2026, 9, 11))


def test_single_polarisation_frames_are_skipped(cfg):
    """2017-01-12 asc-027 over Sri Lanka was '1SSV' (VV only): every backend failed on the missing VH.
    Such frames must never be chosen; the next dual-polarisation image is used instead."""
    def item(day, pol, orbit=27):
        return {"Id": day + pol, "Name": f"S1A_IW_GRDH_1S{pol}_{day.replace('-', '')}T124851_"
                f"{day.replace('-', '')}T124920_014799_0181A9_577B.SAFE",
                "ContentDate": {"Start": f"{day}T12:48:51.000Z"},
                "Attributes": [{"Name": "relativeOrbitNumber", "Value": orbit},
                               {"Name": "orbitDirection", "Value": "ASCENDING"}],
                "GeoFootprint": {"type": "Polygon", "coordinates": [FP]}}
    scenes = parse_s1_items([item("2016-12-31", "DV"), item("2017-01-12", "SV"), item("2017-01-20", "DH"),
                             item("2017-06-05", "DV")])
    assert [s.day.isoformat() for s in scenes] == ["2016-12-31", "2017-06-05"]
    p = select_pairs(scenes, AOI_, date(2017, 5, 30), {**cfg, "event": {"before_date": "2017-01-24"}},
                     event_start=date(2017, 1, 25))[0]
    assert p.pre.day == date(2016, 12, 31) and p.post.day == date(2017, 6, 5)
