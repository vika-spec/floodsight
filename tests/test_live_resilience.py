"""Live-evaluation robustness: odd inputs and failing data services must give a clear answer, not a crash."""
from datetime import timedelta

import pytest


def _fake_fetch(empty_osm=False, two_tracks=False):
    from floodsight.fetch.osm import empty_osm as _empty
    from floodsight.synthetic import make_scene

    def fetch(ctx):
        sc = make_scene(ctx.aoi, ctx.cfg["grid"]["resolution_m"])
        pair = {"track": "asc-085", "pre_date": "2026-08-16", "post_date": "2026-08-28", "gap_days": 12,
                "coverage": 1.0, "incidence_deg": 38.0, "heading_deg": -12.0, "pre_scenes": [], "post_scenes": []}
        pd = [{"pair": pair, "pre": sc["pre"], "post": sc["post"], "backend": "openeo",
               "incidence": 38.0, "heading": -12.0}]
        if two_tracks:
            pd.append({**pd[0], "pair": {**pair, "track": "des-019"}, "incidence": 40.0, "heading": 192.0})
        osm = _empty("2026-07-27") if empty_osm else sc["osm"]
        if empty_osm:
            ctx.note("OpenStreetMap could not be downloaded, so damage and cut-off analysis are missing from this run; "
                     "the flood map is complete. Re-run later to add them.")
        return sc["dem"], osm, None, pd
    return fetch


def test_run_completes_without_osm(tmp_path, cfg, monkeypatch):
    import floodsight.pipeline as P
    from floodsight.synthetic import DEFAULT_AOI
    monkeypatch.setattr(P, "_fetch_all", _fake_fetch(empty_osm=True))
    r = P.run(DEFAULT_AOI, "2026-08-26", cfg, runs_dir=tmp_path, name="noosm")
    assert r["verdict"] == "flood_detected"
    assert any("OpenStreetMap could not be downloaded" in n for n in r["notes"])
    assert (tmp_path / "noosm" / r["reports"]["en"]).exists()


def test_two_tracks_are_merged(tmp_path, cfg, monkeypatch):
    import floodsight.pipeline as P
    from floodsight.synthetic import DEFAULT_AOI
    monkeypatch.setattr(P, "_fetch_all", _fake_fetch(two_tracks=True))
    r = P.run(DEFAULT_AOI, "2026-08-26", cfg, runs_dir=tmp_path, name="two")
    assert len(r["inputs"]["detection_stats"]) == 2
    assert r["access"]["settlements_cut_off"] >= 1


def test_too_large_area_is_refused_clearly(cfg, tmp_path):
    from floodsight.pipeline import run
    with pytest.raises(ValueError, match="km²"):
        run("80,20,82,22", "2026-08-26", cfg, runs_dir=tmp_path)


@pytest.mark.parametrize("bbox", ["85.4,27.9,85.1,28.2", "85,28,86", "abc"])
def test_bad_bbox_is_refused_clearly(cfg, tmp_path, bbox):
    from floodsight.pipeline import run
    with pytest.raises(ValueError):
        run(bbox, "2026-08-26", cfg, runs_dir=tmp_path)


def test_no_satellite_pair_gives_clear_message(cfg, tmp_path, monkeypatch):
    import floodsight.fetch.catalog as C
    from floodsight.pipeline import NoData, run
    monkeypatch.setattr(C, "search_s1", lambda *a, **k: [])
    with pytest.raises(NoData, match="same orbit track"):
        run("85.25,27.95,85.35,28.05", "2026-08-26", cfg, runs_dir=tmp_path)


def test_one_track_failing_does_not_stop_the_run(cfg, tmp_path, monkeypatch):
    """Two tracks found, one fails in processing: the run continues with the other and says so."""
    from datetime import date, datetime

    import floodsight.fetch.catalog as C
    import floodsight.fetch.dem as D
    import floodsight.fetch.osm as O
    import floodsight.fetch.s1 as S1
    import floodsight.fetch.s2 as S2
    import floodsight.pipeline as P
    from floodsight.fetch.catalog import Scene
    from floodsight.synthetic import DEFAULT_AOI, make_scene

    sc = make_scene()
    fp = [[85.0, 27.5], [86.0, 27.5], [86.0, 28.5], [85.0, 28.5], [85.0, 27.5]]
    scenes = [Scene(f"{d}{o}", f"S1A_IW_GRDH_x_{d}", datetime.fromisoformat(d), "S1A", o, dr, fp)
              for d, o, dr in (("2026-08-16", 85, "ASCENDING"), ("2026-08-28", 85, "ASCENDING"),
                               ("2026-08-24", 19, "DESCENDING"), ("2026-09-05", 19, "DESCENDING"))]
    monkeypatch.setattr(C, "search_s1", lambda *a, **k: scenes)
    monkeypatch.setattr(D, "fetch_dem", lambda ctx: sc["dem"])
    monkeypatch.setattr(O, "fetch_osm", lambda ctx: sc["osm"])
    monkeypatch.setattr(S2, "fetch_s2", lambda ctx: None)

    def fake_pair(ctx, pair, dem_ready=None):
        if pair.pre.direction == "DESCENDING":
            raise RuntimeError("[503] no available server")
        return {"pre": sc["pre"], "post": sc["post"], "backend": "openeo"}
    monkeypatch.setattr(S1, "fetch_pair", fake_pair)
    r = P.run(DEFAULT_AOI, "2026-08-26", cfg, runs_dir=tmp_path, name="onefail")
    assert len(r["inputs"]["sentinel1_pairs"]) == 1
    assert any("could not be processed" in n for n in r["notes"])
