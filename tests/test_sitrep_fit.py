"""The situation report is ONE page (the brief asks for a one-page sitrep), even for a busy area:
the layout tightens (smaller map, fewer rows) until all content sits above the footer."""
import copy


def test_busy_area_still_one_page(tmp_path, cfg):
    from floodsight.pipeline import run
    from floodsight.report.sitrep import FOOTER_Y, LAYOUTS, _render, build_pdf

    r = run(None, "2026-08-26", cfg, demo=True, runs_dir=tmp_path / "runs")
    run_dir = tmp_path / "runs" / r["run_id"]
    big = copy.deepcopy(r)
    base = [s for s in r["settlements"] if s["status"] == "cut_off"][0]
    big["settlements"] = [dict(base, name=f"Very long settlement name number {i}", priority=i + 1) for i in range(40)]
    road = {"name": "Pasang Lhamu Highway (F21) between two villages", "highway": "primary", "km_affected": 1.25,
            "certain": False, "lat_lon": [28.1, 85.3]}
    big["blockages"] = {"roads": [road] * 12, "bridges": [{"name": "A suspension bridge with a long name",
                        "highway": "path", "lat_lon": [28.1, 85.3]}] * 12, "roads_total": 30, "bridges_total": 20}
    heli = big["helicopter_access"]
    heli["per_settlement"] = [{"name": f"Village {i}", "status": "cut_off", "priority": i,
                               "helipad": {"name": "District helipad", "km": 3.2, "lat_lon": [28.0, 85.3]},
                               "zone": {"km": 0.4, "lat_lon": [28.0, 85.3], "clear_radius_m": 60, "slope_deg": 3}}
                              for i in range(30)]
    big["notes"] = ["A long system note that explains a caveat in detail, repeated to fill space. " * 3] * 6
    map_png = run_dir / "report" / "map.png"
    for lang in ("en", "ne"):
        _, bottom_loose = _render(big, lang, map_png, LAYOUTS[0])
        assert bottom_loose > FOOTER_Y                       # the busy case really overflows the first layout
        out = build_pdf(big, run_dir, lang, map_png)
        assert out.exists() and out.read_bytes().count(b"/Type /Page\n") <= 1 and b"/Count 1" in out.read_bytes()
    fitted = [b for b in (_render(big, "en", map_png, lay)[1] for lay in LAYOUTS) if b <= FOOTER_Y]
    assert fitted, "some layout must fit"
