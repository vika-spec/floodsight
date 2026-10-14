import json


def test_demo_end_to_end(tmp_path, cfg):
    from floodsight.pipeline import run
    r = run(None, "2026-08-26", cfg, demo=True, runs_dir=tmp_path)
    d = tmp_path / r["run_id"]
    saved = json.loads((d / "results.json").read_text())
    assert saved["verdict"] == "flood_detected"
    assert saved["access"]["settlements_cut_off"] == 2   # blocked by confident flood water
    assert saved["access"]["settlements_likely_cut_off"] == 2   # single-track debris only
    for k in ("en", "ne", "map"):
        assert (d / saved["reports"][k]).exists()
    for k in ("flood", "before", "after"):
        assert (d / saved["web"][k]["png"]).exists()
    assert "Contains modified Copernicus Sentinel data 2026." in saved["attributions"]

