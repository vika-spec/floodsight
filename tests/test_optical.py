"""Sentinel-2 as a second detector: adds optical-only water/debris at low confidence under the same physical
limits, confirms radar detections, and never adds debris outside the allowed (mountain, corridor) zone."""
import numpy as np

from floodsight.detect.baseline import DEBRIS, FLOOD, NO_CHANGE
from floodsight.detect.fuse import apply_optical
from floodsight.fetch.s2 import optical_evidence


def _img(g, r, n, s, shape):
    return np.stack([np.full(shape, v, np.float32) for v in (g, r, n, s)])


def test_optical_detector():
    shape = (60, 60)
    pre = _img(0.05, 0.04, 0.35, 0.20, shape)           # vegetated land: NDVI ~0.8, MNDWI < 0
    post = pre.copy()
    post[:, 5:20, 5:20] = np.array([0.08, 0.06, 0.04, 0.02])[:, None, None]    # new water: MNDWI > 0
    post[:, 30:45, 30:45] = np.array([0.10, 0.12, 0.14, 0.25])[:, None, None]  # bare mud: NDVI ~0.08
    opt = optical_evidence(pre, post)
    assert opt["new_water"][10, 10] and opt["debris"][35, 35]

    cls = np.full(shape, NO_CHANGE, np.uint8)
    conf = np.zeros(shape, np.uint8)
    cls[30:32, 30:45], conf[30:32, 30:45] = DEBRIS, 1                  # radar saw part of the debris
    everywhere = np.ones(shape, bool)
    out, c, st = apply_optical(cls, conf, opt, everywhere, everywhere, everywhere, min_px=4, detect=True)
    assert (out[5:20, 5:20] == FLOOD).all() and (c[5:20, 5:20] == 1).all()     # optical-only: low confidence
    assert (out[30:45, 30:45] == DEBRIS).all()
    assert (c[30:32, 30:45] == 2).all()                                  # both sensors agree: raised
    assert st["added_flood_px"] == 225 and st["confirmed_debris_px"] == 30

    # plains / outside the corridor: no optical debris; outside the flood zone: no optical water
    nowhere = np.zeros(shape, bool)
    out2, _, st2 = apply_optical(np.full(shape, NO_CHANGE, np.uint8), np.zeros(shape, np.uint8), opt,
                                 everywhere, nowhere, nowhere, detect=True)
    assert not ((out2 == FLOOD) | (out2 == DEBRIS)).any()

    # detect=False: confirm only, add nothing
    out3, _, st3 = apply_optical(np.full(shape, NO_CHANGE, np.uint8), np.zeros(shape, np.uint8), opt,
                                 everywhere, everywhere, everywhere, detect=False)
    assert not ((out3 == FLOOD) | (out3 == DEBRIS)).any() and "added_flood_px" not in st3


def test_pipeline_uses_optical_detector(cfg, tmp_path, monkeypatch):
    """Full run with a clear-sky Sentinel-2 pair: optical adds the flood the radar pair did not show."""
    from datetime import datetime

    import floodsight.fetch.catalog as C
    import floodsight.fetch.dem as D
    import floodsight.fetch.osm as O
    import floodsight.fetch.s1 as S1
    import floodsight.fetch.s2 as S2
    import floodsight.pipeline as P
    import floodsight.preprocess.mountain as MT
    from floodsight.fetch.catalog import Scene
    from floodsight.synthetic import DEFAULT_AOI, make_scene

    sc = make_scene()
    fp = [[85.0, 27.5], [86.0, 27.5], [86.0, 28.5], [85.0, 28.5], [85.0, 27.5]]
    scenes = [Scene(f"{d}", f"S1A_IW_GRDH_x_{d}", datetime.fromisoformat(d), "S1A", 85, "ASCENDING", fp)
              for d in ("2026-08-16", "2026-08-28")]
    monkeypatch.setattr(C, "search_s1", lambda *a, **k: scenes)
    monkeypatch.setattr(D, "fetch_dem", lambda ctx: sc["dem"])
    monkeypatch.setattr(O, "fetch_osm", lambda ctx: sc["osm"])
    monkeypatch.setattr(MT, "_read_coarse", lambda *a: (_ for _ in ()).throw(RuntimeError("offline")))
    H, W = sc["dem"].shape
    pre = _img(0.05, 0.04, 0.35, 0.20, (H, W))
    post = pre.copy()
    flood = sc["truth"] == 1
    for b, v in enumerate((0.08, 0.06, 0.04, 0.02)):
        post[b][flood] = v
    monkeypatch.setattr(S2, "fetch_s2", lambda ctx: {"pre": pre, "post": post})
    # the radar pair shows NO change at all: anything mapped as flood must come from the optical sensor
    monkeypatch.setattr(S1, "fetch_pair", lambda ctx, pair, dem_ready=None:
                        {"pre": sc["pre"], "post": sc["pre"], "backend": "openeo"})
    r = P.run(DEFAULT_AOI, "2026-08-26", cfg, runs_dir=tmp_path, name="optical")
    s2 = r["inputs"]["sentinel2"]
    assert s2["used"] and s2["added_flood_km2"] > 0.5
    assert r["areas"]["flood_km2"] >= s2["added_flood_km2"] * 0.99
    assert r["areas"]["flood_high_conf_km2"] == 0                      # optical-only = low confidence
