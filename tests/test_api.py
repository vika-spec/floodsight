"""Web API: start an analysis, follow its progress, open the results, trace a flow path, and refuse bad input."""
import time

import pytest


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from fastapi.testclient import TestClient

    from server.api import create_app
    runs = tmp_path_factory.mktemp("runs")
    return TestClient(create_app(runs_dir=runs, web_dir=runs / "noweb"))


def _wait(client, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = client.get(f"/api/jobs/{jid}").json()
        if j["state"] in ("done", "failed"):
            return j
        time.sleep(0.5)
    raise AssertionError("job did not finish")


def test_demo_job_end_to_end(client):
    st = client.get("/api/status").json()
    assert "checks" in st and st["presets"]
    assert "copernicus_account" in st["checks"] and isinstance(st["checks"]["copernicus_account"], bool)

    r = client.post("/api/jobs", json={"demo": True, "date": "2026-08-26", "force": True}).json()
    again = client.post("/api/jobs", json={"demo": True, "date": "2026-08-26", "force": True}).json()
    assert again["job"]["id"] == r["job"]["id"]                      # double click -> same job
    j = _wait(client, r["job"]["id"])
    assert j["state"] == "done", j
    assert any("Terrain" in s["name"] for s in j["stages"])
    assert j["log"]

    runs = client.get("/api/runs").json()
    assert runs[0]["run_id"] == r["run_id"] and runs[0]["demo"]
    R = client.get(f"/api/runs/{r['run_id']}").json()
    assert R["access"]["settlements_cut_off"] == 2   # blocked by confident flood water
    assert R["access"]["settlements_likely_cut_off"] == 2   # single-track debris only
    png = client.get(f"/api/runs/{r['run_id']}/files/{R['web']['flood']['png']}")
    assert png.status_code == 200 and png.headers["content-type"] == "image/png"
    assert client.get(f"/api/runs/{r['run_id']}/files/{R['reports']['en']}").status_code == 200

    # same request without force: open the existing result instead of recomputing
    reuse = client.post("/api/jobs", json={"demo": True, "date": "2026-08-26"}).json()
    assert reuse["reused"] and reuse["job"] is None

    s = R["settlements"][0]
    # flood path from a point: background job, then the stored result (demo run: offline readers)
    fr = client.post(f"/api/runs/{r['run_id']}/flow", json={"lon": R["aoi"]["bbox"][0] + 0.02,
                                                              "lat": R["aoi"]["center"][1], "km": 20}).json()
    j = _wait(client, fr["job"]["id"])
    assert j["state"] == "done", j
    flow = client.get(f"/api/flows/{fr['key']}").json()
    assert flow["length_km"] > 3 and flow["path"]["geometry"]["coordinates"]
    again = client.post(f"/api/runs/{r['run_id']}/flow", json={"lon": R["aoi"]["bbox"][0] + 0.02,
                                                                 "lat": R["aoi"]["center"][1], "km": 20}).json()
    assert again["reused"] and again["key"] == fr["key"]
    assert client.post(f"/api/runs/{r['run_id']}/flow", json={"lon": 500, "lat": 0}).status_code == 422
    assert s


def test_file_access_is_confined(client):
    runs = client.get("/api/runs").json()
    rid = runs[0]["run_id"]
    for bad in ("../../config.yaml", "..%2F..%2Fconfig.yaml", "results.py", "/etc/passwd"):
        assert client.get(f"/api/runs/{rid}/files/{bad}").status_code == 404
    assert client.get("/api/runs/..%2F..").status_code == 404


@pytest.mark.parametrize("body,msg", [
    ({"bbox": "1,2,3", "date": "2026-08-26"}, "bbox"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2099-01-01"}, "future"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2012-01-01"}, "Sentinel-1"),
    ({"bbox": "80,25,90,30", "date": "2026-08-26"}, "limit"),
    ({"date": "2026-08-26"}, "Draw an area"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "26/08/2026"}, "Date"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2026-08-26", "name": "../x"}, "letters"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2026-08-26", "backend": "magic"}, "backend"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2026-08-26", "before_date": "2026-08-26"}, "earlier"),
    ({"bbox": "85.1,27.9,85.2,28.0", "date": "2026-08-26", "before_date": "20/08/2026"}, "still dry"),
])

def test_bad_input_is_explained(client, body, msg):
    r = client.post("/api/jobs", json=body)
    assert r.status_code == 422
    assert msg in r.json()["error"]


def test_before_date_changes_run_identity():
    from server.api import JobRequest, _config
    from floodsight.core import AOI, run_id_for
    a = JobRequest(bbox="85.1,27.9,85.2,28.0", date="2026-08-26")
    b = JobRequest(bbox="85.1,27.9,85.2,28.0", date="2026-08-26", before_date="2026-08-01")
    aoi = AOI.parse(a.bbox)
    from datetime import date
    assert _config(b)["event"]["before_date"] == "2026-08-01"
    assert run_id_for(aoi, date(2026, 8, 26), _config(a)) != run_id_for(aoi, date(2026, 8, 26), _config(b))
