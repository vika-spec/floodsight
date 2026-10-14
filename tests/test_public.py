"""Public deployment guard: anyone can view and run the demo; new analyses need the code and respect caps;
the access code is never echoed in job status."""
import pytest


@pytest.fixture()
def pub_client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from server.api import create_app
    monkeypatch.setenv("FLOODSIGHT_PUBLIC", "1")
    monkeypatch.setenv("FLOODSIGHT_ACCESS_CODE", "judges-2026")
    monkeypatch.setenv("FLOODSIGHT_MAX_KM2", "500")
    monkeypatch.setenv("FLOODSIGHT_MAX_RUNS_DAY", "1")
    calls = []

    def fake_runner(params, progress):
        calls.append(params)
        return {"inputs": {"sentinelhub_processing_units": 123.0}}

    return TestClient(create_app(runs_dir=tmp_path, runner=fake_runner, web_dir=tmp_path / "noweb")), calls


BOX = "85.25,27.95,85.35,28.05"          # ~110 km2


def test_public_mode_rules(pub_client):
    import time
    c, calls = pub_client
    st = c.get("/api/status").json()
    assert st["public"]["enabled"] and st["public"]["access_required"]

    r = c.post("/api/jobs", json={"bbox": BOX, "date": "2026-08-26"})
    assert r.status_code == 403 and "access code" in r.json()["error"]
    r = c.post("/api/jobs", json={"bbox": BOX, "date": "2026-08-26", "access_code": "wrong"})
    assert r.status_code == 403
    r = c.post("/api/jobs", json={"bbox": "85.0,27.9,85.3,28.1", "date": "2026-08-26", "access_code": "judges-2026"})
    assert r.status_code == 403 and "km²" in r.json()["error"]

    r = c.post("/api/jobs", json={"bbox": BOX, "date": "2026-08-26", "access_code": "judges-2026",
                                  "resolution_m": 10, "backend": "local"})
    assert r.status_code == 200, r.json()
    job = r.json()["job"]
    assert "access_code" not in job["params"]                          # never echoed
    for _ in range(50):
        if c.get(f"/api/jobs/{job['id']}").json()["state"] in ("done", "failed"):
            break
        time.sleep(0.1)
    assert calls[0]["resolution_m"] == 20 and calls[0]["backend"] == "auto"   # fixed on the public server
    u = c.get("/api/status").json()["public"]
    assert u["runs_today"] == 1 and u["pu_today"] == 123.0
    r = c.post("/api/jobs", json={"bbox": "85.20,27.95,85.30,28.05", "date": "2026-08-26", "access_code": "judges-2026"})
    assert r.status_code == 403 and "daily limit" in r.json()["error"]
    # the demo stays open to everyone
    assert c.post("/api/jobs", json={"demo": True, "date": "2026-08-26", "force": True}).status_code == 200


def test_local_mode_unchanged(tmp_path, monkeypatch):
    from server.public import PublicGuard
    monkeypatch.delenv("FLOODSIGHT_PUBLIC", raising=False)
    assert PublicGuard(tmp_path).check(None, 99999) is None


def _wait(c, jid):
    import time
    for _ in range(50):
        if c.get(f"/api/jobs/{jid}").json()["state"] in ("done", "failed"):
            return
        time.sleep(0.1)


def test_visitor_limit_and_own_keys(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from server import api as api_mod
    monkeypatch.setenv("FLOODSIGHT_PUBLIC", "1")
    monkeypatch.setenv("FLOODSIGHT_ACCESS_CODE", "judges-2026")
    monkeypatch.setenv("FLOODSIGHT_RUNS_PER_IP", "2")
    monkeypatch.setenv("FLOODSIGHT_MAX_RUNS_DAY", "100")
    calls = []

    def fake_runner(params, progress):
        calls.append(dict(params))
        return {"inputs": {"sentinelhub_processing_units": 10.0}}

    c = TestClient(api_mod.create_app(runs_dir=tmp_path, runner=fake_runner, web_dir=tmp_path / "noweb"))
    A, B = {"X-Forwarded-For": "1.2.3.4"}, {"X-Forwarded-For": "5.6.7.8"}
    boxes = ["85.25,27.95,85.35,28.05", "85.26,27.95,85.36,28.05", "85.27,27.95,85.37,28.05"]
    for b in boxes[:2]:
        r = c.post("/api/jobs", json={"bbox": b, "date": "2026-08-26", "access_code": "judges-2026"}, headers=A)
        assert r.status_code == 200, r.json()
        _wait(c, r.json()["job"]["id"])
    assert c.get("/api/status", headers=A).json()["public"]["runs_left_for_you"] == 0
    r = c.post("/api/jobs", json={"bbox": boxes[2], "date": "2026-08-26", "access_code": "judges-2026"}, headers=A)
    assert r.status_code == 403 and r.json()["error"].startswith("[own-keys]")      # app offers own keys
    r = c.post("/api/jobs", json={"bbox": boxes[2], "date": "2026-08-26", "access_code": "judges-2026"}, headers=B)
    assert r.status_code == 200                                                        # another visitor still can
    _wait(c, r.json()["job"]["id"])
    pu_before = c.get("/api/status").json()["public"]["pu_today"]
    # own keys: allowed past the visitor limit, never echoed, not counted against our account
    r = c.post("/api/jobs", json={"bbox": "85.28,27.95,85.38,28.05", "date": "2026-08-26", "access_code": "judges-2026",
                                  "sh_client_id": "THEIR-ID", "sh_client_secret": "THEIR-SECRET"}, headers=A)
    assert r.status_code == 200, r.json()
    body = r.text
    assert "THEIR-SECRET" not in body and "THEIR-ID" not in body and "_keys_ref" not in body
    _wait(c, r.json()["job"]["id"])
    assert calls[-1]["_own_keys"] is True and "sh_client_secret" not in calls[-1]
    assert c.get("/api/status").json()["public"]["pu_today"] == pu_before
    assert "THEIR-SECRET" not in c.get("/api/status").text
    for f in tmp_path.rglob("*"):                                  # nothing written to disk holds the keys
        if f.is_file():
            assert b"THEIR-SECRET" not in f.read_bytes()


def test_default_runner_uses_keys_only_during_the_run(tmp_path, monkeypatch):
    from floodsight.fetch import shub
    from server import api as api_mod
    seen = {}

    def fake_run(*a, **k):
        seen["cred"] = shub.credentials()
        return {}
    monkeypatch.setattr("floodsight.pipeline.run", fake_run)
    monkeypatch.setenv("SH_CLIENT_ID", "OURS")
    monkeypatch.setenv("SH_CLIENT_SECRET", "OURS-S")
    api_mod._RUN_KEYS["r1"] = ("THEIRS", "THEIRS-S")
    runner = api_mod._default_runner(tmp_path)
    runner({"bbox": "85.25,27.95,85.35,28.05", "date": "2026-08-26", "_run_id": "x", "_keys_ref": "r1"}, lambda *a: None)
    assert seen["cred"] == ("THEIRS", "THEIRS-S")
    assert "r1" not in api_mod._RUN_KEYS and shub.credentials() == ("OURS", "OURS-S")
    assert not shub.override_active()


def test_quota_detection_and_job_message():
    from floodsight.fetch.shub import _is_quota
    assert _is_quota(403, "Access denied: processing units limit exceeded")
    assert _is_quota(429, "Monthly quota exceeded")
    assert not _is_quota(429, "Too many requests per second")
    assert not _is_quota(429, "Rate limit: processing units limit per minute")       # wait and retry instead
    assert not _is_quota(500, "quota")
