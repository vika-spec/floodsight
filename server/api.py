"""FloodSight web API + static web app.

    python serve.py                      # http://127.0.0.1:8000  (opens the browser)

Routes (JSON unless noted):
  GET  /api/places?q=...           place name -> candidate places with a proposed analysis box
  GET  /api/status                 credentials present?, openEO login cached?, model weights?, presets, limits
  GET  /api/runs                   finished analyses, newest first (one line each)
  GET  /api/runs/{run}             results.json (+ validation.json if a reference comparison exists)
  GET  /api/runs/{run}/files/{p}   a file of that run (PNG, GeoJSON, PDF, GeoTIFF) - read-only, inside the run
  POST /api/runs/{run}/flow        {lon, lat, km} -> background job tracing the flood path from any point
  GET  /api/flows/{key}            the traced path, places along it, height bands, profile
  POST /api/jobs                   start an analysis {bbox, date, demo, name, resolution_m, backend, use_model, force}
  GET  /api/jobs/{id}              progress of an analysis
  GET  /                           the web app (web/dist)

Public deployments (FLOODSIGHT_PUBLIC=1): see server/public.py (access code for new analyses, daily caps).

Security: binds to 127.0.0.1 by default; credentials stay in .env on the server and are never sent to the
browser (status reports only yes/no); file access is limited to the runs folder and known file types; run
names are restricted to [A-Za-z0-9_-].
"""
from __future__ import annotations

import json
import os
import re
from datetime import date
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")   # reports are drawn in a worker thread

from fastapi import FastAPI, HTTPException, Request             # noqa: E402
from fastapi.exceptions import RequestValidationError           # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles                     # noqa: E402
from pydantic import BaseModel, Field                           # noqa: E402

from floodsight.core import AOI, ROOT, load_config, parse_date, run_id_for  # noqa: E402
from floodsight.presets import PRESETS                          # noqa: E402

from .jobs import JobManager                                    # noqa: E402
from .public import PublicGuard                                 # noqa: E402

SENTINEL1_START = date(2014, 10, 3)       # first Sentinel-1 GRD data
NAME_RE = re.compile(r"^[A-Za-z0-9_\-]{1,60}$")
SERVE_EXT = {".png": "image/png", ".jpg": "image/jpeg", ".json": "application/json",
             ".geojson": "application/geo+json", ".pdf": "application/pdf", ".tif": "image/tiff",
             ".csv": "text/csv", ".txt": "text/plain"}


class JobRequest(BaseModel):
    bbox: str | None = None
    date: str = "2026-08-26"
    before_date: str | None = None     # optional: last day the area was still dry (floods that built up for weeks)
    demo: bool = False
    name: str | None = None
    resolution_m: float = Field(20, ge=10, le=60)
    backend: str = Field("auto", pattern="^(auto|sentinelhub|openeo|local)$")
    use_model: bool = True
    force: bool = False
    access_code: str | None = Field(None, max_length=200)   # public deployments only; never stored
    # optional: the user's OWN Sentinel Hub keys for this one run (their account, their processing units).
    # Kept in memory until the run starts, then only for its duration; never stored, logged or returned.
    sh_client_id: str | None = Field(None, max_length=200)
    sh_client_secret: str | None = Field(None, max_length=400)


SECRET_FIELDS = {"access_code", "sh_client_id", "sh_client_secret"}
_RUN_KEYS: dict = {}            # job reference -> (client_id, secret), removed when the run starts


def client_ip(request: Request) -> str:
    """Visitor address; behind a hosting proxy (e.g. Hugging Face) the first X-Forwarded-For entry."""
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


class FlowRequest(BaseModel):
    lon: float = Field(..., ge=-180, le=180)
    lat: float = Field(..., ge=-85, le=85)
    km: float = Field(80, ge=5, le=150)


def _config(req: JobRequest) -> dict:
    ov = {"grid": {"resolution_m": req.resolution_m},
          "sentinel1": {"backend": req.backend},
          "model": {"enabled": req.use_model}}
    if req.before_date and not req.demo:
        ov["event"] = {"before_date": req.before_date}
    return load_config(overrides=ov)


def _check_summary(d: Path) -> dict | None:
    """Headline numbers of a run's comparison with a reference map, for the start page's example cards."""
    v = d / "validation" / "validation.json"
    try:
        if v.exists():
            V = json.loads(v.read_text(encoding="utf-8"))
            c = V.get("flood_extent") or {}
            return {"reference": V.get("reference"), "iou": c.get("iou"), "precision": c.get("precision")}
        K = _label_check(d)
        if K:
            return {"reference": K["reference"], "iou": K["iou"], "precision": K["precision"]}
    except Exception:  # noqa: BLE001 - an old or half-written file just shows no numbers
        return None
    return None


KS_VARIANT = "final map: flood water + debris"


def _label_check(d: Path) -> dict | None:
    """A run's comparison with Kuro Siwo hand labels (validation/kurosiwo_<event>.json), if one was made.
    The whole flagged map (flood water + debris) is reported, the same as for the EMS comparison."""
    for f in sorted((d / "validation").glob("kurosiwo_*.json")):
        K = json.loads(f.read_text(encoding="utf-8"))
        s = (K.get("scores") or {}).get(KS_VARIANT)
        if not s:
            continue
        role = K.get("model_role_for_event")
        return {"reference": f"Kuro Siwo hand labels (event {K.get('event')})", "iou": s.get("iou"),
                "precision": s.get("precision"), "recall": s.get("recall"), "variant": KS_VARIANT,
                "model_role": role if isinstance(role, str) else None}
    return None


def _default_runner(runs_dir: Path):
    def runner(params: dict, progress):
        from floodsight.fetch.shub import use_credentials
        from floodsight.pipeline import run
        req = JobRequest(**{k: v for k, v in params.items() if not k.startswith("_")})
        cred = _RUN_KEYS.pop(params.get("_keys_ref"), None)        # own keys: used for this run only
        with use_credentials(cred):
            return run(req.bbox, req.date, _config(req), progress=progress, demo=req.demo,
                       name=params["_run_id"], runs_dir=runs_dir)
    return runner


def _default_flow_runner(runs_dir: Path):
    def runner(params: dict, progress):
        from floodsight.analysis.flowtrace import flow_trace, run_readers
        d = runs_dir / params["run"]
        R = json.loads((d / "results.json").read_text(encoding="utf-8"))
        demo = any(str(n).startswith("DEMO MODE") for n in R.get("notes", []))
        out = flow_trace(params["lon"], params["lat"], load_config(), params["km"], parse_date(R["flood_date"]),
                         d, progress, **(run_readers(d) if demo else {}))
        out["run"], out["key"] = params["run"], params["_key"]
        f = runs_dir / "_flows" / f"{params['_key']}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(out), encoding="utf-8")
        return out
    return runner


def _metered(runner, guard: PublicGuard):
    """Public mode: count each real analysis and its Sentinel Hub processing units in the daily ledger."""
    def wrapped(params: dict, progress):
        out = runner(params, progress)
        if guard.enabled and not params.get("demo") and not params.get("_own_keys"):
            pu = (out or {}).get("inputs", {}).get("sentinelhub_processing_units", 0) if isinstance(out, dict) else 0
            guard.record(pu)
        return out
    return wrapped


def create_app(runs_dir: str | Path | None = None, runner=None, web_dir: str | Path | None = None,
               flow_runner=None) -> FastAPI:
    runs = Path(runs_dir or ROOT / "runs")
    runs.mkdir(parents=True, exist_ok=True)
    web = Path(web_dir or ROOT / "web" / "dist")
    guard = PublicGuard(runs)
    jobs = JobManager(_metered(runner or _default_runner(runs), guard))
    flow_jobs = JobManager(flow_runner or _default_flow_runner(runs))   # own worker: a path never waits for a run
    app = FastAPI(title="FloodSight", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
    app.state.jobs = jobs

    def run_dir(run: str) -> Path:
        if not NAME_RE.match(run):
            raise HTTPException(404, "Unknown run")
        d = runs / run
        if not (d / "results.json").exists():
            raise HTTPException(404, f"No finished analysis called '{run}'")
        return d

    # ------------------------------------------------------------------------------------------- status
    @app.get("/api/model-card")
    def model_card():
        """What the AI model is and how it did on recorded tests (docs/model_card.json), for the dashboard."""
        f = Path(__file__).resolve().parent.parent / "docs" / "model_card.json"
        if not f.exists():
            raise HTTPException(404, "no model card")
        return json.loads(f.read_text(encoding="utf-8"))

    @app.get("/api/status")
    def status(request: Request):
        cfg = load_config()
        weights = ROOT / cfg["model"]["weights"]
        gpu = None
        try:
            import torch
            gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else False
        except Exception:  # noqa: BLE001
            gpu = None
        login = None
        try:
            from openeo.rest.auth.config import RefreshTokenStore
            login = bool(RefreshTokenStore().load())
        except Exception:  # noqa: BLE001
            login = None
        return {
            "version": "1.0.0",
            "checks": {
                "copernicus_account": bool(os.environ.get("CDSE_USERNAME") and os.environ.get("CDSE_PASSWORD")),
                "sentinelhub_key": bool(os.environ.get("SH_CLIENT_ID", "").strip() and os.environ.get("SH_CLIENT_SECRET", "").strip()),
                "openeo_login": login,
                "osm_key": bool(os.environ.get("OHSOME_API_KEY", "").strip()),
                "model_weights": weights.exists(),
                "gpu": gpu,
            },
            "limits": {"max_area_km2": cfg["grid"]["max_area_km2"], "min_date": SENTINEL1_START.isoformat(),
                       "max_date": date.today().isoformat(), "batch_above_km2": cfg["sentinel1"].get("sync_max_km2", 400)},
            "presets": [{"name": k, **v} for k, v in PRESETS.items()],
            "public": guard.status(client_ip(request)),
            "busy": [j.public(0) for j in jobs.active()],
        }

    # ------------------------------------------------------------------------------------------- runs
    @app.get("/api/runs")
    def list_runs():
        out = []
        for d in runs.iterdir() if runs.exists() else []:
            f = d / "results.json"
            if not f.exists():
                continue
            try:
                R = json.loads(f.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001 - half-written or old file
                continue
            out.append({
                "run_id": d.name, "flood_date": R.get("flood_date"), "bbox": R.get("aoi", {}).get("bbox"),
                "area_km2": R.get("aoi", {}).get("area_km2"), "verdict": R.get("verdict"),
                "generated_utc": R.get("generated_utc"), "mtime": f.stat().st_mtime,
                "demo": any(str(n).startswith("DEMO MODE") for n in R.get("notes", [])),
                "cut_off": R.get("access", {}).get("settlements_cut_off"),
                "debris_km2": R.get("areas", {}).get("debris_km2"), "flood_km2": R.get("areas", {}).get("flood_km2"),
                "validated": (d / "validation" / "validation.json").exists(),
                "check": _check_summary(d),
            })
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    @app.get("/api/runs/{run}")
    def get_run(run: str):
        d = run_dir(run)
        R = json.loads((d / "results.json").read_text(encoding="utf-8"))
        v = d / "validation" / "validation.json"
        if v.exists():
            R["validation"] = json.loads(v.read_text(encoding="utf-8"))
            R["validation"]["figure"] = "validation/comparison.png" if (d / "validation" / "comparison.png").exists() else None
        else:
            try:
                K = _label_check(d)
            except Exception:  # noqa: BLE001
                K = None
            if K:
                R["label_check"] = K
        return R

    @app.get("/api/runs/{run}/files/{path:path}")
    def get_file(run: str, path: str):
        d = run_dir(run).resolve()
        p = (d / path).resolve()
        if d not in p.parents or not p.is_file() or p.suffix.lower() not in SERVE_EXT:
            raise HTTPException(404, "File not found")
        return FileResponse(p, media_type=SERVE_EXT[p.suffix.lower()], filename=p.name if p.suffix in (".pdf", ".tif") else None,
                            headers={"Cache-Control": "no-cache"})

    # ------------------------------------------------------------------------------------------- flood path
    flows = runs / "_flows"

    @app.post("/api/runs/{run}/flow")
    def start_flow(run: str, req: FlowRequest):
        run_dir(run)
        import hashlib
        key = f"{run[:40]}_" + hashlib.sha1(f"{req.lon:.4f},{req.lat:.4f},{req.km:g}".encode()).hexdigest()[:12]
        if (flows / f"{key}.json").exists():
            return {"reused": True, "key": key, "job": None}
        job = flow_jobs.submit(key, {"run": run, "lon": req.lon, "lat": req.lat, "km": req.km, "_key": key})
        return {"reused": False, "key": key, "job": job.public()}

    @app.get("/api/flows/{key}")
    def get_flow(key: str):
        if not NAME_RE.match(key):
            raise HTTPException(404, "Unknown flood path")
        f = flows / f"{key}.json"
        if not f.exists():
            raise HTTPException(404, "Unknown flood path")
        return json.loads(f.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------------------------------- places
    @app.get("/api/places")
    def places(q: str = ""):
        """Place name -> candidates with a proposed box (OpenStreetMap Nominatim). Only helps choose the area."""
        from floodsight.fetch.geocode import search_places
        if len(q.strip()) < 2:
            raise HTTPException(422, "Type a place name, e.g. 'Betrawati, Nuwakot, Nepal'.")
        if len(q) > 200:
            raise HTTPException(422, "That place name is too long.")
        cfg = load_config()
        max_km2 = min(cfg["grid"]["max_area_km2"], guard.max_km2) if guard.enabled else cfg["grid"]["max_area_km2"]
        try:
            return {"results": search_places(q, max_km2),
                    "attribution": "Place search: Nominatim, data © OpenStreetMap contributors"}
        except RuntimeError as e:
            raise HTTPException(503, str(e)) from e

    # ------------------------------------------------------------------------------------------- jobs
    @app.post("/api/jobs")
    def start_job(req: JobRequest, request: Request):
        cfg = load_config()
        try:
            d = parse_date(req.date)
        except ValueError:
            raise HTTPException(422, "Date must look like 2026-08-26.") from None
        if not req.demo:
            if d < SENTINEL1_START:
                raise HTTPException(422, f"Sentinel-1 data starts on {SENTINEL1_START:%d %b %Y}; pick a later date.")
            if d > date.today():
                raise HTTPException(422, "That date is in the future; pick the day the flood happened.")
            if req.before_date:
                try:
                    b = parse_date(req.before_date)
                except ValueError:
                    raise HTTPException(422, "The 'still dry on' date must look like 2026-08-20.") from None
                if b >= d:
                    raise HTTPException(422, "The 'still dry on' date must be earlier than the flood date.")
                if b < SENTINEL1_START:
                    raise HTTPException(422, f"Sentinel-1 data starts on {SENTINEL1_START:%d %b %Y}.")
            if not req.bbox:
                raise HTTPException(422, "Draw an area on the map or type its corners first.")
            try:
                aoi = AOI.parse(req.bbox)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            if aoi.area_km2() > cfg["grid"]["max_area_km2"]:
                raise HTTPException(422, f"The area is {aoi.area_km2():.0f} km²; the limit is "
                                         f"{cfg['grid']['max_area_km2']} km². Draw a smaller box around the valley.")
        own = bool((req.sh_client_id or "").strip() and (req.sh_client_secret or "").strip())
        ip = client_ip(request)
        if guard.enabled and not req.demo:
            refusal = guard.check(req.access_code, AOI.parse(req.bbox).area_km2() if req.bbox else None,
                                  ip=ip, own_keys=own)
            if refusal:
                raise HTTPException(403, refusal)
            req.resolution_m, req.backend, req.use_model = 20, "auto", True   # fixed on the public server
        if req.name and not NAME_RE.match(req.name):
            raise HTTPException(422, "Names can use letters, numbers, - and _ only.")
        run_id = req.name or (f"demo_{d}" if req.demo else run_id_for(AOI.parse(req.bbox), d, _config(req)))
        if not req.force and (runs / run_id / "results.json").exists():
            return {"reused": True, "run_id": run_id, "job": None}
        params = req.model_dump(exclude=SECRET_FIELDS)    # codes and keys are never stored or shown in job status
        params["_run_id"] = run_id
        if own and not req.demo:
            import secrets
            ref = secrets.token_hex(8)
            _RUN_KEYS[ref] = (req.sh_client_id.strip(), req.sh_client_secret.strip())
            params["_keys_ref"], params["_own_keys"] = ref, True
        job = jobs.submit(run_id, params)
        if job.params is not params:                      # joined an identical queued run: drop unused keys
            _RUN_KEYS.pop(params.get("_keys_ref"), None)
        if guard.enabled and not req.demo and not own and job.params is params:
            guard.reserve_ip(ip)                          # counts against this visitor's runs on our account
        return {"reused": False, "run_id": run_id, "job": job.public(), "queue_position": jobs.queue_position(job)}

    @app.get("/api/jobs/{jid}")
    def get_job(jid: str):
        job = jobs.get(jid) or flow_jobs.get(jid)
        if job is None:
            raise HTTPException(404, "Unknown job (the server may have restarted)")
        mgr = jobs if jobs.get(jid) else flow_jobs
        return {**job.public(), "queue_position": mgr.queue_position(job)}

    # ------------------------------------------------------------------------------------------- web app
    if (web / "index.html").exists():
        app.mount("/", StaticFiles(directory=web, html=True), name="web")
    else:
        @app.get("/", response_class=HTMLResponse)
        def missing_ui():
            return ("<h2>FloodSight API is running</h2><p>The web app is not built yet: "
                    "<code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>. API docs: "
                    "<a href='/api/docs'>/api/docs</a></p>")

    @app.exception_handler(HTTPException)
    def http_error(_, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    def bad_request(_, exc: RequestValidationError):
        first = exc.errors()[0] if exc.errors() else {}
        where = ".".join(str(x) for x in first.get("loc", [])[1:])
        return JSONResponse({"error": f"{where}: {first.get('msg', 'invalid value')}"}, status_code=422)

    return app
