"""Background analysis jobs for the web app.

Design:
  * ONE worker thread runs analyses in order. The pipeline already downloads in parallel internally, and
    Copernicus openEO allows only a couple of concurrent requests per account, so running two analyses at
    once would make both slower and risk rate limits.
  * Submitting the same area + date + settings while it is queued or running returns the SAME job
    (idempotent: a double click or a page reload never starts a second download).
  * A finished run is a folder runs/<run_id>/ with results.json. Opening it again costs nothing; running it
    again reuses every downloaded input (cache), so only the analysis is repeated.
  * Progress = the pipeline's stage callbacks + the last log lines, kept in memory and polled by the browser
    (GET /api/jobs/<id>). Polling survives page reloads and proxies; no socket to keep alive.
  * Failures never take the server down: they become a job state with a plain-language message.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field

log = logging.getLogger("floodsight")


@dataclass
class Job:
    id: str
    run_id: str
    params: dict
    state: str = "queued"            # queued | running | done | failed
    stage: str = "Waiting to start"
    frac: float = 0.0
    stages: list = field(default_factory=list)   # [{name, start, end}]
    created: float = field(default_factory=time.time)
    started: float | None = None
    finished: float | None = None
    error: str | None = None
    error_kind: str | None = None    # input | no_data | error
    log: deque = field(default_factory=lambda: deque(maxlen=300))
    activity: dict = field(default_factory=dict)   # data requests: which service, what, how long (no secrets)

    def public(self, log_lines: int = 60) -> dict:
        now = time.time()
        return {
            "id": self.id, "run_id": self.run_id, "state": self.state,
            "params": {k: v for k, v in self.params.items() if k != "_keys_ref"},
            "stage": self.stage, "progress": round(self.frac, 3), "stages": self.stages,
            "elapsed_s": round((self.finished or now) - (self.started or now), 1) if self.started else 0,
            "queued_s": round((self.started or now) - self.created, 1),
            "error": self.error, "error_kind": self.error_kind,
            "log": list(self.log)[-log_lines:], "server_time": now,
            "activity": list(self.activity.values())[-60:],
        }


class _JobLogHandler(logging.Handler):
    """Copies pipeline log lines into the running job (one job runs at a time)."""

    def __init__(self, manager: "JobManager"):
        super().__init__(logging.INFO)
        self.m = manager
        self.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))

    def emit(self, record):
        job = self.m.current
        if job is not None:
            try:
                msg = self.format(record)
                if record.levelno >= logging.WARNING:
                    msg = msg.replace(" ", " ⚠ ", 1)
                job.log.append(msg)
            except Exception:  # noqa: BLE001
                pass


class JobManager:
    def __init__(self, runner):
        """runner(params, progress_cb) -> results dict. Injected so tests can use the demo pipeline."""
        self.runner = runner
        self.jobs: dict[str, Job] = {}
        self.by_run: dict[str, str] = {}
        self.q: queue.Queue[str] = queue.Queue()
        self.current: Job | None = None
        self.lock = threading.Lock()
        fl = logging.getLogger("floodsight")
        if fl.getEffectiveLevel() > logging.INFO:      # progress lines are INFO; make sure they reach the job
            fl.setLevel(logging.INFO)
        fl.addHandler(_JobLogHandler(self))
        from floodsight import activity

        def sink(d):                                   # one job runs at a time
            job = self.current
            if job is not None:
                job.activity[d["id"]] = d
        activity.add_sink(sink)
        self.worker = threading.Thread(target=self._loop, name="floodsight-worker", daemon=True)
        self.worker.start()

    # -------------------------------------------------------------------------------------------- public
    def submit(self, run_id: str, params: dict) -> Job:
        with self.lock:
            jid = self.by_run.get(run_id)
            if jid and self.jobs[jid].state in ("queued", "running"):
                return self.jobs[jid]                     # idempotent
            job = Job(id=uuid.uuid4().hex[:12], run_id=run_id, params=params)
            self.jobs[job.id] = job
            self.by_run[run_id] = job.id
            self._trim()
        self.q.put(job.id)
        return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)

    def active(self) -> list[Job]:
        return [j for j in self.jobs.values() if j.state in ("queued", "running")]

    def queue_position(self, job: Job) -> int:
        waiting = sorted((j for j in self.jobs.values() if j.state == "queued"), key=lambda j: j.created)
        return next((i + 1 for i, j in enumerate(waiting) if j.id == job.id), 0)

    # -------------------------------------------------------------------------------------------- worker
    def _trim(self, keep=50):
        done = sorted((j for j in self.jobs.values() if j.state in ("done", "failed")), key=lambda j: j.created)
        for j in done[:-keep] if len(done) > keep else []:
            self.jobs.pop(j.id, None)

    def _progress(self, job: Job):
        def cb(stage: str, frac: float):
            now = time.time()
            if job.stages and job.stages[-1]["end"] is None:
                job.stages[-1]["end"] = now
            if stage != "Done":
                job.stages.append({"name": stage, "start": now, "end": None})
            job.stage, job.frac = stage, max(job.frac, float(frac))
        return cb

    def _loop(self):
        from floodsight.fetch.shub import ShubQuotaExceeded
        from floodsight.pipeline import NoData

        while True:
            jid = self.q.get()
            job = self.jobs.get(jid)
            if job is None:
                continue
            job.state, job.started, self.current = "running", time.time(), job
            try:
                self.runner(job.params, self._progress(job))
                job.state, job.frac, job.stage = "done", 1.0, "Done"
            except NoData as e:
                job.state, job.error_kind, job.error = "failed", "no_data", str(e)
            except ShubQuotaExceeded as e:
                job.state, job.error_kind = "failed", "quota"
                job.error = (f"{e} You can run it with your own free Copernicus account: enter your Sentinel Hub "
                             "keys under 'Use my own Copernicus keys'.")
            except ValueError as e:
                job.state, job.error_kind, job.error = "failed", "input", str(e)
            except Exception as e:  # noqa: BLE001 - report, keep serving
                job.state, job.error_kind = "failed", "error"
                job.error = f"{type(e).__name__}: {e}"
                for line in traceback.format_exc().splitlines()[-12:]:
                    job.log.append(line)
                log.error("Job %s failed: %s", job.id, job.error)
            finally:
                job.finished = time.time()
                if job.stages and job.stages[-1]["end"] is None:
                    job.stages[-1]["end"] = job.finished
                self.current = None
