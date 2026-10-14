"""What the pipeline is waiting for, right now: one entry per data request (which service, what, how long).

Used so that nothing in a run is a black box: the web app shows these entries live under the running step
("Sentinel Hub · Sentinel-1 after image, track asc-085, 3 Sep 2026 · 41 s · 2 requests · 12.4 MB · 18 PU"),
and the command line prints them. Entries carry no secrets: only service names, dates, sizes and timings.

    with activity.task("Sentinel Hub", "Sentinel-1 after image, track asc-085, 2026-09-03"):
        ...                                  # inside: activity.note("server busy, retrying in 10 s")
    activity.cached("Copernicus DEM", "Elevation")   # reused from an earlier run, no download
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from contextlib import contextmanager

log = logging.getLogger("floodsight")

_ids = itertools.count(1)
_local = threading.local()
_sinks: list = []
_lock = threading.Lock()


class Task:
    __slots__ = ("id", "source", "what", "start", "end", "status", "detail", "requests", "mb", "pu", "waited")

    def __init__(self, source: str, what: str):
        self.id, self.source, self.what = next(_ids), source, what
        self.start, self.end = time.time(), None
        self.status, self.detail = "running", None
        self.requests, self.mb, self.pu, self.waited = 0, 0.0, 0.0, 0.0

    def as_dict(self) -> dict:
        return {"id": self.id, "source": self.source, "what": self.what, "start": self.start, "end": self.end,
                "status": self.status, "detail": self.detail, "requests": self.requests,
                "mb": round(self.mb, 2), "pu": round(self.pu, 2), "waited_s": round(self.waited)}


def add_sink(fn) -> None:
    with _lock:
        _sinks.append(fn)


def remove_sink(fn) -> None:
    with _lock:
        if fn in _sinks:
            _sinks.remove(fn)


def _emit(t: Task) -> None:
    d = t.as_dict()
    for fn in list(_sinks):
        try:
            fn(d)
        except Exception:  # noqa: BLE001 - a display problem must never stop a run
            pass


def current() -> Task | None:
    stack = getattr(_local, "stack", None)
    return stack[-1] if stack else None


def bind(fn):
    """Wrap fn so that, when run in another thread (a tile download), it reports to the caller's task."""
    t = current()

    def run(*a, **k):
        if t is None:
            return fn(*a, **k)
        stack = getattr(_local, "stack", None)
        if stack is None:
            stack = _local.stack = []
        stack.append(t)
        try:
            return fn(*a, **k)
        finally:
            stack.pop()
    return run


@contextmanager
def task(source: str, what: str):
    t = Task(source, what)
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    stack.append(t)
    _emit(t)
    try:
        yield t
    except BaseException as e:
        t.status, t.detail = "failed", f"{type(e).__name__}: {str(e)[:140]}"
        raise
    else:
        if t.status == "running":
            t.status = "done"
            if t.detail and "retrying in" in t.detail:
                t.detail = None
            if t.waited >= 1:
                t.detail = (f"{t.detail}; " if t.detail else "") + f"includes {t.waited:.0f} s waiting for a busy server"
    finally:
        t.end = time.time()
        stack.pop()
        _emit(t)
        extra = []
        if t.requests:
            extra.append(f"{t.requests} request{'s' if t.requests != 1 else ''}")
        if t.mb >= 0.05:
            extra.append(f"{t.mb:.1f} MB")
        if t.pu:
            extra.append(f"{t.pu:.1f} PU")
        log.info("[%s] %s: %s in %.0f s%s", t.source, t.what, t.status, t.end - t.start,
                 f" ({', '.join(extra)})" if extra else "")


def cached(source: str, what: str) -> None:
    """Something reused from an earlier run of the same area (no download)."""
    t = Task(source, what)
    t.status, t.end, t.detail = "cached", t.start, "reused from an earlier run, no download"
    _emit(t)


def note(detail: str) -> None:
    """Live detail for the task running on this thread (e.g. a retry wait)."""
    t = current()
    if t is not None:
        t.detail = detail
        _emit(t)


def wait(seconds: float, reason: str) -> None:
    """The task is about to sleep before a retry (busy server): shown live and summed up at the end."""
    t = current()
    if t is not None:
        with _lock:
            t.waited += float(seconds)
        t.detail = f"{reason}; retrying in {seconds:.0f} s"
        _emit(t)


def add(requests: int = 0, mb: float = 0.0, pu: float = 0.0) -> None:
    t = current()
    if t is not None:
        with _lock:
            t.requests += requests
            t.mb += mb
            t.pu += pu
        _emit(t)
