"""Live data-request tracking: what is being downloaded, from where, how long, and why it is waiting."""
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from floodsight import activity


@pytest.fixture
def events():
    seen = {}
    sink = lambda d: seen.__setitem__(d["id"], d)  # noqa: E731
    activity.add_sink(sink)
    yield seen
    activity.remove_sink(sink)


def test_task_records_source_time_and_counts(events):
    with activity.task("Sentinel Hub (Copernicus)", "Sentinel-1 after image"):
        activity.add(requests=2, mb=3.5, pu=12.0)
    (d,) = events.values()
    assert d["source"] == "Sentinel Hub (Copernicus)" and d["status"] == "done"
    assert d["requests"] == 2 and d["mb"] == 3.5 and d["pu"] == 12.0 and d["end"] >= d["start"]


def test_wait_is_shown_live_then_summed(events):
    live = []
    activity.add_sink(lambda d: live.append(d["detail"]))
    with activity.task("Sentinel Hub (Copernicus)", "x"):
        activity.wait(10, "Sentinel Hub busy (429)")
    assert any(x and "retrying in 10 s" in x for x in live)
    (d,) = events.values()
    assert d["waited_s"] == 10 and "10 s waiting for a busy server" in d["detail"]


def test_failure_is_recorded_and_raised(events):
    with pytest.raises(ValueError):
        with activity.task("OpenStreetMap (ohsome)", "Roads"):
            raise ValueError("timeout")
    (d,) = events.values()
    assert d["status"] == "failed" and "timeout" in d["detail"]


def test_cached_entry(events):
    activity.cached("Copernicus DEM (AWS open data)", "Elevation")
    (d,) = events.values()
    assert d["status"] == "cached" and d["end"] == d["start"]


def test_bind_reports_tile_threads_to_the_parent_task(events):
    def tile(_):
        activity.add(requests=1, mb=1.0)
    with activity.task("Sentinel Hub (Copernicus)", "tiled image"):
        with ThreadPoolExecutor(4) as ex:
            list(ex.map(activity.bind(tile), range(8)))
    (d,) = events.values()
    assert d["requests"] == 8 and d["mb"] == 8.0


def test_parallel_tasks_do_not_mix(events):
    def one(i):
        with activity.task("OpenStreetMap (ohsome)", f"layer {i}"):
            activity.add(requests=i)
    ts = [threading.Thread(target=one, args=(i,)) for i in range(1, 5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    got = {d["what"]: d["requests"] for d in events.values()}
    assert got == {f"layer {i}": i for i in range(1, 5)}


def test_no_secret_fields():
    t = activity.Task("Sentinel Hub (Copernicus)", "x").as_dict()
    assert set(t) == {"id", "source", "what", "start", "end", "status", "detail", "requests", "mb", "pu", "waited_s"}
