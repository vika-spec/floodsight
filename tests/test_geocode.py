"""Place search: response parsing for both geocoders, the proposed box, fallback, and the API/CLI paths.
No network: the HTTP call is replaced."""
import pytest

from floodsight.fetch import geocode as G

NOMI = [  # Nominatim jsonv2: boundingbox = [south, north, west, east] as strings
    {"lat": "27.9705", "lon": "85.1866", "display_name": "Betrawati, Nuwakot, Bagmati Province, Nepal",
     "category": "place", "type": "village", "boundingbox": ["27.9505", "27.9905", "85.1666", "85.2066"]},
    {"lat": "27.95", "lon": "85.25", "display_name": "Nuwakot District, Nepal", "category": "boundary",
     "type": "administrative", "boundingbox": ["27.7", "28.2", "85.0", "85.5"]},
]
PHOTON = {"features": [{"geometry": {"coordinates": [85.1866, 27.9705]},
                        "properties": {"name": "Betrawati", "county": "Nuwakot", "country": "Nepal",
                                       "osm_value": "village", "extent": [85.18, 27.98, 85.19, 27.96]}}]}


@pytest.fixture(autouse=True)
def _clear():
    G._cache.clear()


def test_nominatim_parsing_and_boxes(monkeypatch):
    monkeypatch.setattr(G, "_polite_get", lambda url, params: NOMI)
    hits = G.search_places("Betrawati, Nuwakot, Nepal", max_km2=3000)
    v, d = hits
    assert v["kind"] == "village" and v["source"] == "Nominatim"
    assert v["bbox_from"].startswith("15 km square")              # village outline too small -> square
    w, s, e, n = v["bbox"]
    assert w < 85.1866 < e and s < 27.9705 < n and 200 < v["bbox_km2"] < 250
    assert d["bbox_from"] == "place outline" and d["bbox"] == [85.0, 27.7, 85.5, 28.2]   # ~2700 km² fits 3000
    small = G.search_places("Betrawati, Nuwakot, Nepal ", max_km2=1600)                 # cached query
    assert small[1]["bbox_from"].startswith("15 km square")       # district too big for the public limit


def test_falls_back_to_photon(monkeypatch):
    def get(url, params):
        if url == G.NOMINATIM:
            raise ConnectionError("down")
        return PHOTON
    monkeypatch.setattr(G, "_polite_get", get)
    h = G.search_places("Betrawati")[0]
    assert h["source"] == "Photon" and h["name"] == "Betrawati, Nuwakot, Nepal"
    assert h["outline"] == [85.18, 27.96, 85.19, 27.98]          # extent [W, N, E, S] -> [W, S, E, N]


def test_both_down_is_a_clear_error(monkeypatch):
    monkeypatch.setattr(G, "_polite_get", lambda url, params: (_ for _ in ()).throw(ConnectionError("x")))
    with pytest.raises(RuntimeError, match="Draw the box"):
        G.search_places("Betrawati")


def test_api_places(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from server.api import create_app
    client = TestClient(create_app(runs_dir=tmp_path, web_dir=tmp_path / "noweb"))
    monkeypatch.setattr(G, "_polite_get", lambda url, params: NOMI)
    r = client.get("/api/places", params={"q": "Betrawati"})
    assert r.status_code == 200 and len(r.json()["results"]) == 2
    assert "OpenStreetMap" in r.json()["attribution"]
    assert client.get("/api/places", params={"q": "x"}).status_code == 422


def test_cli_place_lists_boxes_and_does_not_run(monkeypatch, capsys):
    import run
    monkeypatch.setattr(G, "_polite_get", lambda url, params: NOMI)
    assert run.main(["--place", "Betrawati, Nuwakot, Nepal"]) == 0
    out = capsys.readouterr().out
    assert "--bbox" in out and "not the flood area" in out


@pytest.mark.parametrize("text,lonlat", [
    ("27.9705, 85.1866", (85.1866, 27.9705)),               # Google Maps order: lat, lon
    ("27.9705 85.1866", (85.1866, 27.9705)),
    ("95.1866, 27.9705", (95.1866, 27.9705)),               # first value > 90 -> lon, lat
    ("27.97 N, 85.18 E", (85.18, 27.97)),
    ("85.18E 27.97N", (85.18, 27.97)),
    ("33.86 S, 151.21 E", (151.21, -33.86)),
    ("-33.86, 151.21", (151.21, -33.86)),
])
def test_point_coordinates(text, lonlat):
    assert G.parse_point(text) == pytest.approx(lonlat)


@pytest.mark.parametrize("text", ["Betrawati", "Nuwakot 2", "85.1, 27.9, 85.2, 28.0", "95, 200", "Route 66, Texas"])
def test_not_a_point(text):
    assert G.parse_point(text) is None


def test_point_needs_no_network(monkeypatch):
    monkeypatch.setattr(G, "_polite_get", lambda url, params: (_ for _ in ()).throw(AssertionError("no lookup")))
    h, swapped = G.search_places("27.9705, 85.1866")      # both <= 90: both readings offered, lat-lon first
    assert (swapped["lon"], swapped["lat"]) == (27.9705, 85.1866)
    assert len(G.search_places("27.97 N, 85.18 E")) == 1     # letters make it unambiguous
    assert len(G.search_places("27.9705, 95.1866")) == 1     # 95 cannot be a latitude
    assert h["source"] == "typed" and h["bbox_from"].startswith("15 km square")
    w, s, e, n = h["bbox"]
    assert w < 85.1866 < e and s < 27.9705 < n
