"""Find a place by name (village, town, district, river...) and propose an analysis box around it.

    from floodsight.fetch.geocode import search_places
    search_places("Betrawati, Nuwakot, Nepal")
    search_places("27.9705, 85.1866")          # a point (latitude, longitude): no lookup, box around it

This only helps the operator choose WHERE to look. It is not an analysis input: the flood map, damage and
access results still come only from Sentinel-1/2, the Copernicus DEM and pre-event OpenStreetMap.

Source: Nominatim, the OpenStreetMap geocoder (https://nominatim.org), under its usage policy
(https://operations.osmfoundation.org/policies/nominatim/): an identifying User-Agent, at most one request per
second, results cached, searches only when the user presses Search (no search-as-you-type).
Data © OpenStreetMap contributors (ODbL). If Nominatim cannot be reached, Photon (photon.komoot.io, also
OpenStreetMap data) is tried.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict

import requests

log = logging.getLogger("floodsight")

NOMINATIM = "https://nominatim.openstreetmap.org/search"
PHOTON = "https://photon.komoot.io/api/"
UA = {"User-Agent": "FloodSight/1.0 (flood damage mapping prototype; Multimodal AI Hackathon 2026)"}
DEFAULT_SIDE_KM = 15.0          # box proposed around a village or town (a valley stretch, adjustable on the map)
MIN_PLACE_KM2 = 25.0            # a place outline smaller than this gets the default box around its centre

_lock = threading.Lock()
_last = [0.0]
_cache: OrderedDict = OrderedDict()


def _polite_get(url, params):
    with _lock:                                   # never more than one request per second from this server
        wait = 1.05 - (time.monotonic() - _last[0])
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.monotonic()
    r = requests.get(url, params=params, headers=UA, timeout=15)
    r.raise_for_status()
    return r.json()


def _km2(w, s, e, n):
    return (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * (n - s) * 110.57


def square_box(lon, lat, side_km):
    hx = side_km / 2 / (111.32 * max(math.cos(math.radians(lat)), 0.01))
    hy = side_km / 2 / 110.57
    return [round(lon - hx, 5), round(lat - hy, 5), round(lon + hx, 5), round(lat + hy, 5)]


def suggest_box(lon, lat, outline, max_km2, side_km=DEFAULT_SIDE_KM):
    """The place's own outline when it is a sensible analysis size, otherwise a square around its centre.
    Administrative outlines are NOT flood areas: the operator is told to move/resize the box onto the valley."""
    if outline:
        a = _km2(*outline)
        if MIN_PLACE_KM2 <= a <= max_km2:
            return [round(v, 5) for v in outline], "place outline"
    side = min(side_km, math.sqrt(max_km2))
    return square_box(lon, lat, side), f"{side:.0f} km square around the place"


def _nominatim(q, limit):
    out = []
    for it in _polite_get(NOMINATIM, {"q": q, "format": "jsonv2", "limit": limit, "accept-language": "en"}):
        s, n, w, e = (float(v) for v in it["boundingbox"])
        out.append({"name": it.get("display_name", q), "kind": (it.get("type") or it.get("category") or "").replace("_", " "),
                    "lon": float(it["lon"]), "lat": float(it["lat"]), "outline": [w, s, e, n], "source": "Nominatim"})
    return out


def _photon(q, limit):
    out = []
    for f in _polite_get(PHOTON, {"q": q, "limit": limit, "lang": "en"}).get("features", []):
        p, (lon, lat) = f.get("properties", {}), f["geometry"]["coordinates"]
        ext = p.get("extent")                               # [west, north, east, south]
        name = ", ".join(x for x in (p.get("name"), p.get("county") or p.get("city"), p.get("state"),
                                     p.get("country")) if x)
        out.append({"name": name or q, "kind": (p.get("osm_value") or "").replace("_", " "), "lon": lon, "lat": lat,
                    "outline": [ext[0], ext[3], ext[2], ext[1]] if ext else None, "source": "Photon"})
    return out


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_point(q: str):
    """'27.9705, 85.1866' (lat, lon: the order Google Maps and most phones copy) -> (lon, lat).
    Also accepts N/S/E/W letters ('27.97 N 85.18 E') and lon, lat when the first value cannot be a latitude.
    Returns None if the text is not exactly two coordinates."""
    t = q.strip().upper()
    if re.search(r"[A-DF-MO-RT-VX-Z]", t.replace("DEG", "")):        # other letters than N, E, S, W -> a name
        return None
    nums = [float(x) for x in _NUM.findall(t)]
    if len(nums) != 2:
        return None
    a, b = nums
    letters = re.findall(r"[NSEW]", t)
    if len(letters) == 2:
        lat, lon = (a, b) if letters[0] in "NS" else (b, a)
        lat = -abs(lat) if "S" in letters else lat
        lon = -abs(lon) if "W" in letters else lon
    elif abs(a) > 90 >= abs(b):
        lon, lat = a, b                       # first value cannot be a latitude: lon, lat
    else:
        lat, lon = a, b
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lon, lat


def search_places(q: str, max_km2: float = 3000.0, limit: int = 6) -> list[dict]:
    q = " ".join((q or "").split())
    if len(q) < 2:
        return []
    pt = parse_point(q)
    if pt is not None:                        # coordinates: no lookup needed
        lon, lat = pt
        cands = [(lon, lat)]
        # both numbers could be latitudes and no N/E letters: offer the swapped reading too, never guess
        if abs(lon) <= 90 and lon != lat and not re.search(r"[NSEW]", q.upper()):
            cands.append((lat, lon))
        out = []
        for i, (x, y) in enumerate(cands):
            box, how = suggest_box(x, y, None, max_km2)
            out.append({"name": f"Point at latitude {y:.5f}, longitude {x:.5f}"
                                + (" (numbers read the other way round)" if i else ""),
                        "kind": "coordinates", "lon": x, "lat": y, "outline": None, "source": "typed",
                        "bbox": box, "bbox_from": how, "bbox_km2": round(_km2(*box), 1)})
        return out
    key = (q.lower(), limit)
    if key in _cache:
        _cache.move_to_end(key)
        hits = _cache[key]
    else:
        hits, errors = None, []
        for fn in (_nominatim, _photon):
            try:
                hits = fn(q, limit)
                break
            except Exception as e:  # noqa: BLE001 - try the next geocoder
                errors.append(f"{fn.__name__[1:]}: {str(e)[:80]}")
        if hits is None:
            raise RuntimeError("Place search is unavailable right now (" + "; ".join(errors) + "). "
                               "Draw the box on the map or type its corners instead.")
        _cache[key] = hits
        if len(_cache) > 500:
            _cache.popitem(last=False)
    out = []
    for h in hits:
        box, how = suggest_box(h["lon"], h["lat"], h["outline"], max_km2)
        out.append({**h, "bbox": box, "bbox_from": how, "bbox_km2": round(_km2(*box), 1)})
    return out
