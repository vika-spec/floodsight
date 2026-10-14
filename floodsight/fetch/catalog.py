"""Copernicus Data Space catalogue search + automatic Sentinel-1 pair selection.

The catalogue call only returns metadata (a few KB). Pixel data is fetched later, cropped to the AOI.
Pair selection is a pure function so it can be tested offline.
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import requests

from ..core import AOI

log = logging.getLogger("floodsight")

ODATA = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"

# IW swath incidence range (near -> far), degrees
IW_NEAR, IW_FAR = 30.9, 45.9


@dataclass
class Scene:
    id: str
    name: str
    start: datetime
    platform: str
    rel_orbit: int
    direction: str               # ASCENDING / DESCENDING
    footprint: list = field(default_factory=list)  # [[lon, lat], ...] outer ring

    @property
    def day(self) -> date:
        return self.start.date()


@dataclass
class Acquisition:
    """All frames from one satellite pass over the AOI (same day, same track) — processed as one mosaic."""
    day: date
    rel_orbit: int
    direction: str
    scenes: list
    coverage: float              # fraction of AOI covered by the union of frames

    @property
    def track(self) -> str:
        return f"{self.direction[:3].lower()}-{self.rel_orbit:03d}"


@dataclass
class Pair:
    pre: Acquisition
    post: Acquisition
    score: float
    incidence_deg: float
    heading_deg: float
    extra_pres: list = field(default_factory=list)   # earlier passes on the same track (multi-image reference)

    def describe(self) -> str:
        gap = (self.post.day - self.pre.day).days
        return (f"track {self.pre.track}: before {self.pre.day} -> after {self.post.day} "
                f"({gap} days apart, AOI coverage {min(self.pre.coverage, self.post.coverage):.0%})")

    def to_dict(self) -> dict:
        return {"track": self.pre.track, "pre_date": str(self.pre.day), "post_date": str(self.post.day),
                "gap_days": (self.post.day - self.pre.day).days,
                "coverage": round(min(self.pre.coverage, self.post.coverage), 3),
                "incidence_deg": round(self.incidence_deg, 1), "heading_deg": round(self.heading_deg, 1),
                "pre_scenes": [s.name for s in self.pre.scenes], "post_scenes": [s.name for s in self.post.scenes],
                "pre_stack_dates": [str(self.pre.day)] + [str(a.day) for a in self.extra_pres]}


# ---------------------------------------------------------------------------
# Catalogue search
# ---------------------------------------------------------------------------

def _poly_wkt(aoi: AOI) -> str:
    w, s, e, n = aoi.as_list()
    return f"POLYGON(({w} {s},{e} {s},{e} {n},{w} {n},{w} {s}))"


def _attr(item: dict, name: str):
    for a in item.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return None


def _rel_orbit_from_name(name: str) -> int | None:
    m = re.match(r"S1([A-D])_.*?_\d{8}T\d{6}_\d{8}T\d{6}_(\d{6})_", name)
    if not m:
        return None
    sat, absolute = m.group(1), int(m.group(2))
    offsets = {"A": 73, "B": 27}   # published offsets; S1C/S1D rely on the catalogue attribute
    if sat not in offsets:
        return None
    return (absolute - offsets[sat]) % 175 + 1


POL_RE = re.compile(r"_GRD[HM]_1S(DV|SV|DH|SH)_")


def parse_s1_items(items: list[dict]) -> list[Scene]:
    """Scenes usable by FloodSight: IW GRD high resolution, DUAL polarisation VV+VH ("1SDV" in the name).
    Single-polarisation frames ("1SSV": VV only) and HH/HV frames are skipped: water/debris rules and the model
    need both VV and VH, and processing such a frame fails ("no tiff for band VH")."""
    scenes, seen, skipped_pol = [], set(), set()
    for it in items:
        name = it["Name"]
        base = name.replace("_COG", "")
        if base in seen or "_GRDH_" not in name:
            continue
        seen.add(base)
        m = POL_RE.search(name)
        if m and m.group(1) != "DV":
            skipped_pol.add(f"{name[17:25]} 1S{m.group(1)}")
            continue
        rel = _attr(it, "relativeOrbitNumber") or _rel_orbit_from_name(name)
        direction = (_attr(it, "orbitDirection") or "").upper()
        if rel is None or direction not in ("ASCENDING", "DESCENDING"):
            log.warning("Skipping %s: missing orbit metadata", name)
            continue
        start = datetime.fromisoformat(it["ContentDate"]["Start"].replace("Z", "+00:00")).replace(tzinfo=None)
        fp = []
        geo = it.get("GeoFootprint") or {}
        coords = geo.get("coordinates")
        if coords:
            ring = coords[0][0] if geo.get("type") == "MultiPolygon" else coords[0]
            fp = [list(map(float, c[:2])) for c in ring]
        scenes.append(Scene(it["Id"], name, start, name[:3], int(rel), direction, fp))
    if skipped_pol:
        log.info("Skipped %d frame(s) without VV+VH dual polarisation (%s); FloodSight needs both bands",
                 len(skipped_pol), ", ".join(sorted(skipped_pol)[:6]) + (" ..." if len(skipped_pol) > 6 else ""))
    return sorted(scenes, key=lambda s: s.start)


def search_s1(aoi: AOI, start: date, end: date, session: requests.Session | None = None) -> list[Scene]:
    s = session or requests.Session()
    flt = (
        "Collection/Name eq 'SENTINEL-1' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{_poly_wkt(aoi)}') "
        f"and ContentDate/Start ge {start.isoformat()}T00:00:00.000Z "
        f"and ContentDate/Start lt {end.isoformat()}T23:59:59.999Z "
        "and contains(Name,'_IW_GRDH_')"
    )
    params = {"$filter": flt, "$expand": "Attributes", "$top": 200, "$orderby": "ContentDate/Start asc"}
    items = _get_all(s, ODATA, params)
    scenes = parse_s1_items(items)
    log.info("Catalogue: %d Sentinel-1 GRD frames between %s and %s", len(scenes), start, end)
    return scenes


def search_s2(aoi: AOI, start: date, end: date, max_cloud: float,
              session: requests.Session | None = None) -> list[dict]:
    s = session or requests.Session()
    flt = (
        "Collection/Name eq 'SENTINEL-2' "
        f"and OData.CSC.Intersects(area=geography'SRID=4326;{_poly_wkt(aoi)}') "
        f"and ContentDate/Start ge {start.isoformat()}T00:00:00.000Z "
        f"and ContentDate/Start lt {end.isoformat()}T23:59:59.999Z "
        "and contains(Name,'MSIL2A') "
        "and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' "
        f"and att/OData.CSC.DoubleAttribute/Value le {max_cloud})"
    )
    items = _get_all(s, ODATA, {"$filter": flt, "$expand": "Attributes", "$top": 200})
    out = []
    for it in items:
        out.append({"id": it["Id"], "name": it["Name"],
                    "date": it["ContentDate"]["Start"][:10], "cloud": _attr(it, "cloudCover")})
    return out


def _get_all(session: requests.Session, url: str, params: dict, retries: int = 3) -> list[dict]:
    items, next_url, p = [], url, params
    while next_url:
        for attempt in range(retries):
            try:
                r = session.get(next_url, params=p, timeout=60)
                r.raise_for_status()
                break
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise RuntimeError(f"Catalogue search failed: {e}") from e
                time.sleep(2 ** attempt)
        data = r.json()
        items.extend(data.get("value", []))
        next_url, p = data.get("@odata.nextLink"), None
    return items


# ---------------------------------------------------------------------------
# Pair selection (pure logic, tested offline)
# ---------------------------------------------------------------------------

def _coverage(footprints: list[list], aoi: AOI) -> float:
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union

    a = box(*aoi.as_list())
    polys = [Polygon(fp) for fp in footprints if len(fp) >= 4]
    if not polys:
        return 1.0  # unknown footprint: assume full, the pixel mask will tell the truth later
    return float(unary_union(polys).buffer(0).intersection(a).area / a.area)


def group_acquisitions(scenes: list[Scene], aoi: AOI) -> list[Acquisition]:
    groups = defaultdict(list)
    for s in scenes:
        groups[(s.day, s.rel_orbit, s.direction)].append(s)
    return [Acquisition(d, r, dirn, sc, _coverage([x.footprint for x in sc], aoi))
            for (d, r, dirn), sc in sorted(groups.items())]


def estimate_geometry(acq: Acquisition, aoi: AOI) -> tuple[float, float]:
    """Approximate incidence angle at the AOI centre and platform heading.

    Right-looking SAR: ascending passes (heading ~ -12°) look east, so near range is the western edge
    of the swath; descending (heading ~ 192°) look west. Position across the footprint gives incidence.
    """
    lon, lat = aoi.center
    heading = -12.0 if acq.direction == "ASCENDING" else 192.0
    lons = [p[0] for s in acq.scenes for p in s.footprint]
    if not lons:
        return (IW_NEAR + IW_FAR) / 2, heading
    w, e = min(lons), max(lons)
    frac = (lon - w) / max(e - w, 1e-6)
    if acq.direction == "DESCENDING":
        frac = 1 - frac
    frac = min(max(frac, 0.0), 1.0)
    return IW_NEAR + frac * (IW_FAR - IW_NEAR), heading


def select_pairs(scenes: list[Scene], aoi: AOI, flood_date: date, cfg: dict,
                 event_start: date | None = None) -> list[Pair]:
    """Best before/after pair per orbit track, ranked. Same relative orbit = same viewing geometry.

    The after-image is the earliest acquisition on/after the flood date. The before-image is the latest one
    strictly before `event_start` (default: the flood date; earlier when the operator says the area was still
    dry on a given day, see core.event_start)."""
    s1 = cfg["sentinel1"]
    start = event_start or flood_date
    acqs = group_acquisitions(scenes, aoi)
    post_start = flood_date + timedelta(days=s1.get("post_min_offset_days", 0))
    by_track = defaultdict(list)
    for a in acqs:
        by_track[(a.rel_orbit, a.direction)].append(a)

    pairs = []
    for track, lst in by_track.items():
        pres = [a for a in lst if a.day < start and a.coverage >= 0.3]
        posts = [a for a in lst if a.day >= post_start and a.coverage >= 0.3]
        if not pres or not posts:
            continue
        post = min(posts, key=lambda a: a.day)          # earliest after: most useful to rescuers
        pre = max(pres, key=lambda a: a.day)            # latest before: least unrelated change
        pre_gap = (start - pre.day).days
        post_gap = (post.day - flood_date).days
        cov = min(pre.coverage, post.coverage)
        score = 100 * cov - 2.0 * post_gap - 0.5 * pre_gap
        inc, head = estimate_geometry(post, aoi)
        # optional multi-image "before" reference: the next-latest passes of the SAME track (same geometry),
        # only nearly complete ones, newest first; n = 1 means the single latest before-image (default)
        n = int(s1.get("pre_stack_images", 1))
        extra = sorted([a for a in pres if a is not pre and a.coverage >= 0.6], key=lambda a: a.day, reverse=True)
        pairs.append(Pair(pre, post, score, inc, head, extra[:max(n - 1, 0)]))
    pairs.sort(key=lambda p: -p.score)
    return pairs[: s1.get("max_pairs", 2)]


# ---------------------------------------------------------------------------
# Auth (used by the local GRD fallback; openEO has its own login flow)
# ---------------------------------------------------------------------------

def cdse_token() -> str:
    user, pwd = os.environ.get("CDSE_USERNAME"), os.environ.get("CDSE_PASSWORD")
    if not user or not pwd:
        raise RuntimeError("Set CDSE_USERNAME and CDSE_PASSWORD in .env for the local download fallback.")
    r = requests.post(TOKEN_URL, data={"client_id": "cdse-public", "grant_type": "password",
                                       "username": user, "password": pwd}, timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]
