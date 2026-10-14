"""Helicopter access for cut-off settlements: mapped helipads and candidate landing zones.

Two kinds of answer, both from allowed data only:
  1. Mapped helipads / heliports from the PRE-EVENT OpenStreetMap (aeroway=helipad|heliport), each with what
     the flood map shows at that spot (flood/debris detected, not visible to the radar, or nothing detected).
  2. Candidate landing zones found in the terrain: open, flat ground (slope from the Copernicus DEM) where
     the flood map shows no flood, debris or water, the radar could see the ground, no mapped building
     stands, and the clear area is at least `clear_diameter_m` across, kept `flood_clearance_m` away from
     detected flood/debris.
For every settlement whose road access is cut off, likely cut off or unknown: the nearest of each kind
within `max_distance_km` (straight line).

Slope limit 7 degrees: the "acceptable" ground slope for landing in US Army FM 3-21.38 Pathfinder Operations
(skid-equipped helicopters). Power lines mapped in the pre-event OpenStreetMap are kept clear.

What is NOT checked (stated on every report): trees, buildings or power lines missing from OpenStreetMap,
loose or soft ground, wind, the approach path, and flood or debris that our map missed (it misses much of
the debris in mountain valleys). A candidate zone is a place for a pilot to look at, not a certified site.
"""
from __future__ import annotations

import math

import numpy as np

from ..detect.baseline import NO_CHANGE

NEEDS_AIR = ("cut_off", "likely_cut_off", "unknown")
NOT_CHECKED = ("trees, buildings or power lines missing from OpenStreetMap, soft or loose ground, wind, the approach "
               "path, and flood or debris our map missed (it misses much of the debris in mountain valleys)")
SLOPE_SOURCE = "US Army FM 3-21.38 Pathfinder Operations: ground slope up to 7 degrees acceptable for landing"


def _rep_point(geom):
    return geom if geom.geom_type == "Point" else geom.representative_point()


def _occupied(gdf, buffer_m, grid):
    """Pixels within buffer_m of any geometry in gdf (buildings, power lines)."""
    from rasterio.features import rasterize
    if gdf is None or gdf.empty:
        return None
    g = gdf.to_crs(grid.crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty]
    if not len(g):
        return None
    import shapely
    # all outlines widened in one call; quad_segs=16 is what geom.buffer() uses, so the shapes are identical
    shapes = shapely.buffer(g.geometry.values, buffer_m, quad_segs=16)
    return rasterize(((s, 1) for s in shapes), out_shape=grid.shape,
                     transform=grid.transform, fill=0, all_touched=True, dtype="uint8").astype(bool)


def candidate_zones(cls, slope, buildings, grid, cfg, power_lines=None) -> list[dict]:
    """Candidate landing zones on the analysis grid. Returns dicts with x/y in grid CRS, size and slope."""
    from scipy import ndimage as ndi

    L = cfg["landing"]
    res = grid.res
    free = (cls == NO_CHANGE) & (np.nan_to_num(slope, nan=90.0) <= L["max_slope_deg"])
    hazard = ~((cls == NO_CHANGE) | (cls == 255))     # flood, debris, permanent water
    if hazard.any():
        free &= ndi.distance_transform_edt(~hazard) * res >= L["flood_clearance_m"]
    for gdf, buf in ((buildings, L["building_clearance_m"]), (power_lines, L.get("power_line_clearance_m", 30))):
        occ = _occupied(gdf, buf, grid)
        if occ is not None:
            free &= ~occ
    # a pixel can be the centre of a zone only if the whole clear circle around it is free
    rad = max(1, int(round(L["clear_diameter_m"] / 2 / res)))
    yy, xx = np.mgrid[-rad:rad + 1, -rad:rad + 1]
    centre_ok = ndi.binary_erosion(free, structure=(xx ** 2 + yy ** 2) <= rad ** 2, border_value=0)
    lab, n = ndi.label(centre_ok)
    if n == 0:
        return []
    depth = ndi.distance_transform_edt(free) * res        # how far each pixel is from the nearest obstacle
    zones = []
    idx = ndi.maximum_position(depth, lab, range(1, n + 1))
    sizes = ndi.sum(centre_ok, lab, range(1, n + 1))     # pixels where a full clear circle fits
    for k, (r, c) in enumerate(idx):
        x, y = grid.transform * (c + 0.5, r + 0.5)
        zones.append({"x": x, "y": y, "clear_radius_m": round(float(depth[r, c]), 0),
                      "slope_deg": round(float(slope[r, c]), 1), "centre_area_m2": round(float(sizes[k]) * res * res)})
    return zones


def helicopter_access(cls, slope, settlements, helipads, buildings, grid, cfg, power_lines=None) -> dict:
    """Mapped helipads + candidate zones, and the nearest of each for settlements that need air access."""
    from pyproj import Transformer

    L = cfg["landing"]
    to_ll = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)

    def ll(x, y):
        lon, lat = to_ll.transform(x, y)
        return [round(lat, 5), round(lon, 5)]

    pads = []
    if helipads is not None and not helipads.empty:
        hp = helipads.to_crs(grid.crs)
        for _, row in hp.iterrows():
            if row.geometry is None or row.geometry.is_empty:
                continue
            p = _rep_point(row.geometry)
            r, c = grid.xy_to_rowcol(np.array([p.x]), np.array([p.y]))
            r, c = int(r[0]), int(c[0])
            if 0 <= r < grid.height and 0 <= c < grid.width:
                v = int(cls[r, c])
                state = "not visible to the radar" if v == 255 else (
                    "no damage detected" if v == NO_CHANGE else "flood/water/debris detected at the pad")
            else:
                state = "outside the analysed area"
            nm = row.get("name:en") or row.get("name")
            nm = str(nm) if nm is not None and str(nm) not in ("", "nan", "None") else f"unnamed {row.get('aeroway') or 'helipad'}"
            pads.append({"name": nm, "x": p.x, "y": p.y, "lat_lon": ll(p.x, p.y), "state": state,
                         "usable": state in ("no damage detected", "outside the analysed area")})

    # Landing grounds are looked for only when some settlement needs air access: they exist to reach cut-off
    # places. With none (most plains runs), the search over every building of a city is skipped; the
    # per-settlement answers would be empty either way.
    needy = settlements is not None and not settlements.empty and "status" in settlements and \
        bool(settlements["status"].isin(NEEDS_AIR).any())
    zones = candidate_zones(cls, slope, buildings, grid, cfg, power_lines) if needy else []
    for z in zones:
        z["lat_lon"] = ll(z["x"], z["y"])

    per = []
    if settlements is not None and not settlements.empty and "status" in settlements:
        st = settlements.to_crs(grid.crs)
        maxd = L["max_distance_km"] * 1000
        for _, s in st[st["status"].isin(NEEDS_AIR)].iterrows():
            sx, sy = s.geometry.x, s.geometry.y

            def nearest(items, ok=lambda _: True):
                best = None
                for it in items:
                    if not ok(it):
                        continue
                    d = math.hypot(it["x"] - sx, it["y"] - sy)
                    if d <= maxd and (best is None or d < best[0]):
                        best = (d, it)
                return best
            hp = nearest(pads, lambda p: p["usable"])
            zn = nearest(zones)
            per.append({"name": s.get("name"), "status": s["status"], "priority": s.get("priority"),
                        "helipad": {"name": hp[1]["name"], "km": round(hp[0] / 1000, 2), "lat_lon": hp[1]["lat_lon"]} if hp else None,
                        "zone": {"km": round(zn[0] / 1000, 2), "lat_lon": zn[1]["lat_lon"],
                                 "clear_radius_m": zn[1]["clear_radius_m"], "slope_deg": zn[1]["slope_deg"]} if zn else None})
        per.sort(key=lambda p: (p["priority"] if p["priority"] is not None else 10 ** 6))
    clean = lambda items: [{k: v for k, v in it.items() if k not in ("x", "y")} for it in items]
    return {"helipads": clean(pads), "zones_total": len(zones), "zones": clean(zones)[:500], "per_settlement": per,
            "zones_searched": needy,
            "criteria": {**{k: L[k] for k in ("max_slope_deg", "clear_diameter_m", "flood_clearance_m",
                                              "building_clearance_m", "max_distance_km")},
                         "power_line_clearance_m": L.get("power_line_clearance_m", 30),
                         "power_lines_mapped": 0 if power_lines is None else int(len(power_lines)),
                         "slope_source": SLOPE_SOURCE},
            "not_checked": NOT_CHECKED}
