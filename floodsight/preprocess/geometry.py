"""Radar viewing geometry over terrain: slope, local incidence angle, layover and shadow.

Conventions: grid x = east, rows grow southward. Heading = satellite flight direction, degrees clockwise
from north. Sentinel-1 is right-looking, so the look azimuth is heading + 90°.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, map_coordinates


def gradients(dem: np.ndarray, res: float) -> tuple[np.ndarray, np.ndarray]:
    z = np.where(np.isfinite(dem), dem, np.nanmean(dem))
    dzdx = np.gradient(z, axis=1) / res
    dzdy = -np.gradient(z, axis=0) / res   # north-positive
    return dzdx, dzdy


def slope_deg(dem: np.ndarray, res: float) -> np.ndarray:
    dzdx, dzdy = gradients(dem, res)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype(np.float32)


def look_vector(heading_deg: float) -> tuple[float, float]:
    la = np.radians(heading_deg + 90.0)
    return float(np.sin(la)), float(np.cos(la))       # (east, north), pointing away from the sensor


def sensor_ground_vector(heading_deg: float) -> tuple[float, float]:
    lx, ly = look_vector(heading_deg)
    return -lx, -ly                                     # horizontal direction toward the sensor


def local_incidence(dem, res, incidence_deg, heading_deg) -> np.ndarray:
    """Angle between the terrain normal and the direction to the satellite (degrees)."""
    dzdx, dzdy = gradients(dem, res)
    th = np.radians(incidence_deg)
    lx, ly = look_vector(heading_deg)
    sx, sy, sz = -np.sin(th) * lx, -np.sin(th) * ly, np.cos(th)
    norm = np.sqrt(dzdx ** 2 + dzdy ** 2 + 1)
    cosang = (-dzdx * sx - dzdy * sy + sz) / norm
    return np.degrees(np.arccos(np.clip(cosang, -1, 1))).astype(np.float32)


def visibility_mask(dem: np.ndarray, res: float, incidence_deg, heading_deg: float,
                    max_shadow_km: float = 3.0) -> dict:
    """Pixels the radar cannot measure reliably.

    layover: slopes facing the sensor steeper than the incidence angle (signal from several heights
             collapses into one pixel). Dilated by one pixel toward the sensor.
    shadow:  slopes facing away steeper than (90° - incidence), plus cast shadow behind ridges found
             by marching toward the sensor and checking if terrain blocks the ray.
    Returns boolean arrays and a combined 'blind' mask.
    """
    th = np.radians(np.broadcast_to(np.asarray(incidence_deg, dtype=float), dem.shape))
    dzdx, dzdy = gradients(dem, res)
    lx, ly = look_vector(heading_deg)
    g = dzdx * lx + dzdy * ly                                 # rise per metre moving away from sensor
    alpha = np.arctan(g)
    layover = alpha > th
    shadow = (-alpha) > (np.pi / 2 - th)

    # cast shadow: point q is hidden if some point p closer to the sensor (distance d) satisfies
    # z_p > z_q + d * cot(theta)
    z = np.where(np.isfinite(dem), dem, np.nanmin(dem)).astype(np.float32)
    relief = float(np.nanmax(z) - np.nanmin(z))
    dmax = min(max_shadow_km * 1000, relief * float(np.tan(th.max())) + res)
    step = max(res, dmax / 60)
    sx, sy = sensor_ground_vector(heading_deg)
    R, C = np.mgrid[0:z.shape[0], 0:z.shape[1]].astype(np.float32)
    cot = 1.0 / np.tan(th)
    cast = np.zeros(z.shape, bool)
    d = step
    while d <= dmax:
        zp = map_coordinates(z, [R - d * sy / res, C + d * sx / res], order=1, mode="nearest")
        cast |= zp > z + d * cot + 1.0
        d += step
    shadow |= cast

    # small toward-sensor dilation of layover (passive layover zone)
    st = np.zeros((3, 3), bool)
    st[1, 1] = True
    st[1 - int(round(sy)), 1 + int(round(sx))] = True
    layover = binary_dilation(layover, st)
    return {"layover": layover, "shadow": shadow, "blind": layover | shadow}
