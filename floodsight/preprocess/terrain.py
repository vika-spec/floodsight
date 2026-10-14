"""Hydrological terrain layers from the DEM: filled DEM, D8 flow direction, flow accumulation, HAND.

HAND (height above nearest drainage) is the single most useful false-alarm filter in mountains:
a radar-dark pixel 300 m above the river is shadow or smooth rock, not flood water.
"""
from __future__ import annotations

import logging

import numpy as np
from scipy import ndimage

log = logging.getLogger("floodsight")

# pysheds D8 encoding: N, NE, E, SE, S, SW, W, NW
DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)
D8_OFFSETS = {64: (-1, 0), 128: (-1, 1), 1: (0, 1), 2: (1, 1), 4: (1, 0), 8: (1, -1), 16: (0, -1), 32: (-1, -1)}


def fill_nan(dem: np.ndarray) -> np.ndarray:
    bad = ~np.isfinite(dem)
    if not bad.any():
        return dem.astype(np.float64)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return dem[tuple(idx)].astype(np.float64)


def height_above_drainage(fdir: np.ndarray, z: np.ndarray, streams: np.ndarray, max_doublings: int = 40):
    """HAND: height of each cell above the drainage cell its D8 flow path reaches first.

    Drainage = stream cells. A path that leaves the analysed area before meeting a stream ends at the last
    cell inside the area, and that cell is its reference (pysheds returns no value there instead). On flat
    plains this concerns most of the area: on a synthetic 0.2 m/km plain pysheds left 64% without HAND, and
    "no HAND" meant "excluded from the water zone", so real flood on plains was discarded. Measured to the
    exit point, HAND can only be lower than the true value (the path would continue downhill outside), and
    on a plain the relief along the path is small.

    Heights come from the DEM itself (not the depression-filled one), so a hollow keeps its true lower
    height. Negative values (a hollow below its drainage) are set to 0.
    Returns (hand float32, edge_ref bool: reference is the area edge, not a stream)."""
    H, W = z.shape
    n = H * W
    me = np.arange(n, dtype=np.int64)
    rows, cols = np.divmod(me, W)
    nxt = me.copy()
    f = fdir.ravel()
    for code, (dr, dc) in D8_OFFSETS.items():
        sel = f == code
        r2, c2 = rows[sel] + dr, cols[sel] + dc
        inside = (r2 >= 0) & (r2 < H) & (c2 >= 0) & (c2 < W)
        tgt = me[sel].copy()                      # leaves the area: this cell is the last one inside
        tgt[inside] = r2[inside] * W + c2[inside]
        nxt[sel] = tgt
    drain = streams.ravel().astype(bool)
    nxt[drain] = me[drain]                        # streams are their own reference
    # pointer doubling: after k rounds every cell points 2^k steps down its path (stops at fixed points)
    for _ in range(max_doublings):
        new = nxt[nxt]
        if np.array_equal(new, nxt):
            break
        nxt = new
    done = nxt[nxt] == nxt
    zf = z.ravel()
    hand = np.where(done, np.maximum(zf - zf[nxt], 0.0), np.nan).reshape(H, W).astype(np.float32)
    edge_ref = (done & ~drain[nxt]).reshape(H, W)
    return hand, edge_ref


def flow_layers(dem: np.ndarray, grid) -> dict:
    """Only what a flow path needs: D8 flow direction and accumulation on the depression-filled DEM
    (same steps as hydro_layers, without streams and HAND, which take more than half of its time)."""
    if not hasattr(np, "in1d"):
        np.in1d = np.isin
    from pyproj import Proj
    from pysheds.grid import Grid as PGrid
    from pysheds.sview import Raster, ViewFinder

    z = fill_nan(dem)
    vf = ViewFinder(affine=grid.transform, shape=z.shape, crs=Proj(grid.crs), nodata=np.float64(-9999))
    pg = PGrid(viewfinder=vf)
    inflated = pg.resolve_flats(pg.fill_depressions(pg.fill_pits(Raster(z, viewfinder=vf))))
    fdir = pg.flowdir(inflated, dirmap=DIRMAP)
    acc = pg.accumulation(fdir, dirmap=DIRMAP)
    return {"fdir": np.asarray(fdir).astype(np.int32), "acc": np.asarray(acc).astype(np.float32)}


def hydro_layers(dem: np.ndarray, grid, stream_area_km2: float = 2.0) -> dict:
    if not hasattr(np, "in1d"):   # pysheds still calls np.in1d, removed in NumPy 2.4
        np.in1d = np.isin
    from pyproj import Proj
    from pysheds.grid import Grid as PGrid
    from pysheds.sview import Raster, ViewFinder

    z = fill_nan(dem)
    vf = ViewFinder(affine=grid.transform, shape=z.shape, crs=Proj(grid.crs), nodata=np.float64(-9999))
    r = Raster(z, viewfinder=vf)
    pg = PGrid(viewfinder=vf)
    inflated = pg.resolve_flats(pg.fill_depressions(pg.fill_pits(r)))
    fdir = pg.flowdir(inflated, dirmap=DIRMAP)
    acc = pg.accumulation(fdir, dirmap=DIRMAP)
    cells = stream_area_km2 * 1e6 / grid.res ** 2
    streams = np.asarray(acc) > cells
    stream_r = Raster(streams, viewfinder=ViewFinder(affine=grid.transform, shape=z.shape,
                                                     crs=Proj(grid.crs), nodata=False))
    hand, edge_ref = height_above_drainage(np.asarray(fdir), z, streams)
    hand = np.where(np.isfinite(dem), hand, np.nan).astype(np.float32)
    log.info("HAND: %.0f%% of the area drains out of it before meeting a stream (height measured to the "
             "point where the flow leaves the area)", 100 * edge_ref.mean())
    return {
        "fdir": np.asarray(fdir).astype(np.int32),
        "acc": np.asarray(acc).astype(np.float32),
        "streams": streams,
        "hand": hand,
        "hand_edge_fraction": float(edge_ref.mean()),
    }
