"""Raw Level-1 GRD chain vs. Sentinel Hub: same area and dates processed both ways, maps compared.

    python run.py --bbox ... --date ... --name area_shub                      # Sentinel Hub (default)
    python run.py --bbox ... --date ... --name area_raw --backend local       # our own GRD processing
    python scripts/compare_backends.py runs/area_shub runs/area_raw

Prints the backscatter agreement of the after-images (median difference, correlation) and the agreement of the
two flood/debris maps (IoU, area of each class). Both runs can also be scored against a reference with
floodsight.validate.ems. Nothing is tuned on this comparison.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cls(run: Path):
    import rasterio
    with rasterio.open(run / "layers" / "flood_class.tif") as src:
        return src.read(1)


def _after_vv(run: Path, grid):
    from floodsight.core import read_to_grid
    R = json.loads((run / "results.json").read_text())
    p = R["inputs"]["sentinel1_pairs"][0]
    hits = sorted((run / "inputs").glob(f"s1_{p['track']}_post_{p['post_date']}*.tif"))
    if not hits:
        return None, None
    lin = read_to_grid(hits[0], grid)[0]
    return 10 * np.log10(np.clip(lin, 1e-5, None)), hits[0].name


def main(argv=None):
    from floodsight.core import Grid
    from floodsight.detect.baseline import DEBRIS, FLOOD, NODATA

    a, b = (Path(x) for x in (argv or sys.argv[1:])[:2])
    Ra = json.loads((a / "results.json").read_text())
    grid = Grid.from_dict(Ra["grid"])
    ca, cb = _cls(a), _cls(b)
    if ca.shape != cb.shape:
        print("The two runs use different grids: run them with the same --bbox and --res.")
        return 1
    seen = (ca != NODATA) & (cb != NODATA)
    ha = ((ca == FLOOD) | (ca == DEBRIS)) & seen
    hb = ((cb == FLOOD) | (cb == DEBRIS)) & seen
    px = grid.res ** 2 / 1e6
    iou = (ha & hb).sum() / max((ha | hb).sum(), 1)
    print(f"\nflood + debris area: {a.name} {ha.sum() * px:.3f} km2, {b.name} {hb.sum() * px:.3f} km2")
    print(f"agreement of the two maps (IoU, where both see the ground): {iou:.3f}")
    for name, k in (("flood water", FLOOD), ("debris", DEBRIS)):
        print(f"  {name:12s} {((ca == k) & seen).sum() * px:8.3f} km2 vs {((cb == k) & seen).sum() * px:8.3f} km2")
    va, na = _after_vv(a, grid)
    vb, nb = _after_vv(b, grid)
    if va is not None and vb is not None:
        ok = np.isfinite(va) & np.isfinite(vb)
        d = vb[ok] - va[ok]
        r = np.corrcoef(va[ok], vb[ok])[0, 1]
        print(f"after-image VV: median difference {np.median(d):+.2f} dB, correlation {r:.3f}  ({na} vs {nb})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
