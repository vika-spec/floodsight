"""Check that this machine can run FloodSight on real data. Run it first, and again before judging.

    python scripts/check_setup.py
    python scripts/check_setup.py --login      # also opens the openEO login (first time only)

Each line prints OK / FAIL with the fix. Nothing here downloads large data.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OK, BAD, WARN = "\033[32mOK  \033[0m", "\033[31mFAIL\033[0m", "\033[33mWARN\033[0m"


def check(name, fn, fix=""):
    t0 = time.time()
    try:
        info = fn()
        print(f"{OK} {name} ({time.time() - t0:.1f}s){' - ' + str(info) if info else ''}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"{BAD} {name}: {type(e).__name__}: {str(e)[:160]}")
        if fix:
            print(f"      fix: {fix}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--login", action="store_true", help="authenticate openEO now")
    a = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    results = []
    for m in ("numpy", "scipy", "rasterio", "geopandas", "shapely", "pyproj", "networkx", "pysheds", "skimage",
              "openeo", "requests", "pyarrow", "fpdf", "uharfbuzz", "fastapi", "uvicorn", "matplotlib"):
        results.append(check(f"import {m}", lambda m=m: getattr(importlib.import_module(m), "__version__", ""),
                             "pip install -r requirements.txt"))

    def torch_gpu():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(f"torch {torch.__version__} sees no GPU")
        return f"torch {torch.__version__}, {torch.cuda.get_device_name(0)}, " \
               f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
    check("PyTorch + CUDA GPU (only needed for training/model inference)", torch_gpu,
          "install the CUDA 12.8+ build: https://pytorch.org/get-started/locally/")

    from floodsight.core import AOI
    aoi = AOI(85.25, 28.0, 85.35, 28.1)

    def catalog():
        from floodsight.fetch.catalog import search_s1
        s = search_s1(aoi, date(2026, 8, 1), date(2026, 9, 10))
        return f"{len(s)} Sentinel-1 frames around the Trishuli event; tracks: " \
               f"{sorted({(x.direction[:3], x.rel_orbit) for x in s})}"
    results.append(check("Copernicus catalogue search", catalog, "check internet access"))

    def dem():
        import rasterio
        from floodsight.fetch.dem import tile_names
        url = next(tile_names(85.3, 28.0, 85.31, 28.01))
        with rasterio.open(url) as src:
            return f"DEM tile readable ({src.width}x{src.height})"
    results.append(check("Copernicus DEM (public COG tiles)", dem, "check internet; the openEO fallback also works"))

    def osm():
        from floodsight.core import load_config
        from floodsight.fetch.osm import fetch_layer
        cfg = load_config()
        tiny = AOI(85.30, 27.98, 85.33, 28.01)
        gdf, src = fetch_layer("roads", tiny, date(2026, 7, 27), cfg["osm"])
        hint = "" if src == "ohsome v2" else " (tip: add OHSOME_API_KEY to .env for the main source)"
        return f"{len(gdf)} road segments via {src}{hint}"
    results.append(check("Pre-event OpenStreetMap download (tiny test area)", osm,
                         "see the warnings above; add OHSOME_API_KEY to .env (free at account.heigit.org/signup)"))

    def openeo_auth():
        from floodsight.fetch.s1 import openeo_connection
        con = openeo_connection()
        return f"logged in as {con.describe_account().get('user_id', '?')}"
    if a.login:
        results.append(check("openEO login (Copernicus Data Space)", openeo_auth,
                             "create a free account at dataspace.copernicus.eu"))
    else:
        print(f"{WARN} openEO login not tested (run with --login once; the token is then cached)")

    import os
    from floodsight.fetch import shub
    if shub.available():
        def sentinel_hub():
            from datetime import date
            from floodsight.core import AOI, Grid, load_config
            from floodsight.fetch.catalog import Acquisition
            g = Grid.for_aoi(AOI.parse("85.30,27.95,85.32,27.97"), 20)
            acq = Acquisition(day=date(2026, 8, 28), rel_orbit=85, direction="ascending", scenes=[], coverage=1)
            arr, _ = shub._request_grid(g, shub.s1_request_data(acq, load_config()), shub.S1_EVALSCRIPT, 3, "setup test")
            return f"radar tile {g.width}x{g.height} px, {100 * (arr[2] > 0).mean():.0f}% valid, {shub.USAGE['pu']:.1f} PU"
        results.append(check("Sentinel Hub (fast radar processing, tiny test area)", sentinel_hub,
                             "check SH_CLIENT_ID / SH_CLIENT_SECRET in .env (CDSE dashboard -> User settings -> OAuth clients)"))
    else:
        print(f"{WARN} SH_CLIENT_ID not set in .env: radar uses openEO (slower)")

    if os.environ.get("CDSE_USERNAME"):
        def token():
            from floodsight.fetch.catalog import cdse_token
            return f"token ok ({len(cdse_token())} chars)"
        results.append(check("CDSE download token (local GRD fallback)", token, "check CDSE_USERNAME/PASSWORD in .env"))
    else:
        print(f"{WARN} CDSE_USERNAME not set in .env: the local GRD fallback cannot download")

    w = Path(__file__).resolve().parent.parent / "models" / "siamese_unet.pt"
    print(f"{OK if w.exists() else WARN} model weights {'found' if w.exists() else 'missing (rule-based only until trained)'}: {w}")
    print("\nAll required checks passed." if all(results) else "\nSome checks failed; see fixes above.")


if __name__ == "__main__":
    main()
