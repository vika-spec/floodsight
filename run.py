"""FloodSight command line.

    python run.py --bbox 85.25,27.85,85.45,28.15 --date 2026-08-26
    python run.py --demo
    python run.py --place "Betrawati, Nuwakot, Nepal"          # find the box for a place name
    python run.py --bbox ... --date 2022-09-11 --before-date 2022-06-19   # flood that built up for weeks
    python run.py --bbox ... --date ... --backend local      # force local GRD processing
    python run.py --bbox ... --date ... --backend openeo     # force openEO (skip Sentinel Hub)
"""
from __future__ import annotations

import argparse
import json
import sys

from floodsight.core import load_config, setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser(description="Map flood damage and cut-off settlements from satellite data.")
    ap.add_argument("--bbox", help="west,south,east,north in degrees")
    ap.add_argument("--place", help='place name or point instead of --bbox, e.g. "Betrawati, Nuwakot, Nepal" or '
                    '"27.9705, 85.1866" (latitude, longitude): prints a proposed box and stops, so you can check '
                    "it and run with that --bbox")
    ap.add_argument("--date", default="2026-08-26", help="flood date YYYY-MM-DD")
    ap.add_argument("--before-date", help="optional: last day the area was still dry (YYYY-MM-DD). Before-images "
                    "are taken on or before it. Use for floods that built up over days or weeks; default: "
                    "the latest image before --date")
    ap.add_argument("--demo", action="store_true", help="offline synthetic valley (no accounts needed)")
    ap.add_argument("--config", help="alternative config.yaml")
    ap.add_argument("--backend", choices=["auto", "sentinelhub", "openeo", "local"], help="Sentinel-1 processing backend")
    ap.add_argument("--res", type=float, help="grid resolution in metres (default from config)")
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--no-optical", action="store_true", help="skip Sentinel-2 (saves processing units in tests)")
    ap.add_argument("--name", help="run folder name")
    ap.add_argument("--pre-stack", type=int, help="number of before-images averaged as the change reference (test)")
    ap.add_argument("--tracks", type=int, help="maximum number of orbit tracks to use (test)")
    a = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    setup_logging()
    ov = {}
    if a.backend:
        ov.setdefault("sentinel1", {})["backend"] = a.backend
    if a.res:
        ov.setdefault("grid", {})["resolution_m"] = a.res
    if a.no_model:
        ov.setdefault("model", {})["enabled"] = False
    if a.before_date:
        ov["event"] = {"before_date": a.before_date}
    if a.no_optical:
        ov.setdefault("sentinel2", {})["enabled"] = False
    if a.pre_stack:
        ov.setdefault("sentinel1", {})["pre_stack_images"] = a.pre_stack
    if a.tracks:
        ov.setdefault("sentinel1", {})["max_pairs"] = a.tracks
    cfg = load_config(a.config, ov)
    if a.place and not a.bbox:
        from floodsight.fetch.geocode import search_places
        hits = search_places(a.place, cfg["grid"]["max_area_km2"])
        if not hits:
            print(f"No place found for '{a.place}'. Add the district/state and country, or give --bbox.")
            return 2
        print(f"Places matching '{a.place}'" + ("" if hits[0]["source"] == "typed" else
              " (Nominatim, data © OpenStreetMap contributors)") + ":\n")
        for i, h in enumerate(hits, 1):
            box = ",".join(f"{v:.4f}" for v in h["bbox"])
            print(f" {i}. {h['name']}  [{h['kind']}]")
            print(f"    box ({h['bbox_from']}, {h['bbox_km2']:.0f} km²): --bbox {box}")
        print("\nA place outline is not the flood area: check the box covers the river/valley that flooded, "
              "then run again with --bbox.")
        return 0
    if not a.demo and not a.bbox:
        ap.error("--bbox (or --place to find one) is required, or use --demo")

    from floodsight.pipeline import NoData, run

    try:
        res = run(a.bbox, a.date, cfg, demo=a.demo, name=a.name)
    except NoData as e:
        print(f"\nNO DATA: {e}")
        return 2
    s = {k: res[k] for k in ("run_id", "verdict", "areas", "damage", "access")}
    s["reports"] = res["reports"]
    s["seconds"] = res["timings"]["total"]
    print(json.dumps(s, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
