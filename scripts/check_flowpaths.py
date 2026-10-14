"""Independent check of the flood-path bonus on several valleys (list fixed BEFORE any run).

For each start point the path is traced on the Copernicus DEM and compared with rivers mapped in
OpenStreetMap (pre-event snapshot), which the path tracing never uses. Reported per valley: share of the
path within 150 m of a mapped river, the named rivers it follows, length, windows and time.

    python scripts/check_flowpaths.py                    # default valleys below
    python scripts/check_flowpaths.py --point "My valley:85.36,28.27" --km 50

Start points are approximate town or river locations; a click within 300 m of the river snaps onto it.
Nothing is tuned on these results: they go into the report as they come out.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# name, lon, lat, date for the OSM snapshot (any past date: only the river map is used)
DEFAULT = [
    ("Trishuli (Rasuwa, case study start)", 85.36, 28.27, "2026-08-26"),
    ("Bhote Koshi at Tatopani, Nepal", 85.94, 27.95, "2025-07-08"),
    ("Melamchi at Melamchi Bazaar, Nepal", 85.577, 27.828, "2021-06-15"),
    ("Teesta at Chungthang, Sikkim", 88.645, 27.603, "2023-10-04"),
    ("Beas at Manali, Himachal", 77.19, 32.24, "2023-07-10"),
    ("Kosi at Chatara (plains exit), Nepal", 87.16, 26.82, "2024-09-28"),
]


def main(argv=None):
    from floodsight.analysis.flowtrace import flow_trace
    from floodsight.core import load_config, parse_date, setup_logging

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--point", action="append", help='"name:lon,lat" (repeatable); default: the fixed list')
    ap.add_argument("--km", type=float, default=50)
    ap.add_argument("--out", default="validation/flowpath_check.json")
    a = ap.parse_args(argv)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    setup_logging()
    pts = DEFAULT
    if a.point:
        pts = []
        for p in a.point:
            nm, xy = p.rsplit(":", 1)
            lon, lat = (float(v) for v in xy.split(","))
            pts.append((nm, lon, lat, date.today().isoformat()))
    cfg = load_config()
    rows = []
    for nm, lon, lat, d in pts:
        try:
            o = flow_trace(lon, lat, cfg, a.km, parse_date(d))
            rows.append({"valley": nm, "start": [lon, lat], "length_km": o["length_km"], "drop_m": o["drop_m"],
                         "on_mapped_river": o["path_on_mapped_river"], "rivers_followed": o["rivers_followed"],
                         "windows": o["windows"], "seconds": o["seconds"], "stop": o["stop_reason"],
                         "settlements": len(o["settlements"])})
        except Exception as e:  # noqa: BLE001 - report and continue
            rows.append({"valley": nm, "start": [lon, lat], "error": str(e)[:200]})
    print(f"\n{'valley':42s} {'km':>6s} {'on river':>9s} {'windows':>8s} {'sec':>6s}  rivers followed")
    for r in rows:
        if "error" in r:
            print(f"{r['valley']:42s}  ERROR {r['error']}")
            continue
        share = "n/a" if r["on_mapped_river"] is None else f"{100 * r['on_mapped_river']:.0f}%"
        riv = ", ".join(f"{x['name']} {x['from_km']}-{x['to_km']}" for x in r["rivers_followed"]) or "-"
        print(f"{r['valley']:42s} {r['length_km']:6.1f} {share:>9s} {r['windows']:8d} {r['seconds']:6.1f}  {riv}")
    ok = [r["on_mapped_river"] for r in rows if r.get("on_mapped_river") is not None]
    if ok:
        print(f"\nMedian share on a mapped river: {100 * sorted(ok)[len(ok) // 2]:.0f}% over {len(ok)} valleys")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=1))
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
