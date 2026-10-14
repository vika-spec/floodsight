"""Rehearse live evaluation: run the full pipeline on several events and print a scorecard.

    python scripts/rehearse.py                 # all presets (needs internet + CDSE login)
    python scripts/rehearse.py --demo          # offline check of the whole chain
    python scripts/rehearse.py --only Teesta Melamchi

A pass means: the run finished, produced both PDFs, and (for the control case) reported no flood.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from floodsight.core import load_config, setup_logging  # noqa: E402
from floodsight.pipeline import NoData, run  # noqa: E402
from floodsight.presets import PRESETS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    setup_logging()
    cfg = load_config()
    cases = {"Demo valley": {"bbox": None, "date": "2026-08-26", "demo": True}} if a.demo else \
        {k: v for k, v in PRESETS.items() if not a.only or any(o.lower() in k.lower() for o in a.only)}
    rows = []
    for name, c in cases.items():
        t0 = time.time()
        try:
            r = run(c["bbox"], c["date"], cfg, demo=c.get("demo", False))
            expect_none = name.startswith("Control")
            ok = all(Path("runs", r["run_id"], p).exists() for p in (r["reports"]["en"], r["reports"]["ne"]))
            if expect_none:
                ok &= r["verdict"] == "no_significant_change"
            rows.append((name, "PASS" if ok else "CHECK", r["verdict"], r["areas"]["flood_km2"] + r["areas"]["debris_km2"],
                         r["access"].get("settlements_cut_off", 0), time.time() - t0, ""))
        except NoData as e:
            rows.append((name, "NO DATA", "-", 0, 0, time.time() - t0, str(e)[:70]))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            rows.append((name, "FAIL", "-", 0, 0, time.time() - t0, f"{type(e).__name__}: {str(e)[:60]}"))
    print("\n" + "=" * 110)
    print(f"{'Case':52} {'Result':8} {'Verdict':22} {'km² hit':>8} {'cut off':>8} {'sec':>6}")
    for n, res, v, km, co, s, msg in rows:
        print(f"{n[:52]:52} {res:8} {v:22} {km:8.2f} {co:8d} {s:6.0f}  {msg}")
    return 0 if all(r[1] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
