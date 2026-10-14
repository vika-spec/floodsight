"""Copy finished runs into showcase/ for the public deployment (lightweight: no raw satellite inputs).

    python scripts/make_showcase.py trishuli_shub slo_meza demo_2026-08-26

Copies runs/<name>/ without inputs/ (raw downloaded images, large) into showcase/<name>/. Everything the web
app shows is kept: results.json, map layers and PNGs, vectors (incl. priority list), PDFs, validation.
Prints the size so you can stay within the host's limits.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(names):
    if not names:
        print(__doc__)
        return 1
    out = ROOT / "showcase"
    out.mkdir(exist_ok=True)
    for n in names:
        src = ROOT / "runs" / n
        if not (src / "results.json").exists():
            print(f"skip {n}: no finished run at {src}")
            continue
        dst = out / n
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("inputs", "*.part", "__pycache__"))
        mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1e6
        print(f"{n}: {mb:.1f} MB")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1e6
    print(f"showcase/ total {total:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
