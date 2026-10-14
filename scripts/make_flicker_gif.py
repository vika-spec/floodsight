"""Make the before/after "flicker" animation of a finished run as a GIF (for the README or a slide).

    python scripts/make_flicker_gif.py trishuli_shub                 # Sentinel-2 optical + radar
    python scripts/make_flicker_gif.py trishuli_shub --out docs/img

Uses the close-up pictures the run already made (runs/<run>/layers/zoom_*.png): the same pictures the web app
flickers, with FloodSight's detection outlined in yellow. Nothing is recomputed. Each frame is labelled with its
date so a reader always knows which image is showing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT = ROOT / "floodsight" / "report" / "fonts" / "FloodSightSans-Bold.ttf"


def _label(im: Image.Image, text: str, colour) -> Image.Image:
    im = im.convert("RGB")
    d = ImageDraw.Draw(im)
    size = max(16, im.width // 22)
    font = ImageFont.truetype(str(FONT), size) if FONT.exists() else ImageFont.load_default()
    pad = size // 2
    w = d.textlength(text, font=font)
    d.rectangle([pad, pad, pad + w + 2 * pad, pad + size + pad], fill=colour)
    d.text((2 * pad, pad + pad // 3), text, font=font, fill="white")
    return im


def make(run: Path, before: str, after: str, labels: tuple[str, str], out: Path, ms: int = 1000) -> Path | None:
    a, b = run / "layers" / f"zoom_{before}.png", run / "layers" / f"zoom_{after}.png"
    if not (a.exists() and b.exists()):
        print(f"  skipped {out.name}: {a.name} or {b.name} not in this run")
        return None
    f0 = _label(Image.open(a), labels[0], (23, 36, 46))
    f1 = _label(Image.open(b), labels[1], (160, 30, 30))
    out.parent.mkdir(parents=True, exist_ok=True)
    f0.save(out, save_all=True, append_images=[f1], duration=ms, loop=0, optimize=True)
    print(f"  {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", help="run folder name inside runs/ (e.g. trishuli_shub)")
    ap.add_argument("--out", default="docs/img")
    a = ap.parse_args(argv)
    run = ROOT / "runs" / a.run
    if not run.exists():
        raise SystemExit(f"No run folder {run}")
    radar = ("Before (radar)", "After (radar)")
    try:
        from datetime import date
        r = json.loads((run / "results.json").read_text(encoding="utf-8"))
        pair = r["inputs"]["sentinel1_pairs"][0]
        fmt = lambda d: date.fromisoformat(d).strftime("%d %b %Y").lstrip("0")
        radar = (f"Before, {fmt(pair['pre_date'])}", f"After, {fmt(pair['post_date'])}")
    except Exception:  # noqa: BLE001 - labels without dates are still correct
        pass
    out = ROOT / a.out
    make(run, "opt_before", "opt_after", ("Before (Sentinel-2)", "After (Sentinel-2)"), out / f"{a.run}_flicker_optical.gif")
    make(run, "before", "after", radar, out / f"{a.run}_flicker_radar.gif")


if __name__ == "__main__":
    main()