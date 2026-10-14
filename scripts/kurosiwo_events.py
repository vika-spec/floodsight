"""Where and when is each Kuro Siwo flood event, and was the model trained on it? (read-only)

    python scripts/kurosiwo_events.py --root D:\\data\\KuroSiwo\\GRD\\data --weights models\\siamese_unet.pt

For every event (and every area inside it) this prints: what role the event had when the model was trained
(train / val / test, read from the weights file), tile count, share of flood pixels, the bounding box of the
tiles, the image dates found in the tile metadata, and a ready `run.py` command for the most flooded part, using the SAME
after-image date (MS1) and before-image date (SL1, via --before-date) as Kuro Siwo.

Why: Kuro Siwo labels are an ALLOWED reference (hand-checked flood masks on Sentinel-1). Running the full
FloodSight pipeline on an event the model has NEVER seen, then scoring it against these labels with
`python -m floodsight.validate.kurosiwo`, is an honest test of the whole system on a new area.
Output also goes to kurosiwo_events.json.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DATE_RE = re.compile(r"(20\d\d)[-_]?([01]\d)[-_]?([0-3]\d)")


def model_roles(weights):
    if not weights or not Path(weights).exists():
        return {}
    import torch
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    roles = {}
    for role, evs in (ck.get("split") or {}).items():
        for e in evs:
            roles[str(e)] = role
    return roles


def dates_in(obj, out, key=""):
    """Collect date-looking strings from nested JSON with the key they sit under."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            dates_in(v, out, f"{key}.{k}" if key else str(k))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            dates_in(v, out, key)
    elif isinstance(obj, str):
        m = DATE_RE.search(obj)
        if m:
            out[key].add(f"{m.group(1)}-{m.group(2)}-{m.group(3)}")


def tile_info(s):
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.open(s["post_vv"]) as src:
        if src.crs is None:
            return None
        return transform_bounds(src.crs, "EPSG:4326", *src.bounds)


def box_area_km2(b):
    w, s, e, n = b
    return (e - w) * 111.32 * math.cos(math.radians((s + n) / 2)) * (n - s) * 110.57


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--weights", default="models/siamese_unet.pt")
    ap.add_argument("--max-box-km", type=float, default=12.0, help="side of the suggested run box")
    ap.add_argument("--label-sample", type=int, default=400, help="tiles per area read for the flood share")
    a = ap.parse_args(argv)

    from floodsight.model.dataset import _read, find_samples

    roles = model_roles(a.weights)
    print(f"Model split read from {a.weights}: " + (", ".join(f"{r}={sorted(e for e, x in roles.items() if x == r)}"
          for r in ("train", "val", "test")) if roles else "NOT FOUND (roles shown as '?')"))
    print("Scanning Kuro Siwo (metadata only, may take a minute)...\n")
    samples = find_samples(a.root)
    groups = defaultdict(list)
    for s in samples:
        groups[(s["event"], Path(s["folder"]).parent.name)].append(s)

    rng = random.Random(0)
    out = []
    shown_json = set()
    for (ev, aoi), tiles in sorted(groups.items(), key=lambda kv: (len(kv[0][0]), kv[0])):
        dates = defaultdict(set)
        for s in tiles[:50]:
            for f in Path(s["folder"]).iterdir():
                m = DATE_RE.search(f.name)
                if m:
                    dates[f"file:{f.name.split('_')[0]}_{f.name.split('_')[1] if '_' in f.name else ''}"].add(
                        f"{m.group(1)}-{m.group(2)}-{m.group(3)}")
                if f.suffix == ".json":
                    try:
                        j = json.loads(f.read_text())
                    except Exception:  # noqa: BLE001
                        continue
                    dates_in(j, dates)
                    if ev not in shown_json:
                        shown_json.add(ev)
                        print(f"[event {ev}] example tile metadata keys: {list(j.keys()) if isinstance(j, dict) else type(j).__name__}"
                              + (f" / info keys: {list(j['info'].keys())}" if isinstance(j, dict) and isinstance(j.get('info'), dict) else ""))
        boxes, flood = [], []
        pick = set(map(id, rng.sample(tiles, min(a.label_sample, len(tiles)))))
        for s in tiles:
            b = tile_info(s)
            if b is None:
                continue
            fr = None
            if id(s) in pick:
                lab, valid = _read(s["label"]), _read(s["valid"]) == 1
                fr = float(((lab == 2) & valid).sum() / max(valid.sum(), 1))
            boxes.append(b)
            flood.append(fr)
        if not boxes:
            print(f"event {ev} area {aoi}: tiles have no georeference, skipped")
            continue
        B = np.array(boxes)
        full = [float(B[:, 0].min()), float(B[:, 1].min()), float(B[:, 2].max()), float(B[:, 3].max())]
        known = [(b, f) for b, f in zip(boxes, flood) if f is not None]
        wet = [b for b, f in known if f > 0.05] or [b for b, _ in known]
        c = np.array([((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b in wet])
        cx, cy = float(np.median(c[:, 0])), float(np.median(c[:, 1]))
        hx = a.max_box_km / 2 / (111.32 * math.cos(math.radians(cy)))
        hy = a.max_box_km / 2 / 110.57
        sug = [max(cx - hx, full[0]), max(cy - hy, full[1]), min(cx + hx, full[2]), min(cy + hy, full[3])]
        flood_pct = 100 * float(np.mean([f for _, f in known])) if known else float("nan")
        dsum = {k: sorted(v) for k, v in dates.items()}
        post_dates = sorted({d for k, v in dsum.items() if "MS1" in k or "flood" in k.lower() or "post" in k.lower() for d in v})
        row = {"event": ev, "area": aoi, "model_role": roles.get(str(ev), "?" if not roles else "not used"),
               "tiles": len(tiles), "flood_pct": round(flood_pct, 1), "tiles_bbox": [round(x, 4) for x in full],
               "tiles_bbox_km2": round(box_area_km2(full)), "suggested_bbox": [round(x, 4) for x in sug],
               "suggested_km2": round(box_area_km2(sug)), "dates_found": dsum, "post_dates": post_dates}
        out.append(row)
        print(f"event {ev:>8} area {aoi:<10} role={row['model_role']:<9} tiles={len(tiles):<6} flood={flood_pct:5.1f}%  "
              f"box {row['tiles_bbox_km2']} km²")
        for k, v in dsum.items():
            print(f"      dates  {k}: {', '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
        ms1 = sorted({x for k, v in dsum.items() if k.startswith("file:MS1") for x in v})
        sl1 = sorted({x for k, v in dsum.items() if k.startswith("file:SL1") for x in v})
        d = ms1[0] if ms1 else (post_dates[0] if post_dates else "<post-image date>")
        b = f" --before-date {sl1[-1]}" if sl1 else ""
        row["run_date"], row["run_before_date"] = d, (sl1[-1] if sl1 else None)
        print(f"      try:   python run.py --bbox {','.join(f'{x:.4f}' for x in sug)} --date {d}{b} --res 10 "
              f"--no-optical --name ks_{ev}_{aoi}   ({row['suggested_km2']} km²)\n")

    Path("kurosiwo_events.json").write_text(json.dumps(out, indent=1))
    print("Saved kurosiwo_events.json")
    print("Use for an honest test only events whose role is 'test' or 'not used'. 'train'/'val' events were seen "
          "during training or model selection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
