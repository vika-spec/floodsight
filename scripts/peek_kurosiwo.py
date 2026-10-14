"""Find which flood events are inside each Kuro Siwo GRD shard WITHOUT downloading the shards.

A .tar file stores each file's path in a 512-byte header in front of the file. This script reads a few
small pieces (default 16 x 256 KB) spread through every shard using HTTP range requests, finds the tar
headers in them, and reads the event (activation) id and dates from the file names, e.g.
    GRD/data/1111009/01/<tile>/MS1_IVV_1111009_01_20220911.tif  -> event 1111009, AOI 01, date 2022-09-11

Total download: about 35 shards x 16 x 256 KB = ~140 MB instead of ~700 GB.

    python scripts/peek_kurosiwo.py                      # all 35 GRD shards
    python scripts/peek_kurosiwo.py --shards 0 8 16      # only some
    python scripts/peek_kurosiwo.py --samples 32         # look at more places per shard (more reliable)

Because only samples are read, a small event squeezed between two sample points can be missed; the
list is a guide for choosing shards, and `--inspect` after downloading is the final check.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import requests

BASE = "https://huggingface.co/datasets/orion-ai-lab/Kuro-Siwo-GeoTIFFs/resolve/main/GRD/{:05d}.tar"
NAME_RE = re.compile(r"(?:MS1|SL1|SL2|MK0)_[A-Z]{3}_(\d+)_(\d+)_(\d{8})")
PATH_RE = re.compile(r"data[/\\](\d+)[/\\](\d+)[/\\]")


def shard_size(session, url) -> int:
    r = session.head(url, allow_redirects=True, timeout=60)
    r.raise_for_status()
    size = int(r.headers.get("Content-Length") or r.headers.get("x-linked-size") or 0)
    if not size:
        raise RuntimeError("could not read shard size")
    return size


def read_range(session, url, start, length) -> bytes:
    r = session.get(url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=120,
                    allow_redirects=True)
    r.raise_for_status()
    if r.status_code != 206:
        raise RuntimeError("server ignored the byte range (would download the whole file); aborting")
    return r.content


def tar_names(buf: bytes, base_offset: int) -> list[str]:
    """Scan a chunk for tar headers. Headers sit on 512-byte boundaries of the archive."""
    names = []
    first = (-base_offset) % 512
    for off in range(first, len(buf) - 512 + 1, 512):
        blk = buf[off:off + 512]
        if blk[257:262] != b"ustar":
            continue
        name = blk[0:100].split(b"\0", 1)[0].decode("utf-8", "replace")
        prefix = blk[345:500].split(b"\0", 1)[0].decode("utf-8", "replace")
        typeflag = blk[156:157]
        if typeflag == b"L":            # GNU long name: real name is in the next block
            name = buf[off + 512:off + 1024].split(b"\0", 1)[0].decode("utf-8", "replace")
        elif typeflag in (b"x", b"g"):  # PAX header: "NN path=...\n" records in the next block(s)
            m = re.search(rb"\d+ path=([^\n]+)\n", buf[off + 512:off + 2048])
            if m:
                name = m.group(1).decode("utf-8", "replace")
        names.append(f"{prefix}/{name}" if prefix else name)
    return names


def event_of(name: str):
    m = NAME_RE.search(name)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = PATH_RE.search(name)
    if m:
        return m.group(1), m.group(2), None
    return None


def peek(shard: int, samples: int, chunk: int) -> dict:
    s = requests.Session()
    url = BASE.format(shard)
    size = shard_size(s, url)
    events = defaultdict(lambda: {"aois": set(), "dates": set(), "hits": 0})
    starts = [0] + [((size * i) // samples) // 512 * 512 for i in range(1, samples)]
    for st in starts:
        buf = read_range(s, url, st, min(chunk, size - st))
        for n in tar_names(buf, st):
            ev = event_of(n)
            if ev:
                e = events[ev[0]]
                e["aois"].add(ev[1])
                if ev[2]:
                    e["dates"].add(ev[2])
                e["hits"] += 1
    return {"shard": shard, "size_gb": size / 1e9, "events": events}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, nargs="*", default=list(range(35)))
    ap.add_argument("--samples", type=int, default=16, help="pieces read per shard")
    ap.add_argument("--chunk-kb", type=int, default=256)
    a = ap.parse_args()

    results = []
    with ThreadPoolExecutor(6) as ex:
        futs = {ex.submit(peek, sh, a.samples, a.chunk_kb * 1024): sh for sh in a.shards}
        for f, sh in futs.items():
            try:
                results.append(f.result())
            except Exception as e:  # noqa: BLE001
                print(f"shard {sh:05d}: failed ({type(e).__name__}: {str(e)[:80]})", file=sys.stderr)

    all_events = defaultdict(list)
    print(f"\n{'shard':>6} {'GB':>5}  events found (activation id: AOIs, image dates)")
    for r in sorted(results, key=lambda r: r["shard"]):
        parts = []
        for ev, info in sorted(r["events"].items()):
            all_events[ev].append(r["shard"])
            dates = sorted(info["dates"])
            dtxt = f"{dates[0]}..{dates[-1]}" if dates else "?"
            parts.append(f"{ev} (AOI {','.join(sorted(info['aois']))}; {dtxt})")
        print(f"{r['shard']:>6} {r['size_gb']:5.1f}  " + ("; ".join(parts) if parts else "(no headers found)"))

    print(f"\n{len(all_events)} distinct events seen. Event -> shards:")
    for ev, sh in sorted(all_events.items()):
        print(f"  {ev}: {', '.join(f'{x:05d}' for x in sh)}")
    print("\nTip: choose shards that together cover several different events, then verify with --inspect.")


if __name__ == "__main__":
    main()
    