"""Core utilities: config, analysis grid, raster I/O, run folders, timing, and the data-rule guard."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

log = logging.getLogger("floodsight")

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ---------------------------------------------------------------------------
# Data rules (the brief: published damage maps and post-event OSM are NOT inputs)
# ---------------------------------------------------------------------------

FORBIDDEN_INPUT_MARKERS = ("emsr", "copernicus_ems", "unosat", "validation_data", "damage_map")


class DataRuleViolation(RuntimeError):
    pass


def assert_allowed_input(path: str | Path) -> None:
    """Called on every file the pipeline reads. Validation data can never become an input."""
    p = str(path).lower().replace("\\", "/")
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in p:
            raise DataRuleViolation(
                f"Refusing to read '{path}': reference damage maps are for validation only (data rules)."
            )


def event_start(cfg: dict, flood_date: date) -> date:
    """First day that may already show flooding. Every "before" input (Sentinel-1 before-image, Sentinel-2
    before-composite, OSM snapshot) must be strictly earlier than this day.

    Default: the flood date. If the operator sets event.before_date (CLI --before-date), the area is taken as
    still dry on that day, so the event starts the day after. Needed for floods that build up over days or
    weeks (monsoon plains): otherwise the "before" image may already be flooded and that water is mapped as
    normal water."""
    b = (cfg.get("event") or {}).get("before_date")
    if not b:
        return flood_date
    b = parse_date(b)
    if b >= flood_date:
        raise ValueError(f"--before-date ({b}) must be earlier than the flood date ({flood_date}).")
    return date.fromordinal(b.toordinal() + 1)


def osm_snapshot_date(flood_date: date, latest_available: str) -> date:
    """OSM must be as it was BEFORE the event: the day before the flood, capped at what ohsome serves."""
    latest = date.fromisoformat(latest_available)
    day_before = date.fromordinal(flood_date.toordinal() - 1)
    snap = min(latest, day_before)
    if snap >= flood_date:
        raise DataRuleViolation("OSM snapshot must be earlier than the flood date.")
    return snap


# ---------------------------------------------------------------------------
# Analysis grid: one UTM grid shared by every layer, so pixels line up exactly
# ---------------------------------------------------------------------------

@dataclass
class AOI:
    west: float
    south: float
    east: float
    north: float

    @classmethod
    def parse(cls, text: str) -> "AOI":
        vals = [float(v) for v in str(text).replace(" ", "").split(",")]
        if len(vals) != 4:
            raise ValueError("bbox must be 'west,south,east,north' in degrees")
        a = cls(*vals)
        a.validate()
        return a

    def validate(self) -> None:
        if not (-180 <= self.west < self.east <= 180 and -85 <= self.south < self.north <= 85):
            raise ValueError(f"Invalid bbox {self.as_list()} (expected west<east, south<north, degrees)")

    def as_list(self) -> list[float]:
        return [self.west, self.south, self.east, self.north]

    def buffered(self, km: float) -> "AOI":
        dlat = km / 111.0
        dlon = km / (111.0 * max(math.cos(math.radians((self.south + self.north) / 2)), 0.2))
        return AOI(self.west - dlon, self.south - dlat, self.east + dlon, self.north + dlat)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.west + self.east) / 2, (self.south + self.north) / 2)

    def utm_epsg(self) -> int:
        lon, lat = self.center
        zone = int((lon + 180) // 6) + 1
        return (32600 if lat >= 0 else 32700) + zone

    def area_km2(self) -> float:
        lat = math.radians((self.south + self.north) / 2)
        return (self.east - self.west) * 111.32 * math.cos(lat) * (self.north - self.south) * 110.57


@dataclass
class Grid:
    crs: str
    transform: Any  # affine.Affine
    width: int
    height: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def res(self) -> float:
        return float(self.transform.a)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        t = self.transform
        return (t.c, t.f + t.e * self.height, t.c + t.a * self.width, t.f)

    @classmethod
    def for_aoi(cls, aoi: AOI, resolution_m: float) -> "Grid":
        from pyproj import Transformer
        from rasterio.transform import from_origin

        epsg = aoi.utm_epsg()
        tr = Transformer.from_crs(4326, epsg, always_xy=True)
        xs, ys = tr.transform(
            [aoi.west, aoi.east, aoi.west, aoi.east], [aoi.south, aoi.south, aoi.north, aoi.north]
        )
        x0 = math.floor(min(xs) / resolution_m) * resolution_m
        y1 = math.ceil(max(ys) / resolution_m) * resolution_m
        w = int(math.ceil((max(xs) - x0) / resolution_m))
        h = int(math.ceil((y1 - min(ys)) / resolution_m))
        return cls(f"EPSG:{epsg}", from_origin(x0, y1, resolution_m, resolution_m), w, h)

    def to_dict(self) -> dict:
        return {"crs": self.crs, "transform": list(self.transform)[:6], "width": self.width, "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        from affine import Affine

        return cls(d["crs"], Affine(*d["transform"]), d["width"], d["height"])

    def xy_to_rowcol(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t = self.transform   # north-up grid (no rotation)
        c = (np.asarray(x, float) - t.c) / t.a
        r = (np.asarray(y, float) - t.f) / t.e
        # a point that cannot be projected (NaN/inf) is reported as row/col -1: always outside the grid,
        # never an arbitrary integer that could land inside it
        bad = ~(np.isfinite(r) & np.isfinite(c))
        r = np.where(bad, -1.0, np.floor(np.where(bad, 0.0, r)))
        c = np.where(bad, -1.0, np.floor(np.where(bad, 0.0, c)))
        return r.astype(int), c.astype(int)

    def rowcol_to_xy(self, r, c):
        t = self.transform
        return t.c + (np.asarray(c, float) + 0.5) * t.a, t.f + (np.asarray(r, float) + 0.5) * t.e

    def lonlat_to_xy(self, lon, lat):
        from pyproj import Transformer

        return Transformer.from_crs(4326, self.crs, always_xy=True).transform(lon, lat)


# ---------------------------------------------------------------------------
# Raster I/O
# ---------------------------------------------------------------------------

def write_raster(path: str | Path, arr: np.ndarray, grid: Grid, nodata=None) -> Path:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(arr)
    bands = arr[None] if arr.ndim == 2 else arr
    dtype = bands.dtype if bands.dtype != bool else np.uint8
    with rasterio.open(
        path, "w", driver="GTiff", height=grid.height, width=grid.width, count=bands.shape[0],
        dtype=dtype, crs=grid.crs, transform=grid.transform, nodata=nodata,
        compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as dst:
        dst.write(bands.astype(dtype))
    return path


def read_to_grid(path: str | Path, grid: Grid, band: int | None = None, resampling: str = "bilinear",
                 src_nodata=None) -> np.ndarray:
    """Read any raster and reproject it onto the analysis grid (float32, NaN = no data)."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    assert_allowed_input(path)
    with rasterio.open(path) as src:
        if src.crs is None or src.transform.is_identity:
            # without a map position the data would be placed at the wrong location: never continue silently
            raise ValueError(f"{Path(path).name} has no map position (CRS/geotransform); delete it so it is "
                             "downloaded again")
        idx = [band] if band else list(range(1, src.count + 1))
        out = np.full((len(idx), grid.height, grid.width), np.nan, np.float32)
        nod = src_nodata if src_nodata is not None else src.nodata
        for i, b in enumerate(idx):
            reproject(
                source=rasterio.band(src, b), destination=out[i],
                src_transform=src.transform, src_crs=src.crs, src_nodata=nod,
                dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=np.nan,
                resampling=getattr(Resampling, resampling),
            )
    return out[0] if band else out


# ---------------------------------------------------------------------------
# Run context: folder layout, caching, stage timing, progress callbacks
# ---------------------------------------------------------------------------

def run_id_for(aoi: AOI, flood_date: date, cfg: dict) -> str:
    k = {"aoi": [round(v, 5) for v in aoi.as_list()], "date": flood_date.isoformat(),
         "res": cfg["grid"]["resolution_m"]}
    if (cfg.get("event") or {}).get("before_date"):          # a different before-image is a different run
        k["before"] = str(cfg["event"]["before_date"])
    key = json.dumps(k, sort_keys=True)
    return f"{flood_date.isoformat()}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"


@dataclass
class RunContext:
    aoi: AOI
    flood_date: date
    cfg: dict
    runs_dir: Path = ROOT / "runs"
    name: str | None = None
    progress: Callable[[str, float], None] | None = None
    timings: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)   # human-readable caveats that go into the report

    def __post_init__(self):
        self.run_id = self.name or run_id_for(self.aoi, self.flood_date, self.cfg)
        self.dir = Path(self.runs_dir) / self.run_id
        for sub in ("inputs", "layers", "vectors", "report"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.grid = Grid.for_aoi(self.aoi, self.cfg["grid"]["resolution_m"])

    def path(self, sub: str, name: str) -> Path:
        return self.dir / sub / name

    def note(self, text: str) -> None:
        if text not in self.notes:
            log.info("NOTE: %s", text)
            self.notes.append(text)

    @contextmanager
    def stage(self, name: str, frac: float = 0.0):
        if self.progress:
            self.progress(name, frac)
        log.info("▶ %s", name)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            self.timings[name] = round(dt, 2)
            log.info("✓ %s (%.1fs)", name, dt)


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("fontTools", "fpdf", "rasterio", "urllib3", "PIL", "matplotlib", "openeo", "openeo.rest", "pyogrio", "fiona"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def parse_date(s: str | date) -> date:
    if isinstance(s, date):
        return s
    return datetime.strptime(s, "%Y-%m-%d").date()
