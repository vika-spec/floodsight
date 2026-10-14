"""Static map for the situation report (matplotlib, in the analysis UTM grid)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..detect.baseline import DEBRIS, FLOOD, NODATA, PERMANENT

STATUS_COLORS = {"cut_off": "#d62728", "likely_cut_off": "#e4573f", "unknown": "#7a6fa8",
                 "detour": "#f0a202", "reachable": "#2ca02c",
                 "isolated_before": "#7f7f7f", "no_road_data": "#bbbbbb"}


MAX_HOSPITALS = 8


def render_map(run_dir: Path, results: dict, out_png: Path, width_in=7.4, max_h_in=4.6, dpi=200) -> Path:
    import geopandas as gpd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import rasterio
    from matplotlib import font_manager
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # labels can be Nepali (Devanagari); use the report's bundled font, which has those glyphs
    fonts = Path(__file__).resolve().parent / "fonts"
    for f in ("FloodSightSans-Regular.ttf", "FloodSightSans-Bold.ttf"):
        if (fonts / f).exists():
            font_manager.fontManager.addfont(str(fonts / f))
    label_font = (font_manager.FontProperties(fname=str(fonts / "FloodSightSans-Regular.ttf")).get_name()
                  if (fonts / "FloodSightSans-Regular.ttf").exists() else None)

    with rasterio.open(run_dir / "layers" / "flood_class.tif") as src:
        cls = src.read(1)
        b = src.bounds
        crs = src.crs
    with rasterio.open(run_dir / "layers" / "confidence.tif") as src:
        conf = src.read(1)
    hydro = np.load(run_dir / "layers" / "hydro.npz")
    dem = hydro["dem"]

    # hillshade background
    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmean(dem)), results["grid"]["resolution_m"])
    az, alt = np.radians(315), np.radians(45)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)

    H, W = cls.shape
    aspect_ratio = H / W
    h_in = min(max_h_in, width_in * aspect_ratio)
    fig, ax = plt.subplots(figsize=(width_in, h_in), dpi=dpi)
    ext = (b.left, b.right, b.bottom, b.top)
    ax.imshow(hs, cmap="gray", extent=ext, vmin=-0.2, vmax=1.1, interpolation="bilinear")
    rgba = np.zeros((H, W, 4), np.float32)
    col = {FLOOD: (0.12, 0.43, 0.9), DEBRIS: (0.59, 0.37, 0.16), PERMANENT: (0.47, 0.67, 0.86)}
    for k, c in col.items():
        m = cls == k
        rgba[m, :3] = c
        rgba[m, 3] = 0.55 if k == PERMANENT else np.array([0, 0.55, 0.75, 0.92])[np.clip(conf[m], 0, 3)]
    nod = cls == NODATA
    rgba[nod] = (0.5, 0.5, 0.5, 0.35)
    ax.imshow(rgba, extent=ext, interpolation="nearest")

    def load(key):
        p = results.get("vectors", {}).get(key)
        if not p or not (run_dir / p).exists():
            return None
        g = gpd.read_file(run_dir / p)
        return None if g.empty else g.to_crs(crs)

    br = load("broken_roads")
    if br is not None:
        br.plot(ax=ax, color="#d62728", linewidth=2.2, zorder=4)
    rd = load("roads")
    if rd is not None and "is_bridge" in rd:
        bridges = rd[rd["is_bridge"].astype(str).str.lower() == "true"]
        if not bridges.empty:
            bridges.plot(ax=ax, color="black", linewidth=3.0, zorder=5)
    st = load("settlements")
    hl = load("health")
    if hl is not None:
        # only the facilities that settlements with lost or changed access depend on (before or after the flood),
        # at most MAX_HOSPITALS, labelled; the full list is in report/health_facilities.csv
        need = set()
        if st is not None:
            aff = st[st["status"].isin(["cut_off", "likely_cut_off", "unknown", "detour"])]
            for tcol in ("target_before", "target_after"):
                if tcol in aff:
                    need |= {str(v) for v in aff[tcol].dropna() if str(v) not in ("", "nan", "None")}
        hl = hl[hl["name"].astype(str).isin(need)].head(MAX_HOSPITALS) if "name" in hl else hl.iloc[0:0]
        if not hl.empty:
            hl.plot(ax=ax, marker="P", color="white", edgecolor="black", markersize=70, zorder=6)
            for _, r in hl.iterrows():
                p_ = r.geometry if r.geometry.geom_type == "Point" else r.geometry.representative_point()
                ax.annotate(str(r["name"])[:24], (p_.x, p_.y), xytext=(4, -9), textcoords="offset points",
                            fontsize=6, zorder=8, family=label_font, style="italic",
                            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.75))
    if st is not None:
        for s, c in STATUS_COLORS.items():
            sub = st[st["status"] == s]
            if not sub.empty:
                sub.plot(ax=ax, color=c, edgecolor="black", markersize=36, zorder=7)
        for _, r in st[st["status"].isin(["cut_off", "likely_cut_off", "unknown", "detour"])].head(12).iterrows():
            en = str(r.get("name_en", "") or "")
            label = en if en and en != "nan" else str(r.get("name", ""))   # English name when OSM has one
            ax.annotate(label[:22], (r.geometry.x, r.geometry.y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, zorder=8, family=label_font,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))

    ax.set_xlim(b.left, b.right)
    ax.set_ylim(b.bottom, b.top)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel(""); ax.set_ylabel("")
    # scale bar
    span = (b.right - b.left)
    sb = 10 ** np.floor(np.log10(span / 4))
    sb = sb * (2 if span / sb > 8 else 1)
    x0, y0 = b.left + span * 0.03, b.bottom + (b.top - b.bottom) * 0.05
    ax.plot([x0, x0 + sb], [y0, y0], color="black", lw=3, zorder=9)
    ax.text(x0 + sb / 2, y0 + (b.top - b.bottom) * 0.02, f"{sb / 1000:g} km", ha="center", fontsize=7, zorder=9)
    ax.annotate("N", xy=(0.97, 0.93), xycoords="axes fraction", ha="center", fontsize=9, weight="bold")
    ax.annotate("", xy=(0.97, 0.92), xytext=(0.97, 0.80), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="black"))

    handles = [Patch(color=col[FLOOD], label="Flood water"), Patch(color=col[DEBRIS], label="Debris / mud"),
               Patch(color=(0.5, 0.5, 0.5, 0.5), label="Radar blind / no data"),
               Line2D([], [], color="#d62728", lw=2.2, label="Road cut"),
               Line2D([], [], marker="P", ls="", mfc="white", mec="black", label="Hospital used by affected settlements"),
               Line2D([], [], marker="o", ls="", mfc=STATUS_COLORS["cut_off"], mec="black", label="Cut off"),
               Line2D([], [], marker="o", ls="", mfc=STATUS_COLORS["likely_cut_off"], mec="black", label="Likely cut off"),
               Line2D([], [], marker="o", ls="", mfc=STATUS_COLORS["unknown"], mec="black", label="Unknown"),
               Line2D([], [], marker="o", ls="", mfc=STATUS_COLORS["detour"], mec="black", label="Detour"),
               Line2D([], [], marker="o", ls="", mfc=STATUS_COLORS["reachable"], mec="black", label="Road OK")]
    ax.legend(handles=handles, loc="lower right", fontsize=6, framealpha=0.85, ncol=2)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
    return out_png
