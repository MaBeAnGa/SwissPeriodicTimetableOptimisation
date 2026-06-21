
#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Patch, FancyBboxPatch
from matplotlib.ticker import FuncFormatter, MultipleLocator

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures" / "part-II"
SPATIAL = ROOT / "part2_spatial_breakdown_data.csv"
MUNI_GEOM = ROOT / "swiss_municipality_station_geometries.json"
CANTON_BOUNDARIES = ROOT / "historical_swiss_boundaries.js"

YEARS = [2026, 2035]
BASE_VERSION = "v0"
FINAL_VERSION = "optimized_step2"
NEUTRAL_EPS_MIN = 0.002  # Same threshold as the app: about 120 ms.

CANTON_ORDER = [
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE",
    "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH",
]
CANTON_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    "#4d4d4d", "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39",
    "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363", "#9ecae1",
    "#fdae6b", "#74c476", "#bcbddc", "#969696", "#6baed6",
]
COLOR_BY_CANTON = dict(zip(CANTON_ORDER, CANTON_COLORS))

METRIC_COLUMNS = [
    ("Total", "avgTotalMin", "total"),
    ("Rolling", "avgRollingMin", "rolling"),
    ("Dwell", "avgDwellMin", "dwell"),
    ("Transfer", "avgTransferMin", "transfer"),
    ("Waiting", "avgWaitingMin", "waiting"),
]

BLUE = "#1e69cf"
RED = "#d62d34"
NEUTRAL = "#ece6db"
EDGE = "#ffffff"
BORDER = "#b8b8b8"
TEXT = "#202124"
MUTED = "#5f6368"
BACKGROUND = "#fbf7f0"


def load_boundaries() -> dict[str, Any]:
    text = CANTON_BOUNDARIES.read_text(encoding="utf-8")
    match = re.search(r"=\s*(\{.*\});?\s*$", text, re.S)
    if not match:
        raise RuntimeError(f"Could not parse {CANTON_BOUNDARIES}")
    return json.loads(match.group(1))


def read_data() -> pd.DataFrame:
    df = pd.read_csv(SPATIAL)
    df["year"] = df["year"].astype(int)
    return df


def metric_delta_rows(df: pd.DataFrame, year: int, level: str) -> pd.DataFrame:
    base = df[(df["level"] == level) & (df["year"] == year) & (df["version"] == BASE_VERSION)].copy()
    final = df[(df["level"] == level) & (df["year"] == year) & (df["version"] == FINAL_VERSION)].copy()
    key_cols = ["code"]
    keep_cols = [
        "code", "name", "canton", "originCount", "totalWeight", "lat", "lon",
        "avgTotalMin", "avgRollingMin", "avgDwellMin", "avgTransferMin", "avgWaitingMin",
    ]
    # CSV has lat/lon only for municipality/station rows. Missing columns are added for cantons.
    for col in keep_cols:
        if col not in base.columns:
            base[col] = np.nan
        if col not in final.columns:
            final[col] = np.nan
    merged = base[keep_cols].merge(
        final[keep_cols], on=key_cols, suffixes=("_base", "_final"), how="inner"
    )
    merged["name"] = merged["name_final"].fillna(merged["name_base"])
    merged["canton"] = merged["canton_final"].fillna(merged["canton_base"])
    merged["originCount"] = merged["originCount_base"].fillna(merged["originCount_final"])
    merged["totalWeight"] = merged["totalWeight_base"].fillna(merged["totalWeight_final"])
    merged["lat"] = merged["lat_final"].fillna(merged["lat_base"])
    merged["lon"] = merged["lon_final"].fillna(merged["lon_base"])
    for label, col, short in METRIC_COLUMNS:
        merged[f"delta{label}Min"] = merged[f"{col}_final"] - merged[f"{col}_base"]
        merged[f"base{label}Min"] = merged[f"{col}_base"]
        merged[f"final{label}Min"] = merged[f"{col}_final"]
    return merged


def swiss_delta(df: pd.DataFrame, year: int, col: str) -> float:
    base = df[(df["level"] == "canton") & (df["year"] == year) & (df["version"] == BASE_VERSION)]
    final = df[(df["level"] == "canton") & (df["year"] == year) & (df["version"] == FINAL_VERSION)]
    base = base.set_index("code")
    final = final.set_index("code")
    common = base.index.intersection(final.index)
    weight = base.loc[common, "totalWeight"].astype(float)
    return (final.loc[common, col].astype(float).mul(weight).sum() / weight.sum()) - (base.loc[common, col].astype(float).mul(weight).sum() / weight.sum())


def swiss_base(df: pd.DataFrame, year: int, col: str) -> float:
    base = df[(df["level"] == "canton") & (df["year"] == year) & (df["version"] == BASE_VERSION)].set_index("code")
    weight = base["totalWeight"].astype(float)
    return base[col].astype(float).mul(weight).sum() / weight.sum()


def lighten(hex_color: str, amount: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def delta_color(delta_min: float, max_abs_min: float) -> str:
    val = float(delta_min or 0.0)
    if abs(val) < NEUTRAL_EPS_MIN:
        return NEUTRAL
    strength = min(1.0, abs(val) / max(max_abs_min, 1e-9))
    # App-inspired blue/red, but mixed with white for print readability.
    amount = 0.72 - 0.50 * strength
    return lighten(BLUE if val < 0 else RED, max(0.16, min(0.78, amount)))


def format_seconds(delta_min: float, decimals: int = 1) -> str:
    sec = float(delta_min) * 60.0
    sign = "+" if sec > 0 else "−" if sec < 0 else ""
    sec_abs = abs(sec)
    if sec_abs >= 60:
        minutes = int(sec_abs // 60)
        rest = sec_abs - minutes * 60
        if rest < 0.05:
            return f"{sign}{minutes} min"
        return f"{sign}{minutes} min {rest:.{decimals}f} s"
    return f"{sign}{sec_abs:.{decimals}f} s"


def format_ms(delta_min: float) -> str:
    ms = float(delta_min) * 60_000.0
    sign = "+" if ms > 0 else "−" if ms < 0 else ""
    ms_abs = abs(ms)
    if ms_abs >= 1000:
        sec = ms_abs / 1000.0
        if sec >= 60:
            minute = int(sec // 60)
            rest = sec - minute * 60
            return f"{sign}{minute} min {rest:.1f} s"
        return f"{sign}{sec:.3f} s"
    return f"{sign}{ms_abs:.0f} ms"


def polygon_centroid(polygons: list[list[list[float]]]) -> tuple[float, float]:
    sx = sy = area_sum = 0.0
    fallback = []
    for poly in polygons:
        pts = np.asarray(poly, dtype=float)
        if len(pts) < 3:
            continue
        fallback.extend(poly)
        x = pts[:, 0]
        y = pts[:, 1]
        if x[0] != x[-1] or y[0] != y[-1]:
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area = 0.5 * np.sum(cross)
        if abs(area) < 1e-9:
            continue
        cx = np.sum((x[:-1] + x[1:]) * cross) / (6 * area)
        cy = np.sum((y[:-1] + y[1:]) * cross) / (6 * area)
        a = abs(float(area))
        sx += a * float(cx)
        sy += a * float(cy)
        area_sum += a
    if area_sum > 0:
        return sx / area_sum, sy / area_sum
    pts = np.asarray(fallback, dtype=float)
    return float(np.nanmean(pts[:, 0])), float(np.nanmean(pts[:, 1]))


def station_positions() -> dict[str, tuple[float, float]]:
    geo = json.loads(MUNI_GEOM.read_text(encoding="utf-8"))
    out = {}
    for name, rec in geo.get("stationMunicipalities", {}).items():
        east = rec.get("east")
        north = rec.get("north")
        if east is not None and north is not None:
            out[str(name)] = (float(east), float(north))
    return out


def station_extremes(df: pd.DataFrame, year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = metric_delta_rows(df, year, "station")
    best = rows.sort_values("deltaTotalMin", ascending=True).head(3).copy()
    worst = rows.sort_values("deltaTotalMin", ascending=False).head(3).copy()
    return best, worst


def set_swiss_map_extent(ax: plt.Axes, bounds: dict[str, float], pad: float = 8500) -> None:
    ax.set_xlim(float(bounds["minX"]) - pad, float(bounds["maxX"]) + pad)
    ax.set_ylim(float(bounds["minY"]) - pad, float(bounds["maxY"]) + pad)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_station_markers(ax: plt.Axes, best: pd.DataFrame, worst: pd.DataFrame, positions: dict[str, tuple[float, float]]) -> None:
    offsets = {
        # Small nudges where top/worst stations are very close together.
        "Schönenwerd SO": (-6000, 7000),
        "Däniken SO": (0, -6500),
        "Dulliken": (6500, 4500),
        "Flums": (6000, 6000),
        "Sugiez": (-6000, 5500),
        "Murten/Morat": (6500, -4500),
        "Wildegg": (-6000, 6000),
        "Bassersdorf": (5500, 5500),
        "Seebleiche": (6500, 3000),
        "Kloten": (-5000, 6000),
        "Kloten Balsberg": (5200, -4500),
        "Opfikon": (6500, 3500),
    }
    for group, rows, color, edge in [("best", best, BLUE, "#ffffff"), ("worst", worst, RED, "#ffffff")]:
        for idx, row in enumerate(rows.itertuples(index=False), start=1):
            station = str(row.code)
            pos = positions.get(station) or positions.get(str(row.name))
            if not pos:
                continue
            x, y = pos
            dx, dy = offsets.get(str(row.name), offsets.get(station, (0, 0)))
            # Short leader line keeps the exact station position visible while avoiding label overlap.
            ax.plot([x, x + dx], [y, y + dy], color=color, lw=0.8, alpha=0.45, zorder=22)
            ax.scatter([x], [y], s=26, color=color, edgecolor="white", linewidth=0.65, zorder=23)
            ax.scatter([x + dx], [y + dy], s=150, color=color, edgecolor=edge, linewidth=1.25, zorder=24)
            ax.text(
                x + dx, y + dy, str(idx), ha="center", va="center", fontsize=7.5,
                weight="bold", color="white", zorder=25,
                path_effects=[pe.withStroke(linewidth=0.6, foreground=color)],
            )


def draw_station_extreme_box(ax: plt.Axes, best: pd.DataFrame, worst: pd.DataFrame, year: int) -> None:
    """Draw a compact station-marker key in its own panel, not on top of the map."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    box = FancyBboxPatch(
        (0.010, 0.090), 0.980, 0.820,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        transform=ax.transAxes,
        facecolor="white",
        edgecolor="#e5e1d8",
        linewidth=0.8,
        alpha=0.98,
        zorder=1,
    )
    ax.add_patch(box)
    ax.text(0.035, 0.810, f"{year} station markers", transform=ax.transAxes,
            fontsize=8.8, weight="bold", color=TEXT, ha="left", va="top", zorder=2)

    col_specs = [
        ("Most improved", best, BLUE, 0.035, 0.455),
        ("Worst affected", worst, RED, 0.545, 0.965),
    ]
    for label, rows, color, tx, value_x in col_specs:
        ax.text(tx, 0.610, label, transform=ax.transAxes, fontsize=7.8, weight="bold", color=color,
                ha="left", va="top", zorder=2)
        for i, row in enumerate(rows.itertuples(index=False), start=1):
            yy = 0.610 - i * 0.150
            name = str(row.name)
            if len(name) > 20:
                name = name[:18] + "..."
            ax.text(tx, yy, f"{i}. {name}", transform=ax.transAxes, fontsize=7.4, color=TEXT,
                    weight="bold", ha="left", va="center", zorder=2)
            ax.text(value_x, yy, format_seconds(row.deltaTotalMin, decimals=1), transform=ax.transAxes,
                    fontsize=7.4, color=color, weight="bold", ha="right", va="center", zorder=2)


def draw_canton_map(ax: plt.Axes, df: pd.DataFrame, year: int, boundaries: dict[str, Any], positions: dict[str, tuple[float, float]]) -> None:
    rows = metric_delta_rows(df, year, "canton").set_index("code")
    max_abs = max(abs(rows["deltaTotalMin"]).max(), NEUTRAL_EPS_MIN)
    ax.set_facecolor(BACKGROUND)
    for canton in boundaries["cantons"]:
        code = str(canton.get("ak"))
        delta = float(rows.loc[code, "deltaTotalMin"]) if code in rows.index else 0.0
        face = delta_color(delta, max_abs)
        for poly in canton.get("polygons", []):
            ax.add_patch(Polygon(np.asarray(poly), closed=True, facecolor=face, edgecolor=EDGE, linewidth=0.85, zorder=3))
    for canton in boundaries["cantons"]:
        for poly in canton.get("polygons", []):
            ax.add_patch(Polygon(np.asarray(poly), closed=True, facecolor="none", edgecolor=BORDER, linewidth=0.55, zorder=4))
    best, worst = station_extremes(df, year)
    draw_station_markers(ax, best, worst, positions)
    ch = swiss_delta(df, year, "avgTotalMin")
    ax.text(0.020, 0.965, f"{year} canton origins", transform=ax.transAxes, ha="left", va="top", fontsize=12.0, weight="bold", color=TEXT)
    ax.text(0.980, 0.965, f"CH {format_ms(ch)}", transform=ax.transAxes, ha="right", va="top", fontsize=10.0, weight="bold", color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.2", facecolor="white", edgecolor="#e4e2dc", linewidth=0.8))
    set_swiss_map_extent(ax, boundaries["bounds"])


def draw_municipality_map(ax: plt.Axes, df: pd.DataFrame, year: int, geom: dict[str, Any], positions: dict[str, tuple[float, float]]) -> None:
    rows = metric_delta_rows(df, year, "municipality").set_index("code")
    max_abs = max(abs(rows["deltaTotalMin"]).max(), NEUTRAL_EPS_MIN)
    ax.set_facecolor(BACKGROUND)
    municipalities = geom["municipalities"]
    # Draw neutral base first so municipalities without rail origins still complete the Swiss shape.
    for rec in municipalities.values():
        code = rec.get("code")
        delta = float(rows.loc[code, "deltaTotalMin"]) if code in rows.index else 0.0
        face = delta_color(delta, max_abs) if code in rows.index else "#f0ece4"
        for poly in rec.get("polygons", []):
            if len(poly) >= 3:
                ax.add_patch(Polygon(np.asarray(poly), closed=True, facecolor=face, edgecolor="white", linewidth=0.17, zorder=2))
    # Crisp outer/canton border from the canton boundary asset.
    boundaries = load_boundaries()
    for canton in boundaries["cantons"]:
        for poly in canton.get("polygons", []):
            ax.add_patch(Polygon(np.asarray(poly), closed=True, facecolor="none", edgecolor=BORDER, linewidth=0.55, zorder=5))
    best, worst = station_extremes(df, year)
    draw_station_markers(ax, best, worst, positions)
    ch = swiss_delta(df, year, "avgTotalMin")
    ax.text(0.020, 0.965, f"{year} municipality origins", transform=ax.transAxes, ha="left", va="top", fontsize=12.0, weight="bold", color=TEXT)
    ax.text(0.980, 0.965, f"CH {format_ms(ch)}", transform=ax.transAxes, ha="right", va="top", fontsize=10.0, weight="bold", color=TEXT,
            bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.2", facecolor="white", edgecolor="#e4e2dc", linewidth=0.8))
    set_swiss_map_extent(ax, geom["bounds"])


def add_delta_map_legend(fig: plt.Figure, comparison_label: str = "Optimized Step 2 vs Baseline") -> None:
    handles = [
        Patch(facecolor=lighten(BLUE, 0.25), edgecolor="none", label="lower Step 2 total time"),
        Patch(facecolor=NEUTRAL, edgecolor="#d8d2c8", label="unchanged (±120 ms)"),
        Patch(facecolor=lighten(RED, 0.25), edgecolor="none", label="higher Step 2 total time"),
        Line2D([0], [0], marker="o", color=BLUE, markerfacecolor=BLUE, markeredgecolor="white", lw=0, markersize=7, label="top 3 improved stations"),
        Line2D([0], [0], marker="o", color=RED, markerfacecolor=RED, markeredgecolor="white", lw=0, markersize=7, label="top 3 worsened stations"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.020), fontsize=8.5, columnspacing=1.22, handlelength=1.55)
    fig.text(0.025, 0.976, comparison_label, ha="left", va="top", fontsize=13.5, weight="bold", color=TEXT)


def draw_progression(df: pd.DataFrame) -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    def nice_limits(values: list[float]) -> tuple[float, float]:
        vals = [float(v) for v in values if math.isfinite(float(v))]
        vals.append(0.0)
        lo = min(vals)
        hi = max(vals)
        if abs(hi - lo) < 1e-6:
            lo -= 0.5
            hi += 0.5
        span = hi - lo
        pad = max(0.08, span * 0.24)
        lo -= pad
        hi += pad
        # Slightly round the limits so individual panels feel intentional.
        raw_step = (hi - lo) / 4.0
        mag = 10 ** math.floor(math.log10(max(raw_step, 1e-6)))
        for mult in (1, 2, 2.5, 5, 10):
            step = mult * mag
            if raw_step <= step:
                break
        lo = math.floor(lo / step) * step
        hi = math.ceil(hi / step) * step
        return lo, hi

    def label_candidates(rows: pd.DataFrame, label: str) -> list[tuple[str, float]]:
        col = f"delta{label}Min"
        vals = rows[col].astype(float) * 60.0
        best = vals.nsmallest(3)
        worst = vals.nlargest(3)
        candidates: dict[str, float] = {}
        for code, val in pd.concat([best, worst]).items():
            candidates[str(code)] = float(val)
        return sorted(candidates.items(), key=lambda item: abs(item[1]), reverse=True)

    def add_nonoverlapping_labels(ax: plt.Axes, rows: pd.DataFrame, label: str, y_limits: tuple[float, float]) -> None:
        y_min, y_max = y_limits
        y_span = max(y_max - y_min, 1e-6)
        min_gap = y_span * 0.105
        placed: list[float] = []
        for code, delta_sec in label_candidates(rows, label):
            if any(abs(delta_sec - y) < min_gap for y in placed):
                continue
            color = COLOR_BY_CANTON.get(code, "#333333")
            # Keep labels inside the axes vertically while preserving the point position.
            y_text = min(max(delta_sec, y_min + 0.06 * y_span), y_max - 0.06 * y_span)
            va = "center"
            ax.plot([1.0, 1.055], [delta_sec, y_text], color=color, lw=0.55, alpha=0.55, zorder=7)
            ax.text(
                1.075,
                y_text,
                f"{code} {delta_sec:+.1f} s".replace("-", "−"),
                ha="left",
                va=va,
                fontsize=6.6,
                weight="bold",
                color=color,
                zorder=9,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.10", facecolor="white", edgecolor="#e5e1d8", linewidth=0.45, alpha=0.94),
            )
            placed.append(y_text)

    fig, axes = plt.subplots(2, len(METRIC_COLUMNS), figsize=(12.1, 7.15), sharex=False, sharey=False)
    fig.patch.set_facecolor("white")

    for row_idx, year in enumerate(YEARS):
        year_rows = metric_delta_rows(df, year, "canton").set_index("code")
        for col_idx, (label, col, short) in enumerate(METRIC_COLUMNS):
            ax = axes[row_idx, col_idx]
            ax.set_facecolor("#fbfaf7")
            values = [float(v) * 60.0 for v in year_rows[f"delta{label}Min"].astype(float).values]
            ch_delta = swiss_delta(df, year, col) * 60.0
            y_limits = nice_limits(values + [ch_delta])
            ax.set_ylim(*y_limits)
            ax.set_xlim(-0.26, 1.48)

            ax.axhspan(y_limits[0], y_limits[1], xmin=0.42, xmax=0.83, facecolor="#f3efe7", alpha=0.72, zorder=0)
            ax.axhline(0, color="#1f2937", lw=1.0, alpha=0.60, zorder=1)
            ax.grid(True, axis="y", color="#e1e5ea", lw=0.75, zorder=0)
            ax.grid(False, axis="x")

            for code in CANTON_ORDER:
                if code not in year_rows.index:
                    continue
                delta_sec = float(year_rows.loc[code, f"delta{label}Min"] * 60.0)
                color = COLOR_BY_CANTON[code]
                ax.plot([0, 1], [0.0, delta_sec], color=color, lw=1.05, alpha=0.67, zorder=3)
                ax.scatter([0], [0.0], s=10, color="#8d8982", alpha=0.38, linewidth=0, zorder=4)
                ax.scatter([1], [delta_sec], s=16, color=color, edgecolor="white", linewidth=0.30, alpha=0.96, zorder=5)

            ax.plot([0, 1], [0.0, ch_delta], color="#000000", lw=2.25, zorder=8)
            ax.scatter([0], [0.0], s=38, color="#000000", edgecolor="white", linewidth=0.7, zorder=9)
            ax.scatter([1], [ch_delta], s=58, color="#000000", marker="D", edgecolor="white", linewidth=0.7, zorder=10)
            ax.text(
                0.04,
                0.93,
                f"CH {ch_delta:+.2f} s".replace("-", "−"),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=6.8,
                color="#000000",
                weight="bold",
                zorder=11,
                bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.08", facecolor="white", edgecolor="#e5e1d8", linewidth=0.45, alpha=0.92),
            )

            add_nonoverlapping_labels(ax, year_rows, label, y_limits)

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["B", "Δ"], fontsize=7.4, color=MUTED)
            if row_idx == 0:
                ax.set_title(label, fontsize=10.4, weight="bold", color=TEXT, pad=6)
            if col_idx == 0:
                ax.set_ylabel(f"{year}\nStep 2 - Baseline\n[s/trip]", fontsize=8.8, color=TEXT, labelpad=10)
            else:
                ax.set_ylabel("")
            if row_idx == 1:
                ax.set_xlabel(label, fontsize=9.2, weight="bold", color=TEXT, labelpad=8)
            ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, pos: f"{v:.0f}" if abs(v) >= 1 else f"{v:.1f}"))
            ax.tick_params(axis="y", labelsize=7.2, pad=2)
            ax.tick_params(axis="x", length=0, pad=2)
            for spine in ["top", "right", "bottom"]:
                ax.spines[spine].set_visible(False)
            ax.spines["left"].set_color("#b6bcc4")
            ax.spines["left"].set_linewidth(0.75)

    fig.suptitle("Cantonal origin progression from Baseline to Optimized Step 2", x=0.055, y=0.982, ha="left", fontsize=14.1, weight="bold", color=TEXT)
    handles = [Line2D([0], [0], color=COLOR_BY_CANTON[c], lw=2, marker="o", markersize=3.8, label=c) for c in CANTON_ORDER]
    handles.append(Line2D([0], [0], color="#000000", lw=2.3, marker="D", markersize=5.2, label="CH"))
    fig.legend(handles=handles, loc="lower center", ncol=14, frameon=False, bbox_to_anchor=(0.5, 0.012), fontsize=7.1, columnspacing=0.74, handlelength=1.25, handletextpad=0.30)
    fig.subplots_adjust(left=0.078, right=0.985, top=0.915, bottom=0.138, hspace=0.34, wspace=0.060)
    out = FIG / "part2_canton_progression_by_component"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))

def draw_map_figures(df: pd.DataFrame) -> None:
    boundaries = load_boundaries()
    geom = json.loads(MUNI_GEOM.read_text(encoding="utf-8"))
    positions = station_positions()

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    for kind, drawer, fname in [
        ("canton", draw_canton_map, "part2_step2_vs_baseline_canton_origin_maps"),
        ("municipality", draw_municipality_map, "part2_step2_vs_baseline_municipality_origin_maps"),
    ]:
        fig = plt.figure(figsize=(13.2, 5.72), facecolor="white")
        gs = fig.add_gridspec(
            2,
            2,
            height_ratios=[1.0, 0.205],
            left=0.018,
            right=0.990,
            top=0.900,
            bottom=0.118,
            wspace=0.028,
            hspace=0.000,
        )
        map_axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
        panel_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

        for ax, panel_ax, year in zip(map_axes, panel_axes, YEARS):
            if kind == "canton":
                drawer(ax, df, year, boundaries, positions)
            else:
                drawer(ax, df, year, geom, positions)
            best, worst = station_extremes(df, year)
            draw_station_extreme_box(panel_ax, best, worst, year)

        add_delta_map_legend(fig)
        out = FIG / fname
        fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(out.with_suffix(".pdf"))
        print(out.with_suffix(".png"))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = read_data()
    draw_progression(df)
    draw_map_figures(df)


if __name__ == "__main__":
    main()
