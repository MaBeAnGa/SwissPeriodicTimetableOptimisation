#!/usr/bin/env python3
"""Refine Part I thesis figures from already-built raw OD evidence tables.

This script intentionally reads the compact evidence tables rather than rebuilding the
raw route cache. It produces thesis-ready PDFs in figures/part1_results/.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "part1_results_discussion_work"
FIG = PROJECT / "figures" / "part1_results"
YEARS_OBSERVED = ["1982", "2002", "2005", "2008", "2009", "2011", "2012", "2013", "2014 Q1/2", "2014 Q3/4", "2015", "2016", "2017", "2018", "2019", "2020", "2021", "2022", "2023", "2024", "2025", "2026"]

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 11,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_json(name: str) -> dict:
    with (PROJECT / name).open(encoding="utf-8") as f:
        return json.load(f)


def safe_year_num(year: str) -> float:
    if year == "2014 Q1/2":
        return 2014.25
    if year == "2014 Q3/4":
        return 2014.75
    return float(year)


def minutes_label(value: float, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "--"
    sign = "+" if signed and value > 0 else ("-" if signed and value < 0 else "")
    seconds = int(round(abs(value) * 60))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        body = f"{h} h {m} min" + (f" {s} s" if s else "")
    elif m:
        body = f"{m} min" + (f" {s} s" if s else "")
    else:
        body = f"{s} s"
    return f"{sign}{body}"


def format_axis_value(x: float) -> str:
    return f"{x:.0f}" if abs(x - round(x)) < 1e-9 else f"{x:.1f}"


def range_from_values(values: pd.Series, pad: float, floor: float = 0.0, ceil: float = 100.0) -> dict:
    vals = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    raw_min = max(floor, float(vals.min()) - pad)
    raw_max = min(ceil, float(vals.max()) + pad)
    if raw_max - raw_min < pad * 3:
        mid = (raw_min + raw_max) / 2
        raw_min = max(floor, mid - pad * 1.5)
        raw_max = min(ceil, mid + pad * 1.5)
    if raw_max <= raw_min:
        raw_max = min(ceil, raw_min + pad * 3)
    return {"min": math.floor(raw_min * 10) / 10, "max": math.ceil(raw_max * 10) / 10}


def normalized(value: float, axis_range: dict) -> float:
    return min(1.0, max(0.0, (float(value) - axis_range["min"]) / (axis_range["max"] - axis_range["min"])))


def ternary_point(parts: dict[str, float]) -> tuple[float, float]:
    # App-style vertices: rolling at top, dwell bottom-left, transfer bottom-right.
    vertices = {
        "rolling": np.array([0.5, math.sqrt(3) / 2]),
        "dwell": np.array([0.0, 0.0]),
        "transfer": np.array([1.0, 0.0]),
    }
    total = sum(max(0.0, float(parts[k])) for k in vertices) or 1.0
    p = sum((max(0.0, float(parts[k])) / total) * vertices[k] for k in vertices)
    return float(p[0]), float(p[1])


def raw_axis_label(axis: str, fraction: float, ranges: dict) -> str:
    axis_range = ranges[axis]
    value = axis_range["min"] + fraction * (axis_range["max"] - axis_range["min"])
    return f"{axis[:1].upper()} {format_axis_value(value)}%"


def axis_title(axis: str, ranges: dict) -> str:
    labels = {"rolling": "Rolling share", "dwell": "Dwell share", "transfer": "Transfer share"}
    axis_range = ranges[axis]
    return f"{labels[axis]} · {format_axis_value(axis_range['min'])}-{format_axis_value(axis_range['max'])}%"


def plot_zoomed_triangle() -> None:
    agg = pd.read_csv(OUT / "raw_od_part1_spatial_aggregates.csv")
    g = agg[(agg["level"] == "national") & (agg["weightMode"] == "weighted")].sort_values("yearNum").copy()

    # Proper local ternary zoom. The lower bounds are clean percentages below all
    # observed values; the remaining seven percentage points form a true simplex.
    lower = {"rolling": 80.0, "dwell": 3.0, "transfer": 10.0}
    zoom_total = 100.0 - lower["rolling"] - lower["dwell"] - lower["transfer"]
    upper = {axis: lower[axis] + zoom_total for axis in lower}

    def local_point(rolling_pct: float, dwell_pct: float, transfer_pct: float) -> tuple[float, float]:
        r = max(0.0, float(rolling_pct) - lower["rolling"]) / zoom_total
        d = max(0.0, float(dwell_pct) - lower["dwell"]) / zoom_total
        t = max(0.0, float(transfer_pct) - lower["transfer"]) / zoom_total
        total = r + d + t
        if total <= 0:
            total = 1.0
        r, d, t = r / total, d / total, t / total
        return t + 0.5 * r, (math.sqrt(3) / 2.0) * r

    g["x"] = g.apply(lambda r: local_point(r["rollingPct"], r["dwellPct"], r["transferPct"])[0], axis=1)
    g["y"] = g.apply(lambda r: local_point(r["rollingPct"], r["dwellPct"], r["transferPct"])[1], axis=1)

    fig, ax = plt.subplots(figsize=(7.6, 7.0))
    tri = np.array([[0.5, math.sqrt(3) / 2], [1.0, 0.0], [0.0, 0.0], [0.5, math.sqrt(3) / 2]])
    ax.fill(tri[:, 0], tri[:, 1], color="#fffaf3", zorder=0)
    ax.plot(tri[:, 0], tri[:, 1], color="#3f3f46", lw=1.35, zorder=3)

    def zoom_fraction(axis: str, raw_value: float) -> float:
        return (float(raw_value) - lower[axis]) / zoom_total

    def axis_label(axis: str, raw_value: float) -> str:
        prefix = {"rolling": "R", "dwell": "D", "transfer": "T"}[axis]
        if abs(raw_value - round(raw_value)) < 1e-9:
            text = f"{int(round(raw_value))}"
        else:
            text = f"{raw_value:.1f}"
        return f"{prefix} {text}%"

    grid_color = "#d8d1c4"
    r_values = [82.0, 84.0, 86.0]
    d_values = [4.0, 5.0, 6.0, 8.0]
    t_values = [12.0, 14.0, 16.0]

    for value in r_values:
        frac = zoom_fraction("rolling", value)
        a = ternary_point({"rolling": frac, "dwell": 0, "transfer": 1 - frac})
        b = ternary_point({"rolling": frac, "dwell": 1 - frac, "transfer": 0})
        ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.85, zorder=1)
        ax.text(b[0] - 0.028, b[1] + 0.004, axis_label("rolling", value), rotation=0, ha="right", va="center", fontsize=7.5, weight="bold", color="#2457a6")

    for value in d_values:
        frac = zoom_fraction("dwell", value)
        a = ternary_point({"rolling": 0, "dwell": frac, "transfer": 1 - frac})
        b = ternary_point({"rolling": 1 - frac, "dwell": frac, "transfer": 0})
        ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.85, zorder=1)
        if value <= 5.0:
            tx = a[0] + 0.048
            ty = -0.030
            ax.text(tx, ty, axis_label("dwell", value), rotation=-60, ha="left", va="top", fontsize=7.5, weight="bold", color="#9a4f00")
        else:
            tx = a[0] + 0.040
            ty = a[1] + 0.020
            ax.text(tx, ty, axis_label("dwell", value), rotation=-60, ha="left", va="center", fontsize=7.5, weight="bold", color="#9a4f00", bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.35})

    for value in t_values:
        frac = zoom_fraction("transfer", value)
        a = ternary_point({"rolling": 0, "dwell": 1 - frac, "transfer": frac})
        b = ternary_point({"rolling": 1 - frac, "dwell": 0, "transfer": frac})
        ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.85, zorder=1)
        ax.text(b[0] + 0.030, b[1] + 0.004, axis_label("transfer", value), rotation=60, ha="left", va="center", fontsize=7.5, weight="bold", color="#6d28d9")

    ax.plot(g["x"], g["y"], color="#111827", lw=1.25, alpha=0.58, zorder=4)
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "paper_safe_years",
        ["#1e3a8a", "#2563eb", "#0f766e", "#7c2d12", "#991b1b"],
    )
    norm = mpl.colors.Normalize(g["yearNum"].min(), g["yearNum"].max())
    ax.scatter(g["x"], g["y"], c=g["yearNum"], cmap=cmap, norm=norm, s=66, edgecolor="white", linewidth=0.9, zorder=5)

    label_positions = {
        "1982": (0.690, 0.062, "left"),
        "2002": (0.350, 0.250, "right"),
        "2005": (0.336, 0.326, "right"),
        "2012": (0.355, 0.418, "right"),
        "2016": (0.455, 0.678, "left"),
        "2017": (0.455, 0.616, "left"),
        "2021": (0.558, 0.745, "left"),
        "2025": (0.585, 0.806, "left"),
        "2026": (0.570, 0.874, "left"),
        "2035": (0.230, 0.682, "right"),
    }
    for _, row in g.iterrows():
        year = str(row["year"])
        if year in label_positions:
            tx, ty, ha = label_positions[year]
            ax.plot([row["x"], tx], [row["y"], ty], color="#52525b", lw=0.55, alpha=0.55, zorder=4)
            ax.text(
                tx,
                ty,
                year,
                fontsize=8.0,
                weight="bold",
                color="#27272a",
                ha=ha,
                va="center",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.84, "pad": 0.6},
                zorder=6,
            )

    def bounds_label(axis: str) -> str:
        label = {"rolling": "Rolling share", "dwell": "Dwell share", "transfer": "Transfer share"}[axis]
        return f"{label} · {lower[axis]:g}-{upper[axis]:g}%"

    ax.text(0.5, math.sqrt(3) / 2 + 0.080, bounds_label("rolling"), ha="center", va="bottom", fontsize=11.4, weight="bold", color="#2457a6")
    ax.text(-0.060, -0.185, bounds_label("dwell"), ha="left", va="top", fontsize=11.4, weight="bold", color="#9a4f00")
    ax.text(1.060, -0.185, bounds_label("transfer"), ha="right", va="top", fontsize=11.4, weight="bold", color="#6d28d9")

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.285, math.sqrt(3) / 2 + 0.14)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Mean rolling, dwell, and transfer shares of total time", loc="left", fontsize=11, weight="bold", pad=8)
    out = FIG / "raw_od_part1_national_composition_triangle_zoomed"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out.with_suffix(".pdf"))


def canton_patches(boundaries: dict, value_by_code: dict, color_for, edgecolor="white", linewidth=0.55):
    patches, colors = [], []
    for canton in boundaries.get("cantons", []):
        code = canton.get("ak")
        color = color_for(code, value_by_code.get(code))
        for poly in canton.get("polygons", []):
            if len(poly) >= 3:
                patches.append(Polygon(np.asarray(poly), closed=True))
                colors.append(color)
    return PatchCollection(patches, facecolor=colors, edgecolor=edgecolor, linewidth=linewidth)


def draw_delta_map() -> None:
    best = pd.read_csv(OUT / "raw_od_2026_vs_best_observed.csv")
    best = best[(best["level"] == "canton") & (best["weightMode"] == "weighted") & (best["metric"] == "avgTotalMin")]
    best = best[best["code"].notna()].copy()
    boundaries = load_json("historical_swiss_boundaries.json")
    values = dict(zip(best["code"], best["delta2026MinusBestMin"]))
    vmax = max(1.0, float(np.nanpercentile(best["delta2026MinusBestMin"], 95)))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("delta", ["#f4efe6", "#f2b6ad", "#9f1239"])
    norm = mpl.colors.Normalize(0, vmax)
    def color_for(code, val):
        if val is None or not np.isfinite(val): return "#ece8df"
        if abs(val) < 1/60: return "#dbeafe"
        return cmap(norm(val))

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.add_collection(canton_patches(boundaries, values, color_for))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title("2026 excess total time versus each origin canton’s best observed year", loc="left", fontsize=10.5, weight="bold")
    ax.text(0.01, 0.02, "Passenger-weighted total time; grouped by canton of origin", transform=ax.transAxes, fontsize=8, color="#6b7280")
    # Keep the map itself clean; the text cites the largest values, and the colorbar carries the scale.
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.04, pad=0.025)
    cbar.set_label("Minutes by which 2026 is slower than the best observed year", fontsize=8)
    out = FIG / "raw_od_part1_canton_2026_vs_best_total_time_refined"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out.with_suffix(".pdf"))


def draw_best_year_map() -> None:
    best = pd.read_csv(OUT / "raw_od_2026_vs_best_observed.csv")
    best = best[(best["level"] == "canton") & (best["weightMode"] == "weighted") & (best["metric"] == "avgTotalMin")]
    best = best[best["code"].notna()].copy()
    boundaries = load_json("historical_swiss_boundaries.json")
    year_order = [y for y in YEARS_OBSERVED if y in set(best["bestYearObserved"])]
    # Use a qualitative palette but order it chronologically.
    palette = list(mpl.colormaps["tab20"].colors) + list(mpl.colormaps["tab20b"].colors)
    color_by_year = {year: mpl.colors.to_hex(palette[i % len(palette)]) for i, year in enumerate(year_order)}
    values = dict(zip(best["code"], best["bestYearObserved"]))
    def color_for(code, year):
        if year is None or str(year) == "nan": return "#ece8df"
        return color_by_year.get(str(year), "#a3a3a3")

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    ax.add_collection(canton_patches(boundaries, values, color_for))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title("Best observed total-time year by canton of origin", loc="left", fontsize=10.5, weight="bold")
    ax.text(0.01, 0.02, "Passenger-weighted total time; observed years 1982-2026 only", transform=ax.transAxes, fontsize=8, color="#6b7280")
    # Label the main non-2026 cases only; the full year mapping is encoded in the colours and legend.
    label_rows = best[best["delta2026MinusBestMin"] > 0.5].copy()
    manual_offsets = {"GE": (-18000, -12000), "NE": (-12000, 5000), "JU": (0, 9000), "BL": (6000, 9000), "SO": (9000, -2000), "OW": (-3000, -5000)}
    for _, row in label_rows.iterrows():
        canton = next((x for x in boundaries.get("cantons", []) if x.get("ak") == row["code"]), None)
        if not canton: continue
        pts = np.concatenate([np.asarray(poly) for poly in canton.get("polygons", []) if len(poly) >= 3])
        dx, dy = manual_offsets.get(str(row["code"]), (0, 0))
        ax.text(float(pts[:,0].mean()) + dx, float(pts[:,1].mean()) + dy, f"{row['code']}\n{row['bestYearObserved']}", ha="center", va="center", fontsize=6.6, weight="bold", color="#111827")
    # Compact legend arranged below.
    handles = [mpl.patches.Patch(facecolor=color_by_year[y], edgecolor="none", label=y) for y in year_order]
    ax.legend(handles=handles, title="Best year", loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=min(6, max(1, len(handles))), frameon=False, fontsize=7, title_fontsize=8)
    out = FIG / "raw_od_part1_canton_best_observed_year_map"
    fig.tight_layout()
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=260, bbox_inches="tight")
    plt.close(fig)
    print("Wrote", out.with_suffix(".pdf"))


def write_latex_heatmap_table() -> None:
    events = pd.read_csv(OUT / "raw_od_event_windows.csv")
    g = events[(events["level"] == "national") & (events["weightMode"] == "weighted") & (events["metric"] == "avgTotalMin")].copy()
    table_rows = []
    for _, r in g.iterrows():
        table_rows.append((r["window"], r["fromYear"], r["toYear"], float(r["fromValue"]), float(r["toValue"]), float(r["deltaMin"])))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{National passenger-weighted total-time changes across selected historical interpretation windows. Negative values indicate a lower in-system total time in the later year.}",
        r"\label{tab:part1-event-windows-national}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"\textbf{Window} & \textbf{Years} & \textbf{From [min]} & \textbf{To [min]} & \textbf{Change [min]} \\",
        r"\midrule",
    ]
    for label, y0, y1, v0, v1, d in table_rows:
        lines.append(f"{label} & {y0}--{y1} & {v0:.2f} & {v1:.2f} & {d:+.2f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    p = OUT / "part1_event_windows_national_table.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", p)


def cell_for_delta(delta: float) -> str:
    if not np.isfinite(delta):
        return "--"
    vmax = 18.0
    pct = int(min(70, max(8, round(abs(delta) / vmax * 70))))
    value = f"{delta:+.1f}"
    if abs(delta) < 0.05:
        return value
    color = "red" if delta > 0 else "blue"
    text = f"\\textcolor{{white}}{{{value}}}" if pct >= 52 else value
    return f"\\cellcolor{{{color}!{pct}}}{text}"


def write_latex_canton_heatmap_table() -> None:
    events = pd.read_csv(OUT / "raw_od_event_windows.csv")
    g = events[(events["level"] == "canton") & (events["weightMode"] == "weighted") & (events["metric"] == "avgTotalMin")].copy()
    # Keep CH at the top, then Swiss cantons alphabetically; drop border/NaN group if present.
    g = g[g["code"].notna()].copy()
    windows = [
        ("1982--2002", "Long-run reconstruction"),
        ("2002--2005", "Rail 2000/Bahn 2000 opening window"),
        ("2005--2008", "Loetschberg base-tunnel window"),
        ("2016--2017", "Pre/post Gotthard base-tunnel window"),
        ("2020--2021", "Ceneri/late-2010s window"),
        ("2024--2026", "Recent timetable recast"),
        ("2026--2035", "STEP 2035 planning horizon"),
    ]
    pivot = g.pivot_table(index="code", columns="window", values="deltaMin", aggfunc="first")
    ordered = ["CH"] + sorted([c for c in pivot.index if c != "CH" and isinstance(c, str)])
    pivot = pivot.loc[[c for c in ordered if c in pivot.index]]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Passenger-weighted total-time changes by origin canton across selected historical interpretation windows. Negative values indicate a lower in-system total time in the later year. Cell colours are scaled by the signed change in minutes.}",
        r"\label{tab:part1-event-windows-canton-heatmap}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\renewcommand{\arraystretch}{1.08}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"\textbf{Origin} & " + " & ".join(f"\\textbf{{{label}}}" for label, _ in windows) + r" \\",
        r"\midrule",
    ]
    for code, row in pivot.iterrows():
        vals = [cell_for_delta(float(row.get(window, np.nan))) for _, window in windows]
        prefix = r"\textbf{CH}" if code == "CH" else str(code)
        lines.append(prefix + " & " + " & ".join(vals) + r" \\")
        if code == "CH":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}%", r"}", r"\end{table}"]
    p = OUT / "part1_event_windows_canton_heatmap_table.tex"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", p)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plot_zoomed_triangle()
    draw_delta_map()
    draw_best_year_map()
    write_latex_heatmap_table()
    write_latex_canton_heatmap_table()

if __name__ == "__main__":
    main()
