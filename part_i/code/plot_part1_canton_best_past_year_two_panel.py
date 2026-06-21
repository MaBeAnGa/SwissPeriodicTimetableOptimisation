#!/usr/bin/env python3
"""Two-panel Part I map of best past total-time year by canton of origin."""
from __future__ import annotations

import json
from pathlib import Path
import re

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon, Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "part1_results_discussion_work" / "raw_od_part1_spatial_aggregates.csv"
BOUNDARIES = ROOT / "historical_swiss_boundaries.js"
FIG = ROOT / "figures" / "part1_results"
OUT = FIG / "raw_od_part1_canton_best_past_total_time_year_two_panel"

PAST_YEARS = [
    "1982", "2002", "2005", "2008", "2009", "2011", "2012", "2013",
    "2014 Q1/2", "2014 Q3/4", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026",
]

# Same decade = deliberately related colours. Avoids pale yellow for print legibility.
YEAR_COLORS = {
    "1982": "#6b7280",
    "2002": "#1d4ed8",
    "2005": "#2563eb",
    "2008": "#60a5fa",
    "2009": "#93c5fd",
    "2011": "#115e59",
    "2012": "#0f766e",
    "2013": "#0d9488",
    "2014 Q1/2": "#14b8a6",
    "2014 Q3/4": "#2dd4bf",
    "2015": "#16a34a",
    "2016": "#22c55e",
    "2017": "#86efac",
    "2018": "#4ade80",
    "2019": "#15803d",
    "2020": "#92400e",
    "2021": "#b45309",
    "2022": "#d97706",
    "2023": "#f59e0b",
    "2024": "#fb923c",
    "2025": "#f97316",
    "2026": "#ea580c",
}
NO_DATA_COLOR = "#e7e5dc"
EDGE_COLOR = "#ffffff"
LABEL_COLOR = "#111827"
FUTURE_BETTER_LABEL_COLOR = "#111827"
COLOR_SOFTEN = 0.32


def soften_hex(hex_color: str, amount: float = COLOR_SOFTEN) -> str:
    """Mix a colour with white for a calmer print palette."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    r = round(r + (255 - r) * amount)
    g = round(g + (255 - g) * amount)
    b = round(b + (255 - b) * amount)
    return f"#{r:02x}{g:02x}{b:02x}"


def year_color(year: str) -> str:
    return soften_hex(YEAR_COLORS[year]) if year in YEAR_COLORS else "#a3a3a3"

# Small-canton label nudges in LV95 metres. Kept modest so labels remain inside/near their canton.
LABEL_OFFSETS = {
    "BS": (-2600, 5200),
    "BL": (1400, -2500),
    "AG": (-2500, 3500),
    "SO": (3500, -2600),
    "ZG": (1200, 3200),
    "LU": (-2500, -2500),
    "OW": (0, -4200),
    "NW": (5000, 3000),
    "UR": (1500, -3000),
    "SZ": (5000, -1000),
    "AI": (4300, -2200),
    "AR": (-5200, 3600),
    "GL": (3000, 1000),
    "SH": (4200, 2200),
    "SG": (0, -5500),
    "NE": (-1500, 2500),
    "GE": (-4500, -2500),
}

# Drawing order matters for enclave visibility: draw SG before AR/AI, then redraw AR/AI last.
LAST_DRAW_CODES = {"AR", "AI"}


def load_boundaries() -> dict:
    text = BOUNDARIES.read_text(encoding="utf-8")
    match = re.search(r"=\s*(\{.*\});?\s*$", text, re.S)
    if not match:
        raise ValueError(f"Could not parse {BOUNDARIES}")
    return json.loads(match.group(1))


def polygon_area_centroid(poly: list[list[float]]) -> tuple[float, float, float]:
    pts = np.asarray(poly, dtype=float)
    if len(pts) < 3:
        return 0.0, float(np.nanmean(pts[:, 0])), float(np.nanmean(pts[:, 1]))
    x = pts[:, 0]
    y = pts[:, 1]
    if x[0] != x[-1] or y[0] != y[-1]:
        x = np.r_[x, x[0]]
        y = np.r_[y, y[0]]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * np.sum(cross)
    if abs(area) < 1e-9:
        return 0.0, float(np.nanmean(x[:-1])), float(np.nanmean(y[:-1]))
    cx = np.sum((x[:-1] + x[1:]) * cross) / (6 * area)
    cy = np.sum((y[:-1] + y[1:]) * cross) / (6 * area)
    return abs(float(area)), float(cx), float(cy)


def canton_label_position(canton: dict) -> tuple[float, float]:
    weighted_x = weighted_y = total_area = 0.0
    fallback = []
    for poly in canton.get("polygons", []):
        if len(poly) < 3:
            continue
        area, cx, cy = polygon_area_centroid(poly)
        fallback.extend(poly)
        if area > 0:
            weighted_x += area * cx
            weighted_y += area * cy
            total_area += area
    if total_area > 0:
        x = weighted_x / total_area
        y = weighted_y / total_area
    elif fallback:
        pts = np.asarray(fallback, dtype=float)
        x = float(pts[:, 0].mean())
        y = float(pts[:, 1].mean())
    else:
        x = y = 0.0
    dx, dy = LABEL_OFFSETS.get(str(canton.get("ak", "")), (0, 0))
    return x + dx, y + dy


def build_best_table(mode: str) -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype={"year": str})
    sub = df[(df["level"] == "canton") & (df["weightMode"] == mode) & (df["year"].isin(PAST_YEARS + ["2035"]))].copy()
    if sub.empty:
        raise ValueError(f"No {mode} canton rows found in {DATA}")
    rows = []
    for code, g in sub.groupby("code", sort=True):
        past = g[g["year"].isin(PAST_YEARS)].copy()
        future = g[g["year"] == "2035"].copy()
        if past.empty:
            continue
        # Stable tie-break: if two years are numerically equal, prefer the later observed timetable.
        past["pastOrder"] = past["year"].map({year: i for i, year in enumerate(PAST_YEARS)})
        past = past.sort_values(["avgTotalMin", "pastOrder"], ascending=[True, False])
        best = past.iloc[0]
        future_value = float(future.iloc[0]["avgTotalMin"]) if not future.empty else np.nan
        rows.append({
            "code": str(code),
            "bestPastYear": str(best["year"]),
            "bestPastTotalMin": float(best["avgTotalMin"]),
            "total2035Min": future_value,
            "futureLowerThanPastBest": bool(np.isfinite(future_value) and future_value < float(best["avgTotalMin"])),
        })
    return pd.DataFrame(rows)


def draw_panel(ax, boundaries: dict, best: pd.DataFrame, title: str) -> None:
    by_code = best.set_index("code").to_dict(orient="index")
    cantons = list(boundaries.get("cantons", []))
    cantons_sorted = [c for c in cantons if c.get("ak") not in LAST_DRAW_CODES] + [c for c in cantons if c.get("ak") in LAST_DRAW_CODES]

    for canton in cantons_sorted:
        code = str(canton.get("ak", ""))
        row = by_code.get(code)
        face = year_color(row["bestPastYear"]) if row else NO_DATA_COLOR
        z = 2 if code not in LAST_DRAW_CODES else 5
        for poly in canton.get("polygons", []):
            if len(poly) < 3:
                continue
            ax.add_patch(Polygon(
                np.asarray(poly, dtype=float),
                closed=True,
                facecolor=face,
                edgecolor=EDGE_COLOR,
                linewidth=0.62,
                joinstyle="round",
                zorder=z,
            ))

    # Redraw a subtle outer border over all fills for crisp print output.
    for canton in cantons_sorted:
        code = str(canton.get("ak", ""))
        z = 8 if code in LAST_DRAW_CODES else 7
        for poly in canton.get("polygons", []):
            if len(poly) < 3:
                continue
            ax.add_patch(Polygon(
                np.asarray(poly, dtype=float),
                closed=True,
                facecolor="none",
                edgecolor="#f9fafb",
                linewidth=0.78 if code in LAST_DRAW_CODES else 0.54,
                joinstyle="round",
                zorder=z,
            ))

    for canton in cantons:
        code = str(canton.get("ak", ""))
        x, y = canton_label_position(canton)
        row = by_code.get(code)
        is_second_best = bool(row and row["futureLowerThanPastBest"])
        if is_second_best:
            ax.scatter([x], [y], s=255, marker="o", color="#111827", linewidths=0, zorder=11)
        ax.text(
            x,
            y,
            code,
            ha="center",
            va="center",
            fontsize=7.0 if code not in {"BS", "ZG", "AI"} else 6.4,
            weight="bold",
            color="#ffffff" if is_second_best else LABEL_COLOR,
            zorder=12,
            path_effects=[],
        )

    bounds = boundaries.get("bounds", {})
    ax.set_xlim(float(bounds.get("minX", 2480000)) - 6000, float(bounds.get("maxX", 2840000)) + 6000)
    ax.set_ylim(float(bounds.get("minY", 1070000)) - 5000, float(bounds.get("maxY", 1300000)) + 5000)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, loc="left", fontsize=12.5, weight="bold", pad=4)


def legend_handles(years: list[str]) -> list[Patch]:
    handles = [Patch(facecolor=year_color(y), edgecolor="none", label=y) for y in years if y in YEAR_COLORS]
    handles.append(Patch(facecolor=NO_DATA_COLOR, edgecolor="#ffffff", label="No Part I origin data"))
    return handles


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    boundaries = load_boundaries()
    unweighted = build_best_table("unweighted")
    weighted = build_best_table("weighted")
    used_years = sorted(
        set(unweighted["bestPastYear"]).union(set(weighted["bestPastYear"])),
        key=lambda y: PAST_YEARS.index(y),
    )

    mpl.rcParams.update({
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "legend.fontsize": 8.6,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.2), constrained_layout=False)
    draw_panel(axes[0], boundaries, unweighted, "Unweighted mean")
    draw_panel(axes[1], boundaries, weighted, "Passenger-weighted mean")
    fig.suptitle("Best past total-time year by canton", fontsize=16, weight="bold", y=0.985)

    handles = legend_handles(used_years)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(7, len(handles)),
        frameon=False,
        bbox_to_anchor=(0.5, 0.012),
        columnspacing=1.0,
        handlelength=1.35,
        handletextpad=0.42,
    )

    # Label-style legend: normal text means the coloured year is the best overall;
    # a black circled label means 2035 is better, so the colour is the best past year.
    legend_y = 0.125
    fig.text(0.345, legend_y, "CH", ha="right", va="center", fontsize=9.0, weight="bold", color="#6b7280")
    fig.text(0.352, legend_y, "Canton's best year overall", ha="left", va="center", fontsize=9.0, color="#6b7280")
    fig.text(
        0.615, legend_y, "CH", ha="right", va="center", fontsize=7.8, weight="bold", color="#ffffff",
        bbox=dict(boxstyle="circle,pad=0.12", facecolor="#111827", edgecolor="#111827", linewidth=0.0),
    )
    fig.text(0.624, legend_y, "Canton's 2nd best year after 2035", ha="left", va="center", fontsize=9.0, color="#6b7280")
    fig.subplots_adjust(left=0.025, right=0.985, top=0.90, bottom=0.17, wspace=0.035)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=280, bbox_inches="tight")
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".png"))
    for mode, table in [("unweighted", unweighted), ("weighted", weighted)]:
        table.to_csv(ROOT / "part1_results_discussion_work" / f"best_past_total_time_year_by_canton_{mode}.csv", index=False)
        print(mode, table.sort_values("code").to_string(index=False))


if __name__ == "__main__":
    main()
