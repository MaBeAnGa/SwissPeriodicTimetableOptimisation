#!/usr/bin/env python3
"""Plot Part I unweighted total in-system time by origin canton."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "part1_results_discussion_work" / "raw_od_part1_spatial_aggregates.csv"
FIG = ROOT / "figures" / "part1_results"
OUT = FIG / "raw_od_part1_canton_total_time_trends_unweighted"

YEAR_ORDER = [
    "1982", "2002", "2005", "2008", "2009", "2011", "2012", "2013",
    "2014 Q1/2", "2014 Q3/4", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2035",
]

def year_num(year: str) -> float:
    if year == "2014 Q1/2":
        return 2014.0
    if year == "2014 Q3/4":
        return 2014.5
    return float(year)

def minutes_axis(x, _pos=None):
    x = float(x)
    if x >= 60:
        h = int(x // 60)
        m = int(round(x - h * 60))
        return f"{h} h {m} min" if m else f"{h} h"
    return f"{int(round(x))} min"


def non_overlapping_labels(items, min_gap=1.7):
    """Keep labels at natural y-positions only when they do not collide."""
    selected = []
    for item in sorted(items, key=lambda row: row["y"]):
        if all(abs(item["y"] - other["y"]) >= min_gap for other in selected):
            selected.append(item)
    return selected

def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, dtype={"year": str})
    df = df[df["year"].isin(YEAR_ORDER)].copy()
    df["yearNum"] = df["year"].map(year_num)

    canton = df[(df["level"] == "canton") & (df["weightMode"] == "unweighted")].copy()
    national = df[(df["level"] == "national") & (df["weightMode"] == "unweighted")].copy()
    canton["year"] = pd.Categorical(canton["year"], YEAR_ORDER, ordered=True)
    national["year"] = pd.Categorical(national["year"], YEAR_ORDER, ordered=True)
    canton = canton.sort_values(["code", "yearNum"])
    national = national.sort_values("yearNum")
    if canton.empty or national.empty:
        raise SystemExit("No unweighted canton/national rows found in raw_od_part1_spatial_aggregates.csv")


    codes = sorted(canton["code"].dropna().unique())

    # High-contrast categorical palette selected to remain visible on white paper.
    palette = [
        "#1f2937", "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706",
        "#0891b2", "#be123c", "#4d7c0f", "#9333ea", "#0f766e", "#b45309",
        "#64748b", "#16a34a", "#c026d3", "#0284c7", "#ea580c", "#65a30d",
        "#475569", "#991b1b", "#0369a1", "#854d0e", "#166534", "#581c87",
        "#9f1239", "#0e7490",
    ]
    colors = {code: palette[i % len(palette)] for i, code in enumerate(codes)}

    mpl.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(11.8, 14.85))

    line_handles = []
    for code in codes:
        g = canton[canton["code"] == code]
        handle, = ax.plot(
            g["yearNum"],
            g["avgTotalMin"],
            color=colors[code],
            lw=1.9,
            alpha=0.92,
            marker="o",
            ms=3.3,
            markeredgewidth=0.0,
            label=code,
            zorder=2,
        )
        line_handles.append(handle)

    ch_handle, = ax.plot(
        national["yearNum"],
        national["avgTotalMin"],
        color="#000000",
        lw=2.9,
        marker="o",
        ms=4.6,
        markeredgewidth=0.0,
        label="CH mean",
        zorder=5,
    )

    xticks = [1990, 2000, 2010, 2020, 2030]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(x) for x in xticks])
    ax.yaxis.set_major_formatter(FuncFormatter(minutes_axis))
    ax.set_ylabel("Unweighted mean minutes", labelpad=10)
    ax.set_xlabel("Timetable year")
    ax.set_title("Mean total travel time by canton of origin", loc="left", weight="bold", pad=10)
    ax.grid(True, axis="y", color="#e5e7eb", lw=0.9)
    ax.grid(False, axis="x")
    ax.set_xlim(1979.6, 2037.9)
    ymin = max(0, canton["avgTotalMin"].min() - 3.0)
    ymax = canton["avgTotalMin"].max() + 4.0
    ax.set_ylim(ymin, ymax)
    for spine in ax.spines.values():
        spine.set_color("#3f3f46")
        spine.set_linewidth(0.9)

    first_year = min(year_num(y) for y in YEAR_ORDER)
    last_year = max(year_num(y) for y in YEAR_ORDER)
    left_candidates = []
    right_candidates = []
    for code in codes:
        g = canton[canton["code"] == code].sort_values("yearNum")
        if g.empty:
            continue
        left_candidates.append({"code": code, "y": float(g.iloc[0]["avgTotalMin"])})
        right_candidates.append({"code": code, "y": float(g.iloc[-1]["avgTotalMin"])})

    for item in non_overlapping_labels(left_candidates, min_gap=1.65):
        code = item["code"]
        ax.plot([first_year, first_year - 0.55], [item["y"], item["y"]], color=colors[code], lw=0.65, alpha=0.55, zorder=1)
        ax.text(
            first_year - 0.68,
            item["y"],
            code,
            color=colors[code],
            fontsize=9.0,
            va="center",
            ha="right",
            weight="bold",
            zorder=6,
        )
    for item in non_overlapping_labels(right_candidates, min_gap=1.65):
        code = item["code"]
        ax.plot([last_year, last_year + 0.55], [item["y"], item["y"]], color=colors[code], lw=0.65, alpha=0.55, zorder=1)
        ax.text(
            last_year + 0.68,
            item["y"],
            code,
            color=colors[code],
            fontsize=9.0,
            va="center",
            ha="left",
            weight="bold",
            zorder=6,
        )

    handles = line_handles + [ch_handle]
    labels = [h.get_label() for h in handles]
    ax.legend(
        handles,
        labels,
        ncol=12,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        frameon=False,
        handlelength=2.0,
        columnspacing=1.05,
        handletextpad=0.42,
        borderaxespad=0.0,
    )

    fig.subplots_adjust(left=0.125, right=0.965, top=0.90, bottom=0.235)
    fig.savefig(OUT.with_suffix(".png"), dpi=260)
    fig.savefig(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".pdf"))
    print(OUT.with_suffix(".png"))

if __name__ == "__main__":
    main()
