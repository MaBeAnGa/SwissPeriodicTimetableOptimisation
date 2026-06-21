#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "figures" / "part-II"
SPATIAL = ROOT / "part2_spatial_breakdown_data.csv"

BASE_VERSION = "v0"
FINAL_VERSION = "optimized_step2"
BLUE = "#4d83c6"
RED = "#d96a6d"
GRID = "#e5dfd6"
TEXT = "#202124"
MUTED = "#5f6368"


def station_delta_rows(df: pd.DataFrame, year: int) -> pd.DataFrame:
    base = df[(df["level"] == "station") & (df["year"] == year) & (df["version"] == BASE_VERSION)]
    final = df[(df["level"] == "station") & (df["year"] == year) & (df["version"] == FINAL_VERSION)]
    merged = base[["code", "name", "avgTotalMin"]].merge(
        final[["code", "name", "avgTotalMin"]],
        on="code",
        suffixes=("_base", "_final"),
        how="inner",
    )
    merged["name"] = merged["name_final"].fillna(merged["name_base"])
    merged["delta_min"] = merged["avgTotalMin_final"] - merged["avgTotalMin_base"]
    return merged[["code", "name", "delta_min"]]


def value_label(value: float) -> str:
    sign = "+" if value > 0 else "−"
    abs_value = abs(value)
    decimals = 3 if abs_value < 0.1 else 2
    return f"{sign}{abs_value:.{decimals}f} min"


def draw_panel(ax: plt.Axes, rows: pd.DataFrame, title: str, color: str, xlim: tuple[float, float], side: str) -> None:
    rows = rows.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(rows))
    values = rows["delta_min"].to_numpy(float)
    labels = rows["name"].astype(str).to_list()

    ax.barh(y, values, color=color, alpha=0.92, height=0.58, edgecolor="none")
    ax.axvline(0, color="#3f3f3f", lw=1.1)
    ax.set_xlim(*xlim)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.3)
    ax.set_title(title, loc="left", fontsize=12.6, weight="bold", color=TEXT, pad=6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.tick_params(axis="x", labelsize=8.5, colors=TEXT, pad=2)
    ax.tick_params(axis="y", labelsize=9.2, colors=TEXT, pad=5)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)

    span = abs(xlim[1] - xlim[0])
    for yi, value in zip(y, values):
        if side == "negative":
            ax.text(
                -0.035 * span,
                yi,
                value_label(value),
                ha="right",
                va="center",
                fontsize=8.4,
                weight="bold",
                color="white",
                clip_on=True,
            )
        else:
            ax.text(
                value + 0.025 * span,
                yi,
                value_label(value),
                ha="left",
                va="center",
                fontsize=8.4,
                weight="bold",
                color=TEXT,
                clip_on=False,
            )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#666666")
    ax.spines["bottom"].set_color("#666666")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def main() -> None:
    df = pd.read_csv(SPATIAL)
    df["year"] = df["year"].astype(int)

    year_data = {}
    for year in [2026, 2035]:
        rows = station_delta_rows(df, year)
        best = rows.sort_values("delta_min", ascending=True).head(5).reset_index(drop=True)
        worst = rows.sort_values("delta_min", ascending=False).head(5).reset_index(drop=True)
        lim = max(abs(best["delta_min"].min()), abs(worst["delta_min"].max())) * 1.16
        if lim < 2:
            lim = np.ceil(lim * 10) / 10
        else:
            lim = np.ceil(lim * 2) / 2
        year_data[year] = (best, worst, float(lim))

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 5.95), dpi=180)
    fig.patch.set_facecolor("white")
    fig.suptitle(
        "Largest station-origin total-time changes after Step 2",
        fontsize=17.0,
        weight="bold",
        color=TEXT,
        y=0.965,
    )

    for row_idx, year in enumerate([2026, 2035]):
        best, worst, lim = year_data[year]
        draw_panel(axes[row_idx, 0], best, f"{year}: Most improved", BLUE, (-lim, 0), "negative")
        draw_panel(axes[row_idx, 1], worst, f"{year}: Most worsened", RED, (0, lim), "positive")
        axes[row_idx, 0].text(
            -0.16,
            0.5,
            f"{year}",
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=10.8,
            weight="bold",
            color=MUTED,
        )

    fig.text(0.52, 0.045, "Mean total-time change [min/trip]", ha="center", fontsize=10.5, color=TEXT)
    fig.text(0.275, 0.018, "negative values: lower Step 2 time", ha="center", fontsize=8.2, color=MUTED)
    fig.text(0.735, 0.018, "positive values: higher Step 2 time", ha="center", fontsize=8.2, color=MUTED)

    fig.subplots_adjust(left=0.165, right=0.975, top=0.865, bottom=0.145, hspace=0.55, wspace=0.43)

    out = FIG / "part2_station_total_time_extremes"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(out.with_suffix(".pdf"))
    print(out.with_suffix(".png"))


if __name__ == "__main__":
    main()
