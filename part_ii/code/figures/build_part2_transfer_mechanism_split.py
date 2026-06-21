#!/usr/bin/env python3
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "2nd Half of Project (Comparison 2026 and 2035)" / "transfer_optimization_tables" / "accepted_shift_transfer_line_windows" / "accepted_shift_transfer_line_window_summary.csv"
OUT_PDF = Path(__file__).with_name("part2_transfer_mechanism_split.pdf")
OUT_PNG = Path(__file__).with_name("part2_transfer_mechanism_split.png")

ORDER = [
    ("2026", "Step 1", "2026\nStep 1"),
    ("2026", "Step 2", "2026\nStep 2 additions"),
    ("2035", "Step 1", "2035\nStep 1"),
    ("2035", "Step 2", "2035\nStep 2 additions"),
]

COLORS = {
    "existing": "#3F78AC",
    "unlocked": "#69A878",
    "grid": "#DED9CF",
    "text": "#222222",
    "muted": "#656565",
}


def fnum(value: float, digits: int = 0) -> str:
    if digits == 0:
        return f"{value:,.0f}".replace(",", "'")
    return f"{value:,.{digits}f}".replace(",", "'")


def load_data():
    sums = defaultdict(lambda: defaultdict(float))
    counts = defaultdict(lambda: defaultdict(int))
    with SOURCE.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("json_scan_status") != "complete":
                continue
            key = (row["year"], row["step"])
            sums[key]["existing"] += float(row.get("improved_existing_transfer_pax_min_delta_sum") or 0)
            sums[key]["unlocked"] += float(row.get("unlocked_near_miss_pax_min_delta_sum") or 0)
            counts[key]["existing_groups"] += int(float(row.get("improved_existing_transfer_line_time_shifts") or 0))
            counts[key]["unlocked_groups"] += int(float(row.get("unlocked_near_miss_line_time_shifts") or 0))
            counts[key]["rows"] += 1
    return sums, counts


def main() -> None:
    sums, counts = load_data()

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    labels = [label for _, _, label in ORDER]
    existing = [sums[(year, step)]["existing"] for year, step, _ in ORDER]
    unlocked = [sums[(year, step)]["unlocked"] for year, step, _ in ORDER]
    totals = [a + b for a, b in zip(existing, unlocked)]
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(8.7, 4.85), dpi=220)
    fig.patch.set_facecolor("#FBFAF7")
    ax.set_facecolor("#FBFAF7")

    bar_h = 0.58
    ax.barh(y, existing, height=bar_h, color=COLORS["existing"], label="Improved feasible transfers")
    ax.barh(y, unlocked, left=existing, height=bar_h, color=COLORS["unlocked"], label="Newly feasible transfers")

    max_total = max(totals)
    x_max = max_total * 1.30
    ax.set_xlim(0, x_max)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Proxy passenger-minute gain per day")

    ax.xaxis.grid(True, color=COLORS["grid"], linewidth=0.9)
    ax.set_axisbelow(True)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#87827A")
    ax.tick_params(axis="y", length=0, pad=8)
    ax.tick_params(axis="x", colors=COLORS["muted"])

    # Segment labels are only drawn where they fit comfortably inside the bar.
    for yi, ex, un in zip(y, existing, unlocked):
        if ex > 900:
            ax.text(ex / 2, yi, fnum(ex), ha="center", va="center", color="white", fontsize=9.5, weight="bold")
        if un > 900:
            ax.text(ex + un / 2, yi, fnum(un), ha="center", va="center", color="white", fontsize=9.5, weight="bold")

    for yi, (year, step, _), total in zip(y, ORDER, totals):
        c = counts[(year, step)]
        text = (
            f"{fnum(total)} total\n"
            f"{c['rows']} edits; {c['existing_groups']} + {c['unlocked_groups']} transfer groups"
        )
        ax.text(
            total + max_total * 0.035,
            yi,
            text,
            ha="left",
            va="center",
            fontsize=9.2,
            color=COLORS["text"],
            linespacing=1.25,
            bbox={
                "boxstyle": "round,pad=0.34,rounding_size=0.12",
                "facecolor": "white",
                "edgecolor": "#DDD6CC",
                "linewidth": 0.8,
                "alpha": 0.98,
            },
        )

    title = "Transfer mechanisms around accepted timetable edits"
    subtitle = "Realized local transfer gains from shortened feasible transfers and newly feasible transfers"
    fig.text(0.08, 0.965, title, ha="left", va="top", fontsize=16, weight="bold", color="#151515")
    fig.text(0.08, 0.915, subtitle, ha="left", va="top", fontsize=10.4, color=COLORS["muted"])

    leg = ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.025),
        ncol=2,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )
    for t in leg.get_texts():
        t.set_color(COLORS["text"])

    fig.text(
        0.16,
        0.052,
        "Note: transfer groups are unique station-line-time transfer patterns; partly reduced near-misses are not counted as realized gains.",
        ha="left",
        va="bottom",
        fontsize=8.4,
        color=COLORS["muted"],
    )

    plt.subplots_adjust(left=0.16, right=0.96, top=0.77, bottom=0.19)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=320)
    print(OUT_PDF)
    print(OUT_PNG)


if __name__ == "__main__":
    main()
