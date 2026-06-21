#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "part2_spatial_breakdown_data.csv"
OUT = ROOT / "figures" / "part-II"
PDF = OUT / "part2_canton_composition_triangle.pdf"
PNG = OUT / "part2_canton_composition_triangle.png"

YEARS = ["2026", "2035"]
BASELINE = "v0"
FINAL = "optimized_step2"
VERSION_LABELS = {BASELINE: "Baseline", FINAL: "Optimized Step 2"}

# Stable Swiss canton order for colour assignment and legend layout.
CANTON_ORDER = [
    "AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE",
    "NW", "OW", "SG", "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH",
]

# High-contrast qualitative palette, deliberately not tied to political/geographic meaning.
CANTON_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
    "#4d4d4d", "#bcbd22", "#17becf", "#393b79", "#637939", "#8c6d31", "#843c39",
    "#7b4173", "#3182bd", "#e6550d", "#31a354", "#756bb1", "#636363", "#9ecae1",
    "#fdae6b", "#74c476", "#bcbddc", "#969696", "#6baed6",
]
COLOR_BY_CANTON = dict(zip(CANTON_ORDER, CANTON_COLORS))

AXIS_COLORS = {
    "rolling": "#2457a6",
    "dwell": "#9a4f00",
    "transfer": "#6d28d9",
}
AXIS_LABELS = {
    "rolling": "Rolling share",
    "dwell": "Dwell share",
    "transfer": "Transfer share",
}

# Manually chosen clean lower bounds. Each year uses its own local simplex so the
# plotted triangle is scaled to the shown canton values, not to the full 0-100 range.
LOWER_BOUNDS = {
    "2026": {"rolling": 78.0, "dwell": 4.0, "transfer": 3.0},
    "2035": {"rolling": 78.0, "dwell": 5.0, "transfer": 3.0},
}

GRID_VALUES = {
    "2026": {
        "rolling": [80, 84, 88, 92],
        "dwell": [6, 10, 14, 18],
        "transfer": [4, 8, 12, 16],
    },
    "2035": {
        "rolling": [80, 84, 88, 92],
        "dwell": [6, 10, 14, 18],
        "transfer": [4, 8, 12, 16],
    },
}


def pct(row: dict[str, str], key: str) -> float:
    return float(row[key])


def read_canton_rows() -> dict[tuple[str, str, str], dict[str, float]]:
    rows: dict[tuple[str, str, str], dict[str, float]] = {}
    with DATA.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("level") != "canton":
                continue
            year = str(row["year"])
            version = str(row["version"])
            canton = str(row["code"])
            if year not in YEARS or version not in {BASELINE, FINAL} or canton not in CANTON_ORDER:
                continue
            rows[(year, version, canton)] = {
                "rolling": pct(row, "rollingPct"),
                "dwell": pct(row, "dwellPct"),
                "transfer": pct(row, "transferPct"),
                "weight": float(row["totalWeight"]),
                "rolling_min": float(row["avgRollingMin"]),
                "dwell_min": float(row["avgDwellMin"]),
                "transfer_min": float(row["avgTransferMin"]),
            }
    return rows


def swiss_average(rows: dict[tuple[str, str, str], dict[str, float]], year: str, version: str) -> dict[str, float]:
    # Reconstruct the Swiss-origin average from component passenger-minutes so it
    # matches the canton aggregation used in the figure.
    sums = {"rolling": 0.0, "dwell": 0.0, "transfer": 0.0, "weight": 0.0}
    for canton in CANTON_ORDER:
        r = rows[(year, version, canton)]
        w = r["weight"]
        sums["weight"] += w
        sums["rolling"] += r["rolling_min"] * w
        sums["dwell"] += r["dwell_min"] * w
        sums["transfer"] += r["transfer_min"] * w
    denom = sums["rolling"] + sums["dwell"] + sums["transfer"]
    return {
        "rolling": 100.0 * sums["rolling"] / denom,
        "dwell": 100.0 * sums["dwell"] / denom,
        "transfer": 100.0 * sums["transfer"] / denom,
        "weight": sums["weight"],
    }


def local_point(parts: dict[str, float], lower: dict[str, float]) -> tuple[float, float]:
    zoom_total = 100.0 - lower["rolling"] - lower["dwell"] - lower["transfer"]
    r = (parts["rolling"] - lower["rolling"]) / zoom_total
    d = (parts["dwell"] - lower["dwell"]) / zoom_total
    t = (parts["transfer"] - lower["transfer"]) / zoom_total
    # Components should sum to one after the local offset. Renormalise only to
    # absorb floating-point rounding in the source CSV.
    total = r + d + t
    if total <= 0:
        total = 1.0
    r, d, t = r / total, d / total, t / total
    return t + 0.5 * r, (math.sqrt(3.0) / 2.0) * r


def simplex_point(r: float, d: float, t: float) -> tuple[float, float]:
    total = r + d + t
    if total <= 0:
        total = 1.0
    r, d, t = r / total, d / total, t / total
    return t + 0.5 * r, (math.sqrt(3.0) / 2.0) * r


def axis_fraction(axis: str, value: float, lower: dict[str, float]) -> float:
    zoom_total = 100.0 - lower["rolling"] - lower["dwell"] - lower["transfer"]
    return (value - lower[axis]) / zoom_total


def fmt_pct(value: float) -> str:
    return f"{int(round(value))}%" if abs(value - round(value)) < 1e-9 else f"{value:.1f}%"


def axis_range_label(axis: str, lower: dict[str, float]) -> str:
    zoom_total = 100.0 - lower["rolling"] - lower["dwell"] - lower["transfer"]
    lo = lower[axis]
    hi = lower[axis] + zoom_total
    return f"{AXIS_LABELS[axis]} · {fmt_pct(lo)}-{fmt_pct(hi)}"


def validate_local_scales(rows: dict[tuple[str, str, str], dict[str, float]]) -> None:
    for year in YEARS:
        lower = LOWER_BOUNDS[year]
        zoom_total = 100.0 - lower["rolling"] - lower["dwell"] - lower["transfer"]
        upper = {axis: lower[axis] + zoom_total for axis in ("rolling", "dwell", "transfer")}
        points = [
            (version, canton, rows[(year, version, canton)])
            for version in (BASELINE, FINAL)
            for canton in CANTON_ORDER
        ]
        points.extend(
            (version, "CH", swiss_average(rows, year, version))
            for version in (BASELINE, FINAL)
        )
        for version, canton, point in points:
            for axis in ("rolling", "dwell", "transfer"):
                if not (lower[axis] - 1e-9 <= point[axis] <= upper[axis] + 1e-9):
                    raise RuntimeError(
                        f"{year} {version} {canton} {axis}={point[axis]:.3f} "
                        f"outside local scale {lower[axis]:.1f}-{upper[axis]:.1f}"
                    )


# Candidate label offsets in typographic points, tried in this order. Larger
# movements are processed first and receive a few additional, longer offsets.
LABEL_OFFSETS_PT = [
    (13, 0), (-13, 0), (0, 13), (0, -13),
    (11, 9), (-11, 9), (11, -9), (-11, -9),
    (17, 7), (-17, 7), (17, -7), (-17, -7),
]
LARGE_MOVE_EXTRA_OFFSETS_PT = [
    (21, 0), (-21, 0), (0, 20), (0, -20),
    (19, 14), (-19, 14), (19, -14), (-19, -14),
    (27, 7), (-27, 7), (27, -7), (-27, -7),
]


def display_box_for_point(ax: plt.Axes, x: float, y: float, pad_px: float) -> Bbox:
    px, py = ax.transData.transform((x, y))
    return Bbox.from_extents(px - pad_px, py - pad_px, px + pad_px, py + pad_px)


def display_boxes_for_segment(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], pad_px: float = 3.0) -> list[Bbox]:
    boxes: list[Bbox] = []
    for i in range(1, 10):
        t = i / 10.0
        x = p0[0] + (p1[0] - p0[0]) * t
        y = p0[1] + (p1[1] - p0[1]) * t
        boxes.append(display_box_for_point(ax, x, y, pad_px))
    return boxes


def expanded_bbox(bbox: Bbox, pad_px: float = 2.0) -> Bbox:
    return Bbox.from_extents(bbox.x0 - pad_px, bbox.y0 - pad_px, bbox.x1 + pad_px, bbox.y1 + pad_px)


def bbox_inside(inner: Bbox, outer: Bbox, pad_px: float = 2.0) -> bool:
    return (
        inner.x0 >= outer.x0 + pad_px
        and inner.x1 <= outer.x1 - pad_px
        and inner.y0 >= outer.y0 + pad_px
        and inner.y1 <= outer.y1 - pad_px
    )


def bbox_overlaps_any(box: Bbox, others: list[Bbox]) -> bool:
    return any(box.overlaps(other) for other in others)


def existing_text_boxes(ax: plt.Axes, renderer) -> list[Bbox]:
    boxes: list[Bbox] = []
    for artist in ax.texts:
        if artist.get_visible():
            boxes.append(expanded_bbox(artist.get_window_extent(renderer), 2.0))
    return boxes


def build_panel_label_entries(
    year: str,
    rows: dict[tuple[str, str, str], dict[str, float]],
    lower: dict[str, float],
) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for canton in CANTON_ORDER:
        base = rows[(year, BASELINE, canton)]
        final = rows[(year, FINAL, canton)]
        p0 = local_point(base, lower)
        p1 = local_point(final, lower)
        movement = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        midpoint = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
        entries.append({
            "canton": canton,
            "base": p0,
            "final": p1,
            "midpoint": midpoint,
            "movement": movement,
            "color": COLOR_BY_CANTON[canton],
        })
    # Larger scenario movements receive first choice of label space; remaining
    # labels are then placed wherever a clean non-overlapping position exists.
    return sorted(entries, key=lambda item: (-float(item["movement"]), str(item["canton"])))


def add_canton_labels(fig: plt.Figure, axes, rows: dict[tuple[str, str, str], dict[str, float]]) -> dict[str, list[str]]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    placed_by_year: dict[str, list[str]] = {}
    for ax, year in zip(axes, YEARS):
        lower = LOWER_BOUNDS[year]
        entries = build_panel_label_entries(year, rows, lower)
        protected = existing_text_boxes(ax, renderer)
        for entry in entries:
            p0 = entry["base"]
            p1 = entry["final"]
            protected.append(display_box_for_point(ax, p0[0], p0[1], 8.0))
            protected.append(display_box_for_point(ax, p1[0], p1[1], 8.0))
            protected.extend(display_boxes_for_segment(ax, p0, p1, pad_px=2.2))

        # Protect the Swiss-average markers and connecting segment.
        avg_base = swiss_average(rows, year, BASELINE)
        avg_final = swiss_average(rows, year, FINAL)
        ch0 = local_point(avg_base, lower)
        ch1 = local_point(avg_final, lower)
        protected.append(display_box_for_point(ax, ch0[0], ch0[1], 12.0))
        protected.append(display_box_for_point(ax, ch1[0], ch1[1], 15.0))
        protected.extend(display_boxes_for_segment(ax, ch0, ch1, pad_px=4.0))

        accepted: list[Bbox] = []
        placed: list[str] = []
        axes_box = ax.get_window_extent(renderer)
        movement_values = [float(entry["movement"]) for entry in entries]
        large_threshold = sorted(movement_values, reverse=True)[min(7, len(movement_values) - 1)]

        for entry in entries:
            canton = str(entry["canton"])
            movement = float(entry["movement"])
            offsets = LABEL_OFFSETS_PT + (LARGE_MOVE_EXTRA_OFFSETS_PT if movement >= large_threshold else [])
            anchors = [entry["final"], entry["midpoint"], entry["base"]]
            placed_text = None
            for anchor in anchors:
                ax_anchor = anchor  # type: ignore[assignment]
                for dx, dy in offsets:
                    text = ax.annotate(
                        canton,
                        xy=ax_anchor,
                        xytext=(dx, dy),
                        textcoords="offset points",
                        ha="center",
                        va="center",
                        fontsize=6.6,
                        weight="bold",
                        color=str(entry["color"]),
                        zorder=20,
                        bbox={
                            "boxstyle": "round,pad=0.12,rounding_size=0.08",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.78,
                        },
                    )
                    bbox = expanded_bbox(text.get_window_extent(renderer), 2.4)
                    if bbox_inside(bbox, axes_box, pad_px=3.0) and not bbox_overlaps_any(bbox, protected + accepted):
                        accepted.append(bbox)
                        placed.append(canton)
                        placed_text = text
                        break
                    text.remove()
                if placed_text is not None:
                    break
        placed_by_year[year] = placed
    return placed_by_year


def draw_grid(ax: plt.Axes, year: str, lower: dict[str, float]) -> None:
    tri = [(0.5, math.sqrt(3.0) / 2.0), (1.0, 0.0), (0.0, 0.0), (0.5, math.sqrt(3.0) / 2.0)]
    ax.fill([p[0] for p in tri], [p[1] for p in tri], color="#fffaf3", zorder=0)
    ax.plot([p[0] for p in tri], [p[1] for p in tri], color="#3f3f46", lw=1.15, zorder=4)

    grid_color = "#d8d1c4"
    for axis in ("rolling", "dwell", "transfer"):
        for value in GRID_VALUES[year][axis]:
            frac = axis_fraction(axis, float(value), lower)
            if not (0.0 < frac < 1.0):
                continue
            if axis == "rolling":
                a = simplex_point(frac, 1.0 - frac, 0.0)
                b = simplex_point(frac, 0.0, 1.0 - frac)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.75, zorder=1)
                ax.text(a[0] - 0.020, a[1] + 0.002, f"R {int(value)}%", ha="right", va="center", fontsize=6.9, color=AXIS_COLORS[axis], weight="bold")
            elif axis == "dwell":
                a = simplex_point(0.0, frac, 1.0 - frac)
                b = simplex_point(1.0 - frac, frac, 0.0)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.75, zorder=1)
                ax.text(a[0] + 0.012, a[1] - 0.025, f"D {int(value)}%", rotation=-60, ha="left", va="top", fontsize=6.9, color=AXIS_COLORS[axis], weight="bold")
            else:
                a = simplex_point(0.0, 1.0 - frac, frac)
                b = simplex_point(1.0 - frac, 0.0, frac)
                ax.plot([a[0], b[0]], [a[1], b[1]], color=grid_color, lw=0.75, zorder=1)
                ax.text(a[0] - 0.012, a[1] - 0.025, f"T {int(value)}%", rotation=60, ha="right", va="top", fontsize=6.9, color=AXIS_COLORS[axis], weight="bold")

    ax.text(0.5, math.sqrt(3.0) / 2.0 + 0.070, axis_range_label("rolling", lower), ha="center", va="bottom", fontsize=9.5, weight="bold", color=AXIS_COLORS["rolling"])
    ax.text(-0.030, -0.115, axis_range_label("dwell", lower), ha="left", va="top", fontsize=9.5, weight="bold", color=AXIS_COLORS["dwell"])
    ax.text(1.030, -0.115, axis_range_label("transfer", lower), ha="right", va="top", fontsize=9.5, weight="bold", color=AXIS_COLORS["transfer"])

    ax.set_xlim(-0.090, 1.090)
    ax.set_ylim(-0.160, math.sqrt(3.0) / 2.0 + 0.125)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_panel(ax: plt.Axes, year: str, rows: dict[tuple[str, str, str], dict[str, float]]) -> None:
    lower = LOWER_BOUNDS[year]
    draw_grid(ax, year, lower)
    ax.set_title(f"{year}", loc="left", fontsize=13.0, weight="bold", color="#202124", pad=8)

    for canton in CANTON_ORDER:
        base = rows[(year, BASELINE, canton)]
        final = rows[(year, FINAL, canton)]
        c = COLOR_BY_CANTON[canton]
        x0, y0 = local_point(base, lower)
        x1, y1 = local_point(final, lower)
        ax.plot([x0, x1], [y0, y1], color=c, lw=1.05, alpha=0.58, zorder=3)
        ax.scatter(x0, y0, s=28, facecolor="white", edgecolor=c, linewidth=1.05, zorder=5)
        ax.scatter(x1, y1, s=34, facecolor=c, edgecolor="white", linewidth=0.65, zorder=6)

    avg_base = swiss_average(rows, year, BASELINE)
    avg_final = swiss_average(rows, year, FINAL)
    bx, by = local_point(avg_base, lower)
    fx, fy = local_point(avg_final, lower)
    ax.plot([bx, fx], [by, fy], color="#000000", lw=2.1, zorder=8)
    ax.scatter(bx, by, s=75, facecolor="white", edgecolor="#000000", linewidth=1.6, zorder=9)
    ax.scatter(fx, fy, s=135, marker="*", facecolor="#000000", edgecolor="#000000", linewidth=1.0, zorder=10)
    ax.text(fx + 0.018, fy + 0.018, "CH", fontsize=9.2, weight="bold", color="#000000", ha="left", va="bottom", zorder=11)


def main() -> None:
    rows = read_canton_rows()
    missing = [(y, v, c) for y in YEARS for v in (BASELINE, FINAL) for c in CANTON_ORDER if (y, v, c) not in rows]
    if missing:
        raise RuntimeError(f"Missing canton rows: {missing[:5]} (+{len(missing)-5} more)" if len(missing) > 5 else f"Missing canton rows: {missing}")
    validate_local_scales(rows)

    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 7.35))
    for ax, year in zip(axes, YEARS):
        draw_panel(ax, year, rows)

    fig.suptitle("Origin-canton composition of in-system passenger-minutes", x=0.055, y=0.982, ha="left", fontsize=14.2, weight="bold")
    fig.text(0.055, 0.943, "Open circles show Baseline; filled circles show Optimized Step 2. Each panel is locally scaled to the canton values shown.", ha="left", va="top", fontsize=8.8, color="#555555")

    marker_handles = [
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="white", markeredgecolor="#333333", lw=0, markersize=6.2, label="Baseline"),
        Line2D([0], [0], marker="o", color="#333333", markerfacecolor="#333333", markeredgecolor="white", lw=0, markersize=6.2, label="Optimized Step 2"),
        Line2D([0], [0], marker="*", color="#000000", markerfacecolor="#000000", lw=1.5, markersize=9.0, label="Swiss average"),
    ]
    marker_legend = fig.legend(handles=marker_handles, loc="lower center", bbox_to_anchor=(0.5, 0.158), ncol=3, frameon=False, fontsize=8.1, handlelength=1.5, columnspacing=2.0)
    fig.add_artist(marker_legend)

    canton_handles = [
        Line2D([0], [0], marker="o", color=COLOR_BY_CANTON[c], markerfacecolor=COLOR_BY_CANTON[c], lw=1.0, markersize=4.8, label=c)
        for c in CANTON_ORDER
    ]
    fig.legend(handles=canton_handles, loc="lower center", bbox_to_anchor=(0.5, 0.018), ncol=13, frameon=False, fontsize=7.5, handlelength=1.25, columnspacing=1.05, handletextpad=0.35)

    fig.subplots_adjust(left=0.040, right=0.985, top=0.900, bottom=0.245, wspace=0.115)
    placed_labels = add_canton_labels(fig, axes, rows)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(PDF, bbox_inches="tight")
    fig.savefig(PNG, dpi=260, bbox_inches="tight")
    plt.close(fig)
    for year in YEARS:
        labels = ", ".join(placed_labels.get(year, []))
        print(f"Placed {len(placed_labels.get(year, []))} canton labels for {year}: {labels}")
    print(f"Wrote {PDF}")
    print(f"Wrote {PNG}")


if __name__ == "__main__":
    main()
