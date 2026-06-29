#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, ConnectionPatch, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "figures" / "part-II"
THESIS_TEX = ROOT / "thesis" / "thesis.tex"
BOUNDARIES = ROOT / "historical_swiss_boundaries.js"
STATION_GEOM = ROOT / "swiss_municipality_station_geometries.json"
STEP3_SEGMENT_DIR = Path(
    "<EXTERNAL_SCENARIO_STATE_ROOT>/step3_local_outputs/STEP3_BUILD_LOCAL_FINAL/"
    "reversal_surgery_preparation/ic5_ic82_baseline_step3_only/"
    "cell_surgery_20260617_090934/segment_loads_step3_only_corrected"
)
OUT_PDF = FIG_DIR / "part2_timetable_edit_stage_maps.pdf"
OUT_PNG = FIG_DIR / "part2_timetable_edit_stage_maps.png"

YEAR_LABELS = {2026: "2026", 2035: "2035"}
STAGE_COLORS = {
    1: "#2A6FDB",  # blue
    2: "#F28E2B",  # orange
    3: "#2CA25F",  # green
}
BASE_EDGE = "#A8A39A"
BASE_NODE = "#BBB6AE"
CANTON_FILL = "#FBF7F0"
CANTON_EDGE = "#C8C2B8"
TEXT = "#1F2328"
PANEL_BG = "#FFFFFF"


def canonical_station(name: str) -> str:
    name = str(name).strip()
    if name.startswith("Zürich HB"):
        return "Zürich HB"
    return name


def strip_latex(text: str) -> str:
    txt = text.strip()
    txt = txt.replace("\\textbf{", "").replace("}", "")
    txt = txt.replace("\\(+", "+").replace("\\(-", "-").replace("\\)", "")
    txt = txt.replace("\\(", "").replace("\\)", "")
    txt = txt.replace("~", " ")
    txt = txt.replace("\\cellcolor{gray!12}", "")
    txt = txt.replace("\\shortstack{", "")
    txt = txt.replace("\\\\", " ")
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def load_boundaries() -> dict:
    text = BOUNDARIES.read_text(encoding="utf-8")
    match = re.search(r"=\s*(\{.*\});?\s*$", text, re.S)
    if not match:
        raise RuntimeError(f"Could not parse {BOUNDARIES}")
    return json.loads(match.group(1))


def load_station_positions() -> dict[str, tuple[float, float]]:
    data = json.loads(STATION_GEOM.read_text(encoding="utf-8"))
    positions: dict[str, tuple[float, float]] = {}
    for name, rec in data.get("stationMunicipalities", {}).items():
        east = rec.get("east")
        north = rec.get("north")
        if east is None or north is None:
            continue
        positions[canonical_station(name)] = (float(east), float(north))
    return positions


def extract_table_block(tex: str, label: str) -> str:
    start = tex.index(f"\\label{{{label}}}")
    tail = tex[start:]
    end = tail.index("\\bottomrule")
    return tail[:end]


def parse_edit_rows() -> pd.DataFrame:
    text = THESIS_TEX.read_text(encoding="utf-8")
    rows: list[dict[str, object]] = []
    for label, year in [
        ("tab:part2-accepted-edits-2026", 2026),
        ("tab:part2-accepted-edits-2035", 2035),
    ]:
        block = extract_table_block(text, label)
        for raw in block.splitlines():
            line = raw.strip()
            if not line.startswith("Step~"):
                continue
            if "&" not in line or not line.endswith("\\\\"):
                continue
            parts = [strip_latex(part) for part in line[:-2].split("&")]
            if len(parts) < 9:
                continue
            step_name = parts[0]
            stage = 3 if "Step 3" in step_name else 2 if "Step 2" in step_name else 1
            rows.append(
                {
                    "year": year,
                    "step_name": step_name,
                    "stage": stage,
                    "line": parts[1],
                    "origin": canonical_station(parts[2]),
                    "dest": canonical_station(parts[7]),
                    "delta": parts[8],
                }
            )
    return pd.DataFrame(rows)


def load_network(year: int, positions: dict[str, tuple[float, float]]) -> tuple[list[tuple[str, str]], list[str]]:
    path = STEP3_SEGMENT_DIR / f"{year}_STEP3_ONLY_CORRECTED_adjacent_line_service_pax.csv"
    df = pd.read_csv(path)
    edges: set[tuple[str, str]] = set()
    stations: set[str] = set()
    for row in df.to_dict("records"):
        a = canonical_station(row["First station"])
        b = canonical_station(row["Second station"])
        if a == b:
            continue
        if a not in positions or b not in positions:
            continue
        stations.add(a)
        stations.add(b)
        edges.add(tuple(sorted((a, b))))
    return sorted(edges), sorted(stations)


def aggregate_edit_pairs(rows: pd.DataFrame) -> dict[int, list[dict[str, object]]]:
    out: dict[int, list[dict[str, object]]] = {}
    for year, sub in rows.groupby("year"):
        grouped: dict[tuple[str, str, int], int] = defaultdict(int)
        for row in sub.itertuples(index=False):
            pair = tuple(sorted((row.origin, row.dest)))
            grouped[(pair[0], pair[1], int(row.stage))] += 1
        year_rows: list[dict[str, object]] = []
        for (a, b, stage), count in grouped.items():
            year_rows.append({"a": a, "b": b, "stage": stage, "count": count})
        out[int(year)] = year_rows
    return out


def station_stage_counts(rows: pd.DataFrame) -> dict[int, dict[str, Counter]]:
    out: dict[int, dict[str, Counter]] = {}
    for year, sub in rows.groupby("year"):
        counts: dict[str, Counter] = defaultdict(Counter)
        for row in sub.itertuples(index=False):
            counts[row.origin][int(row.stage)] += 1
            counts[row.dest][int(row.stage)] += 1
        out[int(year)] = counts
    return out


def draw_cantons(ax: plt.Axes, boundary_data: dict) -> None:
    cantons = boundary_data["cantons"].values() if isinstance(boundary_data["cantons"], dict) else boundary_data["cantons"]
    for canton in cantons:
        for poly in canton["polygons"]:
            ax.add_patch(
                Polygon(
                    poly,
                    closed=True,
                    facecolor=CANTON_FILL,
                    edgecolor=CANTON_EDGE,
                    linewidth=0.65,
                    zorder=0,
                )
            )


def draw_base_network(
    ax: plt.Axes,
    edges: Iterable[tuple[str, str]],
    stations: Iterable[str],
    positions: dict[str, tuple[float, float]],
    lw: float,
    alpha: float,
    node_size: float,
) -> None:
    for a, b in edges:
        xa, ya = positions[a]
        xb, yb = positions[b]
        ax.plot([xa, xb], [ya, yb], color=BASE_EDGE, lw=lw, alpha=alpha, zorder=1, solid_capstyle="round")
    xs = [positions[s][0] for s in stations if s in positions]
    ys = [positions[s][1] for s in stations if s in positions]
    ax.scatter(xs, ys, s=node_size, color=BASE_NODE, alpha=0.8, linewidths=0, zorder=2)


def offset_segment(x1: float, y1: float, x2: float, y2: float, offset: float) -> tuple[list[float], list[float]]:
    dx = x2 - x1
    dy = y2 - y1
    norm = math.hypot(dx, dy)
    if norm == 0:
        return [x1, x2], [y1, y2]
    px = -dy / norm
    py = dx / norm
    return [x1 + px * offset, x2 + px * offset], [y1 + py * offset, y2 + py * offset]


def draw_edit_lines(
    ax: plt.Axes,
    pairs: list[dict[str, object]],
    positions: dict[str, tuple[float, float]],
    lw_base: float = 2.0,
    alpha: float = 0.96,
) -> None:
    by_pair: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for rec in pairs:
        by_pair[(str(rec["a"]), str(rec["b"]))].append(rec)
    for (a, b), items in by_pair.items():
        if a not in positions or b not in positions:
            continue
        items = sorted(items, key=lambda x: int(x["stage"]))
        offsets = [0.0]
        if len(items) == 2:
            offsets = [-1200.0, 1200.0]
        elif len(items) >= 3:
            offsets = [-1800.0, 0.0, 1800.0]
        xa, ya = positions[a]
        xb, yb = positions[b]
        for rec, off in zip(items, offsets, strict=False):
            xs, ys = offset_segment(xa, ya, xb, yb, off)
            stage = int(rec["stage"])
            count = int(rec["count"])
            lw = lw_base + 0.55 * max(0, count - 1)
            ax.plot(
                xs,
                ys,
                color=STAGE_COLORS[stage],
                lw=lw,
                alpha=alpha,
                zorder=4 + stage / 10.0,
                solid_capstyle="round",
            )


def draw_pie_marker(
    ax: plt.Axes,
    x: float,
    y: float,
    counts: Counter,
    radius: float,
    edgecolor: str = "white",
    linewidth: float = 0.7,
) -> None:
    total = sum(counts.values())
    if total <= 0:
        return
    theta = 90.0
    for stage in [1, 2, 3]:
        c = counts.get(stage, 0)
        if c <= 0:
            continue
        sweep = 360.0 * c / total
        wedge = Wedge(
            center=(x, y),
            r=radius,
            theta1=theta,
            theta2=theta + sweep,
            facecolor=STAGE_COLORS[stage],
            edgecolor=edgecolor,
            linewidth=linewidth,
            zorder=8,
        )
        ax.add_patch(wedge)
        theta += sweep
    ax.add_patch(Circle((x, y), radius, facecolor="none", edgecolor="#3A3A3A", linewidth=0.55, zorder=9))


INSET_SPECS = {
    2026: [
        {
            "title": "Near Aarau",
            "stations": ["Aarau", "Aarau Torfeld", "Binzenhof", "Lenzburg"],
            "pad_x": 2400.0,
            "pad_y": 1800.0,
        },
        {
            "title": "Near Zürich Oerlikon",
            "stations": [
                "Zürich HB",
                "Zürich Oerlikon",
                "Wallisellen",
                "Glattbrugg",
                "Zürich Flughafen",
                "Bassersdorf",
                "Effretikon",
            ],
            "pad_x": 3200.0,
            "pad_y": 2600.0,
        },
    ],
    2035: [
        {
            "title": "Near Aarau",
            "stations": ["Aarau", "Aarau Torfeld", "Binzenhof", "Däniken SO", "Schönenwerd SO"],
            "pad_x": 2600.0,
            "pad_y": 2000.0,
        },
        {
            "title": "Near Zürich Oerlikon",
            "stations": ["Zürich HB", "Zürich Oerlikon", "Zürich Altstetten", "Schlieren", "Zürich Flughafen"],
            "pad_x": 2600.0,
            "pad_y": 2200.0,
        },
    ],
}


def choose_clusters(year: int, positions: dict[str, tuple[float, float]]) -> list[dict[str, object]]:
    specs = INSET_SPECS.get(year, [])
    clusters: list[dict[str, object]] = []
    for spec in specs:
        members = [station for station in spec["stations"] if station in positions]
        if len(members) < 2:
            continue
        pts = np.array([positions[s] for s in members], dtype=float)
        minx, miny = pts.min(axis=0)
        maxx, maxy = pts.max(axis=0)
        clusters.append(
            {
                "title": spec["title"],
                "stations": members,
                "bbox": (
                    minx - float(spec["pad_x"]),
                    maxx + float(spec["pad_x"]),
                    miny - float(spec["pad_y"]),
                    maxy + float(spec["pad_y"]),
                ),
            }
        )
    return clusters


def set_main_extent(ax: plt.Axes, bounds: dict[str, float]) -> None:
    ax.set_xlim(bounds["minX"] - 9000, bounds["maxX"] + 9000)
    ax.set_ylim(bounds["minY"] - 6000, bounds["maxY"] + 8000)
    ax.set_aspect("equal")
    ax.axis("off")


def draw_year_panel(
    main_ax: plt.Axes,
    inset_axes: list[plt.Axes],
    year: int,
    boundary_data: dict,
    edges: list[tuple[str, str]],
    stations: list[str],
    positions: dict[str, tuple[float, float]],
    pair_rows: list[dict[str, object]],
    station_counts: dict[str, Counter],
) -> None:
    draw_cantons(main_ax, boundary_data)
    draw_base_network(main_ax, edges, stations, positions, lw=0.45, alpha=0.35, node_size=2.8)
    draw_edit_lines(main_ax, pair_rows, positions, lw_base=2.25, alpha=0.95)
    for station, counts in station_counts.items():
        if station not in positions:
            continue
        x, y = positions[station]
        radius = 2400.0 + 260.0 * max(0, sum(counts.values()) - 1)
        draw_pie_marker(main_ax, x, y, counts, radius=radius)
    set_main_extent(main_ax, boundary_data["bounds"])
    stage_counter = Counter(int(row["stage"]) for row in pair_rows for _ in range(int(row["count"])))
    main_ax.set_title(
        f"{YEAR_LABELS[year]}\n"
        f"Step 1: {stage_counter.get(1,0)} rows   Step 2: {stage_counter.get(2,0)} rows   Step 3: {stage_counter.get(3,0)} rows",
        loc="left",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
        pad=10,
    )

    clusters = choose_clusters(year, positions)
    for ax in inset_axes:
        ax.set_visible(False)
    for idx, cluster in enumerate(clusters[: len(inset_axes)]):
        ax = inset_axes[idx]
        ax.set_visible(True)
        ax.set_facecolor(PANEL_BG)
        xmin, xmax, ymin, ymax = cluster["bbox"]
        draw_cantons(ax, boundary_data)
        draw_base_network(ax, edges, stations, positions, lw=0.35, alpha=0.24, node_size=2.0)
        draw_edit_lines(ax, pair_rows, positions, lw_base=2.35, alpha=0.98)
        for station, counts in station_counts.items():
            if station not in positions:
                continue
            x, y = positions[station]
            if xmin <= x <= xmax and ymin <= y <= ymax:
                draw_pie_marker(ax, x, y, counts, radius=760.0 + 70.0 * max(0, sum(counts.values()) - 1))
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(cluster["title"], fontsize=9.8, fontweight="bold", color=TEXT, pad=4)
        ax.add_patch(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#66605A", linewidth=0.7))
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#66605A", linewidth=0.8)
        main_ax.add_patch(rect)
        con1 = ConnectionPatch(
            xyA=(xmax, ymax),
            coordsA=main_ax.transData,
            xyB=(0.0, 1.0),
            coordsB=ax.transAxes,
            color="#8A837C",
            lw=0.8,
            alpha=0.85,
        )
        con2 = ConnectionPatch(
            xyA=(xmax, ymin),
            coordsA=main_ax.transData,
            xyB=(0.0, 0.0),
            coordsB=ax.transAxes,
            color="#8A837C",
            lw=0.8,
            alpha=0.85,
        )
        ax.figure.add_artist(con1)
        ax.figure.add_artist(con2)


def build_figure() -> None:
    boundary_data = load_boundaries()
    positions = load_station_positions()
    edits = parse_edit_rows()
    pair_rows = aggregate_edit_pairs(edits)
    stage_counts = station_stage_counts(edits)
    network = {year: load_network(year, positions) for year in (2026, 2035)}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PANEL_BG,
            "figure.facecolor": PANEL_BG,
        }
    )

    fig = plt.figure(figsize=(8.27, 11.69), facecolor=PANEL_BG)
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[4.65, 1.75],
        height_ratios=[1, 1],
        left=0.055,
        right=0.965,
        top=0.945,
        bottom=0.085,
        wspace=0.05,
        hspace=0.13,
    )
    fig.suptitle(
        "Accepted timetable edits by optimization stage",
        fontsize=18,
        fontweight="bold",
        color=TEXT,
        y=0.975,
    )

    all_axes: list[plt.Axes] = []
    for row_idx, year in enumerate((2026, 2035)):
        main_ax = fig.add_subplot(outer[row_idx, 0])
        inset_grid = outer[row_idx, 1].subgridspec(2, 1, hspace=0.12)
        inset_axes = [fig.add_subplot(inset_grid[0, 0]), fig.add_subplot(inset_grid[1, 0])]
        edges, stations = network[year]
        draw_year_panel(
            main_ax,
            inset_axes,
            year,
            boundary_data,
            edges,
            stations,
            positions,
            pair_rows[year],
            stage_counts[year],
        )
        all_axes.extend([main_ax, *inset_axes])

    # Legend including line colors and station marker explanation.
    handles = [
        Line2D([0], [0], color=BASE_EDGE, lw=2.2, alpha=0.55, label="Other rail segments"),
        Line2D([0], [0], color=STAGE_COLORS[1], lw=3.2, label="Step 1 edits"),
        Line2D([0], [0], color=STAGE_COLORS[2], lw=3.2, label="Step 2 edits"),
        Line2D([0], [0], color=STAGE_COLORS[3], lw=3.2, label="Step 3 edits"),
        Line2D([0], [0], marker="o", markersize=4.8, color="none", markerfacecolor=BASE_NODE, markeredgewidth=0, label="Other stations"),
        Line2D([0], [0], marker="o", markersize=8.5, color="#3A3A3A", markerfacecolor="#FFFFFF", label="Edited station (pie if multiple stages)"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9.3,
        bbox_to_anchor=(0.5, 0.025),
        columnspacing=1.6,
        handlelength=2.4,
    )

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=240, bbox_inches="tight")
    print(f"Wrote {OUT_PDF}")
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    build_figure()
