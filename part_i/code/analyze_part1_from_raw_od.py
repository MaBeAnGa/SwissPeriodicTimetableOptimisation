#!/usr/bin/env python3
"""Rebuild Part I thesis evidence directly from raw OD metrics plus OD weights."""

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
OD_METRICS = OUT / "part1_selected_od_metrics.csv.gz"

YEARS = [
    "1982", "2002", "2005", "2008", "2009", "2011", "2012", "2013",
    "2014 Q1/2", "2014 Q3/4", "2015", "2016", "2017", "2018", "2019",
    "2020", "2021", "2022", "2023", "2024", "2025", "2026", "2035",
]
OBSERVED_YEARS = [y for y in YEARS if y != "2035"]
EVENT_WINDOWS = [
    ("1982-2002", "1982", "2002"),
    ("Bahn 2000", "2002", "2005"),
    ("2005-2009", "2005", "2009"),
    ("2011-2014", "2011", "2014 Q1/2"),
    ("Zürich HB Löwenstrasse", "2014 Q1/2", "2016"),
    ("NEAT Base Tunnels", "2016", "2021"),
    ("2021-2026", "2021", "2026"),
    ("STEP 2035 Planning Horizon", "2026", "2035"),
]
METRIC_MAP = {
    "avgTotalMin": "weightedAvgTotalMin",
    "avgRollingMin": "weightedAvgRollingMin",
    "avgDwellMin": "weightedAvgDwellMin",
    "avgTransferMin": "weightedAvgTransferMin",
    "avgWaitMin": "avgWaitMin",
}


def minutes_label(x: float, signed: bool = False) -> str:
    if pd.isna(x):
        return "--"
    sign = ""
    if signed:
        sign = "+" if x > 0 else ("-" if x < 0 else "")
    x_abs = abs(float(x))
    total_seconds = int(round(x_abs * 60))
    h, rem = divmod(total_seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        text = f"{h} h {m} min" + (f" {s} s" if s else "")
    elif m:
        text = f"{m} min" + (f" {s} s" if s else "")
    else:
        text = f"{s} s"
    return f"{sign}{text}"


def safe_year_num(year: str) -> float:
    if year == "2014 Q1/2":
        return 2014.0
    if year == "2014 Q3/4":
        return 2014.5
    return float(year)


def load_json(name: str):
    with (PROJECT / name).open(encoding="utf-8") as f:
        return json.load(f)


def load_station_meta() -> pd.DataFrame:
    muni = load_json("swiss_municipality_station_geometries.json")
    rows = []
    for station, m in muni.get("stationMunicipalities", {}).items():
        rows.append({
            "origin": station,
            "originCantonMeta": m.get("stationCanton") or m.get("municipalityCanton"),
            "municipalityCode": f"{m.get('municipalityCanton')}-{m.get('gdeNr')}",
            "municipalityName": m.get("municipalityName"),
            "municipalityCanton": m.get("municipalityCanton"),
        })
    return pd.DataFrame(rows).drop_duplicates("origin")


def load_od() -> pd.DataFrame:
    df = pd.read_csv(OD_METRICS, dtype={"year": str})
    meta = load_station_meta()
    df = df.merge(meta, on="origin", how="left")
    if "originCanton" in df:
        df["originCanton"] = df["originCanton"].fillna(df["originCantonMeta"])
    else:
        df["originCanton"] = df["originCantonMeta"]
    df["municipalityCode"] = df["municipalityCode"].fillna(df["originCanton"].fillna("XX") + "-" + df["origin"])
    df["municipalityName"] = df["municipalityName"].fillna(df["origin"])
    for c in list(METRIC_MAP.values()) + ["pair_weight"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def aggregate(df: pd.DataFrame, level: str, weighted: bool = True) -> pd.DataFrame:
    if level == "national":
        group_cols = ["year"]
        code_name = ("CH", "Switzerland", "CH")
    elif level == "canton":
        group_cols = ["year", "originCanton"]
    elif level == "municipality":
        group_cols = ["year", "municipalityCode", "municipalityName", "municipalityCanton"]
    elif level == "station":
        group_cols = ["year", "origin", "originCanton", "municipalityCode", "municipalityName"]
    else:
        raise ValueError(level)

    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        year = str(keys[0])
        if weighted:
            w = g["pair_weight"].fillna(0.0)
            total_w = float(w.sum())
            vals = {}
            sums = {}
            for out_col, src_col in METRIC_MAP.items():
                sums[out_col] = float((g[src_col] * w).sum())
                vals[out_col] = sums[out_col] / total_w if total_w else np.nan
        else:
            total_w = float(len(g))
            vals = {out_col: float(g[src_col].mean()) for out_col, src_col in METRIC_MAP.items()}
            sums = {out_col: np.nan for out_col in METRIC_MAP}
        total = vals["avgRollingMin"] + vals["avgDwellMin"] + vals["avgTransferMin"]
        vals["avgTotalMin"] = total
        vals["avgTotalPlusWaitMin"] = total + vals.get("avgWaitMin", np.nan)
        vals["rollingPct"] = 100 * vals["avgRollingMin"] / total if total else np.nan
        vals["dwellPct"] = 100 * vals["avgDwellMin"] / total if total else np.nan
        vals["transferPct"] = 100 * vals["avgTransferMin"] / total if total else np.nan

        if level == "national":
            code, name, canton = code_name
        elif level == "canton":
            code = keys[1]
            name = keys[1]
            canton = keys[1]
        elif level == "municipality":
            code, name, canton = keys[1], keys[2], keys[3]
        else:
            code, name, canton = keys[1], keys[1], keys[2]
        rows.append({
            "level": level,
            "weightMode": "weighted" if weighted else "unweighted",
            "year": year,
            "yearNum": safe_year_num(str(year)),
            "code": code,
            "name": name,
            "canton": canton,
            "originCount": int(g["origin"].nunique()),
            "odCellCount": int(len(g)),
            "totalWeight": total_w,
            **vals,
            "rollingPassengerMinDay": sums["avgRollingMin"],
            "dwellPassengerMinDay": sums["avgDwellMin"],
            "transferPassengerMinDay": sums["avgTransferMin"],
            "waitPassengerMinDay": sums["avgWaitMin"],
        })
    return pd.DataFrame(rows).sort_values(["level", "code", "yearNum"])


def best_year_table(agg: pd.DataFrame, level: str, metric: str, weighted: bool = True) -> pd.DataFrame:
    mode = "weighted" if weighted else "unweighted"
    sub = agg[(agg["level"] == level) & (agg["weightMode"] == mode) & (agg["year"].isin(OBSERVED_YEARS))]
    rows = []
    for (code, name, canton), g in sub.groupby(["code", "name", "canton"], dropna=False):
        g = g.sort_values("yearNum")
        now = g[g["year"] == "2026"]
        if now.empty or metric not in g:
            continue
        best = g.loc[g[metric].idxmin()]
        now = now.iloc[0]
        delta = float(now[metric] - best[metric])
        rows.append({
            "level": level,
            "weightMode": mode,
            "metric": metric,
            "code": code,
            "name": name,
            "canton": canton,
            "bestYearObserved": best["year"],
            "bestValueMin": float(best[metric]),
            "value2026Min": float(now[metric]),
            "delta2026MinusBestMin": delta,
            "delta2026MinusBestLabel": minutes_label(delta, signed=True),
            "totalWeight2026": float(now["totalWeight"]),
            "odCellCount": int(now["odCellCount"]),
        })
    return pd.DataFrame(rows).sort_values("delta2026MinusBestMin", ascending=False)


def event_table(agg: pd.DataFrame, level: str, metric: str, weighted: bool = True) -> pd.DataFrame:
    mode = "weighted" if weighted else "unweighted"
    sub = agg[(agg["level"] == level) & (agg["weightMode"] == mode)]
    rows = []
    for label, y0, y1 in EVENT_WINDOWS:
        a = sub[sub["year"] == y0][["code", "name", "canton", metric, "totalWeight"]].rename(columns={metric: "fromValue", "totalWeight": "fromWeight"})
        b = sub[sub["year"] == y1][["code", metric, "totalWeight"]].rename(columns={metric: "toValue", "totalWeight": "toWeight"})
        m = a.merge(b, on="code")
        m["deltaMin"] = m["toValue"] - m["fromValue"]
        m["deltaLabel"] = m["deltaMin"].map(lambda x: minutes_label(x, signed=True))
        m["window"] = label
        m["fromYear"] = y0
        m["toYear"] = y1
        m["level"] = level
        m["metric"] = metric
        m["weightMode"] = mode
        rows.append(m)
    return pd.concat(rows, ignore_index=True)


def plot_national(agg: pd.DataFrame, out: Path) -> None:
    g = agg[(agg["level"] == "national") & (agg["weightMode"] == "weighted")].sort_values("yearNum")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(g["yearNum"], g["avgTotalMin"], color="#111827", lw=2.5, marker="o", ms=4, label="Total")
    ax.plot(g["yearNum"], g["avgRollingMin"], color="#2457a6", lw=1.8, marker="o", ms=3, label="Rolling")
    ax.plot(g["yearNum"], g["avgTransferMin"], color="#7b3fb3", lw=1.8, marker="o", ms=3, label="Transfer")
    ax.plot(g["yearNum"], g["avgDwellMin"], color="#c46a00", lw=1.8, marker="o", ms=3, label="Dwell")
    ax.plot(g["yearNum"], g["avgWaitMin"], color="#d23b2d", lw=1.4, ls="--", marker="o", ms=3, label="Initial wait")
    ax.set_ylabel("Passenger-weighted minutes")
    ax.set_xlabel("Timetable year")
    ax.grid(True, axis="y", color="#dddddd", lw=0.6)
    ax.set_title("Timetable components across the 131 analysed stations in Part I", loc="left", fontsize=11, weight="bold")
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
    ax.set_xlim(1981, 2036)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_triangle(agg: pd.DataFrame, out: Path) -> None:
    g = agg[(agg["level"] == "national") & (agg["weightMode"] == "weighted")].sort_values("yearNum")
    def xy(row):
        r, d, t = row["rollingPct"] / 100, row["dwellPct"] / 100, row["transferPct"] / 100
        return t + 0.5 * d, math.sqrt(3) / 2 * d
    fig, ax = plt.subplots(figsize=(6.4, 5.7))
    tri = np.array([[0, 0], [1, 0], [0.5, math.sqrt(3) / 2], [0, 0]])
    ax.plot(tri[:, 0], tri[:, 1], color="#444", lw=1.1)
    coords = [xy(r) for _, r in g.iterrows()]
    xs, ys = zip(*coords)
    sc = ax.scatter(xs, ys, c=g["yearNum"], cmap="viridis", s=45, zorder=3)
    ax.plot(xs, ys, color="#111827", lw=1, alpha=0.55)
    for _, r in g.iterrows():
        if r["year"] in {"1982", "2005", "2016", "2017", "2021", "2026", "2035"}:
            x, y = xy(r)
            ax.text(x + 0.012, y + 0.006, r["year"], fontsize=7)
    ax.text(-0.03, -0.04, "Rolling", ha="right", va="top", fontsize=9, weight="bold", color="#2457a6")
    ax.text(1.03, -0.04, "Transfer", ha="left", va="top", fontsize=9, weight="bold", color="#7b3fb3")
    ax.text(0.5, math.sqrt(3) / 2 + 0.03, "Dwell", ha="center", va="bottom", fontsize=9, weight="bold", color="#c46a00")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    cbar.set_label("Calendar year")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Rolling/dwell/transfer composition", loc="left", fontsize=11, weight="bold")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def draw_canton_map(best: pd.DataFrame, out: Path) -> None:
    boundaries = load_json("historical_swiss_boundaries.json")
    values = best.set_index("code")["delta2026MinusBestMin"].to_dict()
    vmax = max(1.0, float(np.nanpercentile(list(values.values()), 95)))
    cmap = mpl.colors.LinearSegmentedColormap.from_list("delta", ["#f2f0ea", "#f7b0a8", "#b4162a"])
    norm = mpl.colors.Normalize(0, vmax)
    patches, colors = [], []
    for canton in boundaries.get("cantons", []):
        val = values.get(canton.get("ak"), np.nan)
        color = "#e9e5dc" if pd.isna(val) else cmap(norm(val))
        for poly in canton.get("polygons", []):
            if len(poly) >= 3:
                patches.append(Polygon(np.asarray(poly), closed=True))
                colors.append(color)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.add_collection(PatchCollection(patches, facecolor=colors, edgecolor="white", linewidth=0.55))
    ax.set_aspect("equal")
    ax.autoscale()
    ax.axis("off")
    ax.set_title("2026 total-time excess versus each canton of origin best observed year", loc="left", fontsize=11, weight="bold")
    for _, row in best.head(5).iterrows():
        c = next((x for x in boundaries.get("cantons", []) if x.get("ak") == row["code"]), None)
        if not c:
            continue
        pts = np.concatenate([np.asarray(p) for p in c.get("polygons", []) if len(p) >= 3])
        ax.text(pts[:, 0].mean(), pts[:, 1].mean(), f"{row['code']}\n{row['delta2026MinusBestLabel']}", ha="center", va="center", fontsize=7, weight="bold")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.035, pad=0.02)
    cbar.set_label("Minutes slower in 2026 than best observed year (1982-2026)")
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=220)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)


def plot_event_heatmap(events: pd.DataFrame, out: Path, weight_mode: str = "weighted") -> None:
    """Plot total-time deltas for the thesis event-window figure.

    Each cell is computed as the later-year mean minus the earlier-year mean.
    In weighted mode, OD pairs are passenger-weighted. In unweighted mode, each
    OD pair starting in the origin canton receives equal weight. Negative values
    therefore indicate that trips originating in that canton are faster in the
    later year.
    """
    if weight_mode not in {"weighted", "unweighted"}:
        raise ValueError(f"Unsupported weight mode: {weight_mode}")
    cols = [label for label, _, _ in EVENT_WINDOWS]
    endpoints = {label: (y0, y1) for label, y0, y1 in EVENT_WINDOWS}
    g = events[(events["metric"] == "avgTotalMin") & (events["weightMode"] == weight_mode)]
    canton = g[g["level"] == "canton"].pivot_table(index="code", columns="window", values="deltaMin", aggfunc="first").reindex(columns=cols)
    national = g[g["level"] == "national"].pivot_table(index="code", columns="window", values="deltaMin", aggfunc="first").reindex(columns=cols)

    if "CH" in national.index:
        top = national.loc[["CH"]]
    else:
        top = pd.DataFrame(columns=cols)
    order_col = "1982-2002" if "1982-2002" in canton else cols[0]
    canton = canton.loc[canton[order_col].sort_values().index]
    pivot = pd.concat([top, canton], axis=0)

    arr = pivot.to_numpy(dtype=float)
    vmax = max(0.25, float(np.nanpercentile(np.abs(arr), 95)))

    # A4-portrait friendly geometry: when inserted at \textwidth, labels remain
    # close to thesis body size rather than being scaled down from a landscape plot.
    fig, ax = plt.subplots(figsize=(7.35, 8.95))
    im = ax.imshow(arr, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_yticks(np.arange(len(pivot.index)))
    ylabels = ["CH mean" if idx == "CH" else idx for idx in pivot.index]
    ax.set_yticklabels(ylabels, fontsize=11.6, fontweight="bold")

    def bold_date_label(date_label: str, scenario_label=None) -> str:
        # Use mathtext only for the date line so the scenario subtitle remains regular.
        math_date = r"$\mathbf{" + date_label.replace(" ", r"\ ").replace("-", r"{-}") + "}$"
        if scenario_label:
            return f"{math_date}\n({scenario_label})"
        return math_date

    def display_label(label: str) -> str:
        if label == "1982-2002":
            return bold_date_label("1982-2002")
        if label == "Bahn 2000":
            return bold_date_label("2002-2005", "Bahn 2000")
        if label == "2005-2009":
            return bold_date_label("2005-2009")
        if label == "2011-2014":
            return bold_date_label("2011-2014")
        if label == "Zürich HB Löwenstrasse":
            return bold_date_label("2014 Q1/2-2016", "Zürich HB Löwenstrasse")
        if label == "NEAT Base Tunnels":
            return bold_date_label("2016-2021", "NEAT Base Tunnels")
        if label == "2021-2026":
            return bold_date_label("2021-2026")
        if label == "STEP 2035 Planning Horizon":
            return bold_date_label("2026-2035", "STEP 2035 Planning Horizon")
        y0, y1 = endpoints[label]
        return bold_date_label(f"{y0}-{y1}", label)

    display_cols = [display_label(c) for c in cols]
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(display_cols, rotation=38, ha="right", va="top", fontsize=9.3, rotation_mode="anchor", linespacing=0.95)
    ax.tick_params(length=0)

    # White minor gridlines make the exact window/canton cells legible in print.
    ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(pivot.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.95)

    if "CH" in pivot.index and len(pivot.index) > 1:
        ax.axhline(0.5, color="#222222", lw=1.2)

    def cell_label(v: float) -> str:
        if pd.isna(v):
            return ""
        if abs(v) < 0.05:
            return "0.0"
        sign = "−" if v < 0 else "+"
        return f"{sign}{abs(v):.1f}"

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            if pd.isna(val):
                continue
            color = "white" if abs(val) > vmax * 0.58 else "#1f2937"
            weight = "bold" if pivot.index[i] == "CH" else "normal"
            ax.text(j, i, cell_label(val), ha="center", va="center", fontsize=10.6, color=color, weight=weight)

    if weight_mode == "weighted":
        title = "Passenger-weighted total-time changes by canton of origin"
        cbar_label = "Change in weighted total time [min]"
    else:
        title = "Unweighted total-time changes by canton of origin"
        cbar_label = "Change in unweighted total time [min]"
    ax.set_title(title, loc="left", fontsize=13.8, weight="bold", pad=20)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label(cbar_label, fontsize=10.8)
    cbar.ax.tick_params(labelsize=10.4)

    fig.subplots_adjust(left=0.115, right=0.865, bottom=0.205, top=0.925)
    fig.savefig(out.with_suffix(".png"), dpi=300)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)

def main() -> None:
    OUT.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True, parents=True)
    df = load_od()
    pieces = []
    for weighted in [True, False]:
        for level in ["national", "canton", "municipality", "station"]:
            pieces.append(aggregate(df, level, weighted))
    agg = pd.concat(pieces, ignore_index=True)
    agg.to_csv(OUT / "raw_od_part1_spatial_aggregates.csv", index=False)

    best_parts = []
    for weighted in [True, False]:
        for level in ["national", "canton", "municipality", "station"]:
            for metric in ["avgTotalMin", "avgRollingMin", "avgTransferMin", "avgDwellMin", "avgWaitMin"]:
                best_parts.append(best_year_table(agg, level, metric, weighted))
    best = pd.concat(best_parts, ignore_index=True)
    best.to_csv(OUT / "raw_od_2026_vs_best_observed.csv", index=False)

    event_parts = []
    for weighted in [True, False]:
        for level in ["national", "canton", "municipality", "station"]:
            for metric in ["avgTotalMin", "avgRollingMin", "avgTransferMin", "avgDwellMin", "avgWaitMin"]:
                event_parts.append(event_table(agg, level, metric, weighted))
    events = pd.concat(event_parts, ignore_index=True)
    events.to_csv(OUT / "raw_od_event_windows.csv", index=False)

    plot_national(agg, FIG / "raw_od_part1_national_component_trends")
    plot_triangle(agg, FIG / "raw_od_part1_national_composition_triangle")
    draw_canton_map(best[(best["weightMode"] == "weighted") & (best["level"] == "canton") & (best["metric"] == "avgTotalMin")].copy(), FIG / "raw_od_part1_canton_2026_vs_best_total_time")
    plot_event_heatmap(events, FIG / "raw_od_part1_canton_event_window_heatmap", "weighted")
    plot_event_heatmap(events, FIG / "raw_od_part1_canton_event_window_heatmap_unweighted", "unweighted")

    summary = {
        "nationalWeighted": agg[(agg["level"] == "national") & (agg["weightMode"] == "weighted")].sort_values("yearNum").to_dict(orient="records"),
        "nationalUnweighted": agg[(agg["level"] == "national") & (agg["weightMode"] == "unweighted")].sort_values("yearNum").to_dict(orient="records"),
        "canton2026VsBestTotal": best[(best["weightMode"] == "weighted") & (best["level"] == "canton") & (best["metric"] == "avgTotalMin")].head(15).to_dict(orient="records"),
        "station2026VsBestTotal": best[(best["weightMode"] == "weighted") & (best["level"] == "station") & (best["metric"] == "avgTotalMin")].head(20).to_dict(orient="records"),
        "municipality2026VsBestTotal": best[(best["weightMode"] == "weighted") & (best["level"] == "municipality") & (best["metric"] == "avgTotalMin")].head(20).to_dict(orient="records"),
        "eventCantonTotal": events[(events["weightMode"] == "weighted") & (events["level"] == "canton") & (events["metric"] == "avgTotalMin")].to_dict(orient="records"),
        "eventCantonTotalUnweighted": events[(events["weightMode"] == "unweighted") & (events["level"] == "canton") & (events["metric"] == "avgTotalMin")].to_dict(orient="records"),
        "eventStationTotal": events[(events["weightMode"] == "weighted") & (events["level"] == "station") & (events["metric"] == "avgTotalMin")].to_dict(orient="records"),
    }
    with (OUT / "raw_od_part1_results_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Wrote raw-OD aggregate evidence:")
    for p in [
        OUT / "raw_od_part1_spatial_aggregates.csv",
        OUT / "raw_od_2026_vs_best_observed.csv",
        OUT / "raw_od_event_windows.csv",
        OUT / "raw_od_part1_results_summary.json",
    ]:
        print(" -", p)
    print("Raw-OD figures:")
    for p in sorted(FIG.glob("raw_od_part1_*.*")):
        print(" -", p)


if __name__ == "__main__":
    main()
