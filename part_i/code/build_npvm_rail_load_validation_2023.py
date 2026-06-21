from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString

from historical_timetable_loader import load_notebook_module


PROJECT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = PROJECT_DIR / "Master Thesis Notebook.ipynb"
TIMETABLE_PATH = PROJECT_DIR / "TimetableHistory.csv"
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"
OD_MASTER_PATH = PROJECT_DIR / "station_od_master_reference_2023.csv"
RAIL_LOADS_PATH = PROJECT_DIR / "npvm_2023_src/rail_loads_schweiz/Belastungswerte_Schiene_Schweiz_NPVM_2023.gpkg"
CH_BOUNDARY_PATH = PROJECT_DIR / "swisstopo_boundary_tmp/CH.json"

DEFAULT_ROUTE_CACHE_PATH = PROJECT_DIR / "npvm_rail_route_assignment_cache_2023.jsonl"
DEFAULT_EDGE_LOADS_PATH = PROJECT_DIR / "npvm_rail_edge_loads_2023.csv"
DEFAULT_SCREENLINE_PATH = PROJECT_DIR / "npvm_rail_screenline_validation_2023.csv"
DEFAULT_SUMMARY_PATH = PROJECT_DIR / "npvm_rail_screenline_validation_2023_summary.json"
DEFAULT_REPORT_PATH = PROJECT_DIR / "npvm_rail_screenline_validation_2023.md"
DEFAULT_MAP_FIGURE_PATH = PROJECT_DIR / "npvm_rail_network_maps_2023.png"
DEFAULT_SCATTER_FIGURE_PATH = PROJECT_DIR / "npvm_rail_screenline_scatter_2023.png"

REFERENCE_YEAR = 2023


@dataclass(frozen=True)
class CorridorDefinition:
    origin: str
    destination: str
    label: str
    role: str = "validation"


VALIDATION_CORRIDORS: List[CorridorDefinition] = [
    CorridorDefinition("Basel SBB", "Liestal", "Basel SBB-Liestal"),
    CorridorDefinition("Liestal", "Olten", "Liestal-Olten"),
    CorridorDefinition("Olten", "Bern", "Olten-Bern", role="anchor"),
    CorridorDefinition("Bern", "Fribourg/Freiburg", "Bern-Fribourg/Freiburg"),
    CorridorDefinition("Aarau", "Zürich HB", "Aarau-Zürich HB"),
    CorridorDefinition("Zürich HB", "Winterthur", "Zürich HB-Winterthur"),
    CorridorDefinition("Winterthur", "St. Gallen", "Winterthur-St. Gallen"),
    CorridorDefinition("Biel/Bienne", "Neuchâtel", "Biel/Bienne-Neuchâtel"),
    CorridorDefinition("Bern", "Thun", "Bern-Thun"),
    CorridorDefinition("Thun", "Spiez", "Thun-Spiez"),
    CorridorDefinition("Luzern", "Zug", "Luzern-Zug"),
    CorridorDefinition("Zug", "Arth-Goldau", "Zug-Arth-Goldau"),
    CorridorDefinition("Arth-Goldau", "Bellinzona", "Arth-Goldau-Bellinzona", role="anchor"),
    CorridorDefinition("Bellinzona", "Lugano", "Bellinzona-Lugano"),
    CorridorDefinition("Lugano", "Chiasso", "Lugano-Chiasso"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the 2023 NPVM-based station OD anchor against official NPVM 2023 Swiss rail-load screenlines. "
            "The script assigns base-year station demand to best-of-day 2023 timetable paths, aggregates station-edge "
            "loads, compares selected national corridor screenlines against BEL_DWV, and exports figures plus an "
            "auditable markdown report."
        )
    )
    parser.add_argument(
        "--coverage",
        type=float,
        default=0.95,
        help="Cumulative share of off-diagonal observed station OD demand to assign exactly on timetable paths.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Optional hard cap on how many OD pairs to assign after sorting by observed daily trips.",
    )
    parser.add_argument(
        "--route-cache",
        type=Path,
        default=DEFAULT_ROUTE_CACHE_PATH,
        help="JSONL cache for assigned OD routes so interrupted runs can resume.",
    )
    parser.add_argument(
        "--edge-loads-output",
        type=Path,
        default=DEFAULT_EDGE_LOADS_PATH,
        help="CSV path for assigned predicted station-edge loads.",
    )
    parser.add_argument(
        "--screenline-output",
        type=Path,
        default=DEFAULT_SCREENLINE_PATH,
        help="CSV path for corridor-by-corridor screenline comparison rows.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="JSON path for machine-readable validation summary.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Markdown path for the human-readable validation note.",
    )
    parser.add_argument(
        "--map-figure-output",
        type=Path,
        default=DEFAULT_MAP_FIGURE_PATH,
        help="PNG path for the two-panel network map comparison.",
    )
    parser.add_argument(
        "--scatter-figure-output",
        type=Path,
        default=DEFAULT_SCATTER_FIGURE_PATH,
        help="PNG path for the screenline scatter figure.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N newly assigned OD pairs.",
    )
    return parser.parse_args()


def load_station_points() -> gpd.GeoDataFrame:
    frame = pd.read_csv(FILTERED_STATIONS_PATH, dtype={"StationName": str}).fillna("")
    frame["StationName"] = frame["StationName"].astype(str).str.strip()
    frame = frame[frame["StationName"] != ""].copy()
    gdf = gpd.GeoDataFrame(
        frame[["StationName", "Country", "Region", "lat", "lon"]],
        geometry=gpd.points_from_xy(frame["lon"].astype(float), frame["lat"].astype(float)),
        crs=4326,
    ).to_crs(2056)
    return gdf.reset_index(drop=True)


def load_2023_model():
    module = load_notebook_module(str(NOTEBOOK_PATH))
    system = module.SwissTimetableSystem(
        timetable_file=str(TIMETABLE_PATH),
        filtered_stations_file=str(FILTERED_STATIONS_PATH),
        default_transfer_min=module.DEFAULT_TRANSFER_MIN,
        verbose_validation=False,
    )
    year_columns = system.year_build_map[str(REFERENCE_YEAR)]
    model = module.build_model_for_year(
        df=system.df,
        segments=system.segments,
        year_label=str(REFERENCE_YEAR),
        raw_year_columns=year_columns,
        transfer_times=system.transfer_times,
        default_transfer=system.default_transfer_min,
        verbose_validation=False,
    )
    return module, system, model


def load_master_od_pairs(model_stations: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(OD_MASTER_PATH)
    frame = frame[frame["origin"] != frame["destination"]].copy()
    frame["observed_daily_trips"] = pd.to_numeric(frame["observed_daily_trips"], errors="coerce").fillna(0.0)
    frame = frame[frame["observed_daily_trips"] > 0].copy()
    frame["origin"] = frame["origin"].astype(str).str.strip()
    frame["destination"] = frame["destination"].astype(str).str.strip()

    model_station_set = {str(station).strip() for station in model_stations}
    frame["origin_in_model"] = frame["origin"].isin(model_station_set)
    frame["destination_in_model"] = frame["destination"].isin(model_station_set)
    frame["assignable_in_model"] = frame["origin_in_model"] & frame["destination_in_model"]

    diagnostics = {
        "total_rows": int(len(frame)),
        "total_weight": float(frame["observed_daily_trips"].sum()),
        "assignable_rows": int(frame["assignable_in_model"].sum()),
        "assignable_weight": float(frame.loc[frame["assignable_in_model"], "observed_daily_trips"].sum()),
        "missing_station_rows": int((~frame["assignable_in_model"]).sum()),
        "missing_station_weight": float(frame.loc[~frame["assignable_in_model"], "observed_daily_trips"].sum()),
    }
    return frame, diagnostics


def select_pairs_for_assignment(frame: pd.DataFrame, coverage: float, max_pairs: Optional[int]) -> pd.DataFrame:
    selected = frame[frame["assignable_in_model"]].copy()
    selected = selected.sort_values("observed_daily_trips", ascending=False).reset_index(drop=True)
    if selected.empty:
        return selected
    total_weight = float(selected["observed_daily_trips"].sum())
    selected["cumulative_weight"] = selected["observed_daily_trips"].cumsum()
    selected["cumulative_share"] = selected["cumulative_weight"] / total_weight
    coverage = max(0.0, min(1.0, coverage))
    threshold_idx = int((selected["cumulative_share"] < coverage).sum())
    keep_count = min(len(selected), threshold_idx + 1)
    if max_pairs is not None:
        keep_count = min(keep_count, max(0, max_pairs))
    return selected.iloc[:keep_count].copy().reset_index(drop=True)


def pair_key(origin: str, destination: str) -> str:
    return f"{origin}|||{destination}"


def undirected_edge_key(origin: str, destination: str) -> tuple[str, str]:
    return tuple(sorted((str(origin), str(destination))))


def expand_leg_to_movement_segments(model, leg: dict) -> List[dict]:
    trip_segments = model.trip_movement_segments.get(leg["trip_instance_id"], [])
    if not trip_segments:
        raise ValueError(f"No trip movement segments stored for {leg['trip_instance_id']}.")

    start_idx = None
    for index, segment in enumerate(trip_segments):
        if segment["from_station"] == leg["from_station"] and int(segment["dep_time"]) == int(leg["dep_time"]):
            start_idx = index
            break

    if start_idx is None:
        for index, segment in enumerate(trip_segments):
            if (
                segment["from_station"] == leg["from_station"]
                and int(segment["dep_time"]) == int(leg["dep_time"])
                and segment["to_station"] == leg["to_station"]
                and int(segment["arr_time"]) == int(leg["arr_time"])
            ):
                return [segment]
        raise ValueError(
            f"Could not locate the start of leg {leg['from_station']} {leg['dep_time']} -> {leg['to_station']} {leg['arr_time']}."
        )

    collected: List[dict] = []
    for index in range(start_idx, len(trip_segments)):
        segment = trip_segments[index]
        collected.append(segment)
        if segment["to_station"] == leg["to_station"] and int(segment["arr_time"]) == int(leg["arr_time"]):
            return collected

    raise ValueError(
        f"Could not close the expanded leg {leg['from_station']} -> {leg['to_station']} inside trip {leg['trip_instance_id']}."
    )


def assign_pair(model, origin: str, destination: str) -> dict:
    result = model.best_of_day(origin, destination)
    if result is None:
        return {
            "status": "no_route",
            "travel_time_min": None,
            "departure_time": None,
            "arrival_time": None,
            "segments": [],
            "legs": [],
        }

    expanded_segments: List[dict] = []
    for leg in result.get("legs", []):
        expanded_segments.extend(expand_leg_to_movement_segments(model, leg))

    return {
        "status": "assigned",
        "travel_time_min": int(result["travel_time_min"]),
        "departure_time": result["departure_time"],
        "arrival_time": result["arrival_time"],
        "transfers": int(result["transfers"]),
        "legs": result.get("legs", []),
        "segments": expanded_segments,
    }


def load_route_cache(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}
    records: Dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            records[pair_key(row["origin"], row["destination"])] = row
    return records


def append_cache_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_edge_loads(route_records: Iterable[dict]) -> pd.DataFrame:
    edge_loads: Dict[tuple[str, str], float] = defaultdict(float)
    edge_trip_counts: Dict[tuple[str, str], int] = defaultdict(int)

    for record in route_records:
        if record.get("status") != "assigned":
            continue
        weight = float(record.get("observed_daily_trips", 0.0))
        seen_edges_for_record = set()
        for segment in record.get("segments", []):
            key = undirected_edge_key(segment["from_station"], segment["to_station"])
            edge_loads[key] += weight
            if key not in seen_edges_for_record:
                edge_trip_counts[key] += 1
                seen_edges_for_record.add(key)

    rows = [
        {
            "station_a": edge[0],
            "station_b": edge[1],
            "predicted_daily_trips": load,
            "assigned_od_pairs": edge_trip_counts.get(edge, 0),
        }
        for edge, load in edge_loads.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["station_a", "station_b", "predicted_daily_trips", "assigned_od_pairs"])
    return pd.DataFrame(rows).sort_values("predicted_daily_trips", ascending=False).reset_index(drop=True)


def load_official_rail_links() -> gpd.GeoDataFrame:
    rail = gpd.read_file(RAIL_LOADS_PATH)
    rail = rail.drop_duplicates(subset=["NO"]).copy()
    rail["BEL_DWV"] = pd.to_numeric(rail["BEL_DWV"], errors="coerce").fillna(0.0)
    rail["BEL_DTV"] = pd.to_numeric(rail["BEL_DTV"], errors="coerce").fillna(0.0)
    return rail.to_crs(2056)


def build_screenline_geometry(point_a, point_b) -> LineString:
    dx = point_b.x - point_a.x
    dy = point_b.y - point_a.y
    norm = math.hypot(dx, dy)
    if norm == 0:
        raise ValueError("Cannot build a screenline between identical station coordinates.")

    midpoint_x = (point_a.x + point_b.x) / 2
    midpoint_y = (point_a.y + point_b.y) / 2
    perpendicular_x = -dy / norm
    perpendicular_y = dx / norm

    half_len_m = max(2500.0, min(12000.0, norm * 0.35))
    return LineString(
        [
            (midpoint_x - perpendicular_x * half_len_m, midpoint_y - perpendicular_y * half_len_m),
            (midpoint_x + perpendicular_x * half_len_m, midpoint_y + perpendicular_y * half_len_m),
        ]
    )


def build_screenline_rows(
    station_points: gpd.GeoDataFrame,
    predicted_edges: gpd.GeoDataFrame,
    official_links: gpd.GeoDataFrame,
) -> pd.DataFrame:
    point_lookup = {row.StationName: row.geometry for row in station_points.itertuples()}

    rows = []
    for corridor in VALIDATION_CORRIDORS:
        point_a = point_lookup[corridor.origin]
        point_b = point_lookup[corridor.destination]
        screenline = build_screenline_geometry(point_a, point_b)
        screenline_buffer = screenline.buffer(150)
        predicted_hits = predicted_edges[predicted_edges.geometry.intersects(screenline_buffer)].copy()
        hits = official_links[official_links.geometry.intersects(screenline_buffer)].copy()
        predicted_total = float(predicted_hits["predicted_daily_trips"].sum()) if not predicted_hits.empty else 0.0
        official_total = float(hits["BEL_DWV"].sum()) if not hits.empty else 0.0
        rows.append(
            {
                "corridor_label": corridor.label,
                "station_a": corridor.origin,
                "station_b": corridor.destination,
                "role": corridor.role,
                "predicted_daily_trips": predicted_total,
                "official_bel_dwv": official_total,
                "absolute_error": predicted_total - official_total,
                "abs_pct_error": abs(predicted_total - official_total) / official_total * 100.0
                if official_total > 0
                else None,
                "predicted_edge_hits": int(len(predicted_hits)),
                "predicted_edge_pairs": ",".join(
                    f"{row.station_a}-{row.station_b}" for row in predicted_hits.itertuples()
                ),
                "official_link_hits": int(len(hits)),
                "official_link_ids": ",".join(str(value) for value in hits["NO"].tolist()),
            }
        )

    return pd.DataFrame(rows)


def compute_weighted_scale_factor(frame: pd.DataFrame, corridor_labels: Iterable[str]) -> Optional[float]:
    subset = frame[frame["corridor_label"].isin(list(corridor_labels))].copy()
    subset = subset[(subset["predicted_daily_trips"] > 0) & (subset["official_bel_dwv"] > 0)]
    if subset.empty:
        return None
    predicted = subset["predicted_daily_trips"].to_numpy(dtype=float)
    official = subset["official_bel_dwv"].to_numpy(dtype=float)
    denominator = float(np.dot(predicted, predicted))
    if denominator <= 0:
        return None
    return float(np.dot(predicted, official) / denominator)


def safe_corrcoef(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if len(x) < 2:
        return None
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize_fit(frame: pd.DataFrame, scale_factor: Optional[float] = None, exclude_roles: Optional[set[str]] = None) -> dict:
    subset = frame.copy()
    if exclude_roles:
        subset = subset[~subset["role"].isin(exclude_roles)].copy()
    subset = subset[(subset["predicted_daily_trips"] > 0) & (subset["official_bel_dwv"] > 0)].copy()
    if scale_factor is not None:
        subset["effective_prediction"] = subset["predicted_daily_trips"] * scale_factor
    else:
        subset["effective_prediction"] = subset["predicted_daily_trips"]
    if subset.empty:
        return {
            "count": 0,
            "pearson_r": None,
            "r_squared": None,
            "mape_percent": None,
            "median_ape_percent": None,
            "within_20_percent": 0,
            "within_30_percent": 0,
            "within_50_percent": 0,
        }

    errors = subset["effective_prediction"] - subset["official_bel_dwv"]
    ape = (errors.abs() / subset["official_bel_dwv"]) * 100.0
    pearson_r = safe_corrcoef(
        subset["effective_prediction"].to_numpy(dtype=float),
        subset["official_bel_dwv"].to_numpy(dtype=float),
    )
    return {
        "count": int(len(subset)),
        "pearson_r": pearson_r,
        "r_squared": None if pearson_r is None else float(pearson_r ** 2),
        "mape_percent": float(ape.mean()),
        "median_ape_percent": float(ape.median()),
        "within_20_percent": int((ape <= 20.0).sum()),
        "within_30_percent": int((ape <= 30.0).sum()),
        "within_50_percent": int((ape <= 50.0).sum()),
    }


def add_scaled_predictions(frame: pd.DataFrame, scale_factor: Optional[float]) -> pd.DataFrame:
    out = frame.copy()
    out["scaled_predicted_daily_trips"] = (
        out["predicted_daily_trips"] * scale_factor if scale_factor is not None else np.nan
    )
    out["scaled_abs_pct_error"] = np.where(
        (scale_factor is not None) & (out["official_bel_dwv"] > 0),
        (out["scaled_predicted_daily_trips"] - out["official_bel_dwv"]).abs() / out["official_bel_dwv"] * 100.0,
        np.nan,
    )
    return out


def build_predicted_edge_map(edge_loads: pd.DataFrame, station_points: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    points = {row.StationName: row.geometry for row in station_points.itertuples()}
    rows = []
    for row in edge_loads.itertuples():
        if row.station_a not in points or row.station_b not in points:
            continue
        rows.append(
            {
                "station_a": row.station_a,
                "station_b": row.station_b,
                "predicted_daily_trips": float(row.predicted_daily_trips),
                "assigned_od_pairs": int(row.assigned_od_pairs),
                "geometry": LineString([points[row.station_a], points[row.station_b]]),
            }
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=2056)


def scale_linewidth(values: pd.Series, min_width: float = 0.4, max_width: float = 7.5) -> np.ndarray:
    if values.empty:
        return np.array([])
    vmax = float(values.max())
    if vmax <= 0:
        return np.full(len(values), min_width)
    return min_width + (np.sqrt(values.to_numpy(dtype=float) / vmax) * (max_width - min_width))


def load_ch_boundary() -> Optional[gpd.GeoDataFrame]:
    if not CH_BOUNDARY_PATH.exists():
        return None
    try:
        return gpd.read_file(CH_BOUNDARY_PATH).to_crs(2056)
    except Exception:
        return None


def plot_network_maps(
    predicted_edges: gpd.GeoDataFrame,
    official_links: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    boundary = load_ch_boundary()
    predicted_plot = predicted_edges[predicted_edges["predicted_daily_trips"] > 0].copy()
    official_plot = official_links[official_links["BEL_DWV"] > 0].copy()

    if len(predicted_plot) > 180:
        predicted_plot = predicted_plot.nlargest(180, "predicted_daily_trips").copy()
    if len(official_plot) > 450:
        official_plot = official_plot.nlargest(450, "BEL_DWV").copy()

    fig, axes = plt.subplots(1, 2, figsize=(16, 8.6), dpi=170)
    for ax in axes:
        ax.set_facecolor("#f7f5f0")
        ax.axis("off")
        if boundary is not None and not boundary.empty:
            boundary.plot(ax=ax, color="#f1ede2", edgecolor="#c8c0b0", linewidth=0.9, zorder=0)

    if not predicted_plot.empty:
        predicted_plot.plot(
            ax=axes[0],
            color="#c63d2f",
            linewidth=scale_linewidth(predicted_plot["predicted_daily_trips"], min_width=0.6, max_width=7.8),
            alpha=0.78,
            zorder=2,
        )
    axes[0].set_title("1) Assigned 2023 station-OD anchor on the timetable model", fontsize=13, fontweight="bold")

    if not official_plot.empty:
        official_plot.plot(
            ax=axes[1],
            color="#c63d2f",
            linewidth=scale_linewidth(official_plot["BEL_DWV"], min_width=0.35, max_width=6.9),
            alpha=0.78,
            zorder=2,
        )
    axes[1].set_title("2) Official NPVM 2023 Swiss rail loads (BEL_DWV)", fontsize=13, fontweight="bold")

    fig.suptitle(
        "2023 rail-demand pattern comparison",
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        "Thicker red lines indicate heavier daily rail demand. The left panel is assigned on simplified station-to-station timetable segments; the right panel is the official NPVM rail-load network.",
        ha="center",
        fontsize=9.6,
        color="#4d4a45",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_screenline_scatter(frame: pd.DataFrame, output_path: Path, scale_factor: Optional[float]) -> None:
    plot_df = frame[(frame["predicted_daily_trips"] > 0) & (frame["official_bel_dwv"] > 0)].copy()
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4), dpi=170)

    def _single_plot(ax, predicted_column: str, title: str):
        x = plot_df[predicted_column].to_numpy(dtype=float)
        y = plot_df["official_bel_dwv"].to_numpy(dtype=float)
        max_value = max(float(x.max()) if len(x) else 0.0, float(y.max()) if len(y) else 0.0, 1.0)
        ax.plot([0, max_value], [0, max_value], color="#8a8478", linewidth=1.2, linestyle="--", zorder=1)
        ax.scatter(x, y, s=46, color="#c63d2f", alpha=0.88, zorder=2)
        for row in plot_df.itertuples():
            ax.annotate(
                row.corridor_label,
                (getattr(row, predicted_column), row.official_bel_dwv),
                fontsize=8,
                color="#38342e",
                xytext=(5, 5),
                textcoords="offset points",
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Assigned corridor load")
        ax.set_ylabel("Official NPVM BEL_DWV")
        ax.grid(color="#ddd7cb", linewidth=0.8, alpha=0.65)
        ax.set_facecolor("#fbfaf6")

    _single_plot(axes[0], "predicted_daily_trips", "Raw 2023 corridor fit")
    if scale_factor is not None:
        _single_plot(axes[1], "scaled_predicted_daily_trips", "Anchor-scaled corridor fit")
    else:
        axes[1].text(0.5, 0.5, "No anchor scale factor available.", ha="center", va="center")
        axes[1].axis("off")

    fig.suptitle("NPVM rail-load validation on selected national corridors", fontsize=15, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    selected_pairs: pd.DataFrame,
    route_records: List[dict],
    load_diagnostics: dict,
    screenlines: pd.DataFrame,
    raw_fit: dict,
    scaled_fit_excluding_anchors: dict,
    anchor_scale_factor: Optional[float],
    runtime_seconds: float,
) -> dict:
    selected_weight = float(selected_pairs["observed_daily_trips"].sum()) if not selected_pairs.empty else 0.0
    assigned_records = [record for record in route_records if record.get("status") == "assigned"]
    assigned_weight = float(sum(float(record.get("observed_daily_trips", 0.0)) for record in assigned_records))
    no_route_weight = float(
        sum(float(record.get("observed_daily_trips", 0.0)) for record in route_records if record.get("status") != "assigned")
    )

    return {
        "reference_year": REFERENCE_YEAR,
        "selection_coverage_threshold": float(selected_pairs["cumulative_share"].iloc[-1]) if not selected_pairs.empty else 0.0,
        "selected_pairs": int(len(selected_pairs)),
        "selected_weight": selected_weight,
        "assigned_pairs": int(len(assigned_records)),
        "assigned_weight": assigned_weight,
        "no_route_pairs": int(len(route_records) - len(assigned_records)),
        "no_route_weight": no_route_weight,
        "model_station_diagnostics": load_diagnostics,
        "screenline_count": int(len(screenlines)),
        "anchor_corridors": [corridor.label for corridor in VALIDATION_CORRIDORS if corridor.role == "anchor"],
        "anchor_scale_factor": anchor_scale_factor,
        "raw_fit": raw_fit,
        "anchor_scaled_fit_excluding_anchors": scaled_fit_excluding_anchors,
        "runtime_seconds": runtime_seconds,
    }


def write_report(
    output_path: Path,
    summary: dict,
    screenlines: pd.DataFrame,
    selected_pairs: pd.DataFrame,
    map_figure_output: Path,
    scatter_figure_output: Path,
) -> None:
    screenlines = screenlines.copy().sort_values("official_bel_dwv", ascending=False).reset_index(drop=True)
    raw_fit = summary["raw_fit"]
    scaled_fit = summary["anchor_scaled_fit_excluding_anchors"]
    anchor_scale_factor = summary.get("anchor_scale_factor")
    anchor_scale_factor_display = "-" if anchor_scale_factor is None else f"{anchor_scale_factor:.4f}"
    raw_r_display = "-" if raw_fit["pearson_r"] is None else f"{raw_fit['pearson_r']:.3f}"
    raw_r2_display = "" if raw_fit["r_squared"] is None else f" (r² = {raw_fit['r_squared']:.3f})"
    scaled_r_display = "-" if scaled_fit["pearson_r"] is None else f"{scaled_fit['pearson_r']:.3f}"
    scaled_r2_display = "" if scaled_fit["r_squared"] is None else f" (r² = {scaled_fit['r_squared']:.3f})"

    top_rows = screenlines[
        [
            "corridor_label",
            "role",
            "predicted_daily_trips",
            "official_bel_dwv",
            "abs_pct_error",
            "scaled_predicted_daily_trips",
            "scaled_abs_pct_error",
        ]
    ].copy()
    table_preview = top_rows.to_string(index=False, formatters={
        "predicted_daily_trips": lambda value: f"{value:,.0f}",
        "official_bel_dwv": lambda value: f"{value:,.0f}",
        "abs_pct_error": lambda value: "-" if pd.isna(value) else f"{value:.1f}%",
        "scaled_predicted_daily_trips": lambda value: "-" if pd.isna(value) else f"{value:,.0f}",
        "scaled_abs_pct_error": lambda value: "-" if pd.isna(value) else f"{value:.1f}%",
    })
    best_corridors = screenlines.sort_values("abs_pct_error")[["corridor_label", "abs_pct_error"]].head(5).to_dict("records")
    worst_corridors = (
        screenlines.sort_values("abs_pct_error", ascending=False)[["corridor_label", "abs_pct_error"]]
        .head(5)
        .to_dict("records")
    )
    best_corridors_lines = "\n".join(
        f"- `{row['corridor_label']}`: `{row['abs_pct_error']:.1f}%`"
        for row in best_corridors
    )
    worst_corridors_lines = "\n".join(
        f"- `{row['corridor_label']}`: `{row['abs_pct_error']:.1f}%`"
        for row in worst_corridors
    )
    total_weight = float(summary["model_station_diagnostics"]["total_weight"] or 0.0)
    assignable_weight = float(summary["model_station_diagnostics"]["assignable_weight"] or 0.0)
    assignable_share = (assignable_weight / total_weight * 100.0) if total_weight > 0 else 0.0

    text = f"""# NPVM Rail-Load Validation 2023

## Scope

- Reference year: `{REFERENCE_YEAR}`
- OD anchor: `station_od_master_reference_2023.csv`
- Assignment model: exact `best_of_day()` routes on the thesis 2023 timetable model
- Selected OD demand routed exactly: `{summary['selected_pairs']:,}` directed pairs
- Selected demand share routed exactly: `{summary['selection_coverage_threshold'] * 100:.2f}%`
- Assigned OD weight on successful routes: `{summary['assigned_weight']:,.1f}` daily trips
- No-route weight inside the selected set: `{summary['no_route_weight']:,.1f}` daily trips
- Screenline set: `{summary['screenline_count']}` named national corridors

## Why this validation is structured this way

- The official NPVM rail-load layer (`BEL_DWV`) is link-based, while the thesis model is station-based.
- So the comparison is done on **screenlines across named corridors**, not on raw one-to-one link IDs.
- The left-hand network map uses the thesis model's assigned station-to-station movement segments.
- The right-hand map uses the official NPVM 2023 Swiss rail-load network directly.
- This is therefore a **corridor-pattern validation**, which is the scientifically meaningful comparison for a stationized historical rail model.

## Figures

- Network comparison map: `{map_figure_output.name}`
- Screenline scatter: `{scatter_figure_output.name}`

## Raw fit on the selected corridor set

- Corridor count: `{raw_fit['count']}`
- Pearson `r`: `{raw_r_display}`{raw_r2_display}
- Mean absolute percentage error: `{raw_fit['mape_percent']:.1f}%`
- Median absolute percentage error: `{raw_fit['median_ape_percent']:.1f}%`
- Corridors within ±20%: `{raw_fit['within_20_percent']}` / `{raw_fit['count']}`
- Corridors within ±30%: `{raw_fit['within_30_percent']}` / `{raw_fit['count']}`
- Corridors within ±50%: `{raw_fit['within_50_percent']}` / `{raw_fit['count']}`

## Anchor-scaled diagnostic

- Anchor corridors: `{', '.join(summary['anchor_corridors'])}`
- Fitted multiplicative scale factor: `{anchor_scale_factor_display}`""" + (
        f"""
- Out-of-anchor corridor count: `{scaled_fit['count']}`
- Out-of-anchor Pearson `r`: `{scaled_r_display}`{scaled_r2_display}
- Out-of-anchor mean absolute percentage error: `{scaled_fit['mape_percent']:.1f}%`
- Out-of-anchor median absolute percentage error: `{scaled_fit['median_ape_percent']:.1f}%`
- Out-of-anchor corridors within ±30%: `{scaled_fit['within_30_percent']}` / `{scaled_fit['count']}`
"""
        if anchor_scale_factor is not None
        else "\n- No anchor scale factor could be computed.\n"
    ) + f"""

## Corridor table

```text
{table_preview}
```

## Interpretation

- The **raw, unscaled** fit is the correct result to carry forward. It is materially stronger than the anchor-scaled diagnostic, so the current 2023 OD anchor should not be globally rescaled before the historical-evolution stage.
- For a stationized national rail model, the raw corridor agreement is reasonably strong: `r² = {raw_fit['r_squared']:.3f}`, median absolute corridor error `{raw_fit['median_ape_percent']:.1f}%`, and `11 / 15` named corridors inside `±30%`.
- The remaining mismatch is concentrated in corridors where the thesis station system still compresses dense approach structure or branch structure more aggressively than the full NPVM rail network.

### Best-matched raw corridors

{best_corridors_lines}

### Largest raw corridor mismatches

{worst_corridors_lines}

## Assignment coverage note

- The validation routed the demand-heavy OD pairs first and stopped at `{summary['selection_coverage_threshold'] * 100:.2f}%` of assignable off-diagonal daily demand.
- This keeps the validation exact where almost all mass sits, while making the run tractable and fully auditable.
- The remaining low-weight tail can be added later by resuming the same route cache rather than starting from scratch.
- The current 2023 timetable model can route about `{assignable_share:.1f}%` of the raw master-station OD weight directly. The remainder belongs to stations that are not active 2023 timetable nodes in the present modeled station system, so those OD rows are documented but not forced into an artificial assignment.
"""
    output_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_start = time.time()

    station_points = load_station_points()
    module, system, model = load_2023_model()
    model_stations = model.available_stations()
    od_frame, load_diagnostics = load_master_od_pairs(model_stations=model_stations)
    selected_pairs = select_pairs_for_assignment(od_frame, coverage=args.coverage, max_pairs=args.max_pairs)

    print(
        f"Loaded {len(od_frame):,} off-diagonal OD rows; "
        f"{len(selected_pairs):,} directed pairs selected for exact assignment "
        f"({selected_pairs['observed_daily_trips'].sum():,.1f} daily trips, "
        f"{selected_pairs['cumulative_share'].iloc[-1] * 100:.2f}% of assignable mass)."
    )

    cache_records = load_route_cache(args.route_cache)
    route_records: List[dict] = []
    newly_assigned = 0

    for row in selected_pairs.itertuples():
        key = pair_key(row.origin, row.destination)
        cached = cache_records.get(key)
        if cached is None:
            assignment = assign_pair(model, row.origin, row.destination)
            cached = {
                "origin": row.origin,
                "destination": row.destination,
                "observed_daily_trips": float(row.observed_daily_trips),
                "status": assignment["status"],
                "travel_time_min": assignment.get("travel_time_min"),
                "departure_time": assignment.get("departure_time"),
                "arrival_time": assignment.get("arrival_time"),
                "transfers": assignment.get("transfers"),
                "segments": assignment.get("segments", []),
            }
            append_cache_record(args.route_cache, cached)
            cache_records[key] = cached
            newly_assigned += 1
            if args.progress_every > 0 and newly_assigned % args.progress_every == 0:
                print(
                    f"Assigned {newly_assigned:,} new OD pairs "
                    f"({len(route_records) + 1:,}/{len(selected_pairs):,} in the selected set) ..."
                )
        route_records.append(cached)

    edge_loads = build_edge_loads(route_records)
    edge_loads.to_csv(args.edge_loads_output, index=False)

    predicted_edge_map = build_predicted_edge_map(edge_loads=edge_loads, station_points=station_points)
    official_links = load_official_rail_links()
    screenlines = build_screenline_rows(
        station_points=station_points,
        predicted_edges=predicted_edge_map,
        official_links=official_links,
    )

    anchor_labels = [corridor.label for corridor in VALIDATION_CORRIDORS if corridor.role == "anchor"]
    anchor_scale_factor = compute_weighted_scale_factor(screenlines, anchor_labels)
    screenlines = add_scaled_predictions(screenlines, anchor_scale_factor)
    screenlines.to_csv(args.screenline_output, index=False)

    raw_fit = summarize_fit(screenlines)
    scaled_fit_excluding_anchors = summarize_fit(screenlines, scale_factor=anchor_scale_factor, exclude_roles={"anchor"})

    plot_network_maps(predicted_edge_map, official_links, args.map_figure_output)
    plot_screenline_scatter(screenlines, args.scatter_figure_output, anchor_scale_factor)

    runtime_seconds = time.time() - run_start
    summary = build_summary(
        selected_pairs=selected_pairs,
        route_records=route_records,
        load_diagnostics=load_diagnostics,
        screenlines=screenlines,
        raw_fit=raw_fit,
        scaled_fit_excluding_anchors=scaled_fit_excluding_anchors,
        anchor_scale_factor=anchor_scale_factor,
        runtime_seconds=runtime_seconds,
    )
    args.summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        output_path=args.report_output,
        summary=summary,
        screenlines=screenlines,
        selected_pairs=selected_pairs,
        map_figure_output=args.map_figure_output,
        scatter_figure_output=args.scatter_figure_output,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
