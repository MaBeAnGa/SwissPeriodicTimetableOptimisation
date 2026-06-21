from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from historical_od_weighting import (
    DEFAULT_DISTANCE_EXPONENT,
    DEFAULT_REFERENCE_YEAR,
    DEFAULT_STATION_POPULATION,
    haversine_km,
)
from historical_station_sets import OVERALL_ANALYSIS_TARGET_STATIONS
from historical_timetable_loader import build_system


PROJECT_DIR = Path(__file__).resolve().parent
BASE_2023_ANALYSIS_PATH = PROJECT_DIR / "historical_od_pair_weights_reference_2023.csv"
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"

DEFAULT_ALL_OD_PATH = PROJECT_DIR / "All_OD_Pairings.csv"
DEFAULT_AUDIT_OUTPUT_PATH = PROJECT_DIR / "All_OD_Pairings_full_audit.csv"
DEFAULT_SUMMARY_PATH = PROJECT_DIR / "All_OD_Pairings_summary.json"
DEFAULT_VALIDATION_PATH = PROJECT_DIR / "All_OD_Pairings_validation.md"
DEFAULT_SKIM_PATH = PROJECT_DIR / "All_OD_Pairings_gcost_skims.csv"
DEFAULT_METHOD_PATH = PROJECT_DIR / "historical_od_evolution_model.md"

OPERATING_DAY_ASSUMPTION = "05:00-00:00"
REPRESENTATIVE_WINDOW_LABEL = "06:00-08:00"
READY_TIME_LABELS = ["06:00", "07:00", "08:00"]
READY_TIME_MINUTES = [6 * 60, 7 * 60, 8 * 60]
UNREACHABLE_COST_MIN = 16 * 60
MISSING_PERIOD_PENALTY_FACTOR = 1.5
IPF_MAX_ITER = 500
IPF_TOL = 1e-8

WEIGHT_MODEL_2023 = "npvm_2023_observed_hybrid_selected_station_projection_with_same_agglomeration_distance_attenuation"
WEIGHT_MODEL_PROJECTED = "npvm_2023_anchor_population_gcost_ipf_projection"
GC_MODEL_NAME = "representative_ready_times_elapsed_time_06_07_08"

CANONICAL_ALL_OD_COLUMNS = [
    "origin",
    "destination",
    "year",
    "reference_year",
    "pair_weight",
    "observed_daily_trips",
    "distance_exponent",
    "weight_model",
    "source",
    "analysis_status",
    "exclude_from_analysis",
]

CANONICAL_SOURCE_BY_MODEL = {
    WEIGHT_MODEL_2023: (
        "Accepted 2023 NPVM-based selected-station anchor with inter-agglomeration projection and calibrated "
        "same-agglomeration distance attenuation."
    ),
    WEIGHT_MODEL_PROJECTED: (
        "Projected from the accepted 2023 selected-station anchor with calibrated same-agglomeration distance "
        "attenuation, representative 06:00, 07:00, and 08:00 generalized-cost skims for the repeating "
        "05:00-00:00 operating day, population-scaled marginals, and IPF balancing."
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the full historical All_OD_Pairings table from the accepted 2023 observed/hybrid "
            "analysis-layer station OD anchor, projected across all timetable years with timetable-based "
            "generalized-cost skims, demographic marginals, and IPF balancing."
        )
    )
    parser.add_argument(
        "--base-analysis-input",
        type=Path,
        default=BASE_2023_ANALYSIS_PATH,
        help="Path to the accepted 2023 website-facing analysis-layer OD table.",
    )
    parser.add_argument(
        "--filtered-stations",
        type=Path,
        default=FILTERED_STATIONS_PATH,
        help="Path to FilteredStations.csv for yearly station populations and service counts.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_ALL_OD_PATH,
        help="Path to the canonical live all-years OD pairing CSV.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT_PATH,
        help="Optional path to the verbose audit export with extra lineage columns. Use an empty string to skip it.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path to the machine-readable summary JSON.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Path to the human-readable validation markdown.",
    )
    parser.add_argument(
        "--skim-output",
        type=Path,
        default=DEFAULT_SKIM_PATH,
        help="Path to the audit CSV of yearly generalized-cost skims used by the OD evolution model.",
    )
    parser.add_argument(
        "--method-output",
        type=Path,
        default=DEFAULT_METHOD_PATH,
        help="Path to the markdown note documenting the implemented historical OD evolution model.",
    )
    parser.add_argument(
        "--years",
        nargs="*",
        default=None,
        help=(
            "Optional subset of timetable years to recompute. When set, the builder refreshes only those years "
            "and merges them back into the existing all-years outputs."
        ),
    )
    return parser.parse_args()


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    numeric = _safe_float(value, None)
    if numeric is None:
        return default
    return int(round(numeric))


def load_station_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["StationName"] = frame["StationName"].astype(str).str.strip()
    frame = frame[frame["StationName"].isin(OVERALL_ANALYSIS_TARGET_STATIONS)].copy()
    if frame.empty:
        raise ValueError("No overall-analysis stations found in FilteredStations.csv.")
    return frame


def station_population(frame: pd.DataFrame, station: str, year_label: str) -> int:
    row = frame.loc[frame["StationName"] == station]
    if row.empty:
        return DEFAULT_STATION_POPULATION
    raw = str(row.iloc[0].get(f"{year_label}_pop", "")).strip()
    if not raw:
        return DEFAULT_STATION_POPULATION
    try:
        value = int(float(raw))
    except ValueError:
        return DEFAULT_STATION_POPULATION
    return value if value > 0 else DEFAULT_STATION_POPULATION


def station_service_count(frame: pd.DataFrame, station: str, year_label: str) -> float:
    row = frame.loc[frame["StationName"] == station]
    if row.empty:
        return 0.0
    raw = str(row.iloc[0].get(year_label, "")).strip()
    if not raw:
        return 0.0
    try:
        return max(float(raw), 0.0)
    except ValueError:
        return 0.0


def load_station_metadata(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for _, row in frame.iterrows():
        station = str(row["StationName"]).strip()
        metadata[station] = {
            "country": str(row.get("Country", "")).strip(),
            "canton": str(row.get("Region", "")).strip(),
            "lat": float(str(row["lat"]).strip()),
            "lon": float(str(row["lon"]).strip()),
        }
    missing = [station for station in OVERALL_ANALYSIS_TARGET_STATIONS if station not in metadata]
    if missing:
        raise ValueError(f"Stations missing from FilteredStations.csv: {', '.join(missing)}")
    return metadata


def load_base_analysis_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["origin"] = frame["origin"].astype(str).str.strip()
    frame["destination"] = frame["destination"].astype(str).str.strip()
    frame["year"] = frame["year"].astype(str).str.strip()
    frame = frame[(frame["origin"] != "") & (frame["destination"] != "")].copy()
    frame = frame[frame["year"] == DEFAULT_REFERENCE_YEAR].copy()
    if frame.empty:
        raise ValueError(f"No 2023 rows found in {path}.")

    numeric_cols = [
        "origin_population",
        "destination_population",
        "distance_km",
        "pair_weight",
        "observed_daily_trips",
        "raw_station_observed_daily_trips",
        "cluster_pair_observed_daily_trips",
        "gravity_weight",
        "distance_exponent",
        "origin_outbound_share",
        "destination_inbound_share",
    ]
    for column in numeric_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["exclude_from_analysis"] = (
        frame["exclude_from_analysis"].astype(str).str.strip().str.lower().isin({"1", "true", "yes"})
    )
    return frame


def sort_year_labels(system: Any, years: list[str]) -> list[str]:
    sortable = pd.DataFrame({"Year": years})
    sorted_df = system._sort_year_df(sortable)
    return [str(item) for item in sorted_df["Year"].tolist()]


def normalize_requested_years(system: Any, available_years: list[str], requested_years: list[str] | None) -> list[str]:
    if not requested_years:
        return list(available_years)
    trimmed = [str(item).strip() for item in requested_years if str(item).strip()]
    if not trimmed:
        return list(available_years)
    available_set = set(available_years)
    missing = [year for year in trimmed if year not in available_set]
    if missing:
        raise ValueError(
            "Requested timetable years are not available in the current model: " + ", ".join(missing)
        )
    deduped = list(dict.fromkeys(trimmed))
    return sort_year_labels(system, deduped)


def load_existing_output_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def merge_year_refresh(
    existing_frame: pd.DataFrame,
    refreshed_frame: pd.DataFrame,
    refreshed_years: list[str],
) -> pd.DataFrame:
    if existing_frame.empty:
        return refreshed_frame.copy()
    existing = existing_frame.copy()
    if "year" in existing.columns:
        existing["year"] = existing["year"].astype(str).str.strip()
        existing = existing[~existing["year"].isin(refreshed_years)].copy()
    merged = pd.concat([existing, refreshed_frame], ignore_index=True, sort=False)
    return merged


def all_dest_ready_time_costs(module: Any, model: Any, origin: str, ready_time_min: int) -> dict[str, dict[str, int]]:
    start_nodes = module.seed_origin_source_edges(model, origin, ready_time_min)
    if not start_nodes:
        return {}

    dist: dict[int, tuple[int, int, int, int]] = {}
    pq: list[tuple[tuple[int, int, int, int], int]] = []
    inf_cost = (float("inf"), float("inf"), float("inf"), float("inf"))

    for start_node in start_nodes:
        start_time = model.nodes[start_node].time
        cost = (start_time, -start_time, 0, 0)
        if cost < dist.get(start_node, inf_cost):
            dist[start_node] = cost
            heapq.heappush(pq, (cost, start_node))

    best_station: dict[str, tuple[int, int, int, int]] = {}
    while pq:
        current_cost, node_id = heapq.heappop(pq)
        if current_cost != dist.get(node_id):
            continue
        node = model.nodes[node_id]

        if node.kind == "an":
            previous = best_station.get(node.station)
            if previous is None or current_cost < previous:
                best_station[node.station] = current_cost

        for edge in model.adj.get(node_id, []):
            to_node = model.nodes[edge.to_node]
            arrival_time = to_node.time
            first_dep_neg = current_cost[1]
            transfers = current_cost[2] + (1 if edge.edge_type == "transfer" else 0)
            movement_segments = current_cost[3] + module._movement_increment(
                model,
                node_id,
                edge.to_node,
                edge.edge_type,
            )
            new_cost = (arrival_time, first_dep_neg, transfers, movement_segments)

            if new_cost < dist.get(edge.to_node, inf_cost):
                dist[edge.to_node] = new_cost
                heapq.heappush(pq, (new_cost, edge.to_node))

    payload: dict[str, dict[str, int]] = {}
    for station, cost in best_station.items():
        arrival_time, first_dep_neg, transfers, movement_segments = cost
        payload[station] = {
            "arrivalTimeMin": int(arrival_time),
            "firstDepartureTimeMin": int(-first_dep_neg),
            "elapsedTimeMin": int(arrival_time - ready_time_min),
            "transfers": int(transfers),
            "movementSegments": int(movement_segments),
        }
    return payload


def build_gc_skims(
    module: Any,
    system: Any,
    selected_stations: list[str],
    skim_output_path: Path,
    years_to_refresh: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    all_years = sort_year_labels(system, list(system.models.keys()))
    requested_years = normalize_requested_years(system, all_years, years_to_refresh)
    cached_reuse_df = pd.DataFrame()
    if skim_output_path.exists():
        cached_df = pd.read_csv(skim_output_path, dtype=str).fillna("")
        required_columns = {"year", "origin", "destination", "gcModel", "gcMinutes", "reachablePeriods", "missingPeriods"}
        if required_columns.issubset(set(cached_df.columns)):
            cached_df["year"] = cached_df["year"].astype(str).str.strip()
            cached_years = sort_year_labels(system, sorted(set(cached_df["year"])))
            cached_models = set(cached_df["gcModel"].astype(str))
            if cached_models == {GC_MODEL_NAME}:
                cached_df["gcMinutes"] = cached_df["gcMinutes"].astype(float)
                cached_df["reachablePeriods"] = cached_df["reachablePeriods"].astype(int)
                cached_df["missingPeriods"] = cached_df["missingPeriods"].astype(int)
                if "meanTransfers" in cached_df.columns:
                    cached_df["meanTransfers"] = pd.to_numeric(cached_df["meanTransfers"], errors="coerce")
                if requested_years == all_years and cached_years == all_years:
                    diagnostics = {
                        "gcModel": GC_MODEL_NAME,
                        "readyTimes": READY_TIME_LABELS,
                        "operatingDayAssumption": OPERATING_DAY_ASSUMPTION,
                        "representativeWindow": REPRESENTATIVE_WINDOW_LABEL,
                        "yearsComputed": all_years,
                        "yearsRefreshed": [],
                        "skimRows": int(len(cached_df)),
                        "meanReadyTimeRuntimeSec": None,
                        "medianReadyTimeRuntimeSec": None,
                        "maxReadyTimeRuntimeSec": None,
                        "meanDestinationsReached": None,
                        "fullyReachablePairShare": round(
                            float((cached_df["reachablePeriods"] == len(READY_TIME_LABELS)).mean()),
                            4,
                        ),
                        "fullyUnreachablePairCount": int((cached_df["reachablePeriods"] == 0).sum()),
                        "reusedExistingSkimCache": True,
                        "skimCacheFile": str(skim_output_path),
                    }
                    print(f"[GC skim] Reusing existing skim cache from {skim_output_path.name}", flush=True)
                    return cached_df, diagnostics

                cached_reuse_df = cached_df[~cached_df["year"].isin(requested_years)].copy()
                missing_other_years = [
                    year for year in all_years if year not in requested_years and year not in set(cached_reuse_df["year"])
                ]
                if missing_other_years:
                    requested_years = sort_year_labels(system, requested_years + missing_other_years)
                if not cached_reuse_df.empty:
                    print(
                        f"[GC skim] Reusing cached years outside refresh scope from {skim_output_path.name}; "
                        f"refreshing {', '.join(requested_years)}.",
                        flush=True,
                    )

    rows: list[dict[str, Any]] = []
    timing_rows: list[dict[str, Any]] = []

    for year_idx, year_label in enumerate(requested_years, start=1):
        model = system.models[year_label]
        available = set(model.available_stations())
        for origin_idx, origin in enumerate(selected_stations, start=1):
            if origin not in available:
                continue
            for ready_label, ready_time_min in zip(READY_TIME_LABELS, READY_TIME_MINUTES):
                started = time.perf_counter()
                result = all_dest_ready_time_costs(module, model, origin, ready_time_min)
                timing_rows.append(
                    {
                        "year": year_label,
                        "origin": origin,
                        "readyTime": ready_label,
                        "runtimeSec": round(time.perf_counter() - started, 4),
                        "destinationsReached": len(result),
                    }
                )
                for destination in selected_stations:
                    if destination == origin:
                        continue
                    entry = result.get(destination)
                    rows.append(
                        {
                            "year": year_label,
                            "origin": origin,
                            "destination": destination,
                            "readyTime": ready_label,
                            "readyTimeMin": ready_time_min,
                            "reachable": bool(entry),
                            "arrivalTimeMin": entry["arrivalTimeMin"] if entry else None,
                            "firstDepartureTimeMin": entry["firstDepartureTimeMin"] if entry else None,
                            "elapsedTimeMin": entry["elapsedTimeMin"] if entry else None,
                            "transfers": entry["transfers"] if entry else None,
                            "movementSegments": entry["movementSegments"] if entry else None,
                        }
                    )
            print(
                f"[GC skim] {year_idx}/{len(requested_years)} {year_label} | {origin_idx}/{len(selected_stations)} {origin}",
                flush=True,
            )

    skim_df = pd.DataFrame(rows)
    if skim_df.empty:
        raise ValueError("No generalized-cost skim rows were generated.")

    aggregated_rows: list[dict[str, Any]] = []
    for (year_label, origin, destination), group in skim_df.groupby(["year", "origin", "destination"], sort=False):
        reachable_costs = [float(value) for value in group["elapsedTimeMin"].dropna().tolist()]
        reachable_transfers = [float(value) for value in group["transfers"].dropna().tolist()]
        reachable_periods = int(group["reachable"].sum())
        missing_periods = int(len(group) - reachable_periods)

        if reachable_costs:
            base_penalty = max(max(reachable_costs), 1.0) * MISSING_PERIOD_PENALTY_FACTOR
            penalty_cost = max(base_penalty, float(UNREACHABLE_COST_MIN))
            effective_costs = list(reachable_costs) + [penalty_cost] * missing_periods
            gc_avg = float(sum(effective_costs) / len(effective_costs))
            avg_transfers = float(sum(reachable_transfers) / len(reachable_transfers)) if reachable_transfers else None
        else:
            gc_avg = float(UNREACHABLE_COST_MIN)
            avg_transfers = None

        aggregated_rows.append(
            {
                "year": year_label,
                "origin": origin,
                "destination": destination,
                "gcModel": GC_MODEL_NAME,
                "gcMinutes": round(gc_avg, 3),
                "reachablePeriods": reachable_periods,
                "missingPeriods": missing_periods,
                "meanTransfers": round(avg_transfers, 3) if avg_transfers is not None else None,
            }
        )

    aggregated_df = pd.DataFrame(aggregated_rows)
    if not cached_reuse_df.empty:
        aggregated_df = pd.concat([cached_reuse_df, aggregated_df], ignore_index=True, sort=False)
    aggregated_df.to_csv(skim_output_path, index=False)

    timing_df = pd.DataFrame(timing_rows)
    diagnostics = {
        "gcModel": GC_MODEL_NAME,
        "readyTimes": READY_TIME_LABELS,
        "operatingDayAssumption": OPERATING_DAY_ASSUMPTION,
        "representativeWindow": REPRESENTATIVE_WINDOW_LABEL,
        "yearsComputed": sort_year_labels(system, sorted(set(aggregated_df["year"].astype(str)))),
        "yearsRefreshed": requested_years,
        "skimRows": int(len(aggregated_df)),
        "meanReadyTimeRuntimeSec": round(float(timing_df["runtimeSec"].mean()), 4) if not timing_df.empty else None,
        "medianReadyTimeRuntimeSec": round(float(timing_df["runtimeSec"].median()), 4) if not timing_df.empty else None,
        "maxReadyTimeRuntimeSec": round(float(timing_df["runtimeSec"].max()), 4) if not timing_df.empty else None,
        "meanDestinationsReached": round(float(timing_df["destinationsReached"].mean()), 2) if not timing_df.empty else None,
        "fullyReachablePairShare": round(
            float((aggregated_df["reachablePeriods"] == len(READY_TIME_LABELS)).mean()),
            4,
        ),
        "fullyUnreachablePairCount": int((aggregated_df["reachablePeriods"] == 0).sum()),
        "reusedExistingSkimCache": False,
        "skimCacheFile": str(skim_output_path),
    }
    return aggregated_df, diagnostics


def calibrate_gc_beta(base_active_df: pd.DataFrame, gc_2023_df: pd.DataFrame) -> dict[str, Any]:
    merged = base_active_df.merge(
        gc_2023_df[["origin", "destination", "gcMinutes"]],
        on=["origin", "destination"],
        how="left",
    )
    merged = merged[(merged["pair_weight"] > 0) & merged["gcMinutes"].notna() & (merged["gcMinutes"] > 0)].copy()
    if merged.empty:
        raise ValueError("No valid 2023 rows available to calibrate the generalized-cost sensitivity.")

    row_totals = merged.groupby("origin")["pair_weight"].sum()
    col_totals = merged.groupby("destination")["pair_weight"].sum()
    grand_total = float(merged["pair_weight"].sum())

    merged["expected_independence"] = merged.apply(
        lambda row: (row_totals[row["origin"]] * col_totals[row["destination"]]) / grand_total,
        axis=1,
    )
    merged = merged[merged["expected_independence"] > 0].copy()
    merged["log_residual"] = np.log(merged["pair_weight"] / merged["expected_independence"])
    merged["log_gc"] = np.log(merged["gcMinutes"])

    weights = np.sqrt(merged["pair_weight"].to_numpy(dtype=float))
    x = merged["log_gc"].to_numpy(dtype=float)
    y = merged["log_residual"].to_numpy(dtype=float)
    w = weights

    x_bar = np.average(x, weights=w)
    y_bar = np.average(y, weights=w)
    denominator = np.sum(w * (x - x_bar) ** 2)
    if denominator <= 0:
        beta = 1.0
        intercept = float(y_bar)
        r_squared = None
    else:
        slope = np.sum(w * (x - x_bar) * (y - y_bar)) / denominator
        intercept = float(y_bar - slope * x_bar)
        beta = max(0.01, float(-slope))
        y_hat = intercept + slope * x
        ss_res = np.sum(w * (y - y_hat) ** 2)
        ss_tot = np.sum(w * (y - y_bar) ** 2)
        r_squared = None if ss_tot <= 0 else float(max(0.0, 1.0 - (ss_res / ss_tot)))

    return {
        "beta": round(beta, 6),
        "intercept": round(intercept, 6),
        "rowsUsed": int(len(merged)),
        "weightedRsq": None if r_squared is None else round(r_squared, 6),
        "grandTotal": round(grand_total, 6),
    }


def build_year_targets(
    base_active_df: pd.DataFrame,
    station_frame: pd.DataFrame,
    selected_stations: list[str],
    year_label: str,
    active_stations_by_year: dict[str, set[str]],
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    base_outbound = base_active_df.groupby("origin")["pair_weight"].sum().to_dict()
    base_inbound = base_active_df.groupby("destination")["pair_weight"].sum().to_dict()

    year_active = active_stations_by_year[year_label]
    raw_origin_targets: dict[str, float] = {}
    raw_destination_targets: dict[str, float] = {}
    inactive_zeroed: list[str] = []

    for station in selected_stations:
        pop_2023 = max(station_population(station_frame, station, DEFAULT_REFERENCE_YEAR), 1)
        pop_year = station_population(station_frame, station, year_label)
        growth_factor = pop_year / pop_2023
        service_count = station_service_count(station_frame, station, year_label)
        is_active = station in year_active and service_count > 0
        if not is_active:
            raw_origin_targets[station] = 0.0
            raw_destination_targets[station] = 0.0
            inactive_zeroed.append(station)
            continue
        raw_origin_targets[station] = float(base_outbound.get(station, 0.0) * growth_factor)
        raw_destination_targets[station] = float(base_inbound.get(station, 0.0) * growth_factor)

    sum_origin = sum(raw_origin_targets.values())
    sum_destination = sum(raw_destination_targets.values())
    if sum_origin > 0 and sum_destination > 0:
        target_total = 0.5 * (sum_origin + sum_destination)
        origin_scale = target_total / sum_origin
        destination_scale = target_total / sum_destination
    elif sum_origin > 0:
        origin_scale = 1.0
        destination_scale = 0.0
        target_total = sum_origin
    elif sum_destination > 0:
        origin_scale = 0.0
        destination_scale = 1.0
        target_total = sum_destination
    else:
        origin_scale = 0.0
        destination_scale = 0.0
        target_total = 0.0

    origin_targets = {station: value * origin_scale for station, value in raw_origin_targets.items()}
    destination_targets = {
        station: value * destination_scale for station, value in raw_destination_targets.items()
    }

    diagnostics = {
        "targetTotal": round(target_total, 6),
        "sumOriginBeforeScaling": round(sum_origin, 6),
        "sumDestinationBeforeScaling": round(sum_destination, 6),
        "originScale": round(origin_scale, 9),
        "destinationScale": round(destination_scale, 9),
        "inactiveStationsZeroed": sorted(inactive_zeroed),
    }
    return origin_targets, destination_targets, diagnostics


def ipf_balance(
    seed_matrix: np.ndarray,
    row_targets: np.ndarray,
    col_targets: np.ndarray,
    active_mask: np.ndarray,
    max_iter: int = IPF_MAX_ITER,
    tol: float = IPF_TOL,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = seed_matrix.copy().astype(float)
    matrix[~active_mask] = 0.0

    if matrix.shape[0] != len(row_targets) or matrix.shape[1] != len(col_targets):
        raise ValueError("IPF dimensions do not align.")

    iterations = 0
    converged = False
    for iterations in range(1, max_iter + 1):
        for row_idx in range(matrix.shape[0]):
            target = row_targets[row_idx]
            current = matrix[row_idx, :].sum()
            if target <= 0 or current <= 0:
                matrix[row_idx, :] = 0.0
            else:
                matrix[row_idx, :] *= target / current

        for col_idx in range(matrix.shape[1]):
            target = col_targets[col_idx]
            current = matrix[:, col_idx].sum()
            if target <= 0 or current <= 0:
                matrix[:, col_idx] = 0.0
            else:
                matrix[:, col_idx] *= target / current

        row_error = np.max(np.abs(matrix.sum(axis=1) - row_targets))
        col_error = np.max(np.abs(matrix.sum(axis=0) - col_targets))
        if max(row_error, col_error) <= tol:
            converged = True
            break

    diagnostics = {
        "iterations": iterations,
        "converged": converged,
        "rowErrorMax": round(float(np.max(np.abs(matrix.sum(axis=1) - row_targets))), 12),
        "colErrorMax": round(float(np.max(np.abs(matrix.sum(axis=0) - col_targets))), 12),
    }
    return matrix, diagnostics


def build_all_pairings(
    base_table: pd.DataFrame,
    station_frame: pd.DataFrame,
    station_metadata: dict[str, dict[str, Any]],
    gc_df: pd.DataFrame,
    beta: float,
    year_labels: list[str],
    active_stations_by_year: dict[str, set[str]],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    selected_stations = list(OVERALL_ANALYSIS_TARGET_STATIONS)
    station_index = {station: idx for idx, station in enumerate(selected_stations)}

    base_table = base_table.copy()
    base_table["base_pair_weight_2023"] = base_table["pair_weight"].astype(float)
    base_table["base_observed_daily_trips_2023"] = pd.to_numeric(
        base_table["observed_daily_trips"],
        errors="coerce",
    )

    base_active = base_table[(~base_table["exclude_from_analysis"]) & (base_table["pair_weight"] > 0)].copy()
    base_active["pair_weight"] = base_active["pair_weight"].astype(float)

    gc_lookup = {
        (str(row["year"]), str(row["origin"]), str(row["destination"])): row
        for _, row in gc_df.iterrows()
    }

    rows_out: list[dict[str, Any]] = []
    year_summaries: dict[str, Any] = {}

    for year_label in year_labels:
        origin_targets, destination_targets, target_diag = build_year_targets(
            base_active,
            station_frame,
            selected_stations,
            year_label,
            active_stations_by_year,
        )

        row_targets_arr = np.array([origin_targets[station] for station in selected_stations], dtype=float)
        col_targets_arr = np.array([destination_targets[station] for station in selected_stations], dtype=float)

        seed_matrix = np.zeros((len(selected_stations), len(selected_stations)), dtype=float)
        active_mask = np.zeros_like(seed_matrix, dtype=bool)

        seed_positive_pairs = 0
        gc_missing_pairs = 0
        inactive_year_pairs = 0

        for _, base_row in base_table.iterrows():
            origin = str(base_row["origin"])
            destination = str(base_row["destination"])
            i = station_index[origin]
            j = station_index[destination]

            if bool(base_row["exclude_from_analysis"]):
                continue

            if row_targets_arr[i] <= 0 or col_targets_arr[j] <= 0:
                inactive_year_pairs += 1
                continue

            base_weight = float(base_row["base_pair_weight_2023"])
            if base_weight <= 0:
                inactive_year_pairs += 1
                continue
            gc_ref_row = gc_lookup.get((DEFAULT_REFERENCE_YEAR, origin, destination))
            gc_year_row = gc_lookup.get((year_label, origin, destination))
            gc_ref = float(gc_ref_row["gcMinutes"]) if gc_ref_row is not None else float(UNREACHABLE_COST_MIN)
            gc_year = float(gc_year_row["gcMinutes"]) if gc_year_row is not None else float(UNREACHABLE_COST_MIN)
            if gc_year_row is None:
                gc_missing_pairs += 1
            gc_ref = max(gc_ref, 1.0)
            gc_year = max(gc_year, 1.0)
            cost_factor = (gc_ref / gc_year) ** beta
            seed_matrix[i, j] = max(base_weight * cost_factor, 1e-12)
            active_mask[i, j] = True
            seed_positive_pairs += 1

        balanced_matrix, ipf_diag = ipf_balance(seed_matrix, row_targets_arr, col_targets_arr, active_mask)
        year_summaries[year_label] = {
            "targetDiagnostics": target_diag,
            "ipfDiagnostics": ipf_diag,
            "seedPositivePairs": seed_positive_pairs,
            "gcMissingPairs": gc_missing_pairs,
            "inactiveYearPairs": inactive_year_pairs,
            "projectedTotal": round(float(balanced_matrix.sum()), 6),
        }

        for _, base_row in base_table.iterrows():
            origin = str(base_row["origin"])
            destination = str(base_row["destination"])
            i = station_index[origin]
            j = station_index[destination]

            pair_distance = float(base_row["distance_km"])
            origin_population = station_population(station_frame, origin, year_label)
            destination_population = station_population(station_frame, destination, year_label)
            gravity_weight = (
                (origin_population * destination_population) / max(pair_distance, 1.0) ** DEFAULT_DISTANCE_EXPONENT
            )

            gc_ref_row = gc_lookup.get((DEFAULT_REFERENCE_YEAR, origin, destination))
            gc_year_row = gc_lookup.get((year_label, origin, destination))
            gc_ref = float(gc_ref_row["gcMinutes"]) if gc_ref_row is not None else float(UNREACHABLE_COST_MIN)
            gc_year = float(gc_year_row["gcMinutes"]) if gc_year_row is not None else float(UNREACHABLE_COST_MIN)
            gc_ratio = gc_ref / gc_year if gc_year > 0 else None

            base_status = str(base_row.get("analysis_status", "")).strip()
            is_same_agglomeration_pair = "same_agglomeration" in base_status

            if bool(base_row["exclude_from_analysis"]):
                pair_weight = 0.0
                observed_daily_trips = (
                    float(base_row["observed_daily_trips"])
                    if year_label == DEFAULT_REFERENCE_YEAR and not pd.isna(base_row["observed_daily_trips"])
                    else None
                )
                analysis_status = "excluded_same_agglomeration"
                weight_model = WEIGHT_MODEL_2023 if year_label == DEFAULT_REFERENCE_YEAR else WEIGHT_MODEL_PROJECTED
                notes = (
                    "This directed pair remains inside the same official BFS AGGL2020 agglomeration (or singleton "
                    "analysis cluster) and therefore stays excluded from the network-wide OD weighting layer in every year."
                )
                exclude_flag = True
                seed_before_ipf = 0.0
            elif row_targets_arr[i] <= 0 or col_targets_arr[j] <= 0 or float(base_row["base_pair_weight_2023"]) <= 0:
                pair_weight = 0.0
                observed_daily_trips = None
                analysis_status = "inactive_station_year"
                weight_model = WEIGHT_MODEL_PROJECTED
                notes = (
                    "This pair is set to zero in this year because the origin or destination station is not active in "
                    "the timetable model or has zero counted services in FilteredStations.csv for that year."
                )
                exclude_flag = False
                seed_before_ipf = 0.0
            else:
                pair_weight = float(balanced_matrix[i, j])
                observed_daily_trips = (
                    float(base_row["observed_daily_trips"])
                    if year_label == DEFAULT_REFERENCE_YEAR and not pd.isna(base_row["observed_daily_trips"])
                    else None
                )
                if is_same_agglomeration_pair:
                    analysis_status = (
                        "observed_active_same_agglomeration_distance_attenuated_2023"
                        if year_label == DEFAULT_REFERENCE_YEAR
                        else "projected_active_same_agglomeration_distance_attenuated_ipf"
                    )
                else:
                    analysis_status = (
                        "observed_active_inter_agglomeration_2023"
                        if year_label == DEFAULT_REFERENCE_YEAR
                        else "projected_active_inter_agglomeration_ipf"
                    )
                weight_model = WEIGHT_MODEL_2023 if year_label == DEFAULT_REFERENCE_YEAR else WEIGHT_MODEL_PROJECTED
                notes = (
                    "Projected from the accepted 2023 selected-station OD anchor with a timetable-based generalized-cost "
                    "pivot factor and year-specific demographic marginals, then balanced with IPF to match the year's "
                    "origin and destination targets."
                    if year_label != DEFAULT_REFERENCE_YEAR
                    else str(base_row["notes"]).strip()
                )
                exclude_flag = False
                seed_before_ipf = float(seed_matrix[i, j])

            row_out = {
                "origin": origin,
                "destination": destination,
                "year": year_label,
                "reference_year": DEFAULT_REFERENCE_YEAR,
                "origin_country": str(base_row["origin_country"]).strip(),
                "destination_country": str(base_row["destination_country"]).strip(),
                "origin_canton": str(base_row["origin_canton"]).strip(),
                "destination_canton": str(base_row["destination_canton"]).strip(),
                "origin_population": origin_population,
                "destination_population": destination_population,
                "distance_km": round(pair_distance, 6),
                "pair_weight": round(pair_weight, 12),
                "observed_daily_trips": (
                    "" if observed_daily_trips is None else round(float(observed_daily_trips), 12)
                ),
                "base_pair_weight_2023": round(float(base_row["base_pair_weight_2023"]), 12),
                "base_observed_daily_trips_2023": (
                    ""
                    if pd.isna(base_row["base_observed_daily_trips_2023"])
                    else round(float(base_row["base_observed_daily_trips_2023"]), 12)
                ),
                "raw_station_observed_daily_trips": (
                    round(float(base_row["raw_station_observed_daily_trips"]), 12)
                    if not pd.isna(base_row["raw_station_observed_daily_trips"])
                    else ""
                ),
                "cluster_pair_observed_daily_trips": (
                    round(float(base_row["cluster_pair_observed_daily_trips"]), 12)
                    if not pd.isna(base_row["cluster_pair_observed_daily_trips"])
                    else ""
                ),
                "gravity_weight": round(gravity_weight, 12),
                "distance_exponent": DEFAULT_DISTANCE_EXPONENT,
                "weight_model": weight_model,
                "analysis_status": analysis_status,
                "exclude_from_analysis": int(exclude_flag),
                "origin_cluster_id": str(base_row["origin_cluster_id"]).strip(),
                "destination_cluster_id": str(base_row["destination_cluster_id"]).strip(),
                "origin_cluster_scope": str(base_row["origin_cluster_scope"]).strip(),
                "destination_cluster_scope": str(base_row["destination_cluster_scope"]).strip(),
                "origin_aggl2020_code": str(base_row["origin_aggl2020_code"]).strip(),
                "destination_aggl2020_code": str(base_row["destination_aggl2020_code"]).strip(),
                "origin_outbound_share": (
                    round(float(base_row["origin_outbound_share"]), 12)
                    if not pd.isna(base_row["origin_outbound_share"])
                    else ""
                ),
                "destination_inbound_share": (
                    round(float(base_row["destination_inbound_share"]), 12)
                    if not pd.isna(base_row["destination_inbound_share"])
                    else ""
                ),
                "origin_share_method": str(base_row["origin_share_method"]).strip(),
                "destination_share_method": str(base_row["destination_share_method"]).strip(),
                "source": (
                    str(base_row["source"]).strip()
                    if year_label == DEFAULT_REFERENCE_YEAR
                    else (
                        "Accepted 2023 NPVM-based selected-station anchor with calibrated same-agglomeration distance "
                        "attenuation carried across years with representative ready-time generalized-cost skims "
                        "(06:00, 07:00, 08:00) for the repeating 05:00-00:00 operating day, population-scaled "
                        "origin/destination marginals, and IPF balancing on the 131-station analysis layer."
                    )
                ),
                "notes": notes,
                "gc_model": GC_MODEL_NAME,
                "gc_minutes_reference_2023": round(gc_ref, 6),
                "gc_minutes_year": round(gc_year, 6),
                "gc_ratio_reference_over_year": "" if gc_ratio is None else round(gc_ratio, 12),
                "reachable_periods_year": (
                    int(gc_year_row["reachablePeriods"]) if gc_year_row is not None else 0
                ),
                "missing_periods_year": (
                    int(gc_year_row["missingPeriods"]) if gc_year_row is not None else len(READY_TIME_LABELS)
                ),
                "mean_transfers_year": (
                    ""
                    if gc_year_row is None or pd.isna(gc_year_row["meanTransfers"])
                    else round(float(gc_year_row["meanTransfers"]), 6)
                ),
                "seed_pair_weight_pre_ipf": round(seed_before_ipf, 12),
                "origin_target_total_year": round(float(row_targets_arr[i]), 12),
                "destination_target_total_year": round(float(col_targets_arr[j]), 12),
            }
            rows_out.append(row_out)

        print(f"[OD projection] built year {year_label}", flush=True)

    all_df = pd.DataFrame(rows_out)
    diagnostics = {
        "years": year_labels,
        "yearSummaries": year_summaries,
    }
    return all_df, diagnostics


def write_validation_markdown(
    path: Path,
    summary: dict[str, Any],
) -> None:
    lines = [
        "# All_OD_Pairings validation",
        "",
        "## Purpose",
        "",
        "This note records the implemented historical OD pairing table that now extends the accepted 2023",
        "selected-station analysis layer across all timetable years in the thesis system.",
        "",
        "## Implemented model",
        "",
        "- Base layer: accepted `historical_od_pair_weights_reference_2023.csv` 2023 analysis table",
        (
            "- Time-varying impedance: representative ready-time timetable skims at `06:00`, `07:00`, and `08:00`, "
            f"used as the repeating service proxy for the thesis operating day `{OPERATING_DAY_ASSUMPTION}`"
        ),
        "- Generalized cost used in the pivot: elapsed time from the ready time to arrival, so initial waiting is included",
        "- Growth of marginals: station `*_pop` values from `FilteredStations.csv`",
        "- Balancing: doubly constrained IPF on the 131-station analysis layer",
        "- Applied-method precedent: DfT matrix-building guidance describes both demand-pivot approaches and the Furness process as standard reference-matrix forecasting tools",
        "",
        "## Calibration",
        "",
        f"- Calibrated beta: `{summary['calibration']['beta']}`",
        f"- Weighted R^2: `{summary['calibration']['weightedRsq']}`",
        f"- Rows used in calibration: `{summary['calibration']['rowsUsed']}`",
        "",
        "## Skim diagnostics",
        "",
        f"- Generalized-cost skim model: `{summary['gcSkims']['gcModel']}`",
        f"- Ready times: `{', '.join(summary['gcSkims']['readyTimes'])}`",
        f"- Thesis operating-day assumption: `{summary['gcSkims']['operatingDayAssumption']}`",
        f"- Representative skim window used in practice: `{summary['gcSkims']['representativeWindow']}`",
        f"- Mean ready-time runtime: `{summary['gcSkims']['meanReadyTimeRuntimeSec']} s`",
        f"- Fully reachable pair share: `{summary['gcSkims']['fullyReachablePairShare']}`",
        f"- Fully unreachable pair count: `{summary['gcSkims']['fullyUnreachablePairCount']}`",
        "",
        "## Output diagnostics",
        "",
        f"- Output rows: `{summary['outputRows']}`",
        f"- Distinct years: `{summary['yearsCount']}`",
        f"- Active projected rows: `{summary['activeRows']}`",
        f"- Explicit zero/excluded rows: `{summary['excludedRows']}`",
        f"- Inactive station-year zero rows: `{summary['inactiveYearRows']}`",
        "",
        "## Interpretation",
        "",
        "The resulting `All_OD_Pairings.csv` is not a set of observed matrices for every year. It is a",
        "year-specific **modelled OD series** anchored in the accepted 2023 observed/hybrid table, then",
        "updated across years with timetable-based generalized cost and population-scaled marginals.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def canonicalize_all_pairings_frame(frame: pd.DataFrame) -> pd.DataFrame:
    canonical = frame.copy()
    canonical["source"] = canonical["weight_model"].map(CANONICAL_SOURCE_BY_MODEL).fillna(
        "Local all-years station-pair OD table."
    )
    return canonical[CANONICAL_ALL_OD_COLUMNS].copy()


def write_method_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Historical OD evolution model (implemented March 2026)",
        "",
        "## Role in the workspace",
        "",
        "This note documents the currently implemented behavior that builds the thesis-wide `All_OD_Pairings.csv` file.",
        "It supersedes the earlier live assumption that the website weighting layer was only a fixed 2023 reference-year table.",
        "",
        "## Current hierarchy",
        "",
        "1. The raw 2023 master anchor remains the audit-level observed/hybrid OD source.",
        "2. The accepted 2023 131-station selected-station analysis table remains the live base layer for the website.",
        "3. `All_OD_Pairings.csv` now evolves that accepted 2023 analysis layer across all timetable years.",
        "4. Gravity remains only a fallback for genuinely missing rows or supporting comparisons.",
        "",
        "## Why the historical evolution is implemented on the 131-station analysis layer",
        "",
        "The accepted 2023 raw 250-station master matrix remains the core audit object, but the year-specific live",
        "OD tables are currently implemented on the 131-station analysis layer because:",
        "",
        "- the website and overall historical metric operate on that analysis layer,",
        "- same-agglomeration demand is kept in the live metric with a calibrated distance attenuation rather than discarded entirely,",
        "- and the full timetable-based generalized-cost skimming can therefore be precomputed with a tractable offline batch",
        "  while staying aligned with the actual public-facing analysis object.",
        "",
        "This is an implementation decision, not a claim that the 250-station historical master is conceptually unimportant.",
        "The raw 2023 master still remains the accepted observed/hybrid anchor.",
        "",
        "## Generalized-cost skim",
        "",
        "For each timetable year and each selected origin station, the implementation runs three representative ready-time skims:",
        "",
        f"- `{READY_TIME_LABELS[0]}`",
        f"- `{READY_TIME_LABELS[1]}`",
        f"- `{READY_TIME_LABELS[2]}`",
        "",
        f"The thesis-wide operating-day assumption is `{OPERATING_DAY_ASSUMPTION}`. The OD evolution builder does not",
        f"recompute every hour of that full day. Instead, it uses the repeating integrated-timetable structure in the",
        f"representative window `{REPRESENTATIVE_WINDOW_LABEL}` as the cost proxy for that operating day.",
        "",
        "Each skim computes the best lexicographic arrival to all selected destinations simultaneously from one origin.",
        "The cost recorded for the OD pair is the full elapsed time from the requested ready time to arrival, so the",
        "initial waiting before boarding is included. This is important because the route engine's internal travel-time",
        "figures exclude that initial waiting by design.",
        "",
        "If a destination is unreachable in one or more ready-time periods, the missing period is penalized rather than dropped,",
        "so the final generalized cost does not look artificially optimistic. If a pair is unreachable in all three representative periods,",
        f"a conservative placeholder cost of `{UNREACHABLE_COST_MIN}` minutes is used for the pivot factor.",
        "",
        "## Cost sensitivity calibration",
        "",
        "The model calibrates a single generalized-cost sensitivity on the accepted 2023 active selected-station rows.",
        "It does this by comparing each observed 2023 pair flow against its independence expectation from the 2023 row and",
        "column totals, then regressing the residual on the logarithm of the 2023 generalized cost.",
        "",
        f"- Calibrated beta: `{summary['calibration']['beta']}`",
        f"- Weighted R^2: `{summary['calibration']['weightedRsq']}`",
        "",
        "This yields a parsimonious elasticity-like parameter without pretending that a complete destination-choice system has",
        "been estimated for every year.",
        "",
        "## Marginal growth",
        "",
        "For every station and year:",
        "",
        "- the 2023 outbound total is scaled by the station's year-specific population ratio,",
        "- the 2023 inbound total is scaled the same way,",
        "- and stations with no active service in a year are set to zero on both sides.",
        "",
        "The year-specific population inputs come from the `_pop` columns in `FilteredStations.csv`, which already embed the",
        "service-weighted split for same-municipality multi-station cases.",
        "",
        "## Balancing",
        "",
        "The pre-balanced seed for each active pair is:",
        "",
        "`T_ij,y^seed = T_ij,2023 * (GC_ij,2023 / GC_ij,y)^beta`",
        "",
        "These seeds are then balanced with iterative proportional fitting (IPF) so that the yearly matrix matches the",
        "population-scaled origin and destination totals. Same-agglomeration pairs keep the accepted 2023 distance-attenuated",
        "status through the yearly projection, while station-year inactive rows stay explicit zeros so they do not silently",
        "fall back to gravity later.",
        "",
        "This follows the applied transport-model tradition of pivoting from a reference matrix and then using Furness/IPF",
        "to restore the target marginals rather than pretending that a fresh observed OD matrix exists for each year.",
        "",
        "## Output semantics",
        "",
        "In `All_OD_Pairings.csv`:",
        "",
        "- `pair_weight` is the live year-specific OD weight used by the website and overall analysis",
        "- `observed_daily_trips` is populated only on the 2023 observed rows",
        "- `weight_model`, `source`, and `analysis_status` explain whether the row is an observed 2023 anchor row, a projected row, or an explicit zero / excluded row",
        "- `exclude_from_analysis = 1` only marks any rows intentionally kept out of the analysis; under the current calibrated attenuation design, same-agglomeration selected-station rows stay active unless their base selected-station weight is zero",
        "",
        "The fuller audit details stay in `All_OD_Pairings_full_audit.csv`, including:",
        "",
        "- the 2023 base pair weight carried behind each row",
        "- generalized-cost pivot fields such as `gc_minutes_reference_2023` and `gc_minutes_year`",
        "- the pre-balanced seed before IPF",
        "- and the origin/destination target totals used in balancing",
        "",
        "## Current implementation scope",
        "",
        "This now means the website's historical overall analysis is no longer driven by a fixed 2023 demand structure.",
        "It is driven by a year-specific OD table series anchored in 2023 and evolved with both demography and timetable-based cost.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    station_frame = load_station_frame(args.filtered_stations)
    station_metadata = load_station_metadata(station_frame)
    base_table = load_base_analysis_table(args.base_analysis_input)

    module, system = build_system(PROJECT_DIR, verbose_validation=False)
    all_year_labels = sort_year_labels(system, list(system.models.keys()))
    year_labels = normalize_requested_years(system, all_year_labels, args.years)
    active_stations_by_year = {
        year_label: set(system.models[year_label].available_stations()) for year_label in all_year_labels
    }

    gc_df, gc_diagnostics = build_gc_skims(
        module,
        system,
        list(OVERALL_ANALYSIS_TARGET_STATIONS),
        args.skim_output,
        years_to_refresh=year_labels,
    )
    gc_2023_df = gc_df[gc_df["year"] == DEFAULT_REFERENCE_YEAR].copy()

    calibration = calibrate_gc_beta(
        base_table[~base_table["exclude_from_analysis"]].copy(),
        gc_2023_df,
    )

    all_pairings_df, pairing_diagnostics = build_all_pairings(
        base_table,
        station_frame,
        station_metadata,
        gc_df,
        calibration["beta"],
        year_labels,
        active_stations_by_year,
    )
    audit_output = None if str(args.audit_output).strip() == "" else args.audit_output
    existing_canonical = pd.DataFrame()
    if args.years:
        existing_canonical = load_existing_output_frame(args.output)
        if audit_output is not None:
            existing_audit = load_existing_output_frame(audit_output)
            all_pairings_df = merge_year_refresh(existing_audit, all_pairings_df, year_labels)
    if audit_output is not None:
        all_pairings_df.to_csv(audit_output, index=False)

    canonical_all_pairings_df = canonicalize_all_pairings_frame(all_pairings_df)
    if args.years and audit_output is None:
        canonical_all_pairings_df = merge_year_refresh(existing_canonical, canonical_all_pairings_df, year_labels)
    canonical_all_pairings_df.to_csv(args.output, index=False)

    exclude_mask = (
        canonical_all_pairings_df["exclude_from_analysis"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin({"1", "true", "yes"})
    )
    pair_weight_numeric = pd.to_numeric(canonical_all_pairings_df["pair_weight"], errors="coerce").fillna(0.0)
    active_rows = int(
        ((~exclude_mask) & (pair_weight_numeric > 0)).sum()
    )
    excluded_rows = int(exclude_mask.sum())
    inactive_year_rows = int((canonical_all_pairings_df["analysis_status"].astype(str) == "inactive_station_year").sum())
    output_year_labels = sort_year_labels(
        system,
        sorted(set(canonical_all_pairings_df["year"].astype(str).str.strip().tolist())),
    )

    summary = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "buildRuntimeSec": round(time.perf_counter() - started, 2),
        "baseAnalysisInput": str(args.base_analysis_input),
        "outputFile": str(args.output),
        "auditOutputFile": None if audit_output is None else str(audit_output),
        "skimFile": str(args.skim_output),
        "refreshedYears": year_labels,
        "gcSkims": gc_diagnostics,
        "calibration": calibration,
        "outputRows": int(len(canonical_all_pairings_df)),
        "yearsCount": len(output_year_labels),
        "yearLabels": output_year_labels,
        "activeRows": active_rows,
        "excludedRows": excluded_rows,
        "inactiveYearRows": inactive_year_rows,
        "weightModels": sorted(set(str(item) for item in canonical_all_pairings_df["weight_model"].unique())),
        "canonicalColumns": CANONICAL_ALL_OD_COLUMNS,
        "pairingDiagnostics": pairing_diagnostics,
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_validation_markdown(args.validation_output, summary)
    write_method_markdown(args.method_output, summary)

    print(
        f"Built {len(canonical_all_pairings_df)} canonical rows across {len(year_labels)} timetable years in "
        f"{summary['buildRuntimeSec']:.2f}s. Output: {args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
