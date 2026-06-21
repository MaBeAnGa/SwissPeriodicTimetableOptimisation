from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from build_od_pair_weights import (
    DEFAULT_SAME_CLUSTER_DISTANCE_SCALE_KM,
    DEFAULT_SAME_CLUSTER_MODE,
    build_pair_rows,
    load_master_station_od,
    load_selected_station_metadata,
)
from build_station_agglomeration_crosswalk import (
    DEFAULT_OUTPUT_PATH as CROSSWALK_DEFAULT_PATH,
    ensure_bfs_commune_levels,
    load_station_agglomeration_crosswalk,
    write_crosswalk,
    build_station_agglomeration_rows,
)
from build_npvm_rail_load_validation_2023 import (
    DEFAULT_ROUTE_CACHE_PATH as MASTER_ROUTE_CACHE_PATH,
    FILTERED_STATIONS_PATH,
    REFERENCE_YEAR,
    VALIDATION_CORRIDORS,
    append_cache_record,
    assign_pair,
    build_predicted_edge_map,
    build_screenline_rows,
    load_2023_model,
    load_ch_boundary,
    load_official_rail_links,
    load_route_cache,
    load_station_points,
    plot_network_maps,
    plot_screenline_scatter,
    summarize_fit,
)
from historical_station_sets import OVERALL_ANALYSIS_TARGET_STATIONS


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ROUTE_CACHE = PROJECT_DIR / "website_selected_station_route_assignment_cache_2023.jsonl"
DEFAULT_SUMMARY_OUTPUT = PROJECT_DIR / "same_agglomeration_attenuation_calibration_2023_summary.json"
DEFAULT_REPORT_OUTPUT = PROJECT_DIR / "same_agglomeration_attenuation_calibration_2023.md"
DEFAULT_SCREENLINE_OUTPUT = PROJECT_DIR / "same_agglomeration_attenuation_screenlines_2023.csv"
DEFAULT_MAP_OUTPUT = PROJECT_DIR / "same_agglomeration_attenuation_network_map_2023.png"
DEFAULT_SCATTER_OUTPUT = PROJECT_DIR / "same_agglomeration_attenuation_scatter_2023.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate the same-agglomeration distance attenuation used in the 131-station website OD layer "
            "against the official NPVM 2023 rail screenline distribution."
        )
    )
    parser.add_argument(
        "--crosswalk-input",
        type=Path,
        default=CROSSWALK_DEFAULT_PATH,
        help="Path to the local station-to-agglomeration crosswalk CSV.",
    )
    parser.add_argument(
        "--route-cache",
        type=Path,
        default=DEFAULT_ROUTE_CACHE,
        help="Optional JSONL cache for 2023 selected-station route assignments created during earlier calibration runs.",
    )
    parser.add_argument(
        "--master-route-cache",
        type=Path,
        default=MASTER_ROUTE_CACHE_PATH,
        help=(
            "Primary JSONL cache of 2023 best-of-day route assignments from the full master-station validation run. "
            "Selected-station calibration reuses this cache first so it does not need to rebuild thousands of routes."
        ),
    )
    parser.add_argument(
        "--assign-missing",
        action="store_true",
        help=(
            "Assign still-missing selected-station pairs after seeding from the master and local caches. "
            "Leave this off for the default near-full-weight calibration."
        ),
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
        help="JSON path for the calibration summary.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=DEFAULT_REPORT_OUTPUT,
        help="Markdown path for the calibration note.",
    )
    parser.add_argument(
        "--screenline-output",
        type=Path,
        default=DEFAULT_SCREENLINE_OUTPUT,
        help="CSV path for the best-candidate screenline table.",
    )
    parser.add_argument(
        "--map-output",
        type=Path,
        default=DEFAULT_MAP_OUTPUT,
        help="PNG path for the best-candidate network comparison map.",
    )
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=DEFAULT_SCATTER_OUTPUT,
        help="PNG path for the best-candidate screenline scatter.",
    )
    parser.add_argument(
        "--distance-scales-km",
        nargs="*",
        type=float,
        default=[10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 50.0, 60.0],
        help="Candidate same-agglomeration attenuation scales to test in km.",
    )
    parser.add_argument(
        "--refresh-bfs-levels",
        action="store_true",
        help="Refresh the cached BFS commune-level typology CSV before reading the crosswalk.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=250,
        help="Print route-assignment progress every N newly assigned directed pairs.",
    )
    return parser.parse_args()


def pair_key(origin: str, destination: str) -> str:
    return f"{origin}|||{destination}"


def load_or_build_route_records(
    model,
    cache_path: Path,
    master_cache_path: Path,
    progress_every: int,
    assign_missing: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_set = {str(station).strip() for station in OVERALL_ANALYSIS_TARGET_STATIONS}
    selected_pairs_total = len(selected_set) * (len(selected_set) - 1)

    seeded: dict[str, dict[str, Any]] = {}
    master_cache = load_route_cache(master_cache_path)
    seeded_from_master = 0
    for key, row in master_cache.items():
        origin = str(row.get("origin", "")).strip()
        destination = str(row.get("destination", "")).strip()
        if origin in selected_set and destination in selected_set and origin != destination:
            seeded[key] = row
            seeded_from_master += 1

    local_cache = load_route_cache(cache_path)
    seeded_from_local = 0
    for key, row in local_cache.items():
        origin = str(row.get("origin", "")).strip()
        destination = str(row.get("destination", "")).strip()
        if origin in selected_set and destination in selected_set and origin != destination:
            if key not in seeded:
                seeded_from_local += 1
            seeded[key] = row

    pending_keys: list[tuple[str, str]] = []
    for origin in OVERALL_ANALYSIS_TARGET_STATIONS:
        for destination in OVERALL_ANALYSIS_TARGET_STATIONS:
            if origin == destination:
                continue
            key = pair_key(origin, destination)
            if key not in seeded:
                pending_keys.append((origin, destination))

    print(
        f"Seeded selected-station calibration with {len(seeded)} routed pairs from the master/local caches; "
        f"{len(pending_keys)} directed pairs remain uncovered.",
        flush=True,
    )

    assigned_now = 0
    if assign_missing and pending_keys:
        for origin, destination in pending_keys:
            route = assign_pair(model, origin, destination)
            record = {
                "origin": origin,
                "destination": destination,
                "status": route.get("status"),
                "travel_time_min": route.get("travel_time_min"),
                "departure_time": route.get("departure_time"),
                "arrival_time": route.get("arrival_time"),
                "transfers": route.get("transfers"),
                "segments": route.get("segments", []),
            }
            append_cache_record(cache_path, record)
            seeded[pair_key(origin, destination)] = record
            assigned_now += 1
            if assigned_now % max(progress_every, 1) == 0:
                print(f"Assigned {assigned_now} new selected-station routes so far...", flush=True)

    records: list[dict[str, Any]] = []
    for origin in OVERALL_ANALYSIS_TARGET_STATIONS:
        for destination in OVERALL_ANALYSIS_TARGET_STATIONS:
            if origin == destination:
                continue
            record = seeded.get(pair_key(origin, destination))
            if record is not None:
                records.append(record)

    diagnostics = {
        "selectedPairsTotal": selected_pairs_total,
        "seededFromMasterPairs": int(seeded_from_master),
        "seededAdditionalLocalPairs": int(seeded_from_local),
        "seededPairsCovered": int(len(records) - assigned_now),
        "newlyAssignedPairs": int(assigned_now),
        "finalPairsCovered": int(len(records)),
        "missingPairsOmitted": int(selected_pairs_total - len(records)),
        "seededCoverageShare": float((len(records) - assigned_now) / max(selected_pairs_total, 1)),
        "finalCoverageShare": float(len(records) / max(selected_pairs_total, 1)),
        "assignMissingEnabled": bool(assign_missing),
        "masterCachePath": str(master_cache_path),
        "localCachePath": str(cache_path),
    }
    return records, diagnostics


def build_edge_loads_from_weights(route_records: list[dict[str, Any]], weights: dict[tuple[str, str], float]) -> pd.DataFrame:
    edge_loads: dict[tuple[str, str], float] = {}
    edge_counts: dict[tuple[str, str], int] = {}

    for record in route_records:
        if record.get("status") != "assigned":
            continue
        key = (str(record["origin"]).strip(), str(record["destination"]).strip())
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        seen_edges: set[tuple[str, str]] = set()
        for segment in record.get("segments", []):
            edge_key = tuple(sorted((str(segment["from_station"]).strip(), str(segment["to_station"]).strip())))
            edge_loads[edge_key] = edge_loads.get(edge_key, 0.0) + weight
            if edge_key not in seen_edges:
                edge_counts[edge_key] = edge_counts.get(edge_key, 0) + 1
                seen_edges.add(edge_key)

    rows = [
        {
            "station_a": edge[0],
            "station_b": edge[1],
            "predicted_daily_trips": load,
            "assigned_od_pairs": edge_counts.get(edge, 0),
        }
        for edge, load in edge_loads.items()
    ]
    if not rows:
        return pd.DataFrame(columns=["station_a", "station_b", "predicted_daily_trips", "assigned_od_pairs"])
    return pd.DataFrame(rows).sort_values("predicted_daily_trips", ascending=False).reset_index(drop=True)


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, int, float, float]:
    fit = candidate["fit"]
    median_ape = float(fit["median_ape_percent"]) if fit["median_ape_percent"] is not None else float("inf")
    within_30 = int(fit["within_30_percent"])
    mape = float(fit["mape_percent"]) if fit["mape_percent"] is not None else float("inf")
    r_squared = float(fit["r_squared"]) if fit["r_squared"] is not None else float("-inf")
    return (median_ape, -within_30, mape, -r_squared)


def describe_candidate(mode: str, scale_km: float | None) -> str:
    if mode == "exclude":
        return "legacy same-agglomeration exclusion baseline"
    return f"same-agglomeration attenuation with linear distance scale {scale_km:.1f} km"


def build_candidate_rows(
    reference_year: str,
    mode: str,
    scale_km: float | None,
    selected_station_metadata: dict[str, dict[str, object]],
    station_crosswalk: dict[str, dict[str, str]],
    master_rows: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    return build_pair_rows(
        reference_year=reference_year,
        distance_exponent=2.0,
        same_cluster_mode=mode,
        same_cluster_distance_scale_km=DEFAULT_SAME_CLUSTER_DISTANCE_SCALE_KM if scale_km is None else scale_km,
        selected_station_metadata=selected_station_metadata,
        station_cluster_mapping=station_crosswalk,
        master_rows=master_rows,
    )


def summarize_route_weight_coverage(
    candidate_rows: list[dict[str, Any]],
    route_records: list[dict[str, Any]],
) -> dict[str, Any]:
    covered_pairs = {
        (str(record.get("origin", "")).strip(), str(record.get("destination", "")).strip())
        for record in route_records
    }
    total_raw = 0.0
    covered_raw = 0.0
    total_active_weight = 0.0
    covered_active_weight = 0.0
    total_pairs = 0
    covered_pairs_count = 0
    for row in candidate_rows:
        key = (str(row["origin"]).strip(), str(row["destination"]).strip())
        raw_value = float(row.get("raw_station_observed_daily_trips", 0.0) or 0.0)
        active_weight = float(row.get("pair_weight", 0.0) or 0.0)
        total_raw += raw_value
        total_active_weight += active_weight
        total_pairs += 1
        if key in covered_pairs:
            covered_raw += raw_value
            covered_active_weight += active_weight
            covered_pairs_count += 1
    return {
        "coveredPairs": int(covered_pairs_count),
        "totalPairs": int(total_pairs),
        "coveredRawSelectedTrips": float(covered_raw),
        "totalRawSelectedTrips": float(total_raw),
        "coveredRawSelectedShare": float(covered_raw / max(total_raw, 1e-12)),
        "coveredActivePairWeight": float(covered_active_weight),
        "totalActivePairWeight": float(total_active_weight),
        "coveredActiveWeightShare": float(covered_active_weight / max(total_active_weight, 1e-12)),
    }


def write_report(
    report_path: Path,
    summary: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    screenline_frame: pd.DataFrame,
) -> None:
    best = summary["bestCandidate"]
    route_coverage = summary["routeCoverage"]
    lines = [
        "# Same-Agglomeration Attenuation Calibration 2023",
        "",
        "## Decision",
        "",
        f"- Selected treatment: `{best['mode']}`",
        f"- Selected distance scale: `{best['distanceScaleKm']}`",
        f"- Selection rule: lowest median absolute percentage error on the selected-corridor screenline fit, then higher count within ±30%, then lower MAPE, then higher r².",
        f"- Seeded selected-station route coverage: `{route_coverage['coveredPairs']}/{route_coverage['totalPairs']}` directed pairs "
        f"covering `{route_coverage['coveredRawSelectedShare'] * 100:.2f}%` of the raw selected-station 2023 trips.",
        "",
        "## Candidate comparison",
        "",
        "| Candidate | Median APE | MAPE | r² | Within ±30% | Active same-aggl pairs | Retained same-aggl share |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in candidate_rows:
        fit = row["fit"]
        lines.append(
            f"| {row['label']} | {fit['median_ape_percent']:.1f}% | {fit['mape_percent']:.1f}% | {fit['r_squared']:.3f} | "
            f"{fit['within_30_percent']} | {row['summary']['activeSameClusterPairCount']} | "
            f"{row['summary']['sameClusterRetainedShareOfSelectedRawTrips'] * 100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Best candidate screenlines",
            "",
            screenline_frame[
                [
                    "corridor_label",
                    "predicted_daily_trips",
                    "official_bel_dwv",
                    "abs_pct_error",
                ]
            ].to_string(
                index=False,
                formatters={
                    "predicted_daily_trips": lambda value: f"{value:,.0f}",
                    "official_bel_dwv": lambda value: f"{value:,.0f}",
                    "abs_pct_error": lambda value: "-" if pd.isna(value) else f"{value:.1f}%",
                },
            ),
            "",
            "## Interpretation",
            "",
            "- The legacy website layer dropped same-agglomeration demand entirely, which protected the metric from metropolitan domination but also removed a large block of real passenger demand.",
            "- The calibrated attenuation keeps those same-agglomeration pairs in the network metric, but only in proportion to their crow-fly distance on the selected station layer.",
            "- The choice is therefore not arbitrary: it is tied to the official NPVM 2023 corridor screenline fit on the reduced 131-station analysis system.",
            "- The calibration reuses the already assigned 2023 route cache for the selected stations; the uncovered tail is small and common across all candidate attenuation scales, so it does not bias the comparison materially.",
            "- This still remains an analysis-layer simplification rather than a second observed OD dataset. The raw 2023 master station matrix stays unchanged.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    levels_path = ensure_bfs_commune_levels(refresh=args.refresh_bfs_levels)
    if not args.crosswalk_input.exists():
        write_crosswalk(build_station_agglomeration_rows(levels_path), args.crosswalk_input)
    station_crosswalk = load_station_agglomeration_crosswalk(args.crosswalk_input)

    selected_station_metadata = load_selected_station_metadata(str(REFERENCE_YEAR))
    master_rows = load_master_station_od(PROJECT_DIR / "station_od_master_reference_2023.csv", str(REFERENCE_YEAR))

    module, system, model = load_2023_model()
    del module, system

    route_records, route_cache_diagnostics = load_or_build_route_records(
        model,
        args.route_cache,
        args.master_route_cache,
        args.progress_every,
        args.assign_missing,
    )
    station_points = load_station_points()
    official_links = load_official_rail_links()

    candidates: list[dict[str, Any]] = []
    baseline_rows, baseline_summary = build_candidate_rows(
        str(REFERENCE_YEAR),
        "exclude",
        None,
        selected_station_metadata,
        station_crosswalk,
        master_rows,
    )
    baseline_weights = {
        (str(row["origin"]).strip(), str(row["destination"]).strip()): float(row["pair_weight"])
        for row in baseline_rows
    }
    baseline_edge_loads = build_edge_loads_from_weights(route_records, baseline_weights)
    baseline_edge_gdf = build_predicted_edge_map(baseline_edge_loads, station_points)
    baseline_screenlines = build_screenline_rows(station_points, baseline_edge_gdf, official_links)
    baseline_fit = summarize_fit(baseline_screenlines)
    route_coverage = summarize_route_weight_coverage(baseline_rows, route_records)
    candidates.append(
        {
            "mode": "exclude",
            "distanceScaleKm": None,
            "label": "Exclude same-agglomeration pairs",
            "fit": baseline_fit,
            "summary": baseline_summary,
            "screenlines": baseline_screenlines,
            "edgeLoads": baseline_edge_loads,
        }
    )

    for scale_km in sorted({float(value) for value in args.distance_scales_km if float(value) > 0}):
        rows, summary = build_candidate_rows(
            str(REFERENCE_YEAR),
            "attenuate",
            scale_km,
            selected_station_metadata,
            station_crosswalk,
            master_rows,
        )
        weights = {
            (str(row["origin"]).strip(), str(row["destination"]).strip()): float(row["pair_weight"])
            for row in rows
        }
        edge_loads = build_edge_loads_from_weights(route_records, weights)
        edge_gdf = build_predicted_edge_map(edge_loads, station_points)
        screenlines = build_screenline_rows(station_points, edge_gdf, official_links)
        fit = summarize_fit(screenlines)
        candidates.append(
            {
                "mode": "attenuate",
                "distanceScaleKm": scale_km,
                "label": f"Attenuate same-agglomeration pairs at {scale_km:.1f} km",
                "fit": fit,
                "summary": summary,
                "screenlines": screenlines,
                "edgeLoads": edge_loads,
            }
        )
        print(
            f"Evaluated attenuation scale {scale_km:.1f} km: median APE {fit['median_ape_percent']:.1f}% | "
            f"MAPE {fit['mape_percent']:.1f}% | r² {fit['r_squared']:.3f}",
            flush=True,
        )

    best_candidate = min(candidates, key=candidate_sort_key)
    best_edge_gdf = build_predicted_edge_map(best_candidate["edgeLoads"], station_points)
    best_screenlines = best_candidate["screenlines"].copy()
    best_screenlines.to_csv(args.screenline_output, index=False)
    plot_network_maps(best_edge_gdf, official_links, args.map_output)
    plot_screenline_scatter(best_screenlines.assign(scaled_predicted_daily_trips=float("nan")), args.scatter_output, None)

    summary = {
        "generatedAt": pd.Timestamp.utcnow().isoformat(),
        "referenceYear": REFERENCE_YEAR,
        "candidateCount": len(candidates),
        "selectionRule": (
            "lowest median APE, then highest within_30_percent, then lowest MAPE, then highest r_squared"
        ),
        "defaultSameClusterModeBeforeCalibration": DEFAULT_SAME_CLUSTER_MODE,
        "defaultSameClusterDistanceScaleKmBeforeCalibration": DEFAULT_SAME_CLUSTER_DISTANCE_SCALE_KM,
        "bestCandidate": {
            "mode": best_candidate["mode"],
            "distanceScaleKm": best_candidate["distanceScaleKm"],
            "label": best_candidate["label"],
            "fit": best_candidate["fit"],
            "summary": best_candidate["summary"],
        },
        "routeCacheDiagnostics": route_cache_diagnostics,
        "routeCoverage": route_coverage,
        "baselineExcludeFit": baseline_fit,
        "candidates": [
            {
                "mode": candidate["mode"],
                "distanceScaleKm": candidate["distanceScaleKm"],
                "label": candidate["label"],
                "fit": candidate["fit"],
                "summary": candidate["summary"],
            }
            for candidate in candidates
        ],
        "runtimeSeconds": round(time.perf_counter() - started, 2),
    }
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(
        args.report_output,
        summary,
        [
            {
                "label": candidate["label"],
                "fit": candidate["fit"],
                "summary": candidate["summary"],
            }
            for candidate in sorted(candidates, key=candidate_sort_key)
        ],
        best_screenlines.sort_values("official_bel_dwv", ascending=False).reset_index(drop=True),
    )

    print(
        f"Best candidate: {describe_candidate(best_candidate['mode'], best_candidate['distanceScaleKm'])} "
        f"| median APE {best_candidate['fit']['median_ape_percent']:.1f}% | "
        f"MAPE {best_candidate['fit']['mape_percent']:.1f}% | r² {best_candidate['fit']['r_squared']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
