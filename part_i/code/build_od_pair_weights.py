from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from build_station_agglomeration_crosswalk import (
    DEFAULT_OUTPUT_PATH as CROSSWALK_DEFAULT_PATH,
    build_station_agglomeration_rows,
    ensure_bfs_commune_levels,
    load_station_agglomeration_crosswalk,
    write_crosswalk,
)
from historical_od_weighting import (
    DEFAULT_DISTANCE_EXPONENT,
    DEFAULT_REFERENCE_YEAR,
    DEFAULT_STATION_POPULATION,
    haversine_km,
)
from historical_station_sets import OVERALL_ANALYSIS_TARGET_STATIONS


PROJECT_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"
MASTER_OD_INPUT_PATH = PROJECT_DIR / "station_od_master_reference_2023.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "historical_od_pair_weights_reference_2023.csv"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_DIR / "historical_od_pair_weights_reference_2023_summary.json"
DEFAULT_VALIDATION_OUTPUT_PATH = PROJECT_DIR / "historical_od_pair_weights_reference_2023_validation.md"

DEFAULT_SAME_CLUSTER_MODE = "attenuate"
DEFAULT_SAME_CLUSTER_DISTANCE_SCALE_KM = 60.0
WEIGHT_MODEL_NAME = "npvm_2023_observed_hybrid_selected_station_projection_with_same_agglomeration_distance_attenuation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical station-pair weighting table used by the website's overall historical analysis. "
            "The new canonical table is derived from the raw 2023 observed/hybrid master station matrix, then "
            "projected onto the 131-station website subset in an agglomeration-aware way: inter-agglomeration "
            "cluster flows are redistributed to the visible stations with observed station shares, while same-"
            "agglomeration selected-station pairs can either be explicitly excluded or retained with a calibrated "
            "distance attenuation."
        )
    )
    parser.add_argument(
        "--reference-year",
        default=DEFAULT_REFERENCE_YEAR,
        help="Reference year written into the generated pair table. The current observed/hybrid pipeline is anchored to 2023.",
    )
    parser.add_argument(
        "--distance-exponent",
        type=float,
        default=DEFAULT_DISTANCE_EXPONENT,
        help="Distance exponent used only for the documented gravity comparison column and the on-the-fly fallback.",
    )
    parser.add_argument(
        "--same-cluster-mode",
        choices=["exclude", "attenuate"],
        default=DEFAULT_SAME_CLUSTER_MODE,
        help=(
            "How same-agglomeration directed pairs should be treated on the selected 131-station analysis layer. "
            "'exclude' keeps the legacy zero-weight exclusion. 'attenuate' keeps the raw selected-station observed "
            "same-agglomeration demand but downweights it by distance."
        ),
    )
    parser.add_argument(
        "--same-cluster-distance-scale-km",
        type=float,
        default=DEFAULT_SAME_CLUSTER_DISTANCE_SCALE_KM,
        help=(
            "Distance scale used when --same-cluster-mode=attenuate. The attenuation factor is min(1, distance_km / scale_km), "
            "so same-agglomeration pairs shorter than this scale are proportionally downweighted while longer ones keep full weight."
        ),
    )
    parser.add_argument(
        "--master-od-input",
        type=Path,
        default=MASTER_OD_INPUT_PATH,
        help="Path to the raw master 2023 station-OD matrix CSV.",
    )
    parser.add_argument(
        "--crosswalk-input",
        type=Path,
        default=CROSSWALK_DEFAULT_PATH,
        help="Path to the local station-to-agglomeration crosswalk CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the canonical website pair-weight CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="Path to the machine-readable summary JSON.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT_PATH,
        help="Path to the human-readable validation markdown.",
    )
    parser.add_argument(
        "--refresh-bfs-levels",
        action="store_true",
        help="Refresh the cached BFS commune-level typology CSV before reading the crosswalk.",
    )
    return parser.parse_args()


def load_station_frame() -> pd.DataFrame:
    frame = pd.read_csv(FILTERED_STATIONS_PATH, dtype=str).fillna("")
    frame["StationName"] = frame["StationName"].astype(str).str.strip()
    frame = frame[frame["StationName"].isin(OVERALL_ANALYSIS_TARGET_STATIONS)].copy()
    if frame.empty:
        raise ValueError("No overall-analysis stations were found in FilteredStations.csv.")
    return frame


def station_population(row: pd.Series, reference_year: str) -> int:
    population_column = f"{reference_year}_pop"
    raw_value = str(row.get(population_column, "")).strip()
    if not raw_value:
        return DEFAULT_STATION_POPULATION
    try:
        numeric = int(float(raw_value))
    except ValueError:
        return DEFAULT_STATION_POPULATION
    return numeric if numeric > 0 else DEFAULT_STATION_POPULATION


def station_service_count(row: pd.Series, reference_year: str) -> float:
    raw_value = str(row.get(reference_year, "")).strip()
    if not raw_value:
        return 0.0
    try:
        return max(float(raw_value), 0.0)
    except ValueError:
        return 0.0


def load_master_station_od(path: Path, reference_year: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Build the raw master 2023 station anchor with build_npvm_station_od_2023.py first."
        )

    frame = pd.read_csv(path, dtype=str).fillna("")
    frame["origin"] = frame["origin"].astype(str).str.strip()
    frame["destination"] = frame["destination"].astype(str).str.strip()
    frame["reference_year"] = frame["reference_year"].astype(str).str.strip()
    frame["year"] = frame["year"].astype(str).str.strip()
    frame = frame[
        (frame["origin"] != "")
        & (frame["destination"] != "")
        & (frame["reference_year"] == reference_year)
    ].copy()
    if frame.empty:
        raise ValueError(f"No master station OD rows found for reference year {reference_year}.")

    frame["observed_daily_trips_numeric"] = (
        pd.to_numeric(frame["observed_daily_trips"], errors="coerce").fillna(0.0)
    )
    frame["pair_weight_numeric"] = pd.to_numeric(frame["pair_weight"], errors="coerce").fillna(0.0)
    return frame


def load_selected_station_metadata(reference_year: str) -> dict[str, dict[str, object]]:
    frame = load_station_frame()
    metadata: dict[str, dict[str, object]] = {}
    for _, row in frame.iterrows():
        station = str(row["StationName"]).strip()
        metadata[station] = {
            "country": str(row.get("Country", "")).strip(),
            "canton": str(row.get("Region", "")).strip(),
            "lat": float(str(row["lat"]).strip()),
            "lon": float(str(row["lon"]).strip()),
            "population": station_population(row, reference_year),
            "service_2023": station_service_count(row, reference_year),
        }
    missing = [station for station in OVERALL_ANALYSIS_TARGET_STATIONS if station not in metadata]
    if missing:
        raise ValueError(f"Stations missing from FilteredStations.csv: {', '.join(missing)}")
    return metadata


def build_selected_shares(
    selected_station_metadata: dict[str, dict[str, object]],
    station_cluster: dict[str, str],
    master_rows: pd.DataFrame,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, str],
    dict[str, str],
    dict[str, object],
]:
    intercluster_origin_totals: Counter[str] = Counter()
    intercluster_destination_totals: Counter[str] = Counter()
    selected_origin_totals_by_cluster: defaultdict[str, float] = defaultdict(float)
    selected_destination_totals_by_cluster: defaultdict[str, float] = defaultdict(float)
    selected_service_by_cluster: defaultdict[str, float] = defaultdict(float)
    selected_stations_by_cluster: defaultdict[str, list[str]] = defaultdict(list)

    for station, meta in selected_station_metadata.items():
        cluster_id = station_cluster[station]
        selected_service_by_cluster[cluster_id] += float(meta.get("service_2023") or 0.0)
        selected_stations_by_cluster[cluster_id].append(station)

    for _, row in master_rows.iterrows():
        origin = str(row["origin"]).strip()
        destination = str(row["destination"]).strip()
        origin_cluster = station_cluster.get(origin, f"STATION:{origin}")
        destination_cluster = station_cluster.get(destination, f"STATION:{destination}")
        if origin_cluster == destination_cluster:
            continue
        trips = float(row["observed_daily_trips_numeric"])
        intercluster_origin_totals[origin] += trips
        intercluster_destination_totals[destination] += trips

    for station in selected_station_metadata:
        cluster_id = station_cluster[station]
        selected_origin_totals_by_cluster[cluster_id] += intercluster_origin_totals[station]
        selected_destination_totals_by_cluster[cluster_id] += intercluster_destination_totals[station]

    origin_shares: dict[str, float] = {}
    destination_shares: dict[str, float] = {}
    origin_share_method: dict[str, str] = {}
    destination_share_method: dict[str, str] = {}

    for cluster_id, stations in selected_stations_by_cluster.items():
        origin_denominator = selected_origin_totals_by_cluster[cluster_id]
        destination_denominator = selected_destination_totals_by_cluster[cluster_id]
        service_denominator = selected_service_by_cluster[cluster_id]

        for station in stations:
            service_value = float(selected_station_metadata[station].get("service_2023") or 0.0)
            if origin_denominator > 0:
                origin_shares[station] = intercluster_origin_totals[station] / origin_denominator
                origin_share_method[station] = "observed_intercluster_origin_share_on_selected_subset"
            elif service_denominator > 0:
                origin_shares[station] = service_value / service_denominator
                origin_share_method[station] = "service_share_fallback_on_selected_subset"
            else:
                origin_shares[station] = 1.0 / max(len(stations), 1)
                origin_share_method[station] = "equal_share_fallback_on_selected_subset"

            if destination_denominator > 0:
                destination_shares[station] = intercluster_destination_totals[station] / destination_denominator
                destination_share_method[station] = "observed_intercluster_destination_share_on_selected_subset"
            elif service_denominator > 0:
                destination_shares[station] = service_value / service_denominator
                destination_share_method[station] = "service_share_fallback_on_selected_subset"
            else:
                destination_shares[station] = 1.0 / max(len(stations), 1)
                destination_share_method[station] = "equal_share_fallback_on_selected_subset"

    diagnostics = {
        "selectedStationsByCluster": {
            cluster: sorted(stations) for cluster, stations in selected_stations_by_cluster.items()
        },
        "originShareMethodCounts": dict(Counter(origin_share_method.values())),
        "destinationShareMethodCounts": dict(Counter(destination_share_method.values())),
        "selectedOriginTotalsByCluster": {
            cluster: value for cluster, value in selected_origin_totals_by_cluster.items()
        },
        "selectedDestinationTotalsByCluster": {
            cluster: value for cluster, value in selected_destination_totals_by_cluster.items()
        },
    }
    return origin_shares, destination_shares, origin_share_method, destination_share_method, diagnostics


def compute_cluster_pair_flows(
    master_rows: pd.DataFrame,
    station_cluster: dict[str, str],
) -> tuple[dict[tuple[str, str], float], dict[str, float]]:
    cluster_pair_flows: defaultdict[tuple[str, str], float] = defaultdict(float)
    same_cluster_flow_by_cluster: defaultdict[str, float] = defaultdict(float)

    for _, row in master_rows.iterrows():
        origin_cluster = station_cluster.get(str(row["origin"]).strip(), f"STATION:{str(row['origin']).strip()}")
        destination_cluster = station_cluster.get(
            str(row["destination"]).strip(),
            f"STATION:{str(row['destination']).strip()}",
        )
        trips = float(row["observed_daily_trips_numeric"])
        if origin_cluster == destination_cluster:
            same_cluster_flow_by_cluster[origin_cluster] += trips
        else:
            cluster_pair_flows[(origin_cluster, destination_cluster)] += trips

    return dict(cluster_pair_flows), dict(same_cluster_flow_by_cluster)


def same_cluster_distance_attenuation(distance_km: float, scale_km: float) -> float:
    numeric_distance = max(float(distance_km), 0.0)
    numeric_scale = max(float(scale_km), 1e-9)
    return min(1.0, numeric_distance / numeric_scale)


def build_pair_rows(
    reference_year: str,
    distance_exponent: float,
    same_cluster_mode: str,
    same_cluster_distance_scale_km: float,
    selected_station_metadata: dict[str, dict[str, object]],
    station_cluster_mapping: dict[str, dict[str, str]],
    master_rows: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    station_cluster = {
        station: station_cluster_mapping.get(station, {}).get("analysis_cluster_id", f"STATION:{station}")
        for station in selected_station_metadata
    }
    cluster_pair_flows, same_cluster_flow_by_cluster = compute_cluster_pair_flows(master_rows, station_cluster)
    (
        origin_shares,
        destination_shares,
        origin_share_method,
        destination_share_method,
        share_diagnostics,
    ) = build_selected_shares(
        selected_station_metadata,
        station_cluster,
        master_rows,
    )

    raw_station_pair = {
        (str(row["origin"]).strip(), str(row["destination"]).strip()): float(row["observed_daily_trips_numeric"])
        for _, row in master_rows.iterrows()
        if str(row["origin"]).strip() in selected_station_metadata
        and str(row["destination"]).strip() in selected_station_metadata
    }

    rows: list[dict[str, object]] = []
    active_weight_sum = 0.0
    excluded_pair_count = 0
    active_pair_count = 0
    projected_from_cluster_pair_count = 0
    raw_selected_intercluster_sum = 0.0
    raw_selected_samecluster_sum = 0.0
    attenuated_samecluster_sum = 0.0
    active_samecluster_pair_count = 0
    cluster_usage_counter: Counter[str] = Counter()
    cluster_scope_counter: Counter[str] = Counter()
    active_pairs_touching_foreign_station = 0
    excluded_pairs_touching_foreign_station = 0
    active_foreign_weight_sum = 0.0

    for origin in OVERALL_ANALYSIS_TARGET_STATIONS:
        origin_meta = selected_station_metadata[origin]
        origin_cluster = station_cluster[origin]
        origin_cluster_info = station_cluster_mapping.get(origin, {})

        for destination in OVERALL_ANALYSIS_TARGET_STATIONS:
            if destination == origin:
                continue

            destination_meta = selected_station_metadata[destination]
            destination_cluster = station_cluster[destination]
            destination_cluster_info = station_cluster_mapping.get(destination, {})

            distance_km = max(
                haversine_km(
                    float(origin_meta["lat"]),
                    float(origin_meta["lon"]),
                    float(destination_meta["lat"]),
                    float(destination_meta["lon"]),
                ),
                1.0,
            )
            gravity_weight = (
                int(origin_meta["population"]) * int(destination_meta["population"])
            ) / (distance_km ** distance_exponent)

            raw_selected_station_trips = raw_station_pair.get((origin, destination), 0.0)
            if origin_cluster == destination_cluster:
                cluster_pair_trips = same_cluster_flow_by_cluster.get(origin_cluster, 0.0)
                same_cluster_attenuation = ""
                raw_selected_samecluster_sum += raw_selected_station_trips
                if same_cluster_mode == "exclude":
                    pair_weight = 0.0
                    exclude_from_analysis = True
                    analysis_status = "excluded_same_agglomeration"
                    notes = (
                        "This directed pair stays inside the same official BFS AGGL2020 agglomeration (or the same singleton cluster) "
                        "and is therefore excluded from the network-wide overall-analysis weighting layer. The raw master 2023 station "
                        "matrix still retains its observed station-level demand separately."
                    )
                    excluded_pair_count += 1
                    if (
                        str(origin_meta["country"]).strip() != "CH"
                        or str(destination_meta["country"]).strip() != "CH"
                    ):
                        excluded_pairs_touching_foreign_station += 1
                else:
                    attenuation_factor = same_cluster_distance_attenuation(
                        distance_km,
                        same_cluster_distance_scale_km,
                    )
                    same_cluster_attenuation = round(attenuation_factor, 12)
                    pair_weight = raw_selected_station_trips * attenuation_factor
                    exclude_from_analysis = False
                    analysis_status = (
                        "active_same_agglomeration_distance_attenuated"
                        if pair_weight > 0
                        else "same_agglomeration_zero_selected_pair"
                    )
                    notes = (
                        "This directed pair stays inside the same official BFS AGGL2020 agglomeration (or the same singleton cluster), "
                        "but it is retained in the network-wide weighting layer with a calibrated linear distance attenuation applied to the "
                        "raw selected-station observed 2023 OD value. The attenuation factor is min(1, crow-fly distance / "
                        f"{same_cluster_distance_scale_km:.1f} km). This keeps nearby metropolitan demand in the metric without letting the "
                        "shortest local pairs dominate the network-wide averages."
                    )
                    attenuated_samecluster_sum += pair_weight
                    if pair_weight > 0:
                        active_pair_count += 1
                        active_samecluster_pair_count += 1
                        active_weight_sum += pair_weight
                        if (
                            str(origin_meta["country"]).strip() != "CH"
                            or str(destination_meta["country"]).strip() != "CH"
                        ):
                            active_pairs_touching_foreign_station += 1
                            active_foreign_weight_sum += pair_weight
            else:
                cluster_pair_trips = cluster_pair_flows.get((origin_cluster, destination_cluster), 0.0)
                same_cluster_attenuation = ""
                pair_weight = cluster_pair_trips * origin_shares[origin] * destination_shares[destination]
                exclude_from_analysis = False
                analysis_status = "active_inter_agglomeration_projection"
                notes = (
                    "Inter-agglomeration pair projected from the raw 2023 master station matrix. The cluster-level observed flow is "
                    "the sum of all master OD rows between the two official BFS AGGL2020 analysis clusters (or singleton clusters). "
                    "That cluster flow is redistributed to the visible website stations with the origin station's observed inter-cluster "
                    "production share and the destination station's observed inter-cluster attraction share, both normalized on the "
                    "selected 131-station website subset. When a selected cluster has zero observed inter-cluster totals on one side, "
                    "the redistribution falls back to 2023 service shares within that selected subset."
                )
                raw_selected_intercluster_sum += raw_selected_station_trips
                active_weight_sum += pair_weight
                active_pair_count += 1
                projected_from_cluster_pair_count += 1
                cluster_usage_counter[origin_cluster] += 1
                cluster_usage_counter[destination_cluster] += 1
                cluster_scope_counter[str(origin_cluster_info.get("cluster_scope", "")).strip() or "unspecified"] += 1
                cluster_scope_counter[str(destination_cluster_info.get("cluster_scope", "")).strip() or "unspecified"] += 1
                if (
                    str(origin_meta["country"]).strip() != "CH"
                    or str(destination_meta["country"]).strip() != "CH"
                ):
                    active_pairs_touching_foreign_station += 1
                    active_foreign_weight_sum += pair_weight

            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "year": reference_year,
                    "reference_year": reference_year,
                    "origin_country": str(origin_meta["country"]),
                    "destination_country": str(destination_meta["country"]),
                    "origin_canton": str(origin_meta["canton"]),
                    "destination_canton": str(destination_meta["canton"]),
                    "origin_population": int(origin_meta["population"]),
                    "destination_population": int(destination_meta["population"]),
                    "distance_km": round(distance_km, 6),
                    "pair_weight": round(pair_weight, 12),
                    "observed_daily_trips": round(pair_weight, 12),
                    "raw_station_observed_daily_trips": round(raw_selected_station_trips, 12),
                    "cluster_pair_observed_daily_trips": round(cluster_pair_trips, 12),
                    "gravity_weight": round(gravity_weight, 12),
                    "distance_exponent": distance_exponent,
                    "weight_model": WEIGHT_MODEL_NAME,
                    "analysis_status": analysis_status,
                    "exclude_from_analysis": 1 if exclude_from_analysis else 0,
                    "origin_cluster_id": origin_cluster,
                    "destination_cluster_id": destination_cluster,
                    "origin_cluster_scope": origin_cluster_info.get("cluster_scope", ""),
                    "destination_cluster_scope": destination_cluster_info.get("cluster_scope", ""),
                    "origin_aggl2020_code": origin_cluster_info.get("aggl2020_code", ""),
                    "destination_aggl2020_code": destination_cluster_info.get("aggl2020_code", ""),
                    "origin_outbound_share": round(origin_shares[origin], 12),
                    "destination_inbound_share": round(destination_shares[destination], 12),
                    "origin_share_method": origin_share_method.get(origin, ""),
                    "destination_share_method": destination_share_method.get(destination, ""),
                    "same_cluster_attenuation_factor": same_cluster_attenuation,
                    "source": (
                        "Derived from the raw 2023 observed/hybrid master station matrix, then projected to the 131-station website "
                        "subset with official BFS AGGL2020 agglomeration clustering and selected-station inter-cluster shares"
                    ),
                    "notes": notes,
                }
            )

    summary = {
        "referenceYear": reference_year,
        "weightModel": WEIGHT_MODEL_NAME,
        "sameClusterMode": same_cluster_mode,
        "sameClusterDistanceScaleKm": same_cluster_distance_scale_km,
        "masterStationCount": int(len(set(master_rows["origin"]).union(set(master_rows["destination"])))),
        "analysisStationCount": len(OVERALL_ANALYSIS_TARGET_STATIONS),
        "analysisClusterCount": len({station_cluster[station] for station in OVERALL_ANALYSIS_TARGET_STATIONS}),
        "activePairCount": active_pair_count,
        "excludedPairCount": excluded_pair_count,
        "activeSameClusterPairCount": active_samecluster_pair_count,
        "projectedWeightSum": active_weight_sum,
        "rawSelectedInterclusterTrips": raw_selected_intercluster_sum,
        "rawSelectedSameClusterTrips": raw_selected_samecluster_sum,
        "attenuatedSameClusterTrips": attenuated_samecluster_sum,
        "sameClusterRawShareOfSelectedTrips": (
            raw_selected_samecluster_sum / max(raw_selected_samecluster_sum + raw_selected_intercluster_sum, 1e-12)
        ),
        "sameClusterExcludedShareOfSelectedRawTrips": (
            raw_selected_samecluster_sum / max(raw_selected_samecluster_sum + raw_selected_intercluster_sum, 1e-12)
        ),
        "sameClusterRetainedShareOfSelectedRawTrips": (
            attenuated_samecluster_sum / max(raw_selected_samecluster_sum + raw_selected_intercluster_sum, 1e-12)
        ),
        "clusterPairCount": len(cluster_pair_flows),
        "sameClusterClusterCount": len(same_cluster_flow_by_cluster),
        "shareDiagnostics": share_diagnostics,
        "clusterUsageCounter": dict(cluster_usage_counter),
        "clusterScopeUsageCounter": dict(cluster_scope_counter),
        "activePairsTouchingForeignStation": active_pairs_touching_foreign_station,
        "excludedPairsTouchingForeignStation": excluded_pairs_touching_foreign_station,
        "activeForeignWeightSum": active_foreign_weight_sum,
        "activeForeignWeightShare": active_foreign_weight_sum / max(active_weight_sum, 1e-12),
    }
    return rows, summary


def write_pair_table(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "origin",
        "destination",
        "year",
        "reference_year",
        "origin_country",
        "destination_country",
        "origin_canton",
        "destination_canton",
        "origin_population",
        "destination_population",
        "distance_km",
        "pair_weight",
        "observed_daily_trips",
        "raw_station_observed_daily_trips",
        "cluster_pair_observed_daily_trips",
        "gravity_weight",
        "distance_exponent",
        "weight_model",
        "analysis_status",
        "exclude_from_analysis",
        "origin_cluster_id",
        "destination_cluster_id",
        "origin_cluster_scope",
        "destination_cluster_scope",
        "origin_aggl2020_code",
        "destination_aggl2020_code",
        "origin_outbound_share",
        "destination_inbound_share",
        "origin_share_method",
        "destination_share_method",
        "same_cluster_attenuation_factor",
        "source",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_validation_markdown(summary: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    share_diagnostics = summary.get("shareDiagnostics", {})
    lines = [
        "# Website OD Pair Weight Validation 2023",
        "",
        "## Scope",
        "",
        f"- Reference year: `{summary['referenceYear']}`",
        f"- Weight model: `{summary['weightModel']}`",
        f"- Raw master station system: `{summary['masterStationCount']}` stations",
        f"- Website analysis station system: `{summary['analysisStationCount']}` stations",
        f"- Distinct analysis clusters represented on the website subset: `{summary['analysisClusterCount']}`",
        "",
        "## Pair Table Diagnostics",
        "",
        f"- Same-agglomeration treatment: `{summary['sameClusterMode']}`",
        f"- Same-agglomeration attenuation distance scale: `{summary['sameClusterDistanceScaleKm']:.1f} km`",
        f"- Active directed pairs in the website metric: `{summary['activePairCount']}`",
        f"- Explicit zero/excluded directed pairs: `{summary['excludedPairCount']}`",
        f"- Active same-agglomeration directed pairs retained in the metric: `{summary['activeSameClusterPairCount']}`",
        f"- Sum of projected active pair weights: `{summary['projectedWeightSum']:.3f}`",
        f"- Raw selected-station inter-agglomeration trips before projection: `{summary['rawSelectedInterclusterTrips']:.3f}`",
        f"- Raw selected-station same-agglomeration trips before attenuation/exclusion: `{summary['rawSelectedSameClusterTrips']:.3f}`",
        f"- Attenuated same-agglomeration trips retained in the overall metric: `{summary['attenuatedSameClusterTrips']:.3f}`",
        f"- Same-agglomeration share of the raw selected-station trips before attenuation: `{summary['sameClusterRawShareOfSelectedTrips'] * 100:.2f}%`",
        f"- Retained same-agglomeration share of the total selected raw trips after attenuation: `{summary['sameClusterRetainedShareOfSelectedRawTrips'] * 100:.2f}%`",
        f"- Distinct inter-cluster directions in the master anchor: `{summary['clusterPairCount']}`",
        f"- Distinct clusters with excluded same-cluster demand: `{summary['sameClusterClusterCount']}`",
        f"- Active directed pairs touching at least one foreign station: `{summary['activePairsTouchingForeignStation']}`",
        f"- Excluded directed pairs touching at least one foreign station: `{summary['excludedPairsTouchingForeignStation']}`",
        f"- Active weight sum touching at least one foreign station: `{summary['activeForeignWeightSum']:.3f}`",
        f"- Foreign-station share of the active website weight sum: `{summary['activeForeignWeightShare'] * 100:.2f}%`",
        "",
        "## Redistribution Diagnostics",
        "",
        f"- Origin-share methods used: `{share_diagnostics.get('originShareMethodCounts', {})}`",
        f"- Destination-share methods used: `{share_diagnostics.get('destinationShareMethodCounts', {})}`",
        f"- Cluster scopes touched by active pairs: `{summary.get('clusterScopeUsageCounter', {})}`",
        "",
        "## Interpretation",
        "",
        "- The website's overall historical analysis no longer reads the old gravity-proxy table as its primary source.",
        "- Instead, it now reads an analysis-ready 2023 table derived from the raw observed/hybrid master station anchor.",
        (
            "- Same-agglomeration pairs are written explicitly with zero weight and an exclusion flag so they do not fall through to gravity fallback by accident."
            if summary["sameClusterMode"] == "exclude"
            else "- Same-agglomeration pairs are retained on the selected-station layer, but only with a calibrated linear distance attenuation applied to their raw observed selected-station demand."
        ),
        "- Inter-agglomeration cluster flows come from the full master station matrix, while their redistribution to the visible website nodes is normalized on the selected 131-station subset.",
        "- This means the website layer is now a projection of the observed/hybrid master demand onto the visible analysis nodes, not a synthetic gravity estimate.",
        "- The gravity formula remains available only as a safety-net fallback for genuinely missing rows outside this canonical table.",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    levels_path = ensure_bfs_commune_levels(refresh=args.refresh_bfs_levels)
    if not args.crosswalk_input.exists():
        write_crosswalk(build_station_agglomeration_rows(levels_path), args.crosswalk_input)
    station_crosswalk = load_station_agglomeration_crosswalk(args.crosswalk_input)
    selected_station_metadata = load_selected_station_metadata(args.reference_year)
    master_rows = load_master_station_od(args.master_od_input, args.reference_year)

    station_cluster = {
        station: station_crosswalk.get(station, {})
        for station in set(master_rows["origin"]).union(set(master_rows["destination"])).union(set(selected_station_metadata))
    }

    raw_rows, summary = build_pair_rows(
        args.reference_year,
        args.distance_exponent,
        args.same_cluster_mode,
        args.same_cluster_distance_scale_km,
        selected_station_metadata,
        station_cluster,
        master_rows,
    )

    write_pair_table(raw_rows, args.output)
    write_summary_json(summary, args.summary_output)
    write_validation_markdown(summary, args.validation_output)

    print(
        f"Wrote {len(raw_rows)} directed website station pairs to {args.output} "
        f"({summary['activePairCount']} active, {summary['excludedPairCount']} explicitly excluded)."
    )


if __name__ == "__main__":
    main()
