from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import numpy as np
import pandas as pd

from historical_od_weighting import DEFAULT_DISTANCE_EXPONENT, haversine_km


PROJECT_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"
SWISS_ZONE_ZIP_PATH = PROJECT_DIR / "npvm_2023_src/1_Verkehrszonen_Schweiz_NPVM_2023.zip"
SWISS_CENTROIDS_PATH = PROJECT_DIR / "npvm_2023_src/1_Verkehrszonen_Schweiz_Zentroide_NPVM_2023.gpkg"
FOREIGN_ZONE_ZIP_PATH = PROJECT_DIR / "npvm_2023_src/1_Verkehrszonen_Ausland_NPVM_2023.zip"
MATRIX_ZIP_PATH = PROJECT_DIR / "npvm_2023_src/3_Wegematrix_OEV_NPVM_2023.zip"
SWISS_ZONE_EXTRACT_DIR = PROJECT_DIR / "npvm_2023_src/verkehrszonen_schweiz"
FOREIGN_ZONE_EXTRACT_DIR = PROJECT_DIR / "npvm_2023_src/verkehrszonen_ausland"

SWISS_ZONE_LAYER = "Verkehrszonen_Schweiz_NPVM_2023"
SWISS_CENTROID_LAYER = "Verkehrszonen_Schweiz_Zentroide"
FOREIGN_ZONE_LAYER = "verkehrszonen_ausland_npvm_2023"
SWISS_ZONE_GPKG_NAME = "1_Verkehrszonen_Schweiz_NPVM_2023.gpkg"
FOREIGN_ZONE_GPKG_NAME = "Verkehrszonen_Ausland_NPVM_2023.gpkg"
MATRIX_MEMBER_NAME = "Wege_OEV_NPVM_2023.mtx"

REFERENCE_YEAR = "2023"
TARGET_STATION_SET_NAME = "modeled_station_system_250"
DEFAULT_ALLOCATION_OUTPUT_PATH = PROJECT_DIR / "npvm_zone_station_allocation_2023_master.csv"
DEFAULT_OD_OUTPUT_PATH = PROJECT_DIR / "station_od_master_reference_2023.csv"
DEFAULT_SUMMARY_OUTPUT_PATH = PROJECT_DIR / "station_od_master_reference_2023_summary.json"
DEFAULT_VALIDATION_OUTPUT_PATH = PROJECT_DIR / "station_od_master_validation_2023.md"


@dataclass(frozen=True)
class AllocationParameters:
    min_candidates: int
    nearest_pool: int
    in_zone_bonus: float
    access_offset_km: float
    distance_exponent: float
    service_exponent: float

    @property
    def allocation_model_name(self) -> str:
        distance_label = f"{self.distance_exponent:.2f}".rstrip("0").rstrip(".")
        service_label = f"{self.service_exponent:.2f}".rstrip("0").rstrip(".")
        access_label = f"{self.access_offset_km:.1f}".rstrip("0").rstrip(".")
        bonus_label = f"{self.in_zone_bonus:.2f}".rstrip("0").rstrip(".")
        return (
            "hybrid_zone_to_station_distance_service_2023"
            f"_k{self.min_candidates}"
            f"_pool{self.nearest_pool}"
            f"_d{distance_label}"
            f"_s{service_label}"
            f"_a{access_label}"
            f"_z{bonus_label}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a defensible 2023 station-OD anchor from the NPVM 2023 PT trip matrix by aggregating "
            "traffic zones to the full modeled station system through a documented hybrid zone-to-station "
            "allocation model. This script now builds the raw master 2023 anchor rather than the narrower "
            "website-ready weighting subset."
        )
    )
    parser.add_argument(
        "--allocation-output",
        type=Path,
        default=DEFAULT_ALLOCATION_OUTPUT_PATH,
        help="CSV path for the zone-to-station allocation table.",
    )
    parser.add_argument(
        "--od-output",
        type=Path,
        default=DEFAULT_OD_OUTPUT_PATH,
        help="CSV path for the aggregated directed station-OD matrix.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT_PATH,
        help="JSON path for the machine-readable validation summary.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_OUTPUT_PATH,
        help="Markdown path for the human-readable validation report.",
    )
    parser.add_argument(
        "--min-candidates",
        type=int,
        default=4,
        help="Minimum number of station candidates retained for each zone.",
    )
    parser.add_argument(
        "--nearest-pool",
        type=int,
        default=6,
        help="How many nearest stations are considered before the minimum-candidate rule is applied.",
    )
    parser.add_argument(
        "--in-zone-bonus",
        type=float,
        default=1.75,
        help="Multiplier applied to stations whose point lies inside the zone polygon.",
    )
    parser.add_argument(
        "--access-offset-km",
        type=float,
        default=3.0,
        help="Distance offset in kilometres that softens near-zero access distances.",
    )
    parser.add_argument(
        "--distance-exponent",
        type=float,
        default=2.0,
        help="Distance exponent used in the zone-to-station allocation formula.",
    )
    parser.add_argument(
        "--service-exponent",
        type=float,
        default=0.5,
        help="Exponent applied to the 2023 station service counts in the allocation formula.",
    )
    return parser.parse_args()


def _safe_positive_float(value: object, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        numeric = float(text)
    except (TypeError, ValueError):
        return default
    return numeric if numeric > 0 else default


def _read_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str).fillna("")


def load_target_stations() -> pd.DataFrame:
    frame = _read_csv_frame(FILTERED_STATIONS_PATH)
    frame["StationName"] = frame["StationName"].astype(str).str.strip()
    frame = frame[frame["StationName"] != ""].copy()

    frame["lat"] = frame["lat"].astype(float)
    frame["lon"] = frame["lon"].astype(float)
    frame["service_2023"] = frame["2023"].map(lambda value: _safe_positive_float(value, 0.0))
    frame["service_score"] = np.power(frame["service_2023"].clip(lower=1.0), 0.5)
    frame["population_2023"] = frame["2023_pop"].map(lambda value: int(round(_safe_positive_float(value, 30000.0))))
    frame = frame.sort_values("StationName").reset_index(drop=True)
    return frame


def ensure_extracted(zip_path: Path, extract_dir: Path, expected_file_name: str) -> Path:
    extract_dir.mkdir(parents=True, exist_ok=True)
    expected_path = extract_dir / expected_file_name
    if expected_path.exists():
        return expected_path
    with ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    if not expected_path.exists():
        raise FileNotFoundError(f"Expected {expected_file_name} inside {zip_path.name}")
    return expected_path


def load_zone_geometries() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    swiss_path = ensure_extracted(SWISS_ZONE_ZIP_PATH, SWISS_ZONE_EXTRACT_DIR, SWISS_ZONE_GPKG_NAME)
    foreign_path = ensure_extracted(FOREIGN_ZONE_ZIP_PATH, FOREIGN_ZONE_EXTRACT_DIR, FOREIGN_ZONE_GPKG_NAME)

    swiss = gpd.read_file(swiss_path, layer=SWISS_ZONE_LAYER)[["No", "N_Gem", "N_KT", "geometry"]].copy()
    swiss["zone_id"] = swiss["No"].round().astype(int)
    swiss["zone_name"] = swiss["N_Gem"].astype(str)
    swiss["country_code"] = "CH"
    swiss = swiss[["zone_id", "zone_name", "country_code", "geometry"]]

    swiss_centroids = gpd.read_file(SWISS_CENTROIDS_PATH, layer=SWISS_CENTROID_LAYER)[["No", "geometry"]].copy()
    swiss_centroids["zone_id"] = swiss_centroids["No"].round().astype(int)
    swiss_centroids = swiss_centroids[["zone_id", "geometry"]].rename(columns={"geometry": "centroid_geometry"})
    swiss_centroids = swiss_centroids.set_geometry("centroid_geometry")
    swiss_centroids_4326 = swiss_centroids.to_crs(4326)
    swiss_centroids["centroid_lat"] = swiss_centroids_4326.geometry.y
    swiss_centroids["centroid_lon"] = swiss_centroids_4326.geometry.x
    swiss = swiss.merge(swiss_centroids, on="zone_id", how="left", validate="one_to_one")
    if swiss["centroid_geometry"].isna().any():
        raise ValueError("Swiss centroid layer did not cover all Swiss zone IDs.")

    foreign_union = gpd.read_file(foreign_path, layer=FOREIGN_ZONE_LAYER)[
        ["ID_Zone", "Name", "NUTS2", "geometry"]
    ].copy()
    foreign_union["zone_id"] = foreign_union["ID_Zone"].round().astype(int)
    swiss_zone_ids = set(swiss["zone_id"])
    foreign_only = foreign_union[~foreign_union["zone_id"].isin(swiss_zone_ids)].copy()
    foreign_only["zone_name"] = foreign_only["Name"].fillna("").astype(str)
    foreign_only["country_code"] = foreign_only["NUTS2"].fillna("").astype(str).str[:2].replace("", "ZZ")
    foreign_only = foreign_only[["zone_id", "zone_name", "country_code", "geometry"]]

    centroid_series = gpd.GeoSeries(
        foreign_only.to_crs(3035).geometry.centroid,
        crs="EPSG:3035",
    ).to_crs(4326)
    foreign_only = foreign_only.to_crs(4326)
    foreign_only["centroid_geometry"] = centroid_series.values

    swiss = swiss.set_geometry("geometry")
    return swiss, foreign_only


def build_zone_station_membership(
    stations: pd.DataFrame,
    swiss_zones: gpd.GeoDataFrame,
    foreign_zones: gpd.GeoDataFrame,
) -> dict[int, list[str]]:
    station_points_4326 = gpd.GeoDataFrame(
        stations[["StationName"]].copy(),
        geometry=gpd.points_from_xy(stations["lon"], stations["lat"]),
        crs="EPSG:4326",
    )
    station_points_2056 = station_points_4326.to_crs(2056)
    swiss_zones_2056 = swiss_zones.to_crs(2056)

    membership: dict[int, set[str]] = defaultdict(set)

    swiss_join = gpd.sjoin(
        station_points_2056,
        swiss_zones_2056[["zone_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    for _, row in swiss_join.iterrows():
        membership[int(row["zone_id"])].add(str(row["StationName"]))

    foreign_join = gpd.sjoin(
        station_points_4326,
        foreign_zones[["zone_id", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    for _, row in foreign_join.iterrows():
        membership[int(row["zone_id"])].add(str(row["StationName"]))

    return {zone_id: sorted(stations_in_zone) for zone_id, stations_in_zone in membership.items()}


def build_zone_records(
    swiss_zones: gpd.GeoDataFrame,
    foreign_zones: gpd.GeoDataFrame,
) -> pd.DataFrame:
    swiss_records = pd.DataFrame(
        {
            "zone_id": swiss_zones["zone_id"].astype(int),
            "zone_name": swiss_zones["zone_name"].astype(str),
            "country_code": swiss_zones["country_code"].astype(str),
            "centroid_lat": swiss_zones["centroid_lat"].astype(float),
            "centroid_lon": swiss_zones["centroid_lon"].astype(float),
            "zone_scope": "swiss_official",
        }
    )
    foreign_records = pd.DataFrame(
        {
            "zone_id": foreign_zones["zone_id"].astype(int),
            "zone_name": foreign_zones["zone_name"].astype(str),
            "country_code": foreign_zones["country_code"].astype(str),
            "centroid_lat": foreign_zones["centroid_geometry"].y.astype(float),
            "centroid_lon": foreign_zones["centroid_geometry"].x.astype(float),
            "zone_scope": "non_swiss_from_union_file",
        }
    )
    records = pd.concat([swiss_records, foreign_records], ignore_index=True)
    records = records.drop_duplicates(subset=["zone_id"], keep="first").sort_values("zone_id").reset_index(drop=True)
    return records


def compute_zone_station_allocations(
    zone_records: pd.DataFrame,
    stations: pd.DataFrame,
    membership: dict[int, list[str]],
    params: AllocationParameters,
) -> tuple[list[dict[str, object]], dict[int, list[tuple[int, float]]], dict[str, object]]:
    station_names = stations["StationName"].tolist()
    station_index_by_name = {name: idx for idx, name in enumerate(station_names)}
    station_lats = stations["lat"].to_numpy(dtype=float)
    station_lons = stations["lon"].to_numpy(dtype=float)
    service_counts = stations["service_2023"].to_numpy(dtype=float)
    service_scores = np.power(np.clip(service_counts, 1.0, None), params.service_exponent)

    allocation_rows: list[dict[str, object]] = []
    allocation_map: dict[int, list[tuple[int, float]]] = {}
    zones_with_in_zone_station = 0
    weighted_in_zone_share_sum = 0.0
    weighted_zone_count = 0
    max_share_values: list[float] = []
    candidate_count_counter: Counter[int] = Counter()

    for _, zone in zone_records.iterrows():
        zone_id = int(zone["zone_id"])
        zone_lat = float(zone["centroid_lat"])
        zone_lon = float(zone["centroid_lon"])
        distances = np.array(
            [haversine_km(zone_lat, zone_lon, lat, lon) for lat, lon in zip(station_lats, station_lons)],
            dtype=float,
        )
        nearest_indices = np.argsort(distances)[: max(params.nearest_pool, params.min_candidates)]
        in_zone_names = membership.get(zone_id, [])
        in_zone_indices = [station_index_by_name[name] for name in in_zone_names if name in station_index_by_name]

        candidate_indices: list[int] = []
        seen: set[int] = set()
        for idx in in_zone_indices:
            if idx not in seen:
                candidate_indices.append(idx)
                seen.add(idx)
        for idx in nearest_indices:
            idx = int(idx)
            if idx not in seen:
                candidate_indices.append(idx)
                seen.add(idx)
            if len(candidate_indices) >= params.min_candidates and len(candidate_indices) >= len(in_zone_indices):
                if len(candidate_indices) >= params.nearest_pool:
                    break

        if len(candidate_indices) < params.min_candidates:
            for idx in np.argsort(distances):
                idx = int(idx)
                if idx in seen:
                    continue
                candidate_indices.append(idx)
                seen.add(idx)
                if len(candidate_indices) >= params.min_candidates:
                    break

        raw_weights: list[float] = []
        candidate_rows: list[tuple[int, float, bool, float, float]] = []
        in_zone_share_raw = 0.0
        for idx in candidate_indices:
            distance_km = float(distances[idx])
            in_zone = idx in in_zone_indices
            base_weight = service_scores[idx] / ((distance_km + params.access_offset_km) ** params.distance_exponent)
            if in_zone:
                base_weight *= params.in_zone_bonus
                in_zone_share_raw += base_weight
            raw_weights.append(base_weight)
            candidate_rows.append((idx, distance_km, in_zone, service_counts[idx], base_weight))

        total_weight = float(sum(raw_weights))
        if total_weight <= 0:
            normalized = [1.0 / len(candidate_rows)] * len(candidate_rows)
        else:
            normalized = [weight / total_weight for weight in raw_weights]

        allocation_map[zone_id] = [
            (idx, share)
            for (idx, _, _, _, _), share in zip(candidate_rows, normalized)
        ]

        if in_zone_indices:
            zones_with_in_zone_station += 1
        weighted_in_zone_share_sum += sum(
            share for (_, _, in_zone, _, _), share in zip(candidate_rows, normalized) if in_zone
        )
        weighted_zone_count += 1
        max_share_values.append(max(normalized))
        candidate_count_counter[len(candidate_rows)] += 1

        for rank, ((idx, distance_km, in_zone, service_count, raw_weight), share) in enumerate(
            zip(candidate_rows, normalized),
            start=1,
        ):
            station_name = station_names[idx]
            allocation_rows.append(
                {
                    "zone_id": zone_id,
                    "zone_name": str(zone["zone_name"]),
                    "zone_country_code": str(zone["country_code"]),
                    "zone_scope": str(zone["zone_scope"]),
                    "centroid_lat": f"{zone_lat:.8f}",
                    "centroid_lon": f"{zone_lon:.8f}",
                    "station": station_name,
                    "station_country": str(stations.iloc[idx]["Country"]).strip(),
                    "station_canton": str(stations.iloc[idx]["Region"]).strip(),
                    "station_rank": rank,
                    "distance_km": f"{distance_km:.6f}",
                    "in_zone": "1" if in_zone else "0",
                    "service_2023": f"{service_count:.6f}",
                    "allocation_share": f"{share:.12f}",
                    "raw_weight": f"{raw_weight:.12f}",
                    "allocation_model": params.allocation_model_name,
                    "min_candidates": params.min_candidates,
                    "nearest_pool": params.nearest_pool,
                    "in_zone_bonus": f"{params.in_zone_bonus:.6f}",
                    "access_offset_km": f"{params.access_offset_km:.6f}",
                    "distance_exponent": f"{params.distance_exponent:.6f}",
                    "service_exponent": f"{params.service_exponent:.6f}",
                }
            )

    diagnostics = {
        "zoneCount": int(len(zone_records)),
        "zonesWithInZoneStation": int(zones_with_in_zone_station),
        "avgInZoneShare": weighted_in_zone_share_sum / max(weighted_zone_count, 1),
        "avgMaxAllocationShare": float(np.mean(max_share_values)) if max_share_values else 0.0,
        "medianMaxAllocationShare": float(np.median(max_share_values)) if max_share_values else 0.0,
        "candidateCountDistribution": {str(key): int(value) for key, value in sorted(candidate_count_counter.items())},
    }
    return allocation_rows, allocation_map, diagnostics


def iter_npvm_matrix_rows(matrix_zip_path: Path) -> tuple[int, int, float]:
    with ZipFile(matrix_zip_path) as archive:
        with archive.open(MATRIX_MEMBER_NAME) as raw_handle:
            text_handle = TextIOWrapper(raw_handle, encoding="utf-8-sig", newline="")
            for line in text_handle:
                stripped = line.strip()
                if (
                    not stripped
                    or stripped.startswith("*")
                    or stripped.startswith("-")
                    or stripped.startswith("$")
                ):
                    continue
                parts = stripped.split()
                if len(parts) < 3:
                    continue
                try:
                    origin_zone = int(parts[0])
                    destination_zone = int(parts[1])
                    trips = float(parts[2])
                except ValueError:
                    continue
                yield origin_zone, destination_zone, trips


def aggregate_station_matrix(
    allocation_map: dict[int, list[tuple[int, float]]],
    station_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    pair_matrix = np.zeros((station_count, station_count), dtype=float)
    production_vector = np.zeros(station_count, dtype=float)
    attraction_vector = np.zeros(station_count, dtype=float)

    total_zone_trips = 0.0
    total_station_trips = 0.0
    diagonal_trips = 0.0
    missing_zone_trips = 0.0
    missing_zone_ids: set[int] = set()
    matrix_zone_ids: set[int] = set()
    line_count = 0

    for origin_zone, destination_zone, trips in iter_npvm_matrix_rows(MATRIX_ZIP_PATH):
        line_count += 1
        matrix_zone_ids.add(origin_zone)
        matrix_zone_ids.add(destination_zone)
        if line_count % 2_000_000 == 0:
            print(f"Processed {line_count:,} matrix rows", flush=True)

        origin_allocation = allocation_map.get(origin_zone)
        destination_allocation = allocation_map.get(destination_zone)
        if not origin_allocation or not destination_allocation:
            missing_zone_trips += trips
            if not origin_allocation:
                missing_zone_ids.add(origin_zone)
            if not destination_allocation:
                missing_zone_ids.add(destination_zone)
            continue

        total_zone_trips += trips
        for origin_idx, origin_share in origin_allocation:
            for destination_idx, destination_share in destination_allocation:
                contribution = trips * origin_share * destination_share
                production_vector[origin_idx] += contribution
                attraction_vector[destination_idx] += contribution
                total_station_trips += contribution
                if origin_idx == destination_idx:
                    diagonal_trips += contribution
                else:
                    pair_matrix[origin_idx, destination_idx] += contribution

    diagnostics = {
        "matrixLineCount": int(line_count),
        "matrixZoneIdCount": int(len(matrix_zone_ids)),
        "matrixZoneIdsMissingAllocation": sorted(int(zone_id) for zone_id in missing_zone_ids),
        "matrixZoneIdsMissingAllocationCount": int(len(missing_zone_ids)),
        "totalZoneTripsProcessed": total_zone_trips,
        "totalStationTripsAllocated": total_station_trips,
        "diagonalTrips": diagonal_trips,
        "offDiagonalTrips": float(pair_matrix.sum()),
        "missingZoneTrips": missing_zone_trips,
    }
    return pair_matrix, production_vector, attraction_vector, diagnostics


def write_allocation_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "zone_id",
        "zone_name",
        "zone_country_code",
        "zone_scope",
        "centroid_lat",
        "centroid_lon",
        "station",
        "station_country",
        "station_canton",
        "station_rank",
        "distance_km",
        "in_zone",
        "service_2023",
        "allocation_share",
        "raw_weight",
        "allocation_model",
        "min_candidates",
        "nearest_pool",
        "in_zone_bonus",
        "access_offset_km",
        "distance_exponent",
        "service_exponent",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_station_od_rows(
    stations: pd.DataFrame,
    pair_matrix: np.ndarray,
    params: AllocationParameters,
) -> list[dict[str, object]]:
    station_names = stations["StationName"].tolist()
    station_meta = [row for _, row in stations.iterrows()]
    rows: list[dict[str, object]] = []
    for origin_idx, origin in enumerate(station_names):
        origin_row = station_meta[origin_idx]
        for destination_idx, destination in enumerate(station_names):
            if destination_idx == origin_idx:
                continue
            destination_row = station_meta[destination_idx]
            observed_daily_trips = float(pair_matrix[origin_idx, destination_idx])
            distance_km = max(
                haversine_km(
                    float(origin_row["lat"]),
                    float(origin_row["lon"]),
                    float(destination_row["lat"]),
                    float(destination_row["lon"]),
                ),
                1.0,
            )
            origin_population = int(origin_row["population_2023"])
            destination_population = int(destination_row["population_2023"])
            gravity_weight = (
                origin_population * destination_population
            ) / (distance_km ** DEFAULT_DISTANCE_EXPONENT)

            rows.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "year": REFERENCE_YEAR,
                    "reference_year": REFERENCE_YEAR,
                    "origin_country": str(origin_row["Country"]).strip(),
                    "destination_country": str(destination_row["Country"]).strip(),
                    "origin_canton": str(origin_row["Region"]).strip(),
                    "destination_canton": str(destination_row["Region"]).strip(),
                    "origin_population": origin_population,
                    "destination_population": destination_population,
                    "distance_km": f"{distance_km:.6f}",
                    "pair_weight": f"{observed_daily_trips:.12f}",
                    "observed_daily_trips": f"{observed_daily_trips:.12f}",
                    "gravity_weight": f"{gravity_weight:.12f}",
                    "distance_exponent": f"{DEFAULT_DISTANCE_EXPONENT:.6f}",
                    "weight_model": "npvm_2023_observed_hybrid_zone_to_station_anchor",
                    "allocation_model": params.allocation_model_name,
                    "station_set": TARGET_STATION_SET_NAME,
                    "source": (
                        "Observed NPVM 2023 PT trip matrix aggregated from zones to the modeled station "
                        "system using a local hybrid station-allocation model based on zone-centroid "
                        "distance, 2023 service intensity, and in-zone station membership."
                    ),
                    "notes": (
                        "pair_weight is intentionally set equal to observed_daily_trips for compatibility with "
                        "the later weighting pipeline. This master base-year anchor is observed at the zone-matrix "
                        "level and hybrid only in the zone-to-station allocation step; any later agglomeration-aware "
                        "website weighting is derived from this master station matrix rather than replacing it."
                    ),
                }
            )
    return rows


def write_station_od_csv(rows: list[dict[str, object]], output_path: Path) -> None:
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
        "gravity_weight",
        "distance_exponent",
        "weight_model",
        "allocation_model",
        "station_set",
        "source",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def top_station_totals(stations: pd.DataFrame, values: np.ndarray, limit: int = 15) -> list[dict[str, object]]:
    indexed = [
        {"station": station, "trips": float(value)}
        for station, value in zip(stations["StationName"].tolist(), values)
    ]
    return sorted(indexed, key=lambda row: row["trips"], reverse=True)[:limit]


def top_od_pairs(stations: pd.DataFrame, pair_matrix: np.ndarray, limit: int = 20) -> list[dict[str, object]]:
    station_names = stations["StationName"].tolist()
    results: list[dict[str, object]] = []
    for origin_idx, origin in enumerate(station_names):
        for destination_idx, destination in enumerate(station_names):
            if origin_idx == destination_idx:
                continue
            trips = float(pair_matrix[origin_idx, destination_idx])
            results.append(
                {
                    "origin": origin,
                    "destination": destination,
                    "trips": trips,
                }
            )
    return sorted(results, key=lambda row: row["trips"], reverse=True)[:limit]


def write_summary_json(payload: dict[str, object], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_validation_markdown(payload: dict[str, object], output_path: Path) -> None:
    allocation = payload["allocationDiagnostics"]
    aggregation = payload["aggregationDiagnostics"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Station OD Validation 2023",
        "",
        "## Scope",
        "",
        f"- Reference year: `{REFERENCE_YEAR}`",
        f"- Station set: `{TARGET_STATION_SET_NAME}` ({payload['stationCount']} stations)",
        f"- Zone count with allocations: `{payload['zoneCount']}`",
        f"- Allocation model: `{payload['allocationModel']}`",
        "",
        "## Allocation Model",
        "",
        "- Candidate construction:",
        f"  - all stations physically inside a zone polygon are retained",
        f"  - nearest stations are then added until at least `{payload['parameters']['minCandidates']}` candidates are present",
        f"  - the nearest-station search pool is `{payload['parameters']['nearestPool']}`",
        "- Weight formula:",
        f"  - service term = service_2023 ^ `{payload['parameters']['serviceExponent']}`",
        f"  - access term = 1 / (distance_km + `{payload['parameters']['accessOffsetKm']}`) ^ `{payload['parameters']['distanceExponent']}`",
        f"  - in-zone stations receive a multiplier of `{payload['parameters']['inZoneBonus']}`",
        "",
        "## Allocation Diagnostics",
        "",
        f"- Zones with at least one in-zone target station: `{allocation['zonesWithInZoneStation']}` / `{allocation['zoneCount']}`",
        f"- Average allocation share assigned to in-zone stations: `{allocation['avgInZoneShare']:.4f}`",
        f"- Average highest single-station share per zone: `{allocation['avgMaxAllocationShare']:.4f}`",
        f"- Median highest single-station share per zone: `{allocation['medianMaxAllocationShare']:.4f}`",
        f"- Candidate-count distribution: `{allocation['candidateCountDistribution']}`",
        "",
        "## Matrix Aggregation Diagnostics",
        "",
        f"- Matrix lines processed: `{aggregation['matrixLineCount']}`",
        f"- Distinct zone IDs seen in the matrix: `{aggregation['matrixZoneIdCount']}`",
        f"- Zone IDs missing an allocation: `{aggregation['matrixZoneIdsMissingAllocationCount']}`",
        f"- Total zone trips processed: `{aggregation['totalZoneTripsProcessed']:.3f}`",
        f"- Total station trips allocated: `{aggregation['totalStationTripsAllocated']:.3f}`",
        f"- Off-diagonal station trips retained in the final directed OD matrix: `{aggregation['offDiagonalTrips']:.3f}`",
        f"- Same-station diagonal trips kept only for validation: `{aggregation['diagonalTrips']:.3f}`",
        f"- Trips skipped because a zone allocation was missing: `{aggregation['missingZoneTrips']:.3f}`",
        "",
        "## Top Origin Stations by Off-Diagonal Production",
        "",
    ]
    for row in payload["topProductions"]:
        lines.append(f"- `{row['station']}`: `{row['trips']:.3f}`")

    lines.extend(["", "## Top Destination Stations by Off-Diagonal Attraction", ""])
    for row in payload["topAttractions"]:
        lines.append(f"- `{row['station']}`: `{row['trips']:.3f}`")

    lines.extend(["", "## Top Directed OD Pairs", ""])
    for row in payload["topPairs"]:
        lines.append(f"- `{row['origin']} -> {row['destination']}`: `{row['trips']:.3f}`")

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This file validates the raw master 2023 station-OD anchor, not yet the historical extrapolation.",
            "- The observed part is the NPVM 2023 zone-to-zone PT matrix itself.",
            "- The hybrid part is the zone-to-station allocation layer, which is fully written out in the allocation CSV for auditability.",
            "- The top production and attraction rankings above are based on the exported off-diagonal station matrix; same-station demand is reported separately through the diagonal total.",
            "- Because this master anchor keeps the full modeled station system, strong metropolitan pairs remain visible here by design. The later website weighting layer can derive an inter-agglomeration analysis view from this master matrix without altering the raw anchor.",
            "- Update: those later steps have now been implemented. The accepted 2023 inter-agglomeration layer, corridor validation, and all-years OD projection model have been added on top of this master anchor; this file remains the stage-specific validation note for the raw 2023 master layer.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    params = AllocationParameters(
        min_candidates=max(args.min_candidates, 1),
        nearest_pool=max(args.nearest_pool, args.min_candidates, 1),
        in_zone_bonus=max(args.in_zone_bonus, 1.0),
        access_offset_km=max(args.access_offset_km, 0.1),
        distance_exponent=max(args.distance_exponent, 0.1),
        service_exponent=max(args.service_exponent, 0.0),
    )

    stations = load_target_stations()
    swiss_zones, foreign_zones = load_zone_geometries()
    membership = build_zone_station_membership(stations, swiss_zones, foreign_zones)
    zone_records = build_zone_records(swiss_zones, foreign_zones)

    allocation_rows, allocation_map, allocation_diagnostics = compute_zone_station_allocations(
        zone_records,
        stations,
        membership,
        params,
    )
    write_allocation_csv(allocation_rows, args.allocation_output)
    print(f"Wrote zone-to-station allocation table to {args.allocation_output.name}", flush=True)

    pair_matrix, production_vector, attraction_vector, aggregation_diagnostics = aggregate_station_matrix(
        allocation_map,
        len(stations),
    )
    od_rows = build_station_od_rows(stations, pair_matrix, params)
    write_station_od_csv(od_rows, args.od_output)
    print(f"Wrote aggregated station OD matrix to {args.od_output.name}", flush=True)

    summary_payload = {
        "referenceYear": REFERENCE_YEAR,
        "stationSet": TARGET_STATION_SET_NAME,
        "stationCount": int(len(stations)),
        "zoneCount": int(len(zone_records)),
        "allocationModel": params.allocation_model_name,
        "parameters": {
            "minCandidates": params.min_candidates,
            "nearestPool": params.nearest_pool,
            "inZoneBonus": params.in_zone_bonus,
            "accessOffsetKm": params.access_offset_km,
            "distanceExponent": params.distance_exponent,
            "serviceExponent": params.service_exponent,
        },
        "allocationDiagnostics": allocation_diagnostics,
        "aggregationDiagnostics": aggregation_diagnostics,
        "topProductions": top_station_totals(stations, pair_matrix.sum(axis=1)),
        "topAttractions": top_station_totals(stations, pair_matrix.sum(axis=0)),
        "topProductionsIncludingDiagonal": top_station_totals(stations, production_vector),
        "topAttractionsIncludingDiagonal": top_station_totals(stations, attraction_vector),
        "topPairs": top_od_pairs(stations, pair_matrix),
        "sourceFiles": {
            "matrixZip": str(MATRIX_ZIP_PATH.name),
            "swissZoneZip": str(SWISS_ZONE_ZIP_PATH.name),
            "swissCentroids": str(SWISS_CENTROIDS_PATH.name),
            "foreignZoneZip": str(FOREIGN_ZONE_ZIP_PATH.name),
            "stationCsv": str(FILTERED_STATIONS_PATH.name),
        },
    }
    write_summary_json(summary_payload, args.summary_output)
    write_validation_markdown(summary_payload, args.validation_output)
    print(f"Wrote machine-readable summary to {args.summary_output.name}", flush=True)
    print(f"Wrote validation report to {args.validation_output.name}", flush=True)


if __name__ == "__main__":
    main()
