from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from sys import intern
from typing import Any


DEFAULT_STATION_POPULATION = 30000
DEFAULT_DISTANCE_EXPONENT = 2.0
DEFAULT_REFERENCE_YEAR = "2023"
OD_PAIR_WEIGHTS_FILENAME = "All_OD_Pairings.csv"

DEFAULT_WEIGHT_SOURCES = {
    "npvm_2023_observed_hybrid_selected_station_projection_with_same_agglomeration_distance_attenuation": (
        "Accepted 2023 NPVM-based selected-station anchor on the 131-station analysis layer, with calibrated "
        "distance attenuation for same-agglomeration station pairs."
    ),
    "npvm_2023_anchor_population_gcost_ipf_projection": (
        "Accepted 2023 selected-station anchor projected across timetable years with representative 06:00, 07:00, "
        "and 08:00 generalized-cost skims for the repeating 05:00-00:00 operating day, calibrated same-agglomeration "
        "distance attenuation, population-scaled marginals, and IPF balancing."
    ),
    "gravity_proxy_station_pairs_reference_2023": "Legacy 2023 gravity proxy table.",
}


def _safe_float(value: Any) -> float | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any, default: int | None = None) -> int | None:
    numeric = _safe_float(value)
    if numeric is None:
        return default
    return int(round(numeric))


def _default_weight_source(model: str, csv_name: str) -> str:
    normalized = str(model or "").strip()
    if normalized in DEFAULT_WEIGHT_SOURCES:
        return DEFAULT_WEIGHT_SOURCES[normalized]
    return f"Local pair-weight table ({csv_name})"


def _pack_pair_record(
    *,
    pair_weight: float,
    weight_model: str,
    weight_source: str,
    reference_year: str,
    observed_daily_trips: float | None,
    base_observed_daily_trips_2023: float | None,
    distance_exponent: float | None,
    excluded_from_analysis: bool,
    analysis_status: str,
) -> tuple[Any, ...]:
    return (
        float(pair_weight),
        intern(str(weight_model)),
        intern(str(weight_source)),
        intern(str(reference_year)),
        observed_daily_trips,
        base_observed_daily_trips_2023,
        distance_exponent,
        bool(excluded_from_analysis),
        intern(str(analysis_status)),
    )


def _unpack_pair_record(record: tuple[Any, ...]) -> dict[str, Any]:
    (
        pair_weight,
        weight_model,
        weight_source,
        reference_year,
        observed_daily_trips,
        base_observed_daily_trips_2023,
        distance_exponent,
        excluded_from_analysis,
        analysis_status,
    ) = record
    return {
        "pairWeight": pair_weight,
        "distanceKm": None,
        "originPopulation": None,
        "destinationPopulation": None,
        "weightModel": weight_model,
        "weightSource": weight_source,
        "referenceYear": reference_year,
        "usedTable": True,
        "fallback": False,
        "observedDailyTrips": observed_daily_trips,
        "baseObservedDailyTrips2023": base_observed_daily_trips_2023,
        "gravityWeight": None,
        "notes": "",
        "distanceExponent": distance_exponent,
        "excludedFromAnalysis": excluded_from_analysis,
        "analysisStatus": analysis_status,
    }


def od_weight_table_content_hash(path: str | Path) -> str | None:
    csv_path = Path(path)
    if not csv_path.exists():
        return None
    return hashlib.sha256(csv_path.read_bytes()).hexdigest()


def build_od_weight_cache_signature(
    file_name: str,
    reference_year: str,
    rows_loaded: int,
    content_hash: str | None,
) -> str:
    digest = (content_hash or "missing")[:16]
    reference = str(reference_year or DEFAULT_REFERENCE_YEAR).strip() or DEFAULT_REFERENCE_YEAR
    return f"{file_name}:{reference}:{int(rows_loaded)}:{digest}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * radius_km * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def station_population(
    station_metadata: dict[str, dict[str, Any]],
    station: str,
    base_year: str,
    default_population: int = DEFAULT_STATION_POPULATION,
) -> int:
    station_meta = station_metadata.get(station, {})
    populations = station_meta.get("population", {})
    value = populations.get(base_year, default_population)
    if value is None or value == "":
        return default_population
    try:
        numeric = int(float(value))
    except (TypeError, ValueError):
        return default_population
    return numeric if numeric > 0 else default_population


def gravity_pair_weight(
    origin: str,
    destination: str,
    base_year: str,
    station_metadata: dict[str, dict[str, Any]],
    station_coords: dict[str, dict[str, float]],
    distance_exponent: float = DEFAULT_DISTANCE_EXPONENT,
    default_population: int = DEFAULT_STATION_POPULATION,
) -> dict[str, Any]:
    origin_coords = station_coords.get(origin)
    destination_coords = station_coords.get(destination)
    if not origin_coords or not destination_coords:
        return {
            "pairWeight": 0.0,
            "distanceKm": None,
            "originPopulation": default_population,
            "destinationPopulation": default_population,
            "weightModel": "gravity_fallback_dynamic",
            "weightSource": (
                "FilteredStations station populations (municipality-based 2023 bases plus official canton scaling, "
                "with same-municipality stations split by planner-year service counts where available) and "
                "crow-fly distance"
            ),
            "referenceYear": base_year,
            "usedTable": False,
            "fallback": True,
            "observedDailyTrips": None,
            "gravityWeight": None,
            "notes": "Coordinates missing for at least one station.",
            "distanceExponent": distance_exponent,
            "excludedFromAnalysis": False,
            "analysisStatus": "fallback_missing_coordinates",
        }

    distance_km = max(
        haversine_km(
            origin_coords["lat"],
            origin_coords["lon"],
            destination_coords["lat"],
            destination_coords["lon"],
        ),
        1.0,
    )
    origin_pop = station_population(
        station_metadata,
        origin,
        base_year,
        default_population=default_population,
    )
    destination_pop = station_population(
        station_metadata,
        destination,
        base_year,
        default_population=default_population,
    )
    gravity_weight = (origin_pop * destination_pop) / (distance_km ** distance_exponent)
    return {
        "pairWeight": gravity_weight,
        "distanceKm": distance_km,
        "originPopulation": origin_pop,
        "destinationPopulation": destination_pop,
        "weightModel": "gravity_fallback_dynamic",
        "weightSource": (
            "FilteredStations station populations (municipality-based 2023 bases plus official canton scaling, "
            "with same-municipality stations split by planner-year service counts where available) and "
            "crow-fly distance"
        ),
        "referenceYear": base_year,
        "usedTable": False,
        "fallback": True,
        "observedDailyTrips": None,
        "gravityWeight": gravity_weight,
        "notes": "Computed on the fly because no station-pair table row was available.",
        "distanceExponent": distance_exponent,
        "excludedFromAnalysis": False,
        "analysisStatus": "fallback_computed",
    }


def load_od_pair_weights(
    path: str | Path,
    fallback_reference_year: str = DEFAULT_REFERENCE_YEAR,
) -> dict[str, Any]:
    csv_path = Path(path)
    content_hash = od_weight_table_content_hash(csv_path)
    payload: dict[str, Any] = {
        "path": str(csv_path),
        "referenceYear": fallback_reference_year,
        "pairs": {},
        "yearsAvailable": [],
        "models": [],
        "rowsLoaded": 0,
        "activeRows": 0,
        "excludedRows": 0,
        "analysisStatuses": {},
        "hasObservedDemand": False,
        "contentHash": content_hash,
        "cacheSignature": build_od_weight_cache_signature(
            csv_path.name,
            fallback_reference_year,
            0,
            content_hash,
        ),
    }
    if not csv_path.exists():
        return payload

    years_available: set[str] = set()
    models: set[str] = set()
    pairs: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    reference_year = fallback_reference_year
    has_observed_demand = False
    rows_loaded = 0
    active_rows = 0
    excluded_rows = 0
    analysis_statuses: dict[str, int] = {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            origin = intern(str(row.get("origin", "")).strip())
            destination = intern(str(row.get("destination", "")).strip())
            year = intern(str(row.get("year", "") or row.get("reference_year", "") or fallback_reference_year).strip())
            pair_weight = _safe_float(row.get("pair_weight"))
            exclude_from_analysis = str(row.get("exclude_from_analysis", "")).strip().lower() in {
                "1",
                "true",
                "yes",
            }
            analysis_status = intern(str(row.get("analysis_status", "")).strip() or (
                "excluded" if exclude_from_analysis else "active"
            ))
            if not origin or not destination or not year or pair_weight is None:
                continue

            row_reference_year = intern(str(row.get("reference_year", "") or year).strip())
            if row_reference_year:
                reference_year = row_reference_year

            model = intern(str(row.get("weight_model", "")).strip() or "station_pair_weight_table")
            source = intern(str(row.get("source", "")).strip() or _default_weight_source(model, csv_path.name))
            observed_daily_trips = _safe_float(row.get("observed_daily_trips"))
            base_observed_daily_trips = _safe_float(row.get("base_observed_daily_trips_2023"))
            distance_exponent = _safe_float(row.get("distance_exponent"))

            rows_loaded += 1
            if exclude_from_analysis:
                excluded_rows += 1
            else:
                active_rows += 1
            analysis_statuses[analysis_status] = analysis_statuses.get(analysis_status, 0) + 1
            years_available.add(year)
            models.add(model)
            has_observed_demand = has_observed_demand or observed_daily_trips is not None or base_observed_daily_trips is not None

            pairs[(year, origin, destination)] = _pack_pair_record(
                pair_weight=pair_weight,
                weight_model=model,
                weight_source=source,
                reference_year=row_reference_year or year,
                observed_daily_trips=observed_daily_trips,
                base_observed_daily_trips_2023=base_observed_daily_trips,
                distance_exponent=distance_exponent,
                excluded_from_analysis=exclude_from_analysis,
                analysis_status=analysis_status,
            )

    payload.update(
        {
            "referenceYear": reference_year,
            "pairs": pairs,
            "yearsAvailable": sorted(years_available),
            "models": sorted(models),
            "rowsLoaded": rows_loaded,
            "activeRows": active_rows,
            "excludedRows": excluded_rows,
            "analysisStatuses": analysis_statuses,
            "hasObservedDemand": has_observed_demand,
            "contentHash": content_hash,
            "cacheSignature": build_od_weight_cache_signature(
                csv_path.name,
                reference_year,
                rows_loaded,
                content_hash,
            ),
        }
    )
    return payload


def resolve_od_pair_weight(
    origin: str,
    destination: str,
    year_label: str,
    station_metadata: dict[str, dict[str, Any]],
    station_coords: dict[str, dict[str, float]],
    od_weight_table: dict[str, Any] | None = None,
    distance_exponent: float = DEFAULT_DISTANCE_EXPONENT,
    default_population: int = DEFAULT_STATION_POPULATION,
) -> dict[str, Any]:
    table = od_weight_table or {}
    pairs = table.get("pairs", {})
    reference_year = str(table.get("referenceYear", "") or DEFAULT_REFERENCE_YEAR)
    year = str(year_label).strip()

    for candidate_year in (year, reference_year):
        if not candidate_year:
            continue
        pair = pairs.get((candidate_year, origin, destination))
        if pair:
            return _unpack_pair_record(pair)

    return gravity_pair_weight(
        origin,
        destination,
        year if year else reference_year,
        station_metadata,
        station_coords,
        distance_exponent=distance_exponent,
        default_population=default_population,
    )
