#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


METRIC_FILE_BY_KEY: dict[str, str] = {
    "frequency": "2026_OD_matrix_Frequency_full_LIVE.csv",
    "fastest": "2026_OD_matrix_Fastest_observed_connection_full_LIVE.csv",
    "wait": "2026_OD_matrix_Wait_time_full_LIVE.csv",
    "rolling": "2026_OD_matrix_Rolling_time_full_LIVE.csv",
    "dwell": "2026_OD_matrix_Dwell_time_full_LIVE.csv",
    "transfer": "2026_OD_matrix_Transfer_time_full_LIVE.csv",
}

DEFAULT_EXCLUDED_STATIONS: tuple[str, ...] = ("Cernier Val-de-Ruz",)
EARTH_RADIUS_KM = 6371.0088


def _now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _parse_excluded_station_args(raw_values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            station = part.strip()
            if station:
                out.append(station)
    return out


def _read_square_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, dtype=str, low_memory=False).fillna("")
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if len(df.index) != len(df.columns):
        raise ValueError(
            f"{path.name}: matrix is not square "
            f"(rows={len(df.index)}, cols={len(df.columns)})."
        )
    return df


def _assert_matrix_order(df: pd.DataFrame, filename: str) -> None:
    if list(df.index) != list(df.columns):
        raise ValueError(
            f"{filename}: row labels and column labels are not in identical order."
        )


def _compute_crowfly_distance_km(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg).astype(np.float64)
    lon = np.radians(lon_deg).astype(np.float64)
    lat_col = lat[:, None]
    lon_col = lon[:, None]
    dlat = lat_col - lat_col.T
    dlon = lon_col - lon_col.T
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_col) * np.cos(lat_col.T) * np.sin(dlon / 2.0) ** 2
    np.clip(a, 0.0, 1.0, out=a)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    out = EARTH_RADIUS_KM * c
    np.fill_diagonal(out, 0.0)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare 2026 gravity-model inputs by removing selected stations "
            "from LIVE matrices, aligning station metadata, and building a "
            "crow-fly distance matrix."
        )
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing 2026 LIVE matrices and FilteredStations2026_2035.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "od_modeling_2026",
        help="Output directory for cleaned 2026 gravity inputs.",
    )
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=Path("FilteredStations2026_2035.csv"),
        help="Station metadata CSV relative to --project-dir unless absolute.",
    )
    parser.add_argument(
        "--exclude-station",
        action="append",
        default=list(DEFAULT_EXCLUDED_STATIONS),
        help=(
            "Station name to remove from 2026 inputs. Repeat flag or pass comma-separated list. "
            f"Default: {', '.join(DEFAULT_EXCLUDED_STATIONS)}"
        ),
    )
    args = parser.parse_args()

    project_dir = args.project_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stations_file = args.stations_file
    if not stations_file.is_absolute():
        stations_file = project_dir / stations_file
    if not stations_file.exists():
        raise FileNotFoundError(f"Station metadata file not found: {stations_file}")

    excluded = _parse_excluded_station_args(args.exclude_station)
    excluded_set = set(excluded)
    if not excluded_set:
        raise ValueError("No excluded stations provided; this prep step expects at least one exclusion.")

    print(f"[{_now_local()}] Preparing cleaned 2026 gravity inputs in: {output_dir}")
    print(f"[{_now_local()}] Excluding stations: {sorted(excluded_set)}")

    # Use frequency matrix as canonical station order.
    frequency_path = project_dir / METRIC_FILE_BY_KEY["frequency"]
    if not frequency_path.exists():
        raise FileNotFoundError(f"Missing required matrix: {frequency_path}")

    frequency_df = _read_square_matrix(frequency_path)
    _assert_matrix_order(frequency_df, frequency_path.name)
    station_order = list(frequency_df.index.astype(str))
    missing_requested = [s for s in excluded_set if s not in station_order]
    if missing_requested:
        raise ValueError(
            "Requested excluded stations were not found in the 2026 frequency matrix: "
            f"{missing_requested}"
        )

    keep_stations = [s for s in station_order if s not in excluded_set]
    print(
        f"[{_now_local()}] Canonical 2026 station count: {len(station_order)} "
        f"-> cleaned count: {len(keep_stations)}"
    )

    # 1) Write cleaned station metadata in matrix order.
    stations_df = pd.read_csv(stations_file, dtype=str).fillna("")
    stations_df["StationName"] = stations_df["StationName"].astype(str).str.strip()
    stations_df = stations_df.drop_duplicates(subset=["StationName"], keep="first")
    stations_df = stations_df.set_index("StationName", drop=False)

    missing_in_station_table = [s for s in keep_stations if s not in stations_df.index]
    if missing_in_station_table:
        raise ValueError(
            "Stations present in matrix but missing from station table. "
            f"Example: {missing_in_station_table[:10]}"
        )

    station_subset = stations_df.loc[keep_stations].copy()
    station_subset.insert(1, "gravity_input_rank_2026", np.arange(len(station_subset), dtype=int))
    station_subset["is_exact_sbb_match"] = (
        station_subset["SBB_pf_match_method"].astype(str).str.lower().eq("exact")
        & station_subset["SBB_pf_latest_dtv"].astype(str).str.strip().ne("")
    ).astype(int)
    station_subset_output = output_dir / "FilteredStations2026_gravity_input.csv"
    station_subset.to_csv(station_subset_output, index=False)
    print(f"[{_now_local()}] Wrote {station_subset_output.name}")

    # 2) Write cleaned matrices (same station order as station_subset).
    cleaned_matrix_files: dict[str, str] = {}
    for metric_key, filename in METRIC_FILE_BY_KEY.items():
        path = project_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing matrix for metric '{metric_key}': {path}")
        matrix_df = _read_square_matrix(path)
        _assert_matrix_order(matrix_df, filename)
        if set(matrix_df.index) != set(station_order):
            raise ValueError(
                f"{filename}: station universe mismatch compared with canonical frequency matrix."
            )
        matrix_df = matrix_df.loc[keep_stations, keep_stations]
        output_name = filename.replace("_LIVE.csv", "_gravity_input.csv")
        output_path = output_dir / output_name
        matrix_df.to_csv(output_path, index=True)
        cleaned_matrix_files[metric_key] = output_name
        print(f"[{_now_local()}] Wrote {output_name}")

    # 3) Build and write crow-fly distance matrix in same station order.
    lat = pd.to_numeric(station_subset["lat"], errors="coerce")
    lon = pd.to_numeric(station_subset["lon"], errors="coerce")
    bad_coord = station_subset.index[lat.isna() | lon.isna()].tolist()
    if bad_coord:
        raise ValueError(
            "Missing/invalid coordinates in station metadata for gravity prep. "
            f"Example: {bad_coord[:10]}"
        )

    dist_km = _compute_crowfly_distance_km(lat.to_numpy(dtype=float), lon.to_numpy(dtype=float))
    dist_df = pd.DataFrame(dist_km, index=keep_stations, columns=keep_stations)
    distance_output = output_dir / "2026_station_crowfly_distance_km_gravity_input.csv"
    dist_df.to_csv(distance_output, index=True, float_format="%.6f")
    print(f"[{_now_local()}] Wrote {distance_output.name}")

    # 4) Write manifest for reproducibility.
    manifest = {
        "created_at": _now_local(),
        "project_dir": str(project_dir),
        "source_station_file": str(stations_file),
        "source_matrices": METRIC_FILE_BY_KEY,
        "excluded_stations": sorted(excluded_set),
        "station_count_before": len(station_order),
        "station_count_after": len(keep_stations),
        "output_station_file": station_subset_output.name,
        "output_distance_file": distance_output.name,
        "output_matrices": cleaned_matrix_files,
    }
    manifest_path = output_dir / "2026_gravity_input_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[{_now_local()}] Wrote {manifest_path.name}")
    print(f"[{_now_local()}] Done.")


if __name__ == "__main__":
    main()

