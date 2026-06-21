#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import gc
import gzip
import heapq
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from build_od_first_column_metrics_2026_2035 import (
    DAY_START_MIN,
    WINDOW_END_MIN,
    _compute_day_metrics_from_pattern,
    _decompose_legs_with_rebalance,
    _load_models_with_cache,
    _sample_two_hour_pattern,
    _to_hhmm,
)


BASE_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_CSV = Path(
    os.environ.get("OD_FILTERED_STATIONS_CSV", str(BASE_DIR / "FilteredStations2026_2035.csv"))
).expanduser()
ANCHOR_CACHE_METHOD_FILE = BASE_DIR / "reduced_anchor_cache_method_2026_2035.md"
FULL_OD_METHOD_FILE = BASE_DIR / "full_od_exact_method_2026_2035.md"

METRIC_ORDER = (
    "frequency",
    "fastest_travel_time",
    "avg_wait",
    "avg_rolling",
    "avg_dwell",
    "avg_transfer",
)

METRIC_TO_FILENAME_STEM = {
    "frequency": "OD_matrix_Frequency_full.csv",
    "fastest_travel_time": "OD_matrix_Fastest_observed_connection_full.csv",
    "avg_wait": "OD_matrix_Wait_time_full.csv",
    "avg_rolling": "OD_matrix_Rolling_time_full.csv",
    "avg_dwell": "OD_matrix_Dwell_time_full.csv",
    "avg_transfer": "OD_matrix_Transfer_time_full.csv",
}

REDUCED_ANCHOR_FILES = {
    "frequency": "OD_matrix_Frequency_reduced_anchor.csv",
    "fastest_travel_time": "OD_matrix_Fastest_observed_connection_reduced_anchor.csv",
    "avg_wait": "OD_matrix_Wait_time_reduced_anchor.csv",
    "avg_rolling": "OD_matrix_Rolling_time_reduced_anchor.csv",
    "avg_dwell": "OD_matrix_Dwell_time_reduced_anchor.csv",
    "avg_transfer": "OD_matrix_Transfer_time_reduced_anchor.csv",
}


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _year_output_path(year: str, metric: str, output_tag: str) -> Path:
    stem = f"{year}_{METRIC_TO_FILENAME_STEM[metric]}"
    if not output_tag:
        return BASE_DIR / stem
    if not stem.endswith(".csv"):
        return BASE_DIR / f"{stem}{output_tag}"
    return BASE_DIR / f"{stem[:-4]}{output_tag}.csv"


def _year_live_output_path(year: str, metric: str, output_tag: str) -> Path:
    stem = f"{year}_{METRIC_TO_FILENAME_STEM[metric]}"
    if stem.endswith(".csv"):
        stem = stem[:-4]
    return BASE_DIR / f"{stem}{output_tag}_LIVE.csv"


def _load_station_table() -> pd.DataFrame:
    df = pd.read_csv(FILTERED_STATIONS_CSV)
    if "StationName" not in df.columns:
        raise ValueError(f"{FILTERED_STATIONS_CSV} missing StationName column")
    if "DTV" not in df.columns:
        df["DTV"] = np.nan
    out = df[["StationName", "DTV"]].copy()
    out["StationName"] = out["StationName"].astype(str)
    out["DTV"] = pd.to_numeric(out["DTV"], errors="coerce")
    return out


def _reorder_anchor_then_dtv(
    station_order: list[str],
    anchor_set: set[str],
    dtv_by_station: dict[str, float],
) -> list[str]:
    anchors = [s for s in station_order if s in anchor_set]
    non_anchors = [s for s in station_order if s not in anchor_set]
    non_anchors = sorted(
        non_anchors,
        key=lambda s: (-float(dtv_by_station.get(s, float("-inf"))), s),
    )
    return anchors + non_anchors


def _load_anchor_cache_for_year(year: str) -> tuple[set[str], dict[tuple[str, str], dict[str, float]]]:
    metric_frames: dict[str, pd.DataFrame] = {}
    for metric, suffix in REDUCED_ANCHOR_FILES.items():
        p = BASE_DIR / f"{year}_{suffix}"
        if not p.exists():
            raise FileNotFoundError(f"Missing reduced-anchor cache file: {p}")
        df = pd.read_csv(p)
        if "StationName" not in df.columns:
            raise ValueError(f"{p} missing StationName column")
        metric_frames[metric] = df.set_index("StationName")

    anchors = set(metric_frames["frequency"].index.astype(str).tolist())

    cache: dict[tuple[str, str], dict[str, float]] = {}
    cols = [str(c) for c in metric_frames["frequency"].columns]
    for o in metric_frames["frequency"].index.astype(str):
        for d in cols:
            cache[(o, d)] = {
                "frequency": float(metric_frames["frequency"].loc[o, d]),
                "fastest_travel_time": float(metric_frames["fastest_travel_time"].loc[o, d]),
                "avg_wait": float(metric_frames["avg_wait"].loc[o, d]),
                "avg_rolling": float(metric_frames["avg_rolling"].loc[o, d]),
                "avg_dwell": float(metric_frames["avg_dwell"].loc[o, d]),
                "avg_transfer": float(metric_frames["avg_transfer"].loc[o, d]),
            }
    return anchors, cache


def _csv_cell(v: float) -> str | float:
    fv = float(v)
    if math.isnan(fv):
        return ""
    return fv


def _write_live_csv_rows(
    *,
    path: Path,
    station_order: list[str],
    matrix: np.memmap,
    row_start: int,
    row_end: int,
    include_header: bool,
) -> None:
    mode = "w" if include_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(["StationName", *station_order])
        for r in range(max(0, row_start), max(0, row_end)):
            row_values = np.asarray(matrix[r, :], dtype=np.float64).tolist()
            writer.writerow([station_order[r], *[_csv_cell(v) for v in row_values]])


def _count_live_csv_data_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            n = sum(1 for _ in f)
        return max(0, n - 1)
    except Exception:
        return 0


def _encode_leg_for_payload(leg: dict[str, Any]) -> dict[str, Any]:
    dep_min = int(leg["dep_time"])
    arr_min = int(leg["arr_time"])
    return {
        "from_station": str(leg.get("from_station", "")),
        "to_station": str(leg.get("to_station", "")),
        "dep_min": dep_min,
        "dep_time": _to_hhmm(dep_min),
        "arr_min": arr_min,
        "arr_time": _to_hhmm(arr_min),
        "service_name": str(leg.get("service_name", "")),
        "trip_instance_id": str(leg.get("trip_instance_id", "")),
        "canonical_trip_id": str(leg.get("canonical_trip_id", "")),
        "terminal_station": str(leg.get("terminal_station", "")),
    }


def _build_station_events_for_payload(encoded_legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not encoded_legs:
        return []
    events: list[dict[str, Any]] = []
    events.append(
        {
            "station": encoded_legs[0]["from_station"],
            "arr_min": None,
            "arr_time": None,
            "dep_min": int(encoded_legs[0]["dep_min"]),
            "dep_time": str(encoded_legs[0]["dep_time"]),
        }
    )
    for i, leg in enumerate(encoded_legs):
        dep_next: Optional[int] = None
        dep_next_hhmm: Optional[str] = None
        if i + 1 < len(encoded_legs) and encoded_legs[i + 1]["from_station"] == leg["to_station"]:
            dep_next = int(encoded_legs[i + 1]["dep_min"])
            dep_next_hhmm = str(encoded_legs[i + 1]["dep_time"])
        events.append(
            {
                "station": str(leg["to_station"]),
                "arr_min": int(leg["arr_min"]),
                "arr_time": str(leg["arr_time"]),
                "dep_min": dep_next,
                "dep_time": dep_next_hhmm,
            }
        )
    return events


def _build_transfer_events_for_payload(encoded_legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(len(encoded_legs) - 1):
        cur = encoded_legs[i]
        nxt = encoded_legs[i + 1]
        if str(cur.get("canonical_trip_id", "")) == str(nxt.get("canonical_trip_id", "")):
            continue
        wait_min = max(int(nxt["dep_min"]) - int(cur["arr_min"]), 0)
        out.append(
            {
                "station": str(cur["to_station"]),
                "arrive_prev_min": int(cur["arr_min"]),
                "arrive_prev_time": str(cur["arr_time"]),
                "depart_next_min": int(nxt["dep_min"]),
                "depart_next_time": str(nxt["dep_time"]),
                "transfer_wait_min": int(wait_min),
                "from_service": str(cur.get("service_name", "")),
                "to_service": str(nxt.get("service_name", "")),
            }
        )
    return out


def _pattern_to_nondominated_cell_payload(
    *,
    origin: str,
    destination: str,
    pattern: Optional[list[dict[str, Any]]],
) -> dict[str, Any]:
    if origin == destination:
        return {
            "window_start": _to_hhmm(int(DAY_START_MIN)),
            "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
            "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
            "same_station": True,
            "path_count": 0,
            "paths": [],
        }
    if not pattern:
        return {
            "window_start": _to_hhmm(int(DAY_START_MIN)),
            "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
            "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
            "same_station": False,
            "path_count": 0,
            "paths": [],
        }

    paths: list[dict[str, Any]] = []
    for rec in sorted(pattern, key=lambda x: int(round(float(x["dep"])))):
        dep = int(round(float(rec["dep"])))
        arr = int(round(float(rec["arr"])))
        travel_time = float(rec.get("travel_time", arr - dep))
        rolling = float(rec.get("rolling", math.nan))
        dwell = float(rec.get("dwell", math.nan))
        transfer = float(rec.get("transfer", math.nan))
        raw_legs = rec.get("legs", []) if isinstance(rec, dict) else []
        encoded_legs = [_encode_leg_for_payload(leg) for leg in list(raw_legs or [])]

        line_names: list[str] = []
        for leg in encoded_legs:
            service_name = str(leg.get("service_name", ""))
            if not line_names or line_names[-1] != service_name:
                line_names.append(service_name)

        nodes_sequence: list[str] = []
        if encoded_legs:
            nodes_sequence.append(str(encoded_legs[0]["from_station"]))
            nodes_sequence.extend(str(leg["to_station"]) for leg in encoded_legs)

        station_events = _build_station_events_for_payload(encoded_legs)
        transfer_events = _build_transfer_events_for_payload(encoded_legs)

        paths.append(
            {
                "dep_min": int(dep),
                "dep_time": _to_hhmm(dep),
                "arr_min": int(arr),
                "arr_time": _to_hhmm(arr),
                "travel_time_min": float(travel_time),
                "rolling_min": float(rolling),
                "dwell_min": float(dwell),
                "transfer_min": float(transfer),
                "num_transfers": int(len(transfer_events)),
                "line_names": line_names,
                "nodes_sequence": nodes_sequence,
                "intermediate_nodes": (nodes_sequence[1:-1] if len(nodes_sequence) >= 2 else []),
                "station_events": station_events,
                "transfer_events": transfer_events,
                "legs": encoded_legs,
            }
        )

    return {
        "window_start": _to_hhmm(int(DAY_START_MIN)),
        "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
        "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
        "same_station": False,
        "path_count": int(len(paths)),
        "paths": paths,
    }


def _nondominated_rows_dir(state_root: Path, year: str) -> Path:
    return _state_dir_for_year(state_root, year) / "nondominated_rows"


def _nondominated_row_path_gz(state_root: Path, year: str, row_idx: int) -> Path:
    return _nondominated_rows_dir(state_root, year) / f"row_{int(row_idx):06d}.json.gz"


def _nondominated_row_path_json(state_root: Path, year: str, row_idx: int) -> Path:
    return _nondominated_rows_dir(state_root, year) / f"row_{int(row_idx):06d}.json"


def _nondominated_row_exists(state_root: Path, year: str, row_idx: int) -> bool:
    return _nondominated_row_path_gz(state_root, year, row_idx).exists() or _nondominated_row_path_json(
        state_root, year, row_idx
    ).exists()


def _nondominated_row_resolve_path(state_root: Path, year: str, row_idx: int) -> Optional[Path]:
    p_gz = _nondominated_row_path_gz(state_root, year, row_idx)
    if p_gz.exists():
        return p_gz
    p_json = _nondominated_row_path_json(state_root, year, row_idx)
    if p_json.exists():
        return p_json
    return None


def _atomic_replace_and_fsync(tmp_path: Path, final_path: Path) -> None:
    tmp_path.replace(final_path)
    try:
        dir_fd = os.open(str(final_path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        pass


class _NondominatedRowWriter:
    """
    Stream row payload directly to compressed JSON to avoid holding huge row arrays in memory.
    """

    def __init__(self, *, path: Path, row_idx: int, origin: str, year: str) -> None:
        self.path = path
        self.tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = gzip.open(self.tmp_path, "wt", encoding="utf-8", compresslevel=6)
        self.first_cell = True
        self.closed = False
        prefix = {
            "row_idx": int(row_idx),
            "origin": str(origin),
            "year": str(year),
            "cells": [],
            "completed_at": "",
        }
        txt = json.dumps(prefix, ensure_ascii=False, separators=(",", ":"))
        marker = '"cells":[],"completed_at":""'
        pos = txt.find(marker)
        if pos < 0:
            raise RuntimeError("Unexpected payload template for streaming writer.")
        self.fh.write(txt[:pos])
        self.fh.write('"cells":[')

    def add_cell(self, cell_json: str) -> None:
        if not self.first_cell:
            self.fh.write(",")
        self.first_cell = False
        self.fh.write(json.dumps(cell_json, ensure_ascii=False, separators=(",", ":")))

    def close(self, *, completed_at: str) -> None:
        if self.closed:
            return
        self.fh.write('],"completed_at":')
        self.fh.write(json.dumps(str(completed_at), ensure_ascii=False, separators=(",", ":")))
        self.fh.write("}")
        self.fh.flush()
        raw_fh = getattr(self.fh, "fileobj", None)
        if raw_fh is not None:
            raw_fh.flush()
            os.fsync(raw_fh.fileno())
        self.fh.close()
        _atomic_replace_and_fsync(self.tmp_path, self.path)
        self.closed = True


def _read_nondominated_row_payload(path: Path) -> dict[str, Any]:
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_nondominated_live_csv_rows(
    *,
    path: Path,
    station_order: list[str],
    state_root: Path,
    year: str,
    row_start: int,
    row_end: int,
    include_header: bool,
) -> None:
    mode = "w" if include_header else "a"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(["StationName", *station_order])
        for r in range(max(0, int(row_start)), max(0, int(row_end))):
            row_file = _nondominated_row_resolve_path(state_root, year, r)
            if row_file is None:
                raise FileNotFoundError(
                    f"Missing nondominated row payload for row {r} in {state_root}/{year}"
                )
            payload = _read_nondominated_row_payload(row_file)
            cells = list(payload.get("cells", []))
            writer.writerow([station_order[r], *cells])


def _init_nondominated_live_csv_file(
    *,
    year: str,
    station_order: list[str],
    state_root: Path,
    existing_rows: int,
) -> Path:
    p = BASE_DIR / f"{year}_OD_matrix_Nondominated_paths_full_LIVE.csv"
    _write_nondominated_live_csv_rows(
        path=p,
        station_order=station_order,
        state_root=state_root,
        year=year,
        row_start=0,
        row_end=max(0, min(len(station_order), int(existing_rows))),
        include_header=True,
    )
    return p


def _append_nondominated_live_csv_rows(
    *,
    live_path: Path,
    station_order: list[str],
    state_root: Path,
    year: str,
    row_start: int,
    row_end: int,
) -> None:
    if row_end <= row_start:
        return
    existing_data_rows = _count_live_csv_data_rows(live_path)
    if existing_data_rows != int(row_start):
        _log(
            f"{year}: nondominated live-csv resync for {live_path.name}: "
            f"existing_rows={existing_data_rows} expected_row_start={row_start}; "
            f"rewriting 0..{row_end}"
        )
        _write_nondominated_live_csv_rows(
            path=live_path,
            station_order=station_order,
            state_root=state_root,
            year=year,
            row_start=0,
            row_end=row_end,
            include_header=True,
        )
    else:
        _write_nondominated_live_csv_rows(
            path=live_path,
            station_order=station_order,
            state_root=state_root,
            year=year,
            row_start=row_start,
            row_end=row_end,
            include_header=False,
        )


def _write_nondominated_full_csv(
    *,
    year: str,
    station_order: list[str],
    state_root: Path,
) -> Path:
    out = BASE_DIR / f"{year}_OD_matrix_Nondominated_paths_full.csv"
    _write_nondominated_live_csv_rows(
        path=out,
        station_order=station_order,
        state_root=state_root,
        year=year,
        row_start=0,
        row_end=len(station_order),
        include_header=True,
    )
    return out


def _init_live_csv_files(
    *,
    year: str,
    station_order: list[str],
    matrices: dict[str, np.memmap],
    output_tag: str,
    existing_rows: int,
) -> dict[str, Path]:
    live_paths: dict[str, Path] = {}
    ready_rows = max(0, min(len(station_order), int(existing_rows)))
    for metric in METRIC_ORDER:
        p = _year_live_output_path(year, metric, output_tag)
        _write_live_csv_rows(
            path=p,
            station_order=station_order,
            matrix=matrices[metric],
            row_start=0,
            row_end=ready_rows,
            include_header=True,
        )
        live_paths[metric] = p
    return live_paths


def _append_live_csv_rows(
    *,
    live_paths: dict[str, Path],
    station_order: list[str],
    matrices: dict[str, np.memmap],
    row_start: int,
    row_end: int,
) -> None:
    if row_end <= row_start:
        return
    for metric in METRIC_ORDER:
        p = live_paths[metric]
        existing_data_rows = _count_live_csv_data_rows(p)
        if existing_data_rows != int(row_start):
            _log(
                f"live-csv resync for {p.name}: existing_rows={existing_data_rows} "
                f"expected_row_start={row_start}; rewriting 0..{row_end}"
            )
            _write_live_csv_rows(
                path=p,
                station_order=station_order,
                matrix=matrices[metric],
                row_start=0,
                row_end=row_end,
                include_header=True,
            )
        else:
            _write_live_csv_rows(
                path=p,
                station_order=station_order,
                matrix=matrices[metric],
                row_start=row_start,
                row_end=row_end,
                include_header=False,
            )


def _empty_metrics_unreachable() -> dict[str, float]:
    return {
        "frequency": 0.0,
        "fastest_travel_time": math.nan,
        "avg_wait": math.nan,
        "avg_rolling": math.nan,
        "avg_dwell": math.nan,
        "avg_transfer": math.nan,
    }


def _zero_metrics_same_station() -> dict[str, float]:
    return {
        "frequency": 0.0,
        "fastest_travel_time": 0.0,
        "avg_wait": 0.0,
        "avg_rolling": 0.0,
        "avg_dwell": 0.0,
        "avg_transfer": 0.0,
    }


def _state_dir_for_year(state_root: Path, year: str) -> Path:
    return state_root / f"{year}_full_od_state"


def _matrix_file_path(state_root: Path, year: str, metric: str) -> Path:
    return _state_dir_for_year(state_root, year) / f"{metric}.npy"


def _open_or_create_matrix(
    path: Path,
    shape: tuple[int, int],
    initialize_nan: bool,
) -> np.memmap:
    if initialize_nan or not path.exists():
        arr = np.memmap(path, dtype="float64", mode="w+", shape=shape)
        arr[:, :] = np.nan
        arr.flush()
        return arr
    return np.memmap(path, dtype="float64", mode="r+", shape=shape)


def _load_or_init_year_state(
    year: str,
    n: int,
    station_order: list[str],
    state_root: Path,
    origin_start: int,
    origin_end: int,
    checkpoint_name: str,
) -> tuple[int, dict[str, int], dict[str, np.memmap]]:
    ydir = _state_dir_for_year(state_root, year)
    ydir.mkdir(parents=True, exist_ok=True)
    checkpoint = _state_dir_for_year(state_root, year) / checkpoint_name

    initialize = True
    next_origin_idx = origin_start
    counts = {
        "pairs_total": 0,
        "pairs_same_station": 0,
        "pairs_unavailable": 0,
        "pairs_anchor_cache": 0,
        "pairs_batched_query": 0,
        "pairs_one_to_many_fallback_direct": 0,
        "pairs_direct_query": 0,
        "pairs_no_pattern": 0,
        "one_to_many_batch_runs": 0,
        "one_to_many_destination_lookups": 0,
        "one_to_many_reconstructed": 0,
        "rows_skipped_precomputed": 0,
        "rows_skipped_locked": 0,
    }

    if checkpoint.exists():
        try:
            with checkpoint.open("rb") as f:
                state = pickle.load(f)
            if (
                isinstance(state, dict)
                and state.get("year") == year
                and state.get("station_order") == station_order
                and int(state.get("origin_start", -1)) == int(origin_start)
                and int(state.get("origin_end", -1)) == int(origin_end)
                and isinstance(state.get("next_origin_idx"), int)
                and isinstance(state.get("counts"), dict)
            ):
                initialize = False
                next_origin_idx = int(state["next_origin_idx"])
                for k in counts.keys():
                    counts[k] = int(state["counts"].get(k, 0))
                _log(f"{year}: resuming from origin index {next_origin_idx}")
        except Exception as exc:
            _log(f"{year}: checkpoint load failed ({exc}); rebuilding state.")

    matrices: dict[str, np.memmap] = {}
    for metric in METRIC_ORDER:
        matrices[metric] = _open_or_create_matrix(
            path=_matrix_file_path(state_root, year, metric),
            shape=(n, n),
            initialize_nan=initialize,
        )

    return next_origin_idx, counts, matrices


def _save_checkpoint(
    year: str,
    station_order: list[str],
    origin_start: int,
    origin_end: int,
    next_origin_idx: int,
    counts: dict[str, int],
    state_root: Path,
    checkpoint_name: str,
) -> None:
    state = {
        "year": year,
        "station_order": station_order,
        "origin_start": int(origin_start),
        "origin_end": int(origin_end),
        "next_origin_idx": int(next_origin_idx),
        "counts": {k: int(v) for k, v in counts.items()},
        "saved_at_epoch": time.time(),
    }
    with (_state_dir_for_year(state_root, year) / checkpoint_name).open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def _flush_matrices(matrices: dict[str, np.memmap]) -> None:
    for arr in matrices.values():
        arr.flush()


def _write_full_csvs_for_year(
    year: str,
    station_order: list[str],
    matrices: dict[str, np.memmap],
    output_tag: str,
) -> None:
    for metric in METRIC_ORDER:
        out_path = _year_output_path(year, metric, output_tag)
        df = pd.DataFrame(np.asarray(matrices[metric]), columns=station_order)
        df.insert(0, "StationName", station_order)
        df.to_csv(out_path, index=False)
        _log(f"{year}: wrote {out_path.name}")


def _write_method_summary() -> None:
    lines = [
        "# Full OD Builder Method (2026/2035)",
        "",
        "- Scope: builds full-station OD skims for frequency, fastest observed in-vehicle travel time, average pre-departure wait, and average rolling/dwell/transfer time components.",
        "- Time window assumptions: pattern search from 05:00 to 07:00 (n+1 query strategy), daily interpretation on 05:00-00:00.",
        "- Route selection rule: identical generalized-cost ordering as notebook `earliest_arrival_query` (arrival time, then later first departure, then fewer transfers, then fewer movement segments).",
        "- Anchor cache rule: reduced-anchor values are used only for anchor->anchor pairs, and only after direct-vs-cache validation.",
        "- One-to-many row acceleration (optional, off by default): destination requests are grouped by origin departure-minute seed (`seed_min`) rather than raw query minute, and one generalized-cost search is run per seed bucket.",
        "- Exactness guard: one-to-many mode is enabled only if sampled OD pairs match the direct per-destination method within numerical tolerance (1e-9) for sampled departure records.",
        "- Fallback behavior: any validation failure disables the shortcut and reverts to direct per-destination queries. Runtime exceptions in one-to-many also fallback per pair to direct querying and are counted.",
        "- Resume idempotency: on startup and per-row execution, already-complete rows are detected from frequency memmap completeness and skipped (no recomputation).",
        "- Row-intent locking (optional): per-row non-blocking advisory locks can be enabled so overlapping workers cannot start the same origin row concurrently.",
        "- Interleaved worker slicing (optional): `--row-modulo M --row-remainder R` restricts a worker to rows where `row_index % M == R`, enabling exact disjoint row assignment such as every fourth row.",
        "- Zero-minute handling: rolling and dwell 0-minute segments are rebalanced with a 20-second minimum while preserving exact end-to-end travel time.",
        "",
        "## Output",
        "",
        "- `<YEAR>_OD_matrix_Frequency_full*.csv`",
        "- `<YEAR>_OD_matrix_Fastest_observed_connection_full*.csv`",
        "- `<YEAR>_OD_matrix_Wait_time_full*.csv`",
        "- `<YEAR>_OD_matrix_Rolling_time_full*.csv`",
        "- `<YEAR>_OD_matrix_Dwell_time_full*.csv`",
        "- `<YEAR>_OD_matrix_Transfer_time_full*.csv`",
        "",
        "## Logging counters",
        "",
        "- `pairs_direct_query`: destination pairs solved via direct per-destination querying.",
        "- `pairs_batched_query`: destination pairs solved through one-to-many row batches.",
        "- `pairs_one_to_many_fallback_direct`: destination pairs that attempted one-to-many but were recomputed with direct query due to runtime fallback.",
        "- `pairs_anchor_cache`: destination pairs solved from reduced-anchor cache.",
        "- `one_to_many_batch_runs`: number of unique origin+query-minute one-to-many searches executed.",
        "- `rows_skipped_precomputed`: rows detected as already complete in memmap and therefore not recomputed.",
        "- `rows_skipped_locked`: rows deferred because a row-intent lock was held by another worker at runtime.",
    ]
    FULL_OD_METHOD_FILE.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def _acquire_state_lock(state_dir: Path, years: list[str]) -> Any:
    """
    Prevent concurrent runs writing to the same state/live outputs.
    Uses advisory flock; lock releases automatically when process exits.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_path = state_dir / ".full_od_run.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        holder = lock_file.read().strip()
        raise RuntimeError(
            f"State dir is already locked by another run: {state_dir}\n"
            f"Lock file: {lock_path}\n"
            f"Holder info: {holder or 'unavailable'}\n"
            f"Stop the other run (or use a different --state-dir)."
        )
    holder_info = {
        "pid": os.getpid(),
        "years": years,
        "acquired_at": _ts(),
        "state_dir": str(state_dir),
    }
    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(json.dumps(holder_info, ensure_ascii=False))
    lock_file.flush()
    _log(f"Acquired state lock: {lock_path}")
    return lock_file


def _try_acquire_row_lock(
    *,
    row_lock_dir: Path,
    year: str,
    row_idx: int,
    origin: str,
    pid: int,
) -> tuple[Optional[Any], Path]:
    """
    Acquire a non-blocking advisory lock for a single origin row.
    Returns (handle, path). If handle is None, another process currently holds it.
    """
    row_lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = row_lock_dir / f"{year}_row_{int(row_idx):06d}.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.close()
        return None, lock_path
    try:
        lock_file.seek(0)
        lock_file.truncate(0)
        lock_file.write(
            json.dumps(
                {
                    "year": str(year),
                    "row_idx": int(row_idx),
                    "origin": str(origin),
                    "pid": int(pid),
                    "claimed_at_epoch": time.time(),
                }
            )
        )
        lock_file.flush()
    except Exception:
        pass
    return lock_file, lock_path


def _validate_anchor_cache_exactness(
    year: str,
    model: Any,
    anchor_set: set[str],
    anchor_cache: dict[tuple[str, str], dict[str, float]],
    max_legs: int,
) -> tuple[bool, list[str]]:
    anchors_sorted = sorted(anchor_set)
    if len(anchors_sorted) < 2:
        return False, [f"{year}: anchor set too small for validation"]

    sample_pairs: list[tuple[str, str]] = []
    sample_pairs.append((anchors_sorted[0], anchors_sorted[min(5, len(anchors_sorted) - 1)]))
    sample_pairs.append((anchors_sorted[len(anchors_sorted) // 2], anchors_sorted[-1]))
    sample_pairs.append((anchors_sorted[-2], anchors_sorted[1]))

    messages: list[str] = []
    ok_all = True
    for origin, destination in sample_pairs:
        cached = anchor_cache.get((origin, destination))
        if cached is None:
            messages.append(f"{year} {origin}->{destination}: missing in anchor cache")
            ok_all = False
            continue

        pattern = _sample_two_hour_pattern(
            model=model,
            origin=origin,
            destination=destination,
            max_legs=max_legs,
            include_legs=False,
        )
        if not pattern:
            messages.append(f"{year} {origin}->{destination}: direct query had no pattern")
            ok_all = False
            continue

        direct = _compute_day_metrics_from_pattern(pattern)
        diffs = {
            k: abs(float(direct[k]) - float(cached[k]))
            for k in METRIC_ORDER
        }
        local_ok = all(v <= 1e-9 for v in diffs.values())
        ok_all = ok_all and local_ok
        messages.append(
            f"{year} {origin}->{destination}: "
            + ", ".join([f"{k} diff={diffs[k]:.3e}" for k in METRIC_ORDER])
            + f", ok={local_ok}"
        )

    return ok_all, messages


def _get_notebook_module_for_model(model: Any) -> Any:
    module_name = model.__class__.__module__
    module = sys.modules.get(module_name)
    if module is None:
        raise RuntimeError(f"Routing module not loaded for model class module '{module_name}'")
    required = (
        "seed_origin_source_edges",
        "_movement_increment",
        "reconstruct_node_path",
        "compress_event_path_to_legs",
        "simplify_legs_by_staying_on_train",
    )
    missing = [name for name in required if not hasattr(module, name)]
    if missing:
        raise RuntimeError(f"Routing module '{module_name}' missing required functions: {missing}")
    return module


class _OneToManyBatchRun:
    """
    One search run for (origin, query_min), with lazy per-destination path decode.
    """

    def __init__(
        self,
        *,
        model: Any,
        origin: str,
        query_min: int,
        max_legs: int,
        include_legs: bool = False,
    ) -> None:
        self.model = model
        self.origin = origin
        self.query_min = int(query_min)
        self.max_legs = int(max_legs)
        self.include_legs = bool(include_legs)
        self.module = _get_notebook_module_for_model(model)

        self._best_dest: dict[str, tuple[int, tuple[int, int, int, int]]] = {}
        self._predecessor: dict[int, tuple[Optional[int], Optional[str], Optional[str], Optional[str]]] = {}
        self._result_cache: dict[str, Optional[dict[str, Any]]] = {}
        self.reconstructed_destination_count = 0
        self.search_elapsed_sec = 0.0

        self._run_search()

    def _run_search(self) -> None:
        t0 = time.time()
        start_nodes = self.module.seed_origin_source_edges(self.model, self.origin, self.query_min)
        if not start_nodes:
            self.search_elapsed_sec = time.time() - t0
            return

        dist: dict[int, tuple[int, int, int, int]] = {}
        pq: list[tuple[tuple[int, int, int, int], int]] = []
        inf_cost = (float("inf"), float("inf"), float("inf"), float("inf"))

        for start_node in start_nodes:
            start_time = int(self.model.nodes[start_node].time)
            cost = (start_time, -start_time, 0, 0)
            if cost < dist.get(start_node, inf_cost):
                dist[start_node] = cost
                self._predecessor[start_node] = (None, "source", None, None)
                heapq.heappush(pq, (cost, start_node))

        while pq:
            current_cost, node_id = heapq.heappop(pq)
            if current_cost != dist.get(node_id):
                continue
            node = self.model.nodes[node_id]
            if node.kind == "an":
                station = str(node.station)
                if station not in self._best_dest:
                    self._best_dest[station] = (node_id, current_cost)

            for edge in self.model.adj.get(node_id, []):
                to_node = self.model.nodes[edge.to_node]
                arrival_time = int(to_node.time)
                first_dep_neg = current_cost[1]
                transfers = current_cost[2] + (1 if edge.edge_type == "transfer" else 0)
                movement_segments = current_cost[3] + self.module._movement_increment(
                    self.model, node_id, edge.to_node, edge.edge_type
                )
                new_cost = (arrival_time, first_dep_neg, transfers, movement_segments)
                if new_cost < dist.get(edge.to_node, inf_cost):
                    dist[edge.to_node] = new_cost
                    self._predecessor[edge.to_node] = (
                        node_id,
                        edge.edge_type,
                        edge.service_name,
                        edge.trip_instance_id,
                    )
                    heapq.heappush(pq, (new_cost, edge.to_node))

        self.search_elapsed_sec = time.time() - t0

    def result_for_destination(self, destination: str) -> Optional[dict[str, Any]]:
        if destination in self._result_cache:
            return self._result_cache[destination]

        best = self._best_dest.get(destination)
        if best is None:
            self._result_cache[destination] = None
            return None

        best_dest_node, best_dest_cost = best
        _, edge_path = self.module.reconstruct_node_path(self._predecessor, best_dest_node)
        legs = self.module.compress_event_path_to_legs(self.model, edge_path)
        legs = self.module.simplify_legs_by_staying_on_train(self.model, legs)
        if len(legs) > self.max_legs:
            self._result_cache[destination] = None
            return None

        first_dep = int(-best_dest_cost[1])
        arrival = int(best_dest_cost[0])
        travel_time = float(arrival - first_dep)
        rolling, dwell, transfer = _decompose_legs_with_rebalance(legs, travel_time)
        out = {
            "first_departure_time": float(first_dep),
            "arrival_time": float(arrival),
            "travel_time": float(travel_time),
            "rolling": float(rolling),
            "dwell": float(dwell),
            "transfer": float(transfer),
        }
        if self.include_legs:
            out["legs"] = legs
        self._result_cache[destination] = out
        self.reconstructed_destination_count += 1
        return out


def _solve_origin_row_one_to_many(
    *,
    year: str,
    row_idx: int,
    row_count: int,
    model: Any,
    origin: str,
    destinations: list[str],
    max_legs: int,
    include_legs: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    dests = list(dict.fromkeys(destinations))
    # Keep these sparse to avoid allocating thousands of empty lists/sets per row.
    patterns: dict[str, list[dict[str, Any]]] = {}
    seen_dep: dict[str, set[int]] = {}
    done: set[str] = set()

    from collections import defaultdict

    dep_times = [int(x) for x in model.station_departure_times.get(origin, [])]

    def _seed_min_for_query_min(query_min: int) -> Optional[int]:
        idx = bisect.bisect_left(dep_times, int(query_min))
        if idx >= len(dep_times):
            return None
        return int(dep_times[idx])

    pending: dict[int, set[str]] = defaultdict(set)
    init_seed_min = _seed_min_for_query_min(DAY_START_MIN)
    if init_seed_min is not None:
        pending[init_seed_min].update(dests)
    else:
        done.update(dests)

    stats = {
        "batch_runs": 0,
        "destination_queries": 0,
        "reconstructed": 0,
    }

    row_t0 = time.time()
    _log(
        f"{year}: row {row_idx}/{row_count} ({origin}) one-to-many warmup start; "
        f"pending_destinations={len(dests)} query_min={_to_hhmm(DAY_START_MIN)} "
        f"seed_min={_to_hhmm(init_seed_min) if init_seed_min is not None else 'none'}"
    )

    while pending:
        seed_min = min(pending.keys())
        current_dests = sorted(pending.pop(seed_min))
        if not current_dests:
            continue
        if seed_min >= WINDOW_END_MIN:
            done.update(current_dests)
            continue

        batch = _OneToManyBatchRun(
            model=model,
            origin=origin,
            query_min=int(seed_min),
            max_legs=max_legs,
            include_legs=bool(include_legs),
        )
        stats["batch_runs"] += 1

        for destination in current_dests:
            if destination in done:
                continue

            stats["destination_queries"] += 1
            result = batch.result_for_destination(destination)
            if result is None:
                done.add(destination)
                continue

            dep = int(result["first_departure_time"])
            arr = int(result["arrival_time"])

            if dep >= WINDOW_END_MIN:
                done.add(destination)
                continue

            destination_seen = seen_dep.setdefault(destination, set())
            if dep in destination_seen:
                next_q = dep + 1
                next_seed_min = _seed_min_for_query_min(next_q)
                if next_seed_min is None or next_seed_min >= WINDOW_END_MIN:
                    done.add(destination)
                else:
                    pending[next_seed_min].add(destination)
                continue

            destination_seen.add(dep)
            patterns.setdefault(destination, []).append(
                {
                    "dep": float(dep),
                    "arr": float(arr),
                    "travel_time": float(result["travel_time"]),
                    "rolling": float(result["rolling"]),
                    "dwell": float(result["dwell"]),
                    "transfer": float(result["transfer"]),
                    **({"legs": result.get("legs", [])} if include_legs else {}),
                }
            )

            next_q = dep + 1
            next_seed_min = _seed_min_for_query_min(next_q)
            if next_seed_min is None or next_seed_min >= WINDOW_END_MIN:
                done.add(destination)
            else:
                pending[next_seed_min].add(destination)

        stats["reconstructed"] += int(batch.reconstructed_destination_count)
        pending_total = sum(len(v) for v in pending.values())
        elapsed = time.time() - row_t0
        avg_batch = elapsed / max(1, stats["batch_runs"])
        eta_batches = len(pending) * avg_batch
        _log(
            f"{year}: row {row_idx}/{row_count} ({origin}) one-to-many batch "
            f"{stats['batch_runs']} seed_min={_to_hhmm(seed_min)} "
            f"dest_q={len(current_dests)} "
            f"batch_search={batch.search_elapsed_sec:.1f}s "
            f"reconstructed={batch.reconstructed_destination_count} "
            f"done={len(done)}/{len(dests)} pending_q={pending_total} "
            f"pending_seed_buckets={len(pending)} "
            f"eta_batch_only={eta_batches/60:.1f}m"
        )

    for recs in patterns.values():
        recs.sort(key=lambda x: x["dep"])

    return patterns, stats


def _patterns_identical(
    left: Optional[list[dict[str, float]]],
    right: Optional[list[dict[str, float]]],
    *,
    tol: float = 1e-9,
) -> tuple[bool, str]:
    if left is None and right is None:
        return True, "both-none"
    if left is None or right is None:
        return False, "one-none"
    if len(left) != len(right):
        return False, f"length {len(left)} != {len(right)}"

    int_keys = ("dep", "arr")
    float_keys = ("travel_time", "rolling", "dwell", "transfer")
    for idx, (lrec, rrec) in enumerate(zip(left, right)):
        for key in int_keys:
            if int(round(float(lrec[key]))) != int(round(float(rrec[key]))):
                return False, f"rec#{idx} key={key} {lrec[key]} != {rrec[key]}"
        for key in float_keys:
            if abs(float(lrec[key]) - float(rrec[key])) > tol:
                return False, f"rec#{idx} key={key} {lrec[key]} != {rrec[key]}"

    return True, "ok"


def _build_validation_pairs(
    station_order: list[str],
    available: set[str],
    sample_pairs: int,
) -> list[tuple[str, str]]:
    stations = [s for s in station_order if s in available]
    n = len(stations)
    if n < 2:
        return []

    idx_candidates = sorted(
        set(
            [
                0,
                1,
                2,
                n // 7,
                n // 5,
                n // 4,
                n // 3,
                n // 2,
                (2 * n) // 3,
                (3 * n) // 4,
                n - 3,
                n - 2,
                n - 1,
            ]
        )
    )
    idx_candidates = [i for i in idx_candidates if 0 <= i < n]

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for i in idx_candidates:
        j = (i * 37 + 17) % n
        origin = stations[i]
        destination = stations[j]
        if origin == destination:
            destination = stations[(j + 1) % n]
        pair = (origin, destination)
        if pair in seen or origin == destination:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= sample_pairs:
            return pairs

    for i in range(min(n, sample_pairs * 2)):
        origin = stations[i]
        destination = stations[-(i + 1)]
        pair = (origin, destination)
        if origin == destination or pair in seen:
            continue
        seen.add(pair)
        pairs.append(pair)
        if len(pairs) >= sample_pairs:
            return pairs

    return pairs


def _validate_one_to_many_exactness(
    *,
    year: str,
    model: Any,
    station_order: list[str],
    max_legs: int,
    sample_pairs: int,
) -> tuple[bool, list[str]]:
    available = set(model.available_stations())
    pairs = _build_validation_pairs(station_order, available, max(1, int(sample_pairs)))
    if not pairs:
        return False, [f"{year}: no usable OD pairs available for one-to-many validation"]

    messages: list[str] = []
    ok_all = True
    _log(f"{year}: one-to-many validation starting ({len(pairs)} sampled OD pairs)")
    for idx, (origin, destination) in enumerate(pairs, start=1):
        t_pair = time.time()
        _log(f"{year}: one-to-many validation pair {idx}/{len(pairs)} -> {origin}->{destination}")
        direct = _sample_two_hour_pattern(
            model=model,
            origin=origin,
            destination=destination,
            max_legs=max_legs,
            include_legs=False,
        )
        solved, stats = _solve_origin_row_one_to_many(
            year=year,
            row_idx=idx,
            row_count=len(pairs),
            model=model,
            origin=origin,
            destinations=[destination],
            max_legs=max_legs,
        )
        batched = solved.get(destination)
        same, reason = _patterns_identical(direct, batched)
        ok_all = ok_all and same
        messages.append(
            f"{year} {origin}->{destination}: ok={same} reason={reason} "
            f"batch_queries={stats['batch_runs']}"
        )
        _log(
            f"{year}: one-to-many validation pair {idx}/{len(pairs)} "
            f"ok={same} batch_queries={stats['batch_runs']} "
            f"elapsed={time.time()-t_pair:.1f}s"
        )

    return ok_all, messages


def _run_year(
    year: str,
    station_order: list[str],
    model: Any,
    anchor_set: set[str],
    anchor_cache: dict[tuple[str, str], dict[str, float]],
    state_root: Path,
    max_legs: int,
    checkpoint_every_rows: int,
    use_anchor_cache: bool,
    use_one_to_many: bool,
    output_tag: str,
    pair_progress_every: int,
    pair_heartbeat_sec: int,
    live_csv: bool,
    live_csv_every_rows: int,
    origin_start: int,
    origin_end: int,
    skip_final_write: bool,
    checkpoint_name: str,
    row_lock_dir: Optional[Path],
    row_modulo: int,
    row_remainder: int,
    write_nondominated_cells: bool,
    nondominated_live_csv: bool,
) -> dict[str, Any]:
    n = len(station_order)
    origin_start = max(0, int(origin_start))
    origin_end = min(n, int(origin_end))
    if origin_start >= origin_end:
        return {
            "year": year,
            "station_count": n,
            "origin_start": origin_start,
            "origin_end": origin_end,
            "counts": {
                "pairs_total": 0,
                "pairs_same_station": 0,
                "pairs_unavailable": 0,
                "pairs_anchor_cache": 0,
                "pairs_batched_query": 0,
                "pairs_one_to_many_fallback_direct": 0,
                "pairs_direct_query": 0,
                "pairs_no_pattern": 0,
                "one_to_many_batch_runs": 0,
                "one_to_many_destination_lookups": 0,
                "one_to_many_reconstructed": 0,
                "rows_skipped_precomputed": 0,
                "rows_skipped_locked": 0,
            },
            "elapsed_sec": 0.0,
            "skipped_empty_range": True,
        }

    next_origin_idx, counts, matrices = _load_or_init_year_state(
        year=year,
        n=n,
        station_order=station_order,
        state_root=state_root,
        origin_start=origin_start,
        origin_end=origin_end,
        checkpoint_name=checkpoint_name,
    )
    if next_origin_idx < origin_start:
        next_origin_idx = origin_start

    nondom_rows_dir = _nondominated_rows_dir(state_root, year) if write_nondominated_cells else None
    if nondom_rows_dir is not None:
        nondom_rows_dir.mkdir(parents=True, exist_ok=True)

    # If checkpoint metadata lags behind already-computed memmap rows (e.g. crash
    # after row write but before checkpoint update), skip those rows to avoid
    # recomputation and wasted CPU.
    advance_from = int(next_origin_idx)
    while next_origin_idx < origin_end:
        freq_row = np.asarray(matrices["frequency"][next_origin_idx, :], dtype=np.float64)
        if np.isnan(freq_row).any():
            break
        if write_nondominated_cells:
            if not _nondominated_row_exists(state_root, year, next_origin_idx):
                break
        next_origin_idx += 1
    if next_origin_idx > advance_from:
        advanced = next_origin_idx - advance_from
        counts["rows_skipped_precomputed"] += int(advanced)
        _log(
            f"{year}: advanced checkpoint cursor by {advanced} precomputed rows "
            f"({advance_from} -> {next_origin_idx}) based on frequency memmap completeness."
        )

    available = set(model.available_stations())
    live_paths: dict[str, Path] = {}
    next_live_row = int(next_origin_idx)
    live_every = max(1, int(live_csv_every_rows))
    nondom_live_path: Optional[Path] = None
    nondom_next_live_row = int(next_origin_idx)
    if live_csv and not skip_final_write:
        live_paths = _init_live_csv_files(
            year=year,
            station_order=station_order,
            matrices=matrices,
            output_tag=output_tag,
            existing_rows=next_origin_idx,
        )
        sample_live = next(iter(live_paths.values()))
        _log(f"{year}: live CSV progress enabled ({sample_live.name} etc), update_every_rows={live_every}")
    elif live_csv and skip_final_write:
        _log(f"{year}: live CSV progress disabled for skip-final-write mode.")
    if write_nondominated_cells and nondominated_live_csv:
        nondom_live_path = _init_nondominated_live_csv_file(
            year=year,
            station_order=station_order,
            state_root=state_root,
            existing_rows=next_origin_idx,
        )
        _log(
            f"{year}: nondominated live CSV enabled ({nondom_live_path.name}), "
            f"update_every_rows={live_every}"
        )

    row_modulo = max(1, int(row_modulo))
    row_remainder = int(row_remainder) % row_modulo

    t0 = time.time()
    rows_done_start = max(next_origin_idx, origin_start)
    next_cursor = int(next_origin_idx)
    paused_on_row_lock = False
    paused_row_idx: Optional[int] = None

    for i in range(next_origin_idx, origin_end):
        if row_modulo > 1 and (i % row_modulo) != row_remainder:
            next_cursor = i + 1
            continue
        origin = station_order[i]
        # Idempotency safeguard: if this row is already complete in memmap,
        # do not recompute it.
        freq_row = np.asarray(matrices["frequency"][i, :], dtype=np.float64)
        row_payload_exists = (not write_nondominated_cells) or _nondominated_row_exists(state_root, year, i)
        if not np.isnan(freq_row).any() and row_payload_exists:
            counts["rows_skipped_precomputed"] += 1
            _log(f"{year}: row {i + 1}/{origin_end} ({origin}) already complete; skipping recompute.")
            if checkpoint_every_rows > 0 and ((i + 1) % checkpoint_every_rows == 0):
                _flush_matrices(matrices)
                _save_checkpoint(
                    year=year,
                    station_order=station_order,
                    origin_start=origin_start,
                    origin_end=origin_end,
                    next_origin_idx=i + 1,
                    counts=counts,
                    state_root=state_root,
                    checkpoint_name=checkpoint_name,
                )
            if live_paths and (((i + 1) % live_every) == 0 or (i + 1) == origin_end):
                _append_live_csv_rows(
                    live_paths=live_paths,
                    station_order=station_order,
                    matrices=matrices,
                    row_start=next_live_row,
                    row_end=i + 1,
                )
                next_live_row = i + 1
            if nondom_live_path is not None and (((i + 1) % live_every) == 0 or (i + 1) == origin_end):
                _append_nondominated_live_csv_rows(
                    live_path=nondom_live_path,
                    station_order=station_order,
                    state_root=state_root,
                    year=year,
                    row_start=nondom_next_live_row,
                    row_end=i + 1,
                )
                nondom_next_live_row = i + 1
            next_cursor = i + 1
            continue

        row_lock_handle: Optional[Any] = None
        if row_lock_dir is not None:
            row_lock_handle, row_lock_path = _try_acquire_row_lock(
                row_lock_dir=row_lock_dir,
                year=year,
                row_idx=i,
                origin=origin,
                pid=os.getpid(),
            )
            if row_lock_handle is None:
                counts["rows_skipped_locked"] += 1
                paused_on_row_lock = True
                paused_row_idx = i
                _log(
                    f"{year}: row {i + 1}/{origin_end} ({origin}) locked by another process "
                    f"({row_lock_path.name}); pausing this worker at row boundary to avoid duplicate row starts."
                )
                _flush_matrices(matrices)
                _save_checkpoint(
                    year=year,
                    station_order=station_order,
                    origin_start=origin_start,
                    origin_end=origin_end,
                    next_origin_idx=i,
                    counts=counts,
                    state_root=state_root,
                    checkpoint_name=checkpoint_name,
                )
                break

        row_writer: Optional[_NondominatedRowWriter] = None
        try:
            row_t0 = time.time()
            pair_t0 = time.time()
            last_pair_log_t = pair_t0
            row_batch_patterns: dict[str, list[dict[str, Any]]] = {}
            row_batch_stats = {
                "batch_runs": 0,
                "destination_queries": 0,
                "reconstructed": 0,
            }
            row_batch_error: Optional[Exception] = None
            row_batch_destinations: list[str] = []
            if write_nondominated_cells:
                row_writer = _NondominatedRowWriter(
                    path=_nondominated_row_path_gz(state_root, year, i),
                    row_idx=i,
                    origin=origin,
                    year=year,
                )
            if use_one_to_many and origin in available:
                for destination in station_order:
                    if origin == destination:
                        continue
                    if destination not in available:
                        continue
                    if use_anchor_cache and origin in anchor_set and destination in anchor_set:
                        cached = anchor_cache.get((origin, destination))
                        if cached is not None:
                            continue
                    row_batch_destinations.append(destination)
                if row_batch_destinations:
                    try:
                        row_batch_patterns, row_batch_stats = _solve_origin_row_one_to_many(
                            year=year,
                            row_idx=i + 1,
                            row_count=origin_end,
                            model=model,
                            origin=origin,
                            destinations=row_batch_destinations,
                            max_legs=max_legs,
                            include_legs=bool(write_nondominated_cells),
                        )
                    except Exception as exc:
                        row_batch_error = exc
                        _log(
                            f"{year}: row {i + 1}/{origin_end} ({origin}) one-to-many row solve failed; "
                            f"falling back to direct per-destination queries. error={exc}"
                        )

            for j, destination in enumerate(station_order):
                counts["pairs_total"] += 1
                pattern_for_payload: Optional[list[dict[str, Any]]] = None

                if origin == destination:
                    metrics = _zero_metrics_same_station()
                    counts["pairs_same_station"] += 1
                elif origin not in available or destination not in available:
                    metrics = _empty_metrics_unreachable()
                    counts["pairs_unavailable"] += 1
                elif use_anchor_cache and origin in anchor_set and destination in anchor_set:
                    cached = anchor_cache.get((origin, destination))
                    if cached is not None:
                        metrics = cached
                        counts["pairs_anchor_cache"] += 1
                        if write_nondominated_cells:
                            pattern_for_payload = _sample_two_hour_pattern(
                                model=model,
                                origin=origin,
                                destination=destination,
                                max_legs=max_legs,
                                include_legs=True,
                            )
                            counts["pairs_direct_query"] += 1
                    else:
                        pattern = _sample_two_hour_pattern(
                            model=model,
                            origin=origin,
                            destination=destination,
                            max_legs=max_legs,
                            include_legs=bool(write_nondominated_cells),
                        )
                        pattern_for_payload = pattern
                        if not pattern:
                            metrics = _empty_metrics_unreachable()
                            counts["pairs_no_pattern"] += 1
                        else:
                            metrics = _compute_day_metrics_from_pattern(pattern)
                        counts["pairs_direct_query"] += 1
                elif use_one_to_many and origin in available and destination in available:
                    if row_batch_error is None and destination in row_batch_patterns:
                        pattern = row_batch_patterns[destination]
                        pattern_for_payload = pattern
                        if not pattern:
                            metrics = _empty_metrics_unreachable()
                            counts["pairs_no_pattern"] += 1
                        else:
                            metrics = _compute_day_metrics_from_pattern(pattern)
                        counts["pairs_batched_query"] += 1
                    else:
                        pattern = _sample_two_hour_pattern(
                            model=model,
                            origin=origin,
                            destination=destination,
                            max_legs=max_legs,
                            include_legs=bool(write_nondominated_cells),
                        )
                        pattern_for_payload = pattern
                        if not pattern:
                            metrics = _empty_metrics_unreachable()
                            counts["pairs_no_pattern"] += 1
                        else:
                            metrics = _compute_day_metrics_from_pattern(pattern)
                        counts["pairs_direct_query"] += 1
                        counts["pairs_one_to_many_fallback_direct"] += 1
                        fb = counts["pairs_one_to_many_fallback_direct"]
                        if fb <= 5 or (fb % 100) == 0:
                            _log(
                                f"{year}: one-to-many fallback to direct for {origin}->{destination} "
                                f"(count={fb}, error={row_batch_error})"
                            )
                else:
                    pattern = _sample_two_hour_pattern(
                        model=model,
                        origin=origin,
                        destination=destination,
                        max_legs=max_legs,
                        include_legs=bool(write_nondominated_cells),
                    )
                    pattern_for_payload = pattern
                    if not pattern:
                        metrics = _empty_metrics_unreachable()
                        counts["pairs_no_pattern"] += 1
                    else:
                        metrics = _compute_day_metrics_from_pattern(pattern)
                    counts["pairs_direct_query"] += 1

                for metric in METRIC_ORDER:
                    matrices[metric][i, j] = float(metrics[metric])

                if write_nondominated_cells:
                    cell_payload = _pattern_to_nondominated_cell_payload(
                        origin=origin,
                        destination=destination,
                        pattern=pattern_for_payload,
                    )
                    row_writer.add_cell(
                        json.dumps(
                            cell_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                    )

                now_t = time.time()
                should_log_count = pair_progress_every > 0 and ((j + 1) % pair_progress_every == 0 or (j + 1) == n)
                should_log_time = pair_heartbeat_sec > 0 and ((now_t - last_pair_log_t) >= pair_heartbeat_sec)
                if should_log_count or should_log_time:
                    pair_elapsed = time.time() - pair_t0
                    _log(
                        f"{year}: row {i + 1}/{n} ({origin}) "
                        f"dest {j + 1}/{n} "
                        f"chunk_time={pair_elapsed:.1f}s "
                        f"direct={counts['pairs_direct_query']} "
                        f"batched={counts['pairs_batched_query']} "
                        f"fallback={counts['pairs_one_to_many_fallback_direct']} "
                        f"batch_runs={counts['one_to_many_batch_runs'] + int(row_batch_stats.get('batch_runs', 0))} "
                        f"cache_anchor={counts['pairs_anchor_cache']}"
                    )
                    pair_t0 = time.time()
                    last_pair_log_t = pair_t0

            if write_nondominated_cells:
                row_writer.close(completed_at=_ts())
                saved_size_mb = float("nan")
                try:
                    saved_size_mb = float(row_writer.path.stat().st_size) / (1024.0 * 1024.0)
                except Exception:
                    pass
                _log(
                    f"{year}: row {i + 1}/{origin_end} ({origin}) nondominated row saved "
                    f"file={row_writer.path.name} size_mb={saved_size_mb:.2f}"
                )

            if use_one_to_many and row_batch_destinations:
                counts["one_to_many_batch_runs"] += int(row_batch_stats.get("batch_runs", 0))
                counts["one_to_many_destination_lookups"] += int(row_batch_stats.get("destination_queries", 0))
                counts["one_to_many_reconstructed"] += int(row_batch_stats.get("reconstructed", 0))

            row_elapsed = time.time() - row_t0

            if checkpoint_every_rows > 0 and ((i + 1) % checkpoint_every_rows == 0):
                _flush_matrices(matrices)
                _save_checkpoint(
                    year=year,
                    station_order=station_order,
                    origin_start=origin_start,
                    origin_end=origin_end,
                    next_origin_idx=i + 1,
                    counts=counts,
                    state_root=state_root,
                    checkpoint_name=checkpoint_name,
                )
                _log(
                    f"{year}: checkpoint saved next_origin_idx={i + 1} "
                    f"checkpoint={checkpoint_name}"
                )

            rows_done = (i + 1) - rows_done_start
            elapsed = time.time() - t0
            rows_left = origin_end - (i + 1)
            avg_row_sec = elapsed / max(rows_done, 1)
            eta_sec = rows_left * avg_row_sec
            _log(
                f"{year}: row {i + 1}/{origin_end} ({origin}) "
                f"row_time={row_elapsed:.1f}s avg_row={avg_row_sec:.1f}s "
                f"eta={eta_sec/3600:.1f}h "
                f"direct={counts['pairs_direct_query']} "
                f"batched={counts['pairs_batched_query']} "
                f"fallback={counts['pairs_one_to_many_fallback_direct']} "
                f"batch_runs={counts['one_to_many_batch_runs']} "
                f"cache_anchor={counts['pairs_anchor_cache']}"
            )

            # Keep per-row memory bounded during long sharded runs.
            row_batch_patterns.clear()
            row_batch_destinations.clear()
            gc.collect()

            if live_paths and (((i + 1) % live_every) == 0 or (i + 1) == origin_end):
                _append_live_csv_rows(
                    live_paths=live_paths,
                    station_order=station_order,
                    matrices=matrices,
                    row_start=next_live_row,
                    row_end=i + 1,
                )
                next_live_row = i + 1
            if nondom_live_path is not None and (((i + 1) % live_every) == 0 or (i + 1) == origin_end):
                _append_nondominated_live_csv_rows(
                    live_path=nondom_live_path,
                    station_order=station_order,
                    state_root=state_root,
                    year=year,
                    row_start=nondom_next_live_row,
                    row_end=i + 1,
                )
                nondom_next_live_row = i + 1
            next_cursor = i + 1
        finally:
            if row_writer is not None and not row_writer.closed:
                try:
                    row_writer.fh.close()
                except Exception:
                    pass
                try:
                    if row_writer.tmp_path.exists():
                        row_writer.tmp_path.unlink()
                except Exception:
                    pass
            if row_lock_handle is not None:
                try:
                    row_lock_handle.close()
                except Exception:
                    pass

    _flush_matrices(matrices)
    _save_checkpoint(
        year=year,
        station_order=station_order,
        origin_start=origin_start,
        origin_end=origin_end,
        next_origin_idx=next_cursor,
        counts=counts,
        state_root=state_root,
        checkpoint_name=checkpoint_name,
    )

    if not skip_final_write and next_cursor >= origin_end:
        _write_full_csvs_for_year(
            year=year,
            station_order=station_order,
            matrices=matrices,
            output_tag=output_tag,
        )
        if write_nondominated_cells:
            out_cells = _write_nondominated_full_csv(
                year=year,
                station_order=station_order,
                state_root=state_root,
            )
            if nondom_live_path is not None:
                _write_nondominated_live_csv_rows(
                    path=nondom_live_path,
                    station_order=station_order,
                    state_root=state_root,
                    year=year,
                    row_start=0,
                    row_end=len(station_order),
                    include_header=True,
                )
            _log(f"{year}: wrote {out_cells.name}")
    elif not skip_final_write:
        _log(
            f"{year}: final CSV write skipped because run paused early at row index {next_cursor} "
            f"(origin_end={origin_end}). Resume to complete before writing final outputs."
        )

    summary = {
        "year": year,
        "station_count": n,
        "origin_start": origin_start,
        "origin_end": origin_end,
        "counts": counts,
        "elapsed_sec": time.time() - t0,
        "skip_final_write": bool(skip_final_write),
        "checkpoint_name": checkpoint_name,
        "next_origin_idx": int(next_cursor),
        "paused_on_row_lock": bool(paused_on_row_lock),
        "paused_row_idx": (int(paused_row_idx) if paused_row_idx is not None else None),
        "row_modulo": int(row_modulo),
        "row_remainder": int(row_remainder),
        "write_nondominated_cells": bool(write_nondominated_cells),
        "nondominated_live_csv": bool(nondominated_live_csv),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build full OD metric matrices for 2026/2035 with robust checkpoints, exact anchor-cache shortcuts, and exact one-to-many row acceleration."
    )
    parser.add_argument(
        "--years",
        default="2026,2035",
        help="Comma-separated list from: 2026,2035",
    )
    parser.add_argument(
        "--max-legs",
        type=int,
        default=50,
        help="Max movement legs per route query.",
    )
    parser.add_argument(
        "--checkpoint-every-rows",
        type=int,
        default=1,
        help="Persist checkpoint every N completed origin rows.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(os.environ.get("OD_MODEL_CACHE", str(BASE_DIR / "od_model_cache_2026_2035.pkl"))),
        help="Path for serialized models cache.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=BASE_DIR / "full_od_build_state",
        help="Directory for resumable state (memmaps + checkpoints).",
    )
    parser.add_argument(
        "--no-anchor-cache",
        action="store_true",
        help="Disable anchor-cache shortcut and force direct queries for all valid pairs.",
    )
    parser.add_argument(
        "--one-to-many",
        action="store_true",
        help="Enable exact one-to-many row acceleration (disabled by default).",
    )
    parser.add_argument(
        "--no-one-to-many",
        action="store_true",
        help="Disable exact one-to-many row acceleration and run destination queries independently.",
    )
    parser.add_argument(
        "--one-to-many-validate-pairs",
        type=int,
        default=6,
        help="Number of direct-vs-batched OD pairs checked before enabling one-to-many acceleration.",
    )
    parser.add_argument(
        "--skip-one-to-many-validation",
        action="store_true",
        help="Skip direct-vs-batched validation warmup and immediately enable one-to-many (faster startup).",
    )
    parser.add_argument(
        "--limit-stations",
        type=int,
        default=0,
        help="Debug option: only use first N stations from FilteredStations2026_2035.csv; 0 means all.",
    )
    parser.add_argument(
        "--output-tag",
        default="",
        help="Optional suffix inserted before .csv in output files (example: _test).",
    )
    parser.add_argument(
        "--pair-progress-every",
        type=int,
        default=0,
        help="Optional intra-row progress: print every N destination pairs; 0 disables.",
    )
    parser.add_argument(
        "--pair-heartbeat-sec",
        type=int,
        default=120,
        help="Time-based intra-row progress heartbeat in seconds; 0 disables.",
    )
    parser.add_argument(
        "--live-csv",
        action="store_true",
        help="Create *_LIVE.csv files up front and append completed rows during the run.",
    )
    parser.add_argument(
        "--live-csv-every-rows",
        type=int,
        default=1,
        help="When --live-csv is enabled, append to live CSVs every N completed rows.",
    )
    parser.add_argument(
        "--anchor-first",
        action="store_true",
        help="Reorder rows/columns so anchors are listed first, then all other stations sorted by DTV (descending).",
    )
    parser.add_argument(
        "--origin-start",
        type=int,
        default=0,
        help="Optional origin row start index (inclusive). Default 0.",
    )
    parser.add_argument(
        "--origin-end",
        type=int,
        default=0,
        help="Optional origin row end index (exclusive). Default 0 means full length.",
    )
    parser.add_argument(
        "--skip-final-write",
        action="store_true",
        help="Process and checkpoint rows but do not emit final CSVs (for sharded orchestration).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint.pkl",
        help="Checkpoint file name inside the year state dir.",
    )
    parser.add_argument(
        "--row-lock-dir",
        type=Path,
        default=None,
        help="Optional directory for per-row advisory locks. If set, a row cannot be started concurrently by another process.",
    )
    parser.add_argument(
        "--row-modulo",
        type=int,
        default=1,
        help="Optional row-assignment modulo. Example with --row-modulo 4 and --row-remainder 1: process rows i where i %% 4 == 1.",
    )
    parser.add_argument(
        "--row-remainder",
        type=int,
        default=0,
        help="Remainder used with --row-modulo for deterministic interleaved row assignment.",
    )
    parser.add_argument(
        "--write-nondominated-cells",
        action="store_true",
        help=(
            "Also write a single OD CSV where each cell stores all nondominated paths "
            "between 05:00 and 06:59 with detailed leg/station/transfer timing."
        ),
    )
    parser.add_argument(
        "--nondominated-live-csv",
        action="store_true",
        help="When --write-nondominated-cells is enabled, also maintain *_Nondominated_paths_full_LIVE.csv during runtime.",
    )
    args = parser.parse_args()

    selected_years = [y.strip() for y in str(args.years).split(",") if y.strip()]
    for y in selected_years:
        if y not in ("2026", "2035"):
            raise ValueError(f"Unsupported year in --years: {y}")

    station_table = _load_station_table()
    if args.limit_stations > 0:
        station_table = station_table.iloc[: args.limit_stations].copy()
    station_order = station_table["StationName"].astype(str).tolist()
    dtv_by_station = {
        str(row.StationName): float(row.DTV) if pd.notna(row.DTV) else float("-inf")
        for row in station_table.itertuples(index=False)
    }
    origin_start = max(0, int(args.origin_start))
    origin_end = int(args.origin_end) if int(args.origin_end) > 0 else len(station_order)
    origin_end = min(len(station_order), origin_end)
    output_tag = str(args.output_tag or "")
    if args.limit_stations > 0 and output_tag == "":
        output_tag = f"_LIMIT{args.limit_stations}"
    models = _load_models_with_cache(args.model_cache)
    args.state_dir.mkdir(parents=True, exist_ok=True)
    if args.row_lock_dir is not None:
        args.row_lock_dir.mkdir(parents=True, exist_ok=True)
    state_lock = _acquire_state_lock(args.state_dir, selected_years)

    try:
        if bool(args.anchor_first):
            anchor_union: set[str] = set()
            for year in selected_years:
                try:
                    anchors, _ = _load_anchor_cache_for_year(year)
                    anchor_union.update(anchors)
                except Exception:
                    pass
            station_order = _reorder_anchor_then_dtv(station_order, anchor_union, dtv_by_station)
            _log(
                f"Station order switched to anchor-first: anchors={sum(1 for s in station_order if s in anchor_union)} "
                f"non_anchors={sum(1 for s in station_order if s not in anchor_union)} (sorted by DTV desc)"
            )

        run_summary: list[dict[str, Any]] = []
        write_nondominated_cells = bool(args.write_nondominated_cells)
        nondominated_live_csv = bool(args.nondominated_live_csv) and write_nondominated_cells
        for year in selected_years:
            if year not in models:
                _log(f"{year}: model unavailable, skipping.")
                continue

            anchor_set, anchor_cache = _load_anchor_cache_for_year(year)
            use_anchor_cache = not args.no_anchor_cache
            if write_nondominated_cells and use_anchor_cache:
                _log(f"{year}: disabling anchor cache because detailed nondominated cell output requires explicit path reconstruction.")
                use_anchor_cache = False
            if use_anchor_cache:
                valid, checks = _validate_anchor_cache_exactness(
                    year=year,
                    model=models[year],
                    anchor_set=anchor_set,
                    anchor_cache=anchor_cache,
                    max_legs=args.max_legs,
                )
                for line in checks:
                    _log(f"anchor-check: {line}")
                if not valid:
                    _log(f"{year}: anchor cache validation failed; disabling anchor-cache shortcut for safety.")
                    use_anchor_cache = False

            use_one_to_many = bool(args.one_to_many) and (not args.no_one_to_many)
            if use_one_to_many:
                if bool(args.skip_one_to_many_validation):
                    _log(f"{year}: skipping one-to-many validation warmup (--skip-one-to-many-validation).")
                else:
                    valid, checks = _validate_one_to_many_exactness(
                        year=year,
                        model=models[year],
                        station_order=station_order,
                        max_legs=args.max_legs,
                        sample_pairs=max(1, int(args.one_to_many_validate_pairs)),
                    )
                    for line in checks:
                        _log(f"1toN-check: {line}")
                    if not valid:
                        _log(f"{year}: one-to-many validation failed; disabling one-to-many shortcut for safety.")
                        use_one_to_many = False

            _log(
                f"{year}: starting full matrix build for {len(station_order)} stations "
                f"({len(station_order)}x{len(station_order)} pairs), origins [{origin_start}:{origin_end}). "
                f"anchor_cache_enabled={use_anchor_cache}, one_to_many_enabled={use_one_to_many}, "
                f"row_modulo={max(1, int(args.row_modulo))}, row_remainder={int(args.row_remainder) % max(1, int(args.row_modulo))}, "
                f"write_nondominated_cells={int(write_nondominated_cells)}, nondominated_live_csv={int(nondominated_live_csv)}"
            )

            summary = _run_year(
                year=year,
                station_order=station_order,
                model=models[year],
                anchor_set=anchor_set,
                anchor_cache=anchor_cache,
                state_root=args.state_dir,
                max_legs=args.max_legs,
                checkpoint_every_rows=max(1, int(args.checkpoint_every_rows)),
                use_anchor_cache=use_anchor_cache,
                use_one_to_many=use_one_to_many,
                output_tag=output_tag,
                pair_progress_every=max(0, int(args.pair_progress_every)),
                pair_heartbeat_sec=max(0, int(args.pair_heartbeat_sec)),
                live_csv=bool(args.live_csv),
                live_csv_every_rows=max(1, int(args.live_csv_every_rows)),
                origin_start=origin_start,
                origin_end=origin_end,
                skip_final_write=bool(args.skip_final_write),
                checkpoint_name=str(args.checkpoint_name),
                row_lock_dir=args.row_lock_dir,
                row_modulo=max(1, int(args.row_modulo)),
                row_remainder=int(args.row_remainder),
                write_nondominated_cells=write_nondominated_cells,
                nondominated_live_csv=nondominated_live_csv,
            )
            run_summary.append(summary)
            _log(f"{year}: full matrix processing finished for requested origin slice.")

        summary_out = BASE_DIR / "full_od_build_summary_2026_2035.json"
        summary_out.write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _log(f"Wrote summary: {summary_out}")
        _write_method_summary()
        _log(f"Wrote method note: {FULL_OD_METHOD_FILE}")
        if ANCHOR_CACHE_METHOD_FILE.exists():
            _log(f"Reference method: {ANCHOR_CACHE_METHOD_FILE.name}")
    finally:
        try:
            state_lock.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
