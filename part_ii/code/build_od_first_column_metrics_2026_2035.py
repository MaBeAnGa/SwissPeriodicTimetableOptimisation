#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from comparison_system_2026_2035 import build_second_half_system


BASE_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_CSV = Path(
    os.environ.get("OD_FILTERED_STATIONS_CSV", str(BASE_DIR / "FilteredStations2026_2035.csv"))
).expanduser()
TIMETABLE_CSV = Path(
    os.environ.get("OD_TIMETABLE_CSV", str(BASE_DIR / "FullTimetableHistory2026_2035.csv"))
).expanduser()
NOTEBOOK_IPYNB = BASE_DIR.parent / "Master Thesis Notebook.ipynb"
THIS_SCRIPT = Path(__file__).resolve()

DAY_START_MIN = 5 * 60   # 05:00
WINDOW_END_MIN = 7 * 60  # 07:00 (exclusive for the two-hour pattern window)
DAY_END_MIN = 24 * 60    # 00:00
DAY_DURATION_MIN = DAY_END_MIN - DAY_START_MIN  # 1140
ZERO_MIN_REPLACEMENT = 1.0 / 3.0  # 20 seconds in minutes


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


class _TimestampedWriter:
    def __init__(self, stream: Any):
        self.stream = stream
        self._buf = ""

    def write(self, s: str) -> int:
        if not isinstance(s, str):
            s = str(s)
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.stream.write(f"[{_ts()}] {line}\n")
        self.stream.flush()
        return len(s)

    def flush(self) -> None:
        if self._buf:
            self.stream.write(f"[{_ts()}] {self._buf}")
            self._buf = ""
        self.stream.flush()


def _to_hhmm(minutes: int) -> str:
    h = (minutes // 60) % 24
    m = minutes % 60
    return f"{h:02d}:{m:02d}"


def _decompose_legs_with_rebalance(legs: List[dict], travel_time_min: float) -> Tuple[float, float, float]:
    if not legs:
        return 0.0, 0.0, 0.0

    rolling_segments: List[float] = []
    dwell_segments: List[float] = []
    transfer_segments: List[float] = []

    for leg in legs:
        seg = max(float(leg["arr_time"]) - float(leg["dep_time"]), 0.0)
        rolling_segments.append(seg)

    for i in range(len(legs) - 1):
        gap = max(float(legs[i + 1]["dep_time"]) - float(legs[i]["arr_time"]), 0.0)
        if legs[i]["canonical_trip_id"] == legs[i + 1]["canonical_trip_id"]:
            dwell_segments.append(gap)
        else:
            transfer_segments.append(gap)

    rolling_raw = float(sum(rolling_segments))
    dwell_raw = float(sum(dwell_segments))
    transfer_raw = float(sum(transfer_segments))

    zero_roll = sum(1 for v in rolling_segments if v == 0.0)
    zero_dwell = sum(1 for v in dwell_segments if v == 0.0)

    rolling_adj = rolling_raw + zero_roll * ZERO_MIN_REPLACEMENT
    dwell_adj = dwell_raw + zero_dwell * ZERO_MIN_REPLACEMENT
    transfer_adj = transfer_raw

    raw_total = rolling_raw + dwell_raw + transfer_raw
    if raw_total == 0.0:
        return 0.0, 0.0, 0.0

    # Preserve the original end-to-end time exactly by removing the extra time
    # introduced by the 20-second minimums from other positive components.
    target_total = float(travel_time_min)
    current_total = rolling_adj + dwell_adj + transfer_adj
    excess = current_total - target_total

    if excess > 1e-12:
        pool_rd = rolling_raw + dwell_raw
        if pool_rd > 0.0:
            rolling_adj -= excess * (rolling_raw / pool_rd)
            dwell_adj -= excess * (dwell_raw / pool_rd)
        elif transfer_adj > 0.0:
            transfer_adj = max(0.0, transfer_adj - excess)

    # Numerical safety.
    rolling_adj = max(0.0, rolling_adj)
    dwell_adj = max(0.0, dwell_adj)
    transfer_adj = max(0.0, transfer_adj)

    # Final tiny balancing step to keep exact sum.
    final_total = rolling_adj + dwell_adj + transfer_adj
    diff = target_total - final_total
    if abs(diff) > 1e-9:
        if rolling_adj >= dwell_adj and rolling_adj > 0:
            rolling_adj = max(0.0, rolling_adj + diff)
        elif dwell_adj > 0:
            dwell_adj = max(0.0, dwell_adj + diff)
        else:
            transfer_adj = max(0.0, transfer_adj + diff)

    return rolling_adj, dwell_adj, transfer_adj


def _sample_two_hour_pattern(
    model: Any,
    origin: str,
    destination: str,
    max_legs: int,
    include_legs: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    query_min = DAY_START_MIN
    departures: List[Dict[str, Any]] = []
    seen_departure_times: set[int] = set()

    # n+1 strategy: each query jumps to one minute after the observed departure.
    while True:
        result = model.query(origin, destination, departure_time=_to_hhmm(query_min), max_legs=max_legs)
        if result is None:
            return None if not departures else departures

        dep = int(result["first_departure_time"])
        arr = int(result["arrival_time"])

        if dep >= WINDOW_END_MIN:
            break

        if dep in seen_departure_times:
            query_min = dep + 1
            if query_min >= WINDOW_END_MIN:
                break
            continue

        seen_departure_times.add(dep)
        travel_time = float(arr - dep)
        rolling, dwell, transfer = _decompose_legs_with_rebalance(result["legs"], travel_time)

        departures.append(
            {
                "dep": dep,
                "arr": arr,
                "travel_time": travel_time,
                "rolling": rolling,
                "dwell": dwell,
                "transfer": transfer,
                **({"legs": result["legs"]} if include_legs else {}),
            }
        )

        query_min = dep + 1
        if query_min >= WINDOW_END_MIN:
            # one extra query after cutoff is conceptually required to confirm
            # the next departure is beyond the two-hour window; the loop top
            # performs that check with dep >= WINDOW_END_MIN.
            pass

    if not departures:
        return None

    departures.sort(key=lambda x: x["dep"])
    return departures


def _build_station_components(model: Any) -> Dict[str, int]:
    adjacency: Dict[str, set[str]] = defaultdict(set)
    stations = set(model.available_stations())

    # Use full graph edges (service + transfer). A service-only view would
    # incorrectly separate transfer-only interchange labels (e.g., Zürich HB
    # aliases), causing false "disconnected" skips.
    for from_node_id, edges in model.adj.items():
        from_station = str(model.nodes[from_node_id].station)
        stations.add(from_station)
        for edge in edges:
            to_station = str(model.nodes[edge.to_node].station)
            stations.add(to_station)
            if from_station == to_station:
                continue
            adjacency[from_station].add(to_station)
            adjacency[to_station].add(from_station)

    component_of: Dict[str, int] = {}
    component_id = 0
    for station in sorted(stations):
        if station in component_of:
            continue
        component_id += 1
        stack = [station]
        component_of[station] = component_id
        while stack:
            cur = stack.pop()
            for nxt in adjacency.get(cur, set()):
                if nxt in component_of:
                    continue
                component_of[nxt] = component_id
                stack.append(nxt)
    return component_of


def _compute_day_metrics_from_pattern(pattern: List[Dict[str, Any]]) -> Dict[str, float]:
    # Evaluate dominant departure per minute within one 2-hour cycle.
    cycle_assignment_idx: List[int] = [-1] * 120
    cycle_wait: List[float] = [math.nan] * 120

    candidates: List[Tuple[int, int, int]] = []  # (dep_abs, arr_abs, base_idx)
    for idx, rec in enumerate(pattern):
        dep = int(rec["dep"])
        arr = int(rec["arr"])
        candidates.append((dep, arr, idx))
        candidates.append((dep + 120, arr + 120, idx))

    for minute in range(DAY_START_MIN, WINDOW_END_MIN):
        best: Optional[Tuple[int, int, int]] = None
        for dep_abs, arr_abs, base_idx in candidates:
            if dep_abs < minute:
                continue
            key = (arr_abs, dep_abs, base_idx)
            if best is None or key < best:
                best = key
        if best is None:
            continue
        arr_abs, dep_abs, base_idx = best
        offset = minute - DAY_START_MIN
        cycle_assignment_idx[offset] = base_idx
        cycle_wait[offset] = float(dep_abs - minute)

    # Per user requirement: wait/rolling/dwell/transfer are averaged strictly
    # on the 120-minute pattern window only (05:00-07:00), independent of
    # end-of-day cutoff effects. Cutoffs influence frequency only.
    weight_minutes = [0.0 for _ in pattern]
    wait_total = 0.0
    total_minutes = 120.0

    for offset in range(120):
        idx = cycle_assignment_idx[offset]
        if idx < 0:
            continue
        weight_minutes[idx] += 1.0
        wait_total += cycle_wait[offset]

    # Frequency extrapolation with end-of-day arrival cutoff.
    frequency_total = 0.0
    for rec in pattern:
        dep0 = int(rec["dep"])
        tt = float(rec["travel_time"])
        k = 0
        while True:
            dep_abs = dep0 + 120 * k
            if dep_abs >= DAY_END_MIN:
                break
            if dep_abs + tt <= DAY_END_MIN:
                frequency_total += 1.0
                k += 1
                continue
            break

    fastest = min(float(rec["travel_time"]) for rec in pattern)

    def _weighted(component: str) -> float:
        num = 0.0
        for idx, rec in enumerate(pattern):
            num += float(rec[component]) * weight_minutes[idx]
        return num / total_minutes if total_minutes > 0 else math.nan

    avg_wait = wait_total / total_minutes if total_minutes > 0 else math.nan
    avg_rolling = _weighted("rolling")
    avg_dwell = _weighted("dwell")
    avg_transfer = _weighted("transfer")

    return {
        "frequency": frequency_total,
        "fastest_travel_time": fastest,
        "avg_wait": avg_wait,
        "avg_rolling": avg_rolling,
        "avg_dwell": avg_dwell,
        "avg_transfer": avg_transfer,
    }


def _init_rows(stations: List[str], destination: str) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for s in stations:
        rows[s] = {
            "StationName": s,
            "DestinationStation": destination,
            "2026_frequency": math.nan,
            "2035_frequency": math.nan,
            "2026_fastest": math.nan,
            "2035_fastest": math.nan,
            "2026_wait": math.nan,
            "2035_wait": math.nan,
            "2026_rolling": math.nan,
            "2035_rolling": math.nan,
            "2026_dwell": math.nan,
            "2035_dwell": math.nan,
            "2026_transfer": math.nan,
            "2035_transfer": math.nan,
        }
    return rows


def _write_metric_csv(path: Path, rows: Dict[str, Dict[str, Any]], col_2026: str, col_2035: str) -> None:
    df = pd.DataFrame([rows[k] for k in rows.keys()])
    out = df[["StationName", col_2026, col_2035]].copy()
    out.columns = ["StationName", "2026", "2035"]
    out.to_csv(path, index=False)


def _file_fingerprint(path: Path) -> Tuple[str, int, int]:
    stat = path.stat()
    return (str(path), int(stat.st_size), int(stat.st_mtime))


def _build_cache_fingerprint() -> Dict[str, Any]:
    return {
        "inputs": [
            _file_fingerprint(TIMETABLE_CSV),
            _file_fingerprint(FILTERED_STATIONS_CSV),
            _file_fingerprint(NOTEBOOK_IPYNB),
            _file_fingerprint(THIS_SCRIPT),
        ],
        "constants": {
            "DAY_START_MIN": DAY_START_MIN,
            "WINDOW_END_MIN": WINDOW_END_MIN,
            "DAY_END_MIN": DAY_END_MIN,
            "ZERO_MIN_REPLACEMENT": ZERO_MIN_REPLACEMENT,
        },
    }


def _load_models_with_cache(cache_path: Path) -> Dict[str, Any]:
    # Always load module definitions first so pickle can resolve classes.
    _module, system = build_second_half_system(verbose_validation=False)
    expected_fp = _build_cache_fingerprint()

    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                cached = pickle.load(f)
            if (
                isinstance(cached, dict)
                and cached.get("fingerprint") == expected_fp
                and isinstance(cached.get("models"), dict)
            ):
                _log(f"Loaded model cache: {cache_path}")
                return cached["models"]
            _log("Model cache fingerprint mismatch; rebuilding models.")
        except Exception as exc:
            _log(f"Model cache load failed ({exc}); rebuilding models.")

    with contextlib.redirect_stdout(_TimestampedWriter(sys.stdout)):
        system.build_all_models()
    models = system.models

    try:
        payload = {"fingerprint": expected_fp, "models": models}
        with cache_path.open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        _log(f"Wrote model cache: {cache_path}")
    except Exception as exc:
        _log(f"Warning: could not write model cache ({exc}).")

    return models


def _load_checkpoint(
    checkpoint_path: Path,
    destination: str,
    stations_to_run: List[str],
) -> Tuple[Dict[str, Dict[str, Any]], int]:
    if not checkpoint_path.exists():
        return _init_rows(stations=stations_to_run, destination=destination), 0

    try:
        with checkpoint_path.open("rb") as f:
            state = pickle.load(f)
        if (
            state.get("destination") == destination
            and state.get("stations_to_run") == stations_to_run
            and isinstance(state.get("rows"), dict)
            and isinstance(state.get("next_index"), int)
        ):
            _log(f"Resuming from checkpoint {checkpoint_path} at origin index {state['next_index']}")
            return state["rows"], state["next_index"]
    except Exception as exc:
        _log(f"Checkpoint load failed ({exc}); starting from scratch.")

    return _init_rows(stations=stations_to_run, destination=destination), 0


def _save_checkpoint(
    checkpoint_path: Path,
    destination: str,
    stations_to_run: List[str],
    rows: Dict[str, Dict[str, Any]],
    next_index: int,
) -> None:
    state = {
        "destination": destination,
        "stations_to_run": stations_to_run,
        "rows": rows,
        "next_index": next_index,
    }
    with checkpoint_path.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build first OD-column metric tables for 2026/2035.")
    parser.add_argument("--destination-index", type=int, default=0, help="0-based destination index in station list.")
    parser.add_argument("--max-legs", type=int, default=50, help="Max legs per route query.")
    parser.add_argument(
        "--limit-origins",
        type=int,
        default=0,
        help="Optional debug limit; 0 means all origins.",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=Path(os.environ.get("OD_MODEL_CACHE", str(BASE_DIR / "od_model_cache_2026_2035.pkl"))),
        help="Path for serialized prebuilt models cache.",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=BASE_DIR / "od_first_column_checkpoint.pkl",
        help="Path for resumable progress checkpoint.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Checkpoint every N processed origins.",
    )
    args = parser.parse_args()

    stations_df = pd.read_csv(FILTERED_STATIONS_CSV)
    stations = stations_df["StationName"].astype(str).tolist()
    if not (0 <= args.destination_index < len(stations)):
        raise IndexError(f"destination-index {args.destination_index} out of bounds for {len(stations)} stations")
    destination = stations[args.destination_index]

    if args.limit_origins > 0:
        stations_to_run = stations[: args.limit_origins]
    else:
        stations_to_run = stations

    models = _load_models_with_cache(args.model_cache)
    available_by_year = {y: set(m.available_stations()) for y, m in models.items()}
    components_by_year = {y: _build_station_components(m) for y, m in models.items()}
    disconnected_pairs_skipped = 0
    rows, start_index = _load_checkpoint(
        checkpoint_path=args.checkpoint_file,
        destination=destination,
        stations_to_run=stations_to_run,
    )

    t0 = time.time()
    for idx, origin in enumerate(stations_to_run, start=1):
        if idx <= start_index:
            continue
        row = rows[origin]
        if origin == destination:
            row["2026_frequency"] = 0.0
            row["2035_frequency"] = 0.0
            row["2026_fastest"] = 0.0
            row["2035_fastest"] = 0.0
            row["2026_wait"] = 0.0
            row["2035_wait"] = 0.0
            row["2026_rolling"] = 0.0
            row["2035_rolling"] = 0.0
            row["2026_dwell"] = 0.0
            row["2035_dwell"] = 0.0
            row["2026_transfer"] = 0.0
            row["2035_transfer"] = 0.0
            continue

        for year in ("2026", "2035"):
            if origin not in available_by_year[year] or destination not in available_by_year[year]:
                row[f"{year}_frequency"] = 0.0
                continue

            comp_map = components_by_year[year]
            if comp_map.get(origin) != comp_map.get(destination):
                row[f"{year}_frequency"] = 0.0
                disconnected_pairs_skipped += 1
                continue

            pattern = _sample_two_hour_pattern(
                model=models[year],
                origin=origin,
                destination=destination,
                max_legs=args.max_legs,
            )
            if not pattern:
                row[f"{year}_frequency"] = 0.0
                continue

            metrics = _compute_day_metrics_from_pattern(pattern)
            row[f"{year}_frequency"] = metrics["frequency"]
            row[f"{year}_fastest"] = metrics["fastest_travel_time"]
            row[f"{year}_wait"] = metrics["avg_wait"]
            row[f"{year}_rolling"] = metrics["avg_rolling"]
            row[f"{year}_dwell"] = metrics["avg_dwell"]
            row[f"{year}_transfer"] = metrics["avg_transfer"]

        if args.checkpoint_every > 0 and (idx % args.checkpoint_every == 0):
            _save_checkpoint(
                checkpoint_path=args.checkpoint_file,
                destination=destination,
                stations_to_run=stations_to_run,
                rows=rows,
                next_index=idx,
            )

        if idx % 25 == 0 or idx == len(stations_to_run):
            elapsed = time.time() - t0
            _log(f"Processed {idx}/{len(stations_to_run)} origins in {elapsed/60:.1f} min")

    out_dir = BASE_DIR
    _write_metric_csv(out_dir / "OD_matrix_Frequency.csv", rows, "2026_frequency", "2035_frequency")
    _write_metric_csv(out_dir / "OD_matrix_Fastest_observed_connection.csv", rows, "2026_fastest", "2035_fastest")
    _write_metric_csv(out_dir / "OD_matrix_Wait_time.csv", rows, "2026_wait", "2035_wait")
    _write_metric_csv(out_dir / "OD_matrix_Rolling_time.csv", rows, "2026_rolling", "2035_rolling")
    _write_metric_csv(out_dir / "OD_matrix_Dwell_time.csv", rows, "2026_dwell", "2035_dwell")
    _write_metric_csv(out_dir / "OD_matrix_Transfer_time.csv", rows, "2026_transfer", "2035_transfer")

    if args.checkpoint_file.exists():
        try:
            args.checkpoint_file.unlink()
        except Exception:
            pass

    _log(f"Done. Destination first-column station: {destination}")
    _log(f"Wrote first-column CSV files to: {out_dir}")
    _log(f"Connectivity short-circuit skips: {disconnected_pairs_skipped}")


if __name__ == "__main__":
    main()
