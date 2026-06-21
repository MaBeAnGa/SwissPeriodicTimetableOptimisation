#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import fcntl
import heapq
import json
import os
import signal
import shutil
import subprocess
import sys
import time
import traceback
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from build_od_first_column_metrics_2026_2035 import (  # noqa: E402
    DAY_START_MIN,
    WINDOW_END_MIN,
    _decompose_legs_with_rebalance,
    _load_models_with_cache,
    _to_hhmm,
)


FILTERED_STATIONS_CSV = BASE_DIR / "FilteredStations2026_2035.csv"
DEFAULT_CACHE_FILE = BASE_DIR / "od_model_cache_2026_2035.pkl"
DEFAULT_STATE_ROOT = BASE_DIR / "od_nondominated_paths_cells_state"
DEFAULT_LOG_FILE = BASE_DIR / "od_nondominated_paths_cells.log"
CONTROLLER_LOCK_FILENAME = ".controller.lock"
STATION_ORDER_MANIFEST_FILENAME = "station_order_manifest.json"
ZURICH_HB_CANONICAL = "Zürich HB"
ZURICH_HB_ALIASES = {
    "Zürich HB (21-22)",
    "Zürich HB (31-34)",
    "Zürich HB (41-44)",
}
ZURICH_HB_3134 = "Zürich HB (31-34)"
ZURICH_HB_FAMILY = {ZURICH_HB_CANONICAL, *ZURICH_HB_ALIASES}
ZURICH_HB_COMPLEX_TRANSFER_MIN = 7
ZURICH_HB_3134_SELF_TRANSFER_MIN = 4


def _canonical_station_name(name: str) -> str:
    n = str(name)
    if n in ZURICH_HB_ALIASES:
        return ZURICH_HB_CANONICAL
    return n


def _is_zurich_hb_family(name: Any) -> bool:
    return str(name) in ZURICH_HB_FAMILY


def _trip_canonical(model: Any, trip_instance_id: Any) -> str:
    trip = str(trip_instance_id or "")
    return str(model.trip_instance_to_canonical.get(trip, trip))


def _station_has_departures(model: Any, station: str) -> bool:
    return bool(model.station_departures.get(station))


def _origin_boarding_stations(model: Any, origin: str) -> list[str]:
    if origin == ZURICH_HB_CANONICAL:
        return [s for s in sorted(ZURICH_HB_FAMILY) if _station_has_departures(model, s)]
    return [origin]


def _transfer_boarding_stations(model: Any, arrival_station: str) -> list[str]:
    if _is_zurich_hb_family(arrival_station):
        return [s for s in sorted(ZURICH_HB_FAMILY) if _station_has_departures(model, s)]
    return [arrival_station]


def _minimum_transfer_between(model: Any, arrival_station: str, departure_station: str) -> Optional[int]:
    """
    Pair-specific station transfer rule used by the route builder.

    Normal stations use FilteredStations2026_2035.MinimumTransferTime. Zürich HB
    is a platform complex: 7 minutes across the complex, except a same-group
    transfer within Zürich HB (31-34), which is 4 minutes.
    """
    arr = str(arrival_station)
    dep = str(departure_station)
    if _is_zurich_hb_family(arr) and _is_zurich_hb_family(dep):
        if arr == ZURICH_HB_3134 and dep == ZURICH_HB_3134:
            return ZURICH_HB_3134_SELF_TRANSFER_MIN
        return ZURICH_HB_COMPLEX_TRANSFER_MIN
    if arr != dep:
        return None
    return int(model.transfer_time(arr))


def _seed_origin_source_edges_safe(model: Any, origin: str, departure_time_min: int) -> list[int]:
    out: list[int] = []
    for station in _origin_boarding_stations(model, origin):
        dep_times = model.station_departure_times.get(station, [])
        dep_nodes = model.station_departures.get(station, [])
        idx = bisect.bisect_left(dep_times, int(departure_time_min))
        if idx < len(dep_nodes):
            out.extend(dep_nodes[idx:])
    return out


def _first_transfer_departure_nodes_safe(model: Any, arrival_node: Any) -> list[int]:
    out: list[int] = []
    arr_station = str(arrival_node.station)
    arr_time = int(arrival_node.time)
    arr_canonical = _trip_canonical(model, arrival_node.trip_instance_id)
    for board_station in _transfer_boarding_stations(model, arr_station):
        min_transfer = _minimum_transfer_between(model, arr_station, board_station)
        if min_transfer is None:
            continue
        dep_times = model.station_departure_times.get(board_station, [])
        dep_nodes = model.station_departures.get(board_station, [])
        idx = bisect.bisect_left(dep_times, arr_time + int(min_transfer))
        while idx < len(dep_nodes):
            dep_node_id = dep_nodes[idx]
            dep_node = model.nodes[dep_node_id]
            if _trip_canonical(model, dep_node.trip_instance_id) != arr_canonical:
                out.append(dep_node_id)
                break
            idx += 1
    return out


def _wait_edge_respects_prior_arrival_transfer_time(
    *,
    model: Any,
    predecessor: dict[int, tuple[Optional[int], Optional[str], Optional[str], Optional[str]]],
    from_node_id: int,
    to_node_id: int,
) -> bool:
    """
    Prevent hidden transfers through departure-node wait edges.

    The notebook graph has service dwell edges from arrival to departure nodes,
    then zero/positive wait edges between departure nodes at the same station.
    Without this guard, a passenger can arrive on one train, ride the same-train
    dwell edge, and then hop to another departure before the station minimum
    transfer time has elapsed.
    """
    from_node = model.nodes[from_node_id]
    to_node = model.nodes[to_node_id]
    if from_node.kind != "ab" or to_node.kind != "ab":
        return True

    pred = predecessor.get(from_node_id)
    if pred is None:
        return True
    previous_node_id, previous_edge_type, _, _ = pred
    if previous_edge_type != "service" or previous_node_id is None:
        return True

    previous_node = model.nodes[previous_node_id]
    if (
        previous_node.kind != "an"
        or previous_node.station != from_node.station
        or previous_node.trip_instance_id != from_node.trip_instance_id
    ):
        return True

    if to_node.trip_instance_id == from_node.trip_instance_id:
        return True
    min_transfer = _minimum_transfer_between(model, str(previous_node.station), str(to_node.station))
    if min_transfer is None:
        return False
    return int(to_node.time) >= int(previous_node.time) + int(min_transfer)


def _normalise_output_terminal_name(name: Any, expected_terminal: str) -> str:
    """Collapse Zürich HB aliases only when the OD endpoint is Zürich HB."""
    text = str(name)
    if expected_terminal == ZURICH_HB_CANONICAL and _canonical_station_name(text) == ZURICH_HB_CANONICAL:
        return ZURICH_HB_CANONICAL
    return text


def _normalise_terminal_path_payload(
    path: dict[str, Any],
    *,
    origin: str,
    destination: str,
) -> dict[str, Any]:
    """
    Keep Zürich HB aliases for internal transfer modelling, but make OD terminals
    match the public station universe. This preserves the original baseline
    convention: Zürich HB aliases can appear inside a route, but an OD that starts
    or ends at Zürich HB is written as Zürich HB, not as a platform-group alias.
    """
    out = dict(path)

    legs = [dict(leg) for leg in (path.get("legs", []) or [])]
    if legs:
        legs[0]["from_station"] = _normalise_output_terminal_name(
            legs[0].get("from_station", ""), origin
        )
        legs[-1]["to_station"] = _normalise_output_terminal_name(
            legs[-1].get("to_station", ""), destination
        )
        for leg in legs:
            leg["terminal_station"] = _canonical_station_name(str(leg.get("terminal_station", "")))
        out["legs"] = legs

    nodes_sequence = list(path.get("nodes_sequence", []) or [])
    if nodes_sequence:
        nodes_sequence[0] = _normalise_output_terminal_name(nodes_sequence[0], origin)
        nodes_sequence[-1] = _normalise_output_terminal_name(nodes_sequence[-1], destination)
        out["nodes_sequence"] = nodes_sequence
        out["intermediate_nodes"] = nodes_sequence[1:-1] if len(nodes_sequence) >= 2 else []

    station_events = [dict(event) for event in (path.get("station_events", []) or [])]
    if station_events:
        station_events[0]["station"] = _normalise_output_terminal_name(
            station_events[0].get("station", ""), origin
        )
        station_events[-1]["station"] = _normalise_output_terminal_name(
            station_events[-1].get("station", ""), destination
        )
        out["station_events"] = station_events

    return out


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _append_log_line(log_file: Path, message: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(f"[{_ts()}] {message}\n")
        f.flush()
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _acquire_named_lock(lock_path: Path, payload: dict[str, Any]) -> Any:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_file = lock_path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_file.seek(0)
        holder = lock_file.read().strip()
        lock_file.close()
        raise RuntimeError(
            f"Another process holds lock: {lock_path}\n"
            f"Holder info: {holder or 'unavailable'}"
        )
    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(json.dumps(payload, ensure_ascii=False))
    lock_file.flush()
    return lock_file


def _release_named_lock(lock_file: Any) -> None:
    if lock_file is None:
        return
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        lock_file.close()
    except Exception:
        pass


def _load_station_order(
    *,
    year: str | None = None,
    station_scope: str = "union",
) -> list[str]:
    df = json.loads(FILTERED_STATIONS_CSV.read_text(encoding="utf-8")) if FILTERED_STATIONS_CSV.suffix == ".json" else None
    if df is not None:
        raise RuntimeError("Unexpected station file format.")
    import pandas as pd

    sdf = pd.read_csv(FILTERED_STATIONS_CSV, dtype=str).fillna("")
    if "StationName" not in sdf.columns:
        raise ValueError(f"{FILTERED_STATIONS_CSV} missing StationName column")
    scope = str(station_scope or "union").strip().lower().replace("_", "-")
    if scope in {"year", "year-specific", "active-year"}:
        if year not in {"2026", "2035"}:
            raise ValueError(f"year-specific station scope requires year 2026 or 2035, got {year!r}")
        line_col = f"{year}lines"
        if line_col not in sdf.columns:
            raise ValueError(f"{FILTERED_STATIONS_CSV} missing {line_col} column")
        line_counts = pd.to_numeric(sdf[line_col], errors="coerce").fillna(0)
        sdf = sdf[line_counts > 0].copy()
    elif scope != "union":
        raise ValueError(f"Unsupported station scope {station_scope!r}; use union or year-specific.")
    raw = sdf["StationName"].astype(str).tolist()
    collapsed: list[str] = []
    seen: set[str] = set()
    for name in raw:
        canon = _canonical_station_name(name)
        if canon in seen:
            continue
        seen.add(canon)
        collapsed.append(canon)
    return collapsed


def _station_order_hash(station_order: list[str]) -> str:
    h = hashlib.sha256()
    for station in station_order:
        h.update(station.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _validate_or_write_station_order_manifest(
    *,
    year_state_dir: Path,
    rows_dir: Path,
    year: str,
    station_scope: str,
    station_order: list[str],
    resume: bool,
) -> None:
    manifest_path = year_state_dir / STATION_ORDER_MANIFEST_FILENAME
    expected = {
        "year": str(year),
        "station_scope": str(station_scope),
        "station_count": len(station_order),
        "station_order_sha256": _station_order_hash(station_order),
        "first_station": station_order[0] if station_order else "",
        "last_station": station_order[-1] if station_order else "",
    }
    if manifest_path.exists():
        current = json.loads(manifest_path.read_text(encoding="utf-8"))
        mismatches = [
            key
            for key in ("year", "station_scope", "station_count", "station_order_sha256")
            if current.get(key) != expected.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"State root station-order manifest mismatch for {year}: {mismatches}. "
                f"Expected {expected}; found {current}. Use a separate state root or clean restart."
            )
        return

    existing_rows = sorted(rows_dir.glob("row_*.json")) if rows_dir.exists() else []
    if resume and existing_rows:
        probe_path = existing_rows[0]
        probe = _read_row_payload(probe_path)
        try:
            probe_idx = int(probe_path.stem.split("_")[1])
        except Exception:
            probe_idx = int(probe.get("row_idx", -1))
        expected_origin = station_order[probe_idx] if 0 <= probe_idx < len(station_order) else None
        if len(probe.get("cells", []) or []) != len(station_order) or str(probe.get("origin", "")) != str(expected_origin):
            raise ValueError(
                f"Existing row state at {rows_dir} does not match requested station universe for {year}. "
                f"Probe={probe_path.name}, origin={probe.get('origin')!r}, cells={len(probe.get('cells', []) or [])}; "
                f"expected_origin={expected_origin!r}, expected_cells={len(station_order)}. "
                f"Use a projected/new state root instead of resuming this one."
            )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(expected, indent=2, ensure_ascii=False), encoding="utf-8")


def _encode_leg(leg: dict[str, Any]) -> dict[str, Any]:
    dep_min = int(leg["dep_time"])
    arr_min = int(leg["arr_time"])
    return {
        "from_station": str(leg["from_station"]),
        "to_station": str(leg["to_station"]),
        "dep_min": dep_min,
        "dep_time": _to_hhmm(dep_min),
        "arr_min": arr_min,
        "arr_time": _to_hhmm(arr_min),
        "service_name": str(leg.get("service_name", "")),
        "trip_instance_id": str(leg.get("trip_instance_id", "")),
        "canonical_trip_id": str(leg.get("canonical_trip_id", "")),
        "terminal_station": str(leg.get("terminal_station", "")),
    }


def _build_station_events(encoded_legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not encoded_legs:
        return []
    events: list[dict[str, Any]] = []
    events.append(
        {
            "station": encoded_legs[0]["from_station"],
            "arr_min": None,
            "arr_time": None,
            "dep_min": encoded_legs[0]["dep_min"],
            "dep_time": encoded_legs[0]["dep_time"],
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
                "station": leg["to_station"],
                "arr_min": int(leg["arr_min"]),
                "arr_time": str(leg["arr_time"]),
                "dep_min": dep_next,
                "dep_time": dep_next_hhmm,
            }
        )
    return events


def _build_transfer_events(encoded_legs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(len(encoded_legs) - 1):
        cur = encoded_legs[i]
        nxt = encoded_legs[i + 1]
        if cur["canonical_trip_id"] == nxt["canonical_trip_id"]:
            continue
        wait_min = max(int(nxt["dep_min"]) - int(cur["arr_min"]), 0)
        out.append(
            {
                "station": cur["to_station"],
                "arrive_prev_min": int(cur["arr_min"]),
                "arrive_prev_time": str(cur["arr_time"]),
                "depart_next_min": int(nxt["dep_min"]),
                "depart_next_time": str(nxt["dep_time"]),
                "transfer_wait_min": wait_min,
                "from_service": str(cur["service_name"]),
                "to_service": str(nxt["service_name"]),
            }
        )
    return out


def _sample_nondominated_paths_detailed(
    *,
    model: Any,
    origin: str,
    destination: str,
    max_legs: int,
) -> list[dict[str, Any]]:
    query_min = int(DAY_START_MIN)
    seen_departures: set[int] = set()
    paths: list[dict[str, Any]] = []

    while True:
        result = _query_detailed_path_safe(
            model=model,
            origin=origin,
            destination=destination,
            query_min=query_min,
            max_legs=max_legs,
        )
        if result is None:
            return paths

        dep = int(result["dep_min"])
        if dep >= int(WINDOW_END_MIN):
            break

        if dep in seen_departures:
            query_min = dep + 1
            if query_min >= int(WINDOW_END_MIN):
                pass
            continue

        seen_departures.add(dep)
        paths.append(result)

        query_min = dep + 1
        if query_min >= int(WINDOW_END_MIN):
            pass

    paths.sort(key=lambda x: int(x["dep_min"]))
    return paths


def _build_cell_payload_direct(
    *,
    model: Any,
    origin: str,
    destination: str,
    max_legs: int,
) -> dict[str, Any]:
    if origin == destination:
        return {
            "window_start": _to_hhmm(int(DAY_START_MIN)),
            "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
            "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
            "same_station": True,
            "paths": [],
        }
    paths = _sample_nondominated_paths_detailed(
        model=model,
        origin=origin,
        destination=destination,
        max_legs=max_legs,
    )
    paths = [
        _normalise_terminal_path_payload(path, origin=origin, destination=destination)
        for path in paths
    ]
    return {
        "window_start": _to_hhmm(int(DAY_START_MIN)),
        "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
        "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
        "same_station": False,
        "path_count": len(paths),
        "paths": paths,
    }


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


class _OneToManyBatchRunDetailed:
    """
    One Dijkstra-like one-to-many run for (origin, query_min), with lazy detailed
    path reconstruction per destination.
    """

    def __init__(
        self,
        *,
        model: Any,
        origin: str,
        query_min: int,
        max_legs: int,
        stop_destination: Optional[str] = None,
    ) -> None:
        self.model = model
        self.origin = origin
        self.query_min = int(query_min)
        self.max_legs = int(max_legs)
        self.stop_destination = stop_destination
        self.module = _get_notebook_module_for_model(model)

        self._best_dest: dict[str, tuple[int, tuple[int, int, int, int]]] = {}
        self._predecessor: dict[int, tuple[Optional[int], Optional[str], Optional[str], Optional[str]]] = {}
        self._result_cache: dict[str, Optional[dict[str, Any]]] = {}
        self.reconstructed_destination_count = 0

        self._run_search()

    def _run_search(self) -> None:
        start_nodes = _seed_origin_source_edges_safe(self.model, self.origin, self.query_min)
        if not start_nodes:
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
            stop_search = False
            if node.kind == "an":
                station = str(node.station)
                destination_keys = {station, _canonical_station_name(station)}
                for destination_key in destination_keys:
                    previous = self._best_dest.get(destination_key)
                    if previous is None or current_cost < previous[1]:
                        self._best_dest[destination_key] = (node_id, current_cost)
                if self.stop_destination is not None and self.stop_destination in destination_keys:
                    stop_search = True
            if stop_search:
                break

            def relax(
                to_node_id: int,
                edge_type: str,
                service_name: Optional[str],
                trip_instance_id: Optional[str],
            ) -> None:
                to_node = self.model.nodes[to_node_id]
                arrival_time = int(to_node.time)
                first_dep_neg = current_cost[1]
                transfers = current_cost[2] + (1 if edge_type == "transfer" else 0)
                movement_segments = current_cost[3] + self.module._movement_increment(
                    self.model, node_id, to_node_id, edge_type
                )
                new_cost = (arrival_time, first_dep_neg, transfers, movement_segments)
                if new_cost < dist.get(to_node_id, inf_cost):
                    dist[to_node_id] = new_cost
                    self._predecessor[to_node_id] = (
                        node_id,
                        edge_type,
                        service_name,
                        trip_instance_id,
                    )
                    heapq.heappush(pq, (new_cost, to_node_id))

            for edge in self.model.adj.get(node_id, []):
                if edge.edge_type == "transfer":
                    # Transfer edges are regenerated here so station minima and
                    # Zürich HB alias rules cannot be bypassed by old graph edges.
                    continue
                if edge.edge_type == "wait" and not _wait_edge_respects_prior_arrival_transfer_time(
                    model=self.model,
                    predecessor=self._predecessor,
                    from_node_id=node_id,
                    to_node_id=edge.to_node,
                ):
                    continue
                relax(edge.to_node, edge.edge_type, edge.service_name, edge.trip_instance_id)

            if node.kind == "an":
                for transfer_node_id in _first_transfer_departure_nodes_safe(self.model, node):
                    relax(transfer_node_id, "transfer", None, None)

    def result_for_destination(self, destination: str) -> Optional[dict[str, Any]]:
        if destination in self._result_cache:
            return self._result_cache[destination]

        best = self._best_dest.get(destination)
        if best is None:
            self._result_cache[destination] = None
            return None

        best_dest_node, best_dest_cost = best
        _, edge_path = self.module.reconstruct_node_path(self._predecessor, best_dest_node)
        raw_legs = self.module.compress_event_path_to_legs(self.model, edge_path)
        raw_legs = self.module.simplify_legs_by_staying_on_train(self.model, raw_legs)
        if len(raw_legs) > self.max_legs:
            self._result_cache[destination] = None
            return None

        dep = int(-best_dest_cost[1])
        arr = int(best_dest_cost[0])
        travel_time = float(arr - dep)
        rolling, dwell, transfer = _decompose_legs_with_rebalance(raw_legs, travel_time)

        encoded_legs = [_encode_leg(leg) for leg in raw_legs]
        transfer_events = _build_transfer_events(encoded_legs)
        station_events = _build_station_events(encoded_legs)
        line_names: list[str] = []
        for leg in encoded_legs:
            service_name = str(leg["service_name"])
            if not line_names or line_names[-1] != service_name:
                line_names.append(service_name)

        node_sequence: list[str] = []
        if encoded_legs:
            node_sequence.append(encoded_legs[0]["from_station"])
            node_sequence.extend(leg["to_station"] for leg in encoded_legs)

        out = {
            "dep_min": dep,
            "dep_time": _to_hhmm(dep),
            "arr_min": arr,
            "arr_time": _to_hhmm(arr),
            "travel_time_min": travel_time,
            "rolling_min": float(rolling),
            "dwell_min": float(dwell),
            "transfer_min": float(transfer),
            "num_transfers": int(len(transfer_events)),
            "line_names": line_names,
            "nodes_sequence": node_sequence,
            "intermediate_nodes": node_sequence[1:-1] if len(node_sequence) >= 2 else [],
            "station_events": station_events,
            "transfer_events": transfer_events,
            "legs": encoded_legs,
        }

        self._result_cache[destination] = out
        self.reconstructed_destination_count += 1
        return out


def _query_detailed_path_safe(
    *,
    model: Any,
    origin: str,
    destination: str,
    query_min: int,
    max_legs: int,
) -> Optional[dict[str, Any]]:
    batch = _OneToManyBatchRunDetailed(
        model=model,
        origin=origin,
        query_min=int(query_min),
        max_legs=int(max_legs),
        stop_destination=destination,
    )
    return batch.result_for_destination(destination)


def _build_cell_payload_from_paths(origin: str, destination: str, paths: list[dict[str, Any]]) -> dict[str, Any]:
    if origin == destination:
        return {
            "window_start": _to_hhmm(int(DAY_START_MIN)),
            "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
            "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
            "same_station": True,
            "paths": [],
        }
    paths_sorted = sorted(
        (
            _normalise_terminal_path_payload(path, origin=origin, destination=destination)
            for path in paths
        ),
        key=lambda x: int(x["dep_min"]),
    )
    return {
        "window_start": _to_hhmm(int(DAY_START_MIN)),
        "window_end_exclusive": _to_hhmm(int(WINDOW_END_MIN)),
        "window_end_inclusive": _to_hhmm(int(WINDOW_END_MIN) - 1),
        "same_station": False,
        "path_count": len(paths_sorted),
        "paths": paths_sorted,
    }


def _build_row_cells_one_to_many_detailed(
    *,
    model: Any,
    origin: str,
    station_order: list[str],
    max_legs: int,
) -> list[dict[str, Any]]:
    destinations = list(station_order)
    destinations_to_route = [d for d in destinations if d != origin]
    paths_by_destination: dict[str, list[dict[str, Any]]] = {d: [] for d in destinations_to_route}
    seen_dep: dict[str, set[int]] = {d: set() for d in destinations_to_route}

    module = _get_notebook_module_for_model(model)
    origin_board_stations = _origin_boarding_stations(model, origin)
    dep_times = sorted(
        {
            int(t)
            for station in origin_board_stations
            for t in model.station_departure_times.get(station, [])
        }
    )

    def _seed_min_for_query_min(query_min: int) -> Optional[int]:
        idx = bisect.bisect_left(dep_times, int(query_min))
        if idx >= len(dep_times):
            return None
        return int(dep_times[idx])

    pending: dict[int, set[str]] = defaultdict(set)
    done: set[str] = set()
    init_seed_min = _seed_min_for_query_min(int(DAY_START_MIN))
    if init_seed_min is None:
        done.update(destinations_to_route)
    else:
        pending[init_seed_min].update(destinations_to_route)

    while pending:
        seed_min = min(pending.keys())
        current_dests = sorted(pending.pop(seed_min))
        if not current_dests:
            continue
        if seed_min >= int(WINDOW_END_MIN):
            done.update(current_dests)
            continue

        batch = _OneToManyBatchRunDetailed(
            model=model,
            origin=origin,
            query_min=int(seed_min),
            max_legs=int(max_legs),
        )

        for destination in current_dests:
            if destination in done:
                continue

            result = batch.result_for_destination(destination)
            if result is None:
                done.add(destination)
                continue

            dep = int(result["dep_min"])
            if dep >= int(WINDOW_END_MIN):
                done.add(destination)
                continue

            if dep in seen_dep[destination]:
                next_q = dep + 1
                next_seed_min = _seed_min_for_query_min(next_q)
                if next_seed_min is None or next_seed_min >= int(WINDOW_END_MIN):
                    done.add(destination)
                else:
                    pending[next_seed_min].add(destination)
                continue

            seen_dep[destination].add(dep)
            paths_by_destination[destination].append(result)

            next_q = dep + 1
            next_seed_min = _seed_min_for_query_min(next_q)
            if next_seed_min is None or next_seed_min >= int(WINDOW_END_MIN):
                done.add(destination)
            else:
                pending[next_seed_min].add(destination)

    return [
        _build_cell_payload_from_paths(
            origin=origin,
            destination=destination,
            paths=paths_by_destination.get(destination, []),
        )
        for destination in station_order
    ]


def _row_file_path(rows_dir: Path, row_idx: int) -> Path:
    return rows_dir / f"row_{row_idx:06d}.json"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    tmp_path.replace(path)
    try:
        dir_fd = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        # Best-effort durability: some filesystems/OS setups may not support fsync on dirs.
        pass


def _validate_row_cells_payload_strict(
    *,
    row_idx: int,
    origin: str,
    station_order: list[str],
    row_cells_payload: list[dict[str, Any]],
) -> None:
    """
    Strict structural + semantic validation to prevent silent row/column drift:
    - row has exactly one cell per destination in station_order
    - diagonal cells are same_station=True
    - off-diagonal cells are same_station=False
    - for non-empty paths, nodes_sequence starts at origin and ends at destination
    """
    n = len(station_order)
    if len(row_cells_payload) != n:
        raise ValueError(
            f"row {row_idx + 1}: expected {n} destination cells, got {len(row_cells_payload)}"
        )

    issues: list[str] = []
    for dest_idx, destination in enumerate(station_order):
        payload = row_cells_payload[dest_idx]
        same_station = bool(payload.get("same_station", False))
        paths = payload.get("paths", []) or []

        if destination == origin:
            if not same_station:
                issues.append(
                    f"dest={destination}: diagonal cell has same_station={same_station}"
                )
            continue

        if same_station:
            issues.append(f"dest={destination}: off-diagonal cell has same_station=True")
            continue

        if not paths:
            continue

        first_path = paths[0]
        node_seq = first_path.get("nodes_sequence", []) or []
        if not node_seq:
            issues.append(f"dest={destination}: non-empty paths but empty nodes_sequence")
            continue
        start_actual = _canonical_station_name(str(node_seq[0]))
        start_expected = _canonical_station_name(origin)
        end_actual = _canonical_station_name(str(node_seq[-1]))
        end_expected = _canonical_station_name(destination)

        if start_actual != start_expected:
            issues.append(
                f"dest={destination}: nodes_sequence starts at '{node_seq[0]}' (expected '{origin}')"
            )
        if end_actual != end_expected:
            issues.append(
                f"dest={destination}: nodes_sequence ends at '{node_seq[-1]}' (expected '{destination}')"
            )

        if origin == ZURICH_HB_CANONICAL or destination == ZURICH_HB_CANONICAL:
            for path_idx, path in enumerate(paths):
                seq = path.get("nodes_sequence", []) or []
                if not seq:
                    continue
                if origin == ZURICH_HB_CANONICAL and str(seq[0]) != ZURICH_HB_CANONICAL:
                    issues.append(
                        f"dest={destination}: path {path_idx + 1} starts at '{seq[0]}' "
                        f"but OD origin is '{ZURICH_HB_CANONICAL}'"
                    )
                if destination == ZURICH_HB_CANONICAL and str(seq[-1]) != ZURICH_HB_CANONICAL:
                    issues.append(
                        f"dest={destination}: path {path_idx + 1} ends at '{seq[-1]}' "
                        f"but OD destination is '{ZURICH_HB_CANONICAL}'"
                    )

                legs = path.get("legs", []) or []
                if legs:
                    if (
                        origin == ZURICH_HB_CANONICAL
                        and str(legs[0].get("from_station", "")) != ZURICH_HB_CANONICAL
                    ):
                        issues.append(
                            f"dest={destination}: path {path_idx + 1} first leg starts at "
                            f"'{legs[0].get('from_station', '')}' but OD origin is "
                            f"'{ZURICH_HB_CANONICAL}'"
                        )
                    if (
                        destination == ZURICH_HB_CANONICAL
                        and str(legs[-1].get("to_station", "")) != ZURICH_HB_CANONICAL
                    ):
                        issues.append(
                            f"dest={destination}: path {path_idx + 1} final leg ends at "
                            f"'{legs[-1].get('to_station', '')}' but OD destination is "
                            f"'{ZURICH_HB_CANONICAL}'"
                        )

                station_events = path.get("station_events", []) or []
                if station_events:
                    if (
                        origin == ZURICH_HB_CANONICAL
                        and str(station_events[0].get("station", "")) != ZURICH_HB_CANONICAL
                    ):
                        issues.append(
                            f"dest={destination}: path {path_idx + 1} first event is "
                            f"'{station_events[0].get('station', '')}' but OD origin is "
                            f"'{ZURICH_HB_CANONICAL}'"
                        )
                    if (
                        destination == ZURICH_HB_CANONICAL
                        and str(station_events[-1].get("station", "")) != ZURICH_HB_CANONICAL
                    ):
                        issues.append(
                            f"dest={destination}: path {path_idx + 1} final event is "
                            f"'{station_events[-1].get('station', '')}' but OD destination is "
                            f"'{ZURICH_HB_CANONICAL}'"
                        )

    if issues:
        preview = "; ".join(issues[:5])
        extra = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        raise ValueError(
            f"Strict row validation failed for row {row_idx + 1} origin='{origin}': {preview}{extra}"
        )


def _worker_main(args: argparse.Namespace) -> int:
    log_file = Path(args.log_file).resolve()
    rows_dir = Path(args.rows_dir).resolve()
    rows_dir.mkdir(parents=True, exist_ok=True)

    try:
        station_order = _load_station_order(
            year=str(args.year),
            station_scope=str(getattr(args, "station_scope", "union")),
        )
        n = len(station_order)
        models = _load_models_with_cache(Path(args.model_cache).resolve())
        if args.year not in models:
            raise KeyError(f"Year model not available: {args.year}")
        model = models[args.year]

        shard_id = int(args.shard_id)
        shards = int(args.shards)
        row_limit = int(getattr(args, "row_limit", 0) or 0)
        row_stop = n if row_limit <= 0 else min(n, row_limit)
        row_locks_dir = rows_dir / ".locks"
        row_locks_dir.mkdir(parents=True, exist_ok=True)
        startup_verbose = True
        _append_log_line(
            log_file,
            (
                f"{args.year} shard{shard_id:02d} worker start "
                f"pid={os.getpid()} shards={shards} stations={n} row_stop={row_stop}"
            ),
        )

        for row_idx in range(shard_id, row_stop, shards):
            row_path = _row_file_path(rows_dir, row_idx)
            if row_path.exists():
                continue
            row_lock_path = row_locks_dir / f"row_{row_idx:06d}.lock"
            row_lock_handle = None
            try:
                row_lock_handle = _acquire_named_lock(
                    row_lock_path,
                    payload={
                        "pid": os.getpid(),
                        "year": str(args.year),
                        "shard_id": int(shard_id),
                        "row_idx": int(row_idx),
                        "acquired_at": _ts(),
                    },
                )
            except RuntimeError:
                # Another shard/process is currently computing this row.
                continue
            if row_path.exists():
                _release_named_lock(row_lock_handle)
                continue

            origin = station_order[row_idx]
            try:
                if startup_verbose:
                    _append_log_line(
                        log_file,
                        (
                            f"{args.year} shard{shard_id:02d} startup first origin begin "
                            f"{row_idx + 1}/{n} {origin}"
                        ),
                    )
                try:
                    row_cells_payload = _build_row_cells_one_to_many_detailed(
                        model=model,
                        origin=origin,
                        station_order=station_order,
                        max_legs=int(args.max_legs),
                    )
                except Exception as row_exc:
                    # Safety fallback: keep progress even if one row fails in one-to-many mode.
                    _append_log_line(
                        log_file,
                        (
                            f"ERROR {args.year} shard{shard_id:02d}: one-to-many failed for "
                            f"origin {row_idx + 1}/{n} {origin}; fallback to direct mode. "
                            f"reason={row_exc}"
                        ),
                    )
                    row_cells_payload = [
                        _build_cell_payload_direct(
                            model=model,
                            origin=origin,
                            destination=destination,
                            max_legs=int(args.max_legs),
                        )
                        for destination in station_order
                    ]

                _validate_row_cells_payload_strict(
                    row_idx=row_idx,
                    origin=origin,
                    station_order=station_order,
                    row_cells_payload=row_cells_payload,
                )

                cells = [
                    json.dumps(
                        cell_payload,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    for cell_payload in row_cells_payload
                ]

                _atomic_write_json(
                    row_path,
                    {
                        "row_idx": int(row_idx),
                        "origin": origin,
                        "year": str(args.year),
                        "shard_id": int(shard_id),
                        "cells": cells,
                        "completed_at": _ts(),
                    },
                )
                _append_log_line(
                    log_file,
                    f"{args.year} shard{shard_id:02d} completed origin {row_idx + 1}/{n} {origin}",
                )
                if startup_verbose:
                    _append_log_line(
                        log_file,
                        (
                            f"{args.year} shard{shard_id:02d} startup first origin complete; "
                            "switching to normal row-complete logging"
                        ),
                    )
                    startup_verbose = False
            finally:
                _release_named_lock(row_lock_handle)
        return 0
    except Exception as exc:
        _append_log_line(
            log_file,
            f"ERROR {args.year} shard{int(args.shard_id):02d}: {exc}\n{traceback.format_exc()}",
        )
        return 1


def _write_csv_header(csv_path: Path, station_order: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StationName", *station_order])


def _append_row_to_live_csv(csv_path: Path, station_order: list[str], row_payload: dict[str, Any]) -> None:
    row_idx = int(row_payload["row_idx"])
    origin = station_order[row_idx]
    cells = row_payload["cells"]
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([origin, *cells])


def _read_row_payload(row_path: Path) -> dict[str, Any]:
    return json.loads(row_path.read_text(encoding="utf-8"))


def _write_final_full_csv(
    *,
    final_csv_path: Path,
    station_order: list[str],
    rows_dir: Path,
    row_count: int | None = None,
) -> None:
    n = len(station_order)
    write_rows = n if row_count is None else min(n, max(0, int(row_count)))
    with final_csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["StationName", *station_order])
        for row_idx in range(write_rows):
            row_path = _row_file_path(rows_dir, row_idx)
            if not row_path.exists():
                raise FileNotFoundError(f"Missing row file for final merge: {row_path}")
            payload = _read_row_payload(row_path)
            payload_row_idx = int(payload.get("row_idx", -1))
            payload_origin = str(payload.get("origin", ""))
            expected_origin = station_order[row_idx]
            if payload_row_idx != row_idx:
                raise ValueError(
                    f"Row payload mismatch at merge: file={row_path.name} row_idx={payload_row_idx} expected={row_idx}"
                )
            if payload_origin != expected_origin:
                raise ValueError(
                    f"Row payload mismatch at merge: file={row_path.name} origin='{payload_origin}' expected='{expected_origin}'"
                )

            cells = payload["cells"]
            if len(cells) != n:
                raise ValueError(
                    f"Row payload length mismatch at merge: file={row_path.name} cells={len(cells)} expected={n}"
                )
            w.writerow([station_order[row_idx], *cells])


def _parse_years(years_csv: str) -> list[str]:
    years = [y.strip() for y in years_csv.split(",") if y.strip()]
    if not years:
        raise ValueError("No years specified.")
    return years


def _controller_run_year(
    *,
    year: str,
    station_order: list[str],
    shards: int,
    poll_sec: float,
    state_root: Path,
    model_cache: Path,
    max_legs: int,
    log_file: Path,
    python_exe: str,
    script_path: Path,
    resume: bool,
    output_dir: Path,
    output_tag: str,
    station_scope: str,
    row_limit: int = 0,
    skip_live_csv: bool = False,
    skip_final_csv: bool = False,
) -> None:
    year_state_dir = state_root / year
    rows_dir = year_state_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    controller_lock_file = _acquire_named_lock(
        year_state_dir / CONTROLLER_LOCK_FILENAME,
        payload={
            "pid": os.getpid(),
            "year": str(year),
            "shards": int(shards),
            "acquired_at": _ts(),
            "host_cwd": str(BASE_DIR),
        },
    )

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        live_csv_path = output_dir / f"{year}_OD_matrix_Nondominated_paths_full{output_tag}_LIVE.csv"
        final_csv_path = output_dir / f"{year}_OD_matrix_Nondominated_paths_full{output_tag}.csv"
        expected_rows = len(station_order) if int(row_limit) <= 0 else min(len(station_order), int(row_limit))

        if not resume:
            if rows_dir.exists():
                shutil.rmtree(rows_dir)
            rows_dir.mkdir(parents=True, exist_ok=True)
            manifest_path = year_state_dir / STATION_ORDER_MANIFEST_FILENAME
            if manifest_path.exists():
                manifest_path.unlink()
            if live_csv_path.exists():
                live_csv_path.unlink()
            if final_csv_path.exists():
                final_csv_path.unlink()

        _validate_or_write_station_order_manifest(
            year_state_dir=year_state_dir,
            rows_dir=rows_dir,
            year=year,
            station_scope=station_scope,
            station_order=station_order,
            resume=resume,
        )

        if not skip_live_csv:
            _write_csv_header(live_csv_path, station_order)
        appended_rows: set[int] = set()
        startup_baseline_rows = 0

        shard_procs: list[subprocess.Popen[Any]] = []

        def _terminate_shards() -> None:
            for p in shard_procs:
                if p.poll() is None:
                    try:
                        p.terminate()
                    except Exception:
                        pass
            deadline = time.time() + 10.0
            while time.time() < deadline:
                if all(p.poll() is not None for p in shard_procs):
                    break
                time.sleep(0.2)
            for p in shard_procs:
                if p.poll() is None:
                    try:
                        p.kill()
                    except Exception:
                        pass

        def _append_new_rows() -> None:
            row_files = sorted(rows_dir.glob("row_*.json"))
            for rf in row_files:
                name = rf.stem
                try:
                    row_idx = int(name.split("_")[1])
                except Exception:
                    continue
                if row_idx >= expected_rows:
                    continue
                if row_idx in appended_rows:
                    continue
                if not skip_live_csv:
                    payload = _read_row_payload(rf)
                    _append_row_to_live_csv(live_csv_path, station_order, payload)
                appended_rows.add(row_idx)

        _append_new_rows()
        startup_baseline_rows = len(appended_rows)
        _append_log_line(
            log_file,
            (
                f"{year} controller start resume={int(resume)} shards={shards} "
                f"existing_rows={startup_baseline_rows} total_rows={expected_rows} "
                f"station_count={len(station_order)} station_scope={station_scope} row_limit={int(row_limit)}"
            ),
        )

        for shard_id in range(shards):
            cmd = [
                python_exe,
                str(script_path),
                "--worker",
                "--year",
                str(year),
                "--shard-id",
                str(shard_id),
                "--shards",
                str(shards),
                "--rows-dir",
                str(rows_dir),
                "--model-cache",
                str(model_cache),
                "--max-legs",
                str(max_legs),
                "--log-file",
                str(log_file),
                "--row-limit",
                str(int(row_limit)),
                "--station-scope",
                str(station_scope),
            ]
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=str(BASE_DIR),
            )
            shard_procs.append(proc)
            _append_log_line(
                log_file,
                f"{year} controller launched shard{shard_id:02d} pid={proc.pid}",
            )

        def _handle_signal(signum: int, _frame: Any) -> None:
            _append_log_line(
                log_file,
                f"controller received signal {signum} for year {year}; terminating shards safely",
            )
            _terminate_shards()
            raise KeyboardInterrupt(f"signal {signum}")

        old_int = signal.getsignal(signal.SIGINT)
        old_term = signal.getsignal(signal.SIGTERM)
        ignore_hup = os.getenv("OD_IGNORE_SIGHUP", "0").strip().lower() in {"1", "true", "yes", "on"}
        old_hup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        if hasattr(signal, "SIGHUP") and not ignore_hup:
            signal.signal(signal.SIGHUP, _handle_signal)
        startup_phase = True
        last_startup_heartbeat = 0.0
        try:
            while True:
                _append_new_rows()
                alive = [p for p in shard_procs if p.poll() is None]
                if startup_phase:
                    if len(appended_rows) > startup_baseline_rows:
                        _append_log_line(
                            log_file,
                            (
                                f"{year} startup complete: first new row finished "
                                f"(rows_now={len(appended_rows)}/{len(station_order)}). "
                                "Returning to normal per-row completion logs."
                            ),
                        )
                        startup_phase = False
                    else:
                        now = time.time()
                        if now - last_startup_heartbeat >= max(20.0, float(poll_sec) * 8.0):
                            lock_dir = rows_dir / ".locks"
                            try:
                                lock_files = sum(1 for p in lock_dir.iterdir() if p.is_file())
                            except Exception:
                                lock_files = -1
                            _append_log_line(
                                log_file,
                            (
                                f"{year} startup heartbeat: rows={len(appended_rows)}/{expected_rows} "
                                f"alive_shards={len(alive)} lock_files={lock_files}"
                            ),
                            )
                            last_startup_heartbeat = now
                if not alive:
                    break
                time.sleep(max(float(poll_sec), 0.2))
            _append_new_rows()
        finally:
            signal.signal(signal.SIGINT, old_int)
            signal.signal(signal.SIGTERM, old_term)
            if hasattr(signal, "SIGHUP") and old_hup is not None and not ignore_hup:
                signal.signal(signal.SIGHUP, old_hup)

        bad_rc = [p.returncode for p in shard_procs if p.returncode not in (0, None)]
        if bad_rc:
            raise RuntimeError(f"{year}: one or more shards failed. return_codes={bad_rc}")

        expected = expected_rows
        completed_relevant = sum(1 for row_idx in appended_rows if row_idx < expected_rows)
        if completed_relevant != expected:
            missing = expected - completed_relevant
            raise RuntimeError(f"{year}: incomplete row set after shard run (missing_rows={missing}).")

        if skip_final_csv:
            _append_log_line(
                log_file,
                (
                    f"{year} controller complete: row_json_state_only rows={expected_rows} "
                    f"skip_live_csv={int(skip_live_csv)} skip_final_csv=1"
                ),
            )
        else:
            _write_final_full_csv(
                final_csv_path=final_csv_path,
                station_order=station_order,
                rows_dir=rows_dir,
                row_count=expected_rows,
            )
            if not skip_live_csv:
                shutil.copy2(final_csv_path, live_csv_path)
            _append_log_line(
                log_file,
                f"{year} controller complete: final_csv={final_csv_path.name} rows={expected_rows}",
            )
    finally:
        _release_named_lock(controller_lock_file)


def _controller_main(args: argparse.Namespace) -> int:
    log_file = Path(args.log_file).resolve()
    state_root = Path(args.state_root).resolve()
    years = _parse_years(args.years)

    script_path = Path(__file__).resolve()
    python_exe = sys.executable

    try:
        year_cache_overrides: dict[str, Path] = {}
        if getattr(args, "model_cache_2026", None):
            year_cache_overrides["2026"] = Path(args.model_cache_2026).resolve()
        if getattr(args, "model_cache_2035", None):
            year_cache_overrides["2035"] = Path(args.model_cache_2035).resolve()

        for year in years:
            station_order = _load_station_order(
                year=year,
                station_scope=str(getattr(args, "station_scope", "union")),
            )
            cache_for_year = year_cache_overrides.get(year, Path(args.model_cache).resolve())
            _controller_run_year(
                year=year,
                station_order=station_order,
                shards=int(args.shards),
                poll_sec=float(args.poll_sec),
                state_root=state_root,
                model_cache=cache_for_year,
                max_legs=int(args.max_legs),
                log_file=log_file,
                python_exe=python_exe,
                script_path=script_path,
                resume=bool(args.resume),
                output_dir=Path(args.output_dir).expanduser().resolve(),
                output_tag=str(args.output_tag or ""),
                station_scope=str(getattr(args, "station_scope", "union")),
                row_limit=int(getattr(args, "row_limit", 0) or 0),
                skip_live_csv=bool(getattr(args, "skip_live_csv", False)),
                skip_final_csv=bool(getattr(args, "skip_final_csv", False)),
            )
        return 0
    except Exception as exc:
        _append_log_line(log_file, f"ERROR controller: {exc}\n{traceback.format_exc()}")
        return 1


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build full-cell OD matrices with encoded nondominated path lists "
            "(05:00-06:59 departures), sharded by every 10th row."
        )
    )
    p.add_argument("--years", default="2026,2035", help="Comma-separated year order to process.")
    p.add_argument("--shards", type=int, default=10, help="Number of shards (default: 10).")
    p.add_argument("--poll-sec", type=float, default=2.0, help="Controller poll interval for live merge.")
    p.add_argument("--state-root", type=Path, default=DEFAULT_STATE_ROOT, help="State root for per-row shard outputs.")
    p.add_argument("--output-dir", type=Path, default=BASE_DIR, help="Directory for final and LIVE CSV outputs.")
    p.add_argument("--output-tag", default="", help="Optional suffix before .csv, e.g. _Optimised_v2.")
    p.add_argument("--model-cache", type=Path, default=DEFAULT_CACHE_FILE, help="Path to model cache pickle.")
    p.add_argument(
        "--model-cache-2026",
        type=Path,
        default=None,
        help="Optional year-specific model cache for 2026 workers (faster/lower-RAM startup).",
    )
    p.add_argument(
        "--model-cache-2035",
        type=Path,
        default=None,
        help="Optional year-specific model cache for 2035 workers (faster/lower-RAM startup).",
    )
    p.add_argument("--max-legs", type=int, default=50, help="Max legs per route query.")
    p.add_argument("--log-file", type=Path, default=DEFAULT_LOG_FILE, help="Monitor log file.")
    p.add_argument("--resume", action="store_true", help="Resume from existing row files instead of clean restart.")
    p.add_argument(
        "--station-scope",
        choices=("union", "year-specific"),
        default="union",
        help=(
            "Station universe to use. union keeps the previous shared canonical station order; "
            "year-specific filters to rows with {year}lines > 0 before collapsing Zürich HB aliases."
        ),
    )
    p.add_argument(
        "--row-limit",
        type=int,
        default=0,
        help="Optional smoke/mini limit: compute only the first N origin rows. 0 means full station set.",
    )
    p.add_argument(
        "--skip-live-csv",
        action="store_true",
        help="Do not materialize/update the large LIVE CSV; keep per-row JSON state only.",
    )
    p.add_argument(
        "--skip-final-csv",
        action="store_true",
        help="Do not materialize the large final CSV after all row JSONs are complete.",
    )

    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--year", default="", help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    p.add_argument("--rows-dir", type=Path, default=Path(""), help=argparse.SUPPRESS)
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.worker:
        if not args.year:
            raise ValueError("--worker requires --year")
        if not args.rows_dir:
            raise ValueError("--worker requires --rows-dir")
        rc = _worker_main(args)
        raise SystemExit(rc)

    rc = _controller_main(args)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
