#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DAY_START = 5 * 60
DAY_END_DEPART = 24 * 60
DAY_END_ARRIVAL = 26 * 60
MAX_THROUGH_DWELL_MIN = 15
DEFAULT_WINDOW_START = 10 * 60
DEFAULT_WINDOW_END = 12 * 60
DASH_VALUES = {"", "-", "—", "–", "―"}
TOKEN_RE = re.compile(r"^\s*(\d{1,2})\s*([inIN]?)\s*([Pp]?)\s*([Qq]?)\s*$")

YEAR_COLUMNS = {
    "2026": ("2026", "2026 (HH)", "2026 alt"),
    "2035": ("2035", "2035 (HH)", "2035 alt"),
}


@dataclass(frozen=True)
class ParsedToken:
    minute: int
    mode: str
    pflag: bool
    qflag: bool
    raw: str


@dataclass(frozen=True)
class ServiceSegment:
    service_name: str
    segment_index: int
    rows: tuple[dict[str, str], ...]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _fmt_time(minute: int) -> str:
    minute = int(minute) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _parse_hhmm(text: str) -> int:
    hh, mm = str(text).strip().split(":", 1)
    return int(hh) * 60 + int(mm)


def _parse_time_list(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(_parse_hhmm(part))
        except Exception:
            continue
    return out


def _parse_wait_min(text: str) -> int:
    text = str(text or "").strip().lower().replace(" ", "")
    if text.endswith("min"):
        text = text[:-3]
    return _as_int(text, -1)


def parse_token(cell: str) -> ParsedToken | None:
    cell = str(cell or "").strip()
    if cell in DASH_VALUES:
        return None
    match = TOKEN_RE.match(cell)
    if not match:
        return None
    minute = int(match.group(1))
    if not (0 <= minute <= 59):
        return None
    suffix = match.group(2).lower()
    if suffix == "":
        mode = "hourly"
    elif suffix == "n":
        mode = "even"
    elif suffix == "i":
        mode = "odd"
    else:
        return None
    return ParsedToken(
        minute=minute,
        mode=mode,
        pflag=bool(match.group(3)),
        qflag=bool(match.group(4)),
        raw=cell,
    )


def allowed_hours_for_token(token: ParsedToken, max_hour: int = 26) -> list[int]:
    if token.mode == "hourly":
        return list(range(5, max_hour + 1))
    if token.mode == "even":
        return [h for h in range(5, max_hour + 1) if h % 2 == 0]
    if token.mode == "odd":
        return [h for h in range(5, max_hour + 1) if h % 2 == 1]
    return []


def token_occurrences(token: ParsedToken, kind: str, max_hour: int = 26) -> list[int]:
    times = [h * 60 + token.minute for h in allowed_hours_for_token(token, max_hour=max_hour)]
    if kind == "ab":
        return [t for t in times if DAY_START <= t < DAY_END_DEPART]
    return [t for t in times if DAY_START <= t <= DAY_END_ARRIVAL]


def next_occurrence_from_sorted_times(sorted_times: list[int], threshold: int) -> int | None:
    idx = bisect.bisect_left(sorted_times, threshold)
    if idx >= len(sorted_times):
        return None
    return sorted_times[idx]


def required_threshold(prev_time: int, prev_row: dict[str, Any], next_row: dict[str, Any]) -> int:
    if prev_row["station"] == next_row["station"] and prev_row["kind"] == "an" and next_row["kind"] == "ab":
        threshold = prev_time
    else:
        threshold = prev_time + 1
    if next_row["token"].pflag:
        threshold = max(threshold, ((prev_time // 60) + 1) * 60)
    if next_row["token"].qflag:
        threshold = max(threshold, prev_time + 60)
    return threshold


def _is_same_station_dwell_pair(arr_row: dict[str, Any], dep_row: dict[str, Any]) -> bool:
    return arr_row["kind"] == "an" and dep_row["kind"] == "ab" and arr_row["station"] == dep_row["station"]


def _is_turnback_split_candidate(active_rows: list[dict[str, Any]], arrival_idx: int) -> bool:
    if arrival_idx <= 0 or arrival_idx + 2 >= len(active_rows):
        return False
    prev_row, arr_row = active_rows[arrival_idx - 1], active_rows[arrival_idx]
    dep_row, next_row = active_rows[arrival_idx + 1], active_rows[arrival_idx + 2]
    if not (prev_row["kind"] == "ab" and arr_row["kind"] == "an" and dep_row["kind"] == "ab" and next_row["kind"] == "an"):
        return False
    if arr_row["station"] != dep_row["station"]:
        return False
    return prev_row["station"] == next_row["station"]


def _next_time_for_row(prev_time: int, prev_row: dict[str, Any], next_row: dict[str, Any], row_times: list[int]) -> int | None:
    return next_occurrence_from_sorted_times(row_times, required_threshold(prev_time, prev_row, next_row))


def _can_continue_same_run(
    arr_time: int,
    dep_time: int,
    arr_row: dict[str, Any],
    dep_row: dict[str, Any],
    *,
    active_rows: list[dict[str, Any]] | None,
    arrival_idx: int | None,
) -> bool:
    if not _is_same_station_dwell_pair(arr_row, dep_row):
        return True
    if active_rows is not None and arrival_idx is not None and _is_turnback_split_candidate(active_rows, arrival_idx):
        return False
    return (dep_time - arr_time) <= MAX_THROUGH_DWELL_MIN


def instantiate_trip_events(active_rows: list[dict[str, Any]]) -> list[list[tuple[str, str, int]]]:
    if not active_rows:
        return []
    row_times = [token_occurrences(r["token"], r["kind"], max_hour=26) for r in active_rows]
    trips: list[list[tuple[str, str, int]]] = []
    departure_start_indices = [i for i, r in enumerate(active_rows) if r["kind"] == "ab"]

    for start_idx in departure_start_indices:
        first = active_rows[start_idx]
        first_times = [t for t in row_times[start_idx] if DAY_START <= t < DAY_END_DEPART]
        for t0 in first_times:
            trip: list[tuple[str, str, int]] = [(first["station"], first["kind"], t0)]
            prev_time = t0
            prev_row = first
            idx = start_idx + 1
            valid = True
            while idx < len(active_rows):
                cur_row = active_rows[idx]
                next_t = _next_time_for_row(prev_time, prev_row, cur_row, row_times[idx])
                if next_t is None or next_t > DAY_END_ARRIVAL:
                    valid = len(trip) >= 2 and trip[-1][1] == "an"
                    break

                trip.append((cur_row["station"], cur_row["kind"], next_t))
                prev_time = next_t
                prev_row = cur_row

                if cur_row["kind"] == "an":
                    if idx == len(active_rows) - 1:
                        valid = True
                        break
                    next_row = active_rows[idx + 1]
                    if _is_same_station_dwell_pair(cur_row, next_row):
                        dep_t = _next_time_for_row(prev_time, cur_row, next_row, row_times[idx + 1])
                        if dep_t is None or dep_t >= DAY_END_DEPART:
                            valid = True
                            break
                        if not _can_continue_same_run(prev_time, dep_t, cur_row, next_row, active_rows=active_rows, arrival_idx=idx):
                            valid = True
                            break
                        trip.append((next_row["station"], next_row["kind"], dep_t))
                        prev_time = dep_t
                        prev_row = next_row
                        idx += 2
                        continue
                idx += 1
            if valid and len(trip) >= 2 and trip[-1][1] == "an":
                trips.append(trip)
    return trips


def trip_signature(trip: list[tuple[str, str, int]]) -> str:
    return "|".join(f"{station}::{kind}::{minute}" for station, kind, minute in trip)


def _trip_hash(trip: list[tuple[str, str, int]]) -> str:
    return hashlib.sha1(trip_signature(trip).encode("utf-8")).hexdigest()[:16]


def _is_split_row(row: dict[str, str]) -> bool:
    return str(row.get("station", "")).strip().lower() == "split" or str(row.get("kind", "")).strip().lower() == "split"


def load_timetable_segments(path: Path) -> list[ServiceSegment]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or len(reader.fieldnames) < 3:
            raise ValueError(f"{path.name} does not have the expected timetable structure.")
        original_columns = list(reader.fieldnames)
        rows: list[dict[str, str]] = []

        first = {"station": original_columns[0], "kind": original_columns[1], "rownum": "1"}
        for col in original_columns[2:]:
            first[col] = col
        rows.append(first)

        for idx, raw in enumerate(reader, start=2):
            row = {"station": str(raw.get(original_columns[0], "")).strip(), "kind": str(raw.get(original_columns[1], "")).strip(), "rownum": str(idx)}
            for col in original_columns[2:]:
                row[col] = str(raw.get(col, "")).strip()
            rows.append(row)

    segments: list[ServiceSegment] = []
    current_service: str | None = None
    current_rows: list[dict[str, str]] = []

    def flush(service_name: str | None, block_rows: list[dict[str, str]]) -> None:
        if not service_name:
            return
        segment_rows: list[dict[str, str]] = []
        segment_index = 1
        for row in block_rows:
            if _is_split_row(row):
                if segment_rows:
                    segments.append(ServiceSegment(service_name=service_name, segment_index=segment_index, rows=tuple(segment_rows)))
                    segment_index += 1
                    segment_rows = []
            else:
                segment_rows.append(row)
        if segment_rows:
            segments.append(ServiceSegment(service_name=service_name, segment_index=segment_index, rows=tuple(segment_rows)))

    for row in rows:
        kind = str(row.get("kind", "")).strip().lower()
        if kind == "an/ab":
            flush(current_service, current_rows)
            current_service = str(row.get("station", "")).strip()
            current_rows = []
        elif current_service is not None:
            current_rows.append(row)

    flush(current_service, current_rows)
    return segments


def active_rows_for_segment(segment: ServiceSegment, year_col: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in segment.rows:
        token = parse_token(row.get(year_col, ""))
        if token is None:
            continue
        rows.append(
            {
                "rownum": row.get("rownum", ""),
                "station": row.get("station", ""),
                "kind": str(row.get("kind", "")).strip().lower(),
                "token": token,
                "raw": row.get(year_col, ""),
            }
        )
    while rows and rows[0]["kind"] == "an":
        rows.pop(0)
    while rows and rows[-1]["kind"] == "ab":
        rows.pop()
    return rows


def valid_ab_an_sequence(rows: list[dict[str, Any]]) -> bool:
    expected = "ab"
    for row in rows:
        if row["kind"] != expected:
            return False
        expected = "an" if expected == "ab" else "ab"
    return bool(rows) and rows[-1]["kind"] == "an"


def _expand_repeating_times(source_times: list[int], start_min: int, end_min: int, cycle_minutes: int = 120) -> list[int]:
    expanded: set[int] = set()
    for source_minute in source_times:
        delta = (source_minute - start_min) % cycle_minutes
        minute = start_min + delta
        while minute < end_min:
            expanded.add(minute)
            minute += cycle_minutes
    return sorted(expanded)


def load_high_flow_transfers(
    path: Path,
    threshold_pax: float,
    *,
    window_start: int,
    window_end: int,
) -> tuple[set[str], set[tuple[str, str, int, str]], dict[tuple[str, str, int, str], float], dict[tuple[str, str, int, str], dict[str, Any]], dict[tuple[str, str, int], dict[str, Any]]]:
    stations: set[str] = set()
    high_keys: set[tuple[str, str, int, str]] = set()
    key_pax: dict[tuple[str, str, int, str], float] = {}
    key_meta: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    event_weights: dict[tuple[str, str, int], dict[str, Any]] = defaultdict(
        lambda: {
            "pattern_total_daily_pax": 0.0,
            "time_share_daily_pax": 0.0,
            "pattern_count": 0,
            "partner_lines": set(),
        }
    )

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pax = _as_float(row.get("Number of Affected Daily Pax"), 0.0)
            if pax < threshold_pax:
                continue
            station = str(row.get("Transfer station", "")).strip()
            from_service = str(row.get("Arrival Line", "")).strip()
            to_service = str(row.get("Departure Line", "")).strip()
            wait_min = _parse_wait_min(str(row.get("Transfer Time", "")))
            if not station or not from_service or not to_service or wait_min < 0:
                continue
            stations.add(station)
            key = (station, from_service, wait_min, to_service)
            high_keys.add(key)
            key_pax[key] = key_pax.get(key, 0.0) + pax
            key_meta[key] = row

            raw_arrival_times = _parse_time_list(str(row.get("Arrival Time", "")))
            raw_departure_times = _parse_time_list(str(row.get("Departure Time", "")))

            # The transfer summary stores representative path times from the
            # morning window. For board inspection we project those patterns
            # onto the requested clock window with the same 120-minute periodic
            # convention used by the path-allocation step.
            arrival_times = _expand_repeating_times(raw_arrival_times, window_start, window_end)
            departure_times = _expand_repeating_times(raw_departure_times, window_start, window_end)
            daily_arrival_occurrences = _expand_repeating_times(raw_arrival_times, DAY_START, DAY_END_DEPART)
            daily_departure_occurrences = _expand_repeating_times(raw_departure_times, DAY_START, DAY_END_DEPART)
            daily_occurrence_count = max(1, len(set(daily_arrival_occurrences) | set(daily_departure_occurrences)))
            pax_share = pax / daily_occurrence_count

            for minute in arrival_times:
                ev = event_weights[(station, from_service, minute)]
                ev["pattern_total_daily_pax"] += pax
                ev["time_share_daily_pax"] += pax_share
                ev["pattern_count"] += 1
                ev["partner_lines"].add(f"to:{to_service}")
            for minute in departure_times:
                ev = event_weights[(station, to_service, minute)]
                ev["pattern_total_daily_pax"] += pax
                ev["time_share_daily_pax"] += pax_share
                ev["pattern_count"] += 1
                ev["partner_lines"].add(f"from:{from_service}")

    return stations, high_keys, key_pax, key_meta, event_weights


def _prev_station(trip: list[tuple[str, str, int]], idx: int) -> str:
    station = trip[idx][0]
    for j in range(idx - 1, -1, -1):
        if trip[j][0] != station:
            return trip[j][0]
    return ""


def _next_station(trip: list[tuple[str, str, int]], idx: int) -> str:
    station = trip[idx][0]
    for j in range(idx + 1, len(trip)):
        if trip[j][0] != station:
            return trip[j][0]
    return ""


def build_board_rows(
    *,
    year: str,
    segments: list[ServiceSegment],
    stations: set[str],
    event_weights: dict[tuple[str, str, int], dict[str, Any]],
    window_start: int,
    window_end: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_trip_sigs: set[str] = set()
    seen_board_rows: set[tuple[Any, ...]] = set()

    for segment in segments:
        for year_col in YEAR_COLUMNS[year]:
            active_rows = active_rows_for_segment(segment, year_col)
            if not active_rows or not valid_ab_an_sequence(active_rows):
                continue
            for trip in instantiate_trip_events(active_rows):
                sig = trip_signature(trip)
                if sig in seen_trip_sigs:
                    continue
                seen_trip_sigs.add(sig)
                trip_hash = _trip_hash(trip)
                for idx, (station, kind, minute) in enumerate(trip):
                    if station not in stations or not (window_start <= minute < window_end):
                        continue
                    weight = event_weights.get((station, segment.service_name, minute))
                    partner_lines: list[str] = []
                    pattern_total = 0.0
                    time_share = 0.0
                    pattern_count = 0
                    if weight:
                        partner_lines = sorted(str(x) for x in weight["partner_lines"])
                        pattern_total = float(weight["pattern_total_daily_pax"])
                        time_share = float(weight["time_share_daily_pax"])
                        pattern_count = int(weight["pattern_count"])
                    out = {
                        "year": year,
                        "transfer_station": station,
                        "board_type": "arrival" if kind == "an" else "departure",
                        "time": _fmt_time(minute),
                        "minute": minute,
                        "line": segment.service_name,
                        "line_previous_station": _prev_station(trip, idx),
                        "line_next_station": _next_station(trip, idx),
                        "segment_index": segment.segment_index,
                        "raw_year_column": year_col,
                        "trip_signature_hash": trip_hash,
                        "high_flow_transfer_pattern_count_touching_event": pattern_count,
                        "high_flow_transfer_daily_pax_time_share_touching_event": round(time_share, 6),
                        "high_flow_transfer_daily_pax_pattern_total_touching_event": round(pattern_total, 6),
                        "high_flow_transfer_partner_lines": "; ".join(partner_lines),
                    }
                    row_key = (
                        out["year"],
                        out["transfer_station"],
                        out["board_type"],
                        out["minute"],
                        out["line"],
                        out["line_previous_station"],
                        out["line_next_station"],
                    )
                    if row_key in seen_board_rows:
                        continue
                    seen_board_rows.add(row_key)
                    rows.append(out)
    # Suffix trip instances are useful in the router, but they can create a
    # board event that looks like it starts or ends at an intermediate station.
    # Keep true terminals, but if a same line/time/direction has a non-empty
    # adjacent station context, prefer that over the artificial suffix row.
    departure_context: dict[tuple[Any, ...], bool] = defaultdict(bool)
    arrival_context: dict[tuple[Any, ...], bool] = defaultdict(bool)
    for row in rows:
        if row["board_type"] == "departure":
            key = (row["year"], row["transfer_station"], row["minute"], row["line"], row["line_next_station"])
            departure_context[key] = departure_context[key] or bool(row["line_previous_station"])
        else:
            key = (row["year"], row["transfer_station"], row["minute"], row["line"], row["line_previous_station"])
            arrival_context[key] = arrival_context[key] or bool(row["line_next_station"])

    pruned_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["board_type"] == "departure":
            key = (row["year"], row["transfer_station"], row["minute"], row["line"], row["line_next_station"])
            if departure_context[key] and not row["line_previous_station"]:
                continue
        else:
            key = (row["year"], row["transfer_station"], row["minute"], row["line"], row["line_previous_station"])
            if arrival_context[key] and not row["line_next_station"]:
                continue
        pruned_rows.append(row)

    pruned_rows.sort(
        key=lambda r: (
            r["year"],
            r["transfer_station"],
            int(r["minute"]),
            r["board_type"],
            r["line"],
            r["line_previous_station"],
            r["line_next_station"],
        )
    )
    return pruned_rows


def write_board_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "year",
        "transfer_station",
        "board_type",
        "time",
        "minute",
        "line",
        "line_previous_station",
        "line_next_station",
        "segment_index",
        "raw_year_column",
        "trip_signature_hash",
        "high_flow_transfer_pattern_count_touching_event",
        "high_flow_transfer_daily_pax_time_share_touching_event",
        "high_flow_transfer_daily_pax_pattern_total_touching_event",
        "high_flow_transfer_partner_lines",
    ]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _empty_baseline_state(year: str) -> dict[str, Any]:
    return {
        "year": year,
        "next_row": 0,
        "complete": False,
        "all_path_pax": 0.0,
        "all_path_weighted_transfer_share": 0.0,
        "all_path_pax_transfer_min": 0.0,
        "all_path_pax_total_min": 0.0,
        "high_flow_event_pax": 0.0,
        "high_flow_event_weighted_transfer_share": 0.0,
        "high_flow_event_pax_wait_min": 0.0,
        "high_flow_event_pax_total_min": 0.0,
        "high_flow_event_records": 0,
        "station": {},
    }


def _load_state(path: Path, year: str, fresh: bool) -> dict[str, Any]:
    if fresh or not path.exists():
        return _empty_baseline_state(year)
    with path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    if state.get("year") != year:
        raise ValueError(f"{path} belongs to year {state.get('year')}, not {year}")
    return state


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(path)


def _station_stats(state: dict[str, Any], station: str) -> dict[str, Any]:
    stats = state["station"].get(station)
    if stats is None:
        stats = {
            "event_pax": 0.0,
            "event_weighted_transfer_share": 0.0,
            "event_pax_wait_min": 0.0,
            "event_pax_total_min": 0.0,
            "event_records": 0,
        }
        state["station"][station] = stats
    return stats


def scan_path_alloc_baseline(
    *,
    year: str,
    input_csv: Path,
    high_keys: set[tuple[str, str, int, str]],
    state_path: Path,
    fresh: bool,
    progress_every_rows: int,
    commit_every_rows: int,
    max_rows: int,
) -> dict[str, Any]:
    csv.field_size_limit(sys.maxsize)
    state = _load_state(state_path, year, fresh=fresh)
    if state.get("complete"):
        _log(f"{year}: baseline state already complete at {state_path}")
        return state

    next_row = int(state.get("next_row", 0))
    _log(f"{year}: baseline scan start input={input_csv} resume_row={next_row}")

    processed_this_run = 0
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "StationName":
            raise ValueError(f"{input_csv.name}: first column must be StationName")
        total_rows = len(header) - 1
        for _ in range(next_row):
            next(reader)
        current_row = next_row

        for row in reader:
            if max_rows and processed_this_run >= max_rows:
                break
            for cell in row[1:]:
                if not cell:
                    continue
                try:
                    obj = json.loads(cell)
                except json.JSONDecodeError:
                    continue
                paths = obj.get("paths") or []
                if not isinstance(paths, list):
                    continue
                for path in paths:
                    if not isinstance(path, dict):
                        continue
                    pax = _as_float(path.get("daily_demand_allocated"), 0.0)
                    if pax <= 0.0:
                        continue
                    transfer_min = _as_float(path.get("transfer_min"), 0.0)
                    dwell_min = _as_float(path.get("dwell_min"), 0.0)
                    rolling_min = _as_float(path.get("rolling_min"), 0.0)
                    total_min = transfer_min + dwell_min + rolling_min
                    if total_min <= 0.0:
                        continue
                    path_share = transfer_min / total_min
                    state["all_path_pax"] += pax
                    state["all_path_weighted_transfer_share"] += pax * path_share
                    state["all_path_pax_transfer_min"] += pax * transfer_min
                    state["all_path_pax_total_min"] += pax * total_min

                    events = path.get("transfer_events") or []
                    if not isinstance(events, list) or not events:
                        continue
                    seen_events: set[tuple[str, str, int, str, int, int]] = set()
                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        station = str(ev.get("station", "")).strip()
                        from_service = str(ev.get("from_service", "")).strip()
                        to_service = str(ev.get("to_service", "")).strip()
                        wait_min = _as_int(ev.get("transfer_wait_min"), -1)
                        arrive_min = _as_int(ev.get("arrive_prev_min"), -1)
                        depart_min = _as_int(ev.get("depart_next_min"), -1)
                        if (station, from_service, wait_min, to_service) not in high_keys:
                            continue
                        dedupe = (station, from_service, wait_min, to_service, arrive_min, depart_min)
                        if dedupe in seen_events:
                            continue
                        seen_events.add(dedupe)
                        event_share = wait_min / total_min if total_min > 0 else 0.0
                        state["high_flow_event_pax"] += pax
                        state["high_flow_event_weighted_transfer_share"] += pax * event_share
                        state["high_flow_event_pax_wait_min"] += pax * wait_min
                        state["high_flow_event_pax_total_min"] += pax * total_min
                        state["high_flow_event_records"] += 1
                        st = _station_stats(state, station)
                        st["event_pax"] += pax
                        st["event_weighted_transfer_share"] += pax * event_share
                        st["event_pax_wait_min"] += pax * wait_min
                        st["event_pax_total_min"] += pax * total_min
                        st["event_records"] += 1

            current_row += 1
            processed_this_run += 1
            state["next_row"] = current_row
            if processed_this_run % commit_every_rows == 0:
                _save_state(state_path, state)
            if processed_this_run % progress_every_rows == 0:
                _log(f"{year}: baseline progress rows={current_row}/{total_rows}")

    if not max_rows or processed_this_run < max_rows:
        state["complete"] = True
    _save_state(state_path, state)
    _log(f"{year}: baseline scan saved complete={state['complete']} rows={state['next_row']}")
    return state


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


def write_baseline_csvs(base_path: Path, station_path: Path, states: list[dict[str, Any]]) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    with base_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "scope",
                "weighted_transfer_share",
                "weighted_transfer_share_percent",
                "weighted_transfer_minutes",
                "weighted_total_minutes",
                "weight_daily_pax_or_event_pax",
                "event_records",
                "source_note",
            ],
        )
        writer.writeheader()
        for state in states:
            year = state["year"]
            all_share = _ratio(float(state["all_path_weighted_transfer_share"]), float(state["all_path_pax"]))
            writer.writerow(
                {
                    "year": year,
                    "scope": "all_paths_with_allocated_demand",
                    "weighted_transfer_share": round(all_share, 8),
                    "weighted_transfer_share_percent": round(all_share * 100.0, 4),
                    "weighted_transfer_minutes": round(_ratio(float(state["all_path_pax_transfer_min"]), float(state["all_path_pax"])), 6),
                    "weighted_total_minutes": round(_ratio(float(state["all_path_pax_total_min"]), float(state["all_path_pax"])), 6),
                    "weight_daily_pax_or_event_pax": round(float(state["all_path_pax"]), 6),
                    "event_records": "",
                    "source_note": "Each path weighted once by daily_demand_allocated; numerator uses path transfer_min.",
                }
            )
            event_share = _ratio(float(state["high_flow_event_weighted_transfer_share"]), float(state["high_flow_event_pax"]))
            writer.writerow(
                {
                    "year": year,
                    "scope": "high_flow_transfer_events_ge_threshold",
                    "weighted_transfer_share": round(event_share, 8),
                    "weighted_transfer_share_percent": round(event_share * 100.0, 4),
                    "weighted_transfer_minutes": round(_ratio(float(state["high_flow_event_pax_wait_min"]), float(state["high_flow_event_pax"])), 6),
                    "weighted_total_minutes": round(_ratio(float(state["high_flow_event_pax_total_min"]), float(state["high_flow_event_pax"])), 6),
                    "weight_daily_pax_or_event_pax": round(float(state["high_flow_event_pax"]), 6),
                    "event_records": int(state["high_flow_event_records"]),
                    "source_note": "Each high-flow transfer event weighted by the path daily_demand_allocated using event wait / path total time.",
                }
            )

    with station_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "transfer_station",
                "weighted_transfer_share",
                "weighted_transfer_share_percent",
                "weighted_transfer_minutes",
                "weighted_total_minutes",
                "weight_event_daily_pax",
                "event_records",
            ],
        )
        writer.writeheader()
        for state in states:
            for station, stats in sorted(state["station"].items()):
                share = _ratio(float(stats["event_weighted_transfer_share"]), float(stats["event_pax"]))
                writer.writerow(
                    {
                        "year": state["year"],
                        "transfer_station": station,
                        "weighted_transfer_share": round(share, 8),
                        "weighted_transfer_share_percent": round(share * 100.0, 4),
                        "weighted_transfer_minutes": round(_ratio(float(stats["event_pax_wait_min"]), float(stats["event_pax"])), 6),
                        "weighted_total_minutes": round(_ratio(float(stats["event_pax_total_min"]), float(stats["event_pax"])), 6),
                        "weight_event_daily_pax": round(float(stats["event_pax"]), 6),
                        "event_records": int(stats["event_records"]),
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare transfer-station arrival/departure boards and baseline weighted transfer-share metrics."
    )
    parser.add_argument("--timetable-csv", type=Path, default=Path("FullTimetableHistory2026_2035.csv"))
    parser.add_argument("--transfer-2026", type=Path, default=Path("Identified_All_Transfers_2026_transfer_discounted.csv"))
    parser.add_argument("--transfer-2035", type=Path, default=Path("Identified_All_Transfers_2035_transfer_discounted.csv"))
    parser.add_argument(
        "--pathalloc-2026",
        type=Path,
        default=Path("<USER_HOME>/od_fast_output_transfer_discounted/2026_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv"),
    )
    parser.add_argument(
        "--pathalloc-2035",
        type=Path,
        default=Path("<USER_HOME>/od_fast_output_transfer_discounted/2035_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv"),
    )
    parser.add_argument("--threshold-pax", type=float, default=95.0)
    parser.add_argument("--window-start", default="10:00")
    parser.add_argument("--window-end", default="12:00")
    parser.add_argument("--output-board-csv", type=Path, default=Path("Transfer_Station_Boards_2026_2035_ge95_1000_1159.csv"))
    parser.add_argument("--output-baseline-csv", type=Path, default=Path("Transfer_Time_Share_Baseline_2026_2035.csv"))
    parser.add_argument("--output-baseline-by-station-csv", type=Path, default=Path("Transfer_Time_Share_Baseline_By_Station_2026_2035.csv"))
    parser.add_argument("--state-dir", type=Path, default=Path("/tmp/transfer_optimization_baseline_state"))
    parser.add_argument("--boards-only", action="store_true", help="Only write the small board CSV; skip streaming the huge path-allocation CSVs.")
    parser.add_argument("--fresh-baseline", action="store_true", help="Discard previous baseline resume state.")
    parser.add_argument("--progress-every-rows", type=int, default=10)
    parser.add_argument("--commit-every-rows", type=int, default=2)
    parser.add_argument("--max-path-rows", type=int, default=0, help="Debug/testing only: stop each path scan after this many newly processed rows.")
    args = parser.parse_args()

    window_start = _parse_hhmm(args.window_start)
    window_end = _parse_hhmm(args.window_end)
    if window_end <= window_start:
        raise ValueError("--window-end must be later than --window-start")

    transfer_paths = {"2026": args.transfer_2026.resolve(), "2035": args.transfer_2035.resolve()}
    high_data: dict[str, dict[str, Any]] = {}
    for year, path in transfer_paths.items():
        if not path.exists():
            raise FileNotFoundError(path)
        stations, high_keys, key_pax, key_meta, event_weights = load_high_flow_transfers(
            path,
            args.threshold_pax,
            window_start=window_start,
            window_end=window_end,
        )
        high_data[year] = {
            "stations": stations,
            "high_keys": high_keys,
            "key_pax": key_pax,
            "key_meta": key_meta,
            "event_weights": event_weights,
        }
        _log(f"{year}: high-flow transfer patterns={len(high_keys)} stations={len(stations)} threshold_pax={args.threshold_pax}")

    timetable_path = args.timetable_csv.resolve()
    if not timetable_path.exists():
        raise FileNotFoundError(timetable_path)
    segments = load_timetable_segments(timetable_path)
    _log(f"Loaded timetable segments={len(segments)} from {timetable_path}")

    board_rows: list[dict[str, Any]] = []
    for year in ("2026", "2035"):
        rows = build_board_rows(
            year=year,
            segments=segments,
            stations=high_data[year]["stations"],
            event_weights=high_data[year]["event_weights"],
            window_start=window_start,
            window_end=window_end,
        )
        board_rows.extend(rows)
        _log(f"{year}: board rows in window {_fmt_time(window_start)}-{_fmt_time(window_end - 1)} = {len(rows)}")
    write_board_csv(args.output_board_csv.resolve(), board_rows)
    _log(f"Wrote board CSV: {args.output_board_csv.resolve()}")

    if args.boards_only:
        _log("boards-only requested; skipping huge path-allocation baseline scan.")
        return

    pathalloc = {"2026": args.pathalloc_2026.resolve(), "2035": args.pathalloc_2035.resolve()}
    states: list[dict[str, Any]] = []
    for year in ("2026", "2035"):
        if not pathalloc[year].exists():
            raise FileNotFoundError(pathalloc[year])
        state_path = args.state_dir.resolve() / f"{year}_baseline_state.json"
        state = scan_path_alloc_baseline(
            year=year,
            input_csv=pathalloc[year],
            high_keys=high_data[year]["high_keys"],
            state_path=state_path,
            fresh=bool(args.fresh_baseline),
            progress_every_rows=max(1, args.progress_every_rows),
            commit_every_rows=max(1, args.commit_every_rows),
            max_rows=max(0, args.max_path_rows),
        )
        states.append(state)

    write_baseline_csvs(args.output_baseline_csv.resolve(), args.output_baseline_by_station_csv.resolve(), states)
    _log(f"Wrote baseline CSV: {args.output_baseline_csv.resolve()}")
    _log(f"Wrote baseline-by-station CSV: {args.output_baseline_by_station_csv.resolve()}")


if __name__ == "__main__":
    main()
