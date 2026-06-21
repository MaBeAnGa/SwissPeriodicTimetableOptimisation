#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from prepare_transfer_boards_and_baseline import (
    DAY_END_DEPART,
    DAY_START,
    YEAR_COLUMNS,
    _as_float,
    _as_int,
    _expand_repeating_times,
    _fmt_time,
    _parse_time_list,
    _parse_wait_min,
    _prev_station,
    _next_station,
    _trip_hash,
    active_rows_for_segment,
    build_board_rows,
    instantiate_trip_events,
    load_high_flow_transfers,
    load_timetable_segments,
    valid_ab_an_sequence,
)


DEFAULT_MIN_TRANSFER = 3


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > 0.0 else 0.0


def _load_min_transfer(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = str(row.get("StationName", "")).strip()
            if not station:
                continue
            val = _as_int(row.get("MinimumTransferTime"), DEFAULT_MIN_TRANSFER)
            out[station] = max(0, val)
    if "Zürich HB" in out:
        for alias in ("Zürich HB (21-22)", "Zürich HB (31-34)", "Zürich HB (41-44)"):
            out.setdefault(alias, out["Zürich HB"])
    # Pair-specific routing treats the Zürich HB complex as 7 min by default,
    # with a special shorter transfer only when both services are within the
    # 31-34 platform group. In these station-level transfer tables, a transfer
    # station of "Zürich HB (31-34)" represents that within-group case.
    if "Zürich HB (31-34)" in out:
        out["Zürich HB (31-34)"] = 4
    return out


def _min_transfer(station: str, lookup: dict[str, int]) -> int:
    return int(lookup.get(station, DEFAULT_MIN_TRANSFER))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _event_id(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(k, ""))
        for k in (
            "year",
            "transfer_station",
            "board_type",
            "minute",
            "line",
            "line_previous_station",
            "line_next_station",
        )
    )


def _build_event_ledger(
    *,
    years: list[str],
    segments: list[Any],
    high_data: dict[str, dict[str, Any]],
    min_transfer_lookup: dict[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for year in years:
        board_rows = build_board_rows(
            year=year,
            segments=segments,
            stations=high_data[year]["stations"],
            event_weights=high_data[year]["event_weights"],
            window_start=DAY_START,
            window_end=DAY_END_DEPART,
        )
        for row in board_rows:
            station = str(row["transfer_station"])
            out = dict(row)
            out["event_id"] = _event_id(out)
            out["minimum_transfer_time"] = _min_transfer(station, min_transfer_lookup)
            rows.append(out)
        _log(f"{year}: event ledger rows={len(board_rows)}")
    return rows


def _compact_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": row["event_id"],
        "line": row["line"],
        "time": row["time"],
        "minute": int(row["minute"]),
        "prev": row["line_previous_station"],
        "next": row["line_next_station"],
        "event_pax_time_share": _as_float(row.get("high_flow_transfer_daily_pax_time_share_touching_event"), 0.0),
        "event_pax_pattern_total": _as_float(row.get("high_flow_transfer_daily_pax_pattern_total_touching_event"), 0.0),
        "pattern_count": _as_int(row.get("high_flow_transfer_pattern_count_touching_event"), 0),
        "partners": row.get("high_flow_transfer_partner_lines", ""),
    }


def _build_near_misses(
    ledger_rows: list[dict[str, Any]],
    *,
    early_window_min: int,
    late_window_min: int,
) -> list[dict[str, Any]]:
    by_station_year: dict[tuple[str, str], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"arrival": [], "departure": []})
    for row in ledger_rows:
        by_station_year[(row["year"], row["transfer_station"])][row["board_type"]].append(row)

    out: list[dict[str, Any]] = []
    for (year, station), groups in sorted(by_station_year.items()):
        arrivals = sorted(groups["arrival"], key=lambda r: int(r["minute"]))
        departures = sorted(groups["departure"], key=lambda r: int(r["minute"]))
        min_transfer = _as_int(arrivals[0].get("minimum_transfer_time") if arrivals else None, DEFAULT_MIN_TRANSFER)
        if not arrivals or not departures:
            continue
        for arr in arrivals:
            arr_min = int(arr["minute"])
            for dep in departures:
                dep_min = int(dep["minute"])
                wait = dep_min - arr_min
                if wait < -early_window_min:
                    continue
                if wait > min_transfer + late_window_min:
                    continue
                if arr["line"] == dep["line"] and arr["line_previous_station"] == dep["line_next_station"]:
                    # Same physical train turning back/continuing is not a passenger transfer.
                    continue
                status = "currently_feasible" if wait >= min_transfer else "currently_infeasible"
                potential = min(
                    _as_float(arr.get("high_flow_transfer_daily_pax_time_share_touching_event"), 0.0),
                    _as_float(dep.get("high_flow_transfer_daily_pax_time_share_touching_event"), 0.0),
                )
                out.append(
                    {
                        "year": year,
                        "transfer_station": station,
                        "minimum_transfer_time": min_transfer,
                        "arrival_event_id": arr["event_id"],
                        "arrival_line": arr["line"],
                        "arrival_prev_station": arr["line_previous_station"],
                        "arrival_next_station": arr["line_next_station"],
                        "arrival_time": arr["time"],
                        "arrival_minute": arr_min,
                        "departure_event_id": dep["event_id"],
                        "departure_line": dep["line"],
                        "departure_prev_station": dep["line_previous_station"],
                        "departure_next_station": dep["line_next_station"],
                        "departure_time": dep["time"],
                        "departure_minute": dep_min,
                        "wait_minutes": wait,
                        "transfer_margin_minutes": wait - min_transfer,
                        "status": status,
                        "estimated_potential_daily_pax_time_share": round(potential, 6),
                        "arrival_event_daily_pax_time_share": arr.get("high_flow_transfer_daily_pax_time_share_touching_event", 0.0),
                        "departure_event_daily_pax_time_share": dep.get("high_flow_transfer_daily_pax_time_share_touching_event", 0.0),
                        "arrival_partner_lines": arr.get("high_flow_transfer_partner_lines", ""),
                        "departure_partner_lines": dep.get("high_flow_transfer_partner_lines", ""),
                    }
                )
    out.sort(
        key=lambda r: (
            r["year"],
            -float(r["estimated_potential_daily_pax_time_share"]),
            abs(int(r["transfer_margin_minutes"])),
            r["transfer_station"],
            int(r["arrival_minute"]),
        )
    )
    return out


def _same_station_dwell_before(trip: list[tuple[str, str, int]], dep_idx: int) -> int | None:
    if dep_idx <= 0:
        return None
    prev = trip[dep_idx - 1]
    dep = trip[dep_idx]
    if prev[0] == dep[0] and prev[1] == "an" and dep[1] == "ab":
        return max(0, dep[2] - prev[2])
    return None


def _same_station_dwell_after(trip: list[tuple[str, str, int]], arr_idx: int) -> int | None:
    if arr_idx + 1 >= len(trip):
        return None
    arr = trip[arr_idx]
    nxt = trip[arr_idx + 1]
    if arr[0] == nxt[0] and arr[1] == "an" and nxt[1] == "ab":
        return max(0, nxt[2] - arr[2])
    return None


def _build_shift_feasibility(
    *,
    years: list[str],
    segments: list[Any],
    ledger_event_ids: set[str],
    max_shift_min: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for year in years:
        for segment in segments:
            seen_trip_sigs: set[str] = set()
            for year_col in YEAR_COLUMNS[year]:
                active_rows = active_rows_for_segment(segment, year_col)
                if not active_rows or not valid_ab_an_sequence(active_rows):
                    continue
                for trip in instantiate_trip_events(active_rows):
                    sig = "|".join(f"{s}::{k}::{m}" for s, k, m in trip)
                    if sig in seen_trip_sigs:
                        continue
                    seen_trip_sigs.add(sig)
                    trip_hash = _trip_hash(trip)
                    for idx, (station, kind, minute) in enumerate(trip):
                        if kind == "ab":
                            if idx + 1 >= len(trip) or trip[idx + 1][1] != "an":
                                continue
                            dep_idx = idx
                            arr_idx = idx + 1
                        else:
                            if idx <= 0 or trip[idx - 1][1] != "ab":
                                continue
                            dep_idx = idx - 1
                            arr_idx = idx

                        dep_station, _, dep_min = trip[dep_idx]
                        arr_station, _, arr_min = trip[arr_idx]
                        if kind == "ab":
                            prev_station = _prev_station(trip, idx)
                            next_station = _next_station(trip, idx)
                        else:
                            prev_station = _prev_station(trip, idx)
                            next_station = _next_station(trip, idx)
                        event_row = {
                            "year": year,
                            "transfer_station": station,
                            "board_type": "arrival" if kind == "an" else "departure",
                            "minute": minute,
                            "line": segment.service_name,
                            "line_previous_station": prev_station,
                            "line_next_station": next_station,
                        }
                        event_id = _event_id(event_row)
                        if event_id not in ledger_event_ids:
                            continue

                        dwell_before = _same_station_dwell_before(trip, dep_idx)
                        dwell_after = _same_station_dwell_after(trip, arr_idx)
                        can_earlier = max_shift_min if dwell_before is None else min(max_shift_min, dwell_before)
                        can_later = max_shift_min if dwell_after is None else min(max_shift_min, dwell_after)
                        key = (
                            year,
                            event_id,
                            segment.service_name,
                            dep_station,
                            dep_min,
                            arr_station,
                            arr_min,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        rows.append(
                            {
                                "year": year,
                                "event_id": event_id,
                                "transfer_station": station,
                                "board_type": "arrival" if kind == "an" else "departure",
                                "event_time": _fmt_time(minute),
                                "event_minute": minute,
                                "line": segment.service_name,
                                "line_previous_station": prev_station,
                                "line_next_station": next_station,
                                "section_from_station": dep_station,
                                "section_departure_time": _fmt_time(dep_min),
                                "section_departure_minute": dep_min,
                                "section_to_station": arr_station,
                                "section_arrival_time": _fmt_time(arr_min),
                                "section_arrival_minute": arr_min,
                                "section_rolling_time_min": arr_min - dep_min,
                                "segment_index": segment.segment_index,
                                "raw_year_column": year_col,
                                "trip_signature_hash": trip_hash,
                                "same_train_dwell_before_section_departure_min": "" if dwell_before is None else dwell_before,
                                "same_train_dwell_after_section_arrival_min": "" if dwell_after is None else dwell_after,
                                "can_shift_earlier_min": can_earlier,
                                "can_shift_later_min": can_later,
                                "feasibility_note": (
                                    ("section_start_unbounded_by_same_train;" if dwell_before is None else "")
                                    + ("section_end_unbounded_by_same_train" if dwell_after is None else "")
                                ).strip(";"),
                            }
                        )
    rows.sort(key=lambda r: (r["year"], r["transfer_station"], int(r["event_minute"]), r["line"], r["board_type"]))
    return rows


def _project_transfer_occurrences(row: dict[str, Any]) -> list[tuple[int, int]]:
    arr_times = _parse_time_list(str(row.get("Arrival Time", "")))
    dep_times = _parse_time_list(str(row.get("Departure Time", "")))
    pairs: set[tuple[int, int]] = set()
    for arr_raw, dep_raw in zip(arr_times, dep_times):
        wait = dep_raw - arr_raw
        while wait < 0:
            wait += 120
        for arr in _expand_repeating_times([arr_raw], DAY_START, DAY_END_DEPART):
            dep = arr + wait
            if DAY_START <= arr < DAY_END_DEPART and DAY_START <= dep < DAY_END_DEPART:
                pairs.add((arr, dep))
    if not pairs and arr_times and dep_times:
        arrs = _expand_repeating_times(arr_times, DAY_START, DAY_END_DEPART)
        deps = _expand_repeating_times(dep_times, DAY_START, DAY_END_DEPART)
        for arr, dep in zip(arrs, deps):
            pairs.add((arr, dep))
    return sorted(pairs)


def _feasibility_lookup(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, int], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[(row["year"], row["transfer_station"], row["line"], int(row["event_minute"]))].append(row)
    return out


def _parse_clock_min(value: str) -> int:
    hh, mm = str(value).split(":", 1)
    return int(hh) * 60 + int(mm)


def _best_shift_rows(feas_rows: list[dict[str, Any]], board_type: str) -> list[dict[str, Any]]:
    return [r for r in feas_rows if r["board_type"] == board_type]


def _add_candidate(
    out: list[dict[str, Any]],
    *,
    year: str,
    candidate_type: str,
    action: str,
    station: str,
    line: str,
    event_minute: int,
    delta_min: int,
    proxy_daily_pax: float,
    proxy_pax_minutes: float,
    reason: str,
    feasibility_row: dict[str, Any],
    extra: dict[str, Any],
) -> None:
    out.append(
        {
            "year": year,
            "candidate_type": candidate_type,
            "action": action,
            "transfer_station": station,
            "line": line,
            "event_time": _fmt_time(event_minute),
            "event_minute": event_minute,
            "delta_min": delta_min,
            "estimated_daily_pax_proxy": round(proxy_daily_pax, 6),
            "estimated_pax_minutes_proxy": round(proxy_pax_minutes, 6),
            "reason": reason,
            "section_from_station": feasibility_row.get("section_from_station", ""),
            "section_departure_time": feasibility_row.get("section_departure_time", ""),
            "section_to_station": feasibility_row.get("section_to_station", ""),
            "section_arrival_time": feasibility_row.get("section_arrival_time", ""),
            "can_shift_earlier_min": feasibility_row.get("can_shift_earlier_min", ""),
            "can_shift_later_min": feasibility_row.get("can_shift_later_min", ""),
            "feasibility_note": feasibility_row.get("feasibility_note", ""),
            **extra,
        }
    )


def _build_candidates(
    *,
    years: list[str],
    transfer_paths: dict[str, Path],
    near_misses: list[dict[str, Any]],
    shift_feasibility: list[dict[str, Any]],
    min_transfer_lookup: dict[str, int],
    threshold_pax: float,
    max_shift_min: int,
) -> list[dict[str, Any]]:
    feas = _feasibility_lookup(shift_feasibility)
    out: list[dict[str, Any]] = []

    for year in years:
        with transfer_paths[year].open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pax = _as_float(row.get("Number of Affected Daily Pax"), 0.0)
                if pax < threshold_pax:
                    continue
                station = str(row.get("Transfer station", "")).strip()
                arr_line = str(row.get("Arrival Line", "")).strip()
                dep_line = str(row.get("Departure Line", "")).strip()
                wait_min = _parse_wait_min(str(row.get("Transfer Time", "")))
                min_transfer = _min_transfer(station, min_transfer_lookup)
                reducible = max(0, wait_min - min_transfer)
                if reducible <= 0:
                    continue
                occurrences = _project_transfer_occurrences(row)
                if not occurrences:
                    continue
                pax_share = pax / len(occurrences)
                for arr_min, dep_min in occurrences:
                    # Existing transfer can be improved by moving arrival later.
                    for fr in _best_shift_rows(feas.get((year, station, arr_line, arr_min), []), "arrival"):
                        delta = min(max_shift_min, reducible, _as_int(fr.get("can_shift_later_min"), 0))
                        if delta > 0:
                            _add_candidate(
                                out,
                                year=year,
                                candidate_type="improve_existing_transfer",
                                action="shift_arrival_line_later",
                                station=station,
                                line=arr_line,
                                event_minute=arr_min,
                                delta_min=delta,
                                proxy_daily_pax=pax_share,
                                proxy_pax_minutes=pax_share * delta,
                                reason=f"Reduce current {wait_min}min transfer toward station minimum {min_transfer}min.",
                                feasibility_row=fr,
                                extra={
                                    "arrival_line": arr_line,
                                    "departure_line": dep_line,
                                    "arrival_time": _fmt_time(arr_min),
                                    "departure_time": _fmt_time(dep_min),
                                    "current_wait_minutes": wait_min,
                                    "minimum_transfer_time": min_transfer,
                                    "source_transfer_daily_pax": pax,
                                },
                            )
                    # Existing transfer can be improved by moving departure earlier.
                    for fr in _best_shift_rows(feas.get((year, station, dep_line, dep_min), []), "departure"):
                        delta = min(max_shift_min, reducible, _as_int(fr.get("can_shift_earlier_min"), 0))
                        if delta > 0:
                            _add_candidate(
                                out,
                                year=year,
                                candidate_type="improve_existing_transfer",
                                action="shift_departure_line_earlier",
                                station=station,
                                line=dep_line,
                                event_minute=dep_min,
                                delta_min=-delta,
                                proxy_daily_pax=pax_share,
                                proxy_pax_minutes=pax_share * delta,
                                reason=f"Reduce current {wait_min}min transfer toward station minimum {min_transfer}min.",
                                feasibility_row=fr,
                                extra={
                                    "arrival_line": arr_line,
                                    "departure_line": dep_line,
                                    "arrival_time": _fmt_time(arr_min),
                                    "departure_time": _fmt_time(dep_min),
                                    "current_wait_minutes": wait_min,
                                    "minimum_transfer_time": min_transfer,
                                    "source_transfer_daily_pax": pax,
                                },
                            )

    for nm in near_misses:
        if nm["status"] != "currently_infeasible":
            continue
        shortfall = -int(nm["transfer_margin_minutes"])
        if shortfall <= 0 or shortfall > max_shift_min:
            continue
        proxy = _as_float(nm.get("estimated_potential_daily_pax_time_share"), 0.0)
        if proxy <= 0:
            continue
        year = nm["year"]
        station = nm["transfer_station"]
        arr_line = nm["arrival_line"]
        dep_line = nm["departure_line"]
        arr_min = int(nm["arrival_minute"])
        dep_min = int(nm["departure_minute"])
        for fr in _best_shift_rows(feas.get((year, station, arr_line, arr_min), []), "arrival"):
            if _as_int(fr.get("can_shift_earlier_min"), 0) >= shortfall:
                _add_candidate(
                    out,
                    year=year,
                    candidate_type="unlock_near_miss",
                    action="shift_arrival_line_earlier",
                    station=station,
                    line=arr_line,
                    event_minute=arr_min,
                    delta_min=-shortfall,
                    proxy_daily_pax=proxy,
                    proxy_pax_minutes=proxy * shortfall,
                    reason="Make currently infeasible near-miss transfer meet station minimum transfer time.",
                    feasibility_row=fr,
                    extra={
                        "arrival_line": arr_line,
                        "departure_line": dep_line,
                        "arrival_time": nm["arrival_time"],
                        "departure_time": nm["departure_time"],
                        "current_wait_minutes": nm["wait_minutes"],
                        "minimum_transfer_time": nm["minimum_transfer_time"],
                        "source_transfer_daily_pax": "",
                    },
                )
        for fr in _best_shift_rows(feas.get((year, station, dep_line, dep_min), []), "departure"):
            if _as_int(fr.get("can_shift_later_min"), 0) >= shortfall:
                _add_candidate(
                    out,
                    year=year,
                    candidate_type="unlock_near_miss",
                    action="shift_departure_line_later",
                    station=station,
                    line=dep_line,
                    event_minute=dep_min,
                    delta_min=shortfall,
                    proxy_daily_pax=proxy,
                    proxy_pax_minutes=proxy * shortfall,
                    reason="Make currently infeasible near-miss transfer meet station minimum transfer time.",
                    feasibility_row=fr,
                    extra={
                        "arrival_line": arr_line,
                        "departure_line": dep_line,
                        "arrival_time": nm["arrival_time"],
                        "departure_time": nm["departure_time"],
                        "current_wait_minutes": nm["wait_minutes"],
                        "minimum_transfer_time": nm["minimum_transfer_time"],
                        "source_transfer_daily_pax": "",
                    },
                )

    aggregated: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in out:
        section_departure_minute = _parse_clock_min(str(row.get("section_departure_time", "00:00")))
        section_arrival_minute = _parse_clock_min(str(row.get("section_arrival_time", "00:00")))
        key = (
            row["year"],
            row["action"],
            row["line"],
            int(row["delta_min"]),
            row["section_from_station"],
            section_departure_minute % 120,
            row["section_to_station"],
            section_arrival_minute % 120,
        )
        item = aggregated.get(key)
        transfer_pair = f"{row.get('arrival_line', '')}->{row.get('departure_line', '')}"
        event_pair = f"{row.get('arrival_time', '')}->{row.get('departure_time', '')}"
        if item is None:
            item = {
                "year": row["year"],
                "action": row["action"],
                "line": row["line"],
                "delta_min": row["delta_min"],
                "section_from_station": row["section_from_station"],
                "section_departure_time_mod120": f"{(section_departure_minute % 120) // 60:02d}:{(section_departure_minute % 120) % 60:02d}",
                "section_to_station": row["section_to_station"],
                "section_arrival_time_mod120": f"{(section_arrival_minute % 120) // 60:02d}:{(section_arrival_minute % 120) % 60:02d}",
                "estimated_daily_pax_proxy_sum": 0.0,
                "estimated_pax_minutes_proxy_sum": 0.0,
                "impact_rows": 0,
                "candidate_types": set(),
                "transfer_stations": set(),
                "affected_transfer_pairs": set(),
                "example_event_pairs": set(),
                "reason_examples": set(),
                "feasibility_notes": set(),
                "min_can_shift_earlier_min": None,
                "min_can_shift_later_min": None,
            }
            aggregated[key] = item
        item["estimated_daily_pax_proxy_sum"] += _as_float(row.get("estimated_daily_pax_proxy"), 0.0)
        item["estimated_pax_minutes_proxy_sum"] += _as_float(row.get("estimated_pax_minutes_proxy"), 0.0)
        item["impact_rows"] += 1
        item["candidate_types"].add(str(row.get("candidate_type", "")))
        item["transfer_stations"].add(str(row.get("transfer_station", "")))
        item["affected_transfer_pairs"].add(transfer_pair)
        item["example_event_pairs"].add(event_pair)
        item["reason_examples"].add(str(row.get("reason", "")))
        if row.get("feasibility_note"):
            item["feasibility_notes"].add(str(row.get("feasibility_note", "")))
        earlier = _as_int(row.get("can_shift_earlier_min"), 0)
        later = _as_int(row.get("can_shift_later_min"), 0)
        item["min_can_shift_earlier_min"] = earlier if item["min_can_shift_earlier_min"] is None else min(item["min_can_shift_earlier_min"], earlier)
        item["min_can_shift_later_min"] = later if item["min_can_shift_later_min"] is None else min(item["min_can_shift_later_min"], later)

    aggregated_rows: list[dict[str, Any]] = []
    for item in aggregated.values():
        aggregated_rows.append(
            {
                "year": item["year"],
                "action": item["action"],
                "line": item["line"],
                "delta_min": item["delta_min"],
                "section_from_station": item["section_from_station"],
                "section_departure_time_mod120": item["section_departure_time_mod120"],
                "section_to_station": item["section_to_station"],
                "section_arrival_time_mod120": item["section_arrival_time_mod120"],
                "estimated_daily_pax_proxy_sum": round(float(item["estimated_daily_pax_proxy_sum"]), 6),
                "estimated_pax_minutes_proxy_sum": round(float(item["estimated_pax_minutes_proxy_sum"]), 6),
                "impact_rows": int(item["impact_rows"]),
                "candidate_types": "; ".join(sorted(x for x in item["candidate_types"] if x)),
                "transfer_stations": "; ".join(sorted(x for x in item["transfer_stations"] if x)[:20]),
                "affected_transfer_pairs": "; ".join(sorted(x for x in item["affected_transfer_pairs"] if x)[:20]),
                "example_event_pairs": "; ".join(sorted(x for x in item["example_event_pairs"] if x)[:20]),
                "reason_examples": "; ".join(sorted(x for x in item["reason_examples"] if x)[:4]),
                "min_can_shift_earlier_min": item["min_can_shift_earlier_min"],
                "min_can_shift_later_min": item["min_can_shift_later_min"],
                "feasibility_notes": "; ".join(sorted(x for x in item["feasibility_notes"] if x)[:10]),
            }
        )
    aggregated_rows.sort(
        key=lambda r: (
            -float(r["estimated_pax_minutes_proxy_sum"]),
            r["year"],
            r["line"],
            r["section_from_station"],
            r["section_to_station"],
        )
    )
    return aggregated_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Set up compact transfer-time optimization tables.")
    parser.add_argument("--timetable-csv", type=Path, default=Path("FullTimetableHistory2026_2035.csv"))
    parser.add_argument("--stations-csv", type=Path, default=Path("FilteredStations2026_2035.csv"))
    parser.add_argument("--transfer-2026", type=Path, default=Path("Identified_All_Transfers_2026_transfer_discounted.csv"))
    parser.add_argument("--transfer-2035", type=Path, default=Path("Identified_All_Transfers_2035_transfer_discounted.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("transfer_optimization_tables"))
    parser.add_argument("--threshold-pax", type=float, default=95.0)
    parser.add_argument("--near-miss-before-min", type=int, default=5)
    parser.add_argument("--near-miss-after-min", type=int, default=10)
    parser.add_argument("--max-shift-min", type=int, default=5)
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    args = parser.parse_args()

    years = list(args.years)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    min_transfer_lookup = _load_min_transfer(args.stations_csv.resolve())
    segments = load_timetable_segments(args.timetable_csv.resolve())
    transfer_paths = {
        "2026": args.transfer_2026.resolve(),
        "2035": args.transfer_2035.resolve(),
    }

    high_data: dict[str, dict[str, Any]] = {}
    for year in years:
        stations, high_keys, key_pax, key_meta, event_weights = load_high_flow_transfers(
            transfer_paths[year],
            args.threshold_pax,
            window_start=DAY_START,
            window_end=DAY_END_DEPART,
        )
        high_data[year] = {
            "stations": stations,
            "high_keys": high_keys,
            "key_pax": key_pax,
            "key_meta": key_meta,
            "event_weights": event_weights,
        }
        _log(f"{year}: high-flow transfer patterns={len(high_keys)} stations={len(stations)}")

    event_ledger = _build_event_ledger(
        years=years,
        segments=segments,
        high_data=high_data,
        min_transfer_lookup=min_transfer_lookup,
    )
    ledger_fields = [
        "year",
        "event_id",
        "transfer_station",
        "minimum_transfer_time",
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
    event_ledger_path = output_dir / "Transfer_Optimization_Event_Ledger_2026_2035.csv"
    _write_csv(event_ledger_path, event_ledger, ledger_fields)
    _log(f"Wrote {event_ledger_path} rows={len(event_ledger)}")

    near_misses = _build_near_misses(
        event_ledger,
        early_window_min=max(0, args.near_miss_before_min),
        late_window_min=max(0, args.near_miss_after_min),
    )
    near_fields = [
        "year",
        "transfer_station",
        "minimum_transfer_time",
        "arrival_event_id",
        "arrival_line",
        "arrival_prev_station",
        "arrival_next_station",
        "arrival_time",
        "arrival_minute",
        "departure_event_id",
        "departure_line",
        "departure_prev_station",
        "departure_next_station",
        "departure_time",
        "departure_minute",
        "wait_minutes",
        "transfer_margin_minutes",
        "status",
        "estimated_potential_daily_pax_time_share",
        "arrival_event_daily_pax_time_share",
        "departure_event_daily_pax_time_share",
        "arrival_partner_lines",
        "departure_partner_lines",
    ]
    near_path = output_dir / "Transfer_Optimization_Near_Misses_2026_2035.csv"
    _write_csv(near_path, near_misses, near_fields)
    _log(f"Wrote {near_path} rows={len(near_misses)}")

    shift_feasibility = _build_shift_feasibility(
        years=years,
        segments=segments,
        ledger_event_ids={row["event_id"] for row in event_ledger},
        max_shift_min=max(1, args.max_shift_min),
    )
    feas_fields = [
        "year",
        "event_id",
        "transfer_station",
        "board_type",
        "event_time",
        "event_minute",
        "line",
        "line_previous_station",
        "line_next_station",
        "section_from_station",
        "section_departure_time",
        "section_departure_minute",
        "section_to_station",
        "section_arrival_time",
        "section_arrival_minute",
        "section_rolling_time_min",
        "segment_index",
        "raw_year_column",
        "trip_signature_hash",
        "same_train_dwell_before_section_departure_min",
        "same_train_dwell_after_section_arrival_min",
        "can_shift_earlier_min",
        "can_shift_later_min",
        "feasibility_note",
    ]
    feas_path = output_dir / "Transfer_Optimization_Shift_Feasibility_2026_2035.csv"
    _write_csv(feas_path, shift_feasibility, feas_fields)
    _log(f"Wrote {feas_path} rows={len(shift_feasibility)}")

    candidates = _build_candidates(
        years=years,
        transfer_paths=transfer_paths,
        near_misses=near_misses,
        shift_feasibility=shift_feasibility,
        min_transfer_lookup=min_transfer_lookup,
        threshold_pax=args.threshold_pax,
        max_shift_min=max(1, args.max_shift_min),
    )
    cand_fields = [
        "year",
        "action",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "estimated_daily_pax_proxy_sum",
        "estimated_pax_minutes_proxy_sum",
        "impact_rows",
        "candidate_types",
        "transfer_stations",
        "affected_transfer_pairs",
        "example_event_pairs",
        "reason_examples",
        "min_can_shift_earlier_min",
        "min_can_shift_later_min",
        "feasibility_notes",
    ]
    cand_path = output_dir / "Transfer_Optimization_Candidates_2026_2035.csv"
    _write_csv(cand_path, candidates, cand_fields)
    _log(f"Wrote {cand_path} rows={len(candidates)}")

    summary = {
        "created_at": _now(),
        "threshold_pax": args.threshold_pax,
        "near_miss_before_min": args.near_miss_before_min,
        "near_miss_after_min": args.near_miss_after_min,
        "max_shift_min": args.max_shift_min,
        "years": years,
        "outputs": {
            "event_ledger": str(event_ledger_path),
            "near_misses": str(near_path),
            "shift_feasibility": str(feas_path),
            "candidates": str(cand_path),
        },
        "row_counts": {
            "event_ledger": len(event_ledger),
            "near_misses": len(near_misses),
            "shift_feasibility": len(shift_feasibility),
            "candidates": len(candidates),
        },
        "notes": [
            "Candidate scores are proxy pax-minutes for screening only.",
            "Final decisions should be validated by targeted rerouting on affected OD paths.",
            "Rolling time is preserved by representing shifts as whole-section offsets.",
        ],
    }
    summary_path = output_dir / "transfer_optimization_setup_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
