#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_od_first_column_metrics_2026_2035 import _decompose_legs_with_rebalance
from build_optimised_v2_timetable import (
    DEFAULT_INPUT_TIMETABLE,
    _discover_year_column_indices,
    _load_rows,
    _parse_segments,
    _write_rows,
)
from generate_proposed_timetable_changes import (
    BASE_DIR,
    DEFAULT_STATIONS,
    DEFAULT_TRANSFER_2026,
    DEFAULT_TRANSFER_2035,
    _affected_keys,
    _fmt,
    _load_existing_transfer_occurrences,
    _load_min_transfer_from_ledger,
    _load_near_miss_index,
    _read_csv,
    _safe_float,
    _safe_int,
    _shift_mod120,
    _try_apply_candidate,
    _write_dict_csv,
)


DEFAULT_SOURCE_DIR = BASE_DIR / "transfer_optimization_tables"
DEFAULT_PHYSICAL = DEFAULT_SOURCE_DIR / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv"
DEFAULT_SEED = (
    DEFAULT_SOURCE_DIR
    / "missed_transfer_station_board_v0"
    / "ProposedTimetableChanges_MissedTransfers_StationBoardStrict_FILTERED.csv"
)
DEFAULT_WORK_DIR = DEFAULT_SOURCE_DIR / "better_optimization_v0"
DEFAULT_OPT_STATE = Path.home() / "od_fast_state" / "nondominated_cells_state_OPTIMIZED_FILTERED"
DEFAULT_V0_STATE = Path.home() / "od_fast_state" / "nondominated_cells_state_v0_restored_from_pathalloc_csv"

ZURICH_HB_CANONICAL = "Zürich HB"
ZURICH_HB_3134 = "Zürich HB (31-34)"
ZURICH_HB_FAMILY = {
    "Zürich HB",
    "Zürich HB (21-22)",
    "Zürich HB (31-34)",
    "Zürich HB (41-44)",
}

DEFAULT_MANUAL_HUBS = (
    "Zürich HB",
    "Bern",
    "Basel SBB",
    "Lausanne",
    "Genève",
    "Luzern",
    "Olten",
    "Zürich Oerlikon",
    "Winterthur",
    "St. Gallen",
    "Biel/Bienne",
    "Thun",
    "Spiez",
    "Visp",
    "Arth-Goldau",
    "Zug",
    "Aarau",
    "Rotkreuz",
    "Ziegelbrücke",
    "Sargans",
    "Chur",
    "Bellinzona",
    "Lugano",
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.is_file():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _split_semicolon(value: Any) -> set[str]:
    out: set[str] = set()
    for part in str(value or "").split(";"):
        part = part.strip()
        if part:
            out.add(part)
    return out


def _physical_id(row: dict[str, Any]) -> str:
    existing = str(row.get("physical_shift_id") or "").strip()
    if existing:
        return existing
    return "|".join(
        [
            str(row.get("year", "")),
            "physical",
            str(row.get("line", "")),
            str(row.get("delta_min", "")),
            str(row.get("section_from_station", "")),
            str(row.get("section_departure_time_mod120", "")),
            str(row.get("section_to_station", "")),
            str(row.get("section_arrival_time_mod120", "")),
        ]
    )


def _candidate_section_key(row: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        str(row.get("year", "")),
        str(row.get("line", "")),
        str(row.get("section_from_station", "")),
        str(row.get("section_to_station", "")),
        _mod120(row.get("section_departure_time_mod120", "00:00")),
        _mod120(row.get("section_arrival_time_mod120", "00:00")),
    )


def _mod120(value: Any) -> int:
    text = str(value or "").strip()
    hh, mm = text.split(":", 1)
    return (int(hh) * 60 + int(mm)) % 120


def _time_to_text(minute: int) -> str:
    minute %= 24 * 60
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _is_hb_family(station: Any) -> bool:
    return str(station) in ZURICH_HB_FAMILY


def _minimum_transfer(
    *,
    station_min_transfer: dict[str, int],
    arrival_station: str,
    departure_station: str,
) -> int | None:
    if _is_hb_family(arrival_station) and _is_hb_family(departure_station):
        if arrival_station == ZURICH_HB_3134 and departure_station == ZURICH_HB_3134:
            return 4
        return 7
    if arrival_station != departure_station:
        return None
    return int(station_min_transfer.get(arrival_station, 0))


def _load_station_min_transfer(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            station = str(row.get("StationName", "")).strip()
            if not station:
                continue
            out[station] = max(0, _safe_int(row.get("MinimumTransferTime"), 0))
    out[ZURICH_HB_CANONICAL] = 7
    out[ZURICH_HB_3134] = 4
    return out


def _load_transfer_station_totals(paths: dict[str, Path]) -> dict[tuple[str, str], dict[str, float]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "daily_pax": 0.0, "od_pairs": 0.0})
    for year, path in paths.items():
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                station = str(row.get("Transfer station", "")).strip()
                if not station:
                    continue
                item = totals[(year, station)]
                item["rows"] += 1
                item["daily_pax"] += _safe_float(row.get("Number of Affected Daily Pax"), 0.0)
                item["od_pairs"] += _safe_float(row.get("Number of Affected OD pairs"), 0.0)
    return totals


def _load_near_miss_station_totals(path: Path, years: set[str]) -> dict[tuple[str, str], dict[str, float]]:
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: {"rows": 0.0, "potential_pax": 0.0})
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            year = str(row.get("year", "")).strip()
            if year not in years or str(row.get("status", "")).strip() != "currently_infeasible":
                continue
            station = str(row.get("transfer_station", "")).strip()
            if not station:
                continue
            item = totals[(year, station)]
            item["rows"] += 1
            item["potential_pax"] += _safe_float(row.get("estimated_potential_daily_pax_time_share"), 0.0)
    return totals


def _select_hubs(
    *,
    years: list[str],
    transfer_paths: dict[str, Path],
    near_miss_path: Path,
    manual_hubs: set[str],
    top_hubs_per_year: int,
    min_transfer_pax: float,
    min_near_miss_pax: float,
) -> tuple[set[tuple[str, str]], list[dict[str, Any]]]:
    years_set = set(years)
    transfer_totals = _load_transfer_station_totals(transfer_paths)
    near_totals = _load_near_miss_station_totals(near_miss_path, years_set)
    selected: set[tuple[str, str]] = set()
    rows: list[dict[str, Any]] = []
    for year in years:
        year_rows: list[dict[str, Any]] = []
        stations = {station for y, station in transfer_totals if y == year} | {station for y, station in near_totals if y == year}
        for station in stations:
            t = transfer_totals.get((year, station), {})
            n = near_totals.get((year, station), {})
            transfer_pax = float(t.get("daily_pax", 0.0))
            near_pax = float(n.get("potential_pax", 0.0))
            score = transfer_pax + 10.0 * near_pax
            reasons: list[str] = []
            if transfer_pax >= min_transfer_pax:
                reasons.append("high_existing_transfer_pax")
            if near_pax >= min_near_miss_pax:
                reasons.append("high_near_miss_potential")
            if station in manual_hubs:
                reasons.append("manual_main_hub")
            year_rows.append(
                {
                    "year": year,
                    "transfer_station": station,
                    "transfer_rows": int(t.get("rows", 0.0)),
                    "affected_daily_pax_sum": round(transfer_pax, 6),
                    "near_miss_rows": int(n.get("rows", 0.0)),
                    "near_miss_potential_daily_pax_sum": round(near_pax, 6),
                    "hub_selection_score": round(score, 6),
                    "selection_reason": "; ".join(reasons),
                }
            )
        year_rows.sort(key=lambda r: (-float(r["hub_selection_score"]), str(r["transfer_station"])))
        for idx, row in enumerate(year_rows, start=1):
            reason = str(row["selection_reason"])
            if idx <= top_hubs_per_year:
                reason = (reason + "; " if reason else "") + "top_ranked_hub"
            row["rank_by_score"] = idx
            row["selection_reason"] = reason
            if reason:
                selected.add((year, str(row["transfer_station"])))
                rows.append(row)
    return selected, rows


def _candidate_stations(row: dict[str, Any]) -> set[str]:
    out = set()
    out.add(str(row.get("section_from_station", "")).strip())
    out.add(str(row.get("section_to_station", "")).strip())
    out.update(_split_semicolon(row.get("transfer_stations")))
    out.update(_split_semicolon(row.get("net_affected_transfer_stations")))
    return {item for item in out if item}


def _station_matches_selected_hub(year: str, station: str, selected_hubs: set[tuple[str, str]]) -> bool:
    if (year, station) in selected_hubs:
        return True
    if station in ZURICH_HB_FAMILY and (year, ZURICH_HB_CANONICAL) in selected_hubs:
        return True
    return False


def _candidate_passes_static_filters(
    row: dict[str, Any],
    *,
    years: set[str],
    selected_hubs: set[tuple[str, str]],
    max_shift_min: int,
    min_static_net: float,
    min_static_gross_gain: float,
    max_gross_loss_ratio: float,
    allow_nonhub: bool,
) -> tuple[bool, str]:
    year = str(row.get("year", "")).strip()
    if year not in years:
        return False, "wrong_year"
    delta = _safe_int(row.get("delta_min"), 0)
    if delta == 0 or abs(delta) > max_shift_min:
        return False, "delta_outside_bounds"
    if delta < 0 and _safe_float(row.get("min_can_shift_earlier_min"), 0.0) < abs(delta):
        return False, "insufficient_local_dwell_absorption_earlier"
    if delta > 0 and _safe_float(row.get("min_can_shift_later_min"), 0.0) < delta:
        return False, "insufficient_local_dwell_absorption_later"
    if _safe_float(row.get("broken_existing_daily_pax_proxy"), 0.0) > 0:
        return False, "breaks_existing_transfer_proxy"
    if _safe_int(row.get("broken_existing_transfer_events"), 0) > 0:
        return False, "breaks_existing_event_proxy"
    if _safe_int(row.get("same_line_pair_count"), 0) > 0:
        return False, "same_line_transfer_pair"
    if str(row.get("feasibility_notes", "")).strip():
        return False, "unbounded_or_boundary_feasibility"
    net = _safe_float(row.get("net_pax_minutes_proxy"), 0.0)
    gross_gain = _safe_float(row.get("gross_gain_pax_minutes_proxy"), 0.0)
    gross_loss = _safe_float(row.get("gross_loss_pax_minutes_proxy"), 0.0)
    if net < min_static_net:
        return False, "low_static_net_proxy"
    if gross_gain < min_static_gross_gain:
        return False, "low_static_gross_gain_proxy"
    if gross_gain > 0 and gross_loss / gross_gain > max_gross_loss_ratio:
        return False, "high_gross_loss_ratio"
    if not allow_nonhub and not any(
        _station_matches_selected_hub(year, station, selected_hubs) for station in _candidate_stations(row)
    ):
        return False, "not_in_selected_hub_subgraph"
    return True, ""


def _path_weight(path: dict[str, Any], total_demand: float) -> float:
    weight = _safe_float(path.get("daily_demand_allocated"), 0.0)
    if weight > 0:
        return weight
    share = max(_safe_float(path.get("daily_demand_share"), 0.0), 0.0)
    return max(total_demand, 0.0) * share


def _leg_section_key(year: str, leg: dict[str, Any]) -> tuple[str, str, str, str, int, int]:
    return (
        year,
        str(leg.get("service_name", "")),
        str(leg.get("from_station", "")),
        str(leg.get("to_station", "")),
        int(leg.get("dep_min", 0)) % 120,
        int(leg.get("arr_min", 0)) % 120,
    )


def _copy_shifted_legs(legs: list[dict[str, Any]], shift_map: dict[tuple[str, str, str, str, int, int], int], year: str) -> tuple[list[dict[str, Any]], bool]:
    shifted: list[dict[str, Any]] = []
    touched = False
    for leg in legs:
        item = dict(leg)
        delta = int(shift_map.get(_leg_section_key(year, item), 0))
        if delta:
            item["dep_min"] = int(item["dep_min"]) + delta
            item["arr_min"] = int(item["arr_min"]) + delta
            item["dep_time"] = _time_to_text(int(item["dep_min"]))
            item["arr_time"] = _time_to_text(int(item["arr_min"]))
            touched = True
        shifted.append(item)
    return shifted, touched


def _path_components_from_legs(
    *,
    legs: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
) -> tuple[bool, dict[str, float]]:
    if not legs:
        return False, {}
    for leg in legs:
        if int(leg["arr_min"]) < int(leg["dep_min"]):
            return False, {}
    for left, right in zip(legs, legs[1:]):
        if str(left.get("canonical_trip_id", "")) == str(right.get("canonical_trip_id", "")):
            if int(right["dep_min"]) < int(left["arr_min"]):
                return False, {}
            continue
        min_transfer = _minimum_transfer(
            station_min_transfer=station_min_transfer,
            arrival_station=str(left.get("to_station", "")),
            departure_station=str(right.get("from_station", "")),
        )
        if min_transfer is None:
            return False, {}
        if int(right["dep_min"]) < int(left["arr_min"]) + min_transfer:
            return False, {}

    total = float(int(legs[-1]["arr_min"]) - int(legs[0]["dep_min"]))
    if total <= 0:
        return False, {}
    rebalance_input = [
        {
            "dep_time": int(leg["dep_min"]),
            "arr_time": int(leg["arr_min"]),
            "canonical_trip_id": str(leg.get("canonical_trip_id", "")),
        }
        for leg in legs
    ]
    rolling, dwell, transfer = _decompose_legs_with_rebalance(rebalance_input, total)
    return True, {
        "total": total,
        "rolling": rolling,
        "dwell": dwell,
        "transfer": transfer,
        "rolling_share": rolling / total if total > 0 else 0.0,
        "dwell_share": dwell / total if total > 0 else 0.0,
        "transfer_share": transfer / total if total > 0 else 0.0,
    }


def _scan_current_paths_for_candidate_scores(
    *,
    years: list[str],
    rows_root: Path,
    candidates: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
    log_path: Path,
    progress_every_rows: int,
    scan_row_limit: int = 0,
) -> dict[str, dict[str, Any]]:
    candidate_by_key: dict[tuple[str, str, str, str, int, int], list[str]] = defaultdict(list)
    candidate_key_by_id: dict[str, tuple[str, str, str, str, int, int]] = {}
    candidate_delta: dict[str, int] = {}
    for row in candidates:
        cid = _physical_id(row)
        section_key = _candidate_section_key(row)
        candidate_by_key[section_key].append(cid)
        candidate_key_by_id[cid] = section_key
        candidate_delta[cid] = _safe_int(row.get("delta_min"), 0)

    scores: dict[str, dict[str, Any]] = {
        _physical_id(row): {
            "exact_existing_path_pax_minutes_saved": 0.0,
            "exact_existing_path_weight_touched": 0.0,
            "exact_existing_path_count_touched": 0,
            "exact_existing_path_invalidated_daily_pax": 0.0,
            "exact_existing_path_invalidated_count": 0,
            "exact_existing_path_rolling_share_delta_weighted": 0.0,
            "example_od": "",
        }
        for row in candidates
    }

    for year in years:
        rows_dir = rows_root / year / "rows"
        row_files = sorted(rows_dir.glob("row_*.json"))
        if not row_files:
            raise FileNotFoundError(f"Missing enriched row JSONs: {rows_dir}")
        if scan_row_limit > 0:
            row_files = row_files[:scan_row_limit]
            _log(log_path, f"{year}: LIMITED smoke scan enabled scan_row_limit={scan_row_limit}")
        _log(log_path, f"{year}: scanning current scenario row JSONs for candidate path-level effects rows={len(row_files)}")
        for idx, row_path in enumerate(row_files, start=1):
            payload = json.loads(row_path.read_text(encoding="utf-8"))
            origin = str(payload.get("origin", ""))
            for dest_idx, raw_cell in enumerate(payload.get("cells", []) or []):
                try:
                    cell = json.loads(raw_cell)
                except Exception:
                    continue
                paths = cell.get("paths")
                if not isinstance(paths, list) or not paths:
                    continue
                total_demand = _safe_float(cell.get("daily_demand_total"), 0.0)
                destination = str(cell.get("destination") or "")
                if not destination:
                    # Cell payloads do not always carry destination; this keeps examples readable enough.
                    destination = f"dest_idx_{dest_idx}"
                for path in paths:
                    if not isinstance(path, dict):
                        continue
                    weight = _path_weight(path, total_demand)
                    if weight <= 0:
                        continue
                    legs = path.get("legs")
                    if not isinstance(legs, list) or not legs:
                        continue
                    touched_ids: set[str] = set()
                    for leg in legs:
                        touched_ids.update(candidate_by_key.get(_leg_section_key(year, leg), []))
                    if not touched_ids:
                        continue
                    old_ok, old_metrics = _path_components_from_legs(
                        legs=[dict(leg) for leg in legs],
                        station_min_transfer=station_min_transfer,
                    )
                    if not old_ok:
                        continue
                    for cid in touched_ids:
                        shift_key = candidate_key_by_id[cid]
                        shifted_legs, touched = _copy_shifted_legs(
                            legs=[dict(leg) for leg in legs],
                            shift_map={shift_key: candidate_delta[cid]},
                            year=year,
                        )
                        if not touched:
                            continue
                        new_ok, new_metrics = _path_components_from_legs(
                            legs=shifted_legs,
                            station_min_transfer=station_min_transfer,
                        )
                        item = scores[cid]
                        item["exact_existing_path_weight_touched"] += weight
                        item["exact_existing_path_count_touched"] += 1
                        if not new_ok:
                            item["exact_existing_path_invalidated_daily_pax"] += weight
                            item["exact_existing_path_invalidated_count"] += 1
                            continue
                        item["exact_existing_path_pax_minutes_saved"] += (old_metrics["total"] - new_metrics["total"]) * weight
                        item["exact_existing_path_rolling_share_delta_weighted"] += (
                            new_metrics["rolling_share"] - old_metrics["rolling_share"]
                        ) * weight
                        if not item["example_od"]:
                            item["example_od"] = f"{origin}->{destination} weight={_fmt(weight, 3)} total {old_metrics['total']:.3f}->{new_metrics['total']:.3f}"
            if idx % max(1, progress_every_rows) == 0 or idx == len(row_files):
                _log(log_path, f"{year}: path-effect scan progress rows={idx}/{len(row_files)}")

    for item in scores.values():
        weight = float(item["exact_existing_path_weight_touched"])
        if weight > 0:
            item["exact_existing_path_mean_total_delta_min"] = -float(item["exact_existing_path_pax_minutes_saved"]) / weight
            item["exact_existing_path_rolling_share_delta"] = float(item["exact_existing_path_rolling_share_delta_weighted"]) / weight
        else:
            item["exact_existing_path_mean_total_delta_min"] = 0.0
            item["exact_existing_path_rolling_share_delta"] = 0.0
    return scores


def _scan_whole_scenario_existing_paths(
    *,
    years: list[str],
    rows_root: Path,
    accepted_rows: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
    log_path: Path,
    progress_every_rows: int,
    scan_row_limit: int = 0,
) -> dict[str, Any]:
    shift_map = {_candidate_section_key(row): _safe_int(row.get("delta_min"), 0) for row in accepted_rows}
    shift_id_by_key = {_candidate_section_key(row): _physical_id(row) for row in accepted_rows}
    total_saved = 0.0
    total_weight_touched = 0.0
    touched_paths = 0
    invalidated_pax = 0.0
    invalidated_paths = 0
    invalidated_by_candidate: dict[str, dict[str, float | int]] = defaultdict(lambda: {"daily_pax": 0.0, "paths": 0})
    rolling_share_delta_weighted = 0.0
    for year in years:
        rows_dir = rows_root / year / "rows"
        row_files = sorted(rows_dir.glob("row_*.json"))
        if scan_row_limit > 0:
            row_files = row_files[:scan_row_limit]
            _log(log_path, f"{year}: LIMITED final smoke scan enabled scan_row_limit={scan_row_limit}")
        _log(log_path, f"{year}: final whole-scenario current-route exact scan rows={len(row_files)}")
        for idx, row_path in enumerate(row_files, start=1):
            payload = json.loads(row_path.read_text(encoding="utf-8"))
            for raw_cell in payload.get("cells", []) or []:
                try:
                    cell = json.loads(raw_cell)
                except Exception:
                    continue
                total_demand = _safe_float(cell.get("daily_demand_total"), 0.0)
                for path in cell.get("paths", []) or []:
                    if not isinstance(path, dict):
                        continue
                    weight = _path_weight(path, total_demand)
                    legs = path.get("legs")
                    if weight <= 0 or not isinstance(legs, list) or not legs:
                        continue
                    touched_ids = {
                        shift_id_by_key[_leg_section_key(year, leg)]
                        for leg in legs
                        if _leg_section_key(year, leg) in shift_id_by_key
                    }
                    shifted_legs, touched = _copy_shifted_legs(
                        legs=[dict(leg) for leg in legs],
                        shift_map=shift_map,
                        year=year,
                    )
                    if not touched:
                        continue
                    old_ok, old_metrics = _path_components_from_legs(
                        legs=[dict(leg) for leg in legs],
                        station_min_transfer=station_min_transfer,
                    )
                    if not old_ok:
                        continue
                    new_ok, new_metrics = _path_components_from_legs(
                        legs=shifted_legs,
                        station_min_transfer=station_min_transfer,
                    )
                    total_weight_touched += weight
                    touched_paths += 1
                    if not new_ok:
                        invalidated_pax += weight
                        invalidated_paths += 1
                        for cid in touched_ids:
                            invalidated_by_candidate[cid]["daily_pax"] = float(invalidated_by_candidate[cid]["daily_pax"]) + weight
                            invalidated_by_candidate[cid]["paths"] = int(invalidated_by_candidate[cid]["paths"]) + 1
                        continue
                    total_saved += (old_metrics["total"] - new_metrics["total"]) * weight
                    rolling_share_delta_weighted += (new_metrics["rolling_share"] - old_metrics["rolling_share"]) * weight
            if idx % max(1, progress_every_rows) == 0 or idx == len(row_files):
                _log(log_path, f"{year}: final scan progress rows={idx}/{len(row_files)}")
    return {
        "accepted_change_rows": len(accepted_rows),
        "current_route_exact_pax_minutes_saved": round(total_saved, 6),
        "current_route_exact_weight_touched": round(total_weight_touched, 6),
        "current_route_exact_touched_paths": int(touched_paths),
        "current_route_exact_invalidated_daily_pax": round(invalidated_pax, 6),
        "current_route_exact_invalidated_paths": int(invalidated_paths),
        "invalidated_by_physical_shift_id": {
            cid: {
                "daily_pax": round(float(values["daily_pax"]), 6),
                "paths": int(values["paths"]),
            }
            for cid, values in sorted(
                invalidated_by_candidate.items(),
                key=lambda item: (-float(item[1]["daily_pax"]), str(item[0])),
            )
        },
        "current_route_exact_mean_total_delta_min": round(-total_saved / total_weight_touched, 9) if total_weight_touched > 0 else 0.0,
        "current_route_exact_rolling_share_delta": round(rolling_share_delta_weighted / total_weight_touched, 9) if total_weight_touched > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Better Optimization timetable-change suggestions. Unlike the wider-subgraph "
            "proxy-only search, this script scans the enriched optimized row JSONs and scores each "
            "candidate by exact current-route OD path total-time effects, while using near-miss "
            "proxy gains only as a discounted secondary signal."
        )
    )
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--timetable-csv", type=Path, default=DEFAULT_INPUT_TIMETABLE)
    parser.add_argument("--physical-shifts-csv", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--seed-proposals-csv", type=Path, default=DEFAULT_SEED)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--transfers-2026", type=Path, default=DEFAULT_TRANSFER_2026)
    parser.add_argument("--transfers-2035", type=Path, default=DEFAULT_TRANSFER_2035)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--current-row-state-root", type=Path, default=DEFAULT_OPT_STATE)
    parser.add_argument("--benchmark-row-state-root", type=Path, default=DEFAULT_V0_STATE)
    parser.add_argument(
        "--score-benchmark-effects",
        type=int,
        default=0,
        help="Also scan benchmark row JSONs, usually v0, for diagnostic candidate effects.",
    )
    parser.add_argument("--seed-with-filtered", type=int, default=1)
    parser.add_argument("--top-hubs-per-year", type=int, default=28)
    parser.add_argument("--manual-hub", action="append", default=list(DEFAULT_MANUAL_HUBS))
    parser.add_argument("--min-transfer-pax", type=float, default=40000.0)
    parser.add_argument("--min-near-miss-pax", type=float, default=500.0)
    parser.add_argument("--max-shift-min", type=int, default=10)
    parser.add_argument("--max-candidate-rows", type=int, default=450)
    parser.add_argument("--min-static-net", type=float, default=50.0)
    parser.add_argument("--min-static-gross-gain", type=float, default=200.0)
    parser.add_argument("--max-gross-loss-ratio", type=float, default=0.75)
    parser.add_argument("--near-miss-discount", type=float, default=0.25)
    parser.add_argument("--min-better-score", type=float, default=50.0)
    parser.add_argument("--max-new-rows-per-year", type=int, default=12)
    parser.add_argument("--max-new-rows-total", type=int, default=20)
    parser.add_argument("--max-new-rows-per-line-year", type=int, default=3)
    parser.add_argument("--allow-nonhub", action="store_true")
    parser.add_argument("--progress-every-rows", type=int, default=25)
    parser.add_argument(
        "--scan-row-limit",
        type=int,
        default=0,
        help="Development/smoke-test only: scan only the first N row JSONs per year. Leave at 0 for thesis-grade full scoring.",
    )
    parser.add_argument("--strict-final-no-invalidated", type=int, default=0)
    parser.add_argument("--final-prune-max-iterations", type=int, default=0)
    parser.add_argument("--require-final-positive-vs-current", type=int, default=0)
    args = parser.parse_args()

    years = list(args.years)
    year_set = set(years)
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "BetterOptimization_timetable_search.log"
    log_path.write_text("", encoding="utf-8")
    _log(log_path, f"Better Optimization search start years={years} seed_with_filtered={bool(args.seed_with_filtered)}")

    source_dir = Path(args.source_dir).expanduser().resolve()
    event_ledger_path = source_dir / "Transfer_Optimization_Event_Ledger_2026_2035.csv"
    near_miss_path = source_dir / "Transfer_Optimization_Near_Misses_2026_2035.csv"
    transfer_paths = {
        "2026": Path(args.transfers_2026).expanduser().resolve(),
        "2035": Path(args.transfers_2035).expanduser().resolve(),
    }
    selected_hubs, hub_rows = _select_hubs(
        years=years,
        transfer_paths=transfer_paths,
        near_miss_path=near_miss_path,
        manual_hubs={str(h) for h in args.manual_hub},
        top_hubs_per_year=max(1, int(args.top_hubs_per_year)),
        min_transfer_pax=float(args.min_transfer_pax),
        min_near_miss_pax=float(args.min_near_miss_pax),
    )
    _write_dict_csv(
        work_dir / "BetterOptimization_SelectedHubs.csv",
        hub_rows,
        [
            "year",
            "rank_by_score",
            "transfer_station",
            "transfer_rows",
            "affected_daily_pax_sum",
            "near_miss_rows",
            "near_miss_potential_daily_pax_sum",
            "hub_selection_score",
            "selection_reason",
        ],
    )
    _log(log_path, f"Selected hub-year subgraphs={len(selected_hubs)}")

    all_physical_rows = _read_csv(Path(args.physical_shifts_csv).expanduser().resolve())
    seed_rows: list[dict[str, str]] = []
    seed_ids: set[str] = set()
    if int(args.seed_with_filtered):
        seed_rows = [row for row in _read_csv(Path(args.seed_proposals_csv).expanduser().resolve()) if str(row.get("year", "")) in year_set]
        seed_ids = {_physical_id(row) for row in seed_rows}
    _log(log_path, f"Loaded seed rows={len(seed_rows)}")

    rejected_static: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for row in all_physical_rows:
        pid = _physical_id(row)
        if pid in seed_ids:
            continue
        ok, reason = _candidate_passes_static_filters(
            row,
            years=year_set,
            selected_hubs=selected_hubs,
            max_shift_min=max(1, int(args.max_shift_min)),
            min_static_net=float(args.min_static_net),
            min_static_gross_gain=float(args.min_static_gross_gain),
            max_gross_loss_ratio=float(args.max_gross_loss_ratio),
            allow_nonhub=bool(args.allow_nonhub),
        )
        if not ok:
            rejected_static.append({"physical_shift_id": pid, "year": row.get("year", ""), "status": "rejected_static", "reason": reason})
            continue
        candidate_rows.append(dict(row))

    candidate_rows.sort(
        key=lambda r: (
            -_safe_float(r.get("net_pax_minutes_proxy"), 0.0),
            -_safe_float(r.get("gross_gain_pax_minutes_proxy"), 0.0),
            _safe_float(r.get("gross_loss_pax_minutes_proxy"), 0.0),
            str(r.get("year", "")),
            _physical_id(r),
        )
    )
    candidate_rows = candidate_rows[: max(1, int(args.max_candidate_rows))]
    _log(log_path, f"Static-filtered path-evaluation candidate rows={len(candidate_rows)}")

    station_min_transfer = _load_station_min_transfer(Path(args.stations_csv).expanduser().resolve())
    exact_scores = _scan_current_paths_for_candidate_scores(
        years=years,
        rows_root=Path(args.current_row_state_root).expanduser().resolve(),
        candidates=candidate_rows,
        station_min_transfer=station_min_transfer,
        log_path=log_path,
        progress_every_rows=max(1, int(args.progress_every_rows)),
        scan_row_limit=max(0, int(args.scan_row_limit)),
    )
    benchmark_scores: dict[str, dict[str, Any]] = {}
    if int(args.score_benchmark_effects):
        benchmark_root = Path(args.benchmark_row_state_root).expanduser().resolve()
        _log(log_path, f"Benchmark diagnostic scan enabled root={benchmark_root}")
        benchmark_scores = _scan_current_paths_for_candidate_scores(
            years=years,
            rows_root=benchmark_root,
            candidates=candidate_rows,
            station_min_transfer=station_min_transfer,
            log_path=log_path,
            progress_every_rows=max(1, int(args.progress_every_rows)),
            scan_row_limit=max(0, int(args.scan_row_limit)),
        )

    scored_rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        pid = _physical_id(row)
        score = exact_scores[pid]
        exact_saved = float(score["exact_existing_path_pax_minutes_saved"])
        near_gain = _safe_float(row.get("near_miss_gain_pax_minutes"), 0.0)
        existing_worsened = _safe_float(row.get("existing_worsened_pax_minutes"), 0.0)
        invalidated_pax = float(score["exact_existing_path_invalidated_daily_pax"])
        discounted_near = max(0.0, near_gain) * float(args.near_miss_discount)
        better_score = exact_saved + discounted_near - 0.50 * max(0.0, existing_worsened)
        item = {
            **row,
            "physical_shift_id": pid,
            "exact_existing_path_pax_minutes_saved": round(exact_saved, 6),
            "exact_existing_path_weight_touched": round(float(score["exact_existing_path_weight_touched"]), 6),
            "exact_existing_path_count_touched": int(score["exact_existing_path_count_touched"]),
            "exact_existing_path_invalidated_daily_pax": round(invalidated_pax, 6),
            "exact_existing_path_invalidated_count": int(score["exact_existing_path_invalidated_count"]),
            "exact_existing_path_mean_total_delta_min": round(float(score["exact_existing_path_mean_total_delta_min"]), 9),
            "exact_existing_path_rolling_share_delta": round(float(score["exact_existing_path_rolling_share_delta"]), 9),
            "discounted_near_miss_pax_minutes": round(discounted_near, 6),
            "better_score_pax_minutes": round(better_score, 6),
            "path_effect_example_od": score["example_od"],
        }
        if benchmark_scores:
            benchmark = benchmark_scores.get(pid, {})
            item.update(
                {
                    "benchmark_v0_exact_path_pax_minutes_saved": round(float(benchmark.get("exact_existing_path_pax_minutes_saved", 0.0)), 6),
                    "benchmark_v0_exact_path_weight_touched": round(float(benchmark.get("exact_existing_path_weight_touched", 0.0)), 6),
                    "benchmark_v0_exact_path_invalidated_daily_pax": round(float(benchmark.get("exact_existing_path_invalidated_daily_pax", 0.0)), 6),
                    "benchmark_v0_exact_path_mean_total_delta_min": round(float(benchmark.get("exact_existing_path_mean_total_delta_min", 0.0)), 9),
                    "benchmark_v0_exact_path_rolling_share_delta": round(float(benchmark.get("exact_existing_path_rolling_share_delta", 0.0)), 9),
                }
            )
        scored_rows.append(item)

    scored_rows.sort(
        key=lambda r: (
            -float(r["better_score_pax_minutes"]),
            -float(r["exact_existing_path_pax_minutes_saved"]),
            str(r.get("year", "")),
            str(r.get("physical_shift_id", "")),
        )
    )

    timetable_rows = _load_rows(Path(args.timetable_csv).expanduser().resolve())
    year_col_indices = _discover_year_column_indices(timetable_rows[0])
    segments = _parse_segments(timetable_rows)
    row_to_segment = {row_idx: segment for segment in segments for row_idx in segment.row_indices}
    working_rows = timetable_rows
    seed_written: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    used_event_keys: set[tuple[str, str, str, str, int]] = set()

    for rank, row in enumerate(seed_rows, start=1):
        ok, new_rows, info = _try_apply_candidate(
            raw_row=row,
            rank=rank,
            rows=working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        if not ok:
            raise RuntimeError(f"Seed row failed to apply safely: {_physical_id(row)} {info}")
        working_rows = new_rows
        used_ids.add(_physical_id(row))
        used_event_keys.update(_affected_keys(row))
        seed_written.append({"scenario_role": "seed_filtered", **row, **info})
    _log(log_path, f"Applied seed rows safely={len(seed_written)}")

    accepted: list[dict[str, Any]] = []
    rejected_dynamic: list[dict[str, Any]] = list(rejected_static)
    accepted_by_year: Counter[str] = Counter()
    accepted_by_line_year: Counter[tuple[str, str]] = Counter()
    cumulative_better = 0.0
    cumulative_exact_existing = 0.0
    for row in scored_rows:
        pid = str(row["physical_shift_id"])
        year = str(row.get("year", ""))
        line = str(row.get("line", ""))
        reason = ""
        if float(row["better_score_pax_minutes"]) < float(args.min_better_score):
            reason = "below_better_score_threshold"
        elif float(row["exact_existing_path_invalidated_daily_pax"]) > 0:
            reason = "invalidates_current_path_transfer"
        elif pid in used_ids:
            reason = "duplicate_or_seed_physical_shift"
        elif set(_affected_keys(row)) & used_event_keys:
            reason = "overlaps_already_selected_event"
        elif accepted_by_year[year] >= int(args.max_new_rows_per_year):
            reason = "year_cap_reached"
        elif accepted_by_line_year[(year, line)] >= int(args.max_new_rows_per_line_year):
            reason = "line_year_cap_reached"
        elif len(accepted) >= int(args.max_new_rows_total):
            reason = "global_cap_reached"
        if reason:
            rejected_dynamic.append({"physical_shift_id": pid, "year": year, "status": "rejected_dynamic", "reason": reason, **row})
            if reason == "global_cap_reached":
                break
            continue
        ok, new_rows, info = _try_apply_candidate(
            raw_row=row,
            rank=len(seed_written) + len(accepted) + 1,
            rows=working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        if not ok:
            rejected_dynamic.append({"physical_shift_id": pid, "year": year, "status": "rejected_timetable_apply", "reason": info.get("reason", ""), **row})
            continue
        working_rows = new_rows
        used_ids.add(pid)
        used_event_keys.update(_affected_keys(row))
        accepted_by_year[year] += 1
        accepted_by_line_year[(year, line)] += 1
        cumulative_better += float(row["better_score_pax_minutes"])
        cumulative_exact_existing += float(row["exact_existing_path_pax_minutes_saved"])
        delta = _safe_int(row.get("delta_min"), 0)
        accepted_row = {
            "selection_order": len(accepted) + 1,
            "scenario_role": "new_better_optimization",
            "year": year,
            "action": row.get("actions") or row.get("action", ""),
            "line": line,
            "delta_min": delta,
            "section_from_station": row.get("section_from_station", ""),
            "section_departure_time_mod120": row.get("section_departure_time_mod120", ""),
            "section_departure_time_new_mod120": _shift_mod120(row.get("section_departure_time_mod120", "00:00"), delta),
            "section_to_station": row.get("section_to_station", ""),
            "section_arrival_time_mod120": row.get("section_arrival_time_mod120", ""),
            "section_arrival_time_new_mod120": _shift_mod120(row.get("section_arrival_time_mod120", "00:00"), delta),
            "physical_shift_id": pid,
            "source_candidate_ids": row.get("source_candidate_ids", ""),
            "static_net_pax_minutes_proxy": row.get("net_pax_minutes_proxy", ""),
            "static_gross_gain_pax_minutes_proxy": row.get("gross_gain_pax_minutes_proxy", ""),
            "static_gross_loss_pax_minutes_proxy": row.get("gross_loss_pax_minutes_proxy", ""),
            "near_miss_gain_pax_minutes": row.get("near_miss_gain_pax_minutes", ""),
            "existing_worsened_pax_minutes": row.get("existing_worsened_pax_minutes", ""),
            "exact_existing_path_pax_minutes_saved": row["exact_existing_path_pax_minutes_saved"],
            "discounted_near_miss_pax_minutes": row["discounted_near_miss_pax_minutes"],
            "better_score_pax_minutes": row["better_score_pax_minutes"],
            "cumulative_better_score_pax_minutes": round(cumulative_better, 6),
            "cumulative_exact_existing_path_pax_minutes_saved": round(cumulative_exact_existing, 6),
            "exact_existing_path_weight_touched": row["exact_existing_path_weight_touched"],
            "exact_existing_path_count_touched": row["exact_existing_path_count_touched"],
            "exact_existing_path_mean_total_delta_min": row["exact_existing_path_mean_total_delta_min"],
            "exact_existing_path_rolling_share_delta": row["exact_existing_path_rolling_share_delta"],
            "path_effect_example_od": row["path_effect_example_od"],
            "matched_cells": info.get("matched_cells", ""),
            "changed_cells": info.get("changed_cells", ""),
            "selection_rule": "exact_current_route_pax_minutes_plus_discounted_near_miss",
        }
        accepted.append(accepted_row)
        _log(
            log_path,
            f"accepted={len(accepted)} year={year} line={line} {row.get('section_from_station')}->{row.get('section_to_station')} "
            f"delta={delta} exact_existing={accepted_row['exact_existing_path_pax_minutes_saved']} "
            f"discounted_near={accepted_row['discounted_near_miss_pax_minutes']} better={accepted_row['better_score_pax_minutes']}",
        )

    final_pruned_rows: list[dict[str, Any]] = []
    final_exact: dict[str, Any] = {}
    prune_iterations = max(0, int(args.final_prune_max_iterations))
    current_root = Path(args.current_row_state_root).expanduser().resolve()
    if int(args.strict_final_no_invalidated):
        for iteration in range(prune_iterations + 1):
            final_exact = _scan_whole_scenario_existing_paths(
                years=years,
                rows_root=current_root,
                accepted_rows=accepted,
                station_min_transfer=station_min_transfer,
                log_path=log_path,
                progress_every_rows=max(1, int(args.progress_every_rows)),
                scan_row_limit=max(0, int(args.scan_row_limit)),
            )
            invalidated = float(final_exact.get("current_route_exact_invalidated_daily_pax", 0.0))
            if invalidated <= 1e-9:
                break
            invalid_by = final_exact.get("invalidated_by_physical_shift_id") or {}
            if not invalid_by or iteration >= prune_iterations:
                break
            accepted_by_id = {str(row.get("physical_shift_id", "")): row for row in accepted}
            prune_id = max(
                (cid for cid in invalid_by if cid in accepted_by_id),
                key=lambda cid: (
                    float(invalid_by[cid].get("daily_pax", 0.0)),
                    -float(accepted_by_id[cid].get("better_score_pax_minutes", 0.0)),
                    str(cid),
                ),
                default="",
            )
            if not prune_id:
                break
            removed = accepted_by_id[prune_id]
            accepted = [row for row in accepted if str(row.get("physical_shift_id", "")) != prune_id]
            removed = {
                **removed,
                "status": "rejected_final_combined_invalidated",
                "reason": (
                    "combined scenario invalidated current paths; candidate attributed "
                    f"{invalid_by[prune_id].get('daily_pax', 0.0)} daily pax in iteration {iteration + 1}"
                ),
            }
            final_pruned_rows.append(removed)
            rejected_dynamic.append(removed)
            _log(
                log_path,
                f"final-prune iteration={iteration + 1} removed={prune_id} "
                f"attributed_invalidated_daily_pax={invalid_by[prune_id].get('daily_pax', 0.0)} "
                f"remaining_accepted={len(accepted)}",
            )
    else:
        final_exact = _scan_whole_scenario_existing_paths(
            years=years,
            rows_root=current_root,
            accepted_rows=accepted,
            station_min_transfer=station_min_transfer,
            log_path=log_path,
            progress_every_rows=max(1, int(args.progress_every_rows)),
            scan_row_limit=max(0, int(args.scan_row_limit)),
        )
    if not final_exact:
        final_exact = _scan_whole_scenario_existing_paths(
            years=years,
            rows_root=current_root,
            accepted_rows=accepted,
            station_min_transfer=station_min_transfer,
            log_path=log_path,
            progress_every_rows=max(1, int(args.progress_every_rows)),
            scan_row_limit=max(0, int(args.scan_row_limit)),
        )
    scenario_passes_acceptance = True
    acceptance_reasons: list[str] = []
    if int(args.strict_final_no_invalidated) and float(final_exact.get("current_route_exact_invalidated_daily_pax", 0.0)) > 1e-9:
        scenario_passes_acceptance = False
        acceptance_reasons.append("combined scenario still invalidates current paths")
    if int(args.require_final_positive_vs_current):
        if float(final_exact.get("current_route_exact_pax_minutes_saved", 0.0)) <= 0:
            scenario_passes_acceptance = False
            acceptance_reasons.append("combined scenario does not save positive pax-minutes versus current scenario")
        if float(final_exact.get("current_route_exact_mean_total_delta_min", 0.0)) >= 0:
            scenario_passes_acceptance = False
            acceptance_reasons.append("combined scenario does not lower weighted mean total time versus current scenario")
    final_discounted_near = sum(float(row["discounted_near_miss_pax_minutes"]) for row in accepted)
    final_exact["discounted_near_miss_pax_minutes_selected"] = round(final_discounted_near, 6)
    final_exact["better_score_selected_total"] = round(float(final_exact["current_route_exact_pax_minutes_saved"]) + final_discounted_near, 6)

    # If the strict final check pruned rows, rebuild the preview timetable from scratch so it
    # exactly matches the final accepted list rather than the earlier greedy working copy.
    cumulative_better = 0.0
    cumulative_exact_existing = 0.0
    for idx, row in enumerate(accepted, start=1):
        row["selection_order"] = idx
        cumulative_better += float(row.get("better_score_pax_minutes", 0.0))
        cumulative_exact_existing += float(row.get("exact_existing_path_pax_minutes_saved", 0.0))
        row["cumulative_better_score_pax_minutes"] = round(cumulative_better, 6)
        row["cumulative_exact_existing_path_pax_minutes_saved"] = round(cumulative_exact_existing, 6)

    final_working_rows = timetable_rows
    for rank, row in enumerate(seed_rows, start=1):
        ok, final_working_rows, info = _try_apply_candidate(
            raw_row=row,
            rank=rank,
            rows=final_working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        if not ok:
            raise RuntimeError(f"Seed row failed during final preview rebuild: {_physical_id(row)} {info}")
    for rank, row in enumerate(accepted, start=len(seed_rows) + 1):
        ok, final_working_rows, info = _try_apply_candidate(
            raw_row=row,
            rank=rank,
            rows=final_working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        if not ok:
            raise RuntimeError(f"Accepted row failed during final preview rebuild: {_physical_id(row)} {info}")
    working_rows = final_working_rows

    fields = [
        "selection_order",
        "scenario_role",
        "year",
        "action",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_departure_time_new_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "section_arrival_time_new_mod120",
        "physical_shift_id",
        "source_candidate_ids",
        "static_net_pax_minutes_proxy",
        "static_gross_gain_pax_minutes_proxy",
        "static_gross_loss_pax_minutes_proxy",
        "near_miss_gain_pax_minutes",
        "existing_worsened_pax_minutes",
        "exact_existing_path_pax_minutes_saved",
        "discounted_near_miss_pax_minutes",
        "better_score_pax_minutes",
        "cumulative_better_score_pax_minutes",
        "cumulative_exact_existing_path_pax_minutes_saved",
        "exact_existing_path_weight_touched",
        "exact_existing_path_count_touched",
        "exact_existing_path_mean_total_delta_min",
        "exact_existing_path_rolling_share_delta",
        "path_effect_example_od",
        "matched_cells",
        "changed_cells",
        "selection_rule",
    ]
    _write_dict_csv(work_dir / "BetterOptimization_ProposedNewTimetableChanges_2026_2035.csv", accepted, fields)
    seed_fields = sorted({key for row in seed_written for key in row})
    if seed_written:
        _write_dict_csv(work_dir / "BetterOptimization_SeededFilteredChanges.csv", seed_written, seed_fields)
    scenario_rows = [*seed_written, *accepted]
    scenario_fields = sorted({key for row in scenario_rows for key in row})
    if "scenario_role" in scenario_fields:
        scenario_fields = ["scenario_role", *[f for f in scenario_fields if f != "scenario_role"]]
    _write_dict_csv(work_dir / "BetterOptimization_AllScenarioChanges_SeedPlusNew_2026_2035.csv", scenario_rows, scenario_fields)
    _write_dict_csv(
        work_dir / "BetterOptimization_RejectedOrDeferred_2026_2035.csv",
        rejected_dynamic,
        sorted({key for row in rejected_dynamic for key in row}) if rejected_dynamic else ["status", "reason"],
    )
    _write_dict_csv(
        work_dir / "BetterOptimization_CandidateScores_2026_2035.csv",
        scored_rows,
        sorted({key for row in scored_rows for key in row}),
    )
    _write_rows(work_dir / "FullTimetableHistory2026_2035_BETTER_OPTIMIZATION_preview.csv", working_rows)

    final_accepted_by_year = Counter(str(row.get("year", "")) for row in accepted)
    summary = {
        "created_at": _now(),
        "years": years,
        "method": "Greedy exact-current-route path-level evaluation plus discounted near-miss proxy; final full reroute validation still recommended before thesis adoption.",
        "seed_with_filtered": bool(args.seed_with_filtered),
        "seed_rows": len(seed_written),
        "candidate_rows_after_static_filters": len(candidate_rows),
        "accepted_new_rows": len(accepted),
        "final_pruned_rows": len(final_pruned_rows),
        "accepted_new_rows_by_year": dict(final_accepted_by_year),
        "near_miss_discount": float(args.near_miss_discount),
        "selected_hub_year_subgraphs": len(selected_hubs),
        "scan_row_limit": max(0, int(args.scan_row_limit)),
        "current_row_state_root": str(Path(args.current_row_state_root).expanduser().resolve()),
        "benchmark_row_state_root": str(Path(args.benchmark_row_state_root).expanduser().resolve()) if int(args.score_benchmark_effects) else "",
        "scenario_passes_acceptance": scenario_passes_acceptance,
        "scenario_acceptance_reasons": acceptance_reasons,
        "final_existing_route_exact_summary": final_exact,
        "outputs": {
            "selected_hubs": str(work_dir / "BetterOptimization_SelectedHubs.csv"),
            "candidate_scores": str(work_dir / "BetterOptimization_CandidateScores_2026_2035.csv"),
            "proposed_new_changes": str(work_dir / "BetterOptimization_ProposedNewTimetableChanges_2026_2035.csv"),
            "scenario_changes_seed_plus_new": str(work_dir / "BetterOptimization_AllScenarioChanges_SeedPlusNew_2026_2035.csv"),
            "preview_timetable": str(work_dir / "FullTimetableHistory2026_2035_BETTER_OPTIMIZATION_preview.csv"),
            "log": str(log_path),
        },
    }
    (work_dir / "BetterOptimization_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(log_path, f"Run complete accepted_new_rows={len(accepted)} final_summary={json.dumps(final_exact, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
