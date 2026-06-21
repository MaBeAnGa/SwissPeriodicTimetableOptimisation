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

from build_optimised_v2_timetable import (
    DEFAULT_INPUT_TIMETABLE,
    YEAR_COLUMNS,
    _as_float,
    _candidate_is_strict_safe,
    _discover_year_column_indices,
    _format_mod120,
    _load_rows,
    _matching_section_edits,
    _parse_mod120,
    _parse_segments,
    _row_to_candidate,
    _write_rows,
)
from build_optimised_v4_compatible_timetable import _copy_rows, _validate_segment_column_compatibility
from build_transfer_shift_net_impact import (
    _affected_keys,
    _load_existing_transfer_occurrences,
    _load_min_transfer_from_ledger,
    _load_near_miss_index,
    build_net_impact,
)
from setup_transfer_optimization_tables import (
    DAY_END_DEPART,
    DAY_START,
    _build_candidates,
    _build_event_ledger,
    _build_near_misses,
    _build_shift_feasibility,
    _load_min_transfer,
    _write_csv,
    load_high_flow_transfers,
    load_timetable_segments,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = BASE_DIR / "v0 CSVs"
DEFAULT_TRANSFER_2026 = DEFAULT_INPUT_DIR / "Transfers_2026_v0.csv"
DEFAULT_TRANSFER_2035 = DEFAULT_INPUT_DIR / "Transfers_2035_v0.csv"
DEFAULT_STATIONS = BASE_DIR / "FilteredStations2026_2035.csv"
DEFAULT_WORK_DIR = BASE_DIR / "transfer_optimization_tables" / "proposed_timetable_changes_v0"
DEFAULT_PROPOSED = DEFAULT_WORK_DIR / "ProposedTimetableChanges.csv"
DEFAULT_REJECTED = DEFAULT_WORK_DIR / "ProposedTimetableChanges_RejectedOrDeferred.csv"
DEFAULT_TIMETABLE = DEFAULT_WORK_DIR / "FullTimetableHistory2026_2035_ProposedTimetableChanges_preview.csv"
DEFAULT_SUMMARY = DEFAULT_WORK_DIR / "ProposedTimetableChanges_summary.json"
DEFAULT_LOG = DEFAULT_WORK_DIR / "ProposedTimetableChanges.log"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _fmt(value: float, digits: int = 6) -> str:
    if not math.isfinite(float(value)):
        return ""
    return f"{float(value):.{digits}f}".rstrip("0").rstrip(".")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _log(log_path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_dict_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def _shift_mod120(value: str, delta: int) -> str:
    return _format_mod120(_parse_mod120(value) + int(delta))


def _candidate_sort_key(row: dict[str, str]) -> tuple[Any, ...]:
    return (
        -_safe_float(row.get("net_pax_minutes_proxy"), 0.0),
        -_safe_float(row.get("gross_gain_pax_minutes_proxy"), 0.0),
        _safe_float(row.get("gross_loss_pax_minutes_proxy"), 0.0),
        str(row.get("year", "")),
        str(row.get("physical_shift_id", "")),
    )


def _effect_event_keys(effect: dict[str, Any]) -> tuple[tuple[str, str, str, str, int], tuple[str, str, str, str, int]]:
    year = str(effect["year"])
    station = str(effect["transfer_station"])
    arr_key = (
        year,
        station,
        "arrival",
        str(effect["arrival_line"]),
        int(effect["arrival_minute"]) % 120,
    )
    dep_key = (
        year,
        station,
        "departure",
        str(effect["departure_line"]),
        int(effect["departure_minute"]) % 120,
    )
    return arr_key, dep_key


def _effect_contribution(
    effect: dict[str, Any],
    *,
    event_shift_min: dict[tuple[str, str, str, str, int], int],
    min_transfer_lookup: dict[tuple[str, str], int],
) -> dict[str, Any]:
    arr_key, dep_key = _effect_event_keys(effect)
    arr_shift = int(event_shift_min.get(arr_key, 0))
    dep_shift = int(event_shift_min.get(dep_key, 0))
    old_wait = int(effect["old_wait_minutes"])
    new_wait = old_wait + dep_shift - arr_shift
    pax = float(effect["daily_pax_proxy"])
    year = str(effect["year"])
    station = str(effect["transfer_station"])
    min_transfer = min_transfer_lookup.get((year, station))
    if min_transfer is None:
        min_transfer = _safe_int(effect.get("minimum_transfer_time"), 0)

    source = str(effect.get("source", ""))
    if source == "existing_transfer":
        if new_wait < min_transfer:
            status = "broken_existing_transfer"
            # Broken existing transfers are hard-rejected at selection time. The
            # penalty below only gives them a stable negative ordering in logs.
            contribution = -pax * (min_transfer - new_wait + 60)
        else:
            delta_wait = old_wait - new_wait
            if delta_wait > 0:
                status = "improved_existing_transfer"
            elif delta_wait < 0:
                status = "worsened_existing_transfer"
            else:
                status = "unchanged_existing_transfer"
            contribution = pax * delta_wait
    elif source == "near_miss":
        old_shortfall = max(0, min_transfer - old_wait)
        new_shortfall = max(0, min_transfer - new_wait)
        closed = old_shortfall - new_shortfall
        if old_shortfall > 0 and new_shortfall == 0:
            status = "unlocked_near_miss"
        elif closed > 0:
            status = "partly_improved_near_miss"
        elif closed < 0:
            status = "worsened_near_miss"
        else:
            status = "unchanged_near_miss"
        contribution = pax * closed
    else:
        status = "unknown_effect"
        contribution = 0.0

    return {
        "status": status,
        "contribution": float(contribution),
        "new_wait_minutes": int(new_wait),
        "old_wait_minutes": int(old_wait),
        "minimum_transfer_time": int(min_transfer),
        "daily_pax": float(pax),
        "arrival_shift_min": arr_shift,
        "departure_shift_min": dep_shift,
    }


def _candidate_effects(
    candidate: dict[str, str],
    *,
    existing_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    near_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
) -> dict[tuple[str, str], dict[str, Any]]:
    effects: dict[tuple[str, str], dict[str, Any]] = {}
    for key in _affected_keys(candidate):
        for effect in existing_index.get(key, []):
            effects[(str(effect["source"]), str(effect["effect_id"]))] = effect
        for effect in near_index.get(key, []):
            effects[(str(effect["source"]), str(effect["effect_id"]))] = effect
    return effects


def _score_candidate_current_state(
    candidate: dict[str, str],
    *,
    existing_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    near_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    event_shift_min: dict[tuple[str, str, str, str, int], int],
    min_transfer_lookup: dict[tuple[str, str], int],
) -> dict[str, Any]:
    effects = _candidate_effects(candidate, existing_index=existing_index, near_index=near_index)
    delta = int(candidate["delta_min"])
    trial_shift = dict(event_shift_min)
    for key in _affected_keys(candidate):
        trial_shift[key] = int(trial_shift.get(key, 0)) + delta

    current_total = 0.0
    trial_total = 0.0
    counts = Counter()
    pax_by_status = defaultdict(float)
    examples: list[str] = []

    for effect in effects.values():
        before = _effect_contribution(
            effect,
            event_shift_min=event_shift_min,
            min_transfer_lookup=min_transfer_lookup,
        )
        after = _effect_contribution(
            effect,
            event_shift_min=trial_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        current_total += float(before["contribution"])
        trial_total += float(after["contribution"])
        status = str(after["status"])
        counts[status] += 1
        pax_by_status[status] += float(after["daily_pax"])
        if len(examples) < 8 and status not in {"unchanged_existing_transfer", "unchanged_near_miss"}:
            examples.append(
                f"{effect['transfer_station']} {effect['arrival_line']}->{effect['departure_line']} "
                f"{effect['arrival_time']}->{effect['departure_time']} "
                f"wait {before['new_wait_minutes']}->{after['new_wait_minutes']} "
                f"min={after['minimum_transfer_time']} pax={_fmt(after['daily_pax'], 3)} status={status}"
            )

    gross_gain = 0.0
    gross_loss = 0.0
    for effect in effects.values():
        before = _effect_contribution(
            effect,
            event_shift_min=event_shift_min,
            min_transfer_lookup=min_transfer_lookup,
        )
        after = _effect_contribution(
            effect,
            event_shift_min=trial_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        diff = float(after["contribution"]) - float(before["contribution"])
        if diff >= 0:
            gross_gain += diff
        else:
            gross_loss += -diff

    return {
        "current_objective_on_affected_effects": current_total,
        "trial_objective_on_affected_effects": trial_total,
        "marginal_net_pax_minutes": trial_total - current_total,
        "marginal_gross_gain_pax_minutes": gross_gain,
        "marginal_gross_loss_pax_minutes": gross_loss,
        "effect_count": len(effects),
        "status_counts": dict(counts),
        "pax_by_status": {k: round(v, 6) for k, v in sorted(pax_by_status.items())},
        "broken_existing_daily_pax": pax_by_status.get("broken_existing_transfer", 0.0),
        "worsened_existing_daily_pax": pax_by_status.get("worsened_existing_transfer", 0.0),
        "improved_existing_daily_pax": pax_by_status.get("improved_existing_transfer", 0.0),
        "unlocked_near_miss_daily_pax": pax_by_status.get("unlocked_near_miss", 0.0),
        "partly_improved_near_miss_daily_pax": pax_by_status.get("partly_improved_near_miss", 0.0),
        "examples": " | ".join(examples),
    }


def _build_source_tables(
    *,
    args: argparse.Namespace,
    log_path: Path,
    work_dir: Path,
    transfer_paths: dict[str, Path],
    years: list[str],
) -> dict[str, Path]:
    output_dir = work_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(log_path, "Building source transfer-optimization tables from current Transfers_YEAR_v0.csv files")
    min_transfer_lookup = _load_min_transfer(Path(args.stations_csv).expanduser().resolve())
    segments = load_timetable_segments(Path(args.timetable_csv).expanduser().resolve())

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
        _log(log_path, f"{year}: high-flow transfer patterns={len(high_keys)} stations={len(stations)}")

    event_ledger = _build_event_ledger(
        years=years,
        segments=segments,
        high_data=high_data,
        min_transfer_lookup=min_transfer_lookup,
    )
    event_ledger_path = output_dir / "Transfer_Optimization_Event_Ledger_2026_2035.csv"
    _write_csv(
        event_ledger_path,
        event_ledger,
        [
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
        ],
    )
    _log(log_path, f"Wrote event ledger rows={len(event_ledger)}")

    near_misses = _build_near_misses(
        event_ledger,
        early_window_min=max(0, int(args.near_miss_before_min)),
        late_window_min=max(0, int(args.near_miss_after_min)),
    )
    near_path = output_dir / "Transfer_Optimization_Near_Misses_2026_2035.csv"
    _write_csv(
        near_path,
        near_misses,
        [
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
        ],
    )
    _log(log_path, f"Wrote near misses rows={len(near_misses)}")

    shift_feasibility = _build_shift_feasibility(
        years=years,
        segments=segments,
        ledger_event_ids={row["event_id"] for row in event_ledger},
        max_shift_min=max(1, int(args.max_shift_min)),
    )
    feasibility_path = output_dir / "Transfer_Optimization_Shift_Feasibility_2026_2035.csv"
    _write_csv(
        feasibility_path,
        shift_feasibility,
        [
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
        ],
    )
    _log(log_path, f"Wrote shift feasibility rows={len(shift_feasibility)}")

    candidates = _build_candidates(
        years=years,
        transfer_paths=transfer_paths,
        near_misses=near_misses,
        shift_feasibility=shift_feasibility,
        min_transfer_lookup=min_transfer_lookup,
        threshold_pax=float(args.threshold_pax),
        max_shift_min=max(1, int(args.max_shift_min)),
    )
    candidates_path = output_dir / "Transfer_Optimization_Candidates_2026_2035.csv"
    _write_csv(
        candidates_path,
        candidates,
        [
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
        ],
    )
    _log(log_path, f"Wrote first-pass candidates rows={len(candidates)}")

    net_summary = build_net_impact(
        candidates_path=candidates_path,
        event_ledger_path=event_ledger_path,
        near_miss_path=near_path,
        transfer_paths=transfer_paths,
        output_dir=output_dir,
        threshold_pax=float(args.threshold_pax),
        detail_top_n=int(args.detail_top_n),
    )
    _log(log_path, f"Wrote net-impact tables summary={json.dumps(net_summary.get('row_counts', {}), ensure_ascii=False)}")

    return {
        "event_ledger": event_ledger_path,
        "near_misses": near_path,
        "shift_feasibility": feasibility_path,
        "candidates": candidates_path,
        "physical_shifts": output_dir / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv",
    }


def _unique_edits(edits: list[tuple[int, int, str, str, str]]) -> list[tuple[int, int, str, str, str]]:
    out: list[tuple[int, int, str, str, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for edit in edits:
        key = (edit[0], edit[1], edit[3])
        if key in seen:
            continue
        seen.add(key)
        out.append(edit)
    return out


def _try_apply_candidate(
    *,
    raw_row: dict[str, str],
    rank: int,
    rows: list[list[str]],
    segments: list[Any],
    row_to_segment: dict[int, Any],
    year_col_indices: dict[str, list[int]],
) -> tuple[bool, list[list[str]], dict[str, Any]]:
    candidate = _row_to_candidate(raw_row, rank=rank)
    try:
        edits = _matching_section_edits(
            rows=rows,
            segments=segments,
            year_column_indices=year_col_indices,
            candidate=candidate,
        )
    except Exception as exc:
        return False, rows, {"status": "skipped_match_error", "reason": str(exc), "matched_cells": 0, "changed_cells": 0}

    unique_edits = _unique_edits(edits)
    if not unique_edits:
        return False, rows, {
            "status": "skipped_unmatched_after_prior_edits",
            "reason": "no exact current timetable token match after previously accepted edits",
            "matched_cells": 0,
            "changed_cells": 0,
        }

    trial_rows = _copy_rows(rows)
    touched_segment_cols: set[tuple[int, int]] = set()
    changed = 0
    for row_idx, col_idx, old_value, new_value, _role in unique_edits:
        if trial_rows[row_idx][col_idx] != old_value:
            return False, rows, {
                "status": "skipped_stale_token",
                "reason": (
                    f"current token changed: row={row_idx + 1} col={trial_rows[0][col_idx]!r} "
                    f"expected={old_value!r} found={trial_rows[row_idx][col_idx]!r}"
                ),
                "matched_cells": len(unique_edits),
                "changed_cells": 0,
            }
        trial_rows[row_idx][col_idx] = new_value
        if old_value != new_value:
            changed += 1
        segment = row_to_segment.get(row_idx)
        if segment is not None:
            touched_segment_cols.add((id(segment), col_idx))

    id_to_segment = {id(segment): segment for segment in segments}
    validation_errors: list[str] = []
    for segment_id, col_idx in sorted(touched_segment_cols, key=lambda item: (item[0], item[1])):
        segment = id_to_segment[segment_id]
        ok, reason = _validate_segment_column_compatibility(
            before_rows=rows,
            after_rows=trial_rows,
            segment=segment,
            col_idx=col_idx,
        )
        if not ok:
            validation_errors.append(reason)
    if validation_errors:
        return False, rows, {
            "status": "skipped_cumulative_incompatible",
            "reason": " | ".join(validation_errors[:5]),
            "matched_cells": len(unique_edits),
            "changed_cells": 0,
        }

    return True, trial_rows, {
        "status": "accepted",
        "reason": "",
        "matched_cells": len(unique_edits),
        "changed_cells": changed,
    }


def _selection_fields() -> list[str]:
    return [
        "selection_order",
        "pass_number",
        "priority_rank_static",
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
        "marginal_net_pax_minutes",
        "marginal_gross_gain_pax_minutes",
        "marginal_gross_loss_pax_minutes",
        "cumulative_net_pax_minutes_after",
        "effect_count",
        "status_counts",
        "pax_by_status",
        "broken_existing_daily_pax",
        "worsened_existing_daily_pax",
        "improved_existing_daily_pax",
        "unlocked_near_miss_daily_pax",
        "partly_improved_near_miss_daily_pax",
        "matched_cells",
        "changed_cells",
        "physical_shift_id",
        "source_candidate_ids",
        "input_candidate_rows_merged",
        "static_net_pax_minutes_proxy",
        "static_gross_gain_pax_minutes_proxy",
        "static_gross_loss_pax_minutes_proxy",
        "existing_gain_daily_pax_proxy",
        "near_miss_gain_daily_pax_proxy",
        "existing_worsened_daily_pax_proxy",
        "candidate_types",
        "transfer_stations",
        "affected_transfer_pairs",
        "net_affected_transfer_stations",
        "net_affected_transfer_pairs",
        "example_event_pairs",
        "dynamic_effect_examples",
        "selection_rule",
    ]


def _rejection_fields() -> list[str]:
    return [
        "last_pass_seen",
        "priority_rank_static",
        "year",
        "status",
        "reason",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "marginal_net_pax_minutes",
        "marginal_gross_gain_pax_minutes",
        "marginal_gross_loss_pax_minutes",
        "broken_existing_daily_pax",
        "worsened_existing_daily_pax",
        "effect_count",
        "status_counts",
        "physical_shift_id",
        "static_net_pax_minutes_proxy",
        "dynamic_effect_examples",
    ]


def _accepted_record(
    *,
    raw_row: dict[str, str],
    score: dict[str, Any],
    selection_order: int,
    pass_number: int,
    priority_rank: int,
    cumulative_net: float,
    apply_info: dict[str, Any],
) -> dict[str, Any]:
    delta = int(raw_row["delta_min"])
    return {
        "selection_order": selection_order,
        "pass_number": pass_number,
        "priority_rank_static": priority_rank,
        "year": raw_row.get("year", ""),
        "action": raw_row.get("actions") or raw_row.get("action", ""),
        "line": raw_row.get("line", ""),
        "delta_min": delta,
        "section_from_station": raw_row.get("section_from_station", ""),
        "section_departure_time_mod120": raw_row.get("section_departure_time_mod120", ""),
        "section_departure_time_new_mod120": _shift_mod120(raw_row.get("section_departure_time_mod120", "00:00"), delta),
        "section_to_station": raw_row.get("section_to_station", ""),
        "section_arrival_time_mod120": raw_row.get("section_arrival_time_mod120", ""),
        "section_arrival_time_new_mod120": _shift_mod120(raw_row.get("section_arrival_time_mod120", "00:00"), delta),
        "marginal_net_pax_minutes": _fmt(score["marginal_net_pax_minutes"]),
        "marginal_gross_gain_pax_minutes": _fmt(score["marginal_gross_gain_pax_minutes"]),
        "marginal_gross_loss_pax_minutes": _fmt(score["marginal_gross_loss_pax_minutes"]),
        "cumulative_net_pax_minutes_after": _fmt(cumulative_net),
        "effect_count": score["effect_count"],
        "status_counts": json.dumps(score["status_counts"], ensure_ascii=False, sort_keys=True),
        "pax_by_status": json.dumps(score["pax_by_status"], ensure_ascii=False, sort_keys=True),
        "broken_existing_daily_pax": _fmt(score["broken_existing_daily_pax"]),
        "worsened_existing_daily_pax": _fmt(score["worsened_existing_daily_pax"]),
        "improved_existing_daily_pax": _fmt(score["improved_existing_daily_pax"]),
        "unlocked_near_miss_daily_pax": _fmt(score["unlocked_near_miss_daily_pax"]),
        "partly_improved_near_miss_daily_pax": _fmt(score["partly_improved_near_miss_daily_pax"]),
        "matched_cells": apply_info.get("matched_cells", ""),
        "changed_cells": apply_info.get("changed_cells", ""),
        "physical_shift_id": raw_row.get("physical_shift_id", ""),
        "source_candidate_ids": raw_row.get("source_candidate_ids", ""),
        "input_candidate_rows_merged": raw_row.get("input_candidate_rows_merged", ""),
        "static_net_pax_minutes_proxy": raw_row.get("net_pax_minutes_proxy", ""),
        "static_gross_gain_pax_minutes_proxy": raw_row.get("gross_gain_pax_minutes_proxy", ""),
        "static_gross_loss_pax_minutes_proxy": raw_row.get("gross_loss_pax_minutes_proxy", ""),
        "existing_gain_daily_pax_proxy": raw_row.get("existing_gain_daily_pax_proxy", ""),
        "near_miss_gain_daily_pax_proxy": raw_row.get("near_miss_gain_daily_pax_proxy", ""),
        "existing_worsened_daily_pax_proxy": raw_row.get("existing_worsened_daily_pax_proxy", ""),
        "candidate_types": raw_row.get("candidate_types", ""),
        "transfer_stations": raw_row.get("transfer_stations", ""),
        "affected_transfer_pairs": raw_row.get("affected_transfer_pairs", ""),
        "net_affected_transfer_stations": raw_row.get("net_affected_transfer_stations", ""),
        "net_affected_transfer_pairs": raw_row.get("net_affected_transfer_pairs", ""),
        "example_event_pairs": raw_row.get("example_event_pairs", ""),
        "dynamic_effect_examples": score.get("examples", ""),
        "selection_rule": "accepted iff current marginal net pax-minutes > min threshold, no existing high-flow transfer is broken, and cumulative timetable compatibility validation passes",
    }


def _rejected_record(
    *,
    raw_row: dict[str, str],
    score: dict[str, Any] | None,
    last_pass: int,
    priority_rank: int,
    status: str,
    reason: str,
) -> dict[str, Any]:
    score = score or {}
    return {
        "last_pass_seen": last_pass,
        "priority_rank_static": priority_rank,
        "year": raw_row.get("year", ""),
        "status": status,
        "reason": reason,
        "line": raw_row.get("line", ""),
        "delta_min": raw_row.get("delta_min", ""),
        "section_from_station": raw_row.get("section_from_station", ""),
        "section_departure_time_mod120": raw_row.get("section_departure_time_mod120", ""),
        "section_to_station": raw_row.get("section_to_station", ""),
        "section_arrival_time_mod120": raw_row.get("section_arrival_time_mod120", ""),
        "marginal_net_pax_minutes": _fmt(_safe_float(score.get("marginal_net_pax_minutes"), 0.0)),
        "marginal_gross_gain_pax_minutes": _fmt(_safe_float(score.get("marginal_gross_gain_pax_minutes"), 0.0)),
        "marginal_gross_loss_pax_minutes": _fmt(_safe_float(score.get("marginal_gross_loss_pax_minutes"), 0.0)),
        "broken_existing_daily_pax": _fmt(_safe_float(score.get("broken_existing_daily_pax"), 0.0)),
        "worsened_existing_daily_pax": _fmt(_safe_float(score.get("worsened_existing_daily_pax"), 0.0)),
        "effect_count": score.get("effect_count", ""),
        "status_counts": json.dumps(score.get("status_counts", {}), ensure_ascii=False, sort_keys=True),
        "physical_shift_id": raw_row.get("physical_shift_id", ""),
        "static_net_pax_minutes_proxy": raw_row.get("net_pax_minutes_proxy", ""),
        "dynamic_effect_examples": score.get("examples", ""),
    }


def _select_proposals(
    *,
    args: argparse.Namespace,
    log_path: Path,
    source_paths: dict[str, Path],
    transfer_paths: dict[str, Path],
    years: list[str],
) -> dict[str, Any]:
    rows = _load_rows(Path(args.timetable_csv).expanduser().resolve())
    year_col_indices = _discover_year_column_indices(rows[0])
    segments = _parse_segments(rows)
    row_to_segment: dict[int, Any] = {}
    for segment in segments:
        for row_idx in segment.row_indices:
            row_to_segment[row_idx] = segment

    physical_rows = [
        row
        for row in _read_csv(source_paths["physical_shifts"])
        if _candidate_is_strict_safe(row, set(years))
        and _safe_float(row.get("net_pax_minutes_proxy"), 0.0) > 0.0
    ]
    physical_rows.sort(key=_candidate_sort_key)
    if int(args.max_candidates_considered) > 0:
        physical_rows = physical_rows[: int(args.max_candidates_considered)]
    _log(log_path, f"Loaded strict-safe positive physical-shift candidates={len(physical_rows)}")

    selected_transfer_paths = {year: transfer_paths[year] for year in years}
    existing_index, existing_counts = _load_existing_transfer_occurrences(
        selected_transfer_paths,
        threshold_pax=float(args.threshold_pax),
    )
    near_index, near_counts = _load_near_miss_index(source_paths["near_misses"])
    year_set = set(years)
    near_index = {key: value for key, value in near_index.items() if key[0] in year_set}
    near_counts = {year: count for year, count in near_counts.items() if year in year_set}
    min_transfer_lookup = _load_min_transfer_from_ledger(source_paths["event_ledger"])
    _log(log_path, f"Loaded effect indexes existing={existing_counts} near_miss={near_counts}")

    accepted: list[dict[str, Any]] = []
    rejected_latest: dict[str, dict[str, Any]] = {}
    event_shift_min: dict[tuple[str, str, str, str, int], int] = {}
    accepted_ids: set[str] = set()
    permanently_rejected: set[str] = set()
    cumulative_net = 0.0

    max_accept = int(args.max_accepted)
    max_passes = int(args.max_passes)
    min_net = float(args.min_marginal_net_pax_minutes)
    allow_broken = float(args.allow_broken_existing_daily_pax)

    for pass_number in range(1, max_passes + 1):
        accepted_this_pass = 0
        considered_this_pass = 0
        _log(log_path, f"Selection pass {pass_number}/{max_passes} start accepted={len(accepted)} cumulative_net={_fmt(cumulative_net)}")
        for priority_rank, raw_row in enumerate(physical_rows, start=1):
            physical_id = str(raw_row.get("physical_shift_id", ""))
            if physical_id in accepted_ids or physical_id in permanently_rejected:
                continue
            if max_accept > 0 and len(accepted) >= max_accept:
                break
            considered_this_pass += 1
            score = _score_candidate_current_state(
                raw_row,
                existing_index=existing_index,
                near_index=near_index,
                event_shift_min=event_shift_min,
                min_transfer_lookup=min_transfer_lookup,
            )
            net = float(score["marginal_net_pax_minutes"])
            broken_pax = float(score["broken_existing_daily_pax"])
            if broken_pax > allow_broken:
                permanently_rejected.add(physical_id)
                rejected_latest[physical_id] = _rejected_record(
                    raw_row=raw_row,
                    score=score,
                    last_pass=pass_number,
                    priority_rank=priority_rank,
                    status="rejected_breaks_existing_high_flow_transfer",
                    reason=f"trial would break existing high-flow transfer daily pax={_fmt(broken_pax)} > allowed {_fmt(allow_broken)}",
                )
                continue
            if net <= min_net:
                rejected_latest[physical_id] = _rejected_record(
                    raw_row=raw_row,
                    score=score,
                    last_pass=pass_number,
                    priority_rank=priority_rank,
                    status="deferred_non_positive_current_marginal_net",
                    reason=f"current marginal net pax-minutes {_fmt(net)} <= threshold {_fmt(min_net)}",
                )
                continue

            ok, trial_rows, apply_info = _try_apply_candidate(
                raw_row=raw_row,
                rank=priority_rank,
                rows=rows,
                segments=segments,
                row_to_segment=row_to_segment,
                year_col_indices=year_col_indices,
            )
            if not ok:
                permanently_rejected.add(physical_id)
                rejected_latest[physical_id] = _rejected_record(
                    raw_row=raw_row,
                    score=score,
                    last_pass=pass_number,
                    priority_rank=priority_rank,
                    status=str(apply_info.get("status", "skipped")),
                    reason=str(apply_info.get("reason", "")),
                )
                continue

            rows = trial_rows
            delta = int(raw_row["delta_min"])
            for key in _affected_keys(raw_row):
                event_shift_min[key] = int(event_shift_min.get(key, 0)) + delta
            cumulative_net += net
            accepted_ids.add(physical_id)
            accepted_this_pass += 1
            record = _accepted_record(
                raw_row=raw_row,
                score=score,
                selection_order=len(accepted) + 1,
                pass_number=pass_number,
                priority_rank=priority_rank,
                cumulative_net=cumulative_net,
                apply_info=apply_info,
            )
            accepted.append(record)
            if physical_id in rejected_latest:
                del rejected_latest[physical_id]

            if len(accepted) % int(args.log_every_accepts) == 0:
                _log(
                    log_path,
                    f"accepted={len(accepted)} pass={pass_number} latest={raw_row.get('year')} {raw_row.get('line')} "
                    f"{raw_row.get('section_from_station')}->{raw_row.get('section_to_station')} "
                    f"delta={delta} marginal_net={_fmt(net)} cumulative_net={_fmt(cumulative_net)}",
                )

        _log(
            log_path,
            f"Selection pass {pass_number} complete considered={considered_this_pass} "
            f"accepted_this_pass={accepted_this_pass} total_accepted={len(accepted)}",
        )
        if accepted_this_pass == 0:
            break
        if max_accept > 0 and len(accepted) >= max_accept:
            break

    rejected = list(rejected_latest.values())
    rejected.sort(key=lambda r: (str(r.get("status", "")), int(r.get("priority_rank_static") or 0)))

    proposed_path = Path(args.output_csv).expanduser().resolve()
    rejected_path = Path(args.rejected_csv).expanduser().resolve()
    _write_dict_csv(proposed_path, accepted, _selection_fields())
    _write_dict_csv(rejected_path, rejected, _rejection_fields())
    _write_rows(Path(args.output_timetable).expanduser().resolve(), rows)

    summary = {
        "created_at": _now(),
        "years": years,
        "transfer_inputs": {year: str(path) for year, path in transfer_paths.items()},
        "source_tables": {key: str(path) for key, path in source_paths.items()},
        "output_csv": str(proposed_path),
        "rejected_csv": str(rejected_path),
        "output_timetable_preview": str(Path(args.output_timetable).expanduser().resolve()),
        "strict_safe_positive_physical_candidates_considered": len(physical_rows),
        "accepted_changes": len(accepted),
        "rejected_or_deferred_rows_written": len(rejected),
        "cumulative_net_pax_minutes": round(cumulative_net, 6),
        "threshold_pax": float(args.threshold_pax),
        "max_shift_min": int(args.max_shift_min),
        "max_passes": int(args.max_passes),
        "max_accepted": int(args.max_accepted),
        "allow_broken_existing_daily_pax": float(args.allow_broken_existing_daily_pax),
        "method": [
            "Fresh high-flow event ledger, near-miss table, shift-feasibility table, and physical-shift net-impact table are generated from Transfers_YEAR_v0.csv.",
            "Candidates are limited to strict-safe physical shifts with positive static net pax-minute proxy.",
            "Selection is cumulative: accepted timetable edits are immediately applied to the working timetable.",
            "Each later candidate is scored against the current accepted state using all existing high-flow transfer effects and all near-miss effects touched by its endpoint events.",
            "A candidate is accepted only if marginal net pax-minutes remain positive, no existing high-flow transfer is broken, and the touched timetable segment preserves trip count, station/kind run structure, rolling durations, and reasonable dwell bounds.",
            "i/n/iP/nP suffixes are shifted using the existing suffix-aware token logic, so 59i + 1 min becomes 0n and P/Q flags are preserved.",
        ],
    }
    Path(args.summary_json).expanduser().resolve().write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(log_path, f"Wrote {proposed_path} accepted={len(accepted)}")
    _log(log_path, f"Wrote {rejected_path} rejected_or_deferred={len(rejected)}")
    _log(log_path, f"Wrote timetable preview {Path(args.output_timetable).expanduser().resolve()}")
    _log(log_path, f"Wrote summary {Path(args.summary_json).expanduser().resolve()}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate cumulative, network-scored, strict-compatible timetable-minute proposals "
            "from the validated Transfers_YEAR_v0.csv baselines."
        )
    )
    parser.add_argument("--timetable-csv", type=Path, default=DEFAULT_INPUT_TIMETABLE)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--transfer-2026", type=Path, default=DEFAULT_TRANSFER_2026)
    parser.add_argument("--transfer-2035", type=Path, default=DEFAULT_TRANSFER_2035)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_PROPOSED)
    parser.add_argument("--rejected-csv", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--output-timetable", type=Path, default=DEFAULT_TIMETABLE)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--threshold-pax", type=float, default=95.0)
    parser.add_argument("--near-miss-before-min", type=int, default=10)
    parser.add_argument("--near-miss-after-min", type=int, default=10)
    parser.add_argument("--max-shift-min", type=int, default=10)
    parser.add_argument("--detail-top-n", type=int, default=500)
    parser.add_argument("--max-passes", type=int, default=8)
    parser.add_argument("--max-accepted", type=int, default=0, help="0 means no explicit accepted-change cap.")
    parser.add_argument("--max-candidates-considered", type=int, default=0, help="0 means all strict-safe positive candidates.")
    parser.add_argument("--min-marginal-net-pax-minutes", type=float, default=0.0)
    parser.add_argument("--allow-broken-existing-daily-pax", type=float, default=0.0)
    parser.add_argument("--log-every-accepts", type=int, default=10)
    parser.add_argument("--reuse-source-tables", action="store_true")
    args = parser.parse_args()

    years = list(args.years)
    work_dir = Path(args.work_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_source_tables:
        log_path.write_text("", encoding="utf-8")
    _log(log_path, f"Proposed timetable changes run start years={years} work_dir={work_dir}")

    transfer_paths = {
        "2026": Path(args.transfer_2026).expanduser().resolve(),
        "2035": Path(args.transfer_2035).expanduser().resolve(),
    }
    for year in years:
        if not transfer_paths[year].exists():
            raise FileNotFoundError(f"Missing transfer input for {year}: {transfer_paths[year]}")

    if args.reuse_source_tables:
        source_paths = {
            "event_ledger": work_dir / "Transfer_Optimization_Event_Ledger_2026_2035.csv",
            "near_misses": work_dir / "Transfer_Optimization_Near_Misses_2026_2035.csv",
            "shift_feasibility": work_dir / "Transfer_Optimization_Shift_Feasibility_2026_2035.csv",
            "candidates": work_dir / "Transfer_Optimization_Candidates_2026_2035.csv",
            "physical_shifts": work_dir / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv",
        }
        missing = [str(path) for path in source_paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("Cannot reuse source tables; missing: " + "; ".join(missing))
        _log(log_path, "Reusing existing source tables in work-dir")
    else:
        source_paths = _build_source_tables(
            args=args,
            log_path=log_path,
            work_dir=work_dir,
            transfer_paths=transfer_paths,
            years=years,
        )

    summary = _select_proposals(
        args=args,
        log_path=log_path,
        source_paths=source_paths,
        transfer_paths=transfer_paths,
        years=years,
    )
    _log(log_path, f"Complete accepted={summary['accepted_changes']} cumulative_net={_fmt(summary['cumulative_net_pax_minutes'])}")


if __name__ == "__main__":
    main()
