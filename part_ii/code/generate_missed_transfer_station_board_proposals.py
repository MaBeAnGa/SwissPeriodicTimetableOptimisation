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

from build_optimised_v2_timetable import (
    DEFAULT_INPUT_TIMETABLE,
    _as_float,
    _discover_year_column_indices,
    _load_rows,
    _parse_segments,
    _row_to_candidate,
    _write_rows,
)
from build_transfer_shift_net_impact import _load_existing_transfer_occurrences, _load_min_transfer_from_ledger, _load_near_miss_index
from generate_proposed_timetable_changes import (
    BASE_DIR,
    DEFAULT_STATIONS,
    DEFAULT_TRANSFER_2026,
    DEFAULT_TRANSFER_2035,
    _accepted_record,
    _build_source_tables,
    _candidate_effects,
    _candidate_sort_key,
    _effect_contribution,
    _effect_event_keys,
    _fmt,
    _read_csv,
    _rejected_record,
    _score_candidate_current_state,
    _selection_fields,
    _shift_mod120,
    _try_apply_candidate,
    _write_dict_csv,
)
from repair_transfer_v0_duplicate_line_context import (
    LineProfile,
    _group_line_stops,
    _line_termini,
    _timings_for_line_direction,
    _year_columns,
)


DEFAULT_WORK_DIR = BASE_DIR / "transfer_optimization_tables" / "missed_transfer_station_board_v0"
DEFAULT_OUTPUT = DEFAULT_WORK_DIR / "ProposedTimetableChanges_MissedTransfers_StationBoardStrict.csv"
DEFAULT_AUDIT = DEFAULT_WORK_DIR / "MissedTransferCandidateAudit_StationBoardStrict.csv"
DEFAULT_REJECTED = DEFAULT_WORK_DIR / "MissedTransferRejectedOrDeferred_StationBoardStrict.csv"
DEFAULT_TIMETABLE = DEFAULT_WORK_DIR / "FullTimetableHistory2026_2035_MissedTransferStationBoardStrict_preview.csv"
DEFAULT_SUMMARY = DEFAULT_WORK_DIR / "MissedTransferStationBoardStrict_summary.json"
DEFAULT_LOG = DEFAULT_WORK_DIR / "MissedTransferStationBoardStrict.log"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(str(value).strip())
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _profiles_from_rows(rows: list[list[str]]) -> tuple[dict[str, list[LineProfile]], dict[str, list[str]]]:
    """Build Transfers_v0-style line profiles from the current working timetable."""

    line_blocks: list[tuple[str, list[list[str]]]] = []
    current_line: str | None = None
    current_rows: list[list[str]] = []
    header: list[str] | None = None

    for row in rows:
        if len(row) > 1 and str(row[1]).strip() == "an/ab":
            if current_line is not None:
                line_blocks.append((current_line, current_rows))
            current_line = str(row[0]).strip()
            current_rows = []
            header = row
            continue
        if current_line is not None:
            current_rows.append(row)
    if current_line is not None:
        line_blocks.append((current_line, current_rows))

    idx_by_name = {name: idx for idx, name in enumerate(header or [])}
    profiles: dict[str, list[LineProfile]] = defaultdict(list)
    for block_index, (line, block_rows) in enumerate(line_blocks):
        stops = _group_line_stops(block_rows, idx_by_name)
        profiles[line].append(
            LineProfile(
                line=line,
                block_index=block_index,
                stops=stops,
                termini=_line_termini([stop.station for stop in stops]),
            )
        )
    year_cols = {year: _year_columns(idx_by_name, year) for year in ("2026", "2035")}
    return dict(profiles), year_cols


def _profiles_for_revision(
    rows: list[list[str]],
    revision: int,
    cache: dict[int, tuple[dict[str, list[LineProfile]], dict[str, list[str]]]],
) -> tuple[dict[str, list[LineProfile]], dict[str, list[str]]]:
    if revision not in cache:
        cache[revision] = _profiles_from_rows(rows)
    return cache[revision]


def _proposal_line_timings(
    raw_row: dict[str, str],
    *,
    rows: list[list[str]],
    revision: int,
    profile_cache: dict[int, tuple[dict[str, list[LineProfile]], dict[str, list[str]]]],
) -> dict[str, str]:
    """Return line timing displays for the current/final working timetable.

    The arrival-side context is the shifted line arriving at the section's
    downstream station; the departure-side context is the same line departing
    from the section's upstream station. The imported Transfers_v0 helper only
    expands displayed runs departing in 05:00-06:59, so a shifted :00 token that
    becomes :59 appears as 05:59/06:59 rather than a misleading 04:59 run.
    """

    profiles, year_cols = _profiles_for_revision(rows, revision, profile_cache)
    line = str(raw_row.get("line", "")).strip()
    year = str(raw_row.get("year", "")).strip()
    section_from = str(raw_row.get("section_from_station", "")).strip()
    section_to = str(raw_row.get("section_to_station", "")).strip()

    arrival = _timings_for_line_direction(
        line=line,
        transfer_station=section_to,
        prior_nodes=[section_from] if section_from else [],
        following_nodes=[],
        year=year,
        profiles=profiles,
        year_cols=year_cols,
        mode="arrival",
    )
    departure = _timings_for_line_direction(
        line=line,
        transfer_station=section_from,
        prior_nodes=[],
        following_nodes=[section_to] if section_to else [],
        year=year,
        profiles=profiles,
        year_cols=year_cols,
        mode="departure",
    )
    return {
        "Arrival Line Timings": arrival,
        "Departure Line Timings": departure,
    }


def _candidate_is_missed_transfer_retiming(row: dict[str, str], years: set[str]) -> bool:
    year = str(row.get("year", "")).strip()
    if year not in years:
        return False
    if _safe_float(row.get("net_pax_minutes_proxy"), 0.0) <= 0.0:
        return False
    if str(row.get("recommended_next_step", "")).strip() != "promising_for_targeted_reroute":
        return False
    if str(row.get("safety_flags", "")).strip() or str(row.get("feasibility_notes", "")).strip():
        return False
    if _safe_float(row.get("broken_existing_daily_pax_proxy"), 0.0) > 0.0:
        return False
    candidate_types = str(row.get("candidate_types", "")).lower()
    if "unlock_near_miss" not in candidate_types:
        return False
    actions = str(row.get("actions") or row.get("action", "")).lower()
    # These are the two movements matching the missed-transfer problem:
    # arrive earlier, or depart later.
    return "shift_arrival_line_earlier" in actions or "shift_departure_line_later" in actions


def _board_window_examples(
    raw_row: dict[str, str],
    *,
    existing_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    near_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    current_shift: dict[tuple[str, str, str, str, int], int],
    trial_shift: dict[tuple[str, str, str, str, int], int],
    min_transfer_lookup: dict[tuple[str, str], int],
    window_start: int,
    window_end: int,
    limit: int = 30,
) -> str:
    effects = _candidate_effects(raw_row, existing_index=existing_index, near_index=near_index)
    examples: list[str] = []
    for effect in sorted(
        effects.values(),
        key=lambda e: (
            str(e.get("transfer_station", "")),
            int(e.get("arrival_minute", 0)),
            int(e.get("departure_minute", 0)),
            str(e.get("arrival_line", "")),
            str(e.get("departure_line", "")),
        ),
    ):
        arr_min = int(effect.get("arrival_minute", 0))
        dep_min = int(effect.get("departure_minute", 0))
        if not (window_start <= arr_min < window_end or window_start <= dep_min < window_end):
            continue
        before = _effect_contribution(
            effect,
            event_shift_min=current_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        after = _effect_contribution(
            effect,
            event_shift_min=trial_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        diff = float(after["contribution"]) - float(before["contribution"])
        if abs(diff) < 1e-9 and after["new_wait_minutes"] == before["new_wait_minutes"]:
            continue
        examples.append(
            f"{effect['transfer_station']} {effect['arrival_line']}->{effect['departure_line']} "
            f"{effect['arrival_time']}->{effect['departure_time']} "
            f"wait {before['new_wait_minutes']}->{after['new_wait_minutes']} "
            f"min={after['minimum_transfer_time']} pax={_fmt(float(after['daily_pax']), 3)} "
            f"delta_pm={_fmt(diff, 3)} status={after['status']}"
        )
        if len(examples) >= limit:
            break
    return " | ".join(examples)


def _effect_summary_between_shifts(
    raw_row: dict[str, str],
    *,
    existing_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    near_index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    before_shift: dict[tuple[str, str, str, str, int], int],
    after_shift: dict[tuple[str, str, str, str, int], int],
    min_transfer_lookup: dict[tuple[str, str], int],
    window_start: int = 10 * 60,
    window_end: int = 12 * 60,
    example_limit: int = 30,
) -> dict[str, Any]:
    """Summarise this physical shift's touched effects after all accepted edits.

    This is intentionally post-hoc: marginal values are still recorded at the
    moment of acceptance, while these final-state fields show what the same
    touched board effects look like after later compatible edits have also
    been applied.
    """

    effects = _candidate_effects(raw_row, existing_index=existing_index, near_index=near_index)
    gross_gain = 0.0
    gross_loss = 0.0
    counts: dict[str, int] = {}
    pax_by_status: dict[str, float] = {}
    examples: list[str] = []
    for effect in effects.values():
        before = _effect_contribution(
            effect,
            event_shift_min=before_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        after = _effect_contribution(
            effect,
            event_shift_min=after_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        diff = float(after["contribution"]) - float(before["contribution"])
        if diff >= 0:
            gross_gain += diff
        else:
            gross_loss += -diff
        status = str(after["status"])
        counts[status] = counts.get(status, 0) + 1
        pax_by_status[status] = pax_by_status.get(status, 0.0) + float(after["daily_pax"])

        arr_min = int(effect.get("arrival_minute", 0))
        dep_min = int(effect.get("departure_minute", 0))
        if len(examples) < example_limit and (
            window_start <= arr_min < window_end or window_start <= dep_min < window_end
        ):
            if abs(diff) >= 1e-9 or after["new_wait_minutes"] != before["new_wait_minutes"]:
                examples.append(
                    f"{effect['transfer_station']} {effect['arrival_line']}->{effect['departure_line']} "
                    f"{effect['arrival_time']}->{effect['departure_time']} "
                    f"wait {before['new_wait_minutes']}->{after['new_wait_minutes']} "
                    f"min={after['minimum_transfer_time']} pax={_fmt(float(after['daily_pax']), 3)} "
                    f"delta_pm={_fmt(diff, 3)} status={after['status']}"
                )

    return {
        "final_net_pax_minutes_vs_v0_on_touched_effects": gross_gain - gross_loss,
        "final_gross_gain_pax_minutes_vs_v0_on_touched_effects": gross_gain,
        "final_gross_loss_pax_minutes_vs_v0_on_touched_effects": gross_loss,
        "final_status_counts_on_touched_effects": json.dumps(counts, ensure_ascii=False, sort_keys=True),
        "final_pax_by_status_on_touched_effects": json.dumps(
            {key: round(value, 6) for key, value in sorted(pax_by_status.items())},
            ensure_ascii=False,
            sort_keys=True,
        ),
        "final_station_board_10_12_effect_examples": " | ".join(examples),
    }


def _trial_shift_for_candidate(row: dict[str, str], current_shift: dict[tuple[str, str, str, str, int], int]) -> dict[tuple[str, str, str, str, int], int]:
    from build_transfer_shift_net_impact import _affected_keys

    trial = dict(current_shift)
    delta = int(row["delta_min"])
    for key in _affected_keys(row):
        trial[key] = int(trial.get(key, 0)) + delta
    return trial


def _audit_fields() -> list[str]:
    return [
        "candidate_status",
        "selection_order",
        "last_pass_seen",
        "priority_rank_static",
        "year",
        "actions",
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
        "broken_existing_daily_pax",
        "worsened_existing_daily_pax",
        "unlocked_near_miss_daily_pax",
        "partly_improved_near_miss_daily_pax",
        "effect_count",
        "status_counts",
        "pax_by_status",
        "Arrival Line Timings",
        "Departure Line Timings",
        "station_board_10_12_effect_examples",
        "reason",
        "physical_shift_id",
        "static_net_pax_minutes_proxy",
        "static_gross_gain_pax_minutes_proxy",
        "static_gross_loss_pax_minutes_proxy",
        "candidate_types",
        "transfer_stations",
        "affected_transfer_pairs",
        "net_affected_transfer_stations",
        "net_affected_transfer_pairs",
        "example_event_pairs",
    ]


def _audit_record(
    *,
    raw_row: dict[str, str],
    status: str,
    reason: str,
    score: dict[str, Any] | None,
    pass_number: int,
    priority_rank: int,
    selection_order: int | str = "",
    board_examples: str = "",
) -> dict[str, Any]:
    score = score or {}
    delta = int(raw_row.get("delta_min") or 0)
    return {
        "candidate_status": status,
        "selection_order": selection_order,
        "last_pass_seen": pass_number,
        "priority_rank_static": priority_rank,
        "year": raw_row.get("year", ""),
        "actions": raw_row.get("actions") or raw_row.get("action", ""),
        "line": raw_row.get("line", ""),
        "delta_min": raw_row.get("delta_min", ""),
        "section_from_station": raw_row.get("section_from_station", ""),
        "section_departure_time_mod120": raw_row.get("section_departure_time_mod120", ""),
        "section_departure_time_new_mod120": _shift_mod120(raw_row.get("section_departure_time_mod120", "00:00"), delta),
        "section_to_station": raw_row.get("section_to_station", ""),
        "section_arrival_time_mod120": raw_row.get("section_arrival_time_mod120", ""),
        "section_arrival_time_new_mod120": _shift_mod120(raw_row.get("section_arrival_time_mod120", "00:00"), delta),
        "marginal_net_pax_minutes": _fmt(_safe_float(score.get("marginal_net_pax_minutes"), 0.0)),
        "marginal_gross_gain_pax_minutes": _fmt(_safe_float(score.get("marginal_gross_gain_pax_minutes"), 0.0)),
        "marginal_gross_loss_pax_minutes": _fmt(_safe_float(score.get("marginal_gross_loss_pax_minutes"), 0.0)),
        "broken_existing_daily_pax": _fmt(_safe_float(score.get("broken_existing_daily_pax"), 0.0)),
        "worsened_existing_daily_pax": _fmt(_safe_float(score.get("worsened_existing_daily_pax"), 0.0)),
        "unlocked_near_miss_daily_pax": _fmt(_safe_float(score.get("unlocked_near_miss_daily_pax"), 0.0)),
        "partly_improved_near_miss_daily_pax": _fmt(_safe_float(score.get("partly_improved_near_miss_daily_pax"), 0.0)),
        "effect_count": score.get("effect_count", ""),
        "status_counts": json.dumps(score.get("status_counts", {}), ensure_ascii=False, sort_keys=True),
        "pax_by_status": json.dumps(score.get("pax_by_status", {}), ensure_ascii=False, sort_keys=True),
        "station_board_10_12_effect_examples": board_examples,
        "reason": reason,
        "physical_shift_id": raw_row.get("physical_shift_id", ""),
        "static_net_pax_minutes_proxy": raw_row.get("net_pax_minutes_proxy", ""),
        "static_gross_gain_pax_minutes_proxy": raw_row.get("gross_gain_pax_minutes_proxy", ""),
        "static_gross_loss_pax_minutes_proxy": raw_row.get("gross_loss_pax_minutes_proxy", ""),
        "candidate_types": raw_row.get("candidate_types", ""),
        "transfer_stations": raw_row.get("transfer_stations", ""),
        "affected_transfer_pairs": raw_row.get("affected_transfer_pairs", ""),
        "net_affected_transfer_stations": raw_row.get("net_affected_transfer_stations", ""),
        "net_affected_transfer_pairs": raw_row.get("net_affected_transfer_pairs", ""),
        "example_event_pairs": raw_row.get("example_event_pairs", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a stricter missed-transfer timetable proposal table. This focuses only on "
            "arrival-earlier/departure-later changes that unlock near-miss transfers, and scores "
            "their 10:00-11:59 station-board side-effects cumulatively."
        )
    )
    parser.add_argument("--timetable-csv", type=Path, default=DEFAULT_INPUT_TIMETABLE)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--transfer-2026", type=Path, default=DEFAULT_TRANSFER_2026)
    parser.add_argument("--transfer-2035", type=Path, default=DEFAULT_TRANSFER_2035)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT)
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
    parser.add_argument("--max-accepted", type=int, default=0, help="0 means no accepted-change cap.")
    parser.add_argument("--max-candidates-considered", type=int, default=0, help="0 means all missed-transfer candidates.")
    parser.add_argument("--min-marginal-net-pax-minutes", type=float, default=0.0)
    parser.add_argument("--min-near-miss-improved-daily-pax", type=float, default=0.0)
    parser.add_argument("--allow-broken-existing-daily-pax", type=float, default=0.0)
    parser.add_argument("--log-every-accepts", type=int, default=5)
    parser.add_argument("--reuse-source-tables", action="store_true")
    args = parser.parse_args()

    years = list(args.years)
    year_set = set(years)
    work_dir = Path(args.work_dir).expanduser().resolve()
    log_path = Path(args.log).expanduser().resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.reuse_source_tables:
        log_path.write_text("", encoding="utf-8")
    _log(log_path, f"Missed-transfer station-board proposal run start years={years}")

    transfer_paths = {
        "2026": Path(args.transfer_2026).expanduser().resolve(),
        "2035": Path(args.transfer_2035).expanduser().resolve(),
    }
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
    else:
        source_paths = _build_source_tables(
            args=args,
            log_path=log_path,
            work_dir=work_dir,
            transfer_paths=transfer_paths,
            years=years,
        )

    rows = _load_rows(Path(args.timetable_csv).expanduser().resolve())
    year_col_indices = _discover_year_column_indices(rows[0])
    segments = _parse_segments(rows)
    row_to_segment: dict[int, Any] = {}
    for segment in segments:
        for row_idx in segment.row_indices:
            row_to_segment[row_idx] = segment

    candidate_rows = [
        row
        for row in _read_csv(source_paths["physical_shifts"])
        if _candidate_is_missed_transfer_retiming(row, year_set)
    ]
    candidate_rows.sort(key=_candidate_sort_key)
    if int(args.max_candidates_considered) > 0:
        candidate_rows = candidate_rows[: int(args.max_candidates_considered)]
    _log(log_path, f"Loaded missed-transfer retiming candidates={len(candidate_rows)}")

    selected_transfer_paths = {year: transfer_paths[year] for year in years}
    existing_index, existing_counts = _load_existing_transfer_occurrences(selected_transfer_paths, threshold_pax=float(args.threshold_pax))
    near_index, near_counts = _load_near_miss_index(source_paths["near_misses"])
    near_index = {key: value for key, value in near_index.items() if key[0] in year_set}
    near_counts = {year: count for year, count in near_counts.items() if year in year_set}
    min_transfer_lookup = _load_min_transfer_from_ledger(source_paths["event_ledger"])
    _log(log_path, f"Loaded effect indexes existing={existing_counts} near_miss={near_counts}")

    current_shift: dict[tuple[str, str, str, str, int], int] = {}
    accepted_ids: set[str] = set()
    permanently_rejected: set[str] = set()
    accepted: list[dict[str, Any]] = []
    accepted_raw_rows: list[dict[str, str]] = []
    audit_rows: list[dict[str, Any]] = []
    rejected_latest: dict[str, dict[str, Any]] = {}
    cumulative_net = 0.0
    timetable_revision = 0
    profile_cache: dict[int, tuple[dict[str, list[LineProfile]], dict[str, list[str]]]] = {}

    for pass_number in range(1, int(args.max_passes) + 1):
        accepted_this_pass = 0
        _log(log_path, f"Pass {pass_number}/{args.max_passes} start accepted={len(accepted)} cumulative_net={_fmt(cumulative_net)}")
        for priority_rank, raw_row in enumerate(candidate_rows, start=1):
            physical_id = str(raw_row.get("physical_shift_id", ""))
            if physical_id in accepted_ids or physical_id in permanently_rejected:
                continue
            if int(args.max_accepted) > 0 and len(accepted) >= int(args.max_accepted):
                break
            timing_rows = rows
            timing_revision = timetable_revision

            score = _score_candidate_current_state(
                raw_row,
                existing_index=existing_index,
                near_index=near_index,
                event_shift_min=current_shift,
                min_transfer_lookup=min_transfer_lookup,
            )
            trial_shift = _trial_shift_for_candidate(raw_row, current_shift)
            board_examples = _board_window_examples(
                raw_row,
                existing_index=existing_index,
                near_index=near_index,
                current_shift=current_shift,
                trial_shift=trial_shift,
                min_transfer_lookup=min_transfer_lookup,
                window_start=10 * 60,
                window_end=12 * 60,
            )

            net = float(score["marginal_net_pax_minutes"])
            near_pax = float(score["unlocked_near_miss_daily_pax"]) + float(score["partly_improved_near_miss_daily_pax"])
            broken_pax = float(score["broken_existing_daily_pax"])
            if broken_pax > float(args.allow_broken_existing_daily_pax):
                status = "rejected_breaks_existing_high_flow_transfer"
                reason = f"broken existing daily pax {_fmt(broken_pax)} exceeds allowed {_fmt(float(args.allow_broken_existing_daily_pax))}"
                permanently_rejected.add(physical_id)
            elif near_pax <= float(args.min_near_miss_improved_daily_pax):
                status = "deferred_no_current_near_miss_gain"
                reason = f"current near-miss improved daily pax {_fmt(near_pax)} <= threshold {_fmt(float(args.min_near_miss_improved_daily_pax))}"
            elif net <= float(args.min_marginal_net_pax_minutes):
                status = "deferred_non_positive_current_marginal_net"
                reason = f"current marginal net pax-minutes {_fmt(net)} <= threshold {_fmt(float(args.min_marginal_net_pax_minutes))}"
            else:
                ok, trial_rows, apply_info = _try_apply_candidate(
                    raw_row=raw_row,
                    rank=priority_rank,
                    rows=rows,
                    segments=segments,
                    row_to_segment=row_to_segment,
                    year_col_indices=year_col_indices,
                )
                if ok:
                    rows = trial_rows
                    timetable_revision += 1
                    timing_rows = rows
                    timing_revision = timetable_revision
                    current_shift = trial_shift
                    accepted_ids.add(physical_id)
                    accepted_this_pass += 1
                    cumulative_net += net
                    accepted_record = _accepted_record(
                        raw_row=raw_row,
                        score=score,
                        selection_order=len(accepted) + 1,
                        pass_number=pass_number,
                        priority_rank=priority_rank,
                        cumulative_net=cumulative_net,
                        apply_info=apply_info,
                    )
                    accepted_record["station_board_10_12_effect_examples"] = board_examples
                    accepted_record.update(
                        _proposal_line_timings(
                            raw_row,
                            rows=timing_rows,
                            revision=timing_revision,
                            profile_cache=profile_cache,
                        )
                    )
                    accepted.append(accepted_record)
                    accepted_raw_rows.append(dict(raw_row))
                    status = "accepted"
                    reason = ""
                    if physical_id in rejected_latest:
                        del rejected_latest[physical_id]
                    if len(accepted) % int(args.log_every_accepts) == 0:
                        _log(
                            log_path,
                            f"accepted={len(accepted)} latest={raw_row.get('year')} {raw_row.get('line')} "
                            f"{raw_row.get('section_from_station')}->{raw_row.get('section_to_station')} "
                            f"delta={raw_row.get('delta_min')} net={_fmt(net)} near_pax={_fmt(near_pax)} cumulative={_fmt(cumulative_net)}",
                        )
                else:
                    status = str(apply_info.get("status", "skipped_timetable_incompatible"))
                    reason = str(apply_info.get("reason", ""))
                    permanently_rejected.add(physical_id)

            audit = _audit_record(
                raw_row=raw_row,
                status=status,
                reason=reason,
                score=score,
                pass_number=pass_number,
                priority_rank=priority_rank,
                selection_order=len(accepted) if status == "accepted" else "",
                board_examples=board_examples,
            )
            audit.update(
                _proposal_line_timings(
                    raw_row,
                    rows=timing_rows,
                    revision=timing_revision,
                    profile_cache=profile_cache,
                )
            )
            audit_rows.append(audit)
            if status != "accepted":
                rejected_latest[physical_id] = {
                    **_rejected_record(
                        raw_row=raw_row,
                        score=score,
                        last_pass=pass_number,
                        priority_rank=priority_rank,
                        status=status,
                        reason=reason,
                    ),
                    **_proposal_line_timings(
                        raw_row,
                        rows=timing_rows,
                        revision=timing_revision,
                        profile_cache=profile_cache,
                    ),
                    "station_board_10_12_effect_examples": board_examples,
                }

        _log(log_path, f"Pass {pass_number} complete accepted_this_pass={accepted_this_pass} total_accepted={len(accepted)}")
        if accepted_this_pass == 0:
            break
        if int(args.max_accepted) > 0 and len(accepted) >= int(args.max_accepted):
            break

    output_csv = Path(args.output_csv).expanduser().resolve()
    audit_csv = Path(args.audit_csv).expanduser().resolve()
    rejected_csv = Path(args.rejected_csv).expanduser().resolve()
    final_extra_fields = [
        "Arrival Line Timings",
        "Departure Line Timings",
        "station_board_10_12_effect_examples",
        "final_net_pax_minutes_vs_v0_on_touched_effects",
        "final_gross_gain_pax_minutes_vs_v0_on_touched_effects",
        "final_gross_loss_pax_minutes_vs_v0_on_touched_effects",
        "final_status_counts_on_touched_effects",
        "final_pax_by_status_on_touched_effects",
        "final_station_board_10_12_effect_examples",
    ]
    for record, raw_row in zip(accepted, accepted_raw_rows):
        record.update(
            _proposal_line_timings(
                raw_row,
                rows=rows,
                revision=timetable_revision,
                profile_cache=profile_cache,
            )
        )
        final_summary = _effect_summary_between_shifts(
            raw_row,
            existing_index=existing_index,
            near_index=near_index,
            before_shift={},
            after_shift=current_shift,
            min_transfer_lookup=min_transfer_lookup,
        )
        record.update(
            {
                "final_net_pax_minutes_vs_v0_on_touched_effects": _fmt(
                    float(final_summary["final_net_pax_minutes_vs_v0_on_touched_effects"])
                ),
                "final_gross_gain_pax_minutes_vs_v0_on_touched_effects": _fmt(
                    float(final_summary["final_gross_gain_pax_minutes_vs_v0_on_touched_effects"])
                ),
                "final_gross_loss_pax_minutes_vs_v0_on_touched_effects": _fmt(
                    float(final_summary["final_gross_loss_pax_minutes_vs_v0_on_touched_effects"])
                ),
                "final_status_counts_on_touched_effects": final_summary["final_status_counts_on_touched_effects"],
                "final_pax_by_status_on_touched_effects": final_summary["final_pax_by_status_on_touched_effects"],
                "final_station_board_10_12_effect_examples": final_summary["final_station_board_10_12_effect_examples"],
            }
        )
    _write_dict_csv(output_csv, accepted, [*_selection_fields(), *final_extra_fields])
    _write_dict_csv(audit_csv, audit_rows, _audit_fields())
    _write_dict_csv(
        rejected_csv,
        list(rejected_latest.values()),
        [
            *list(_rejected_record(raw_row={}, score={}, last_pass=0, priority_rank=0, status="", reason="").keys()),
            "Arrival Line Timings",
            "Departure Line Timings",
            "station_board_10_12_effect_examples",
        ],
    )
    _write_rows(Path(args.output_timetable).expanduser().resolve(), rows)

    summary = {
        "created_at": _now(),
        "years": years,
        "accepted_changes": len(accepted),
        "candidate_rows_audited": len(audit_rows),
        "cumulative_net_pax_minutes": round(cumulative_net, 6),
        "method": [
            "Only physical section shifts that touch a missed-transfer-unlocking endpoint are considered: arrival earlier at the downstream station and/or departure later at the upstream station. The opposite endpoint side-effects of the same physical section shift are scored too.",
            "Each candidate is rescored cumulatively after all previously accepted edits.",
            "Every accepted edit must improve near-miss daily pax, have positive current net pax-minutes, break no existing high-flow transfer, and pass timetable-structure validation.",
            "The 10:00-11:59 station-board effect examples list surrounding existing and near-miss transfer consequences for audit.",
            "Accepted rows include both their marginal-at-acceptance fields and final-state touched-effect fields after all later compatible edits.",
            "Arrival Line Timings and Departure Line Timings are generated from the current working timetable and refreshed to the final accepted timetable before writing the accepted proposal CSV.",
            "The proposal still uses transfer-table demand as fixed demand. Exact OD rerouting proof requires building the resulting nondominated-path matrix and comparing v1 against v0.",
        ],
        "outputs": {
            "accepted": str(output_csv),
            "audit": str(audit_csv),
            "rejected": str(rejected_csv),
            "timetable_preview": str(Path(args.output_timetable).expanduser().resolve()),
        },
    }
    Path(args.summary_json).expanduser().resolve().write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(log_path, f"Complete accepted={len(accepted)} cumulative_net={_fmt(cumulative_net)}")
    _log(log_path, f"Wrote {output_csv}")
    _log(log_path, f"Wrote {audit_csv}")
    _log(log_path, f"Wrote {rejected_csv}")


if __name__ == "__main__":
    main()
