#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

from build_optimised_v2_timetable import (
    DEFAULT_CANDIDATES,
    DEFAULT_INPUT_TIMETABLE,
    YEAR_COLUMNS,
    Segment,
    _as_float,
    _candidate_is_strict_safe,
    _discover_year_column_indices,
    _load_rows,
    _matching_section_edits,
    _parse_segments,
    _row_to_candidate,
    _write_rows,
)
from prepare_transfer_boards_and_baseline import (
    MAX_THROUGH_DWELL_MIN,
    ServiceSegment,
    active_rows_for_segment,
    instantiate_trip_events,
    valid_ab_an_sequence,
)


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_TIMETABLE = BASE_DIR / "FullTimetableHistory2026_2035_Optimised_v4.csv"
DEFAULT_MANIFEST = BASE_DIR / "transfer_optimization_tables" / "Optimised_v4_compatible_timetable_manifest.csv"
DEFAULT_SUMMARY = BASE_DIR / "transfer_optimization_tables" / "Optimised_v4_compatible_timetable_summary.json"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _load_candidate_rows(path: Path, years: set[str]) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = [row for row in csv.DictReader(f) if _candidate_is_strict_safe(row, years)]
    rows.sort(
        key=lambda r: (
            -_as_float(r.get("net_pax_minutes_proxy"), 0.0),
            str(r.get("physical_shift_id", "")),
        )
    )
    return rows


def _service_segment_from_rows(rows: list[list[str]], segment: Segment) -> ServiceSegment:
    header = rows[0]
    segment_rows: list[dict[str, str]] = []
    for row_idx in segment.row_indices:
        row = rows[row_idx]
        item = {
            "station": str(row[0]).strip(),
            "kind": str(row[1]).strip(),
            "rownum": str(row_idx + 1),
        }
        for col_idx, col_name in enumerate(header[2:], start=2):
            item[str(col_name)] = str(row[col_idx]).strip()
        segment_rows.append(item)
    return ServiceSegment(
        service_name=segment.service_name,
        segment_index=segment.segment_index,
        rows=tuple(segment_rows),
    )


def _trip_signature_for_compat(trip: list[tuple[str, str, int]]) -> tuple[Any, ...]:
    sequence = tuple((station, kind) for station, kind, _minute in trip)
    rolling: list[tuple[str, str, int]] = []
    for left, right in zip(trip, trip[1:]):
        left_station, left_kind, left_minute = left
        right_station, right_kind, right_minute = right
        if left_kind == "ab" and right_kind == "an":
            rolling.append((left_station, right_station, right_minute - left_minute))
    return (sequence, tuple(rolling))


def _validate_segment_column_compatibility(
    *,
    before_rows: list[list[str]],
    after_rows: list[list[str]],
    segment: Segment,
    col_idx: int,
) -> tuple[bool, str]:
    col_name = str(before_rows[0][col_idx])
    before_segment = _service_segment_from_rows(before_rows, segment)
    after_segment = _service_segment_from_rows(after_rows, segment)

    before_active = active_rows_for_segment(before_segment, col_name)
    after_active = active_rows_for_segment(after_segment, col_name)

    if bool(before_active) != bool(after_active):
        return False, f"{segment.service_name} col={col_name}: active-row presence changed"
    if not before_active:
        return True, ""
    if not valid_ab_an_sequence(after_active):
        return False, f"{segment.service_name} col={col_name}: invalid ab/an sequence after edits"

    before_trips = instantiate_trip_events(before_active)
    after_trips = instantiate_trip_events(after_active)
    if len(after_trips) != len(before_trips):
        return (
            False,
            f"{segment.service_name} col={col_name}: trip-instance count changed "
            f"{len(before_trips)}->{len(after_trips)}",
        )

    before_signatures = sorted(_trip_signature_for_compat(trip) for trip in before_trips)
    after_signatures = sorted(_trip_signature_for_compat(trip) for trip in after_trips)
    if before_signatures != after_signatures:
        return (
            False,
            f"{segment.service_name} col={col_name}: run structure/rolling/dwell signature changed",
        )

    for trip in after_trips:
        for left, right in zip(trip, trip[1:]):
            if left[1] == "an" and right[1] == "ab" and left[0] == right[0]:
                dwell = right[2] - left[2]
                if dwell < 0 or dwell > MAX_THROUGH_DWELL_MIN:
                    return (
                        False,
                        f"{segment.service_name} col={col_name}: unreasonable dwell at {left[0]} = {dwell} min",
                    )
    return True, ""


def _copy_rows(rows: list[list[str]]) -> list[list[str]]:
    return [list(row) for row in rows]


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "priority_rank",
        "selection_order",
        "year",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_departure_time_new_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "section_arrival_time_new_mod120",
        "net_pax_minutes_proxy",
        "gross_loss_pax_minutes_proxy",
        "broken_existing_daily_pax_proxy",
        "min_can_shift_earlier_min",
        "min_can_shift_later_min",
        "status",
        "reason",
        "matched_cells",
        "changed_cells",
        "change_description",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field, "") for field in fields})


def _candidate_base_record(candidate: dict[str, str]) -> dict[str, Any]:
    return {
        key: candidate.get(key, "")
        for key in (
            "priority_rank",
            "year",
            "line",
            "delta_min",
            "section_from_station",
            "section_departure_time_mod120",
            "section_departure_time_new_mod120",
            "section_to_station",
            "section_arrival_time_mod120",
            "section_arrival_time_new_mod120",
            "net_pax_minutes_proxy",
            "gross_loss_pax_minutes_proxy",
            "broken_existing_daily_pax_proxy",
            "min_can_shift_earlier_min",
            "min_can_shift_later_min",
            "change_description",
        )
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a cumulative-compatible optimised timetable. Candidates are scanned by descending "
            "pax-minute proxy, and every accepted edit is applied before testing the next candidate."
        )
    )
    parser.add_argument("--input-timetable", type=Path, default=DEFAULT_INPUT_TIMETABLE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-timetable", type=Path, default=DEFAULT_OUTPUT_TIMETABLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--max-targets", type=int, default=50, help="Stop after this many compatible accepted candidates. 0 means all.")
    parser.add_argument("--years", default="2026,2035", help="Comma-separated years to alter.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_timetable = Path(args.input_timetable).expanduser().resolve()
    candidates_path = Path(args.candidates).expanduser().resolve()
    output_timetable = Path(args.output_timetable).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    summary_path = Path(args.summary).expanduser().resolve()
    years = {y.strip() for y in str(args.years).split(",") if y.strip()}
    invalid_years = years.difference(YEAR_COLUMNS)
    if invalid_years:
        raise ValueError(f"Unsupported years: {sorted(invalid_years)}")

    rows = _load_rows(input_timetable)
    year_col_indices = _discover_year_column_indices(rows[0])
    segments = _parse_segments(rows)
    row_to_segment: dict[int, Segment] = {}
    for segment in segments:
        for row_idx in segment.row_indices:
            row_to_segment[row_idx] = segment

    strict_rows = _load_candidate_rows(candidates_path, years)
    max_targets = int(args.max_targets)
    _log(
        "Optimised_v4 compatible build start "
        f"strict_safe_candidates={len(strict_rows)} max_targets={max_targets} "
        f"years={sorted(years)} input={input_timetable.name}"
    )

    records: list[dict[str, Any]] = []
    applied_candidates = 0
    skipped_unmatched = 0
    skipped_validation = 0
    skipped_error = 0
    considered = 0

    for priority_rank, raw_row in enumerate(strict_rows, start=1):
        if max_targets > 0 and applied_candidates >= max_targets:
            break
        considered += 1
        candidate = _row_to_candidate(raw_row, rank=priority_rank)
        base_record = _candidate_base_record(candidate)
        try:
            edits = _matching_section_edits(
                rows=rows,
                segments=segments,
                year_column_indices=year_col_indices,
                candidate=candidate,
            )
        except Exception as exc:
            records.append(
                {
                    **base_record,
                    "selection_order": "",
                    "status": "skipped_error",
                    "reason": str(exc),
                    "matched_cells": 0,
                    "changed_cells": 0,
                }
            )
            skipped_error += 1
            continue

        unique_edits: list[tuple[int, int, str, str, str]] = []
        seen_edit_keys: set[tuple[int, int, str]] = set()
        for edit in edits:
            key = (edit[0], edit[1], edit[3])
            if key in seen_edit_keys:
                continue
            seen_edit_keys.add(key)
            unique_edits.append(edit)

        if not unique_edits:
            records.append(
                {
                    **base_record,
                    "selection_order": "",
                    "status": "skipped_unmatched_after_prior_edits",
                    "reason": "no exact current timetable token match after previously accepted edits",
                    "matched_cells": 0,
                    "changed_cells": 0,
                }
            )
            skipped_unmatched += 1
            continue

        trial_rows = _copy_rows(rows)
        changed = 0
        touched_segment_cols: set[tuple[int, int]] = set()
        for row_idx, col_idx, old_value, new_value, _role in unique_edits:
            if trial_rows[row_idx][col_idx] != old_value:
                records.append(
                    {
                        **base_record,
                        "selection_order": "",
                        "status": "skipped_validation",
                        "reason": (
                            f"current token changed while preparing trial: row={row_idx + 1} "
                            f"col={trial_rows[0][col_idx]!r} expected={old_value!r} "
                            f"found={trial_rows[row_idx][col_idx]!r}"
                        ),
                        "matched_cells": len(unique_edits),
                        "changed_cells": 0,
                    }
                )
                skipped_validation += 1
                break
            trial_rows[row_idx][col_idx] = new_value
            if old_value != new_value:
                changed += 1
            segment = row_to_segment.get(row_idx)
            if segment is not None:
                touched_segment_cols.add((id(segment), col_idx))
        else:
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
                records.append(
                    {
                        **base_record,
                        "selection_order": "",
                        "status": "skipped_cumulative_incompatible",
                        "reason": " | ".join(validation_errors[:5]),
                        "matched_cells": len(unique_edits),
                        "changed_cells": 0,
                    }
                )
                skipped_validation += 1
                continue

            rows = trial_rows
            applied_candidates += 1
            records.append(
                {
                    **base_record,
                    "selection_order": applied_candidates,
                    "status": "applied",
                    "reason": "",
                    "matched_cells": len(unique_edits),
                    "changed_cells": changed,
                }
            )

    _write_manifest(manifest_path, records)
    summary = {
        "created_at": _ts(),
        "input_timetable": str(input_timetable),
        "candidate_table": str(candidates_path),
        "output_timetable": str(output_timetable),
        "manifest": str(manifest_path),
        "max_targets": max_targets,
        "years": sorted(years),
        "segments_scanned": len(segments),
        "strict_safe_candidates": len(strict_rows),
        "selected_rows": applied_candidates,
        "selected_by_year": {
            year: sum(1 for record in records if record.get("status") == "applied" and record.get("year") == year)
            for year in sorted(years)
        },
        "candidates_considered_until_stop": considered,
        "candidate_rows_applied": applied_candidates,
        "candidate_rows_skipped_unmatched_after_prior_edits": skipped_unmatched,
        "candidate_rows_skipped_cumulative_validation": skipped_validation,
        "candidate_rows_skipped_error": skipped_error,
        "dry_run": bool(args.dry_run),
        "selection_rule": (
            "Greedy cumulative compatibility: strict-safe candidates are sorted by descending "
            "net pax-minute proxy. Each accepted candidate is immediately applied to the working "
            "timetable. Later candidates must match that current timetable and must preserve the "
            "affected line segment's trip-instance count, station/kind run structure, rolling "
            "durations, and reasonable through-dwell bounds."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        _log(f"Dry run complete. Compatible applied={applied_candidates}; manifest={manifest_path}")
    else:
        _write_rows(output_timetable, rows)
        _log(f"Wrote {output_timetable}")
    _log(f"Wrote manifest={manifest_path}")
    _log(f"Wrote summary={summary_path}")


if __name__ == "__main__":
    main()
