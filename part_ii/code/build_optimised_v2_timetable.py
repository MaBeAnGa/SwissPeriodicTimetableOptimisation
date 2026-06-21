#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_TIMETABLE = BASE_DIR / "FullTimetableHistory2026_2035.csv"
DEFAULT_CANDIDATES = BASE_DIR / "transfer_optimization_tables" / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv"
DEFAULT_OUTPUT_TIMETABLE = BASE_DIR / "FullTimetableHistory2026_2035_Optimised_v2.csv"
DEFAULT_MANIFEST = BASE_DIR / "transfer_optimization_tables" / "Optimised_v2_timetable_manifest.csv"
DEFAULT_SUMMARY = BASE_DIR / "transfer_optimization_tables" / "Optimised_v2_timetable_summary.json"

DASH_VALUES = {"", "-", "—", "–", "―"}
TOKEN_RE = re.compile(r"^\s*(\d{1,2})\s*([inIN]?)\s*([Pp]?)\s*([Qq]?)\s*$")
DAY_START = 5 * 60
TWO_HOUR_END = 7 * 60

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
class Segment:
    service_name: str
    segment_index: int
    row_indices: tuple[int, ...]


@dataclass(frozen=True)
class ActiveRow:
    row_index: int
    station: str
    kind: str
    token: ParsedToken


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        text = str(value).strip()
        if not text:
            return default
        return int(float(text))
    except Exception:
        return default


def _parse_mod120(value: str) -> int:
    text = str(value).strip()
    hour, minute = text.split(":", 1)
    out = int(hour) * 60 + int(minute)
    if not (0 <= out < 120):
        raise ValueError(f"Expected HH:MM within 00:00..01:59, got {value!r}")
    return out


def _format_mod120(value: int) -> str:
    value %= 120
    return f"{value // 60:02d}:{value % 60:02d}"


def _parse_token(cell: str) -> ParsedToken | None:
    text = str(cell).strip()
    if text in DASH_VALUES:
        return None
    match = TOKEN_RE.match(text)
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
        raw=text,
    )


def _token_to_text(token: ParsedToken) -> str:
    suffix = ""
    if token.mode == "even":
        suffix = "n"
    elif token.mode == "odd":
        suffix = "i"
    if token.pflag:
        suffix += "P"
    if token.qflag:
        suffix += "Q"
    return f"{token.minute}{suffix}"


def _shift_token(token: ParsedToken, delta_min: int) -> ParsedToken:
    raw_minute = int(token.minute) + int(delta_min)
    hour_delta = raw_minute // 60
    new_minute = raw_minute % 60
    mode = token.mode
    if mode in {"even", "odd"} and hour_delta % 2:
        mode = "odd" if mode == "even" else "even"
    return ParsedToken(
        minute=int(new_minute),
        mode=mode,
        pflag=bool(token.pflag),
        qflag=bool(token.qflag),
        raw="",
    )


def _allowed_hours_for_token(token: ParsedToken) -> list[int]:
    if token.mode == "hourly":
        return [5, 6]
    if token.mode == "even":
        return [6]
    if token.mode == "odd":
        return [5]
    return []


def _mods_for_token(token: ParsedToken, kind: str) -> set[int]:
    mods: set[int] = set()
    for hour in _allowed_hours_for_token(token):
        t = hour * 60 + token.minute
        if kind == "ab" and not (DAY_START <= t < TWO_HOUR_END):
            continue
        if kind == "an" and not (DAY_START <= t <= TWO_HOUR_END):
            continue
        mods.add((t - DAY_START) % 120)
    return mods


def _load_rows(path: Path) -> list[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows or len(rows[0]) < 3:
        raise ValueError(f"{path} does not look like FullTimetableHistory2026_2035.csv")
    width = len(rows[0])
    for row in rows:
        if len(row) < width:
            row.extend([""] * (width - len(row)))
        elif len(row) > width:
            raise ValueError(f"{path}: row has {len(row)} cells but header has {width}: {row[:3]}")
    return rows


def _discover_year_column_indices(header_row: list[str]) -> dict[str, list[int]]:
    by_year: dict[str, list[int]] = {}
    for year, labels in YEAR_COLUMNS.items():
        indices: list[int] = []
        for label in labels:
            try:
                indices.append(header_row.index(label))
            except ValueError:
                raise ValueError(f"Timetable header is missing required column {label!r}") from None
        by_year[year] = indices
    return by_year


def _parse_segments(rows: list[list[str]]) -> list[Segment]:
    segments: list[Segment] = []
    current_service: str | None = None
    current_segment_rows: list[int] = []
    segment_index = 1

    def flush_segment() -> None:
        nonlocal current_segment_rows, segment_index
        if current_service and current_segment_rows:
            segments.append(Segment(current_service, segment_index, tuple(current_segment_rows)))
            segment_index += 1
            current_segment_rows = []

    for idx, row in enumerate(rows):
        station = str(row[0]).strip()
        kind = str(row[1]).strip().lower()
        if kind == "an/ab":
            flush_segment()
            current_service = station
            current_segment_rows = []
            segment_index = 1
            continue
        if current_service is None:
            continue
        if station.lower() == "split" or kind == "split":
            flush_segment()
            continue
        current_segment_rows.append(idx)
    flush_segment()
    return segments


def _active_rows(rows: list[list[str]], segment: Segment, col_idx: int) -> list[ActiveRow]:
    out: list[ActiveRow] = []
    for row_idx in segment.row_indices:
        row = rows[row_idx]
        token = _parse_token(row[col_idx])
        if token is None:
            continue
        kind = str(row[1]).strip().lower()
        if kind not in {"an", "ab"}:
            continue
        out.append(
            ActiveRow(
                row_index=row_idx,
                station=str(row[0]).strip(),
                kind=kind,
                token=token,
            )
        )
    return out


def _matching_section_edits(
    *,
    rows: list[list[str]],
    segments: list[Segment],
    year_column_indices: dict[str, list[int]],
    candidate: dict[str, str],
) -> list[tuple[int, int, str, str, str]]:
    year = str(candidate["year"])
    line = str(candidate["line"]).strip()
    delta = int(candidate["delta_min"])
    from_station = str(candidate["section_from_station"]).strip()
    to_station = str(candidate["section_to_station"]).strip()
    old_dep = _parse_mod120(candidate["section_departure_time_mod120"])
    old_arr = _parse_mod120(candidate["section_arrival_time_mod120"])
    new_dep = _parse_mod120(candidate["section_departure_time_new_mod120"])
    new_arr = _parse_mod120(candidate["section_arrival_time_new_mod120"])

    edits: list[tuple[int, int, str, str, str]] = []
    for segment in segments:
        if segment.service_name != line:
            continue
        for col_idx in year_column_indices[year]:
            active = _active_rows(rows, segment, col_idx)
            for left, right in zip(active, active[1:]):
                if not (
                    left.station == from_station
                    and left.kind == "ab"
                    and right.station == to_station
                    and right.kind == "an"
                ):
                    continue
                if old_dep not in _mods_for_token(left.token, left.kind):
                    continue
                if old_arr not in _mods_for_token(right.token, right.kind):
                    continue
                dep_new_token = _shift_token(left.token, delta)
                arr_new_token = _shift_token(right.token, delta)
                if new_dep not in _mods_for_token(dep_new_token, left.kind):
                    raise ValueError(
                        f"Shift validation failed for {line} {from_station} dep: "
                        f"{left.token.raw}->{_token_to_text(dep_new_token)} does not yield {candidate['section_departure_time_new_mod120']}"
                    )
                if new_arr not in _mods_for_token(arr_new_token, right.kind):
                    raise ValueError(
                        f"Shift validation failed for {line} {to_station} arr: "
                        f"{right.token.raw}->{_token_to_text(arr_new_token)} does not yield {candidate['section_arrival_time_new_mod120']}"
                    )
                edits.append((left.row_index, col_idx, rows[left.row_index][col_idx], _token_to_text(dep_new_token), "departure"))
                edits.append((right.row_index, col_idx, rows[right.row_index][col_idx], _token_to_text(arr_new_token), "arrival"))
    return edits


def _candidate_is_strict_safe(row: dict[str, str], years: set[str]) -> bool:
    year = str(row.get("year", "")).strip()
    if year not in years:
        return False
    if str(row.get("recommended_next_step", "")).strip() != "promising_for_targeted_reroute":
        return False
    delta = _as_int(row.get("delta_min"), 0)
    net = _as_float(row.get("net_pax_minutes_proxy"), 0.0)
    gross_loss = _as_float(row.get("gross_loss_pax_minutes_proxy"), 0.0)
    broken_events = _as_int(row.get("broken_existing_transfer_events"), 0)
    broken_pax = _as_float(row.get("broken_existing_daily_pax_proxy"), 0.0)
    safety_flags = str(row.get("safety_flags", "")).strip()
    feasibility_notes = str(row.get("feasibility_notes", "")).strip()
    if net <= 0.0 or gross_loss != 0.0 or broken_events != 0 or broken_pax != 0.0:
        return False
    if safety_flags or feasibility_notes:
        return False
    earlier = _as_float(row.get("min_can_shift_earlier_min"), 0.0)
    later = _as_float(row.get("min_can_shift_later_min"), 0.0)
    if delta < 0:
        return earlier >= abs(delta)
    if delta > 0:
        return later >= delta
    return False


def _row_to_candidate(row: dict[str, str], rank: int) -> dict[str, str]:
    delta = _as_int(row.get("delta_min"), 0)
    dep_old = _parse_mod120(row["section_departure_time_mod120"])
    arr_old = _parse_mod120(row["section_arrival_time_mod120"])
    dep_new = _format_mod120(dep_old + delta)
    arr_new = _format_mod120(arr_old + delta)
    out = dict(row)
    out["priority_rank"] = str(rank)
    out["section_departure_time_new_mod120"] = dep_new
    out["section_arrival_time_new_mod120"] = arr_new
    out["change_description"] = (
        f"Move {row['line']} section {row['section_from_station']} -> {row['section_to_station']} "
        f"{abs(delta)} min {'earlier' if delta < 0 else 'later'}: "
        f"depart {row['section_departure_time_mod120']}->{dep_new}, "
        f"arrive {row['section_arrival_time_mod120']}->{arr_new}."
    )
    return out


def _load_candidates(path: Path, *, max_targets: int, years: set[str]) -> tuple[list[dict[str, str]], dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        all_rows = list(csv.DictReader(f))

    strict_rows = [row for row in all_rows if _candidate_is_strict_safe(row, years)]
    strict_rows.sort(
        key=lambda r: (
            -_as_float(r.get("net_pax_minutes_proxy"), 0.0),
            str(r.get("physical_shift_id", "")),
        )
    )
    selected_rows = strict_rows if max_targets <= 0 else strict_rows[:max_targets]
    candidates = [_row_to_candidate(row, rank=i + 1) for i, row in enumerate(selected_rows)]
    summary = {
        "source_rows_total": len(all_rows),
        "strict_safe_promising_rows": len(strict_rows),
        "strict_safe_promising_by_year": {
            year: sum(1 for row in strict_rows if str(row.get("year", "")).strip() == year)
            for year in sorted(years)
        },
        "selected_rows": len(candidates),
        "selected_by_year": {
            year: sum(1 for row in candidates if str(row.get("year", "")).strip() == year)
            for year in sorted(years)
        },
    }
    return candidates, summary


def _write_rows(path: Path, rows: list[list[str]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    tmp.replace(path)


def _write_manifest(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Optimised_v2 timetable using strict-safe net-positive transfer retiming candidates."
    )
    parser.add_argument("--input-timetable", type=Path, default=DEFAULT_INPUT_TIMETABLE)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-timetable", type=Path, default=DEFAULT_OUTPUT_TIMETABLE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument(
        "--max-targets",
        type=int,
        default=50,
        help="Apply at most this many strict-safe candidates by net pax-minute proxy. Use 0 for all strict-safe candidates.",
    )
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
    candidates, candidate_summary = _load_candidates(
        candidates_path,
        max_targets=int(args.max_targets),
        years=years,
    )

    _log(
        "Optimised_v2 timetable build start "
        f"strict_safe_promising={candidate_summary['strict_safe_promising_rows']} "
        f"selected={len(candidates)} max_targets={args.max_targets} "
        f"years={sorted(years)} input={input_timetable.name}"
    )

    planned: dict[tuple[int, int], str] = {}
    records: list[dict[str, Any]] = []
    applied_candidates = 0
    skipped_unmatched = 0
    skipped_conflict = 0

    for candidate in candidates:
        base_record = {
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
        try:
            edits = _matching_section_edits(
                rows=rows,
                segments=segments,
                year_column_indices=year_col_indices,
                candidate=candidate,
            )
        except Exception as exc:
            records.append({**base_record, "status": "skipped_error", "reason": str(exc), "matched_cells": 0, "changed_cells": 0})
            skipped_conflict += 1
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
            records.append({**base_record, "status": "skipped_unmatched", "reason": "no exact timetable token match", "matched_cells": 0, "changed_cells": 0})
            skipped_unmatched += 1
            continue

        conflicts: list[str] = []
        for row_idx, col_idx, old_value, new_value, role in unique_edits:
            existing = planned.get((row_idx, col_idx))
            if existing is not None and existing != new_value:
                conflicts.append(
                    f"row={row_idx + 1} col={rows[0][col_idx]!r} {role}: existing_plan={existing!r}, new_plan={new_value!r}"
                )
        if conflicts:
            records.append(
                {
                    **base_record,
                    "status": "skipped_conflict",
                    "reason": " | ".join(conflicts[:5]),
                    "matched_cells": len(unique_edits),
                    "changed_cells": 0,
                }
            )
            skipped_conflict += 1
            continue

        changed = 0
        for row_idx, col_idx, old_value, new_value, role in unique_edits:
            if old_value != new_value:
                planned[(row_idx, col_idx)] = new_value
                changed += 1
            elif (row_idx, col_idx) not in planned:
                planned[(row_idx, col_idx)] = new_value
        records.append(
            {
                **base_record,
                "status": "applied",
                "reason": "",
                "matched_cells": len(unique_edits),
                "changed_cells": changed,
            }
        )
        applied_candidates += 1

    for (row_idx, col_idx), new_value in planned.items():
        rows[row_idx][col_idx] = new_value

    _write_manifest(manifest_path, records)
    summary = {
        "created_at": _ts(),
        "input_timetable": str(input_timetable),
        "candidate_table": str(candidates_path),
        "output_timetable": str(output_timetable),
        "manifest": str(manifest_path),
        "max_targets": int(args.max_targets),
        "years": sorted(years),
        "segments_scanned": len(segments),
        **candidate_summary,
        "candidate_rows_applied": applied_candidates,
        "candidate_rows_skipped_unmatched": skipped_unmatched,
        "candidate_rows_skipped_conflict_or_error": skipped_conflict,
        "physical_cells_planned": len(planned),
        "dry_run": bool(args.dry_run),
        "selection_rule": (
            "Strict-safe candidates require: recommended_next_step=promising_for_targeted_reroute, "
            "positive net pax-minute proxy, zero gross-loss proxy, zero broken existing transfers, "
            "empty safety flags, empty feasibility notes, and neighbouring-section slack sufficient "
            "for the requested delta."
        ),
        "note": (
            "Optimised_v2 edits only exact matching timetable tokens for the selected year, line, "
            "section, old departure mod120, and old arrival mod120. Rolling time is preserved by "
            "moving the section departure and following section arrival by the same minute delta. "
            "Conflicting edits are skipped and recorded in the manifest."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if args.dry_run:
        _log(f"Dry run complete. Would write {len(planned)} physical cells; manifest={manifest_path}")
    else:
        _write_rows(output_timetable, rows)
        _log(f"Wrote {output_timetable}")
    _log(f"Wrote manifest={manifest_path}")
    _log(f"Wrote summary={summary_path}")


if __name__ == "__main__":
    main()
