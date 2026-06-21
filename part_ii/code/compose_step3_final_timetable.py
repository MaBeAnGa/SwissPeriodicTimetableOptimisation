#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from generate_step3_timetable_suggestionsResume import (  # noqa: E402
    DEFAULT_STEP2_TIMETABLE,
    _discover_year_column_indices,
    _load_rows,
    _parse_segments,
    _try_apply_candidate,
)
from build_optimised_v2_timetable import _write_rows  # noqa: E402



def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.expanduser().open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _candidate_id(row: dict[str, Any]) -> str:
    for key in ("physical_shift_id", "physicalShiftId", "source_physical_shift_id"):
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return "|".join(
        str(row.get(key, "")).strip()
        for key in ("year", "line", "delta_min", "section_from_station", "section_to_station", "from", "to")
    )


def _rank_value(row: dict[str, Any], fallback: int) -> tuple[int, int]:
    for key in ("step3_acceptance_rank", "step3_resume_seed_row", "step3_resume_new_rank", "selectionOrder"):
        value = str(row.get(key, "")).strip()
        if value:
            try:
                return (0, int(float(value)))
            except Exception:
                pass
    return (1, fallback)


def _load_accepted_rows(paths: list[Path], years: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        if not path or not path.expanduser().is_file():
            raise FileNotFoundError(f"Accepted Step 3 CSV not found: {path}")
        for idx, row in enumerate(_read_csv(path), start=1):
            year = str(row.get("year", "")).strip()
            if year not in years:
                continue
            status = str(row.get("status", "")).lower()
            if status and "accepted" not in status:
                continue
            cid = _candidate_id(row)
            if cid in seen:
                continue
            seen.add(cid)
            row["_source_csv"] = str(path.expanduser())
            row["_source_order"] = idx
            rows.append(row)
    rows.sort(key=lambda r: (str(r.get("year", "")), *_rank_value(r, int(r.get("_source_order", 0) or 0))))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose a final Step 3 timetable from the Step 2 timetable and accepted Step 3 CSV rows."
    )
    parser.add_argument("--step2-timetable-csv", type=Path, default=DEFAULT_STEP2_TIMETABLE)
    parser.add_argument("--accepted-csv", type=Path, action="append", required=True)
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    args = parser.parse_args()

    years = set(str(y) for y in args.years)
    accepted_rows = _load_accepted_rows(args.accepted_csv, years)
    if not accepted_rows:
        raise RuntimeError("No accepted Step 3 rows found for requested years.")

    rows = _load_rows(args.step2_timetable_csv.expanduser().resolve())
    segments = _parse_segments(rows)
    row_to_segment = {row_idx: segment for segment in segments for row_idx in segment.row_indices}
    year_col_indices = _discover_year_column_indices(rows[0])

    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    working_rows = rows
    for rank, row in enumerate(accepted_rows, start=1):
        ok, new_rows, info = _try_apply_candidate(
            raw_row=row,
            rank=rank,
            rows=working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        rec = {
            "rank": rank,
            "year": row.get("year"),
            "line": row.get("line"),
            "delta_min": row.get("delta_min"),
            "section_from_station": row.get("section_from_station"),
            "section_to_station": row.get("section_to_station"),
            "physical_shift_id": _candidate_id(row),
            "source_csv": row.get("_source_csv"),
            **{f"apply_{k}": v for k, v in info.items()},
        }
        if ok:
            working_rows = new_rows
            applied.append(rec)
        else:
            skipped.append(rec)
            raise RuntimeError(
                "Could not apply accepted Step 3 row "
                f"rank={rank} id={_candidate_id(row)} reason={info.get('reason', info.get('status', 'unknown'))}"
            )

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_rows(output_csv, working_rows)

    summary = {
        "created_at": _now(),
        "step2_timetable_csv": str(args.step2_timetable_csv.expanduser().resolve()),
        "accepted_csvs": [str(p.expanduser().resolve()) for p in args.accepted_csv],
        "years": sorted(years),
        "accepted_rows_loaded": len(accepted_rows),
        "applied_rows": len(applied),
        "skipped_rows": len(skipped),
        "output_csv": str(output_csv),
        "applied": applied,
        "skipped": skipped,
    }
    summary_json = args.summary_json.expanduser().resolve()
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "applied_rows": len(applied)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
