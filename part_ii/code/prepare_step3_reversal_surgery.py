#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from build_optimised_v2_timetable import (  # noqa: E402
    _discover_year_column_indices,
    _format_mod120,
    _load_rows,
    _parse_mod120,
    _parse_segments,
    _write_rows,
)
from generate_proposed_timetable_changes import _try_apply_candidate  # noqa: E402


DEFAULT_RUN_ROOT = Path.home() / "od_fast_state" / "step3_local_outputs" / "STEP3_BUILD_LOCAL_FINAL"
DEFAULT_STEP3_TIMETABLE = DEFAULT_RUN_ROOT / "timetable" / "FullTimetableHistory2026_2035_STEP3_FINAL.csv"
DEFAULT_OUTPUT_DIR = DEFAULT_RUN_ROOT / "reversal_surgery_preparation"
DEFAULT_STEP3_ROOT = DEFAULT_RUN_ROOT / "row_states" / "nondominated_cells_state_STEP3_FINAL"
DEFAULT_V0_ROOT = Path.home() / "od_fast_state" / "nondominated_cells_state_v0_restored_from_pathalloc_csv"
DEFAULT_STEP2_ROOT = Path.home() / "od_fast_state" / "nondominated_cells_state_ACTUALLY_BETTER_OPTIMIZED"

REVERSAL_CANDIDATES: dict[str, dict[str, str]] = {
    "ic5_neuchatel_biel": {
        "source_edit": "2035|physical|IC 5|-2|Neuchâtel|01:27|Biel/Bienne|01:45",
        "year": "2035",
        "line": "IC 5",
        "section_from_station": "Neuchâtel",
        "section_to_station": "Biel/Bienne",
        "original_departure_time_mod120": "01:27",
        "original_arrival_time_mod120": "01:45",
        "original_delta_min": "-2",
        "reason": "Reversal candidate for the inherited IC 5 edit associated with Neuchâtel/Biel attribution losses.",
    },
    "ic82_zofingen_bern": {
        "source_edit": "2035|physical|IC 82|-1|Zofingen|01:32|Bern|00:00",
        "year": "2035",
        "line": "IC 82",
        "section_from_station": "Zofingen",
        "section_to_station": "Bern",
        "original_departure_time_mod120": "01:32",
        "original_arrival_time_mod120": "00:00",
        "original_delta_min": "-1",
        "reason": "Reversal candidate for the inherited IC 82 edit with negative row-level attribution.",
    },
}


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _candidate_to_inverse_row(key: str, source: dict[str, str], rank: int) -> dict[str, str]:
    original_delta = _safe_int(source["original_delta_min"])
    inverse_delta = -original_delta
    current_dep = _format_mod120(_parse_mod120(source["original_departure_time_mod120"]) + original_delta)
    current_arr = _format_mod120(_parse_mod120(source["original_arrival_time_mod120"]) + original_delta)
    return {
        "selection_order": str(rank),
        "year": source["year"],
        "action": "reverse_inherited_edit",
        "line": source["line"],
        "delta_min": str(inverse_delta),
        "section_from_station": source["section_from_station"],
        "section_departure_time_mod120": current_dep,
        "section_departure_time_new_mod120": source["original_departure_time_mod120"],
        "section_to_station": source["section_to_station"],
        "section_arrival_time_mod120": current_arr,
        "section_arrival_time_new_mod120": source["original_arrival_time_mod120"],
        "physical_shift_id": f"reversal|{key}|of|{source['source_edit']}",
        "source_physical_shift_id": source["source_edit"],
        "reversal_key": key,
        "reversal_reason": source["reason"],
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_paths(cell_raw: Any) -> list[dict[str, Any]]:
    try:
        cell = json.loads(str(cell_raw))
    except Exception:
        return []
    paths = cell.get("paths")
    return paths if isinstance(paths, list) else []


def _leg_matches(leg: dict[str, Any], target: dict[str, Any]) -> bool:
    if str(leg.get("service_name", "")).strip() != target["line"]:
        return False
    if str(leg.get("from_station", "")).strip() != target["from"]:
        return False
    if str(leg.get("to_station", "")).strip() != target["to"]:
        return False
    try:
        dep_mod = int(float(leg.get("dep_min"))) % 120
        arr_mod = int(float(leg.get("arr_min"))) % 120
    except Exception:
        return False
    return dep_mod in target["dep_mods"] and arr_mod in target["arr_mods"]


def _path_matches(path: dict[str, Any], targets: list[dict[str, Any]]) -> list[str]:
    matched: list[str] = []
    legs = path.get("legs")
    if not isinstance(legs, list):
        return matched
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        for target in targets:
            if _leg_matches(leg, target):
                matched.append(str(target["key"]))
    return matched


def _scan_root_for_impacted_origins(
    *,
    root: Path,
    year: str,
    targets: list[dict[str, Any]],
    source_label: str,
) -> dict[int, dict[str, Any]]:
    rows_dir = root / year / "rows"
    if not rows_dir.is_dir():
        return {}
    out: dict[int, dict[str, Any]] = {}
    row_paths = sorted(rows_dir.glob("row_*.json"))
    for pos, row_path in enumerate(row_paths, start=1):
        raw_text = row_path.read_text(encoding="utf-8")
        # Most origin rows never use the affected long-distance sections. Avoid
        # expensive JSON decoding unless the row text contains a plausible hit.
        if not any(
            target["line"] in raw_text
            and target["from"] in raw_text
            and target["to"] in raw_text
            for target in targets
        ):
            if pos % 100 == 0 or pos == len(row_paths):
                print(
                    f"[{_now()}] {source_label}: scanned rows={pos}/{len(row_paths)} matched={len(out)}",
                    flush=True,
                )
            continue
        payload = json.loads(raw_text)
        row_idx = int(payload.get("row_idx", row_path.stem.split("_")[-1]))
        origin = str(payload.get("origin", ""))
        cells = payload.get("cells") or []
        row_matches = 0
        cell_matches = 0
        matched_reversals: set[str] = set()
        matched_destinations: set[int] = set()
        for dest_idx, cell_raw in enumerate(cells):
            paths = _cell_paths(cell_raw)
            cell_hit = False
            for path in paths:
                if not isinstance(path, dict):
                    continue
                hits = _path_matches(path, targets)
                if hits:
                    row_matches += 1
                    cell_hit = True
                    matched_reversals.update(hits)
            if cell_hit:
                cell_matches += 1
                matched_destinations.add(dest_idx)
        if row_matches:
            out[row_idx] = {
                "row_idx": row_idx,
                "origin": origin,
                "source": source_label,
                "matched_paths": row_matches,
                "matched_cells": cell_matches,
                "matched_destinations": len(matched_destinations),
                "matched_reversals": sorted(matched_reversals),
            }
        if pos % 100 == 0 or pos == len(row_paths):
            print(
                f"[{_now()}] {source_label}: scanned rows={pos}/{len(row_paths)} matched={len(out)}",
                flush=True,
            )
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _prepare_targets(selected: list[str]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for key in selected:
        source = REVERSAL_CANDIDATES[key]
        original_delta = _safe_int(source["original_delta_min"])
        old_dep = _parse_mod120(source["original_departure_time_mod120"])
        old_arr = _parse_mod120(source["original_arrival_time_mod120"])
        current_dep = (old_dep + original_delta) % 120
        current_arr = (old_arr + original_delta) % 120
        targets.append(
            {
                "key": key,
                "line": source["line"],
                "from": source["section_from_station"],
                "to": source["section_to_station"],
                "dep_mods": {old_dep % 120, current_dep},
                "arr_mods": {old_arr % 120, current_arr},
            }
        )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a safe surgical reroute for reversing inherited Step 1/2 edits in the final Step 3 timetable. "
            "By default this writes only corrected timetable/manifest files; it does not remove row JSONs."
        )
    )
    parser.add_argument("--step3-timetable-csv", type=Path, default=DEFAULT_STEP3_TIMETABLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--v0-root", type=Path, default=DEFAULT_V0_ROOT)
    parser.add_argument("--step2-root", type=Path, default=DEFAULT_STEP2_ROOT)
    parser.add_argument("--step3-root", type=Path, default=DEFAULT_STEP3_ROOT)
    parser.add_argument(
        "--remove",
        action="append",
        choices=sorted(REVERSAL_CANDIDATES),
        help="Reversal key to include. Can be passed multiple times. Defaults to ic5_neuchatel_biel.",
    )
    parser.add_argument("--include-all-negative-known", action="store_true")
    args = parser.parse_args()

    selected = list(args.remove or [])
    if args.include_all_negative_known:
        selected = sorted(set(selected) | set(REVERSAL_CANDIDATES))
    if not selected:
        selected = ["ic5_neuchatel_biel"]

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(args.step3_timetable_csv.expanduser().resolve())
    segments = _parse_segments(rows)
    row_to_segment = {row_idx: segment for segment in segments for row_idx in segment.row_indices}
    year_col_indices = _discover_year_column_indices(rows[0])

    working_rows = rows
    reversal_rows: list[dict[str, str]] = []
    apply_records: list[dict[str, Any]] = []
    for rank, key in enumerate(selected, start=1):
        inverse_row = _candidate_to_inverse_row(key, REVERSAL_CANDIDATES[key], rank)
        ok, new_rows, info = _try_apply_candidate(
            raw_row=inverse_row,
            rank=rank,
            rows=working_rows,
            segments=segments,
            row_to_segment=row_to_segment,
            year_col_indices=year_col_indices,
        )
        apply_records.append({"reversal_key": key, "ok": ok, **inverse_row, **info})
        if not ok:
            raise RuntimeError(f"Could not apply reversal {key}: {info}")
        working_rows = new_rows
        reversal_rows.append(inverse_row)

    corrected_timetable = output_dir / "FullTimetableHistory2026_2035_STEP3_REVERSAL_PREPARED.csv"
    _write_rows(corrected_timetable, working_rows)

    reversal_csv = output_dir / "step3_reversal_candidates.csv"
    reversal_fields = [
        "selection_order",
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
        "source_physical_shift_id",
        "reversal_key",
        "reversal_reason",
    ]
    _write_csv(reversal_csv, reversal_rows, reversal_fields)

    apply_csv = output_dir / "step3_reversal_apply_validation.csv"
    _write_csv(
        apply_csv,
        apply_records,
        sorted({key for row in apply_records for key in row.keys()}),
    )

    targets = _prepare_targets(selected)
    source_roots = [
        ("baseline_v0", args.v0_root.expanduser().resolve()),
        ("step2", args.step2_root.expanduser().resolve()),
        ("step3_current", args.step3_root.expanduser().resolve()),
    ]
    by_row: dict[int, dict[str, Any]] = {}
    source_summaries: list[dict[str, Any]] = []
    for label, root in source_roots:
        matches = _scan_root_for_impacted_origins(root=root, year="2035", targets=targets, source_label=label)
        source_summaries.append(
            {
                "source": label,
                "root": str(root),
                "matched_origin_rows": len(matches),
                "matched_paths": sum(int(v["matched_paths"]) for v in matches.values()),
                "matched_cells": sum(int(v["matched_cells"]) for v in matches.values()),
            }
        )
        for row_idx, rec in matches.items():
            entry = by_row.setdefault(
                row_idx,
                {
                    "year": "2035",
                    "row_idx": row_idx,
                    "origin": rec["origin"],
                    "matched_sources": set(),
                    "matched_reversals": set(),
                    "matched_paths": 0,
                    "matched_cells_source_sum": 0,
                },
            )
            entry["matched_sources"].add(label)
            entry["matched_reversals"].update(rec["matched_reversals"])
            entry["matched_paths"] += int(rec["matched_paths"])
            entry["matched_cells_source_sum"] += int(rec["matched_cells"])

    manifest_rows: list[dict[str, Any]] = []
    for row_idx, rec in sorted(by_row.items()):
        manifest_rows.append(
            {
                "year": rec["year"],
                "row_idx": row_idx,
                "row_file": f"row_{row_idx:06d}.json",
                "origin": rec["origin"],
                "matched_sources": ";".join(sorted(rec["matched_sources"])),
                "matched_reversals": ";".join(sorted(rec["matched_reversals"])),
                "matched_paths_across_sources": rec["matched_paths"],
                "matched_cells_source_sum": rec["matched_cells_source_sum"],
            }
        )

    manifest_csv = output_dir / "step3_reversal_impacted_origin_rows.csv"
    _write_csv(
        manifest_csv,
        manifest_rows,
        [
            "year",
            "row_idx",
            "row_file",
            "origin",
            "matched_sources",
            "matched_reversals",
            "matched_paths_across_sources",
            "matched_cells_source_sum",
        ],
    )

    summary = {
        "created_at": _now(),
        "selected_reversals": selected,
        "step3_timetable_csv": str(args.step3_timetable_csv.expanduser().resolve()),
        "corrected_timetable_csv": str(corrected_timetable),
        "reversal_candidates_csv": str(reversal_csv),
        "apply_validation_csv": str(apply_csv),
        "impacted_origin_rows_csv": str(manifest_csv),
        "impacted_origin_rows": len(manifest_rows),
        "source_summaries": source_summaries,
        "apply_records": apply_records,
        "targets": [
            {
                **target,
                "dep_mods": sorted(target["dep_mods"]),
                "arr_mods": sorted(target["arr_mods"]),
            }
            for target in targets
        ],
    }
    summary_path = output_dir / "step3_reversal_surgery_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
