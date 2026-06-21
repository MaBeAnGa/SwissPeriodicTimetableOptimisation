#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_optimised_v2_timetable import (
    _discover_year_column_indices,
    _load_rows,
    _parse_segments,
    _write_rows,
)
from generate_better_optimization_timetable_suggestions import (
    DEFAULT_PHYSICAL,
    DEFAULT_SOURCE_DIR,
    _candidate_section_key,
    _candidate_stations,
    _copy_shifted_legs,
    _fmt,
    _leg_section_key,
    _load_station_min_transfer,
    _path_components_from_legs,
    _path_weight,
    _physical_id,
    _scan_whole_scenario_existing_paths,
)
from generate_proposed_timetable_changes import (
    DEFAULT_STATIONS,
    DEFAULT_TRANSFER_2026,
    DEFAULT_TRANSFER_2035,
    _affected_keys,
    _read_csv,
    _safe_float,
    _safe_int,
    _try_apply_candidate,
    _write_dict_csv,
)


DEFAULT_STEP2_DIR = DEFAULT_SOURCE_DIR / "ACTUALLY_BETTER_OPTIMIZED"
DEFAULT_STEP2_TIMETABLE = DEFAULT_STEP2_DIR / "FullTimetableHistory2026_2035_BETTER_OPTIMIZATION_preview.csv"
DEFAULT_STEP2_ACCEPTED = DEFAULT_STEP2_DIR / "BetterOptimization_AllScenarioChanges_SeedPlusNew_2026_2035.csv"
DEFAULT_STEP2_STATE = Path.home() / "od_fast_state" / "nondominated_cells_state_ACTUALLY_BETTER_OPTIMIZED"
DEFAULT_WORK_DIR = DEFAULT_SOURCE_DIR / "STEP3_SEARCH"
DEFAULT_PREPARED_INPUT_DIR = DEFAULT_WORK_DIR / "prepared_transfer_threshold_19"


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(path: Path, message: str) -> None:
    line = f"[{_now()}] {message}"
    print(line, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def _round(value: Any, digits: int = 6) -> float:
    try:
        return round(float(value), digits)
    except Exception:
        return 0.0


def _as_years(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value).strip()
        if text in {"2026", "2035"} and text not in out:
            out.append(text)
    if not out:
        raise ValueError("At least one model year must be selected.")
    return out


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str, str]:
    return (
        -_safe_float(row.get("net_pax_minutes_proxy"), 0.0),
        -_safe_float(row.get("gross_gain_pax_minutes_proxy"), 0.0),
        _safe_float(row.get("gross_loss_pax_minutes_proxy"), 0.0),
        str(row.get("year", "")),
        _physical_id(row),
    )


def _has_static_breakage(row: dict[str, Any]) -> bool:
    return (
        _safe_float(row.get("broken_existing_daily_pax_proxy"), 0.0) > 0
        or _safe_int(row.get("broken_existing_transfer_events"), 0) > 0
        or _safe_int(row.get("same_line_pair_count"), 0) > 0
        or bool(str(row.get("feasibility_notes", "")).strip())
    )


def _delta_allowed_by_source_leeway(row: dict[str, Any], delta: int) -> bool:
    if delta == 0:
        return False
    if delta < 0:
        return _safe_float(row.get("min_can_shift_earlier_min"), 0.0) >= abs(delta)
    return _safe_float(row.get("min_can_shift_later_min"), 0.0) >= delta


def _without_physical_id(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out.pop("physical_shift_id", None)
    return out


def _make_candidate_variant(row: dict[str, Any], delta: int, source_id: str) -> dict[str, Any]:
    variant = _without_physical_id(row)
    variant["delta_min"] = str(delta)
    variant["step3_source_physical_shift_id"] = source_id
    variant["step3_generation_method"] = "expanded_delta_from_existing_transfer_context"
    variant["physical_shift_id"] = _physical_id(variant)
    return variant


def _load_prior_accepted(
    path: Path,
    years: set[str],
) -> tuple[list[dict[str, str]], set[str], set[tuple[str, str, str, str, int]]]:
    if not path.is_file():
        return [], set(), set()
    rows = [row for row in _read_csv(path) if str(row.get("year", "")).strip() in years]
    ids = {_physical_id(row) for row in rows}
    keys: set[tuple[str, str, str, str, int]] = set()
    for row in rows:
        keys.update(_affected_keys(row))
    return rows, ids, keys


def _run_logged_command(cmd: list[str], log_path: Path) -> None:
    _log(log_path, "Running: " + " ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    with log_path.open("a", encoding="utf-8") as log_file:
        for line in proc.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
    return_code = proc.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def _prepare_step3_transfer_inputs(
    *,
    prepare: bool,
    force: bool,
    years: list[str],
    timetable_csv: Path,
    stations_csv: Path,
    transfer_2026: Path,
    transfer_2035: Path,
    output_dir: Path,
    threshold_pax: float,
    max_shift_min: int,
    near_miss_before_min: int,
    near_miss_after_min: int,
    fallback_physical_shifts_csv: Path,
    log_path: Path,
) -> Path:
    if not prepare:
        _log(log_path, f"Using existing physical-shift table without Step 3 preparation: {fallback_physical_shifts_csv}")
        return fallback_physical_shifts_csv

    output_dir.mkdir(parents=True, exist_ok=True)
    physical_path = output_dir / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv"
    setup_summary = output_dir / "transfer_optimization_setup_summary.json"
    net_summary = output_dir / "transfer_optimization_net_impact_summary.json"
    if physical_path.is_file() and setup_summary.is_file() and net_summary.is_file() and not force:
        _log(log_path, f"Reusing prepared Step 3 transfer inputs: {physical_path}")
        return physical_path

    script_dir = Path(__file__).resolve().parent
    setup_script = script_dir / "setup_transfer_optimization_tables.py"
    net_script = script_dir / "build_transfer_shift_net_impact.py"
    year_args = [str(year) for year in years]

    _log(
        log_path,
        "Preparing Step 3 transfer-context tables "
        f"threshold_pax={threshold_pax} max_shift_min={max_shift_min} output_dir={output_dir}",
    )
    _run_logged_command(
        [
            sys.executable,
            str(setup_script),
            "--timetable-csv",
            str(timetable_csv),
            "--stations-csv",
            str(stations_csv),
            "--transfer-2026",
            str(transfer_2026),
            "--transfer-2035",
            str(transfer_2035),
            "--output-dir",
            str(output_dir),
            "--threshold-pax",
            str(threshold_pax),
            "--near-miss-before-min",
            str(near_miss_before_min),
            "--near-miss-after-min",
            str(near_miss_after_min),
            "--max-shift-min",
            str(max_shift_min),
            "--years",
            *year_args,
        ],
        log_path,
    )
    _run_logged_command(
        [
            sys.executable,
            str(net_script),
            "--tables-dir",
            str(output_dir),
            "--transfer-2026",
            str(transfer_2026),
            "--transfer-2035",
            str(transfer_2035),
            "--threshold-pax",
            str(threshold_pax),
            "--detail-top-n",
            "200",
        ],
        log_path,
    )
    if not physical_path.is_file():
        raise FileNotFoundError(f"Prepared physical-shift table was not created: {physical_path}")
    _log(log_path, f"Prepared Step 3 physical-shift table: {physical_path}")
    return physical_path


def _build_step3_candidate_pool(
    *,
    physical_rows: list[dict[str, str]],
    years: set[str],
    prior_ids: set[str],
    prior_event_keys: set[tuple[str, str, str, str, int]],
    max_shift_min: int,
    expand_deltas: bool,
    block_prior_event_keys: bool,
    reject_static_breakage: bool,
    min_static_net: float,
    min_static_gross_gain: float,
    max_static_loss_ratio: float,
    max_candidate_pool: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rejected: list[dict[str, Any]] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}

    source_rows = sorted(physical_rows, key=_candidate_sort_key)
    for source in source_rows:
        year = str(source.get("year", "")).strip()
        source_id = _physical_id(source)
        if year not in years:
            continue
        if source_id in prior_ids:
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "already_accepted_in_step1_or_step2"})
            continue
        if block_prior_event_keys and (set(_affected_keys(source)) & prior_event_keys):
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "touches_step1_or_step2_event_key"})
            continue
        if reject_static_breakage and _has_static_breakage(source):
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "static_transfer_breakage_or_boundary_flag"})
            continue
        gross_gain = _safe_float(source.get("gross_gain_pax_minutes_proxy"), 0.0)
        gross_loss = _safe_float(source.get("gross_loss_pax_minutes_proxy"), 0.0)
        net = _safe_float(source.get("net_pax_minutes_proxy"), 0.0)
        if net < min_static_net:
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "below_static_net_floor"})
            continue
        if gross_gain < min_static_gross_gain:
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "below_static_gross_gain_floor"})
            continue
        if gross_gain > 0 and gross_loss / gross_gain > max_static_loss_ratio:
            rejected.append({"physical_shift_id": source_id, "year": year, "status": "rejected_pool", "reason": "above_static_loss_ratio"})
            continue

        deltas: list[int]
        if expand_deltas:
            deltas = [delta for delta in range(-max_shift_min, max_shift_min + 1) if _delta_allowed_by_source_leeway(source, delta)]
        else:
            delta = _safe_int(source.get("delta_min"), 0)
            deltas = [delta] if delta and abs(delta) <= max_shift_min and _delta_allowed_by_source_leeway(source, delta) else []
        for delta in deltas:
            variant = _make_candidate_variant(source, delta, source_id)
            pid = _physical_id(variant)
            if pid in prior_ids:
                continue
            if block_prior_event_keys and (set(_affected_keys(variant)) & prior_event_keys):
                continue
            previous = candidates_by_id.get(pid)
            if previous is None or _safe_float(variant.get("net_pax_minutes_proxy"), 0.0) > _safe_float(previous.get("net_pax_minutes_proxy"), 0.0):
                candidates_by_id[pid] = variant

    candidates = sorted(candidates_by_id.values(), key=_candidate_sort_key)
    if max_candidate_pool > 0:
        candidates = candidates[:max_candidate_pool]
    return candidates, rejected


def _candidate_year_station_summary(row: dict[str, Any]) -> str:
    stations = sorted(_candidate_stations(row))
    if not stations:
        return ""
    return "; ".join(stations[:8])


def _scan_marginal_candidate_scores(
    *,
    years: list[str],
    rows_root: Path,
    candidates: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
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

    base_shift_map = {_candidate_section_key(row): _safe_int(row.get("delta_min"), 0) for row in accepted_rows}

    scores: dict[str, dict[str, Any]] = {
        _physical_id(row): {
            "marginal_reference_pax_minutes_saved": 0.0,
            "marginal_reference_weight_touched": 0.0,
            "marginal_reference_paths_touched": 0,
            "marginal_reference_invalidated_daily_pax": 0.0,
            "marginal_reference_invalidated_paths": 0,
            "marginal_reference_rolling_share_delta_weighted": 0.0,
            "marginal_example_od": "",
        }
        for row in candidates
    }

    for year in years:
        rows_dir = rows_root / year / "rows"
        row_files = sorted(rows_dir.glob("row_*.json"))
        if not row_files:
            raise FileNotFoundError(f"Missing Step 2 row-state JSONs: {rows_dir}")
        if scan_row_limit > 0:
            row_files = row_files[:scan_row_limit]
            _log(log_path, f"{year}: LIMITED marginal scan row_count={len(row_files)}")
        _log(
            log_path,
            f"{year}: marginal scan candidates={len(candidates)} accepted_base={len(accepted_rows)} rows={len(row_files)}",
        )
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
                destination = str(cell.get("destination") or f"dest_idx_{dest_idx}")
                for path in paths:
                    if not isinstance(path, dict):
                        continue
                    weight = _path_weight(path, total_demand)
                    if weight <= 0:
                        continue
                    legs = path.get("legs")
                    if not isinstance(legs, list) or not legs:
                        continue

                    touched_candidate_ids: set[str] = set()
                    for leg in legs:
                        touched_candidate_ids.update(candidate_by_key.get(_leg_section_key(year, leg), []))
                    if not touched_candidate_ids:
                        continue

                    base_legs, base_touched = _copy_shifted_legs(
                        legs=[dict(leg) for leg in legs],
                        shift_map=base_shift_map,
                        year=year,
                    )
                    if base_touched:
                        base_ok, base_metrics = _path_components_from_legs(
                            legs=base_legs,
                            station_min_transfer=station_min_transfer,
                        )
                    else:
                        base_ok, base_metrics = _path_components_from_legs(
                            legs=[dict(leg) for leg in legs],
                            station_min_transfer=station_min_transfer,
                        )
                    if not base_ok:
                        continue

                    for cid in touched_candidate_ids:
                        shift_map = dict(base_shift_map)
                        shift_map[candidate_key_by_id[cid]] = candidate_delta[cid]
                        shifted_legs, touched = _copy_shifted_legs(
                            legs=[dict(leg) for leg in legs],
                            shift_map=shift_map,
                            year=year,
                        )
                        if not touched:
                            continue
                        new_ok, new_metrics = _path_components_from_legs(
                            legs=shifted_legs,
                            station_min_transfer=station_min_transfer,
                        )
                        item = scores[cid]
                        item["marginal_reference_weight_touched"] += weight
                        item["marginal_reference_paths_touched"] += 1
                        if not new_ok:
                            item["marginal_reference_invalidated_daily_pax"] += weight
                            item["marginal_reference_invalidated_paths"] += 1
                            continue
                        item["marginal_reference_pax_minutes_saved"] += (base_metrics["total"] - new_metrics["total"]) * weight
                        item["marginal_reference_rolling_share_delta_weighted"] += (
                            new_metrics["rolling_share"] - base_metrics["rolling_share"]
                        ) * weight
                        if not item["marginal_example_od"]:
                            item["marginal_example_od"] = (
                                f"{origin}->{destination} weight={_fmt(weight, 3)} "
                                f"total {base_metrics['total']:.3f}->{new_metrics['total']:.3f}"
                            )
            if idx % max(1, progress_every_rows) == 0 or idx == len(row_files):
                _log(log_path, f"{year}: marginal scan progress rows={idx}/{len(row_files)}")

    for item in scores.values():
        weight = float(item["marginal_reference_weight_touched"])
        saved = float(item["marginal_reference_pax_minutes_saved"])
        if weight > 0:
            item["marginal_reference_mean_total_delta_min"] = -saved / weight
            item["marginal_reference_rolling_share_delta"] = (
                float(item["marginal_reference_rolling_share_delta_weighted"]) / weight
            )
        else:
            item["marginal_reference_mean_total_delta_min"] = 0.0
            item["marginal_reference_rolling_share_delta"] = 0.0
    return scores



def _compact_leg(leg: dict[str, Any]) -> dict[str, Any]:
    return {
        "service_name": str(leg.get("service_name", "")),
        "from_station": str(leg.get("from_station", "")),
        "to_station": str(leg.get("to_station", "")),
        "dep_min": int(leg.get("dep_min", 0)),
        "arr_min": int(leg.get("arr_min", 0)),
        "dep_time": str(leg.get("dep_time", "")),
        "arr_time": str(leg.get("arr_time", "")),
        "canonical_trip_id": str(leg.get("canonical_trip_id", "")),
    }


def _build_reference_path_cache(
    *,
    years: list[str],
    rows_root: Path,
    candidates: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
    log_path: Path,
    progress_every_rows: int,
    scan_row_limit: int = 0,
) -> list[dict[str, Any]]:
    candidate_keys = {_candidate_section_key(row) for row in candidates}
    records: list[dict[str, Any]] = []
    total_paths_seen = 0
    total_paths_cached = 0
    for year in years:
        rows_dir = rows_root / year / "rows"
        row_files = sorted(rows_dir.glob("row_*.json"))
        if not row_files:
            raise FileNotFoundError(f"Missing Step 2 row-state JSONs: {rows_dir}")
        if scan_row_limit > 0:
            row_files = row_files[:scan_row_limit]
            _log(log_path, f"{year}: LIMITED reference-cache build row_count={len(row_files)}")
        _log(log_path, f"{year}: building reference-path cache rows={len(row_files)} candidate_section_keys={len(candidate_keys)}")
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
                destination = str(cell.get("destination") or f"dest_idx_{dest_idx}")
                for path in paths:
                    if not isinstance(path, dict):
                        continue
                    total_paths_seen += 1
                    weight = _path_weight(path, total_demand)
                    if weight <= 0:
                        continue
                    raw_legs = path.get("legs")
                    if not isinstance(raw_legs, list) or not raw_legs:
                        continue
                    legs = [_compact_leg(dict(leg)) for leg in raw_legs]
                    touched_keys = {_leg_section_key(year, leg) for leg in legs if _leg_section_key(year, leg) in candidate_keys}
                    if not touched_keys:
                        continue
                    old_ok, old_metrics = _path_components_from_legs(
                        legs=[dict(leg) for leg in legs],
                        station_min_transfer=station_min_transfer,
                    )
                    if not old_ok:
                        continue
                    total_paths_cached += 1
                    records.append(
                        {
                            "year": year,
                            "origin": origin,
                            "destination": destination,
                            "weight": weight,
                            "legs": legs,
                            "candidate_keys": tuple(sorted(touched_keys)),
                            "base_metrics_unshifted": old_metrics,
                        }
                    )
            if idx % max(1, progress_every_rows) == 0 or idx == len(row_files):
                _log(
                    log_path,
                    f"{year}: reference-cache progress rows={idx}/{len(row_files)} cached_paths={total_paths_cached}",
                )
    _log(log_path, f"Reference-path cache complete paths_seen={total_paths_seen} cached_paths={len(records)}")
    return records


def _score_candidates_from_reference_cache(
    *,
    candidates: list[dict[str, Any]],
    accepted_rows: list[dict[str, Any]],
    path_cache: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
    log_path: Path,
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

    base_shift_map = {_candidate_section_key(row): _safe_int(row.get("delta_min"), 0) for row in accepted_rows}
    scores: dict[str, dict[str, Any]] = {
        _physical_id(row): {
            "marginal_reference_pax_minutes_saved": 0.0,
            "marginal_reference_weight_touched": 0.0,
            "marginal_reference_paths_touched": 0,
            "marginal_reference_invalidated_daily_pax": 0.0,
            "marginal_reference_invalidated_paths": 0,
            "marginal_reference_rolling_share_delta_weighted": 0.0,
            "marginal_example_od": "",
        }
        for row in candidates
    }

    _log(log_path, f"Scoring from reference-path cache records={len(path_cache)} candidates={len(candidates)} accepted_base={len(accepted_rows)}")
    for record in path_cache:
        touched_candidate_ids: set[str] = set()
        for key in record["candidate_keys"]:
            touched_candidate_ids.update(candidate_by_key.get(key, []))
        if not touched_candidate_ids:
            continue

        year = str(record["year"])
        legs = [dict(leg) for leg in record["legs"]]
        base_legs, base_touched = _copy_shifted_legs(
            legs=[dict(leg) for leg in legs],
            shift_map=base_shift_map,
            year=year,
        )
        if base_touched:
            base_ok, base_metrics = _path_components_from_legs(
                legs=base_legs,
                station_min_transfer=station_min_transfer,
            )
        else:
            base_ok = True
            base_metrics = dict(record["base_metrics_unshifted"])
        if not base_ok:
            continue

        weight = float(record["weight"])
        for cid in touched_candidate_ids:
            shift_map = dict(base_shift_map)
            shift_map[candidate_key_by_id[cid]] = candidate_delta[cid]
            shifted_legs, touched = _copy_shifted_legs(
                legs=[dict(leg) for leg in legs],
                shift_map=shift_map,
                year=year,
            )
            if not touched:
                continue
            new_ok, new_metrics = _path_components_from_legs(
                legs=shifted_legs,
                station_min_transfer=station_min_transfer,
            )
            item = scores[cid]
            item["marginal_reference_weight_touched"] += weight
            item["marginal_reference_paths_touched"] += 1
            if not new_ok:
                item["marginal_reference_invalidated_daily_pax"] += weight
                item["marginal_reference_invalidated_paths"] += 1
                continue
            item["marginal_reference_pax_minutes_saved"] += (base_metrics["total"] - new_metrics["total"]) * weight
            item["marginal_reference_rolling_share_delta_weighted"] += (
                new_metrics["rolling_share"] - base_metrics["rolling_share"]
            ) * weight
            if not item["marginal_example_od"]:
                item["marginal_example_od"] = (
                    f"{record['origin']}->{record['destination']} weight={_fmt(weight, 3)} "
                    f"total {base_metrics['total']:.3f}->{new_metrics['total']:.3f}"
                )

    for item in scores.values():
        weight = float(item["marginal_reference_weight_touched"])
        saved = float(item["marginal_reference_pax_minutes_saved"])
        if weight > 0:
            item["marginal_reference_mean_total_delta_min"] = -saved / weight
            item["marginal_reference_rolling_share_delta"] = float(item["marginal_reference_rolling_share_delta_weighted"]) / weight
        else:
            item["marginal_reference_mean_total_delta_min"] = 0.0
            item["marginal_reference_rolling_share_delta"] = 0.0
    return scores


def _scan_combined_from_reference_cache(
    *,
    accepted_rows: list[dict[str, Any]],
    path_cache: list[dict[str, Any]],
    station_min_transfer: dict[str, int],
    log_path: Path,
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
    _log(log_path, f"Combined reference-cache check accepted_rows={len(accepted_rows)} records={len(path_cache)}")
    for record in path_cache:
        touched_keys = [key for key in record["candidate_keys"] if key in shift_map]
        if not touched_keys:
            continue
        year = str(record["year"])
        legs = [dict(leg) for leg in record["legs"]]
        shifted_legs, touched = _copy_shifted_legs(
            legs=[dict(leg) for leg in legs],
            shift_map=shift_map,
            year=year,
        )
        if not touched:
            continue
        old_metrics = dict(record["base_metrics_unshifted"])
        new_ok, new_metrics = _path_components_from_legs(
            legs=shifted_legs,
            station_min_transfer=station_min_transfer,
        )
        weight = float(record["weight"])
        total_weight_touched += weight
        touched_paths += 1
        touched_ids = {shift_id_by_key[key] for key in touched_keys if key in shift_id_by_key}
        if not new_ok:
            invalidated_pax += weight
            invalidated_paths += 1
            for cid in touched_ids:
                invalidated_by_candidate[cid]["daily_pax"] = float(invalidated_by_candidate[cid]["daily_pax"]) + weight
                invalidated_by_candidate[cid]["paths"] = int(invalidated_by_candidate[cid]["paths"]) + 1
            continue
        total_saved += (old_metrics["total"] - new_metrics["total"]) * weight
        rolling_share_delta_weighted += (new_metrics["rolling_share"] - old_metrics["rolling_share"]) * weight
    return {
        "accepted_change_rows": len(accepted_rows),
        "current_route_exact_pax_minutes_saved": round(total_saved, 6),
        "current_route_exact_weight_touched": round(total_weight_touched, 6),
        "current_route_exact_touched_paths": int(touched_paths),
        "current_route_exact_invalidated_daily_pax": round(invalidated_pax, 6),
        "current_route_exact_invalidated_paths": int(invalidated_paths),
        "invalidated_by_physical_shift_id": {
            cid: {"daily_pax": round(float(values["daily_pax"]), 6), "paths": int(values["paths"])}
            for cid, values in sorted(invalidated_by_candidate.items(), key=lambda item: (-float(item[1]["daily_pax"]), str(item[0])))
        },
        "current_route_exact_mean_total_delta_min": round(-total_saved / total_weight_touched, 9) if total_weight_touched > 0 else 0.0,
        "current_route_exact_rolling_share_delta": round(rolling_share_delta_weighted / total_weight_touched, 9) if total_weight_touched > 0 else 0.0,
        "scan_source": "reference_path_cache",
    }


def _merge_candidate_score(
    row: dict[str, Any],
    score: dict[str, Any],
    iteration: int,
    near_miss_weight: float,
) -> dict[str, Any]:
    near_gain = max(0.0, _safe_float(row.get("near_miss_gain_pax_minutes"), 0.0))
    exact_saved = float(score.get("marginal_reference_pax_minutes_saved", 0.0))
    invalidated = float(score.get("marginal_reference_invalidated_daily_pax", 0.0))
    # Exact current-path savings are observed directly. Discounted near-miss gains represent
    # newly feasible transfers that cannot be observed until a full route rebuild is run.
    selection_score = exact_saved + near_miss_weight * near_gain
    return {
        **row,
        "physical_shift_id": _physical_id(row),
        "iteration_scored": iteration,
        "candidate_stations": _candidate_year_station_summary(row),
        "marginal_reference_pax_minutes_saved": _round(exact_saved),
        "marginal_reference_weight_touched": _round(score.get("marginal_reference_weight_touched", 0.0)),
        "marginal_reference_paths_touched": int(score.get("marginal_reference_paths_touched", 0)),
        "marginal_reference_invalidated_daily_pax": _round(invalidated),
        "marginal_reference_invalidated_paths": int(score.get("marginal_reference_invalidated_paths", 0)),
        "marginal_reference_mean_total_delta_min": _round(score.get("marginal_reference_mean_total_delta_min", 0.0), 9),
        "marginal_reference_rolling_share_delta": _round(score.get("marginal_reference_rolling_share_delta", 0.0), 9),
        "step3_near_miss_score_pax_minutes": _round(near_miss_weight * near_gain),
        "step3_selection_score": _round(selection_score),
        "marginal_example_od": score.get("marginal_example_od", ""),
    }


def _candidate_is_eligible(
    row: dict[str, Any],
    *,
    min_selection_score: float,
    min_exact_reference_saved: float,
    max_invalidated_daily_pax: float,
    used_ids: set[str],
    used_event_keys: set[tuple[str, str, str, str, int]],
) -> tuple[bool, str]:
    pid = str(row.get("physical_shift_id", ""))
    if pid in used_ids:
        return False, "already_selected_or_previously_used"
    if set(_affected_keys(row)) & used_event_keys:
        return False, "overlaps_already_selected_event_key"
    if _safe_float(row.get("marginal_reference_invalidated_daily_pax"), 0.0) > max_invalidated_daily_pax:
        return False, "invalidates_reference_path_transfer"
    if _safe_float(row.get("marginal_reference_pax_minutes_saved"), 0.0) < min_exact_reference_saved:
        return False, "negative_or_insufficient_exact_reference_effect"
    if _safe_float(row.get("step3_selection_score"), 0.0) < min_selection_score:
        return False, "below_step3_selection_score_threshold"
    return True, ""


def _write_rows_union(path: Path, rows: list[dict[str, Any]], preferred: list[str]) -> None:
    fields: list[str] = []
    for field in preferred:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    _write_dict_csv(path, rows, fields)


def _arg_path(value: Any) -> str:
    return str(Path(value).expanduser())


def _run_sequential_year_subprocesses(args: argparse.Namespace, years: list[str], parent_log_path: Path) -> None:
    """Run each selected year in a separate Python process so large caches are freed between years."""
    per_year_hours = float(args.per_year_time_limit_hours)
    if per_year_hours <= 0:
        per_year_hours = max(1.0 / 60.0, float(args.time_limit_hours) / max(1, len(years)))

    base_work_dir = Path(args.work_dir).expanduser().resolve()
    base_prepared_dir = Path(args.prepared_input_dir).expanduser().resolve()
    script_path = Path(__file__).resolve()
    child_dirs: dict[str, Path] = {}

    for year in years:
        child_work_dir = base_work_dir / f"year_{year}"
        child_prepared_dir = base_prepared_dir / f"year_{year}"
        child_dirs[year] = child_work_dir
        cmd = [
            sys.executable,
            str(script_path),
            "--years", year,
            "--step2-timetable-csv", _arg_path(args.step2_timetable_csv),
            "--step2-accepted-csv", _arg_path(args.step2_accepted_csv),
            "--physical-shifts-csv", _arg_path(args.physical_shifts_csv),
            "--stations-csv", _arg_path(args.stations_csv),
            "--transfer-2026", _arg_path(args.transfer_2026),
            "--transfer-2035", _arg_path(args.transfer_2035),
            "--step2-row-state-root", _arg_path(args.step2_row_state_root),
            "--work-dir", str(child_work_dir),
            "--prepare-transfer-inputs", str(int(args.prepare_transfer_inputs)),
            "--force-prepare-transfer-inputs", str(int(args.force_prepare_transfer_inputs)),
            "--prepared-input-dir", str(child_prepared_dir),
            "--transfer-threshold-pax", str(args.transfer_threshold_pax),
            "--near-miss-before-min", str(args.near_miss_before_min),
            "--near-miss-after-min", str(args.near_miss_after_min),
            "--max-shift-min", str(args.max_shift_min),
            "--expand-deltas", str(int(args.expand_deltas)),
            "--max-candidate-pool", str(args.max_candidate_pool),
            "--min-static-net", str(args.min_static_net),
            "--min-static-gross-gain", str(args.min_static_gross_gain),
            "--max-static-loss-ratio", str(args.max_static_loss_ratio),
            "--reject-static-breakage", str(int(args.reject_static_breakage)),
            "--block-step2-event-keys", str(int(args.block_step2_event_keys)),
            "--min-marginal-pax-minutes", str(args.min_marginal_pax_minutes),
            "--min-exact-reference-pax-minutes", str(args.min_exact_reference_pax_minutes),
            "--max-invalidated-daily-pax", str(args.max_invalidated_daily_pax),
            "--near-miss-tiebreak-weight", str(args.near_miss_tiebreak_weight),
            "--time-limit-hours", str(per_year_hours),
            "--max-iterations", str(args.max_iterations),
            "--combined-check-each-accept", str(int(args.combined_check_each_accept)),
            "--use-reference-path-cache", str(int(args.use_reference_path_cache)),
            "--progress-every-rows", str(args.progress_every_rows),
            "--scan-row-limit", str(args.scan_row_limit),
            "--sequential-years", "0",
        ]
        _log(parent_log_path, f"Sequential year run start year={year} time_limit_hours={per_year_hours} work_dir={child_work_dir}")
        _run_logged_command(cmd, parent_log_path)
        _log(parent_log_path, f"Sequential year run complete year={year}")

    _merge_sequential_year_outputs(
        years=years,
        child_dirs=child_dirs,
        parent_work_dir=base_work_dir,
        step2_timetable_csv=Path(args.step2_timetable_csv).expanduser().resolve(),
        log_path=parent_log_path,
    )


def _read_csv_if_exists(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _merge_named_csvs(child_dirs: dict[str, Path], filename: str, output_path: Path, preferred: list[str] | None = None) -> int:
    rows: list[dict[str, Any]] = []
    for year, child_dir in child_dirs.items():
        for row in _read_csv_if_exists(child_dir / filename):
            row.setdefault("sequential_run_year", year)
            rows.append(row)
    _write_rows_union(output_path, rows, preferred or [])
    return len(rows)


def _merge_sequential_previews(
    *,
    years: list[str],
    child_dirs: dict[str, Path],
    parent_work_dir: Path,
    step2_timetable_csv: Path,
    log_path: Path,
) -> None:
    base_rows = _load_rows(step2_timetable_csv)
    if not base_rows:
        return
    base_year_cols = _discover_year_column_indices(base_rows[0])
    for year in years:
        child_preview = child_dirs[year] / "FullTimetableHistory2026_2035_STEP3_preview.csv"
        if not child_preview.is_file():
            _log(log_path, f"Sequential merge: missing preview for year={year}: {child_preview}")
            continue
        child_rows = _load_rows(child_preview)
        if len(child_rows) != len(base_rows):
            _log(log_path, f"Sequential merge: cannot copy preview columns for year={year}; row counts differ")
            continue
        child_year_cols = _discover_year_column_indices(child_rows[0])
        if year not in base_year_cols or year not in child_year_cols:
            _log(log_path, f"Sequential merge: year column not found for year={year}")
            continue
        for base_idx, child_idx in zip(base_year_cols[year], child_year_cols[year]):
            for row_idx in range(len(base_rows)):
                base_rows[row_idx][base_idx] = child_rows[row_idx][child_idx]
    _write_rows(parent_work_dir / "FullTimetableHistory2026_2035_STEP3_preview.csv", base_rows)


def _merge_sequential_year_outputs(
    *,
    years: list[str],
    child_dirs: dict[str, Path],
    parent_work_dir: Path,
    step2_timetable_csv: Path,
    log_path: Path,
) -> None:
    parent_work_dir.mkdir(parents=True, exist_ok=True)
    accepted_count = _merge_named_csvs(
        child_dirs,
        "Step3_SuggestedTimetableChanges_2026_2035.csv",
        parent_work_dir / "Step3_SuggestedTimetableChanges_2026_2035.csv",
        ["sequential_run_year", "step3_acceptance_rank", "status", "year", "line", "delta_min", "physical_shift_id"],
    )
    rejected_count = _merge_named_csvs(
        child_dirs,
        "Step3_RejectedOrDeferredCandidates_2026_2035.csv",
        parent_work_dir / "Step3_RejectedOrDeferredCandidates_2026_2035.csv",
    )
    latest_count = _merge_named_csvs(
        child_dirs,
        "Step3_LatestMarginalCandidateScores_2026_2035.csv",
        parent_work_dir / "Step3_LatestMarginalCandidateScores_2026_2035.csv",
    )
    _merge_sequential_previews(
        years=years,
        child_dirs=child_dirs,
        parent_work_dir=parent_work_dir,
        step2_timetable_csv=step2_timetable_csv,
        log_path=log_path,
    )

    child_summaries: dict[str, Any] = {}
    for year, child_dir in child_dirs.items():
        summary_path = child_dir / "Step3_summary.json"
        if summary_path.is_file():
            try:
                child_summaries[year] = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                child_summaries[year] = {"summary_read_error": str(exc)}
        else:
            child_summaries[year] = {"summary_missing": str(summary_path)}
    summary = {
        "mode": "sequential_year_subprocesses",
        "years": years,
        "accepted_rows_merged": accepted_count,
        "rejected_or_deferred_rows_merged": rejected_count,
        "latest_score_rows_merged": latest_count,
        "child_work_dirs": {year: str(path) for year, path in child_dirs.items()},
        "child_summaries": child_summaries,
    }
    (parent_work_dir / "Step3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _log(log_path, f"Sequential merge complete accepted={accepted_count} rejected={rejected_count} latest_scores={latest_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Step 3 timetable suggestion search. Starts from the Step 2 timetable and Step 2 path state, "
            "then sequentially accepts the highest marginal-benefit compatible section shifts until no "
            "remaining candidate shows a positive exact reference-path improvement."
        )
    )
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--step2-timetable-csv", type=Path, default=DEFAULT_STEP2_TIMETABLE)
    parser.add_argument("--step2-accepted-csv", type=Path, default=DEFAULT_STEP2_ACCEPTED)
    parser.add_argument("--physical-shifts-csv", type=Path, default=DEFAULT_PHYSICAL)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--transfer-2026", type=Path, default=DEFAULT_TRANSFER_2026)
    parser.add_argument("--transfer-2035", type=Path, default=DEFAULT_TRANSFER_2035)
    parser.add_argument("--step2-row-state-root", type=Path, default=DEFAULT_STEP2_STATE)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--prepare-transfer-inputs", type=int, default=1)
    parser.add_argument("--force-prepare-transfer-inputs", type=int, default=1)
    parser.add_argument("--prepared-input-dir", type=Path, default=DEFAULT_PREPARED_INPUT_DIR)
    parser.add_argument("--transfer-threshold-pax", type=float, default=19.0)
    parser.add_argument("--near-miss-before-min", type=int, default=5)
    parser.add_argument("--near-miss-after-min", type=int, default=10)
    parser.add_argument("--max-shift-min", type=int, default=10)
    parser.add_argument("--expand-deltas", type=int, default=1)
    parser.add_argument(
        "--max-candidate-pool",
        type=int,
        default=0,
        help="Maximum scored candidate variants after static sorting. The default 0 scores every generated variant.",
    )
    parser.add_argument("--min-static-net", type=float, default=0.0)
    parser.add_argument("--min-static-gross-gain", type=float, default=0.0)
    parser.add_argument("--max-static-loss-ratio", type=float, default=1.25)
    parser.add_argument("--reject-static-breakage", type=int, default=1)
    parser.add_argument("--block-step2-event-keys", type=int, default=1)
    parser.add_argument(
        "--min-marginal-pax-minutes",
        type=float,
        default=50.0,
        help=(
            "Stop when the best compatible remaining edit has a Step 3 selection score below this value. "
            "The score is exact current-reference-path savings plus the discounted near-miss term."
        ),
    )
    parser.add_argument(
        "--min-exact-reference-pax-minutes",
        type=float,
        default=0.0,
        help="Reject edits whose exact current-reference-path effect is below this value before near-miss credit is added.",
    )
    parser.add_argument("--max-invalidated-daily-pax", type=float, default=0.0)
    parser.add_argument(
        "--near-miss-tiebreak-weight",
        type=float,
        default=0.25,
        help=(
            "Weight on lower-threshold near-miss proxy in the Step 3 selection score. "
            "The default 0.25 matches the conservative discount used in Step 2."
        ),
    )
    parser.add_argument("--time-limit-hours", type=float, default=8.0)
    parser.add_argument("--max-iterations", type=int, default=0, help="Safety/debug cap. Default 0 means no accepted-edit cap.")
    parser.add_argument("--combined-check-each-accept", type=int, default=1)
    parser.add_argument("--use-reference-path-cache", type=int, default=1)
    parser.add_argument("--progress-every-rows", type=int, default=25)
    parser.add_argument("--scan-row-limit", type=int, default=0, help="Smoke-test only. Leave 0 for full overnight run.")
    parser.add_argument(
        "--sequential-years",
        type=int,
        default=1,
        help=(
            "When more than one year is selected, run one child process per year so the large "
            "reference-path cache for the previous year is released before the next year starts."
        ),
    )
    parser.add_argument(
        "--per-year-time-limit-hours",
        type=float,
        default=0.0,
        help="Per-year time limit in sequential mode. Default 0 divides --time-limit-hours by the number of selected years.",
    )
    args = parser.parse_args()

    years = _as_years(args.years)
    year_set = set(years)
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "Step3_timetable_suggestion_search.log"
    log_path.write_text("", encoding="utf-8")
    start_time = time.time()
    time_limit_seconds = max(60.0, float(args.time_limit_hours) * 3600.0)

    if int(args.sequential_years) and len(years) > 1:
        _log(log_path, f"Step 3 sequential-year wrapper start years={years}")
        _run_sequential_year_subprocesses(args, years, log_path)
        _log(log_path, "Step 3 sequential-year wrapper complete")
        return

    _log(log_path, f"Step 3 search start years={years}")
    _log(log_path, f"Step 2 timetable={Path(args.step2_timetable_csv).expanduser().resolve()}")
    _log(log_path, f"Step 2 row state={Path(args.step2_row_state_root).expanduser().resolve()}")

    prepared_physical_shifts_csv = _prepare_step3_transfer_inputs(
        prepare=bool(int(args.prepare_transfer_inputs)),
        force=bool(int(args.force_prepare_transfer_inputs)),
        years=years,
        timetable_csv=Path(args.step2_timetable_csv).expanduser().resolve(),
        stations_csv=Path(args.stations_csv).expanduser().resolve(),
        transfer_2026=Path(args.transfer_2026).expanduser().resolve(),
        transfer_2035=Path(args.transfer_2035).expanduser().resolve(),
        output_dir=Path(args.prepared_input_dir).expanduser().resolve(),
        threshold_pax=float(args.transfer_threshold_pax),
        max_shift_min=max(1, int(args.max_shift_min)),
        near_miss_before_min=max(0, int(args.near_miss_before_min)),
        near_miss_after_min=max(0, int(args.near_miss_after_min)),
        fallback_physical_shifts_csv=Path(args.physical_shifts_csv).expanduser().resolve(),
        log_path=log_path,
    )

    prior_rows, prior_ids, prior_event_keys = _load_prior_accepted(
        Path(args.step2_accepted_csv).expanduser().resolve(),
        year_set,
    )
    _log(log_path, f"Loaded prior Step 1/2 accepted rows={len(prior_rows)} ids={len(prior_ids)}")

    physical_rows = _read_csv(prepared_physical_shifts_csv)
    _log(log_path, f"Loaded Step 3 physical-shift source rows={len(physical_rows)} source={prepared_physical_shifts_csv}")
    candidates, rejected_pool = _build_step3_candidate_pool(
        physical_rows=physical_rows,
        years=year_set,
        prior_ids=prior_ids,
        prior_event_keys=prior_event_keys,
        max_shift_min=max(1, int(args.max_shift_min)),
        expand_deltas=bool(int(args.expand_deltas)),
        block_prior_event_keys=bool(int(args.block_step2_event_keys)),
        reject_static_breakage=bool(int(args.reject_static_breakage)),
        min_static_net=float(args.min_static_net),
        min_static_gross_gain=float(args.min_static_gross_gain),
        max_static_loss_ratio=float(args.max_static_loss_ratio),
        max_candidate_pool=max(0, int(args.max_candidate_pool)),
    )
    _log(log_path, f"Generated Step 3 candidate variants={len(candidates)} rejected_pool={len(rejected_pool)}")

    station_min_transfer = _load_station_min_transfer(Path(args.stations_csv).expanduser().resolve())
    timetable_rows = _load_rows(Path(args.step2_timetable_csv).expanduser().resolve())
    year_col_indices = _discover_year_column_indices(timetable_rows[0])
    segments = _parse_segments(timetable_rows)
    row_to_segment = {row_idx: segment for segment in segments for row_idx in segment.row_indices}
    working_rows = timetable_rows

    reference_path_cache: list[dict[str, Any]] = []
    if int(args.use_reference_path_cache):
        reference_path_cache = _build_reference_path_cache(
            years=years,
            rows_root=Path(args.step2_row_state_root).expanduser().resolve(),
            candidates=candidates,
            station_min_transfer=station_min_transfer,
            log_path=log_path,
            progress_every_rows=max(1, int(args.progress_every_rows)),
            scan_row_limit=max(0, int(args.scan_row_limit)),
        )

    accepted: list[dict[str, Any]] = []
    rejected_dynamic: list[dict[str, Any]] = list(rejected_pool)
    latest_scores: list[dict[str, Any]] = []
    used_ids: set[str] = set(prior_ids)
    used_event_keys: set[tuple[str, str, str, str, int]] = set(prior_event_keys) if int(args.block_step2_event_keys) else set()
    current_combined_saved = 0.0
    current_combined_summary: dict[str, Any] = {}
    remaining_by_id: dict[str, dict[str, Any]] = {_physical_id(row): row for row in candidates}

    iteration = 0
    stop_reason = ""
    while remaining_by_id:
        iteration += 1
        if int(args.max_iterations) > 0 and len(accepted) >= int(args.max_iterations):
            stop_reason = "max_iterations_reached"
            break
        if time.time() - start_time >= time_limit_seconds:
            stop_reason = "time_limit_reached_before_next_scan"
            break

        remaining = list(remaining_by_id.values())
        _log(log_path, f"Iteration {iteration}: scoring remaining candidate variants={len(remaining)} accepted_so_far={len(accepted)}")
        if reference_path_cache:
            scores = _score_candidates_from_reference_cache(
                candidates=remaining,
                accepted_rows=accepted,
                path_cache=reference_path_cache,
                station_min_transfer=station_min_transfer,
                log_path=log_path,
            )
        else:
            scores = _scan_marginal_candidate_scores(
                years=years,
                rows_root=Path(args.step2_row_state_root).expanduser().resolve(),
                candidates=remaining,
                accepted_rows=accepted,
                station_min_transfer=station_min_transfer,
                log_path=log_path,
                progress_every_rows=max(1, int(args.progress_every_rows)),
                scan_row_limit=max(0, int(args.scan_row_limit)),
            )

        scored = [
            _merge_candidate_score(
                row,
                scores[_physical_id(row)],
                iteration,
                near_miss_weight=float(args.near_miss_tiebreak_weight),
            )
            for row in remaining
        ]
        scored.sort(
            key=lambda row: (
                -_safe_float(row.get("step3_selection_score"), 0.0),
                -_safe_float(row.get("marginal_reference_pax_minutes_saved"), 0.0),
                -_safe_float(row.get("marginal_reference_weight_touched"), 0.0),
                abs(_safe_int(row.get("delta_min"), 0)),
                str(row.get("year", "")),
                str(row.get("physical_shift_id", "")),
            )
        )
        latest_scores = scored

        accepted_this_iteration = False
        best_rejection_reason = ""
        for rank, row in enumerate(scored, start=1):
            eligible, reason = _candidate_is_eligible(
                row,
                min_selection_score=float(args.min_marginal_pax_minutes),
                min_exact_reference_saved=float(args.min_exact_reference_pax_minutes),
                max_invalidated_daily_pax=float(args.max_invalidated_daily_pax),
                used_ids=used_ids,
                used_event_keys=used_event_keys,
            )
            if not eligible:
                if not best_rejection_reason:
                    best_rejection_reason = reason
                if reason in {"already_selected_or_previously_used", "overlaps_already_selected_event_key"}:
                    rejected_dynamic.append({"status": "rejected_iteration", "reason": reason, **row})
                    remaining_by_id.pop(str(row.get("physical_shift_id", "")), None)
                continue

            ok, trial_rows, info = _try_apply_candidate(
                raw_row=row,
                rank=len(prior_rows) + len(accepted) + 1,
                rows=working_rows,
                segments=segments,
                row_to_segment=row_to_segment,
                year_col_indices=year_col_indices,
            )
            if not ok:
                rejected_dynamic.append({"status": "rejected_apply", "reason": str(info.get("reason", info.get("status", ""))), **row, **info})
                remaining_by_id.pop(str(row.get("physical_shift_id", "")), None)
                continue

            combined_summary: dict[str, Any] = {}
            combined_saved = current_combined_saved + _safe_float(row.get("marginal_reference_pax_minutes_saved"), 0.0)
            if int(args.combined_check_each_accept):
                _log(log_path, f"Iteration {iteration}: combined reference-path check for candidate rank={rank}")
                if reference_path_cache:
                    combined_summary = _scan_combined_from_reference_cache(
                        accepted_rows=accepted + [row],
                        path_cache=reference_path_cache,
                        station_min_transfer=station_min_transfer,
                        log_path=log_path,
                    )
                else:
                    combined_summary = _scan_whole_scenario_existing_paths(
                        years=years,
                        rows_root=Path(args.step2_row_state_root).expanduser().resolve(),
                        accepted_rows=accepted + [row],
                        station_min_transfer=station_min_transfer,
                        log_path=log_path,
                        progress_every_rows=max(1, int(args.progress_every_rows)),
                        scan_row_limit=max(0, int(args.scan_row_limit)),
                    )
                combined_saved = _safe_float(combined_summary.get("current_route_exact_pax_minutes_saved"), 0.0)
                invalidated_pax = _safe_float(combined_summary.get("current_route_exact_invalidated_daily_pax"), 0.0)
                if invalidated_pax > float(args.max_invalidated_daily_pax):
                    rejected_dynamic.append(
                        {
                            "status": "rejected_combined_check",
                            "reason": "combined_reference_paths_invalidated",
                            "combined_invalidated_daily_pax": invalidated_pax,
                            **row,
                        }
                    )
                    remaining_by_id.pop(str(row.get("physical_shift_id", "")), None)
                    continue
                combined_increment = combined_saved - current_combined_saved
                near_increment = _safe_float(row.get("step3_near_miss_score_pax_minutes"), 0.0)
                combined_selection_increment = combined_increment + near_increment
                if combined_increment < float(args.min_exact_reference_pax_minutes):
                    rejected_dynamic.append(
                        {
                            "status": "rejected_combined_check",
                            "reason": "combined_exact_reference_effect_below_floor",
                            "combined_reference_increment_pax_minutes": combined_increment,
                            "previous_combined_reference_pax_minutes_saved": current_combined_saved,
                            **row,
                        }
                    )
                    remaining_by_id.pop(str(row.get("physical_shift_id", "")), None)
                    continue
                if combined_selection_increment < float(args.min_marginal_pax_minutes):
                    rejected_dynamic.append(
                        {
                            "status": "rejected_combined_check",
                            "reason": "combined_selection_score_below_threshold",
                            "combined_reference_increment_pax_minutes": combined_increment,
                            "combined_near_miss_score_pax_minutes": near_increment,
                            "combined_selection_increment_pax_minutes": combined_selection_increment,
                            "previous_combined_reference_pax_minutes_saved": current_combined_saved,
                            **row,
                        }
                    )
                    remaining_by_id.pop(str(row.get("physical_shift_id", "")), None)
                    continue

            accepted_row = {
                "step3_acceptance_rank": len(accepted) + 1,
                "status": "accepted_step3_suggestion",
                "apply_status": str(info.get("status", "")),
                "apply_reason": str(info.get("reason", "")),
                "matched_cells": int(info.get("matched_cells", 0)),
                "changed_cells": int(info.get("changed_cells", 0)),
                "combined_reference_pax_minutes_saved_after_acceptance": _round(combined_saved),
                "combined_reference_pax_minutes_saved_before_acceptance": _round(current_combined_saved),
                "combined_reference_increment_pax_minutes": _round(combined_saved - current_combined_saved),
                **row,
            }
            accepted.append(accepted_row)
            working_rows = trial_rows
            current_combined_saved = combined_saved
            current_combined_summary = combined_summary
            pid = str(row.get("physical_shift_id", ""))
            used_ids.add(pid)
            used_event_keys.update(_affected_keys(row))
            remaining_by_id.pop(pid, None)
            for other_pid, other in list(remaining_by_id.items()):
                if set(_affected_keys(other)) & used_event_keys:
                    rejected_dynamic.append({"status": "rejected_after_acceptance", "reason": "overlaps_newly_accepted_event_key", **other})
                    remaining_by_id.pop(other_pid, None)
            _log(
                log_path,
                "Accepted Step 3 suggestion "
                f"rank={len(accepted)} year={row.get('year')} line={row.get('line')} "
                f"delta={row.get('delta_min')} section={row.get('section_from_station')}->{row.get('section_to_station')} "
                f"marginal_saved={row.get('marginal_reference_pax_minutes_saved')} combined_saved={_fmt(combined_saved, 3)}",
            )
            accepted_this_iteration = True
            break

        if not accepted_this_iteration:
            if scored:
                best = scored[0]
                stop_reason = (
                    "no_compatible_candidate_above_threshold"
                    f"; best_saved={best.get('marginal_reference_pax_minutes_saved')}"
                    f"; best_invalidated={best.get('marginal_reference_invalidated_daily_pax')}"
                    f"; first_rejection={best_rejection_reason}"
                )
            else:
                stop_reason = "no_remaining_candidates"
            break

    if not stop_reason:
        stop_reason = "candidate_pool_exhausted"

    _log(log_path, f"Step 3 search stopping: {stop_reason}")
    if accepted and not current_combined_summary:
        if reference_path_cache:
            current_combined_summary = _scan_combined_from_reference_cache(
                accepted_rows=accepted,
                path_cache=reference_path_cache,
                station_min_transfer=station_min_transfer,
                log_path=log_path,
            )
        else:
            current_combined_summary = _scan_whole_scenario_existing_paths(
                years=years,
                rows_root=Path(args.step2_row_state_root).expanduser().resolve(),
                accepted_rows=accepted,
                station_min_transfer=station_min_transfer,
                log_path=log_path,
                progress_every_rows=max(1, int(args.progress_every_rows)),
                scan_row_limit=max(0, int(args.scan_row_limit)),
            )

    suggested_path = work_dir / "Step3_SuggestedTimetableChanges_2026_2035.csv"
    rejected_path = work_dir / "Step3_RejectedOrDeferredCandidates_2026_2035.csv"
    latest_path = work_dir / "Step3_LatestMarginalCandidateScores_2026_2035.csv"
    preview_path = work_dir / "FullTimetableHistory2026_2035_STEP3_preview.csv"
    summary_path = work_dir / "Step3_summary.json"

    preferred_fields = [
        "step3_acceptance_rank",
        "status",
        "year",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "physical_shift_id",
        "step3_source_physical_shift_id",
        "marginal_reference_pax_minutes_saved",
        "marginal_reference_weight_touched",
        "marginal_reference_paths_touched",
        "marginal_reference_invalidated_daily_pax",
        "marginal_reference_mean_total_delta_min",
        "combined_reference_increment_pax_minutes",
        "combined_reference_pax_minutes_saved_after_acceptance",
        "candidate_stations",
        "matched_cells",
        "changed_cells",
        "marginal_example_od",
    ]
    _write_rows_union(suggested_path, accepted, preferred_fields)
    _write_rows_union(rejected_path, rejected_dynamic, ["status", "reason", "year", "line", "delta_min", "physical_shift_id"])
    _write_rows_union(latest_path, latest_scores, preferred_fields)
    _write_rows(preview_path, working_rows)

    accepted_by_year = Counter(str(row.get("year", "")) for row in accepted)
    accepted_by_line = Counter(f"{row.get('year')}|{row.get('line')}" for row in accepted)
    summary = {
        "created_at": _now(),
        "years": years,
        "stop_reason": stop_reason,
        "runtime_seconds": round(time.time() - start_time, 3),
        "input_step2_timetable_csv": str(Path(args.step2_timetable_csv).expanduser().resolve()),
        "input_step2_row_state_root": str(Path(args.step2_row_state_root).expanduser().resolve()),
        "prepared_transfer_threshold_pax": float(args.transfer_threshold_pax),
        "prepared_transfer_input_dir": str(Path(args.prepared_input_dir).expanduser().resolve()),
        "physical_shift_source_csv": str(prepared_physical_shifts_csv),
        "candidate_variants_scored_initially": len(candidates),
        "accepted_step3_suggestions": len(accepted),
        "accepted_by_year": dict(sorted(accepted_by_year.items())),
        "accepted_by_line_year": dict(sorted(accepted_by_line.items())),
        "min_step3_selection_score": float(args.min_marginal_pax_minutes),
        "min_exact_reference_pax_minutes": float(args.min_exact_reference_pax_minutes),
        "near_miss_score_weight": float(args.near_miss_tiebreak_weight),
        "max_invalidated_daily_pax": float(args.max_invalidated_daily_pax),
        "combined_check_each_accept": bool(int(args.combined_check_each_accept)),
        "use_reference_path_cache": bool(int(args.use_reference_path_cache)),
        "reference_path_cache_records": len(reference_path_cache),
        "final_combined_reference_scan": current_combined_summary,
        "output_suggested_csv": str(suggested_path),
        "output_rejected_csv": str(rejected_path),
        "output_latest_scores_csv": str(latest_path),
        "output_preview_timetable_csv": str(preview_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(log_path, f"Wrote suggestions: {suggested_path}")
    _log(log_path, f"Wrote preview timetable: {preview_path}")
    _log(log_path, f"Wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
