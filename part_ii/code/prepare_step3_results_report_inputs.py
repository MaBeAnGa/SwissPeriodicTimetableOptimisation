#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PART2_ROOT = Path(__file__).resolve().parent
if str(PART2_ROOT) not in sys.path:
    sys.path.insert(0, str(PART2_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_optimized_timetable_adjustment_effects import (  # noqa: E402
    Agg,
    Proposal,
    _cell_metrics_and_hits as audit_cell_metrics_and_hits,
    _fmt as audit_fmt,
    _proposal_index,
)
from build_step3_analysis_from_row_states import (  # noqa: E402
    METRICS,
    _aggregate_variant,
    _as_float,
    _cell_metrics,
    _finalize_acc,
    _load_station_meta,
    _spatial_rows,
    _station_order,
)

DEFAULT_RUN_ROOT = Path.home() / "od_fast_state" / "step3_local_outputs" / "STEP3_BUILD_LOCAL_FINAL"
DEFAULT_SURGERY_ROOT = (
    DEFAULT_RUN_ROOT
    / "reversal_surgery_preparation"
    / "ic5_ic82_baseline_step3_only"
    / "cell_surgery_20260617_090934"
)
DEFAULT_V0_ROOT = Path.home() / "od_fast_state" / "nondominated_cells_state_v0_restored_from_pathalloc_csv"
DEFAULT_STEP3_ROOT = DEFAULT_RUN_ROOT / "row_states" / "nondominated_cells_state_STEP3_FINAL"
DEFAULT_PROPOSALS = DEFAULT_RUN_ROOT / "analysis" / "step3_final_accepted_proposals_for_attribution_audit.csv"
DEFAULT_STATIONS = PART2_ROOT / "FilteredStations2026_2035.csv"
DEFAULT_MUNICIPALITIES = PROJECT_ROOT / "swiss_municipality_station_geometries.json"
DEFAULT_OUTPUT_DIR = DEFAULT_SURGERY_ROOT / "report_inputs_step3_corrected"
YEARS = ("2026", "2035")
EPS = 1e-9


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def parse_cell(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(str(raw or ""))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _path_in_system_cost(path: dict[str, Any]) -> float:
    rolling = _as_float(path.get("rolling_min"), float("nan"))
    dwell = _as_float(path.get("dwell_min"), float("nan"))
    transfer = _as_float(path.get("transfer_min"), float("nan"))
    total = rolling + dwell + transfer
    return total if math.isfinite(total) and total > 0.0 else float("nan")


def _dominance_beta_by_path(paths: list[dict[str, Any]], *, cycle: int = 120, day_start: int = 300, day_end: int = 1440) -> dict[int, float]:
    instances: list[tuple[float, int]] = []
    for path_idx, path in enumerate(paths):
        dep = _as_float(path.get("dep_min"), float("nan"))
        cost = _path_in_system_cost(path)
        if not math.isfinite(dep) or not math.isfinite(cost):
            continue
        n = 0
        while dep + n * cycle < day_end:
            u = dep + n * cycle
            if day_start <= u < day_end and u + cost < day_end:
                instances.append((u, path_idx))
            n += 1
            if n > 20:
                break
    if not instances:
        return {}
    instances.sort(key=lambda item: item[0])
    beta = {idx: 0.0 for idx, _path in enumerate(paths)}
    prev = float(day_start)
    for u, path_idx in instances:
        if u > prev:
            beta[path_idx] = beta.get(path_idx, 0.0) + (u - prev)
        prev = max(prev, u)
    return beta


def _cell_metrics_with_manifest_demand(raw: Any, demand: float) -> dict[str, float] | None:
    cell = parse_cell(raw)
    if not cell:
        return None
    paths = cell.get("paths")
    if not isinstance(paths, list) or not paths:
        return None
    demand = max(_as_float(demand), 0.0)
    if demand <= 0.0:
        return None
    path_dicts = [p for p in paths if isinstance(p, dict) and math.isfinite(_path_in_system_cost(p))]
    if not path_dicts:
        return None
    beta_all = _dominance_beta_by_path(paths)
    beta = {idx: beta_all.get(idx, 0.0) for idx, p in enumerate(paths) if isinstance(p, dict) and math.isfinite(_path_in_system_cost(p))}
    beta_sum = sum(max(v, 0.0) for v in beta.values())
    equal_weight = demand / max(len(path_dicts), 1)
    weight_sum = 0.0
    rolling_sum = 0.0
    dwell_sum = 0.0
    transfer_sum = 0.0
    total_sum = 0.0
    for idx, path in enumerate(paths):
        if not isinstance(path, dict):
            continue
        total = _path_in_system_cost(path)
        if not math.isfinite(total):
            continue
        if beta_sum > 0.0:
            weight = demand * max(beta.get(idx, 0.0), 0.0) / beta_sum
        else:
            weight = equal_weight
        if weight <= 0.0:
            continue
        rolling = _as_float(path.get("rolling_min"), 0.0)
        dwell = _as_float(path.get("dwell_min"), 0.0)
        transfer = _as_float(path.get("transfer_min"), 0.0)
        weight_sum += weight
        rolling_sum += rolling * weight
        dwell_sum += dwell * weight
        transfer_sum += transfer * weight
        total_sum += total * weight
    if weight_sum <= 0.0:
        return None
    return {
        "weight": weight_sum,
        "rolling": rolling_sum / weight_sum,
        "dwell": dwell_sum / weight_sum,
        "transfer": transfer_sum / weight_sum,
        "total": total_sum / weight_sum,
    }


def row_state_path(root: Path, year: str, row_idx: int) -> Path:
    return root.expanduser().resolve() / year / "rows" / f"row_{row_idx:06d}.json"


def require_inputs(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s):\n" + "\n".join(missing))


def run_command(args: list[str], *, cwd: Path = PART2_ROOT) -> None:
    log("Running: " + " ".join(str(a) for a in args))
    subprocess.run([str(a) for a in args], cwd=str(cwd), check=True)


def python_exe() -> str:
    return sys.executable or "<USER_HOME>/opt/miniforge3/bin/python3"


def done_or_force(path: Path, force: bool, label: str) -> bool:
    if path.exists() and not force:
        log(f"Skipping {label}; output already exists: {path}")
        return True
    return False


def run_attribution_audit_v2(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir / "edit_attribution_step3_corrected_v2"
    by_change = out_dir / "TimetableAdjustmentEffectAudit_by_change.csv"
    by_change_v2 = out_dir / "TimetableAdjustmentEffectAudit_by_change_v2.csv"
    if done_or_force(by_change_v2, args.force, "Step 3 attribution audit v2"):
        return by_change_v2
    if args.force and out_dir.exists():
        shutil.rmtree(out_dir)
    require_inputs([args.proposals, args.v0_state_root / "2026" / "metadata.json", args.v0_state_root / "2035" / "metadata.json"])
    run_command(
        [
            python_exe(),
            PART2_ROOT / "audit_optimized_timetable_adjustment_effects.py",
            "--proposals",
            args.proposals,
            "--v0-row-state-root",
            args.v0_state_root,
            "--optimized-row-state-root",
            args.step3_state_root,
            "--output-dir",
            out_dir,
            "--years",
            "2026,2035",
            "--progress-every-rows",
            str(args.progress_every_rows),
            "--example-limit",
            "3",
        ]
    )
    if not by_change.exists():
        raise FileNotFoundError(f"Audit did not produce {by_change}")
    shutil.copy2(by_change, by_change_v2)
    rows = read_csv(by_change_v2)
    ids = [row.get("proposal_id", "") for row in rows]
    if len(ids) != len(set(ids)):
        dupes = sorted({x for x in ids if ids.count(x) > 1})[:10]
        raise RuntimeError(f"Step 3 attribution audit still has duplicate proposal_id values: {dupes}")
    log(f"Wrote Step 3 attribution audit v2: {by_change_v2}")
    return by_change_v2


def parse_mod120(time_string: str) -> int | None:
    try:
        h, m = str(time_string).strip().split(":", 1)
        return (int(h) * 60 + int(m)) % 120
    except Exception:
        return None


def reversal_proposals() -> list[Proposal]:
    return [
        Proposal(
            proposal_id="2035|revert|IC 82|+1|Zofingen|01:31|Bern|01:59",
            selection_order=1,
            year="2035",
            line="IC 82",
            delta_min=1,
            from_station="Zofingen",
            to_station="Bern",
            dep_old_mod=parse_mod120("01:31"),
            arr_old_mod=parse_mod120("01:59"),
            dep_new_mod=parse_mod120("01:32"),
            arr_new_mod=parse_mod120("00:00"),
            marginal_net_pax_minutes=0.0,
            marginal_gross_gain_pax_minutes=0.0,
            marginal_gross_loss_pax_minutes=0.0,
            action="revert_step1_shift",
        ),
        Proposal(
            proposal_id="2035|revert|IC 5|+2|Neuchâtel|01:25|Biel/Bienne|01:43",
            selection_order=2,
            year="2035",
            line="IC 5",
            delta_min=2,
            from_station="Neuchâtel",
            to_station="Biel/Bienne",
            dep_old_mod=parse_mod120("01:25"),
            arr_old_mod=parse_mod120("01:43"),
            dep_new_mod=parse_mod120("01:27"),
            arr_new_mod=parse_mod120("01:45"),
            marginal_net_pax_minutes=0.0,
            marginal_gross_gain_pax_minutes=0.0,
            marginal_gross_loss_pax_minutes=0.0,
            action="revert_step2_shift",
        ),
    ]


def split_semicolon_values(value: Any) -> list[str]:
    return [part.strip() for part in str(value or "").split(";") if part.strip()]


def _reversal_report_keys(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    candidates_path = args.run_root / "reversal_surgery_preparation" / "ic5_ic82_baseline_step3_only" / "step3_reversal_candidates.csv"
    require_inputs([candidates_path])
    out: dict[str, dict[str, str]] = {}
    for row in read_csv(candidates_path):
        key = row.get("reversal_key", "")
        if key:
            out[key] = row
    return out


def run_reversal_attribution(args: argparse.Namespace) -> Path:
    out_path = args.output_dir / "step3_reversal_edit_attribution.csv"
    if done_or_force(out_path, args.force, "Step 3 reversal attribution"):
        return out_path
    backup_dir = args.surgery_root / "backup_original_step3_rows_2035"
    manifest_path = args.run_root / "reversal_surgery_preparation" / "ic5_ic82_baseline_step3_only" / "step3_reversal_impacted_od_pairs.csv"
    require_inputs([backup_dir, args.step3_state_root / "2035" / "rows", args.v0_state_root / "2035" / "metadata.json", manifest_path])
    reversal_rows = _reversal_report_keys(args)
    order = _station_order(args.v0_state_root, "2035")

    # The grey Step 3 revert rows should describe the marginal effect of the
    # completed surgery: pre-reversal Step 3 row states versus corrected Step 3
    # row states, restricted to the OD cells selected by the reversal manifest.
    cells_by_row: dict[int, dict[str, dict[int, float]]] = {}
    manifest_seen: set[tuple[int, int, str]] = set()
    for row in read_csv(manifest_path):
        try:
            row_idx = int(row.get("row_idx", ""))
            dest_idx = int(row.get("dest_idx", ""))
        except Exception:
            continue
        for reversal_key in split_semicolon_values(row.get("matched_reversals")):
            token = (row_idx, dest_idx, reversal_key)
            if token in manifest_seen:
                continue
            manifest_seen.add(token)
            try:
                demand = max(_as_float(row.get("daily_demand_total_last_seen")), 0.0)
            except Exception:
                demand = 0.0
            by_dest = cells_by_row.setdefault(row_idx, {}).setdefault(reversal_key, {})
            by_dest[dest_idx] = max(by_dest.get(dest_idx, 0.0), demand)

    stats: dict[str, dict[str, float]] = {}
    for reversal_key in sorted({key for by_key in cells_by_row.values() for key in by_key} | set(reversal_rows)):
        stats[reversal_key] = {
            "daily_pax_weight": 0.0,
            "affected_od_pairs": 0.0,
            "changed_od_pairs": 0.0,
            "improved_od_pairs": 0.0,
            "worsened_od_pairs": 0.0,
            "unchanged_od_pairs": 0.0,
            "pre_reversal_total_weighted_min": 0.0,
            "corrected_step3_total_weighted_min": 0.0,
            "total_pax_minutes_delta_corrected_minus_pre_reversal": 0.0,
        }

    row_items = sorted(cells_by_row.items())
    log(f"Reversal attribution scanning {len(row_items)} backed-up 2035 rows and {len(manifest_seen)} manifest OD-cell links")
    for counter, (row_idx, reversal_map) in enumerate(row_items, start=1):
        before_path = backup_dir / f"row_{row_idx:06d}.json"
        if not before_path.exists():
            continue
        after_path = row_state_path(args.step3_state_root, "2035", row_idx)
        before_payload = read_json(before_path)
        after_payload = read_json(after_path)
        before_cells = before_payload.get("cells") or []
        after_cells = after_payload.get("cells") or []
        for reversal_key, dest_map in reversal_map.items():
            acc = stats.setdefault(reversal_key, {})
            for dest_idx, manifest_demand in sorted(dest_map.items()):
                if dest_idx >= len(order) or dest_idx >= len(before_cells) or dest_idx >= len(after_cells):
                    continue
                before_metrics = _cell_metrics_with_manifest_demand(before_cells[dest_idx], manifest_demand)
                after_metrics = _cell_metrics_with_manifest_demand(after_cells[dest_idx], manifest_demand)
                if before_metrics is None or after_metrics is None:
                    continue
                weight = max(_as_float(before_metrics.get("weight")), _as_float(after_metrics.get("weight")))
                if weight <= 0.0:
                    continue
                before_total = _as_float(before_metrics.get("total"))
                after_total = _as_float(after_metrics.get("total"))
                delta_total = after_total - before_total
                acc["daily_pax_weight"] = acc.get("daily_pax_weight", 0.0) + weight
                acc["affected_od_pairs"] = acc.get("affected_od_pairs", 0.0) + 1.0
                acc["pre_reversal_total_weighted_min"] = acc.get("pre_reversal_total_weighted_min", 0.0) + before_total * weight
                acc["corrected_step3_total_weighted_min"] = acc.get("corrected_step3_total_weighted_min", 0.0) + after_total * weight
                acc["total_pax_minutes_delta_corrected_minus_pre_reversal"] = (
                    acc.get("total_pax_minutes_delta_corrected_minus_pre_reversal", 0.0) + delta_total * weight
                )
                if abs(delta_total) <= EPS:
                    acc["unchanged_od_pairs"] = acc.get("unchanged_od_pairs", 0.0) + 1.0
                else:
                    acc["changed_od_pairs"] = acc.get("changed_od_pairs", 0.0) + 1.0
                    if delta_total < 0.0:
                        acc["improved_od_pairs"] = acc.get("improved_od_pairs", 0.0) + 1.0
                    else:
                        acc["worsened_od_pairs"] = acc.get("worsened_od_pairs", 0.0) + 1.0
        if counter % max(1, args.progress_every_rows) == 0 or counter == len(row_items):
            log(f"Reversal attribution progress rows={counter}/{len(row_items)}")

    rows: list[dict[str, Any]] = []
    for reversal_key, acc in sorted(stats.items()):
        meta = reversal_rows.get(reversal_key, {})
        weight = acc.get("daily_pax_weight", 0.0)
        pre_avg = acc.get("pre_reversal_total_weighted_min", 0.0) / weight if weight > 0.0 else 0.0
        corrected_avg = acc.get("corrected_step3_total_weighted_min", 0.0) / weight if weight > 0.0 else 0.0
        delta_total = corrected_avg - pre_avg if weight > 0.0 else 0.0
        total_pax_delta = acc.get("total_pax_minutes_delta_corrected_minus_pre_reversal", 0.0)
        rows.append(
            {
                "reversal_key": reversal_key,
                "proposal_id": meta.get("physical_shift_id", f"reversal|{reversal_key}"),
                "source_physical_shift_id": meta.get("source_physical_shift_id", ""),
                "year": meta.get("year", "2035"),
                "action": meta.get("action", "reverse_inherited_edit"),
                "line": meta.get("line", ""),
                "delta_min": meta.get("delta_min", ""),
                "section_from_station": meta.get("section_from_station", ""),
                "section_departure_time_mod120": meta.get("section_departure_time_mod120", ""),
                "section_departure_time_new_mod120": meta.get("section_departure_time_new_mod120", ""),
                "section_to_station": meta.get("section_to_station", ""),
                "section_arrival_time_mod120": meta.get("section_arrival_time_mod120", ""),
                "section_arrival_time_new_mod120": meta.get("section_arrival_time_new_mod120", ""),
                "affected_od_pairs": int(acc.get("affected_od_pairs", 0.0)),
                "changed_od_pairs": int(acc.get("changed_od_pairs", 0.0)),
                "improved_od_pairs": int(acc.get("improved_od_pairs", 0.0)),
                "worsened_od_pairs": int(acc.get("worsened_od_pairs", 0.0)),
                "unchanged_od_pairs": int(acc.get("unchanged_od_pairs", 0.0)),
                "affected_demand_pax_day": audit_fmt(weight, 6),
                "pre_reversal_total_min": audit_fmt(pre_avg, 6),
                "corrected_step3_total_min": audit_fmt(corrected_avg, 6),
                "delta_total_min": audit_fmt(delta_total, 6),
                "attributed_saving_pax_min_day": audit_fmt(-total_pax_delta, 6),
                "total_pax_minutes_delta_corrected_minus_pre_reversal": audit_fmt(total_pax_delta, 6),
                "status": "ok" if weight > 0.0 else "no_positive_demand_cells",
            }
        )
    write_csv(out_path, rows)
    log(f"Wrote Step 3 reversal attribution from surgery manifest: {out_path}")
    return out_path


def load_spatial_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return read_csv(path)


def run_spatial_deltas(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    delta_path = args.output_dir / "step3_spatial_delta_by_origin.csv"
    extremes_path = args.output_dir / "step3_spatial_extremes.csv"
    counts_path = args.output_dir / "step3_spatial_delta_counts.csv"
    if all(p.exists() for p in (delta_path, extremes_path, counts_path)) and not args.force:
        log(f"Skipping spatial deltas; outputs already exist in {args.output_dir}")
        return delta_path, extremes_path, counts_path
    step3_spatial_path = args.surgery_root / "analysis_step3_only_corrected" / "step3_spatial_breakdown_data.csv"
    require_inputs([step3_spatial_path, args.stations_csv, args.municipalities_json])
    meta = _load_station_meta(args.stations_csv, args.municipalities_json)
    base_rows: list[dict[str, Any]] = []
    for year in YEARS:
        order = _station_order(args.v0_state_root, year)
        base_variant = _aggregate_variant(args.v0_state_root, year, order)
        base_rows.extend(_spatial_rows(year, "v0", base_variant["stationStats"], meta))
    base_path = args.output_dir / "step3_spatial_baseline_by_origin.csv"
    write_csv(base_path, base_rows)
    step3_rows = load_spatial_existing(step3_spatial_path)
    base_by_key = {(r["year"], r["level"], r["code"]): r for r in base_rows}
    step3_by_key = {(r["year"], r["level"], r["code"]): r for r in step3_rows}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(base_by_key) & set(step3_by_key)):
        b = base_by_key[key]
        s = step3_by_key[key]
        weight = max(_as_float(b.get("totalWeight")), _as_float(s.get("totalWeight")))
        row = {
            "year": key[0],
            "level": key[1],
            "code": key[2],
            "name": s.get("name") or b.get("name"),
            "canton": s.get("canton") or b.get("canton"),
            "originCount": s.get("originCount") or b.get("originCount"),
            "totalWeight": weight,
        }
        for metric, col in (
            ("total", "avgTotalMin"),
            ("rolling", "avgRollingMin"),
            ("dwell", "avgDwellMin"),
            ("transfer", "avgTransferMin"),
            ("waiting", "avgWaitingMin"),
        ):
            bv = _as_float(b.get(col), float("nan"))
            sv = _as_float(s.get(col), float("nan"))
            delta = sv - bv
            row[f"baseline_{metric}_min"] = bv
            row[f"step3_{metric}_min"] = sv
            row[f"delta_{metric}_min"] = delta
            row[f"delta_{metric}_sec"] = delta * 60.0
            row[f"delta_{metric}_pax_min_day"] = delta * weight if math.isfinite(delta) else ""
        rows.append(row)
    write_csv(delta_path, rows)
    threshold_sec = float(args.spatial_unchanged_threshold_seconds)
    count_rows: list[dict[str, Any]] = []
    extreme_rows: list[dict[str, Any]] = []
    for year in YEARS:
        for level in ("canton", "municipality", "station"):
            group = [r for r in rows if r["year"] == year and r["level"] == level]
            improved = [r for r in group if _as_float(r.get("delta_total_sec")) < -threshold_sec]
            worsened = [r for r in group if _as_float(r.get("delta_total_sec")) > threshold_sec]
            unchanged = len(group) - len(improved) - len(worsened)
            count_rows.append(
                {
                    "year": year,
                    "level": level,
                    "groups": len(group),
                    "improved_groups": len(improved),
                    "worsened_groups": len(worsened),
                    "unchanged_groups": unchanged,
                    "unchanged_threshold_sec": threshold_sec,
                }
            )
            for label, subset, reverse in (
                ("largest_improvements", group, False),
                ("largest_deteriorations", group, True),
            ):
                sorted_subset = sorted(subset, key=lambda r: _as_float(r.get("delta_total_sec")), reverse=reverse)
                for rank, r in enumerate(sorted_subset[:10], start=1):
                    extreme_rows.append(
                        {
                            "year": year,
                            "level": level,
                            "ranking": label,
                            "rank": rank,
                            "code": r.get("code"),
                            "name": r.get("name"),
                            "canton": r.get("canton"),
                            "delta_total_sec": r.get("delta_total_sec"),
                            "delta_total_pax_min_day": r.get("delta_total_pax_min_day"),
                            "totalWeight": r.get("totalWeight"),
                        }
                    )
    write_csv(extremes_path, extreme_rows)
    write_csv(counts_path, count_rows)
    log(f"Wrote spatial Step 3 report inputs: {delta_path}, {extremes_path}, {counts_path}")
    return delta_path, extremes_path, counts_path


def run_segment_load_deltas(args: argparse.Namespace) -> Path:
    out_path = args.output_dir / "step3_segment_load_delta_top.csv"
    if done_or_force(out_path, args.force, "segment-load delta table"):
        return out_path
    baseline_dir = args.output_dir / "segment_loads_baseline_from_row_states"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    for year in YEARS:
        baseline_csv = baseline_dir / f"{year}_BASELINE_adjacent_line_service_pax.csv"
        baseline_db = baseline_dir / f"{year}_BASELINE_adjacent_line_service_pax.sqlite"
        if not baseline_csv.exists():
            run_command(
                [
                    python_exe(),
                    PROJECT_ROOT / "aggregate_adjacent_line_service_pax_from_row_state.py",
                    "--year",
                    year,
                    "--row-state-root",
                    args.v0_state_root,
                    "--output-csv",
                    baseline_csv,
                    "--state-db",
                    baseline_db,
                    "--resume",
                    "--progress-every-rows",
                    str(args.progress_every_rows),
                ],
                cwd=PROJECT_ROOT,
            )
    rows_out: list[dict[str, Any]] = []
    for year in YEARS:
        baseline_csv = baseline_dir / f"{year}_BASELINE_adjacent_line_service_pax.csv"
        step3_csv = args.surgery_root / "segment_loads_step3_only_corrected" / f"{year}_STEP3_ONLY_CORRECTED_adjacent_line_service_pax.csv"
        require_inputs([baseline_csv, step3_csv])
        def norm_station(value: str) -> str:
            station = str(value or "")
            if station.startswith("Zürich HB ("):
                return "Zürich HB"
            return station

        def key(row: dict[str, str]) -> tuple[str, str, str]:
            return (norm_station(row.get("First station", "")), norm_station(row.get("Second station", "")), row.get("Line", ""))
        base: dict[tuple[str, str, str], float] = {}
        step3: dict[tuple[str, str, str], float] = {}
        for row in read_csv(baseline_csv):
            base[key(row)] = base.get(key(row), 0.0) + _as_float(row.get("Daily pax"))
        for row in read_csv(step3_csv):
            step3[key(row)] = step3.get(key(row), 0.0) + _as_float(row.get("Daily pax"))
        all_keys = set(base) | set(step3)
        deltas = []
        for k in all_keys:
            b = base.get(k, 0.0)
            s = step3.get(k, 0.0)
            deltas.append((s - b, k, b, s))
        for rank, (delta, k, b, s) in enumerate(sorted(deltas, key=lambda x: x[0], reverse=True)[:20], start=1):
            rows_out.append({"year": year, "ranking": "largest_load_increases", "rank": rank, "first_station": k[0], "second_station": k[1], "line": k[2], "baseline_daily_pax": b, "step3_daily_pax": s, "delta_daily_pax": delta})
        for rank, (delta, k, b, s) in enumerate(sorted(deltas, key=lambda x: x[0])[:20], start=1):
            rows_out.append({"year": year, "ranking": "largest_load_decreases", "rank": rank, "first_station": k[0], "second_station": k[1], "line": k[2], "baseline_daily_pax": b, "step3_daily_pax": s, "delta_daily_pax": delta})
    write_csv(out_path, rows_out)
    log(f"Wrote segment-load delta table: {out_path}")
    return out_path


@dataclass
class ODExtreme:
    score: float
    row: dict[str, Any]


def cell_paths(raw: Any) -> list[dict[str, Any]]:
    obj = parse_cell(raw)
    if not obj:
        return []
    paths = obj.get("paths")
    return paths if isinstance(paths, list) else []


def rolling_share(metrics: dict[str, float] | None) -> float | None:
    if not metrics:
        return None
    total = _as_float(metrics.get("total"), 0.0)
    if total <= 0:
        return None
    return _as_float(metrics.get("rolling"), 0.0) / total


def make_od_row(year: str, origin: str, destination: str, base: dict[str, float], step3: dict[str, float]) -> dict[str, Any]:
    weight = max(_as_float(base.get("weight")), _as_float(step3.get("weight")))
    saved_total_min = _as_float(base.get("total")) - _as_float(step3.get("total"))
    base_share = rolling_share(base)
    step3_share = rolling_share(step3)
    return {
        "year": year,
        "origin": origin,
        "destination": destination,
        "daily_pax": weight,
        "baseline_total_min": base.get("total"),
        "step3_total_min": step3.get("total"),
        "saved_total_min": saved_total_min,
        "saved_total_sec": saved_total_min * 60.0,
        "saved_pax_min_day": saved_total_min * weight,
        "baseline_rolling_min": base.get("rolling"),
        "step3_rolling_min": step3.get("rolling"),
        "baseline_dwell_min": base.get("dwell"),
        "step3_dwell_min": step3.get("dwell"),
        "baseline_transfer_min": base.get("transfer"),
        "step3_transfer_min": step3.get("transfer"),
        "baseline_waiting_min": base.get("waiting"),
        "step3_waiting_min": step3.get("waiting"),
        "waiting_delta_min": _as_float(step3.get("waiting")) - _as_float(base.get("waiting")),
        "baseline_rolling_share": base_share,
        "step3_rolling_share": step3_share,
        "delta_rolling_share": (step3_share - base_share) if base_share is not None and step3_share is not None else "",
    }


def sorted_top(rows: list[dict[str, Any]], key: str, reverse: bool, n: int = 3) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda r: _as_float(r.get(key), -math.inf if reverse else math.inf), reverse=reverse)[:n]


def run_od_extremes(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    out_2026 = args.output_dir / "step3_od_extremes_2026.csv"
    out_2035 = args.output_dir / "step3_od_extremes_2035.csv"
    rolling_out = args.output_dir / "step3_od_rolling_share_extremes.csv"
    window_out = args.output_dir / "step3_od_path_window_2035.csv"
    if all(p.exists() for p in (out_2026, out_2035, rolling_out, window_out)) and not args.force:
        log("Skipping OD extremes/path-window extraction; outputs already exist")
        return out_2026, out_2035, rolling_out, window_out
    all_extreme_rows_by_year: dict[str, list[dict[str, Any]]] = {}
    rolling_rows_all: list[dict[str, Any]] = []
    selected_window_pairs: list[dict[str, Any]] = []
    for year in YEARS:
        order = _station_order(args.v0_state_root, year)
        candidate_rows: list[dict[str, Any]] = []
        for row_idx, origin in enumerate(order):
            base_payload = read_json(row_state_path(args.v0_state_root, year, row_idx))
            step3_payload = read_json(row_state_path(args.step3_state_root, year, row_idx))
            base_cells = base_payload.get("cells") or []
            step3_cells = step3_payload.get("cells") or []
            for dest_idx, destination in enumerate(order):
                if dest_idx == row_idx:
                    continue
                base_metrics = _cell_metrics(base_cells[dest_idx])
                step3_metrics = _cell_metrics(step3_cells[dest_idx])
                if base_metrics is None or step3_metrics is None:
                    continue
                row = make_od_row(year, origin, destination, base_metrics, step3_metrics)
                if abs(_as_float(row.get("saved_total_min"))) > EPS or abs(_as_float(row.get("delta_rolling_share"))) > EPS:
                    row["origin_row_idx"] = row_idx
                    row["destination_idx"] = dest_idx
                    candidate_rows.append(row)
            if (row_idx + 1) % max(1, args.progress_every_rows) == 0 or row_idx + 1 == len(order):
                log(f"{year}: OD extremes scan rows={row_idx+1}/{len(order)} candidates={len(candidate_rows)}")
        rows_out: list[dict[str, Any]] = []
        categories = [
            ("largest_per_trip_improvements", "saved_total_min", True),
            ("largest_passenger_minute_improvements", "saved_pax_min_day", True),
            ("largest_per_trip_deteriorations", "saved_total_min", False),
            ("largest_passenger_minute_deteriorations", "saved_pax_min_day", False),
        ]
        for category, key, reverse in categories:
            for rank, row in enumerate(sorted_top(candidate_rows, key, reverse, 10), start=1):
                out_row = {"category": category, "rank": rank, **row}
                rows_out.append(out_row)
                if year == "2035" and rank == 1 and category in {
                    "largest_per_trip_improvements",
                    "largest_passenger_minute_improvements",
                    "largest_per_trip_deteriorations",
                    "largest_passenger_minute_deteriorations",
                }:
                    selected_window_pairs.append(out_row)
        for category, reverse in (("largest_rolling_share_increases", True), ("largest_rolling_share_decreases", False)):
            for rank, row in enumerate(sorted_top(candidate_rows, "delta_rolling_share", reverse, 10), start=1):
                rolling_rows_all.append({"category": category, "rank": rank, **row})
        all_extreme_rows_by_year[year] = rows_out
        write_csv(out_2026 if year == "2026" else out_2035, rows_out)
    write_csv(rolling_out, rolling_rows_all)
    # Path-window extraction for the four leading 2035 OD examples.
    window_rows: list[dict[str, Any]] = []
    order_2035 = _station_order(args.v0_state_root, "2035")
    seen_pairs = set()
    for example in selected_window_pairs:
        pair_key = (example["category"], example["origin"], example["destination"])
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        row_idx = int(example["origin_row_idx"])
        dest_idx = int(example["destination_idx"])
        base_payload = read_json(row_state_path(args.v0_state_root, "2035", row_idx))
        step3_payload = read_json(row_state_path(args.step3_state_root, "2035", row_idx))
        for scenario, payload in (("baseline", base_payload), ("optimized_step3", step3_payload)):
            cell_raw = (payload.get("cells") or [])[dest_idx]
            for path_idx, path in enumerate(cell_paths(cell_raw), start=1):
                dep_min = _as_float(path.get("dep_min"), float("nan"))
                if not math.isfinite(dep_min) or not (300 <= dep_min < 420):
                    continue
                dominance = _as_float(path.get("dominance_minutes_day"), 0.0)
                instances = _as_float(path.get("daily_departure_instances"), 0.0)
                window_rows.append(
                    {
                        "category": example["category"],
                        "origin": example["origin"],
                        "destination": example["destination"],
                        "scenario": scenario,
                        "path_index": path_idx,
                        "dep_min": path.get("dep_min"),
                        "dep_time": path.get("dep_time"),
                        "arr_min": path.get("arr_min"),
                        "arr_time": path.get("arr_time"),
                        "travel_time_min": path.get("travel_time_min"),
                        "rolling_min": path.get("rolling_min"),
                        "dwell_min": path.get("dwell_min"),
                        "transfer_min": path.get("transfer_min"),
                        "num_transfers": path.get("num_transfers"),
                        "dominance_minutes_day": dominance,
                        "daily_departure_instances": instances,
                        "daily_demand_share": path.get("daily_demand_share"),
                        "daily_demand_allocated": path.get("daily_demand_allocated"),
                        "line_names": "; ".join(str(x) for x in (path.get("line_names") or [])),
                        "nodes_sequence": " -> ".join(str(x) for x in (path.get("nodes_sequence") or [])),
                        "station_events_json": json.dumps(path.get("station_events") or [], ensure_ascii=False),
                        "transfer_events_json": json.dumps(path.get("transfer_events") or [], ensure_ascii=False),
                        "legs_json": json.dumps(path.get("legs") or [], ensure_ascii=False),
                    }
                )
    write_csv(window_out, window_rows)
    log(f"Wrote OD extreme/path-window report inputs: {out_2026}, {out_2035}, {rolling_out}, {window_out}")
    return out_2026, out_2035, rolling_out, window_out


def write_manifest(args: argparse.Namespace, outputs: list[Path]) -> None:
    manifest = {
        "created_at": now(),
        "run_root": str(args.run_root),
        "surgery_root": str(args.surgery_root),
        "v0_state_root": str(args.v0_state_root),
        "step3_state_root": str(args.step3_state_root),
        "proposals": str(args.proposals),
        "outputs": [str(path) for path in outputs],
    }
    write_json(args.output_dir / "step3_results_report_inputs_manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the remaining Step 3 Results II report inputs from corrected row-state JSONs: "
            "unique edit attribution, reversal attribution, spatial deltas, segment-load deltas, "
            "OD extremes, rolling-share extremes, and path-window data."
        )
    )
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--surgery-root", type=Path, default=DEFAULT_SURGERY_ROOT)
    parser.add_argument("--v0-state-root", type=Path, default=DEFAULT_V0_ROOT)
    parser.add_argument("--step3-state-root", type=Path, default=DEFAULT_STEP3_ROOT)
    parser.add_argument("--proposals", type=Path, default=DEFAULT_PROPOSALS)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--municipalities-json", type=Path, default=DEFAULT_MUNICIPALITIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--progress-every-rows", type=int, default=25)
    parser.add_argument("--spatial-unchanged-threshold-seconds", type=float, default=0.120)
    parser.add_argument("--force", action="store_true", help="Regenerate outputs even if they already exist.")
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-reversal", action="store_true")
    parser.add_argument("--skip-spatial", action="store_true")
    parser.add_argument("--skip-segment", action="store_true")
    parser.add_argument("--skip-od", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_root = args.run_root.expanduser().resolve()
    args.surgery_root = args.surgery_root.expanduser().resolve()
    args.v0_state_root = args.v0_state_root.expanduser().resolve()
    args.step3_state_root = args.step3_state_root.expanduser().resolve()
    args.proposals = args.proposals.expanduser().resolve()
    args.stations_csv = args.stations_csv.expanduser().resolve()
    args.municipalities_json = args.municipalities_json.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    require_inputs(
        [
            args.surgery_root,
            args.v0_state_root / "2026" / "metadata.json",
            args.v0_state_root / "2035" / "metadata.json",
            args.step3_state_root / "2026" / "rows",
            args.step3_state_root / "2035" / "rows",
            args.proposals,
        ]
    )
    outputs: list[Path] = []
    log(f"Writing report inputs to {args.output_dir}")
    if not args.skip_audit:
        outputs.append(run_attribution_audit_v2(args))
    if not args.skip_reversal:
        outputs.append(run_reversal_attribution(args))
    if not args.skip_spatial:
        outputs.extend(run_spatial_deltas(args))
    if not args.skip_segment:
        outputs.append(run_segment_load_deltas(args))
    if not args.skip_od:
        outputs.extend(run_od_extremes(args))
    write_manifest(args, outputs)
    log("All requested Step 3 report inputs are ready.")


if __name__ == "__main__":
    main()
