#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from array import array
from pathlib import Path
from typing import Any

from rebalance_demand_matrices_shrunk_unmatched import (
    _as_float,
    _balance_ipf,
    _build_corrected_targets,
    _load_station_fit,
    _read_matrix,
    _throughput,
    _write_matrix,
)


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _canonical_transfer_station(name: Any) -> str:
    text = str(name or "").strip()
    if text.startswith("Zürich HB"):
        return "Zürich HB"
    return text


def _load_station_lines(path: Path, year: int) -> dict[str, float]:
    column = f"{year}lines"
    lines: dict[str, float] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "StationName" not in (reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing StationName column")
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path.name}: missing {column} column")
        for row in reader:
            name = str(row.get("StationName", "")).strip()
            if name:
                lines[name] = max(_as_float(row.get(column, 0.0), 0.0), 0.0)
    return lines


def _load_transfer_pax(path: Path) -> dict[str, float]:
    transfers: dict[str, float] = {}
    if not path.exists():
        _log(f"Transfer proxy missing, skipping transfer discounts: {path}")
        return transfers
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"Transfer station", "Number of Affected Daily Pax"}
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name}: missing required columns {missing}")
        for row in reader:
            station = _canonical_transfer_station(row.get("Transfer station", ""))
            if not station:
                continue
            transfers[station] = transfers.get(station, 0.0) + max(
                _as_float(row.get("Number of Affected Daily Pax", 0.0), 0.0),
                0.0,
            )
    return transfers


def _apply_transfer_discount(
    targets: dict[str, float],
    transfer_pax: dict[str, float],
    *,
    transfer_strength: float,
    max_transfer_discount: float,
    min_target_factor: float,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    out: dict[str, float] = {}
    audit: dict[str, dict[str, float]] = {}
    strength = max(0.0, float(transfer_strength))
    cap = max(0.0, min(0.95, float(max_transfer_discount)))
    floor = max(0.0, min(1.0, float(min_target_factor)))

    for name, target in targets.items():
        old = max(float(target), 0.0)
        transfers = max(float(transfer_pax.get(name, 0.0)), 0.0)
        # SBB station frequency counts a transfer as one alighting and one boarding.
        transfer_events = 2.0 * transfers
        if old > 0.0:
            raw_share = transfer_events / old
            discount_share = min(cap, strength * raw_share)
            new_target = max(old * floor, old * (1.0 - discount_share))
        else:
            raw_share = 0.0
            discount_share = 0.0
            new_target = 0.0
        out[name] = new_target
        audit[name] = {
            "transfer_daily_pax_proxy": transfers,
            "transfer_station_frequency_events_proxy": transfer_events,
            "transfer_events_to_target_ratio": raw_share,
            "transfer_discount_share": discount_share,
            "transfer_discount_delta": new_target - old,
        }
    return out, audit


def _distance_factor(
    distance_km: float,
    *,
    max_km: float,
    min_factor: float,
    power: float,
) -> float:
    if not math.isfinite(distance_km) or distance_km <= 0.0 or distance_km >= max_km:
        return 1.0
    ratio = max(0.0, min(1.0, distance_km / max_km))
    return min_factor + (1.0 - min_factor) * (ratio ** power)


def _apply_short_urban_seed_damping(
    *,
    matrix: list[array],
    names: list[str],
    distance_path: Path,
    station_lines: dict[str, float],
    enabled: bool,
    max_km: float,
    min_factor: float,
    power: float,
    line_threshold: float,
) -> dict[str, Any]:
    if not enabled:
        return {"enabled": False}

    _log(f"Applying short-urban seed damping from {distance_path.name}")
    name_index = {name: i for i, name in enumerate(names)}
    damped_cells = 0
    factor_sum = 0.0
    min_seen = 1.0
    max_seen = 1.0

    with distance_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        distance_names = [str(x) for x in header[1:]]
        distance_cols = [(j, name_index.get(name)) for j, name in enumerate(distance_names)]
        for row in reader:
            if not row:
                continue
            origin = str(row[0])
            i = name_index.get(origin)
            if i is None:
                continue
            line_i = station_lines.get(origin, 0.0)
            for csv_j, j in distance_cols:
                if j is None or i == j:
                    continue
                dest = names[j]
                if max(line_i, station_lines.get(dest, 0.0)) < line_threshold:
                    continue
                if csv_j + 1 >= len(row):
                    continue
                dist = _as_float(row[csv_j + 1], float("nan"))
                factor = _distance_factor(
                    dist,
                    max_km=max_km,
                    min_factor=min_factor,
                    power=power,
                )
                if factor >= 0.999999:
                    continue
                matrix[i][j] *= factor
                damped_cells += 1
                factor_sum += factor
                min_seen = min(min_seen, factor)
                max_seen = max(max_seen, factor)

    return {
        "enabled": True,
        "distance_path": str(distance_path),
        "max_km": max_km,
        "min_factor": min_factor,
        "power": power,
        "line_threshold": line_threshold,
        "damped_cells": damped_cells,
        "mean_factor": factor_sum / damped_cells if damped_cells else 1.0,
        "min_factor_seen": min_seen,
        "max_factor_seen": max_seen,
    }


def _write_transfer_audit(
    *,
    path: Path,
    names: list[str],
    old_fit: dict[str, dict[str, float | bool]],
    base_targets: dict[str, float],
    final_targets: dict[str, float],
    transfer_audit: dict[str, dict[str, float]],
    final_matrix: list[array],
) -> None:
    through = _throughput(final_matrix)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "StationName",
                "is_exact_sbb_match",
                "old_target_dtv",
                "raw_gravity_throughput",
                "base_target_after_unmatched_shrink",
                "transfer_daily_pax_proxy",
                "transfer_station_frequency_events_proxy",
                "transfer_events_to_target_ratio",
                "transfer_discount_share",
                "new_target_dtv",
                "final_matrix_throughput",
                "target_delta_vs_old",
                "target_ratio_vs_old",
            ]
        )
        for i, name in enumerate(names):
            rec = old_fit.get(name, {})
            old = float(rec.get("target", 0.0))
            raw = float(rec.get("raw_model", 0.0))
            base = float(base_targets.get(name, old))
            new = float(final_targets.get(name, base))
            ta = transfer_audit.get(name, {})
            writer.writerow(
                [
                    name,
                    int(bool(rec.get("exact", False))),
                    f"{old:.8f}",
                    f"{raw:.8f}",
                    f"{base:.8f}",
                    f"{float(ta.get('transfer_daily_pax_proxy', 0.0)):.8f}",
                    f"{float(ta.get('transfer_station_frequency_events_proxy', 0.0)):.8f}",
                    f"{float(ta.get('transfer_events_to_target_ratio', 0.0)):.8f}",
                    f"{float(ta.get('transfer_discount_share', 0.0)):.8f}",
                    f"{new:.8f}",
                    f"{through[i]:.8f}",
                    f"{new - old:.8f}",
                    f"{new / old if old > 0.0 else 0.0:.8f}",
                ]
            )


def _process_year(
    *,
    root: Path,
    year: int,
    matrix_path: Path,
    station_fit_path: Path,
    base_targets: dict[str, float],
    final_targets: dict[str, float],
    transfer_audit: dict[str, dict[str, float]],
    output_matrix_path: Path,
    output_audit_path: Path,
    station_lines: dict[str, float],
    short_urban_damping: bool,
    short_urban_km: float,
    short_urban_min_factor: float,
    short_urban_power: float,
    short_urban_line_threshold: float,
    max_iter: int,
    rel_tol: float,
    progress_every: int,
    dry_run: bool,
) -> dict[str, Any]:
    fit = _load_station_fit(station_fit_path)
    names, matrix = _read_matrix(matrix_path)
    target_vec = [max(float(final_targets.get(name, fit.get(name, {}).get("target", 0.0))), 0.0) for name in names]
    old_total = sum(float(fit.get(name, {}).get("target", 0.0)) for name in names)
    base_total = sum(float(base_targets.get(name, fit.get(name, {}).get("target", 0.0))) for name in names)
    new_total = sum(target_vec)
    _log(
        f"{year}: old target total={old_total:.3f}, base target total={base_total:.3f}, "
        f"transfer-discounted target total={new_total:.3f}, ratio={new_total / max(old_total, 1e-12):.6f}"
    )

    if dry_run:
        return {
            "year": year,
            "old_total": old_total,
            "base_total": base_total,
            "new_total": new_total,
            "dry_run": True,
        }

    damping_diag = _apply_short_urban_seed_damping(
        matrix=matrix,
        names=names,
        distance_path=root / f"od_modeling_{year}/{year}_station_crowfly_distance_km_gravity_input.csv",
        station_lines=station_lines,
        enabled=short_urban_damping,
        max_km=short_urban_km,
        min_factor=short_urban_min_factor,
        power=short_urban_power,
        line_threshold=short_urban_line_threshold,
    )
    diag = _balance_ipf(
        matrix,
        target_vec,
        max_iter=max_iter,
        rel_tol=rel_tol,
        progress_every=progress_every,
    )
    _write_matrix(output_matrix_path, names, matrix)
    _write_transfer_audit(
        path=output_audit_path,
        names=names,
        old_fit=fit,
        base_targets=base_targets,
        final_targets=final_targets,
        transfer_audit=transfer_audit,
        final_matrix=matrix,
    )
    return {
        "year": year,
        "old_total": old_total,
        "base_total": base_total,
        "new_total": new_total,
        "target_ratio_vs_old": new_total / max(old_total, 1e-12),
        "seed_damping": damping_diag,
        "ipf": diag,
        "output_matrix": str(output_matrix_path),
        "output_audit": str(output_audit_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild OD demand matrices with two safeguards: shrink uncertain station targets, "
            "then discount station targets for transfer double-counting before IPF/RAS."
        )
    )
    parser.add_argument("--project-dir", type=Path, default=Path("."))
    parser.add_argument("--unmatched-blend-alpha", type=float, default=0.50)
    parser.add_argument("--transfer-strength", type=float, default=0.40)
    parser.add_argument("--max-transfer-discount", type=float, default=0.70)
    parser.add_argument("--min-target-factor", type=float, default=0.30)
    parser.add_argument("--disable-short-urban-damping", action="store_true")
    parser.add_argument("--short-urban-km", type=float, default=12.0)
    parser.add_argument("--short-urban-min-factor", type=float, default=0.25)
    parser.add_argument("--short-urban-power", type=float, default=0.75)
    parser.add_argument("--short-urban-line-threshold", type=float, default=8.0)
    parser.add_argument("--max-iter", type=int, default=350)
    parser.add_argument("--rel-tol", type=float, default=1e-8)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.project_dir.resolve()
    fit26 = _load_station_fit(root / "od_modeling_2026/2026_gravity_station_fit.csv")
    fit35 = _load_station_fit(root / "od_modeling_2035/2035_gravity_station_fit.csv")
    base26, base35, unmatched_diag = _build_corrected_targets(
        fit26,
        fit35,
        unmatched_blend_alpha=float(args.unmatched_blend_alpha),
    )

    transfers26 = _load_transfer_pax(root / "Identified_All_Transfers_2026_improved.csv")
    transfers35 = _load_transfer_pax(root / "Identified_All_Transfers_2035_improved.csv")
    final26, transfer_audit26 = _apply_transfer_discount(
        base26,
        transfers26,
        transfer_strength=float(args.transfer_strength),
        max_transfer_discount=float(args.max_transfer_discount),
        min_target_factor=float(args.min_target_factor),
    )
    final35, transfer_audit35 = _apply_transfer_discount(
        base35,
        transfers35,
        transfer_strength=float(args.transfer_strength),
        max_transfer_discount=float(args.max_transfer_discount),
        min_target_factor=float(args.min_target_factor),
    )

    lines26 = _load_station_lines(root / "FilteredStations2026_2035.csv", 2026)
    lines35 = _load_station_lines(root / "FilteredStations2026_2035.csv", 2035)

    summary: dict[str, Any] = {
        "created_at": _now(),
        "method": {
            "unmatched_blend_alpha": float(args.unmatched_blend_alpha),
            "transfer_strength": float(args.transfer_strength),
            "max_transfer_discount": float(args.max_transfer_discount),
            "min_target_factor": float(args.min_target_factor),
            "transfer_formula": (
                "target_after_transfer = max(min_target_factor * target_before_transfer, "
                "target_before_transfer * (1 - min(max_transfer_discount, "
                "transfer_strength * 2 * transfer_daily_pax_proxy / target_before_transfer)))"
            ),
            "short_urban_seed_damping_enabled": not bool(args.disable_short_urban_damping),
            "short_urban_seed_damping": {
                "max_km": float(args.short_urban_km),
                "min_factor": float(args.short_urban_min_factor),
                "power": float(args.short_urban_power),
                "line_threshold": float(args.short_urban_line_threshold),
            },
        },
        "unmatched_target_adjustment": unmatched_diag,
    }

    summary["2026"] = _process_year(
        root=root,
        year=2026,
        matrix_path=root / "od_modeling_2026/2026_OD_matrix_Demand_gravity_best.csv",
        station_fit_path=root / "od_modeling_2026/2026_gravity_station_fit.csv",
        base_targets=base26,
        final_targets=final26,
        transfer_audit=transfer_audit26,
        output_matrix_path=root / "od_modeling_2026/2026_OD_matrix_Demand_gravity_best_transfer_discounted.csv",
        output_audit_path=root / "od_modeling_2026/2026_demand_target_transfer_discount_audit.csv",
        station_lines=lines26,
        short_urban_damping=not bool(args.disable_short_urban_damping),
        short_urban_km=float(args.short_urban_km),
        short_urban_min_factor=float(args.short_urban_min_factor),
        short_urban_power=float(args.short_urban_power),
        short_urban_line_threshold=float(args.short_urban_line_threshold),
        max_iter=int(args.max_iter),
        rel_tol=float(args.rel_tol),
        progress_every=int(args.progress_every),
        dry_run=bool(args.dry_run),
    )
    summary["2035"] = _process_year(
        root=root,
        year=2035,
        matrix_path=root / "od_modeling_2035/2035_OD_matrix_Demand_gravity_best.csv",
        station_fit_path=root / "od_modeling_2035/2035_gravity_station_fit.csv",
        base_targets=base35,
        final_targets=final35,
        transfer_audit=transfer_audit35,
        output_matrix_path=root / "od_modeling_2035/2035_OD_matrix_Demand_gravity_best_transfer_discounted.csv",
        output_audit_path=root / "od_modeling_2035/2035_demand_target_transfer_discount_audit.csv",
        station_lines=lines35,
        short_urban_damping=not bool(args.disable_short_urban_damping),
        short_urban_km=float(args.short_urban_km),
        short_urban_min_factor=float(args.short_urban_min_factor),
        short_urban_power=float(args.short_urban_power),
        short_urban_line_threshold=float(args.short_urban_line_threshold),
        max_iter=int(args.max_iter),
        rel_tol=float(args.rel_tol),
        progress_every=int(args.progress_every),
        dry_run=bool(args.dry_run),
    )

    summary_path = root / "od_demand_transfer_discount_summary_2026_2035.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
