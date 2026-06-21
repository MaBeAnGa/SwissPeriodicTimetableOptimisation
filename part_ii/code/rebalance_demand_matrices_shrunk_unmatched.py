#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from array import array
from pathlib import Path
from typing import Any


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if not math.isfinite(out):
        return default
    return out


def _load_station_fit(path: Path) -> dict[str, dict[str, float | bool]]:
    out: dict[str, dict[str, float | bool]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "StationName",
            "predicted_station_throughput_raw_model",
            "target_dtv_for_constraint",
            "is_exact_sbb_match",
        }
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path.name}: missing required columns: {missing}")
        for row in reader:
            name = str(row["StationName"]).strip()
            if not name:
                continue
            out[name] = {
                "raw_model": max(_as_float(row["predicted_station_throughput_raw_model"]), 0.0),
                "target": max(_as_float(row["target_dtv_for_constraint"]), 0.0),
                "exact": str(row["is_exact_sbb_match"]).strip() in {"1", "1.0", "true", "True"},
            }
    return out


def _blend_targets(current_target: float, raw_model: float, alpha: float) -> float:
    current_target = max(float(current_target), 0.0)
    raw_model = max(float(raw_model), 0.0)
    if current_target <= 0.0:
        return raw_model
    if raw_model <= 0.0:
        return current_target
    alpha = max(0.0, min(1.0, float(alpha)))
    return math.exp(alpha * math.log(current_target) + (1.0 - alpha) * math.log(raw_model))


def _build_corrected_targets(
    fit_2026: dict[str, dict[str, float | bool]],
    fit_2035: dict[str, dict[str, float | bool]],
    *,
    unmatched_blend_alpha: float,
) -> tuple[dict[str, float], dict[str, float], dict[str, Any]]:
    corrected_2026: dict[str, float] = {}
    changed_2026 = 0
    exact_count_2026 = 0
    unmatched_count_2026 = 0

    for name, rec in fit_2026.items():
        target = float(rec["target"])
        raw = float(rec["raw_model"])
        if bool(rec["exact"]):
            corrected = target
            exact_count_2026 += 1
        else:
            corrected = _blend_targets(target, raw, unmatched_blend_alpha)
            unmatched_count_2026 += 1
        corrected_2026[name] = corrected
        if abs(corrected - target) > max(1e-6, target * 1e-9):
            changed_2026 += 1

    corrected_2035: dict[str, float] = {}
    changed_2035 = 0
    exact_count_2035 = 0
    unmatched_count_2035 = 0
    scenario_only = []
    for name, rec35 in fit_2035.items():
        target35 = float(rec35["target"])
        raw35 = float(rec35["raw_model"])
        if bool(rec35["exact"]):
            corrected = target35
            exact_count_2035 += 1
        else:
            unmatched_count_2035 += 1
            rec26 = fit_2026.get(name)
            corr26 = corrected_2026.get(name)
            if rec26 is not None and corr26 is not None and float(rec26["target"]) > 0.0:
                # Keep the already-computed 2035 population/accessibility uplift,
                # but apply it to the corrected 2026 target instead of the old
                # overconfident fallback DTV.
                factor = target35 / max(float(rec26["target"]), 1e-12)
                corrected = corr26 * factor
            else:
                scenario_only.append(name)
                corrected = _blend_targets(target35, raw35, unmatched_blend_alpha)
        corrected_2035[name] = corrected
        if abs(corrected - target35) > max(1e-6, target35 * 1e-9):
            changed_2035 += 1

    diagnostics = {
        "unmatched_blend_alpha": unmatched_blend_alpha,
        "rule": (
            "exact SBB stations keep target_dtv_for_constraint; unmatched 2026 stations use "
            "geometric blend old_target^alpha * raw_gravity_throughput^(1-alpha); "
            "unmatched 2035 stations inherit the old 2035/2026 uplift factor from corrected 2026"
        ),
        "2026": {
            "exact_count": exact_count_2026,
            "unmatched_count": unmatched_count_2026,
            "changed_targets": changed_2026,
            "old_total": sum(float(v["target"]) for v in fit_2026.values()),
            "new_total": sum(corrected_2026.values()),
        },
        "2035": {
            "exact_count": exact_count_2035,
            "unmatched_count": unmatched_count_2035,
            "changed_targets": changed_2035,
            "scenario_only": scenario_only,
            "old_total": sum(float(v["target"]) for v in fit_2035.values()),
            "new_total": sum(corrected_2035.values()),
        },
    }
    return corrected_2026, corrected_2035, diagnostics


def _read_matrix(path: Path) -> tuple[list[str], list[array]]:
    _log(f"Loading matrix {path}")
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] not in {"", "StationName"}:
            raise ValueError(f"{path.name}: expected first column to be StationName or empty index label.")
        names = [str(x) for x in header[1:]]
        matrix: list[array] = []
        row_names: list[str] = []
        for row in reader:
            if not row:
                continue
            row_names.append(str(row[0]))
            vals = array("d", (_as_float(x, 0.0) for x in row[1:]))
            if len(vals) != len(names):
                raise ValueError(f"{path.name}: row {len(row_names)} width mismatch.")
            matrix.append(vals)
    if row_names != names:
        raise ValueError(f"{path.name}: row labels and column labels are not identical/in-order.")
    return names, matrix


def _write_matrix(path: Path, names: list[str], matrix: list[array]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Writing matrix {path}")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["StationName", *names])
        for name, vals in zip(names, matrix):
            writer.writerow([name, *(f"{float(v):.8f}" for v in vals)])
    tmp.replace(path)


def _throughput(matrix: list[array]) -> list[float]:
    n = len(matrix)
    col = [0.0] * n
    row_sums = [0.0] * n
    for i, row in enumerate(matrix):
        s = 0.0
        for j, v in enumerate(row):
            fv = float(v)
            s += fv
            col[j] += fv
        row_sums[i] = s
    return [row_sums[i] + col[i] for i in range(n)]


def _balance_ipf(
    matrix: list[array],
    targets: list[float],
    *,
    max_iter: int,
    rel_tol: float,
    progress_every: int,
) -> dict[str, Any]:
    n = len(matrix)
    row_targets = [max(t, 0.0) * 0.5 for t in targets]
    col_targets = row_targets[:]
    total_target = sum(row_targets)
    if total_target <= 0:
        raise ValueError("Target throughput is all zero.")

    # Feasibility safety for a strict zero diagonal OD matrix.
    if max(row_targets) >= total_target:
        raise ValueError("At least one row target is >= total network row total; zero-diagonal IPF infeasible.")

    col_sums = [0.0] * n
    final_diag: dict[str, Any] = {}
    for it in range(1, max_iter + 1):
        for i, row in enumerate(matrix):
            s = sum(row)
            target = row_targets[i]
            if target <= 0.0:
                for j in range(n):
                    row[j] = 0.0
            elif s > 0.0:
                scale = target / s
                for j in range(n):
                    row[j] *= scale

        for j in range(n):
            col_sums[j] = 0.0
        for row in matrix:
            for j, v in enumerate(row):
                col_sums[j] += float(v)
        for j, target in enumerate(col_targets):
            if target <= 0.0:
                for i in range(n):
                    matrix[i][j] = 0.0
            elif col_sums[j] > 0.0:
                scale = target / col_sums[j]
                for i in range(n):
                    matrix[i][j] *= scale

        if it % max(1, progress_every) == 0 or it == 1:
            max_row_err = 0.0
            max_col_err = 0.0
            for i, row in enumerate(matrix):
                target = row_targets[i]
                if target > 0:
                    max_row_err = max(max_row_err, abs(sum(row) - target) / target)
            for j in range(n):
                c = 0.0
                for i in range(n):
                    c += float(matrix[i][j])
                target = col_targets[j]
                if target > 0:
                    max_col_err = max(max_col_err, abs(c - target) / target)
            _log(f"IPF iter={it} max_rel_row_err={max_row_err:.3e} max_rel_col_err={max_col_err:.3e}")
            final_diag = {
                "iterations_checked": it,
                "max_rel_row_error": max_row_err,
                "max_rel_col_error": max_col_err,
            }
            if max(max_row_err, max_col_err) <= rel_tol:
                final_diag["converged"] = True
                final_diag["iterations"] = it
                return final_diag

    final_diag["converged"] = False
    final_diag["iterations"] = max_iter
    return final_diag


def _write_target_audit(
    path: Path,
    names: list[str],
    old_fit: dict[str, dict[str, float | bool]],
    new_targets: dict[str, float],
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
                "new_target_dtv",
                "final_matrix_throughput",
                "target_delta",
                "target_ratio",
            ]
        )
        for i, name in enumerate(names):
            rec = old_fit.get(name, {})
            old = float(rec.get("target", 0.0))
            raw = float(rec.get("raw_model", 0.0))
            new = float(new_targets.get(name, old))
            ratio = new / old if old > 0 else 0.0
            writer.writerow(
                [
                    name,
                    int(bool(rec.get("exact", False))),
                    f"{old:.8f}",
                    f"{raw:.8f}",
                    f"{new:.8f}",
                    f"{through[i]:.8f}",
                    f"{(new - old):.8f}",
                    f"{ratio:.8f}",
                ]
            )


def _process_year(
    *,
    year: int,
    matrix_path: Path,
    station_fit_path: Path,
    targets: dict[str, float],
    output_matrix_path: Path,
    output_audit_path: Path,
    max_iter: int,
    rel_tol: float,
    progress_every: int,
    dry_run: bool,
) -> dict[str, Any]:
    fit = _load_station_fit(station_fit_path)
    names, matrix = _read_matrix(matrix_path)
    target_vec = [max(float(targets.get(name, fit.get(name, {}).get("target", 0.0))), 0.0) for name in names]
    old_total = sum(float(fit.get(name, {}).get("target", 0.0)) for name in names)
    new_total = sum(target_vec)
    _log(f"{year}: old target total={old_total:.3f}, new target total={new_total:.3f}, ratio={new_total / max(old_total, 1e-12):.6f}")
    if dry_run:
        return {"year": year, "old_total": old_total, "new_total": new_total, "dry_run": True}
    diag = _balance_ipf(
        matrix,
        target_vec,
        max_iter=max_iter,
        rel_tol=rel_tol,
        progress_every=progress_every,
    )
    _write_matrix(output_matrix_path, names, matrix)
    _write_target_audit(output_audit_path, names, fit, targets, matrix)
    return {"year": year, "old_total": old_total, "new_total": new_total, "ipf": diag}


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Improve 2026/2035 OD demand matrices by shrinking only estimated/unmatched station "
            "marginals toward the raw gravity throughput, then rerunning IPF/RAS."
        )
    )
    p.add_argument("--project-dir", type=Path, default=Path("."))
    p.add_argument("--unmatched-blend-alpha", type=float, default=0.50)
    p.add_argument("--max-iter", type=int, default=300)
    p.add_argument("--rel-tol", type=float, default=1e-8)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = args.project_dir.resolve()
    fit26 = _load_station_fit(root / "od_modeling_2026/2026_gravity_station_fit.csv")
    fit35 = _load_station_fit(root / "od_modeling_2035/2035_gravity_station_fit.csv")
    targets26, targets35, target_diag = _build_corrected_targets(
        fit26,
        fit35,
        unmatched_blend_alpha=float(args.unmatched_blend_alpha),
    )

    outputs: dict[str, Any] = {
        "created_at": _now(),
        "target_adjustment": target_diag,
        "outputs": {},
    }
    outputs["2026"] = _process_year(
        year=2026,
        matrix_path=root / "od_modeling_2026/2026_OD_matrix_Demand_gravity_best.csv",
        station_fit_path=root / "od_modeling_2026/2026_gravity_station_fit.csv",
        targets=targets26,
        output_matrix_path=root / "od_modeling_2026/2026_OD_matrix_Demand_gravity_best_improved.csv",
        output_audit_path=root / "od_modeling_2026/2026_demand_target_improvement_audit.csv",
        max_iter=int(args.max_iter),
        rel_tol=float(args.rel_tol),
        progress_every=int(args.progress_every),
        dry_run=bool(args.dry_run),
    )
    outputs["2035"] = _process_year(
        year=2035,
        matrix_path=root / "od_modeling_2035/2035_OD_matrix_Demand_gravity_best.csv",
        station_fit_path=root / "od_modeling_2035/2035_gravity_station_fit.csv",
        targets=targets35,
        output_matrix_path=root / "od_modeling_2035/2035_OD_matrix_Demand_gravity_best_improved.csv",
        output_audit_path=root / "od_modeling_2035/2035_demand_target_improvement_audit.csv",
        max_iter=int(args.max_iter),
        rel_tol=float(args.rel_tol),
        progress_every=int(args.progress_every),
        dry_run=bool(args.dry_run),
    )

    summary_path = root / "od_demand_improvement_summary_2026_2035.json"
    summary_path.write_text(json.dumps(outputs, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"Wrote summary {summary_path}")


if __name__ == "__main__":
    main()
