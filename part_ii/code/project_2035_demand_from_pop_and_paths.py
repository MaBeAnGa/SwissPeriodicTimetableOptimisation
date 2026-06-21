#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from calibrate_2026_gravity_model import _balance_to_station_throughput_ipf


EPS = 1e-12


def _now_local() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(msg: str) -> None:
    print(f"[{_now_local()}] {msg}", flush=True)


def _safe_path_count(cell: str) -> int:
    # Fast, allocation-light extraction from the cell JSON string.
    key = '"path_count":'
    pos = cell.find(key)
    if pos < 0:
        # Same-station cells are encoded without path_count.
        return 0
    i = pos + len(key)
    n = len(cell)
    while i < n and cell[i] in " \t\r\n":
        i += 1
    j = i
    while j < n and cell[j].isdigit():
        j += 1
    if j <= i:
        return 0
    try:
        return int(cell[i:j])
    except ValueError:
        return 0


def _read_station_order_from_od_header(path: Path) -> list[str]:
    csv.field_size_limit(sys.maxsize)
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
    if not header or header[0] != "StationName":
        raise ValueError(f"{path.name}: expected first header column to be 'StationName'.")
    if len(header) < 2:
        raise ValueError(f"{path.name}: no station columns found.")
    return [str(x) for x in header[1:]]


def _resolve_column_indices(header: list[str], station_order: list[str], csv_name: str) -> list[int]:
    pos = {name: i for i, name in enumerate(header)}
    missing = [s for s in station_order if s not in pos]
    if missing:
        raise ValueError(
            f"{csv_name}: missing station columns compared with reference order. "
            f"Missing examples: {missing[:10]}"
        )
    return [int(pos[s]) for s in station_order]


@dataclass
class ExtractSummary:
    created_at: str
    input_2026_csv: str
    input_2035_csv: str
    output_ratio_npy: str
    output_count_2026_npy: str
    output_count_2035_npy: str
    output_station_order_json: str
    station_count: int
    smoothing: float
    rows_processed: int
    ratio_min: float
    ratio_p10: float
    ratio_p50: float
    ratio_p90: float
    ratio_max: float
    note: str


def _extract_pathcount_ratio(args: argparse.Namespace) -> None:
    input_2026 = Path(args.input_2026_csv).resolve()
    input_2035 = Path(args.input_2035_csv).resolve()
    out_ratio = Path(args.output_ratio_npy).resolve()
    out_count_2026 = Path(args.output_count_2026_npy).resolve()
    out_count_2035 = Path(args.output_count_2035_npy).resolve()
    out_station_order = Path(args.output_station_order_json).resolve()
    checkpoint_path = Path(args.checkpoint_json).resolve()
    summary_path = Path(args.summary_json).resolve()

    for p in (input_2026, input_2035):
        if not p.exists():
            raise FileNotFoundError(f"Missing input CSV: {p}")

    out_ratio.parent.mkdir(parents=True, exist_ok=True)
    out_count_2026.parent.mkdir(parents=True, exist_ok=True)
    out_count_2035.parent.mkdir(parents=True, exist_ok=True)
    out_station_order.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    station_order = _read_station_order_from_od_header(input_2035)
    n = len(station_order)
    out_station_order.write_text(
        json.dumps({"station_order": station_order}, ensure_ascii=False),
        encoding="utf-8",
    )
    _log(f"Reference station order loaded from {input_2035.name} (n={n}).")

    start_row = 0
    if args.resume and checkpoint_path.exists() and out_ratio.exists() and out_count_2026.exists() and out_count_2035.exists():
        try:
            ckpt = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            start_row = int(ckpt.get("next_row", 0))
        except Exception:
            start_row = 0
        start_row = max(0, min(start_row, n))
        mode = "r+"
        _log(f"Resuming extraction from row {start_row}/{n}.")
    else:
        mode = "w+"
        _log("Starting fresh extraction run.")

    ratio = np.lib.format.open_memmap(out_ratio, mode=mode, dtype=np.float32, shape=(n, n))
    cnt26 = np.lib.format.open_memmap(out_count_2026, mode=mode, dtype=np.uint16, shape=(n, n))
    cnt35 = np.lib.format.open_memmap(out_count_2035, mode=mode, dtype=np.uint16, shape=(n, n))

    csv.field_size_limit(sys.maxsize)
    with input_2026.open("r", encoding="utf-8", newline="") as f26, input_2035.open("r", encoding="utf-8", newline="") as f35:
        r26 = csv.reader(f26)
        r35 = csv.reader(f35)
        h26 = next(r26)
        h35 = next(r35)

        if not h26 or h26[0] != "StationName":
            raise ValueError(f"{input_2026.name}: expected first header column 'StationName'.")
        if not h35 or h35[0] != "StationName":
            raise ValueError(f"{input_2035.name}: expected first header column 'StationName'.")

        idx26 = _resolve_column_indices(h26, station_order, input_2026.name)
        idx35 = _resolve_column_indices(h35, station_order, input_2035.name)
        origin_col_26 = int(h26.index("StationName"))
        origin_col_35 = int(h35.index("StationName"))

        for _ in range(start_row):
            try:
                next(r26)
                next(r35)
            except StopIteration as exc:
                raise RuntimeError("Reached EOF while skipping to resume row.") from exc

        smoothing = float(args.smoothing)
        progress_every = max(1, int(args.progress_every_rows))
        checkpoint_every = max(1, int(args.checkpoint_every_rows))

        for i in range(start_row, n):
            try:
                row26 = next(r26)
                row35 = next(r35)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Unexpected EOF while processing row {i}. "
                    f"Expected {n} station rows."
                ) from exc

            origin_ref = station_order[i]
            origin26 = row26[origin_col_26]
            origin35 = row35[origin_col_35]
            if origin26 != origin_ref or origin35 != origin_ref:
                raise ValueError(
                    f"Row-order mismatch at row {i}: "
                    f"expected '{origin_ref}', got 2026='{origin26}', 2035='{origin35}'."
                )

            for j in range(n):
                c26 = _safe_path_count(row26[idx26[j]])
                c35 = _safe_path_count(row35[idx35[j]])
                cnt26[i, j] = np.uint16(min(max(c26, 0), 65535))
                cnt35[i, j] = np.uint16(min(max(c35, 0), 65535))
                ratio[i, j] = np.float32((c35 + smoothing) / (c26 + smoothing))

            if (i + 1) % progress_every == 0 or (i + 1) == n:
                _log(f"extract progress: {i + 1}/{n} rows")
            if (i + 1) % checkpoint_every == 0 or (i + 1) == n:
                ratio.flush()
                cnt26.flush()
                cnt35.flush()
                checkpoint_path.write_text(
                    json.dumps(
                        {
                            "next_row": i + 1,
                            "station_count": n,
                            "updated_at": _now_local(),
                        }
                    ),
                    encoding="utf-8",
                )

    ratio.flush()
    cnt26.flush()
    cnt35.flush()

    flat = np.asarray(ratio, dtype=np.float32).reshape(-1)
    summary = ExtractSummary(
        created_at=_now_local(),
        input_2026_csv=str(input_2026),
        input_2035_csv=str(input_2035),
        output_ratio_npy=str(out_ratio),
        output_count_2026_npy=str(out_count_2026),
        output_count_2035_npy=str(out_count_2035),
        output_station_order_json=str(out_station_order),
        station_count=n,
        smoothing=float(args.smoothing),
        rows_processed=n,
        ratio_min=float(np.min(flat)),
        ratio_p10=float(np.percentile(flat, 10)),
        ratio_p50=float(np.percentile(flat, 50)),
        ratio_p90=float(np.percentile(flat, 90)),
        ratio_max=float(np.max(flat)),
        note="ratio = (path_count_2035 + smoothing) / (path_count_2026 + smoothing)",
    )
    summary_path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    checkpoint_path.unlink(missing_ok=True)
    _log(f"Extraction complete. Wrote ratio/count arrays and summary at {summary_path}.")


def _write_matrix_csv(path: Path, station_order: list[str], matrix: np.ndarray) -> None:
    df = pd.DataFrame(matrix, index=station_order, columns=station_order)
    df.to_csv(path, index=True, float_format="%.10f")


def _load_station_order_json(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    order = payload.get("station_order")
    if not isinstance(order, list) or not order:
        raise ValueError(f"{path.name}: missing or invalid 'station_order'.")
    return [str(x) for x in order]


@dataclass
class BuildSummary:
    created_at: str
    station_count: int
    input_2026_demand_matrix: str
    input_ratio_npy: str
    input_wait_2026_matrix: str
    input_wait_2035_matrix: str
    input_station_table: str
    output_raw_matrix: str
    output_balanced_matrix: str
    output_station_projection: str
    service_elasticity: float
    target_service_elasticity: float
    wait_elasticity: float
    wait_smoothing_min: float
    smoothing_note: str
    pair_ratio_clip_min: float
    pair_ratio_clip_max: float
    pop_growth_clip_min: float
    pop_growth_clip_max: float
    doubling_service_multiplier_pair_level: float
    doubling_service_multiplier_target_level: float
    target_total_2026_base: float
    target_total_2035_projected: float
    dtv_ipf: dict[str, Any]
    note: str


def _build_2035_demand_matrix(args: argparse.Namespace) -> None:
    in_2026_matrix = Path(args.input_2026_demand_matrix).resolve()
    in_ratio_path = Path(args.input_ratio_npy).resolve()
    in_station_order = Path(args.input_station_order_json).resolve()
    in_stations = Path(args.input_station_table_csv).resolve()
    in_wait_2026 = Path(args.input_wait_2026_matrix).resolve()
    in_wait_2035 = Path(args.input_wait_2035_matrix).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not in_2026_matrix.exists():
        raise FileNotFoundError(f"Missing 2026 demand matrix: {in_2026_matrix}")
    if not in_ratio_path.exists():
        raise FileNotFoundError(f"Missing ratio matrix: {in_ratio_path}")
    if not in_station_order.exists():
        raise FileNotFoundError(f"Missing station-order JSON: {in_station_order}")
    if not in_stations.exists():
        raise FileNotFoundError(f"Missing station table CSV: {in_stations}")
    if not in_wait_2026.exists():
        raise FileNotFoundError(f"Missing 2026 wait matrix CSV: {in_wait_2026}")
    if not in_wait_2035.exists():
        raise FileNotFoundError(f"Missing 2035 wait matrix CSV: {in_wait_2035}")

    station_order = _load_station_order_json(in_station_order)
    n = len(station_order)
    station_index = {s: i for i, s in enumerate(station_order)}

    ratio = np.load(in_ratio_path, mmap_mode="r")
    if ratio.shape != (n, n):
        raise ValueError(f"Ratio matrix shape mismatch: expected {(n, n)}, got {ratio.shape}")

    stations_df = pd.read_csv(in_stations, dtype=str).fillna("")
    if "StationName" not in stations_df.columns:
        raise ValueError("Station table missing StationName column.")
    stations_df["StationName"] = stations_df["StationName"].astype(str)
    stations_df = stations_df.drop_duplicates(subset=["StationName"], keep="first").set_index("StationName")

    missing_meta = [s for s in station_order if s not in stations_df.index]
    if missing_meta:
        raise ValueError(
            "Station metadata missing stations from OD order. "
            f"Examples: {missing_meta[:10]}"
        )

    pop2026 = pd.to_numeric(stations_df.loc[station_order, "2026_distributedpop"], errors="coerce").to_numpy(dtype=float)
    pop2035 = pd.to_numeric(stations_df.loc[station_order, "2035_distributedpop"], errors="coerce").to_numpy(dtype=float)
    dtv_base = pd.to_numeric(stations_df.loc[station_order, "DTV"], errors="coerce").to_numpy(dtype=float)

    pop_growth = pop2035 / np.maximum(pop2026, EPS)
    pop_growth[~np.isfinite(pop_growth)] = 1.0
    pop_growth = np.clip(pop_growth, float(args.pop_growth_clip_min), float(args.pop_growth_clip_max))

    d26 = pd.read_csv(in_2026_matrix, index_col=0, low_memory=False)
    d26.index = d26.index.astype(str)
    d26.columns = d26.columns.astype(str)
    d26 = d26.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    full_2026 = np.zeros((n, n), dtype=np.float64)
    common = [s for s in station_order if s in d26.index and s in d26.columns]
    if not common:
        raise ValueError("No overlapping stations between 2026 demand matrix and station order.")
    sub = d26.loc[common, common].to_numpy(dtype=np.float64)
    idx = [station_index[s] for s in common]
    full_2026[np.ix_(idx, idx)] = sub

    wait26_df = pd.read_csv(in_wait_2026, index_col=0, low_memory=False)
    wait35_df = pd.read_csv(in_wait_2035, index_col=0, low_memory=False)
    wait26_df.index = wait26_df.index.astype(str)
    wait26_df.columns = wait26_df.columns.astype(str)
    wait35_df.index = wait35_df.index.astype(str)
    wait35_df.columns = wait35_df.columns.astype(str)
    wait26_df = wait26_df.apply(pd.to_numeric, errors="coerce").fillna(np.nan)
    wait35_df = wait35_df.apply(pd.to_numeric, errors="coerce").fillna(np.nan)

    wait26 = np.full((n, n), np.nan, dtype=np.float64)
    wait35 = np.full((n, n), np.nan, dtype=np.float64)
    common_wait = [s for s in station_order if s in wait26_df.index and s in wait26_df.columns and s in wait35_df.index and s in wait35_df.columns]
    if not common_wait:
        raise ValueError("No overlapping stations between wait matrices and station order.")
    widx = [station_index[s] for s in common_wait]
    wait26[np.ix_(widx, widx)] = wait26_df.loc[common_wait, common_wait].to_numpy(dtype=np.float64)
    wait35[np.ix_(widx, widx)] = wait35_df.loc[common_wait, common_wait].to_numpy(dtype=np.float64)

    finite_wait = np.isfinite(wait26) & np.isfinite(wait35)
    if np.any(finite_wait):
        fill26 = float(np.nanmedian(wait26[finite_wait]))
        fill35 = float(np.nanmedian(wait35[finite_wait]))
    else:
        fill26 = 10.0
        fill35 = 10.0
    wait26 = np.where(np.isfinite(wait26), np.maximum(wait26, 0.0), fill26)
    wait35 = np.where(np.isfinite(wait35), np.maximum(wait35, 0.0), fill35)
    np.fill_diagonal(wait26, 0.0)
    np.fill_diagonal(wait35, 0.0)

    np.fill_diagonal(full_2026, 0.0)
    full_2026 = np.maximum(full_2026, 0.0)

    ratio_clipped = np.clip(
        ratio.astype(np.float64),
        float(args.pair_ratio_clip_min),
        float(args.pair_ratio_clip_max),
    )
    service_elasticity = float(args.service_elasticity)
    target_service_elasticity = (
        float(args.target_service_elasticity)
        if args.target_service_elasticity is not None
        else service_elasticity
    )
    pair_service_factor = np.power(ratio_clipped, service_elasticity, dtype=np.float64)

    wait_elasticity = float(args.wait_elasticity)
    wait_smoothing_min = float(args.wait_smoothing_min)
    wait_ratio = (wait35 + wait_smoothing_min) / np.maximum(wait26 + wait_smoothing_min, EPS)
    wait_ratio = np.clip(
        wait_ratio,
        float(args.wait_ratio_clip_min),
        float(args.wait_ratio_clip_max),
    )
    wait_factor = np.power(wait_ratio, wait_elasticity, dtype=np.float64)
    pair_service_factor = pair_service_factor * wait_factor

    pop_pair_factor = np.sqrt(np.outer(pop_growth, pop_growth))
    raw_2035 = full_2026 * pop_pair_factor * pair_service_factor
    np.fill_diagonal(raw_2035, 0.0)

    offdiag_mask = ~np.eye(n, dtype=bool)
    positive = raw_2035[offdiag_mask & (raw_2035 > 0.0)]
    median_positive = float(np.median(positive)) if positive.size else 1.0
    seed_eps = max(median_positive * 1e-9, 1e-12)
    raw_2035[offdiag_mask] = np.maximum(raw_2035[offdiag_mask], seed_eps)
    np.fill_diagonal(raw_2035, 0.0)

    throughput_2026 = full_2026.sum(axis=1) + full_2026.sum(axis=0)
    dtv_base = np.where(np.isfinite(dtv_base), np.maximum(dtv_base, 0.0), 0.0)
    throughput_2026 = np.where(throughput_2026 > 0.0, throughput_2026, dtv_base)

    row_w = full_2026.copy()
    col_w = full_2026.copy()
    row_sum = row_w.sum(axis=1)
    col_sum = col_w.sum(axis=0)
    out_ratio_station = np.zeros(n, dtype=np.float64)
    in_ratio_station = np.zeros(n, dtype=np.float64)
    out_wait_ratio_station = np.zeros(n, dtype=np.float64)
    in_wait_ratio_station = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if row_sum[i] > 0.0:
            out_ratio_station[i] = float(np.dot(row_w[i, :], ratio_clipped[i, :]) / max(row_sum[i], EPS))
            out_wait_ratio_station[i] = float(np.dot(row_w[i, :], wait_ratio[i, :]) / max(row_sum[i], EPS))
        else:
            vals = ratio_clipped[i, :].copy()
            vals = vals[np.arange(n) != i]
            out_ratio_station[i] = float(np.median(vals)) if vals.size else 1.0
            vwait = wait_ratio[i, :].copy()
            vwait = vwait[np.arange(n) != i]
            out_wait_ratio_station[i] = float(np.median(vwait)) if vwait.size else 1.0
        if col_sum[i] > 0.0:
            in_ratio_station[i] = float(np.dot(col_w[:, i], ratio_clipped[:, i]) / max(col_sum[i], EPS))
            in_wait_ratio_station[i] = float(np.dot(col_w[:, i], wait_ratio[:, i]) / max(col_sum[i], EPS))
        else:
            vals = ratio_clipped[:, i].copy()
            vals = vals[np.arange(n) != i]
            in_ratio_station[i] = float(np.median(vals)) if vals.size else 1.0
            vwait = wait_ratio[:, i].copy()
            vwait = vwait[np.arange(n) != i]
            in_wait_ratio_station[i] = float(np.median(vwait)) if vwait.size else 1.0

    station_service_ratio = np.sqrt(np.maximum(out_ratio_station, EPS) * np.maximum(in_ratio_station, EPS))
    station_service_ratio = np.clip(
        station_service_ratio,
        float(args.pair_ratio_clip_min),
        float(args.pair_ratio_clip_max),
    )
    station_wait_ratio = np.sqrt(np.maximum(out_wait_ratio_station, EPS) * np.maximum(in_wait_ratio_station, EPS))
    station_wait_ratio = np.clip(
        station_wait_ratio,
        float(args.wait_ratio_clip_min),
        float(args.wait_ratio_clip_max),
    )
    station_service_factor = (
        np.power(station_service_ratio, target_service_elasticity)
        * np.power(station_wait_ratio, wait_elasticity)
    )

    target_2035 = throughput_2026 * pop_growth * station_service_factor
    target_2035 = np.where(np.isfinite(target_2035), np.maximum(target_2035, 0.0), 0.0)

    if np.sum(target_2035) <= 0:
        raise ValueError("Projected 2035 station targets are non-positive.")

    balanced_2035, ipf_diag = _balance_to_station_throughput_ipf(
        matrix=raw_2035,
        target_throughput=target_2035,
        zero_diagonal=True,
        max_iter=max(int(args.ipf_max_iter), 100),
        rel_tol=max(float(args.ipf_rel_tol), 1e-14),
    )

    out_raw = out_dir / "2035_OD_matrix_Demand_gravity_best_raw.csv"
    out_best = out_dir / "2035_OD_matrix_Demand_gravity_best.csv"
    out_station_proj = out_dir / "2035_station_throughput_projection.csv"
    out_summary = out_dir / "2035_demand_projection_summary.json"

    _write_matrix_csv(out_raw, station_order, raw_2035)
    _write_matrix_csv(out_best, station_order, balanced_2035)

    final_row = balanced_2035.sum(axis=1)
    final_col = balanced_2035.sum(axis=0)
    final_throughput = final_row + final_col
    station_table = pd.DataFrame(
        {
            "StationName": station_order,
            "throughput_2026_base": throughput_2026,
            "pop_growth_2035_vs_2026": pop_growth,
            "station_service_ratio_2035_vs_2026": station_service_ratio,
            "station_wait_ratio_2035_vs_2026": station_wait_ratio,
            "station_service_factor_applied": station_service_factor,
            "target_throughput_2035_before_ipf": target_2035,
            "final_row_sum_2035": final_row,
            "final_col_sum_2035": final_col,
            "final_throughput_2035": final_throughput,
            "residual_vs_target_after_ipf": final_throughput - target_2035,
        }
    )
    station_table.to_csv(out_station_proj, index=False, float_format="%.8f")

    summary = BuildSummary(
        created_at=_now_local(),
        station_count=n,
        input_2026_demand_matrix=str(in_2026_matrix),
        input_ratio_npy=str(in_ratio_path),
        input_wait_2026_matrix=str(in_wait_2026),
        input_wait_2035_matrix=str(in_wait_2035),
        input_station_table=str(in_stations),
        output_raw_matrix=str(out_raw),
        output_balanced_matrix=str(out_best),
        output_station_projection=str(out_station_proj),
        service_elasticity=service_elasticity,
        target_service_elasticity=target_service_elasticity,
        wait_elasticity=wait_elasticity,
        wait_smoothing_min=wait_smoothing_min,
        smoothing_note=(
            "Path-count ratio generated with smoothing in extract stage: "
            "ratio=(count_2035+smoothing)/(count_2026+smoothing)."
        ),
        pair_ratio_clip_min=float(args.pair_ratio_clip_min),
        pair_ratio_clip_max=float(args.pair_ratio_clip_max),
        pop_growth_clip_min=float(args.pop_growth_clip_min),
        pop_growth_clip_max=float(args.pop_growth_clip_max),
        doubling_service_multiplier_pair_level=float(2.0**service_elasticity),
        doubling_service_multiplier_target_level=float(2.0**target_service_elasticity),
        target_total_2026_base=float(np.sum(throughput_2026)),
        target_total_2035_projected=float(np.sum(target_2035)),
        dtv_ipf=dict(ipf_diag),
        note=(
            "2035 demand projection uses 2026 OD matrix as base and applies: "
            "(i) station population growth factor and "
            "(ii) nondominated-path-count service factor and wait-time factor, "
            "then enforces station marginals via IPF."
        ),
    )
    out_summary.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")
    _log(f"Wrote {out_raw.name}")
    _log(f"Wrote {out_best.name}")
    _log(f"Wrote {out_station_proj.name}")
    _log(f"Wrote {out_summary.name}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Crash-safe projection of 2035 OD demand from 2026 demand + population growth "
            "+ nondominated path-count change."
        )
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_extract = sub.add_parser(
        "extract-pathcount-ratio",
        help="Stream huge 2026/2035 nondominated CSVs and write compact path-count ratio arrays.",
    )
    p_extract.add_argument(
        "--input-2026-csv",
        type=Path,
        default=Path("2026_OD_matrix_Nondominated_paths_full.csv"),
    )
    p_extract.add_argument(
        "--input-2035-csv",
        type=Path,
        default=Path("2035_OD_matrix_Nondominated_paths_full.csv"),
    )
    p_extract.add_argument(
        "--output-ratio-npy",
        type=Path,
        default=Path("od_modeling_2035/2035_pathcount_ratio_2035_vs_2026.npy"),
    )
    p_extract.add_argument(
        "--output-count-2026-npy",
        type=Path,
        default=Path("od_modeling_2035/2026_pathcount_matrix.npy"),
    )
    p_extract.add_argument(
        "--output-count-2035-npy",
        type=Path,
        default=Path("od_modeling_2035/2035_pathcount_matrix.npy"),
    )
    p_extract.add_argument(
        "--output-station-order-json",
        type=Path,
        default=Path("od_modeling_2035/station_order_for_projection.json"),
    )
    p_extract.add_argument(
        "--summary-json",
        type=Path,
        default=Path("od_modeling_2035/2035_pathcount_ratio_summary.json"),
    )
    p_extract.add_argument(
        "--checkpoint-json",
        type=Path,
        default=Path("od_modeling_2035/2035_pathcount_ratio_checkpoint.json"),
    )
    p_extract.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Additive smoothing in ratio=(c2035+s)/(c2026+s).",
    )
    p_extract.add_argument(
        "--progress-every-rows",
        type=int,
        default=10,
    )
    p_extract.add_argument(
        "--checkpoint-every-rows",
        type=int,
        default=10,
    )
    p_extract.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint and existing array files if present.",
    )

    p_build = sub.add_parser(
        "build-2035-demand",
        help="Build 2035 demand matrix with IPF using extracted path-count ratios.",
    )
    p_build.add_argument(
        "--input-2026-demand-matrix",
        type=Path,
        default=Path("od_modeling_2026/2026_OD_matrix_Demand_gravity_best.csv"),
    )
    p_build.add_argument(
        "--input-ratio-npy",
        type=Path,
        default=Path("od_modeling_2035/2035_pathcount_ratio_2035_vs_2026.npy"),
    )
    p_build.add_argument(
        "--input-station-order-json",
        type=Path,
        default=Path("od_modeling_2035/station_order_for_projection.json"),
    )
    p_build.add_argument(
        "--input-station-table-csv",
        type=Path,
        default=Path("FilteredStations2026_2035.csv"),
    )
    p_build.add_argument(
        "--output-dir",
        type=Path,
        default=Path("od_modeling_2035"),
    )
    p_build.add_argument(
        "--service-elasticity",
        type=float,
        default=0.50,
        help=(
            "Elasticity linking path-count ratio to pair-level demand multiplier. "
            "If service doubles, multiplier=2^elasticity."
        ),
    )
    p_build.add_argument(
        "--wait-elasticity",
        type=float,
        default=0.0,
        help=(
            "Elasticity linking wait-time ratio to demand multiplier. "
            "Multiplier uses ((wait_2035+s)/(wait_2026+s))^wait_elasticity."
        ),
    )
    p_build.add_argument(
        "--input-wait-2026-matrix",
        type=Path,
        default=Path("od_modeling_2026/2026_OD_matrix_Wait_time_full_gravity_input.csv"),
    )
    p_build.add_argument(
        "--input-wait-2035-matrix",
        type=Path,
        default=Path("od_modeling_2035/2035_OD_matrix_Wait_time_full_gravity_input.csv"),
    )
    p_build.add_argument(
        "--wait-smoothing-min",
        type=float,
        default=1.0,
    )
    p_build.add_argument(
        "--wait-ratio-clip-min",
        type=float,
        default=0.50,
    )
    p_build.add_argument(
        "--wait-ratio-clip-max",
        type=float,
        default=2.50,
    )
    p_build.add_argument(
        "--target-service-elasticity",
        type=float,
        default=None,
        help=(
            "Optional separate elasticity for station-target uplift. "
            "Defaults to --service-elasticity."
        ),
    )
    p_build.add_argument(
        "--pair-ratio-clip-min",
        type=float,
        default=0.50,
    )
    p_build.add_argument(
        "--pair-ratio-clip-max",
        type=float,
        default=2.50,
    )
    p_build.add_argument(
        "--pop-growth-clip-min",
        type=float,
        default=0.50,
    )
    p_build.add_argument(
        "--pop-growth-clip-max",
        type=float,
        default=3.00,
    )
    p_build.add_argument(
        "--ipf-max-iter",
        type=int,
        default=8000,
    )
    p_build.add_argument(
        "--ipf-rel-tol",
        type=float,
        default=1e-8,
    )

    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.cmd == "extract-pathcount-ratio":
        _extract_pathcount_ratio(args)
    elif args.cmd == "build-2035-demand":
        _build_2035_demand_matrix(args)
    else:
        raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
