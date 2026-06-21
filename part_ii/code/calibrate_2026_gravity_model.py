#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EPS = 1e-12
LOG_CLIP_MIN = -60.0
LOG_CLIP_MAX = 60.0


@dataclass(frozen=True)
class ParamSpec:
    name: str
    low: float
    high: float


@dataclass(frozen=True)
class PopulationScalingDiagnostics:
    path: str
    target_year: int
    rows_loaded: int
    unique_canton_year_pairs: int
    skipped_rows: int
    stations_with_scaling_pair: int
    stations_missing_scaling_pair: int
    stations_with_nontrivial_scaling: int


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= EPS or y_std <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


PARAM_SPECS: tuple[ParamSpec, ...] = (
    ParamSpec("origin_pop_exp", 0.20, 1.80),
    ParamSpec("dest_pop_exp", 0.20, 1.80),
    ParamSpec("origin_lines_exp", 0.00, 1.20),
    ParamSpec("dest_lines_exp", 0.00, 1.20),
    ParamSpec("freq_coef", 0.00, 1.40),
    ParamSpec("dist_log_coef", 0.20, 2.50),
    ParamSpec("fastest_coef", 0.000, 0.060),
    ParamSpec("wait_coef", 0.000, 0.120),
    ParamSpec("rolling_coef", 0.000, 0.050),
    ParamSpec("dwell_coef", 0.000, 0.040),
    ParamSpec("transfer_coef", 0.000, 0.180),
    ParamSpec("hub_transfer_bonus", 0.000, 0.800),
)


def _now_local() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _parse_optional_year(value: Any) -> int | None:
    txt = str(value).strip()
    if not txt:
        return None
    try:
        year = int(float(txt))
    except ValueError:
        return None
    if year < 1800 or year > 2200:
        return None
    return year


def _load_population_scale_lookup(path: Path) -> tuple[dict[tuple[str, int], float], int, int]:
    if not path.exists():
        raise FileNotFoundError(f"Population scaling table not found: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    required = {"canton", "source_population_year", "growth_factor_vs_2023"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{path.name}: missing required population scaling columns: {missing}"
        )

    lookup: dict[tuple[str, int], float] = {}
    skipped_rows = 0
    duplicate_rows = 0
    for row in df.itertuples(index=False):
        canton = str(getattr(row, "canton", "")).strip().upper()
        year = _parse_optional_year(getattr(row, "source_population_year", ""))
        growth_txt = str(getattr(row, "growth_factor_vs_2023", "")).strip()
        if not canton or year is None or not growth_txt:
            skipped_rows += 1
            continue
        try:
            growth = float(growth_txt)
        except ValueError:
            skipped_rows += 1
            continue
        if not np.isfinite(growth) or growth <= 0:
            skipped_rows += 1
            continue
        key = (canton, year)
        if key in lookup:
            duplicate_rows += 1
            continue
        lookup[key] = growth
    return lookup, skipped_rows, duplicate_rows


def _load_numeric_square_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, low_memory=False)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    if list(df.index) != list(df.columns):
        raise ValueError(f"{path.name}: row/column labels are not aligned in identical order.")
    return df.apply(pd.to_numeric, errors="coerce")


def _fill_nan_with_median(values: np.ndarray) -> np.ndarray:
    out = values.astype(np.float64, copy=True)
    if np.isnan(out).any():
        med = np.nanmedian(out)
        if not np.isfinite(med):
            med = 0.0
        out = np.nan_to_num(out, nan=float(med), posinf=float(med), neginf=float(med))
    return out


def _resolve_input_file(path_value: Path | None, *, input_dir: Path, default_name: str) -> Path:
    candidate = path_value if path_value is not None else Path(default_name)
    if not candidate.is_absolute():
        candidate = input_dir / candidate
    return candidate


def _build_matrix_paths(
    *,
    input_dir: Path,
    model_year: int,
    distance_file: Path,
) -> dict[str, Path]:
    year = int(model_year)
    return {
        "frequency": input_dir / f"{year}_OD_matrix_Frequency_full_gravity_input.csv",
        "fastest": input_dir / f"{year}_OD_matrix_Fastest_observed_connection_full_gravity_input.csv",
        "wait": input_dir / f"{year}_OD_matrix_Wait_time_full_gravity_input.csv",
        "rolling": input_dir / f"{year}_OD_matrix_Rolling_time_full_gravity_input.csv",
        "dwell": input_dir / f"{year}_OD_matrix_Dwell_time_full_gravity_input.csv",
        "transfer": input_dir / f"{year}_OD_matrix_Transfer_time_full_gravity_input.csv",
        "distance": distance_file,
    }


def _build_pair_term(params: dict[str, float], pair_features: dict[str, np.ndarray]) -> np.ndarray:
    return (
        params["freq_coef"] * pair_features["log_freq"]
        - params["dist_log_coef"] * pair_features["log_dist"]
        - params["fastest_coef"] * pair_features["fastest"]
        - params["wait_coef"] * pair_features["wait"]
        - params["rolling_coef"] * pair_features["rolling"]
        - params["dwell_coef"] * pair_features["dwell"]
        - params["transfer_coef"] * pair_features["transfer"]
    )


def _compute_accessibility_from_pair_term(
    pair_term: np.ndarray,
    *,
    zero_diagonal: bool,
) -> np.ndarray:
    term = np.array(pair_term, dtype=np.float64, copy=True)
    np.clip(term, LOG_CLIP_MIN, LOG_CLIP_MAX, out=term)
    access = np.exp(term)
    if zero_diagonal:
        np.fill_diagonal(access, 0.0)
    return np.maximum(access.sum(axis=1), EPS)


def _load_fixed_params(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"Fixed params file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}: JSON root must be an object.")

    candidate_obj = payload.get("best_params", payload)
    if not isinstance(candidate_obj, dict):
        raise ValueError(
            f"{path.name}: expected either a top-level params object or a 'best_params' object."
        )

    params: dict[str, float] = {}
    for spec in PARAM_SPECS:
        value = candidate_obj.get(spec.name)
        if value is None:
            raise ValueError(f"{path.name}: missing parameter '{spec.name}'.")
        try:
            params[spec.name] = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path.name}: invalid value for '{spec.name}': {value!r}") from exc
    return params


def _compute_induced_demand_adjustment(
    *,
    scenario_pair_term: np.ndarray,
    baseline_pair_term: np.ndarray,
    base_target_throughput: np.ndarray,
    origin_elasticity: float,
    dest_elasticity: float,
    factor_min: float,
    factor_max: float,
    preserve_total_throughput: bool,
    zero_diagonal: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    baseline_access = _compute_accessibility_from_pair_term(
        baseline_pair_term, zero_diagonal=zero_diagonal
    )
    scenario_access = _compute_accessibility_from_pair_term(
        scenario_pair_term, zero_diagonal=zero_diagonal
    )
    return _compute_induced_demand_adjustment_from_access(
        scenario_access=scenario_access,
        baseline_access=baseline_access,
        base_target_throughput=base_target_throughput,
        origin_elasticity=origin_elasticity,
        dest_elasticity=dest_elasticity,
        factor_min=factor_min,
        factor_max=factor_max,
        preserve_total_throughput=preserve_total_throughput,
    )


def _compute_induced_demand_adjustment_from_access(
    *,
    scenario_access: np.ndarray,
    baseline_access: np.ndarray,
    base_target_throughput: np.ndarray,
    origin_elasticity: float,
    dest_elasticity: float,
    factor_min: float,
    factor_max: float,
    preserve_total_throughput: bool,
) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    if scenario_access.shape != baseline_access.shape:
        raise ValueError(
            "Scenario/baseline accessibility vectors must have identical shapes "
            f"(got {scenario_access.shape} vs {baseline_access.shape})."
        )
    if scenario_access.shape != base_target_throughput.shape:
        raise ValueError(
            "Accessibility vectors and throughput target must have identical shapes "
            f"(got {scenario_access.shape} vs {base_target_throughput.shape})."
        )

    delta_log_access = np.log(np.maximum(scenario_access, EPS)) - np.log(
        np.maximum(baseline_access, EPS)
    )
    origin_factor = np.exp(origin_elasticity * delta_log_access)
    dest_factor = np.exp(dest_elasticity * delta_log_access)
    origin_factor = np.clip(origin_factor, factor_min, factor_max)
    dest_factor = np.clip(dest_factor, factor_min, factor_max)

    # Throughput target remains a single station vector; use symmetric blend.
    throughput_factor = np.sqrt(origin_factor * dest_factor)
    adjusted_target = np.maximum(base_target_throughput * throughput_factor, 0.0)

    pre_total = float(np.sum(base_target_throughput))
    post_total = float(np.sum(adjusted_target))
    if preserve_total_throughput and pre_total > EPS and post_total > EPS:
        adjusted_target *= pre_total / post_total
        post_total = float(np.sum(adjusted_target))

    diag: dict[str, Any] = {
        "enabled": True,
        "origin_elasticity": float(origin_elasticity),
        "dest_elasticity": float(dest_elasticity),
        "factor_min": float(factor_min),
        "factor_max": float(factor_max),
        "preserve_total_throughput": bool(preserve_total_throughput),
        "baseline_total_target_throughput": pre_total,
        "adjusted_total_target_throughput": post_total,
        "network_uplift_factor": (post_total / pre_total) if pre_total > EPS else float("nan"),
        "delta_log_access_min": float(np.min(delta_log_access)),
        "delta_log_access_p10": float(np.percentile(delta_log_access, 10)),
        "delta_log_access_p50": float(np.percentile(delta_log_access, 50)),
        "delta_log_access_p90": float(np.percentile(delta_log_access, 90)),
        "delta_log_access_max": float(np.max(delta_log_access)),
        "throughput_factor_min": float(np.min(throughput_factor)),
        "throughput_factor_p10": float(np.percentile(throughput_factor, 10)),
        "throughput_factor_p50": float(np.percentile(throughput_factor, 50)),
        "throughput_factor_p90": float(np.percentile(throughput_factor, 90)),
        "throughput_factor_max": float(np.max(throughput_factor)),
    }
    return adjusted_target, diag, delta_log_access


def _build_candidate_diagnostics(
    candidates: pd.DataFrame,
    *,
    param_names: list[str],
) -> dict[str, Any]:
    metrics = ["score", "mape_pct", "rmse", "mae", "nrmse", "r2"]
    quantiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    metric_quantiles: dict[str, dict[str, float]] = {}
    for metric in metrics:
        if metric not in candidates.columns:
            continue
        vals = pd.to_numeric(candidates[metric], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        metric_quantiles[metric] = {
            f"p{q:02d}": float(np.percentile(vals, q)) for q in quantiles
        }

    if "score" in candidates.columns:
        rank_cutoff = max(1, int(np.ceil(len(candidates) * 0.10)))
        top_slice = candidates.head(rank_cutoff)
    else:
        top_slice = candidates

    corr_rows: list[dict[str, Any]] = []
    top_summary: dict[str, dict[str, float]] = {}
    for name in param_names:
        vals = pd.to_numeric(candidates[name], errors="coerce").to_numpy(dtype=float)
        score_vals = pd.to_numeric(candidates.get("score"), errors="coerce").to_numpy(dtype=float)
        mape_vals = pd.to_numeric(candidates.get("mape_pct"), errors="coerce").to_numpy(dtype=float)

        pearson_score = _safe_corr(vals, score_vals)
        spearman_score = _safe_corr(pd.Series(vals).rank().to_numpy(dtype=float), pd.Series(score_vals).rank().to_numpy(dtype=float))
        pearson_mape = _safe_corr(vals, mape_vals)
        spearman_mape = _safe_corr(pd.Series(vals).rank().to_numpy(dtype=float), pd.Series(mape_vals).rank().to_numpy(dtype=float))
        corr_rows.append(
            {
                "parameter": name,
                "pearson_corr_score": pearson_score,
                "spearman_corr_score": spearman_score,
                "pearson_corr_mape_pct": pearson_mape,
                "spearman_corr_mape_pct": spearman_mape,
                "best_value": float(candidates.iloc[0][name]),
            }
        )

        top_vals = pd.to_numeric(top_slice[name], errors="coerce").to_numpy(dtype=float)
        top_vals = top_vals[np.isfinite(top_vals)]
        if top_vals.size:
            top_summary[name] = {
                "mean_top_decile": float(np.mean(top_vals)),
                "std_top_decile": float(np.std(top_vals)),
                "p10_top_decile": float(np.percentile(top_vals, 10)),
                "p50_top_decile": float(np.percentile(top_vals, 50)),
                "p90_top_decile": float(np.percentile(top_vals, 90)),
            }

    return {
        "candidate_count": int(len(candidates)),
        "metric_quantiles": metric_quantiles,
        "top_decile_count": int(len(top_slice)),
        "parameter_top_decile_summary": top_summary,
        "parameter_correlations": corr_rows,
    }


def _compute_metrics(pred_all: np.ndarray, obs_mask: np.ndarray, obs_vals: np.ndarray) -> dict[str, float]:
    pred_obs = pred_all[obs_mask]
    denom = float(np.dot(pred_obs, pred_obs)) + EPS
    scale = max(float(np.dot(obs_vals, pred_obs)) / denom, EPS)
    pred_scaled = pred_obs * scale
    residual = pred_scaled - obs_vals

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mape = float(np.mean(np.abs(residual) / np.maximum(obs_vals, EPS)) * 100.0)
    obs_mean = float(np.mean(obs_vals))
    nrmse = rmse / max(obs_mean, EPS)

    centered_obs = obs_vals - obs_mean
    ss_tot = float(np.dot(centered_obs, centered_obs))
    ss_res = float(np.dot(residual, residual))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > EPS else np.nan)

    # Composite objective: balance relative RMSE and MAPE.
    score = nrmse + 0.5 * (mape / 100.0)

    return {
        "scale_factor": scale,
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "nrmse": nrmse,
        "r2": r2,
        "score": score,
    }


def _evaluate_candidate(
    params: dict[str, float],
    pair_features: dict[str, np.ndarray],
    station_features: dict[str, np.ndarray],
    obs_mask: np.ndarray,
    obs_vals: np.ndarray,
    zero_diagonal: bool,
) -> tuple[dict[str, float], np.ndarray]:
    log_pop = station_features["log_pop"]
    log_lines = station_features["log_lines"]
    hub_norm = station_features["hub_norm"]

    pair_term = (
        params["freq_coef"] * pair_features["log_freq"]
        - params["dist_log_coef"] * pair_features["log_dist"]
        - params["fastest_coef"] * pair_features["fastest"]
        - params["wait_coef"] * pair_features["wait"]
        - params["rolling_coef"] * pair_features["rolling"]
        - params["dwell_coef"] * pair_features["dwell"]
        - params["transfer_coef"] * pair_features["transfer"]
    )
    origin_term = (
        params["origin_pop_exp"] * log_pop
        + params["origin_lines_exp"] * log_lines
    )
    dest_term = (
        params["dest_pop_exp"] * log_pop
        + params["dest_lines_exp"] * log_lines
    )

    log_flow = pair_term + origin_term[:, None] + dest_term[None, :]
    np.clip(log_flow, LOG_CLIP_MIN, LOG_CLIP_MAX, out=log_flow)

    flow = np.exp(log_flow)
    if zero_diagonal:
        np.fill_diagonal(flow, 0.0)

    throughput = flow.sum(axis=1) + flow.sum(axis=0)
    if params["hub_transfer_bonus"] > 0:
        throughput = throughput * (1.0 + params["hub_transfer_bonus"] * hub_norm)

    metrics = _compute_metrics(throughput, obs_mask=obs_mask, obs_vals=obs_vals)
    return metrics, throughput


def _write_matrix_csv(path: Path, station_names: list[str], matrix: np.ndarray) -> None:
    out = pd.DataFrame(matrix, index=station_names, columns=station_names)
    out.to_csv(path, index=True, float_format="%.6f")


def _compute_unscaled_error_metrics(pred: np.ndarray, obs: np.ndarray) -> dict[str, float]:
    residual = pred - obs
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    mape = float(np.mean(np.abs(residual) / np.maximum(obs, EPS)) * 100.0)
    obs_mean = float(np.mean(obs))
    centered_obs = obs - obs_mean
    ss_tot = float(np.dot(centered_obs, centered_obs))
    ss_res = float(np.dot(residual, residual))
    r2 = 1.0 - (ss_res / ss_tot if ss_tot > EPS else np.nan)
    return {
        "mae": mae,
        "rmse": rmse,
        "mape_pct": mape,
        "r2": r2,
    }


def _balance_to_station_throughput_ipf(
    matrix: np.ndarray,
    target_throughput: np.ndarray,
    zero_diagonal: bool,
    max_iter: int,
    rel_tol: float,
) -> tuple[np.ndarray, dict[str, float | int | bool]]:
    """
    Enforce station marginals with IPF/RAS under the assumption:
    row_target_i = col_target_i = target_throughput_i / 2.
    """
    target = np.maximum(target_throughput.astype(np.float64), 0.0)
    n = matrix.shape[0]
    if matrix.shape != (n, n):
        raise ValueError("Matrix must be square for IPF balancing.")
    if target.shape[0] != n:
        raise ValueError("Target throughput vector length must match matrix dimensions.")

    total_target = float(np.sum(target))
    if total_target <= EPS:
        return np.zeros_like(matrix, dtype=np.float64), {
            "enabled": True,
            "converged": True,
            "iterations": 0,
            "max_abs_row_error": 0.0,
            "max_abs_col_error": 0.0,
            "max_rel_row_error": 0.0,
            "max_rel_col_error": 0.0,
            "max_abs_throughput_error": 0.0,
            "mae_throughput_error": 0.0,
        }

    if zero_diagonal and np.any(target > (total_target - target) + 1e-9):
        raise ValueError(
            "DTV throughput targets are infeasible with strict zero-diagonal OD matrix "
            "(at least one station exceeds half of total network throughput)."
        )

    row_target = 0.5 * target
    col_target = 0.5 * target

    x = np.maximum(matrix.astype(np.float64), 0.0)
    if zero_diagonal:
        diag_mask = np.eye(n, dtype=bool)
        x[~diag_mask] = np.maximum(x[~diag_mask], EPS)
        x[diag_mask] = 0.0
    else:
        x = np.maximum(x, EPS)

    x *= (float(np.sum(row_target)) / max(float(np.sum(x)), EPS))

    converged = False
    last_row_err = np.inf
    last_col_err = np.inf
    last_row_rel = np.inf
    last_col_rel = np.inf

    for it in range(1, max_iter + 1):
        row_sum = x.sum(axis=1)
        row_scale = np.ones_like(row_target)
        row_pos = row_target > 0
        row_scale[row_pos] = row_target[row_pos] / np.maximum(row_sum[row_pos], EPS)
        row_scale[~row_pos] = 0.0
        x *= row_scale[:, None]
        if zero_diagonal:
            np.fill_diagonal(x, 0.0)

        col_sum = x.sum(axis=0)
        col_scale = np.ones_like(col_target)
        col_pos = col_target > 0
        col_scale[col_pos] = col_target[col_pos] / np.maximum(col_sum[col_pos], EPS)
        col_scale[~col_pos] = 0.0
        x *= col_scale[None, :]
        if zero_diagonal:
            np.fill_diagonal(x, 0.0)

        if it == 1 or (it % 5 == 0):
            row_err = np.abs(x.sum(axis=1) - row_target)
            col_err = np.abs(x.sum(axis=0) - col_target)
            last_row_err = float(np.max(row_err))
            last_col_err = float(np.max(col_err))
            last_row_rel = float(np.max(row_err / np.maximum(row_target, 1.0)))
            last_col_rel = float(np.max(col_err / np.maximum(col_target, 1.0)))
            if max(last_row_rel, last_col_rel) <= rel_tol:
                converged = True
                break

    final_row = x.sum(axis=1)
    final_col = x.sum(axis=0)
    throughput_err = np.abs((final_row + final_col) - target)
    diagnostics: dict[str, float | int | bool] = {
        "enabled": True,
        "converged": bool(converged),
        "iterations": int(it),
        "max_abs_row_error": float(np.max(np.abs(final_row - row_target))),
        "max_abs_col_error": float(np.max(np.abs(final_col - col_target))),
        "max_rel_row_error": float(np.max(np.abs(final_row - row_target) / np.maximum(row_target, 1.0))),
        "max_rel_col_error": float(np.max(np.abs(final_col - col_target) / np.maximum(col_target, 1.0))),
        "max_abs_throughput_error": float(np.max(throughput_err)),
        "mae_throughput_error": float(np.mean(throughput_err)),
    }
    return x, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate (or replay) a station-level gravity model and export a DTV-constrained "
            "OD matrix, with optional accessibility-induced demand uplift."
        )
    )
    parser.add_argument(
        "--model-year",
        type=int,
        default=2026,
        help="Year label used for model inputs/outputs (e.g., 2026 or 2035).",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "od_modeling_2026",
        help="Directory containing gravity input matrices/station metadata.",
    )
    parser.add_argument(
        "--stations-file",
        type=Path,
        default=None,
        help="Station metadata CSV relative to --input-dir unless absolute.",
    )
    parser.add_argument(
        "--distance-file",
        type=Path,
        default=None,
        help="Distance matrix CSV relative to --input-dir unless absolute.",
    )
    parser.add_argument(
        "--population-column",
        type=str,
        default="",
        help="Station population column in station metadata (defaults to '<model_year>_distributedpop').",
    )
    parser.add_argument(
        "--lines-column",
        type=str,
        default="",
        help="Station service-lines column in station metadata (defaults to '<model_year>lines').",
    )
    parser.add_argument(
        "--population-scaling-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "canton_population_scaling_factors.csv",
        help=(
            "CSV with canton/year growth factors (growth_factor_vs_2023), used to scale "
            "mixed-year SBB station counts to the target model year."
        ),
    )
    parser.add_argument(
        "--target-year",
        type=int,
        default=None,
        help="Target model year for population-based station-count scaling.",
    )
    parser.add_argument(
        "--fixed-params-file",
        type=Path,
        default=None,
        help=(
            "JSON file with fixed gravity parameters (either top-level params object or "
            "{'best_params': {...}}). If provided, random calibration search is skipped."
        ),
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1200,
        help="Number of random parameter combinations to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260413,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many top calibration candidates to persist in CSV output.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Log progress every N candidate evaluations.",
    )
    parser.add_argument(
        "--allow-self-demand",
        action="store_true",
        help="If set, keeps diagonal OD demand instead of forcing self-demand to zero.",
    )
    parser.add_argument(
        "--disable-dtv-marginal-fit",
        action="store_true",
        help=(
            "Disable final IPF/RAS balancing to station DTV marginals. "
            "By default, the exported OD matrix is DTV-constrained."
        ),
    )
    parser.add_argument(
        "--dtv-marginal-max-iter",
        type=int,
        default=8000,
        help="Maximum iterations for IPF/RAS balancing to DTV marginals.",
    )
    parser.add_argument(
        "--dtv-marginal-rel-tol",
        type=float,
        default=1e-8,
        help="Relative tolerance for IPF/RAS marginal convergence.",
    )
    parser.add_argument(
        "--enable-induced-demand",
        action="store_true",
        help=(
            "Apply accessibility-based induced-demand uplift to station throughput targets "
            "before IPF balancing."
        ),
    )
    parser.add_argument(
        "--induced-base-input-dir",
        type=Path,
        default=None,
        help=(
            "Baseline gravity-input directory used to compute accessibility deltas "
            "(defaults to current --input-dir)."
        ),
    )
    parser.add_argument(
        "--induced-base-model-year",
        type=int,
        default=2026,
        help="Baseline year label for induced-demand accessibility comparison.",
    )
    parser.add_argument(
        "--induced-base-stations-file",
        type=Path,
        default=None,
        help="Baseline stations file (defaults to FilteredStations<induced_base_model_year>_gravity_input.csv).",
    )
    parser.add_argument(
        "--induced-base-distance-file",
        type=Path,
        default=None,
        help="Baseline distance matrix file (defaults to <induced_base_model_year>_station_crowfly_distance_km_gravity_input.csv).",
    )
    parser.add_argument(
        "--induced-origin-elasticity",
        type=float,
        default=0.25,
        help="Elasticity from accessibility delta to origin-side induced demand factor.",
    )
    parser.add_argument(
        "--induced-dest-elasticity",
        type=float,
        default=0.25,
        help="Elasticity from accessibility delta to destination-side induced demand factor.",
    )
    parser.add_argument(
        "--induced-factor-min",
        type=float,
        default=0.70,
        help="Lower clip for induced station demand multiplier.",
    )
    parser.add_argument(
        "--induced-factor-max",
        type=float,
        default=1.80,
        help="Upper clip for induced station demand multiplier.",
    )
    parser.add_argument(
        "--induced-preserve-total-throughput",
        action="store_true",
        help="If set, re-normalize induced targets to preserve network total throughput.",
    )
    parser.add_argument(
        "--write-detailed-diagnostics",
        action="store_true",
        help="Write extra diagnostics (quantiles/correlations) for high-sample runs.",
    )
    args = parser.parse_args()

    model_year = int(args.model_year)
    target_year = int(args.target_year) if args.target_year is not None else model_year

    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    stations_file = _resolve_input_file(
        args.stations_file,
        input_dir=input_dir,
        default_name=f"FilteredStations{model_year}_gravity_input.csv",
    )
    distance_file = _resolve_input_file(
        args.distance_file,
        input_dir=input_dir,
        default_name=f"{model_year}_station_crowfly_distance_km_gravity_input.csv",
    )
    population_scaling_file = args.population_scaling_file
    if not population_scaling_file.is_absolute():
        population_scaling_file = input_dir / population_scaling_file

    matrix_paths = _build_matrix_paths(
        input_dir=input_dir,
        model_year=model_year,
        distance_file=distance_file,
    )
    for key, path in matrix_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing required {key} matrix: {path}")
    if not stations_file.exists():
        raise FileNotFoundError(f"Missing station metadata file: {stations_file}")
    if not population_scaling_file.exists():
        raise FileNotFoundError(
            f"Missing population scaling file: {population_scaling_file}"
        )

    print(f"[{_now_local()}] Loading {model_year} gravity input package from {input_dir}")
    stations = pd.read_csv(stations_file, dtype=str).fillna("")
    stations["StationName"] = stations["StationName"].astype(str)
    station_names = stations["StationName"].tolist()
    n = len(station_names)
    if n < 2:
        raise ValueError("Station list is too small for OD calibration.")

    matrices: dict[str, pd.DataFrame] = {
        key: _load_numeric_square_matrix(path) for key, path in matrix_paths.items()
    }
    for key, df in matrices.items():
        if list(df.index) != station_names:
            raise ValueError(
                f"{key} matrix station order does not match station metadata order."
            )

    population_column = args.population_column.strip() or f"{model_year}_distributedpop"
    lines_column = args.lines_column.strip() or f"{model_year}lines"
    if population_column not in stations.columns:
        raise ValueError(
            f"Station metadata missing population column '{population_column}'."
        )
    if lines_column not in stations.columns:
        raise ValueError(
            f"Station metadata missing lines column '{lines_column}'."
        )

    pop = pd.to_numeric(stations[population_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    lines = pd.to_numeric(stations[lines_column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    log_pop = np.log1p(np.maximum(pop, 0.0)).astype(np.float64)
    log_lines = np.log1p(np.maximum(lines, 0.0)).astype(np.float64)
    line_min = float(np.min(lines))
    line_max = float(np.max(lines))
    if line_max > line_min:
        hub_norm = ((lines - line_min) / (line_max - line_min)).astype(np.float64)
    else:
        hub_norm = np.zeros_like(lines, dtype=np.float64)

    pop_scale_lookup, pop_scale_skipped_rows, pop_scale_duplicate_rows = _load_population_scale_lookup(
        population_scaling_file
    )
    scale_to_target = np.ones(n, dtype=np.float64)
    stations_with_scaling_pair = 0
    stations_missing_scaling_pair = 0
    stations_with_nontrivial_scaling = 0
    station_canton = stations["NPVM_zone_canton"].astype(str)
    station_latest_year = stations["SBB_pf_latest_year"].astype(str)
    for i in range(n):
        canton = station_canton.iat[i].strip().upper()
        source_year = _parse_optional_year(station_latest_year.iat[i])
        if not canton or source_year is None:
            continue
        src_key = (canton, source_year)
        tgt_key = (canton, target_year)
        src_growth = pop_scale_lookup.get(src_key)
        tgt_growth = pop_scale_lookup.get(tgt_key)
        if src_growth is None or tgt_growth is None:
            stations_missing_scaling_pair += 1
            continue
        scale_val = max(float(tgt_growth) / float(src_growth), EPS)
        scale_to_target[i] = scale_val
        stations_with_scaling_pair += 1
        if abs(scale_val - 1.0) > 1e-10:
            stations_with_nontrivial_scaling += 1

    sbb_latest_dtv_raw = pd.to_numeric(stations["SBB_pf_latest_dtv"], errors="coerce").to_numpy(dtype=float)
    sbb_latest_dtv_scaled = sbb_latest_dtv_raw * scale_to_target

    exact_mask = (
        stations["SBB_pf_match_method"].astype(str).str.lower().eq("exact")
        & np.isfinite(sbb_latest_dtv_scaled)
        & (sbb_latest_dtv_scaled > 0.0)
    )
    obs_mask = exact_mask.to_numpy(dtype=bool)
    obs_vals = sbb_latest_dtv_scaled[obs_mask]
    obs_years = stations.loc[exact_mask, "SBB_pf_latest_year"].astype(str).tolist()
    target_dtv = pd.to_numeric(stations["DTV"], errors="coerce").to_numpy(dtype=float)
    missing_target = ~np.isfinite(target_dtv)
    target_dtv[missing_target] = sbb_latest_dtv_scaled[missing_target]
    # Exact SBB matches are always replaced by year-aligned SBB values.
    target_dtv[obs_mask] = sbb_latest_dtv_scaled[obs_mask]
    target_dtv = np.nan_to_num(target_dtv, nan=0.0, posinf=0.0, neginf=0.0)
    target_dtv = np.maximum(target_dtv, 0.0)

    pop_scale_diag = PopulationScalingDiagnostics(
        path=str(population_scaling_file),
        target_year=target_year,
        rows_loaded=len(pop_scale_lookup) + pop_scale_skipped_rows + pop_scale_duplicate_rows,
        unique_canton_year_pairs=len(pop_scale_lookup),
        skipped_rows=int(pop_scale_skipped_rows + pop_scale_duplicate_rows),
        stations_with_scaling_pair=int(stations_with_scaling_pair),
        stations_missing_scaling_pair=int(stations_missing_scaling_pair),
        stations_with_nontrivial_scaling=int(stations_with_nontrivial_scaling),
    )

    if args.fixed_params_file is None and obs_mask.sum() < 200:
        raise ValueError(
            f"Too few exact SBB matches for robust calibration: {int(obs_mask.sum())}."
        )
    if args.fixed_params_file is None and np.all(obs_vals <= 0):
        raise ValueError("Observed SBB DTV values are all non-positive.")

    print(
        f"[{_now_local()}] Stations={n}, exact SBB validation stations={int(obs_mask.sum())}, "
        f"n_samples={args.n_samples}, target_year={target_year}, "
        f"scaled_stations={pop_scale_diag.stations_with_scaling_pair}, "
        f"nontrivial_scales={pop_scale_diag.stations_with_nontrivial_scaling}"
    )

    # Pair-level features as dense arrays for fast candidate evaluation.
    freq = _fill_nan_with_median(matrices["frequency"].to_numpy(dtype=np.float64))
    fastest = _fill_nan_with_median(matrices["fastest"].to_numpy(dtype=np.float64))
    wait = _fill_nan_with_median(matrices["wait"].to_numpy(dtype=np.float64))
    rolling = _fill_nan_with_median(matrices["rolling"].to_numpy(dtype=np.float64))
    dwell = _fill_nan_with_median(matrices["dwell"].to_numpy(dtype=np.float64))
    transfer = _fill_nan_with_median(matrices["transfer"].to_numpy(dtype=np.float64))
    distance = _fill_nan_with_median(matrices["distance"].to_numpy(dtype=np.float64))

    pair_features = {
        "log_freq": np.log1p(np.maximum(freq, 0.0)),
        "log_dist": np.log1p(np.maximum(distance, 0.0)),
        "fastest": np.maximum(fastest, 0.0),
        "wait": np.maximum(wait, 0.0),
        "rolling": np.maximum(rolling, 0.0),
        "dwell": np.maximum(dwell, 0.0),
        "transfer": np.maximum(transfer, 0.0),
    }
    station_features = {
        "log_pop": log_pop,
        "log_lines": log_lines,
        "hub_norm": hub_norm,
    }

    best_params: dict[str, float] | None = None
    best_metrics: dict[str, float] | None = None
    best_throughput: np.ndarray | None = None
    rows: list[dict[str, Any]] = []

    zero_diagonal = not args.allow_self_demand
    calibration_mode = "random_search"
    if args.fixed_params_file is not None:
        fixed_params_file = args.fixed_params_file
        if not fixed_params_file.is_absolute():
            fixed_params_file = input_dir / fixed_params_file
        params = _load_fixed_params(fixed_params_file)
        calibration_mode = "fixed_params_replay"
        if int(obs_mask.sum()) > 0:
            metrics, throughput = _evaluate_candidate(
                params=params,
                pair_features=pair_features,
                station_features=station_features,
                obs_mask=obs_mask,
                obs_vals=obs_vals,
                zero_diagonal=zero_diagonal,
            )
        else:
            pair_term_for_fixed = _build_pair_term(params, pair_features)
            origin_term_for_fixed = (
                params["origin_pop_exp"] * station_features["log_pop"]
                + params["origin_lines_exp"] * station_features["log_lines"]
            )
            dest_term_for_fixed = (
                params["dest_pop_exp"] * station_features["log_pop"]
                + params["dest_lines_exp"] * station_features["log_lines"]
            )
            log_flow_for_fixed = pair_term_for_fixed + origin_term_for_fixed[:, None] + dest_term_for_fixed[None, :]
            np.clip(log_flow_for_fixed, LOG_CLIP_MIN, LOG_CLIP_MAX, out=log_flow_for_fixed)
            flow_for_fixed = np.exp(log_flow_for_fixed)
            if zero_diagonal:
                np.fill_diagonal(flow_for_fixed, 0.0)
            throughput = flow_for_fixed.sum(axis=1) + flow_for_fixed.sum(axis=0)
            throughput = throughput * (1.0 + params["hub_transfer_bonus"] * station_features["hub_norm"])
            metrics = {
                "scale_factor": 1.0,
                "mae": float("nan"),
                "rmse": float("nan"),
                "mape_pct": float("nan"),
                "nrmse": float("nan"),
                "r2": float("nan"),
                "score": float("nan"),
            }
            print(
                f"[{_now_local()}] fixed-params mode: no exact SBB validation stations; "
                "using scale_factor=1.0 and skipping calibration metrics."
            )

        rows.append(
            {
                "candidate_id": 0,
                **params,
                **metrics,
            }
        )
        best_params = params
        best_metrics = metrics
        best_throughput = throughput
    else:
        rng = np.random.default_rng(args.seed)
        candidates = pd.DataFrame(
            {
                spec.name: rng.uniform(spec.low, spec.high, size=args.n_samples)
                for spec in PARAM_SPECS
            }
        )
        for idx, row in candidates.iterrows():
            params = {name: float(row[name]) for name in candidates.columns}
            metrics, throughput = _evaluate_candidate(
                params=params,
                pair_features=pair_features,
                station_features=station_features,
                obs_mask=obs_mask,
                obs_vals=obs_vals,
                zero_diagonal=zero_diagonal,
            )
            rec: dict[str, Any] = {
                "candidate_id": int(idx),
                **params,
                **metrics,
            }
            rows.append(rec)

            if best_metrics is None or metrics["score"] < best_metrics["score"]:
                best_metrics = metrics
                best_params = params
                best_throughput = throughput

            if (idx + 1) % max(args.progress_every, 1) == 0:
                print(
                    f"[{_now_local()}] evaluated {idx + 1}/{args.n_samples}; "
                    f"current_best_score={best_metrics['score']:.6f}, "
                    f"best_mape={best_metrics['mape_pct']:.2f}%"
                )

    if best_params is None or best_metrics is None or best_throughput is None:
        raise RuntimeError("Calibration loop failed to produce a best candidate.")

    print(
        f"[{_now_local()}] Best candidate score={best_metrics['score']:.6f}, "
        f"MAPE={best_metrics['mape_pct']:.2f}%, RMSE={best_metrics['rmse']:.2f}"
    )

    # Recompute best full OD matrix for outputs.
    best_pair_term = _build_pair_term(best_params, pair_features)
    best_origin_term = (
        best_params["origin_pop_exp"] * station_features["log_pop"]
        + best_params["origin_lines_exp"] * station_features["log_lines"]
    )
    best_dest_term = (
        best_params["dest_pop_exp"] * station_features["log_pop"]
        + best_params["dest_lines_exp"] * station_features["log_lines"]
    )
    best_log_flow = best_pair_term + best_origin_term[:, None] + best_dest_term[None, :]
    np.clip(best_log_flow, LOG_CLIP_MIN, LOG_CLIP_MAX, out=best_log_flow)
    best_flow = np.exp(best_log_flow)
    if zero_diagonal:
        np.fill_diagonal(best_flow, 0.0)

    hub_multiplier = 1.0 + best_params["hub_transfer_bonus"] * station_features["hub_norm"]
    throughput_unscaled = (best_flow.sum(axis=1) + best_flow.sum(axis=0)) * hub_multiplier
    scale_factor = best_metrics["scale_factor"]
    best_flow_scaled = best_flow * scale_factor
    throughput_scaled = throughput_unscaled * scale_factor

    target_dtv_adjusted = np.array(target_dtv, copy=True)
    induced_demand_diag: dict[str, Any] = {"enabled": False}
    delta_log_access = np.zeros(n, dtype=np.float64)
    if args.enable_induced_demand:
        induced_base_input_dir = (
            args.induced_base_input_dir.resolve()
            if args.induced_base_input_dir is not None
            else input_dir
        )
        if not induced_base_input_dir.exists():
            raise FileNotFoundError(
                f"Induced-demand baseline input dir not found: {induced_base_input_dir}"
            )

        induced_base_year = int(args.induced_base_model_year)
        induced_base_stations_file = _resolve_input_file(
            args.induced_base_stations_file,
            input_dir=induced_base_input_dir,
            default_name=f"FilteredStations{induced_base_year}_gravity_input.csv",
        )
        induced_base_distance_file = _resolve_input_file(
            args.induced_base_distance_file,
            input_dir=induced_base_input_dir,
            default_name=f"{induced_base_year}_station_crowfly_distance_km_gravity_input.csv",
        )
        base_matrix_paths = _build_matrix_paths(
            input_dir=induced_base_input_dir,
            model_year=induced_base_year,
            distance_file=induced_base_distance_file,
        )
        for key, path in base_matrix_paths.items():
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing induced-demand baseline {key} matrix: {path}"
                )
        if not induced_base_stations_file.exists():
            raise FileNotFoundError(
                f"Missing induced-demand baseline stations file: {induced_base_stations_file}"
            )

        base_stations = pd.read_csv(induced_base_stations_file, dtype=str).fillna("")
        if "StationName" not in base_stations.columns:
            raise ValueError(
                "Induced-demand baseline stations file is missing required column: StationName"
            )
        base_station_names = base_stations["StationName"].astype(str).tolist()
        if len(base_station_names) != len(set(base_station_names)):
            raise ValueError(
                "Induced-demand baseline stations file contains duplicate StationName values."
            )

        base_matrices_df = {
            key: _load_numeric_square_matrix(path)
            for key, path in base_matrix_paths.items()
        }
        for key, df in base_matrices_df.items():
            if list(df.index) != base_station_names:
                raise ValueError(
                    f"Induced-demand baseline {key} matrix station order does not match "
                    f"{induced_base_stations_file.name}."
                )

        scenario_station_set = set(station_names)
        baseline_station_set = set(base_station_names)
        shared_station_names = [s for s in station_names if s in baseline_station_set]
        scenario_only_station_names = [s for s in station_names if s not in baseline_station_set]
        baseline_only_station_names = [s for s in base_station_names if s not in scenario_station_set]
        if not shared_station_names:
            raise ValueError(
                "No shared stations between scenario and induced-demand baseline station sets."
            )
        print(
            f"[{_now_local()}] induced-demand station alignment: "
            f"scenario={len(station_names)}, baseline={len(base_station_names)}, "
            f"shared={len(shared_station_names)}, "
            f"scenario_only={len(scenario_only_station_names)}, "
            f"baseline_only={len(baseline_only_station_names)}"
        )

        base_matrices = {
            key: _fill_nan_with_median(df.to_numpy(dtype=np.float64))
            for key, df in base_matrices_df.items()
        }
        base_pair_features = {
            "log_freq": np.log1p(np.maximum(base_matrices["frequency"], 0.0)),
            "log_dist": np.log1p(np.maximum(base_matrices["distance"], 0.0)),
            "fastest": np.maximum(base_matrices["fastest"], 0.0),
            "wait": np.maximum(base_matrices["wait"], 0.0),
            "rolling": np.maximum(base_matrices["rolling"], 0.0),
            "dwell": np.maximum(base_matrices["dwell"], 0.0),
            "transfer": np.maximum(base_matrices["transfer"], 0.0),
        }
        base_pair_term = _build_pair_term(best_params, base_pair_features)
        base_access_full = _compute_accessibility_from_pair_term(
            base_pair_term, zero_diagonal=zero_diagonal
        )
        scenario_access = _compute_accessibility_from_pair_term(
            best_pair_term, zero_diagonal=zero_diagonal
        )

        # Map baseline accessibility into scenario station order by station name.
        # For scenario-only stations not present in baseline, use delta=0 fallback.
        base_name_to_idx = {name: i for i, name in enumerate(base_station_names)}
        baseline_access_for_scenario = np.array(scenario_access, copy=True)
        for i, station_name in enumerate(station_names):
            j = base_name_to_idx.get(station_name)
            if j is not None:
                baseline_access_for_scenario[i] = base_access_full[j]

        target_dtv_adjusted, induced_demand_diag, delta_log_access = _compute_induced_demand_adjustment_from_access(
            scenario_access=scenario_access,
            baseline_access=baseline_access_for_scenario,
            base_target_throughput=target_dtv,
            origin_elasticity=float(args.induced_origin_elasticity),
            dest_elasticity=float(args.induced_dest_elasticity),
            factor_min=float(args.induced_factor_min),
            factor_max=float(args.induced_factor_max),
            preserve_total_throughput=bool(args.induced_preserve_total_throughput),
        )
        induced_demand_diag["baseline_input_dir"] = str(induced_base_input_dir)
        induced_demand_diag["baseline_year"] = induced_base_year
        induced_demand_diag["baseline_stations_file"] = str(induced_base_stations_file)
        induced_demand_diag["station_alignment_mode"] = "name_based_accessibility_mapping"
        induced_demand_diag["scenario_station_count"] = int(len(station_names))
        induced_demand_diag["baseline_station_count"] = int(len(base_station_names))
        induced_demand_diag["shared_station_count"] = int(len(shared_station_names))
        induced_demand_diag["scenario_only_station_count"] = int(len(scenario_only_station_names))
        induced_demand_diag["baseline_only_station_count"] = int(len(baseline_only_station_names))
        induced_demand_diag["scenario_only_station_samples"] = scenario_only_station_names[:20]
        induced_demand_diag["baseline_only_station_samples"] = baseline_only_station_names[:20]
        induced_demand_diag["scenario_only_fallback"] = (
            "baseline_access <- scenario_access (delta_log_access = 0) for scenario-only stations"
        )

    marginal_fit_diag: dict[str, float | int | bool]
    if args.disable_dtv_marginal_fit:
        final_matrix = best_flow_scaled
        final_row_sum = final_matrix.sum(axis=1)
        final_col_sum = final_matrix.sum(axis=0)
        final_throughput = final_row_sum + final_col_sum
        throughput_err = np.abs(final_throughput - target_dtv_adjusted)
        marginal_fit_diag = {
            "enabled": False,
            "converged": False,
            "iterations": 0,
            "max_abs_row_error": float("nan"),
            "max_abs_col_error": float("nan"),
            "max_rel_row_error": float("nan"),
            "max_rel_col_error": float("nan"),
            "max_abs_throughput_error": float(np.max(throughput_err)),
            "mae_throughput_error": float(np.mean(throughput_err)),
        }
    else:
        final_matrix, marginal_fit_diag = _balance_to_station_throughput_ipf(
            matrix=best_flow_scaled,
            target_throughput=target_dtv_adjusted,
            zero_diagonal=zero_diagonal,
            max_iter=max(int(args.dtv_marginal_max_iter), 10),
            rel_tol=max(float(args.dtv_marginal_rel_tol), 1e-14),
        )
        final_row_sum = final_matrix.sum(axis=1)
        final_col_sum = final_matrix.sum(axis=0)
        final_throughput = final_row_sum + final_col_sum

    # Write outputs.
    output_candidates = pd.DataFrame(rows)
    if "score" in output_candidates.columns:
        output_candidates = output_candidates.sort_values("score", ascending=True, kind="stable")
    output_prefix = str(model_year)

    output_candidates_path = input_dir / f"{output_prefix}_gravity_calibration_candidates.csv"
    output_candidates.to_csv(output_candidates_path, index=False, float_format="%.10f")

    top_k = max(1, min(args.top_k, len(output_candidates)))
    output_top_path = input_dir / f"{output_prefix}_gravity_calibration_top_candidates.csv"
    output_candidates.head(top_k).to_csv(output_top_path, index=False, float_format="%.10f")

    output_raw_matrix_path = input_dir / f"{output_prefix}_OD_matrix_Demand_gravity_best_raw.csv"
    _write_matrix_csv(output_raw_matrix_path, station_names, best_flow_scaled)

    output_matrix_path = input_dir / f"{output_prefix}_OD_matrix_Demand_gravity_best.csv"
    _write_matrix_csv(output_matrix_path, station_names, final_matrix)

    output_diagnostics_path = input_dir / f"{output_prefix}_gravity_calibration_diagnostics.json"
    output_corr_path = input_dir / f"{output_prefix}_gravity_parameter_score_correlations.csv"
    output_diagnostics: dict[str, Any] | None = None
    if args.write_detailed_diagnostics or args.n_samples >= 5000 or args.fixed_params_file is not None:
        output_diagnostics = _build_candidate_diagnostics(
            output_candidates,
            param_names=[spec.name for spec in PARAM_SPECS],
        )
        output_diagnostics_path.write_text(
            json.dumps(output_diagnostics, indent=2),
            encoding="utf-8",
        )
        corr_rows = output_diagnostics.get("parameter_correlations", [])
        if corr_rows:
            pd.DataFrame(corr_rows).to_csv(output_corr_path, index=False, float_format="%.10f")

    throughput_table = pd.DataFrame(
        {
            "StationName": station_names,
            "predicted_station_throughput_raw_model": throughput_scaled,
            "predicted_station_throughput": final_throughput,
            "target_dtv_for_constraint_base": target_dtv,
            "target_dtv_for_constraint": target_dtv_adjusted,
            "final_row_sum": final_row_sum,
            "final_col_sum": final_col_sum,
            "residual_to_target_dtv": final_throughput - target_dtv_adjusted,
            "is_exact_sbb_match": obs_mask.astype(int),
            "SBB_pf_latest_year": stations["SBB_pf_latest_year"].astype(str).tolist(),
            "SBB_pf_latest_dtv_raw": sbb_latest_dtv_raw,
            "source_year_to_target_scale_factor": scale_to_target,
            "SBB_pf_latest_dtv_scaled_to_target_year": sbb_latest_dtv_scaled,
            "SBB_pf_latest_dtv": sbb_latest_dtv_scaled,
            "induced_delta_log_accessibility": delta_log_access,
        }
    )
    throughput_table["residual"] = np.nan
    throughput_table["abs_pct_error"] = np.nan
    obs_raw = throughput_table.loc[obs_mask, "SBB_pf_latest_dtv"].to_numpy(dtype=float)
    pred_raw = throughput_table.loc[obs_mask, "predicted_station_throughput"].to_numpy(dtype=float)
    residual = pred_raw - obs_raw
    throughput_table.loc[obs_mask, "residual"] = residual
    throughput_table.loc[obs_mask, "abs_pct_error"] = (
        np.abs(residual) / np.maximum(obs_raw, EPS) * 100.0
    )
    throughput_output_path = input_dir / f"{output_prefix}_gravity_station_fit.csv"
    throughput_table.to_csv(throughput_output_path, index=False, float_format="%.8f")

    if pred_raw.size > 0 and obs_raw.size > 0:
        post_exact_metrics = _compute_unscaled_error_metrics(
            pred=pred_raw,
            obs=obs_raw,
        )
    else:
        post_exact_metrics = {
            "mae": float("nan"),
            "rmse": float("nan"),
            "mape_pct": float("nan"),
            "r2": float("nan"),
        }

    summary = {
        "created_at": _now_local(),
        "input_dir": str(input_dir),
        "model_year": model_year,
        "calibration_mode": calibration_mode,
        "station_count": int(n),
        "target_year": target_year,
        "population_column": population_column,
        "lines_column": lines_column,
        "population_scaling": {
            "path": pop_scale_diag.path,
            "target_year": pop_scale_diag.target_year,
            "rows_loaded": pop_scale_diag.rows_loaded,
            "unique_canton_year_pairs": pop_scale_diag.unique_canton_year_pairs,
            "skipped_rows": pop_scale_diag.skipped_rows,
            "stations_with_scaling_pair": pop_scale_diag.stations_with_scaling_pair,
            "stations_missing_scaling_pair": pop_scale_diag.stations_missing_scaling_pair,
            "stations_with_nontrivial_scaling": pop_scale_diag.stations_with_nontrivial_scaling,
        },
        "exact_sbb_validation_station_count": int(obs_mask.sum()),
        "validation_latest_year_counts": pd.Series(obs_years).value_counts().to_dict(),
        "n_samples": int(args.n_samples),
        "seed": int(args.seed),
        "allow_self_demand": bool(args.allow_self_demand),
        "induced_demand": induced_demand_diag,
        "dtv_marginal_fit": marginal_fit_diag,
        "best_params": {k: float(v) for k, v in best_params.items()},
        "best_metrics": {k: float(v) for k, v in best_metrics.items()},
        "post_constraint_exact_metrics": post_exact_metrics,
        "outputs": {
            "all_candidates": output_candidates_path.name,
            "top_candidates": output_top_path.name,
            "best_demand_matrix_raw": output_raw_matrix_path.name,
            "best_demand_matrix": output_matrix_path.name,
            "station_fit_table": throughput_output_path.name,
        },
    }
    if output_diagnostics is not None:
        summary["outputs"]["diagnostics_json"] = output_diagnostics_path.name
        if output_corr_path.exists():
            summary["outputs"]["parameter_correlations_csv"] = output_corr_path.name
    summary_path = input_dir / f"{output_prefix}_gravity_calibration_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[{_now_local()}] Wrote {output_candidates_path.name}")
    print(f"[{_now_local()}] Wrote {output_top_path.name}")
    print(f"[{_now_local()}] Wrote {output_raw_matrix_path.name}")
    print(f"[{_now_local()}] Wrote {output_matrix_path.name}")
    print(f"[{_now_local()}] Wrote {throughput_output_path.name}")
    if output_diagnostics is not None:
        print(f"[{_now_local()}] Wrote {output_diagnostics_path.name}")
        if output_corr_path.exists():
            print(f"[{_now_local()}] Wrote {output_corr_path.name}")
    print(f"[{_now_local()}] Wrote {summary_path.name}")
    print(f"[{_now_local()}] Done.")


if __name__ == "__main__":
    main()
