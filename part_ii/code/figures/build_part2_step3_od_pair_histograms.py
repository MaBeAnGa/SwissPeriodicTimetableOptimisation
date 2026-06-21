#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from array import array
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
V0 = Path.home() / 'od_fast_state' / 'nondominated_cells_state_v0_restored_from_pathalloc_csv'
STEP3 = Path.home() / 'od_fast_state' / 'step3_local_outputs' / 'STEP3_BUILD_LOCAL_FINAL' / 'row_states' / 'nondominated_cells_state_STEP3_FINAL'
OUT_PDF = ROOT / 'figures' / 'part-II' / 'part2_step3_od_pair_time_change_histograms.pdf'
OUT_PNG = ROOT / 'figures' / 'part-II' / 'part2_step3_od_pair_time_change_histograms.png'
OUT_JSON = ROOT / 'figures' / 'part-II' / 'part2_step3_od_pair_time_change_histograms_summary.json'
EPS = 1e-12
CHUNK_ROWS = 16
DEFAULT_WORKERS = min(8, os.cpu_count() or 4)
BIN_WIDTH_MIN = 0.5
UNCHANGED_HALF_WINDOW_MIN = 5.0 / 60.0
VISIBLE_Q_LOW = 0.005
VISIBLE_Q_HIGH = 0.995


def now() -> str:
    return time.strftime('%Y-%m-%d %H:%M:%S %Z')


def log(msg: str) -> None:
    print(f'[{now()}] {msg}', flush=True)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def station_order(year: str) -> list[str]:
    payload = json.loads((V0 / year / 'metadata.json').read_text(encoding='utf-8'))
    order = payload.get('station_order') or []
    if not order:
        raise RuntimeError(f'No station_order for {year}')
    return [str(x) for x in order]


def parse_cell(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    text = str(raw or '').strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        return None


def cell_metrics(raw: Any) -> tuple[float, float] | None:
    cell = parse_cell(raw)
    if not isinstance(cell, dict):
        return None
    paths = cell.get('paths')
    if not isinstance(paths, list) or not paths:
        return None

    fastest = math.inf
    total_weight = 0.0
    weighted_sum = 0.0
    cell_demand = max(as_float(cell.get('daily_demand_total'), 0.0), 0.0)

    for path in paths:
        if not isinstance(path, dict):
            continue
        rolling = as_float(path.get('rolling_min'), float('nan'))
        dwell = as_float(path.get('dwell_min'), float('nan'))
        transfer = as_float(path.get('transfer_min'), float('nan'))
        total = as_float(path.get('travel_time_min'), float('nan'))
        if not math.isfinite(total):
            total = rolling + dwell + transfer
        if not math.isfinite(total) or total <= EPS:
            continue
        if total < fastest:
            fastest = total

        weight = as_float(path.get('daily_demand_allocated'), 0.0)
        if weight <= EPS:
            weight = cell_demand * max(as_float(path.get('daily_demand_share'), 0.0), 0.0)
        if weight > EPS:
            total_weight += weight
            weighted_sum += weight * total

    if not math.isfinite(fastest) or total_weight <= EPS:
        return None
    return fastest, weighted_sum / total_weight


def _empty_stats() -> dict[str, float]:
    return {
        'count': 0,
        'sum': 0.0,
        'min': math.inf,
        'max': -math.inf,
        'positive': 0,
        'negative': 0,
        'zero_exact': 0,
        'unchanged_window': 0,
    }


def _update_stats(stats: dict[str, float], value: float) -> None:
    stats['count'] += 1
    stats['sum'] += value
    stats['min'] = min(stats['min'], value)
    stats['max'] = max(stats['max'], value)
    if value == 0:
        stats['zero_exact'] += 1
    if abs(value) <= UNCHANGED_HALF_WINDOW_MIN + 1e-12:
        stats['unchanged_window'] += 1
    elif value > 0:
        stats['positive'] += 1
    elif value < 0:
        stats['negative'] += 1


def _merge_stats(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
    return {
        'count': a['count'] + b['count'],
        'sum': a['sum'] + b['sum'],
        'min': min(a['min'], b['min']),
        'max': max(a['max'], b['max']),
        'positive': a['positive'] + b['positive'],
        'negative': a['negative'] + b['negative'],
        'zero_exact': a['zero_exact'] + b['zero_exact'],
        'unchanged_window': a['unchanged_window'] + b['unchanged_window'],
    }


def process_chunk_stats(year: str, row_ids: list[int], order_len: int) -> dict[str, Any]:
    fastest = _empty_stats()
    weighted = _empty_stats()
    rows_done = 0
    missing_rows = 0

    for row_id in row_ids:
        base_path = V0 / year / 'rows' / f'row_{row_id:06d}.json'
        step3_path = STEP3 / year / 'rows' / f'row_{row_id:06d}.json'
        if not base_path.is_file() or not step3_path.is_file():
            missing_rows += 1
            continue
        try:
            base_cells = json.loads(base_path.read_text(encoding='utf-8')).get('cells') or []
            step3_cells = json.loads(step3_path.read_text(encoding='utf-8')).get('cells') or []
        except Exception:
            missing_rows += 1
            continue

        rows_done += 1
        limit = min(len(base_cells), len(step3_cells), order_len)
        for dest_id in range(limit):
            if dest_id == row_id:
                continue
            base_metrics = cell_metrics(base_cells[dest_id])
            step3_metrics = cell_metrics(step3_cells[dest_id])
            if base_metrics is None or step3_metrics is None:
                continue
            base_fastest, base_weighted = base_metrics
            step3_fastest, step3_weighted = step3_metrics
            _update_stats(fastest, base_fastest - step3_fastest)
            _update_stats(weighted, base_weighted - step3_weighted)

    return {'rows_done': rows_done, 'missing_rows': missing_rows, 'fastest': fastest, 'weighted': weighted}


def process_chunk_changed_values(year: str, row_ids: list[int], order_len: int) -> dict[str, Any]:
    fastest = array('f')
    weighted = array('f')
    rows_done = 0
    missing_rows = 0

    for row_id in row_ids:
        base_path = V0 / year / 'rows' / f'row_{row_id:06d}.json'
        step3_path = STEP3 / year / 'rows' / f'row_{row_id:06d}.json'
        if not base_path.is_file() or not step3_path.is_file():
            missing_rows += 1
            continue
        try:
            base_cells = json.loads(base_path.read_text(encoding='utf-8')).get('cells') or []
            step3_cells = json.loads(step3_path.read_text(encoding='utf-8')).get('cells') or []
        except Exception:
            missing_rows += 1
            continue

        rows_done += 1
        limit = min(len(base_cells), len(step3_cells), order_len)
        for dest_id in range(limit):
            if dest_id == row_id:
                continue
            base_metrics = cell_metrics(base_cells[dest_id])
            step3_metrics = cell_metrics(step3_cells[dest_id])
            if base_metrics is None or step3_metrics is None:
                continue
            base_fastest, base_weighted = base_metrics
            step3_fastest, step3_weighted = step3_metrics
            d_fast = base_fastest - step3_fastest
            d_weight = base_weighted - step3_weighted
            if abs(d_fast) > UNCHANGED_HALF_WINDOW_MIN + 1e-12:
                fastest.append(float(d_fast))
            if abs(d_weight) > UNCHANGED_HALF_WINDOW_MIN + 1e-12:
                weighted.append(float(d_weight))

    return {'rows_done': rows_done, 'missing_rows': missing_rows, 'fastest': fastest, 'weighted': weighted}


def collect_stats_year(year: str, workers: int) -> dict[str, Any]:
    order = station_order(year)
    row_chunks = [list(range(i, min(i + CHUNK_ROWS, len(order)))) for i in range(0, len(order), CHUNK_ROWS)]
    agg_fast = _empty_stats()
    agg_weighted = _empty_stats()
    rows_done = 0
    missing_rows = 0
    t0 = time.time()
    log(f'{year}: pass 1/2 (ranges and means) with {len(order)} rows in {len(row_chunks)} chunks using {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_chunk_stats, year, chunk, len(order)) for chunk in row_chunks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            payload = fut.result()
            agg_fast = _merge_stats(agg_fast, payload['fastest'])
            agg_weighted = _merge_stats(agg_weighted, payload['weighted'])
            rows_done += int(payload['rows_done'])
            missing_rows += int(payload['missing_rows'])
            if idx % 10 == 0 or idx == len(futures):
                log(f'{year}: pass1 chunks {idx}/{len(futures)} rows={rows_done}/{len(order)} compared_od={agg_fast["count"]} elapsed={time.time()-t0:.1f}s')
    return {'year': year, 'rows': len(order), 'rows_done': rows_done, 'missing_rows': missing_rows, 'fastest': agg_fast, 'weighted': agg_weighted}


def collect_changed_values_year(year: str, workers: int) -> dict[str, Any]:
    order = station_order(year)
    row_chunks = [list(range(i, min(i + CHUNK_ROWS, len(order)))) for i in range(0, len(order), CHUNK_ROWS)]
    fastest = array('f')
    weighted = array('f')
    rows_done = 0
    missing_rows = 0
    t0 = time.time()
    log(f'{year}: pass 2/2 (collect changed values) with {len(order)} rows in {len(row_chunks)} chunks using {workers} workers')
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(process_chunk_changed_values, year, chunk, len(order)) for chunk in row_chunks]
        for idx, fut in enumerate(as_completed(futures), start=1):
            payload = fut.result()
            fastest.extend(payload['fastest'])
            weighted.extend(payload['weighted'])
            rows_done += int(payload['rows_done'])
            missing_rows += int(payload['missing_rows'])
            if idx % 10 == 0 or idx == len(futures):
                log(f'{year}: pass2 chunks {idx}/{len(futures)} rows={rows_done}/{len(order)} changed_fast={len(fastest):,} changed_weighted={len(weighted):,} elapsed={time.time()-t0:.1f}s')
    return {
        'year': year,
        'rows_done': rows_done,
        'missing_rows': missing_rows,
        'fastest': np.asarray(fastest, dtype=np.float32),
        'weighted': np.asarray(weighted, dtype=np.float32),
    }


def summarise(stats: dict[str, float], changed: np.ndarray) -> dict[str, Any]:
    count = int(stats['count'])
    mean = stats['sum'] / count if count else None
    out = {
        'count': count,
        'mean_improvement_min': mean,
        'min_improvement_min': None if count == 0 else stats['min'],
        'max_improvement_min': None if count == 0 else stats['max'],
        'positive_count_outside_unchanged': int(stats['positive']),
        'negative_count_outside_unchanged': int(stats['negative']),
        'zero_exact_count': int(stats['zero_exact']),
        'unchanged_within_5s_count': int(stats['unchanged_window']),
        'changed_outside_5s_count': int(stats['positive'] + stats['negative']),
    }
    if changed.size:
        out.update({
            'changed_p005_min': float(np.quantile(changed, VISIBLE_Q_LOW)),
            'changed_p995_min': float(np.quantile(changed, VISIBLE_Q_HIGH)),
            'changed_median_min': float(np.median(changed)),
        })
    return out


def choose_visible_range(changed_payloads: dict[str, dict[str, np.ndarray]]) -> tuple[float, float]:
    vals = []
    for year in ('2026', '2035'):
        for metric in ('fastest', 'weighted'):
            arr = changed_payloads[year][metric]
            if arr.size:
                vals.append(np.quantile(arr, VISIBLE_Q_LOW))
                vals.append(np.quantile(arr, VISIBLE_Q_HIGH))
    low = min(vals)
    high = max(vals)
    xmin = math.floor(low / BIN_WIDTH_MIN) * BIN_WIDTH_MIN
    xmax = math.ceil(high / BIN_WIDTH_MIN) * BIN_WIDTH_MIN
    if xmin >= -UNCHANGED_HALF_WINDOW_MIN:
        xmin = -UNCHANGED_HALF_WINDOW_MIN - BIN_WIDTH_MIN
    if xmax <= UNCHANGED_HALF_WINDOW_MIN:
        xmax = UNCHANGED_HALF_WINDOW_MIN + BIN_WIDTH_MIN
    return xmin, xmax


def make_edges(xmin: float, xmax: float) -> np.ndarray:
    neg_edges = np.arange(xmin, -UNCHANGED_HALF_WINDOW_MIN + 1e-12, BIN_WIDTH_MIN, dtype=np.float64)
    pos_edges = np.arange(UNCHANGED_HALF_WINDOW_MIN, xmax + BIN_WIDTH_MIN + 1e-12, BIN_WIDTH_MIN, dtype=np.float64)
    return np.unique(np.round(np.concatenate([neg_edges, pos_edges]), 10))


def hist_for_visible(arr: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, int, int]:
    left = int(np.sum(arr < edges[0])) if arr.size else 0
    right = int(np.sum(arr >= edges[-1])) if arr.size else 0
    inside = arr[(arr >= edges[0]) & (arr < edges[-1])] if arr.size else arr
    counts, _ = np.histogram(inside, bins=edges)
    return counts.astype(np.int64), left, right


def plot(stats_payloads: dict[str, dict[str, Any]], changed_payloads: dict[str, dict[str, np.ndarray]], edges: np.ndarray) -> None:
    centers = (edges[:-1] + edges[1:]) / 2.0
    widths = np.diff(edges)
    colors = np.where(centers > 0, '#4f81bd', '#d97979')

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.0), sharex=True)
    plt.subplots_adjust(hspace=0.30, wspace=0.14, left=0.07, right=0.985, top=0.90, bottom=0.10)

    layout = [
        ('2026', 'fastest', '2026: Fastest retained path'),
        ('2026', 'weighted', '2026: Dominance-weighted mean path'),
        ('2035', 'fastest', '2035: Fastest retained path'),
        ('2035', 'weighted', '2035: Dominance-weighted mean path'),
    ]

    for ax, (year, metric, title) in zip(axes.flat, layout):
        arr = changed_payloads[year][metric]
        stats = stats_payloads[year][metric]
        counts, left_off, right_off = hist_for_visible(arr, edges)
        plot_counts = np.where(counts > 0, counts, np.nan)
        ax.bar(centers, plot_counts, width=widths, color=colors, edgecolor='none', align='center')
        ax.axvline(0, color='black', linewidth=1.0, alpha=0.9)
        ax.set_yscale('log')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.grid(axis='y', color='#d9d9d9', linewidth=0.8, alpha=0.8)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.text(
            0.985,
            0.96,
            (
                f"Compared OD pairs: {int(stats['count']):,}\n"
                f"Unchanged (±5 s): {int(stats['unchanged_window']):,}\n"
                f"Shown in bars: {arr.size - left_off - right_off:,}\n"
                f"Off-scale left/right: {left_off:,} / {right_off:,}\n"
                f"Mean improvement: {stats['sum']/stats['count']:+.2f} min"
            ),
            transform=ax.transAxes,
            ha='right',
            va='top',
            fontsize=10,
            bbox=dict(boxstyle='round,pad=0.28', facecolor='white', edgecolor='#d0d0d0', alpha=0.96),
        )

    axes[0, 0].set_ylabel('OD-pair count (log scale)')
    axes[1, 0].set_ylabel('OD-pair count (log scale)')
    axes[1, 0].set_xlabel('Baseline minus post-optimization travel time [min]')
    axes[1, 1].set_xlabel('Baseline minus post-optimization travel time [min]')

    fig.suptitle('Step 3 OD-pair travel-time changes relative to Baseline', fontsize=18, fontweight='bold', y=0.965)
    fig.text(
        0.5,
        0.935,
        'Positive values mean lower post-optimization in-system time. Histograms use 0.5-minute bins and show the central changed range only; the ±5-second unchanged band and off-scale tails are reported in each panel.',
        ha='center',
        fontsize=11,
    )

    fig.savefig(OUT_PDF, bbox_inches='tight')
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    workers = DEFAULT_WORKERS
    log(f'Using {workers} workers')
    stats = {
        '2026': collect_stats_year('2026', workers),
        '2035': collect_stats_year('2035', workers),
    }
    changed = {
        '2026': collect_changed_values_year('2026', workers),
        '2035': collect_changed_values_year('2035', workers),
    }
    xmin, xmax = choose_visible_range({y: {'fastest': changed[y]['fastest'], 'weighted': changed[y]['weighted']} for y in ('2026','2035')})
    edges = make_edges(xmin, xmax)
    summary = {
        year: {
            'fastest': summarise(stats[year]['fastest'], changed[year]['fastest']),
            'weighted': summarise(stats[year]['weighted'], changed[year]['weighted']),
            'rows': int(stats[year]['rows']),
            'rows_done': int(stats[year]['rows_done']),
            'missing_rows': int(stats[year]['missing_rows']),
        }
        for year in ('2026', '2035')
    }
    summary['bin_width_min'] = BIN_WIDTH_MIN
    summary['unchanged_half_window_min'] = UNCHANGED_HALF_WINDOW_MIN
    summary['visible_quantiles'] = {'low': VISIBLE_Q_LOW, 'high': VISIBLE_Q_HIGH}
    summary['visible_x_range_min'] = [float(xmin), float(xmax)]
    summary['bin_edges_min'] = [float(x) for x in edges.tolist()]
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    plot(stats, {y: {'fastest': changed[y]['fastest'], 'weighted': changed[y]['weighted']} for y in ('2026','2035')}, edges)
    log(f'Wrote {OUT_PDF}')
    log(f'Wrote {OUT_PNG}')
    log(f'Wrote {OUT_JSON}')


if __name__ == '__main__':
    main()
