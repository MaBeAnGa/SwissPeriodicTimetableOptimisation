#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PART2 = Path(__file__).resolve().parent
DEFAULT_V0_STATE = Path.home() / "od_fast_state" / "nondominated_cells_state_v0_restored_from_pathalloc_csv"
DEFAULT_STEP2_STATE = Path.home() / "od_fast_state" / "nondominated_cells_state_ACTUALLY_BETTER_OPTIMIZED"
DEFAULT_STATIONS = PART2 / "FilteredStations2026_2035.csv"
DEFAULT_MUNICIPALITIES = ROOT / "swiss_municipality_station_geometries.json"
METRICS = ("total", "rolling", "dwell", "transfer", "waiting")
QUANTILES = (0.01, 0.02, 0.03, 0.05, 0.25, 0.50, 0.75, 0.95, 0.97, 0.98, 0.99)
EPS = 1e-9


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_cell(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(str(raw or ""))
    except Exception:
        return None


def _path_weight(path: dict[str, Any], cell_demand: float) -> float:
    w = _as_float(path.get("daily_demand_allocated"), 0.0)
    if w > 0:
        return w
    share = max(_as_float(path.get("daily_demand_share"), 0.0), 0.0)
    return max(cell_demand, 0.0) * share


def _path_waiting(path: dict[str, Any]) -> float:
    instances = max(_as_float(path.get("daily_departure_instances"), 1.0), 1.0)
    dominance = _as_float(path.get("dominance_minutes_day"), float("nan"))
    if math.isfinite(dominance) and dominance >= 0:
        return dominance / instances / 2.0
    return 0.0


def _empty_acc() -> dict[str, float]:
    return {
        "weight": 0.0,
        "rolling_sum": 0.0,
        "dwell_sum": 0.0,
        "transfer_sum": 0.0,
        "waiting_sum": 0.0,
        "cells": 0.0,
        "paths": 0.0,
    }


def _add_acc(acc: dict[str, float], cm: dict[str, float]) -> None:
    w = cm.get("weight", 0.0)
    if w <= 0:
        return
    acc["weight"] += w
    acc["rolling_sum"] += cm["rolling"] * w
    acc["dwell_sum"] += cm["dwell"] * w
    acc["transfer_sum"] += cm["transfer"] * w
    acc["waiting_sum"] += cm["waiting"] * w
    acc["cells"] += 1.0
    acc["paths"] += cm.get("paths", 0.0)


def _merge_acc(target: dict[str, float], source: dict[str, float]) -> None:
    for k, v in source.items():
        target[k] = target.get(k, 0.0) + _as_float(v)


def _cell_metrics(raw: Any) -> dict[str, float] | None:
    cell = _parse_cell(raw)
    if not isinstance(cell, dict):
        return None
    paths = cell.get("paths")
    if not isinstance(paths, list) or not paths:
        return None
    cell_demand = max(_as_float(cell.get("daily_demand_total"), 0.0), 0.0)
    acc = _empty_acc()
    for path in paths:
        if not isinstance(path, dict):
            continue
        w = _path_weight(path, cell_demand)
        rolling = _as_float(path.get("rolling_min"), float("nan"))
        dwell = _as_float(path.get("dwell_min"), float("nan"))
        transfer = _as_float(path.get("transfer_min"), float("nan"))
        total = rolling + dwell + transfer
        if w <= 0 or not all(math.isfinite(v) for v in (rolling, dwell, transfer, total)) or total <= 0:
            continue
        acc["weight"] += w
        acc["rolling_sum"] += rolling * w
        acc["dwell_sum"] += dwell * w
        acc["transfer_sum"] += transfer * w
        acc["waiting_sum"] += _path_waiting(path) * w
        acc["paths"] += 1.0
    if acc["weight"] <= 0:
        return None
    w = acc["weight"]
    rolling = acc["rolling_sum"] / w
    dwell = acc["dwell_sum"] / w
    transfer = acc["transfer_sum"] / w
    waiting = acc["waiting_sum"] / w
    return {
        "weight": w,
        "rolling": rolling,
        "dwell": dwell,
        "transfer": transfer,
        "total": rolling + dwell + transfer,
        "waiting": waiting,
        "paths": acc["paths"],
    }


def _finalize_acc(acc: dict[str, float]) -> dict[str, Any]:
    w = acc.get("weight", 0.0)
    if w <= 0:
        return {"totalWeight": 0.0, "cellsWithDemand": int(acc.get("cells", 0)), "pathsWithDemand": int(acc.get("paths", 0))}
    rolling = acc["rolling_sum"] / w
    dwell = acc["dwell_sum"] / w
    transfer = acc["transfer_sum"] / w
    waiting = acc["waiting_sum"] / w
    total = rolling + dwell + transfer
    denom = max(acc["rolling_sum"] + acc["dwell_sum"] + acc["transfer_sum"], EPS)
    return {
        "totalWeight": round(w, 6),
        "avgTotalMin": round(total, 6),
        "avgRollingMin": round(rolling, 6),
        "avgDwellMin": round(dwell, 6),
        "avgTransferMin": round(transfer, 6),
        "avgWaitingMin": round(waiting, 6),
        "rollingPct": round(acc["rolling_sum"] / denom * 100.0, 6),
        "dwellPct": round(acc["dwell_sum"] / denom * 100.0, 6),
        "transferPct": round(acc["transfer_sum"] / denom * 100.0, 6),
        "cellsWithDemand": int(acc.get("cells", 0)),
        "pathsWithDemand": int(acc.get("paths", 0)),
    }


def _station_order(v0_root: Path, year: str) -> list[str]:
    meta = _read_json(v0_root / year / "metadata.json")
    order = list(meta.get("station_order") or [])
    if not order:
        raise RuntimeError(f"No station_order in {v0_root / year / 'metadata.json'}")
    return order


def _load_station_meta(stations_csv: Path, municipalities_json: Path) -> dict[str, dict[str, Any]]:
    muni_by_gde: dict[str, dict[str, str]] = {}
    if municipalities_json.is_file():
        data = _read_json(municipalities_json)
        for item in (data.get("municipalities") or {}).values():
            gde = str(item.get("gdeNr") or item.get("id") or "").strip()
            if gde:
                muni_by_gde[gde] = {
                    "municipalityCode": gde,
                    "municipalityName": str(item.get("name") or gde),
                    "municipalityCanton": str(item.get("canton") or ""),
                }
    out: dict[str, dict[str, Any]] = {}
    with stations_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station = str(row.get("StationName") or "").strip()
            if not station:
                continue
            raw_code = str(row.get("NPVM_zone_municipality_code") or "").strip()
            if raw_code.endswith(".0"):
                raw_code = raw_code[:-2]
            muni = muni_by_gde.get(raw_code, {})
            canton = str(row.get("Region") or row.get("NPVM_zone_canton") or "").strip().upper()
            out[station] = {
                "station": station,
                "country": str(row.get("Country") or "").strip(),
                "canton": canton,
                "municipalityCode": raw_code,
                "municipalityName": muni.get("municipalityName") or raw_code or station,
                "municipalityCanton": muni.get("municipalityCanton") or canton,
            }
    return out


def _aggregate_variant(state_root: Path, year: str, station_order: list[str]) -> dict[str, Any]:
    rows_dir = state_root / year / "rows"
    if not rows_dir.is_dir():
        raise FileNotFoundError(f"Missing row-state directory: {rows_dir}")
    national = _empty_acc()
    station_stats: dict[str, dict[str, float]] = {}
    for row_idx, origin in enumerate(station_order):
        row_path = rows_dir / f"row_{row_idx:06d}.json"
        payload = _read_json(row_path)
        cells = payload.get("cells") or []
        acc = _empty_acc()
        for dest_idx, raw_cell in enumerate(cells):
            if dest_idx == row_idx:
                continue
            cm = _cell_metrics(raw_cell)
            if cm is None:
                continue
            _add_acc(acc, cm)
            _add_acc(national, cm)
        if acc["weight"] > 0:
            station_stats[origin] = acc
        if (row_idx + 1) % 50 == 0 or row_idx + 1 == len(station_order):
            _log(f"{year} {state_root.name}: variant aggregation rows={row_idx+1}/{len(station_order)}")
    return {"national": national, "stationStats": station_stats}


def _weighted_quantile(values: list[tuple[float, float]], q: float) -> float | None:
    if not values:
        return None
    total = sum(w for _, w in values)
    if total <= 0:
        return None
    target = q * total
    run = 0.0
    for value, weight in sorted(values, key=lambda x: x[0]):
        run += weight
        if run >= target:
            return value
    return values[-1][0]


def _compare_variants(base_root: Path, comp_root: Path, year: str, station_order: list[str], label: str) -> dict[str, Any]:
    buckets: dict[str, list[tuple[float, float]]] = {m: [] for m in METRICS}
    sums = {m: 0.0 for m in METRICS}
    total_weight = 0.0
    changed = improved = worsened = 0
    changed_weight = improved_weight = worsened_weight = 0.0
    reach: dict[str, dict[str, Any]] = {}
    for row_idx, origin in enumerate(station_order):
        base_payload = _read_json(base_root / year / "rows" / f"row_{row_idx:06d}.json")
        comp_payload = _read_json(comp_root / year / "rows" / f"row_{row_idx:06d}.json")
        base_cells = base_payload.get("cells") or []
        comp_cells = comp_payload.get("cells") or []
        r = {"origin": origin, "fasterDestinations": 0, "slowerDestinations": 0, "paxMinutesSaved": 0.0, "paxMinutesLost": 0.0, "maxImprovementMin": 0.0, "maxImprovementDestination": "", "maxWorseningMin": 0.0, "maxWorseningDestination": ""}
        for dest_idx, (a_raw, b_raw) in enumerate(zip(base_cells, comp_cells)):
            if dest_idx == row_idx:
                continue
            a = _cell_metrics(a_raw)
            b = _cell_metrics(b_raw)
            if a is None or b is None:
                continue
            w = max(a["weight"], b["weight"])
            if w <= 0:
                continue
            total_weight += w
            deltas = {m: b[m] - a[m] for m in METRICS}
            for m, d in deltas.items():
                buckets[m].append((d, w))
                sums[m] += d * w
            dtotal = deltas["total"]
            if abs(dtotal) > EPS:
                changed += 1
                changed_weight += w
                if dtotal < 0:
                    improved += 1
                    improved_weight += w
                    saved = -dtotal * w
                    r["fasterDestinations"] += 1
                    r["paxMinutesSaved"] += saved
                    if -dtotal > r["maxImprovementMin"]:
                        r["maxImprovementMin"] = -dtotal
                        r["maxImprovementDestination"] = station_order[dest_idx]
                else:
                    worsened += 1
                    worsened_weight += w
                    lost = dtotal * w
                    r["slowerDestinations"] += 1
                    r["paxMinutesLost"] += lost
                    if dtotal > r["maxWorseningMin"]:
                        r["maxWorseningMin"] = dtotal
                        r["maxWorseningDestination"] = station_order[dest_idx]
        reach[origin] = r
        if (row_idx + 1) % 25 == 0 or row_idx + 1 == len(station_order):
            _log(f"{year} {label}: comparison rows={row_idx+1}/{len(station_order)} changed={changed}")
    stats: dict[str, Any] = {}
    for m in METRICS:
        vals = buckets[m]
        metric = {
            "weightedMean": sums[m] / total_weight if total_weight > 0 else None,
            "weightedTotal": total_weight,
            "totalPaxMinutes": sums[m],
            "min": min((v for v, _ in vals), default=None),
            "max": max((v for v, _ in vals), default=None),
        }
        for q in QUANTILES:
            metric[f"p{int(q*100):02d}"] = _weighted_quantile(vals, q)
        stats[m] = metric
    return {
        "label": label,
        "cellsCompared": len(buckets["total"]),
        "changedCells": changed,
        "changedCellShare": changed / len(buckets["total"]) if buckets["total"] else None,
        "improvedCells": improved,
        "worsenedCells": worsened,
        "changedWeight": changed_weight,
        "improvedWeight": improved_weight,
        "worsenedWeight": worsened_weight,
        "metricStats": stats,
        "reachabilityByOrigin": reach,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _spatial_rows(year: str, version: str, station_stats: dict[str, dict[str, float]], meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    def add(level: str, code: str, name: str, canton: str, metric: dict[str, float], origin_count: int = 1) -> None:
        key = (level, code, name)
        if key not in groups:
            groups[key] = {"level": level, "year": year, "version": version, "code": code, "name": name, "canton": canton, "originCount": 0, "acc": _empty_acc()}
        groups[key]["originCount"] += origin_count
        _merge_acc(groups[key]["acc"], metric)
    for station, metric in station_stats.items():
        m = meta.get(station, {})
        country = str(m.get("country") or "")
        canton = str(m.get("canton") or "")
        add("station", station, station, canton, metric, 1)
        if country == "CH" and canton:
            add("canton", canton, canton, canton, metric, 1)
            muni_code = str(m.get("municipalityCode") or "").strip()
            muni_name = str(m.get("municipalityName") or muni_code or station)
            if muni_code:
                add("municipality", muni_code, muni_name, canton, metric, 1)
    out = []
    for item in groups.values():
        metrics = _finalize_acc(item.pop("acc"))
        out.append({**item, **metrics})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step 3 report summaries from Part II row-state JSONs.")
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--v0-state-root", type=Path, default=DEFAULT_V0_STATE)
    parser.add_argument("--step2-state-root", type=Path, default=DEFAULT_STEP2_STATE)
    parser.add_argument("--step3-state-root", type=Path, required=True)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--municipalities-json", type=Path, default=DEFAULT_MUNICIPALITIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = _load_station_meta(args.stations_csv.expanduser().resolve(), args.municipalities_json.expanduser().resolve())

    payload: dict[str, Any] = {
        "createdAt": _now(),
        "stateRoots": {
            "v0": str(args.v0_state_root.expanduser().resolve()),
            "step2": str(args.step2_state_root.expanduser().resolve()),
            "step3": str(args.step3_state_root.expanduser().resolve()),
        },
        "years": {},
    }
    national_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    spatial_all: list[dict[str, Any]] = []
    reach_rows: list[dict[str, Any]] = []

    for year in args.years:
        order = _station_order(args.v0_state_root.expanduser().resolve(), year)
        variants = {
            "v0": _aggregate_variant(args.v0_state_root.expanduser().resolve(), year, order),
            "step2": _aggregate_variant(args.step2_state_root.expanduser().resolve(), year, order),
            "step3": _aggregate_variant(args.step3_state_root.expanduser().resolve(), year, order),
        }
        year_payload: dict[str, Any] = {"stationCount": len(order), "variants": {}, "comparisons": {}}
        for version, data in variants.items():
            fin = _finalize_acc(data["national"])
            year_payload["variants"][version] = fin
            national_rows.append({"year": year, "version": version, **fin})
            spatial_all.extend(_spatial_rows(year, version, data["stationStats"], meta))
        for label, base, comp in (
            ("step2_vs_v0", "v0", "step2"),
            ("step3_vs_v0", "v0", "step3"),
            ("step3_vs_step2", "step2", "step3"),
        ):
            comp_payload = _compare_variants(
                args.__dict__[f"{base}_state_root"].expanduser().resolve() if base in {"v0", "step2"} else args.step3_state_root.expanduser().resolve(),
                args.__dict__[f"{comp}_state_root"].expanduser().resolve() if comp in {"v0", "step2"} else args.step3_state_root.expanduser().resolve(),
                year,
                order,
                label,
            )
            year_payload["comparisons"][label] = {k: v for k, v in comp_payload.items() if k != "reachabilityByOrigin"}
            for metric, stats in comp_payload["metricStats"].items():
                delta_rows.append({"year": year, "comparison": label, "metric": metric, **stats, "changedCells": comp_payload["changedCells"], "changedCellShare": comp_payload["changedCellShare"], "improvedCells": comp_payload["improvedCells"], "worsenedCells": comp_payload["worsenedCells"]})
            if label in {"step3_vs_v0", "step3_vs_step2"}:
                for r in comp_payload["reachabilityByOrigin"].values():
                    reach_rows.append({"year": year, "comparison": label, **r})
        payload["years"][year] = year_payload

    (out / "step3_analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_csv(out / "step3_national_summary.csv", national_rows)
    _write_csv(out / "step3_od_delta_stats.csv", delta_rows)
    _write_csv(out / "step3_spatial_breakdown_data.csv", spatial_all)
    _write_csv(out / "step3_reachability_by_origin.csv", reach_rows)

    # Compact top/bottom helper tables for quick reporting.
    top_rows = []
    for year in args.years:
        for comparison in ("step3_vs_v0", "step3_vs_step2"):
            rr = [r for r in reach_rows if r["year"] == year and r["comparison"] == comparison]
            for mode, key, reverse in (("most_faster_destinations", "fasterDestinations", True), ("largest_pax_minutes_saved", "paxMinutesSaved", True), ("most_slower_destinations", "slowerDestinations", True), ("largest_pax_minutes_lost", "paxMinutesLost", True)):
                for rank, r in enumerate(sorted(rr, key=lambda x: _as_float(x.get(key)), reverse=reverse)[:20], start=1):
                    top_rows.append({"year": year, "comparison": comparison, "ranking": mode, "rank": rank, **r})
    _write_csv(out / "step3_reachability_top_bottom.csv", top_rows)

    _log(f"Wrote Step 3 analysis outputs to {out}")


if __name__ == "__main__":
    main()
