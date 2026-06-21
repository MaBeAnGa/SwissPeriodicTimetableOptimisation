#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from prepare_transfer_boards_and_baseline import (
    DAY_END_DEPART,
    DAY_START,
    _as_float,
    _as_int,
    _fmt_time,
    _parse_wait_min,
)
from setup_transfer_optimization_tables import _project_transfer_occurrences


DEFAULT_THRESHOLD_PAX = 95.0


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp.replace(path)


def _mod_min(value: str) -> int:
    hh, mm = str(value).split(":", 1)
    return (int(hh) * 60 + int(mm)) % 120


def _same_line_pair(pair: str) -> bool:
    if "->" not in pair:
        return False
    left, right = pair.split("->", 1)
    return left.strip() == right.strip()


def _candidate_id(row: dict[str, Any], sequence: int) -> str:
    return (
        f"{row['year']}|{sequence:05d}|{row['action']}|{row['line']}|{row['delta_min']}|"
        f"{row['section_from_station']}|{row['section_departure_time_mod120']}|"
        f"{row['section_to_station']}|{row['section_arrival_time_mod120']}"
    )


def _load_existing_transfer_occurrences(
    transfer_paths: dict[str, Path],
    *,
    threshold_pax: float,
) -> tuple[
    dict[tuple[str, str, str, str, int], list[dict[str, Any]]],
    dict[str, int],
]:
    index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    counts: dict[str, int] = {}
    for year, path in transfer_paths.items():
        row_count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row_number, row in enumerate(reader, start=1):
                pax = _as_float(row.get("Number of Affected Daily Pax"), 0.0)
                if pax < threshold_pax:
                    continue
                station = str(row.get("Transfer station", "")).strip()
                arr_line = str(row.get("Arrival Line", "")).strip()
                dep_line = str(row.get("Departure Line", "")).strip()
                wait = _parse_wait_min(str(row.get("Transfer Time", "")))
                occurrences = _project_transfer_occurrences(row)
                if not occurrences:
                    continue
                pax_share = pax / len(occurrences)
                od_pairs = _as_int(row.get("Number of Affected OD pairs"), 0)
                for occ_number, (arr_min, dep_min) in enumerate(occurrences):
                    if not (DAY_START <= arr_min < DAY_END_DEPART and DAY_START <= dep_min < DAY_END_DEPART):
                        continue
                    effect = {
                        "source": "existing_transfer",
                        "year": year,
                        "transfer_station": station,
                        "arrival_line": arr_line,
                        "departure_line": dep_line,
                        "arrival_minute": arr_min,
                        "departure_minute": dep_min,
                        "arrival_time": _fmt_time(arr_min),
                        "departure_time": _fmt_time(dep_min),
                        "old_wait_minutes": wait,
                        "minimum_transfer_time": None,
                        "daily_pax_proxy": pax_share,
                        "daily_pax_pattern_total": pax,
                        "od_pairs": od_pairs,
                        "effect_id": f"{year}|xfer|{row_number}|{occ_number}",
                    }
                    index[(year, station, "arrival", arr_line, arr_min % 120)].append(effect)
                    index[(year, station, "departure", dep_line, dep_min % 120)].append(effect)
                    row_count += 1
        counts[year] = row_count
        _log(f"{year}: indexed existing high-flow transfer occurrences={row_count}")
    return index, counts


def _load_near_miss_index(
    near_miss_path: Path,
) -> tuple[dict[tuple[str, str, str, str, int], list[dict[str, Any]]], dict[str, int]]:
    index: dict[tuple[str, str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    with near_miss_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, row in enumerate(reader, start=1):
            if row.get("status") != "currently_infeasible":
                continue
            year = str(row["year"])
            station = str(row["transfer_station"])
            arr_line = str(row["arrival_line"])
            dep_line = str(row["departure_line"])
            arr_min = _as_int(row.get("arrival_minute"), 0)
            dep_min = _as_int(row.get("departure_minute"), 0)
            effect = {
                "source": "near_miss",
                "year": year,
                "transfer_station": station,
                "arrival_line": arr_line,
                "departure_line": dep_line,
                "arrival_minute": arr_min,
                "departure_minute": dep_min,
                "arrival_time": row.get("arrival_time", _fmt_time(arr_min)),
                "departure_time": row.get("departure_time", _fmt_time(dep_min)),
                "old_wait_minutes": _as_int(row.get("wait_minutes"), dep_min - arr_min),
                "minimum_transfer_time": _as_int(row.get("minimum_transfer_time"), 0),
                "daily_pax_proxy": _as_float(row.get("estimated_potential_daily_pax_time_share"), 0.0),
                "daily_pax_pattern_total": "",
                "od_pairs": "",
                "effect_id": f"{year}|near|{row_number}",
            }
            index[(year, station, "arrival", arr_line, arr_min % 120)].append(effect)
            index[(year, station, "departure", dep_line, dep_min % 120)].append(effect)
            counts[year] += 1
    _log(f"Indexed infeasible near-miss events={sum(counts.values())}")
    return index, dict(counts)


def _affected_keys(candidate: dict[str, str]) -> list[tuple[str, str, str, str, int]]:
    year = candidate["year"]
    line = candidate["line"]
    return [
        (
            year,
            candidate["section_from_station"],
            "departure",
            line,
            _mod_min(candidate["section_departure_time_mod120"]),
        ),
        (
            year,
            candidate["section_to_station"],
            "arrival",
            line,
            _mod_min(candidate["section_arrival_time_mod120"]),
        ),
    ]


def _shifted_wait(effect: dict[str, Any], shifted_side: str, delta: int) -> int:
    if shifted_side == "departure":
        return int(effect["old_wait_minutes"]) + delta
    if shifted_side == "arrival":
        return int(effect["old_wait_minutes"]) - delta
    raise ValueError(f"Unexpected shifted side: {shifted_side}")


def _score_existing_effect(
    effect: dict[str, Any],
    *,
    shifted_side: str,
    delta: int,
    min_transfer_lookup: dict[tuple[str, str], int],
) -> dict[str, Any]:
    year = effect["year"]
    station = effect["transfer_station"]
    min_transfer = min_transfer_lookup.get((year, station))
    if min_transfer is None:
        min_transfer = int(effect.get("minimum_transfer_time") or 0)
    old_wait = int(effect["old_wait_minutes"])
    new_wait = _shifted_wait(effect, shifted_side, delta)
    pax = float(effect["daily_pax_proxy"])
    if new_wait < min_transfer:
        status = "broken_existing_transfer"
        pax_min_delta = -pax * max(0, old_wait - new_wait)
    else:
        change = old_wait - new_wait
        if change > 0:
            status = "improved_existing_transfer"
        elif change < 0:
            status = "worsened_existing_transfer"
        else:
            status = "unchanged_existing_transfer"
        pax_min_delta = pax * change
    return {
        **effect,
        "minimum_transfer_time": min_transfer,
        "new_wait_minutes": new_wait,
        "effect_status": status,
        "pax_minutes_delta": pax_min_delta,
        "shifted_side": shifted_side,
    }


def _score_near_miss_effect(effect: dict[str, Any], *, shifted_side: str, delta: int) -> dict[str, Any]:
    min_transfer = int(effect["minimum_transfer_time"])
    old_wait = int(effect["old_wait_minutes"])
    new_wait = _shifted_wait(effect, shifted_side, delta)
    pax = float(effect["daily_pax_proxy"])
    old_shortfall = max(0, min_transfer - old_wait)
    new_shortfall = max(0, min_transfer - new_wait)
    closed = max(0, old_shortfall - new_shortfall)
    unlocked = old_shortfall > 0 and new_shortfall == 0
    if closed > 0:
        status = "unlocked_near_miss" if unlocked else "partly_improved_near_miss"
    else:
        status = "unhelped_near_miss"
    return {
        **effect,
        "new_wait_minutes": new_wait,
        "effect_status": status,
        "pax_minutes_delta": pax * closed,
        "shifted_side": shifted_side,
    }


def _load_min_transfer_from_ledger(event_ledger_path: Path) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    with event_ledger_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[(row["year"], row["transfer_station"])] = _as_int(row.get("minimum_transfer_time"), 0)
    return out


def _classify_candidate(row: dict[str, Any]) -> tuple[str, str]:
    flags: list[str] = []
    if _as_float(row["broken_existing_daily_pax_proxy"], 0.0) > 0:
        flags.append("breaks_existing_high_flow_transfer")
    if _as_float(row["net_pax_minutes_proxy"], 0.0) <= 0:
        flags.append("non_positive_net_proxy")
    if _as_int(row["same_line_pair_count"], 0) > 0:
        flags.append("contains_same_line_transfer_pair")
    if str(row.get("feasibility_notes", "")).strip():
        flags.append("section_boundary_or_unbounded_feasibility")
    if _as_int(row["affected_existing_transfer_events"], 0) == 0 and _as_int(row["affected_near_miss_events"], 0) == 0:
        flags.append("no_matched_endpoint_effects")

    if "breaks_existing_high_flow_transfer" in flags:
        step = "exclude_or_targeted_reroute_only"
    elif "non_positive_net_proxy" in flags:
        step = "deprioritize"
    elif "contains_same_line_transfer_pair" in flags or "section_boundary_or_unbounded_feasibility" in flags:
        step = "manual_review_before_reroute"
    else:
        step = "promising_for_targeted_reroute"
    return "; ".join(flags), step


def _physical_shift_key(row: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        str(row["year"]),
        str(row["line"]),
        str(row["delta_min"]),
        str(row["section_from_station"]),
        str(row["section_departure_time_mod120"]),
        str(row["section_to_station"]),
        str(row["section_arrival_time_mod120"]),
    )


def _merge_semicolon_values(*values: Any, limit: int = 80) -> str:
    items: set[str] = set()
    for value in values:
        for part in str(value or "").split(";"):
            part = part.strip()
            if part:
                items.add(part)
    return "; ".join(sorted(items)[:limit])


def _build_physical_shift_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    merged_actions: dict[tuple[str, str, str, str, str, str, str], set[str]] = defaultdict(set)
    merged_candidate_ids: dict[tuple[str, str, str, str, str, str, str], set[str]] = defaultdict(set)

    for row in summary_rows:
        key = _physical_shift_key(row)
        merged_actions[key].add(str(row["action"]))
        merged_candidate_ids[key].add(str(row["candidate_id"]))
        item = grouped.get(key)
        if item is None or float(row["net_pax_minutes_proxy"]) > float(item["net_pax_minutes_proxy"]):
            grouped[key] = dict(row)
        else:
            # Keep the best-scoring numeric row, but do not lose text context from duplicate action labels.
            item["candidate_types"] = _merge_semicolon_values(item.get("candidate_types"), row.get("candidate_types"))
            item["transfer_stations"] = _merge_semicolon_values(item.get("transfer_stations"), row.get("transfer_stations"))
            item["affected_transfer_pairs"] = _merge_semicolon_values(item.get("affected_transfer_pairs"), row.get("affected_transfer_pairs"))
            item["net_affected_transfer_stations"] = _merge_semicolon_values(
                item.get("net_affected_transfer_stations"), row.get("net_affected_transfer_stations")
            )
            item["net_affected_transfer_pairs"] = _merge_semicolon_values(
                item.get("net_affected_transfer_pairs"), row.get("net_affected_transfer_pairs")
            )
            item["example_event_pairs"] = _merge_semicolon_values(item.get("example_event_pairs"), row.get("example_event_pairs"))
            item["reason_examples"] = _merge_semicolon_values(item.get("reason_examples"), row.get("reason_examples"), limit=10)

    out: list[dict[str, Any]] = []
    for key, row in grouped.items():
        actions = "; ".join(sorted(merged_actions[key]))
        candidate_ids = "; ".join(sorted(merged_candidate_ids[key])[:20])
        physical_id = (
            f"{row['year']}|physical|{row['line']}|{row['delta_min']}|"
            f"{row['section_from_station']}|{row['section_departure_time_mod120']}|"
            f"{row['section_to_station']}|{row['section_arrival_time_mod120']}"
        )
        merged = dict(row)
        merged["physical_shift_id"] = physical_id
        merged["actions"] = actions
        merged["source_candidate_ids"] = candidate_ids
        merged["input_candidate_rows_merged"] = len(merged_candidate_ids[key])
        out.append(merged)
    out.sort(
        key=lambda r: (
            r["recommended_next_step"] != "promising_for_targeted_reroute",
            -float(r["net_pax_minutes_proxy"]),
            -float(r["gross_gain_pax_minutes_proxy"]),
            r["year"],
            r["line"],
        )
    )
    return out


def build_net_impact(
    *,
    candidates_path: Path,
    event_ledger_path: Path,
    near_miss_path: Path,
    transfer_paths: dict[str, Path],
    output_dir: Path,
    threshold_pax: float,
    detail_top_n: int,
) -> dict[str, Any]:
    candidates = _read_csv(candidates_path)
    min_transfer_lookup = _load_min_transfer_from_ledger(event_ledger_path)
    existing_index, existing_counts = _load_existing_transfer_occurrences(transfer_paths, threshold_pax=threshold_pax)
    near_index, near_counts = _load_near_miss_index(near_miss_path)

    summary_rows: list[dict[str, Any]] = []
    detail_rows_by_candidate: dict[str, list[dict[str, Any]]] = {}
    all_detail_candidate_scores: list[tuple[float, str]] = []

    for sequence, candidate in enumerate(candidates, start=1):
        cid = _candidate_id(candidate, sequence)
        delta = int(candidate["delta_min"])
        effects: dict[tuple[str, str], dict[str, Any]] = {}

        for key in _affected_keys(candidate):
            shifted_side = key[2]
            for effect in existing_index.get(key, []):
                scored = _score_existing_effect(
                    effect,
                    shifted_side=shifted_side,
                    delta=delta,
                    min_transfer_lookup=min_transfer_lookup,
                )
                effects[(effect["source"], effect["effect_id"])] = scored
            for effect in near_index.get(key, []):
                scored = _score_near_miss_effect(effect, shifted_side=shifted_side, delta=delta)
                if scored["effect_status"] != "unhelped_near_miss":
                    effects[(effect["source"], effect["effect_id"])] = scored

        counts = Counter(str(e["effect_status"]) for e in effects.values())
        existing_gain_pax = 0.0
        existing_gain_pm = 0.0
        existing_worse_pax = 0.0
        existing_worse_pm = 0.0
        broken_pax = 0.0
        broken_pm = 0.0
        near_pax = 0.0
        near_pm = 0.0
        affected_transfer_pairs: set[str] = set()
        affected_stations: set[str] = set()
        same_line_pair_count = 0

        details: list[dict[str, Any]] = []
        for effect in effects.values():
            status = str(effect["effect_status"])
            pax = float(effect["daily_pax_proxy"])
            pm = float(effect["pax_minutes_delta"])
            pair = f"{effect['arrival_line']}->{effect['departure_line']}"
            affected_transfer_pairs.add(pair)
            affected_stations.add(str(effect["transfer_station"]))
            if _same_line_pair(pair):
                same_line_pair_count += 1
            if status == "improved_existing_transfer":
                existing_gain_pax += pax
                existing_gain_pm += pm
            elif status == "worsened_existing_transfer":
                existing_worse_pax += pax
                existing_worse_pm += -pm
            elif status == "broken_existing_transfer":
                broken_pax += pax
                broken_pm += -pm
            elif status in {"unlocked_near_miss", "partly_improved_near_miss"}:
                near_pax += pax
                near_pm += pm
            details.append(
                {
                    "candidate_id": cid,
                    "year": candidate["year"],
                    "action": candidate["action"],
                    "line": candidate["line"],
                    "delta_min": delta,
                    "source": effect["source"],
                    "effect_status": status,
                    "transfer_station": effect["transfer_station"],
                    "arrival_line": effect["arrival_line"],
                    "arrival_time": effect["arrival_time"],
                    "departure_line": effect["departure_line"],
                    "departure_time": effect["departure_time"],
                    "old_wait_minutes": effect["old_wait_minutes"],
                    "new_wait_minutes": effect["new_wait_minutes"],
                    "minimum_transfer_time": effect["minimum_transfer_time"],
                    "daily_pax_proxy": round(pax, 6),
                    "pax_minutes_delta": round(pm, 6),
                    "shifted_side": effect["shifted_side"],
                }
            )

        gross_gain_pm = existing_gain_pm + near_pm
        gross_loss_pm = existing_worse_pm + broken_pm
        net_pm = gross_gain_pm - gross_loss_pm
        out = {
            **candidate,
            "candidate_id": cid,
            "existing_gain_daily_pax_proxy": round(existing_gain_pax, 6),
            "existing_gain_pax_minutes": round(existing_gain_pm, 6),
            "near_miss_gain_daily_pax_proxy": round(near_pax, 6),
            "near_miss_gain_pax_minutes": round(near_pm, 6),
            "existing_worsened_daily_pax_proxy": round(existing_worse_pax, 6),
            "existing_worsened_pax_minutes": round(existing_worse_pm, 6),
            "broken_existing_daily_pax_proxy": round(broken_pax, 6),
            "broken_existing_pax_minutes_proxy": round(broken_pm, 6),
            "gross_gain_pax_minutes_proxy": round(gross_gain_pm, 6),
            "gross_loss_pax_minutes_proxy": round(gross_loss_pm, 6),
            "net_pax_minutes_proxy": round(net_pm, 6),
            "affected_existing_transfer_events": (
                counts["improved_existing_transfer"]
                + counts["worsened_existing_transfer"]
                + counts["broken_existing_transfer"]
                + counts["unchanged_existing_transfer"]
            ),
            "affected_near_miss_events": counts["unlocked_near_miss"] + counts["partly_improved_near_miss"],
            "unlocked_near_miss_events": counts["unlocked_near_miss"],
            "partly_improved_near_miss_events": counts["partly_improved_near_miss"],
            "broken_existing_transfer_events": counts["broken_existing_transfer"],
            "same_line_pair_count": same_line_pair_count,
            "net_affected_transfer_stations": "; ".join(sorted(affected_stations)[:30]),
            "net_affected_transfer_pairs": "; ".join(sorted(affected_transfer_pairs)[:40]),
        }
        flags, next_step = _classify_candidate(out)
        out["safety_flags"] = flags
        out["recommended_next_step"] = next_step
        summary_rows.append(out)
        detail_rows_by_candidate[cid] = sorted(details, key=lambda r: (-abs(float(r["pax_minutes_delta"])), r["transfer_station"]))[:200]
        all_detail_candidate_scores.append((float(out["net_pax_minutes_proxy"]), cid))

    summary_rows.sort(
        key=lambda r: (
            r["recommended_next_step"] != "promising_for_targeted_reroute",
            -float(r["net_pax_minutes_proxy"]),
            -float(r["gross_gain_pax_minutes_proxy"]),
            r["year"],
            r["line"],
        )
    )

    summary_fields = [
        "candidate_id",
        "year",
        "recommended_next_step",
        "safety_flags",
        "action",
        "line",
        "delta_min",
        "section_from_station",
        "section_departure_time_mod120",
        "section_to_station",
        "section_arrival_time_mod120",
        "net_pax_minutes_proxy",
        "gross_gain_pax_minutes_proxy",
        "gross_loss_pax_minutes_proxy",
        "existing_gain_daily_pax_proxy",
        "existing_gain_pax_minutes",
        "near_miss_gain_daily_pax_proxy",
        "near_miss_gain_pax_minutes",
        "existing_worsened_daily_pax_proxy",
        "existing_worsened_pax_minutes",
        "broken_existing_daily_pax_proxy",
        "broken_existing_pax_minutes_proxy",
        "affected_existing_transfer_events",
        "affected_near_miss_events",
        "unlocked_near_miss_events",
        "partly_improved_near_miss_events",
        "broken_existing_transfer_events",
        "same_line_pair_count",
        "estimated_daily_pax_proxy_sum",
        "estimated_pax_minutes_proxy_sum",
        "impact_rows",
        "candidate_types",
        "transfer_stations",
        "affected_transfer_pairs",
        "net_affected_transfer_stations",
        "net_affected_transfer_pairs",
        "example_event_pairs",
        "reason_examples",
        "min_can_shift_earlier_min",
        "min_can_shift_later_min",
        "feasibility_notes",
    ]
    summary_path = output_dir / "Transfer_Optimization_Net_Impact_Candidates_2026_2035.csv"
    _write_csv(summary_path, summary_rows, summary_fields)

    physical_rows = _build_physical_shift_rows(summary_rows)
    physical_fields = [
        "physical_shift_id",
        "actions",
        "source_candidate_ids",
        "input_candidate_rows_merged",
        *summary_fields,
    ]
    physical_path = output_dir / "Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv"
    _write_csv(physical_path, physical_rows, physical_fields)

    selected_detail_ids = {
        cid
        for _, cid in sorted(all_detail_candidate_scores, reverse=True)[: max(0, detail_top_n)]
    }
    detail_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        cid = row["candidate_id"]
        if cid in selected_detail_ids:
            detail_rows.extend(detail_rows_by_candidate.get(cid, []))
    detail_fields = [
        "candidate_id",
        "year",
        "action",
        "line",
        "delta_min",
        "source",
        "effect_status",
        "transfer_station",
        "arrival_line",
        "arrival_time",
        "departure_line",
        "departure_time",
        "old_wait_minutes",
        "new_wait_minutes",
        "minimum_transfer_time",
        "daily_pax_proxy",
        "pax_minutes_delta",
        "shifted_side",
    ]
    detail_path = output_dir / "Transfer_Optimization_Net_Impact_Details_Top.csv"
    _write_csv(detail_path, detail_rows, detail_fields)

    rec_counts = Counter(row["recommended_next_step"] for row in summary_rows)
    physical_rec_counts = Counter(row["recommended_next_step"] for row in physical_rows)
    flag_counts = Counter()
    for row in summary_rows:
        for flag in str(row["safety_flags"]).split(";"):
            flag = flag.strip()
            if flag:
                flag_counts[flag] += 1
    physical_flag_counts = Counter()
    for row in physical_rows:
        for flag in str(row["safety_flags"]).split(";"):
            flag = flag.strip()
            if flag:
                physical_flag_counts[flag] += 1
    by_year = Counter(row["year"] for row in summary_rows)
    physical_by_year = Counter(row["year"] for row in physical_rows)
    top_promising = [
        {
            "physical_shift_id": row["physical_shift_id"],
            "year": row["year"],
            "actions": row["actions"],
            "line": row["line"],
            "delta_min": row["delta_min"],
            "section": f"{row['section_from_station']}->{row['section_to_station']}",
            "net_pax_minutes_proxy": row["net_pax_minutes_proxy"],
            "gross_gain_pax_minutes_proxy": row["gross_gain_pax_minutes_proxy"],
            "gross_loss_pax_minutes_proxy": row["gross_loss_pax_minutes_proxy"],
            "safety_flags": row["safety_flags"],
        }
        for row in physical_rows
        if row["recommended_next_step"] == "promising_for_targeted_reroute"
    ][:20]
    summary = {
        "created_at": _now(),
        "threshold_pax": threshold_pax,
        "inputs": {
            "candidates": str(candidates_path),
            "event_ledger": str(event_ledger_path),
            "near_misses": str(near_miss_path),
            "transfers": {year: str(path) for year, path in transfer_paths.items()},
        },
        "outputs": {
            "net_impact_candidates": str(summary_path),
            "net_impact_physical_shifts": str(physical_path),
            "top_details": str(detail_path),
        },
        "row_counts": {
            "candidate_rows": len(candidates),
            "net_impact_rows": len(summary_rows),
            "net_impact_physical_shift_rows": len(physical_rows),
            "top_detail_rows": len(detail_rows),
            "existing_transfer_occurrences_indexed": existing_counts,
            "near_miss_events_indexed": near_counts,
        },
        "recommended_next_step_counts": dict(rec_counts),
        "physical_recommended_next_step_counts": dict(physical_rec_counts),
        "safety_flag_counts": dict(flag_counts),
        "physical_safety_flag_counts": dict(physical_flag_counts),
        "candidate_rows_by_year": dict(by_year),
        "physical_shift_rows_by_year": dict(physical_by_year),
        "top_promising_candidates": top_promising,
        "method_notes": [
            "Existing transfer effects use the transfer-discounted high-flow transfer summaries projected across the day.",
            "Near-miss gains remain proxy estimates based on event passenger shares, so they are suitable for screening but not final proof.",
            "Broken existing high-flow transfers are surfaced separately and should be excluded or rerouted explicitly before acceptance.",
            "Net proxy = existing transfer time savings + near-miss shortfall closure - worsened/broken existing transfer time.",
        ],
    }
    summary_json = output_dir / "transfer_optimization_net_impact_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build net-impact screening scores for transfer retiming candidates.")
    parser.add_argument("--tables-dir", type=Path, default=Path("transfer_optimization_tables"))
    parser.add_argument("--transfer-2026", type=Path, default=Path("Identified_All_Transfers_2026_transfer_discounted.csv"))
    parser.add_argument("--transfer-2035", type=Path, default=Path("Identified_All_Transfers_2035_transfer_discounted.csv"))
    parser.add_argument("--threshold-pax", type=float, default=DEFAULT_THRESHOLD_PAX)
    parser.add_argument("--detail-top-n", type=int, default=100)
    args = parser.parse_args()

    tables_dir = args.tables_dir.resolve()
    summary = build_net_impact(
        candidates_path=tables_dir / "Transfer_Optimization_Candidates_2026_2035.csv",
        event_ledger_path=tables_dir / "Transfer_Optimization_Event_Ledger_2026_2035.csv",
        near_miss_path=tables_dir / "Transfer_Optimization_Near_Misses_2026_2035.csv",
        transfer_paths={
            "2026": args.transfer_2026.resolve(),
            "2035": args.transfer_2035.resolve(),
        },
        output_dir=tables_dir,
        threshold_pax=args.threshold_pax,
        detail_top_n=args.detail_top_n,
    )
    _log(
        "Net-impact complete: "
        f"rows={summary['row_counts']['net_impact_rows']} "
        f"physical_shifts={summary['row_counts']['net_impact_physical_shift_rows']} "
        f"promising={summary['recommended_next_step_counts'].get('promising_for_targeted_reroute', 0)}"
    )


if __name__ == "__main__":
    main()
