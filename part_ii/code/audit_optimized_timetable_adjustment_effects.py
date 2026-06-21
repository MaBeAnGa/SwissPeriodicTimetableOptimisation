#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from build_od_nondominated_paths_cells_2026_2035 import _load_station_order


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


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).strip()))
    except Exception:
        return default


def _parse_mod120(value: str) -> int | None:
    try:
        h, m = str(value).strip().split(":", 1)
        return (int(h) * 60 + int(m)) % 120
    except Exception:
        return None


@dataclass(frozen=True)
class Proposal:
    proposal_id: str
    selection_order: int
    year: str
    line: str
    delta_min: int
    from_station: str
    to_station: str
    dep_old_mod: int | None
    arr_old_mod: int | None
    dep_new_mod: int | None
    arr_new_mod: int | None
    marginal_net_pax_minutes: float
    marginal_gross_gain_pax_minutes: float
    marginal_gross_loss_pax_minutes: float
    action: str


@dataclass
class CellMetrics:
    weight: float
    rolling: float
    transfer: float
    dwell: float
    total: float
    rolling_pct: float


@dataclass
class Agg:
    weight: float = 0.0
    cells: int = 0
    improved_cells: int = 0
    worsened_cells: int = 0
    unchanged_cells: int = 0
    v0_rolling: float = 0.0
    opt_rolling: float = 0.0
    v0_transfer: float = 0.0
    opt_transfer: float = 0.0
    v0_dwell: float = 0.0
    opt_dwell: float = 0.0
    v0_total: float = 0.0
    opt_total: float = 0.0
    v0_rolling_pct: float = 0.0
    opt_rolling_pct: float = 0.0
    best_examples: list[tuple[float, dict[str, Any]]] = field(default_factory=list)
    worst_examples: list[tuple[float, dict[str, Any]]] = field(default_factory=list)

    def add(
        self,
        *,
        origin: str,
        destination: str,
        v0: CellMetrics,
        opt: CellMetrics,
        example_limit: int,
    ) -> None:
        weight = max(v0.weight, opt.weight)
        if weight <= 0.0:
            return
        self.weight += weight
        self.cells += 1
        self.v0_rolling += v0.rolling * weight
        self.opt_rolling += opt.rolling * weight
        self.v0_transfer += v0.transfer * weight
        self.opt_transfer += opt.transfer * weight
        self.v0_dwell += v0.dwell * weight
        self.opt_dwell += opt.dwell * weight
        self.v0_total += v0.total * weight
        self.opt_total += opt.total * weight
        self.v0_rolling_pct += v0.rolling_pct * weight
        self.opt_rolling_pct += opt.rolling_pct * weight

        delta_total = opt.total - v0.total
        if delta_total < -1e-9:
            self.improved_cells += 1
        elif delta_total > 1e-9:
            self.worsened_cells += 1
        else:
            self.unchanged_cells += 1

        if example_limit > 0 and abs(delta_total) > 1e-9:
            record = {
                "origin": origin,
                "destination": destination,
                "daily_pax_weight": weight,
                "delta_total_min": delta_total,
                "delta_transfer_min": opt.transfer - v0.transfer,
                "delta_dwell_min": opt.dwell - v0.dwell,
                "delta_rolling_min": opt.rolling - v0.rolling,
                "delta_rolling_pct": opt.rolling_pct - v0.rolling_pct,
                "v0_total_min": v0.total,
                "opt_total_min": opt.total,
                "v0_transfer_min": v0.transfer,
                "opt_transfer_min": opt.transfer,
            }
            gain_score = -delta_total * weight
            loss_score = delta_total * weight
            if delta_total < 0.0:
                heapq.heappush(self.best_examples, (gain_score, record))
                if len(self.best_examples) > example_limit:
                    heapq.heappop(self.best_examples)
            else:
                heapq.heappush(self.worst_examples, (loss_score, record))
                if len(self.worst_examples) > example_limit:
                    heapq.heappop(self.worst_examples)

    def mean(self, attr: str) -> float:
        if self.weight <= 0.0:
            return float("nan")
        return getattr(self, attr) / self.weight

    def summary(self) -> dict[str, Any]:
        if self.weight <= 0.0:
            return {
                "cells": self.cells,
                "daily_pax_weight": self.weight,
                "status": "no_attributed_cells",
            }
        v0_total = self.mean("v0_total")
        opt_total = self.mean("opt_total")
        v0_transfer = self.mean("v0_transfer")
        opt_transfer = self.mean("opt_transfer")
        v0_dwell = self.mean("v0_dwell")
        opt_dwell = self.mean("opt_dwell")
        v0_rolling = self.mean("v0_rolling")
        opt_rolling = self.mean("opt_rolling")
        v0_pct = self.mean("v0_rolling_pct")
        opt_pct = self.mean("opt_rolling_pct")
        delta_total = opt_total - v0_total
        delta_pct = opt_pct - v0_pct
        if delta_total < -1e-6 and delta_pct >= -1e-9:
            status = "improved_total_and_rolling_share"
        elif delta_total < -1e-6:
            status = "improved_total_mixed_share"
        elif delta_total > 1e-6:
            status = "worsened_total"
        else:
            status = "unchanged_total"
        return {
            "cells": self.cells,
            "improved_cells": self.improved_cells,
            "worsened_cells": self.worsened_cells,
            "unchanged_cells": self.unchanged_cells,
            "daily_pax_weight": self.weight,
            "v0_total_min": v0_total,
            "optimized_total_min": opt_total,
            "delta_total_min": delta_total,
            "v0_transfer_min": v0_transfer,
            "optimized_transfer_min": opt_transfer,
            "delta_transfer_min": opt_transfer - v0_transfer,
            "v0_dwell_min": v0_dwell,
            "optimized_dwell_min": opt_dwell,
            "delta_dwell_min": opt_dwell - v0_dwell,
            "v0_rolling_min": v0_rolling,
            "optimized_rolling_min": opt_rolling,
            "delta_rolling_min": opt_rolling - v0_rolling,
            "v0_rolling_pct": v0_pct,
            "optimized_rolling_pct": opt_pct,
            "delta_rolling_pct": delta_pct,
            "total_pax_minutes_delta": (opt_total - v0_total) * self.weight,
            "transfer_pax_minutes_delta": (opt_transfer - v0_transfer) * self.weight,
            "status": status,
        }


def _load_proposals(path: Path, years: set[str]) -> list[Proposal]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    out: list[Proposal] = []
    for row in rows:
        year = str(row.get("year", "")).strip()
        if year not in years:
            continue
        selection_order = _as_int(row.get("selection_order"), len(out) + 1)
        proposal_id = str(row.get("physical_shift_id") or "").strip()
        if not proposal_id:
            proposal_id = "|".join(
                [
                    year,
                    str(row.get("line", "")).strip(),
                    str(row.get("delta_min", "")).strip(),
                    str(row.get("section_from_station", "")).strip(),
                    str(row.get("section_departure_time_mod120", "")).strip(),
                    str(row.get("section_to_station", "")).strip(),
                    str(row.get("section_arrival_time_mod120", "")).strip(),
                ]
            )
        out.append(
            Proposal(
                proposal_id=proposal_id,
                selection_order=selection_order,
                year=year,
                line=str(row.get("line", "")).strip(),
                delta_min=_as_int(row.get("delta_min"), 0),
                from_station=str(row.get("section_from_station", "")).strip(),
                to_station=str(row.get("section_to_station", "")).strip(),
                dep_old_mod=_parse_mod120(row.get("section_departure_time_mod120", "")),
                arr_old_mod=_parse_mod120(row.get("section_arrival_time_mod120", "")),
                dep_new_mod=_parse_mod120(row.get("section_departure_time_new_mod120", "")),
                arr_new_mod=_parse_mod120(row.get("section_arrival_time_new_mod120", "")),
                marginal_net_pax_minutes=_as_float(row.get("marginal_net_pax_minutes"), 0.0),
                marginal_gross_gain_pax_minutes=_as_float(row.get("marginal_gross_gain_pax_minutes"), 0.0),
                marginal_gross_loss_pax_minutes=_as_float(row.get("marginal_gross_loss_pax_minutes"), 0.0),
                action=str(row.get("action", "")).strip(),
            )
        )
    out.sort(key=lambda proposal: proposal.selection_order)
    return out


def _proposal_index(proposals: list[Proposal]) -> dict[tuple[str, str, str, str], list[Proposal]]:
    index: dict[tuple[str, str, str, str], list[Proposal]] = {}
    for proposal in proposals:
        key = (proposal.year, proposal.line, proposal.from_station, proposal.to_station)
        index.setdefault(key, []).append(proposal)
    return index


def _parse_canonical_events(canonical_trip_id: str) -> list[tuple[str, str, int]]:
    out: list[tuple[str, str, int]] = []
    for raw in str(canonical_trip_id or "").split("|"):
        try:
            station, role, minute = raw.rsplit("::", 2)
            out.append((station, role, int(minute) % 120))
        except Exception:
            continue
    return out


def _leg_hits_proposal(leg: dict[str, Any], proposal: Proposal) -> bool:
    if str(leg.get("service_name", "")).strip() != proposal.line:
        return False
    old_pair = (proposal.dep_old_mod, proposal.arr_old_mod)
    new_pair = (proposal.dep_new_mod, proposal.arr_new_mod)

    dep_mod = None
    arr_mod = None
    try:
        dep_mod = int(leg.get("dep_min")) % 120
        arr_mod = int(leg.get("arr_min")) % 120
    except Exception:
        pass
    if (
        str(leg.get("from_station", "")).strip() == proposal.from_station
        and str(leg.get("to_station", "")).strip() == proposal.to_station
        and (dep_mod, arr_mod) in {old_pair, new_pair}
    ):
        return True

    events = _parse_canonical_events(str(leg.get("canonical_trip_id", "")))
    for left, right in zip(events, events[1:]):
        if (
            left[0] == proposal.from_station
            and left[1] == "ab"
            and right[0] == proposal.to_station
            and right[1] == "an"
            and (left[2], right[2]) in {old_pair, new_pair}
        ):
            return True
    return False


def _path_hits(path: dict[str, Any], proposals_by_section: dict[tuple[str, str, str, str], list[Proposal]], year: str) -> set[str]:
    hits: set[str] = set()
    for leg in path.get("legs") or []:
        if not isinstance(leg, dict):
            continue
        service = str(leg.get("service_name", "")).strip()
        if not service:
            continue
        # First try direct leg endpoints.
        direct_key = (
            year,
            service,
            str(leg.get("from_station", "")).strip(),
            str(leg.get("to_station", "")).strip(),
        )
        for proposal in proposals_by_section.get(direct_key, []):
            if _leg_hits_proposal(leg, proposal):
                hits.add(proposal.proposal_id)
        # Then try all proposals on that service because a leg can span several
        # internal timetable sections inside its canonical_trip_id.
        for key, proposals in proposals_by_section.items():
            key_year, key_service, _from, _to = key
            if key_year != year or key_service != service:
                continue
            for proposal in proposals:
                if proposal.proposal_id in hits:
                    continue
                if _leg_hits_proposal(leg, proposal):
                    hits.add(proposal.proposal_id)
    return hits


def _cell_metrics_and_hits(
    cell: str,
    *,
    proposals_by_section: dict[tuple[str, str, str, str], list[Proposal]],
    year: str,
) -> tuple[CellMetrics | None, set[str]]:
    try:
        obj = json.loads(cell)
    except Exception:
        return None, set()
    paths = obj.get("paths")
    if not isinstance(paths, list) or not paths:
        return None, set()
    daily_demand = _as_float(obj.get("daily_demand_total"), 0.0)
    weight_sum = 0.0
    rolling_sum = 0.0
    transfer_sum = 0.0
    dwell_sum = 0.0
    total_sum = 0.0
    pct_sum = 0.0
    hits: set[str] = set()
    for path in paths:
        if not isinstance(path, dict):
            continue
        weight = _as_float(path.get("daily_demand_allocated"), 0.0)
        if weight <= 0.0:
            weight = daily_demand * max(_as_float(path.get("daily_demand_share"), 0.0), 0.0)
        rolling = _as_float(path.get("rolling_min"), float("nan"))
        transfer = _as_float(path.get("transfer_min"), float("nan"))
        dwell = _as_float(path.get("dwell_min"), float("nan"))
        total = rolling + transfer + dwell
        if weight > 0.0 and all(math.isfinite(x) for x in (rolling, transfer, dwell, total)) and total > 0.0:
            weight_sum += weight
            rolling_sum += rolling * weight
            transfer_sum += transfer * weight
            dwell_sum += dwell * weight
            total_sum += total * weight
            pct_sum += (rolling / total) * weight
        hits.update(_path_hits(path, proposals_by_section, year))
    if weight_sum <= 0.0:
        return None, hits
    return (
        CellMetrics(
            weight=weight_sum,
            rolling=rolling_sum / weight_sum,
            transfer=transfer_sum / weight_sum,
            dwell=dwell_sum / weight_sum,
            total=total_sum / weight_sum,
            rolling_pct=pct_sum / weight_sum,
        ),
        hits,
    )


def _row_is_enriched(payload: dict[str, Any]) -> bool:
    cells = payload.get("cells")
    if not isinstance(cells, list) or not cells:
        return False
    try:
        obj = json.loads(cells[0])
    except Exception:
        return False
    return "daily_demand_total" in obj and "path_allocation_method" in obj


def _fmt(value: Any, digits: int = 6) -> str:
    try:
        out = float(value)
    except Exception:
        return ""
    if not math.isfinite(out):
        return ""
    return f"{out:.{digits}f}".rstrip("0").rstrip(".")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    tmp.replace(path)


def audit(
    *,
    proposals: list[Proposal],
    v0_root: Path,
    optimized_root: Path,
    output_dir: Path,
    years: list[str],
    progress_every_rows: int,
    example_limit: int,
) -> dict[str, Any]:
    proposal_by_id = {proposal.proposal_id: proposal for proposal in proposals}
    proposals_by_section = _proposal_index(proposals)
    aggs = {proposal.proposal_id: Agg() for proposal in proposals}
    global_aggs = {year: Agg() for year in years}

    summary: dict[str, Any] = {
        "created_at": _now(),
        "v0_root": str(v0_root.expanduser().resolve()),
        "optimized_root": str(optimized_root.expanduser().resolve()),
        "years": {},
        "important_limitation": (
            "This is a final-state attribution audit from v0 and OPTIMIZED row JSONs. "
            "It is not a true one-change-at-a-time counterfactual rebuild. Cells are "
            "attributed to an accepted proposal when their v0 or optimized paths traverse "
            "that proposal's exact line section and old/new clockface times."
        ),
    }

    for year in years:
        station_order = _load_station_order(year=year, station_scope="year-specific")
        year_changed_cells = 0
        year_hit_cells = 0
        for row_idx, origin in enumerate(station_order):
            v0_path = v0_root.expanduser().resolve() / year / "rows" / f"row_{row_idx:06d}.json"
            opt_path = optimized_root.expanduser().resolve() / year / "rows" / f"row_{row_idx:06d}.json"
            if not v0_path.exists():
                raise FileNotFoundError(f"Missing v0 row JSON: {v0_path}")
            if not opt_path.exists():
                raise FileNotFoundError(f"Missing optimized row JSON: {opt_path}")
            v0_payload = json.loads(v0_path.read_text(encoding="utf-8"))
            opt_payload = json.loads(opt_path.read_text(encoding="utf-8"))
            if not _row_is_enriched(v0_payload):
                raise ValueError(f"v0 row is not enriched: {v0_path}")
            if not _row_is_enriched(opt_payload):
                raise ValueError(f"optimized row is not enriched: {opt_path}")
            if v0_payload.get("origin") != opt_payload.get("origin") or v0_payload.get("origin") != origin:
                raise ValueError(f"{year}: origin mismatch at row {row_idx}")
            v0_cells = v0_payload.get("cells")
            opt_cells = opt_payload.get("cells")
            if not isinstance(v0_cells, list) or not isinstance(opt_cells, list) or len(v0_cells) != len(station_order) or len(opt_cells) != len(station_order):
                raise ValueError(f"{year}: bad cell width at row {row_idx}")

            for dest_idx, destination in enumerate(station_order):
                v0_cell = v0_cells[dest_idx]
                opt_cell = opt_cells[dest_idx]
                if v0_cell == opt_cell:
                    # Unchanged cells add to the global all-network denominator,
                    # but cannot hit a changed old/new section.
                    metrics, _hits = _cell_metrics_and_hits(
                        v0_cell,
                        proposals_by_section=proposals_by_section,
                        year=year,
                    )
                    if metrics is not None:
                        global_aggs[year].add(
                            origin=origin,
                            destination=destination,
                            v0=metrics,
                            opt=metrics,
                            example_limit=0,
                        )
                    continue

                v0_metrics, v0_hits = _cell_metrics_and_hits(
                    v0_cell,
                    proposals_by_section=proposals_by_section,
                    year=year,
                )
                opt_metrics, opt_hits = _cell_metrics_and_hits(
                    opt_cell,
                    proposals_by_section=proposals_by_section,
                    year=year,
                )
                if v0_metrics is None or opt_metrics is None:
                    continue
                year_changed_cells += 1
                global_aggs[year].add(
                    origin=origin,
                    destination=destination,
                    v0=v0_metrics,
                    opt=opt_metrics,
                    example_limit=0,
                )
                hits = v0_hits | opt_hits
                if hits:
                    year_hit_cells += 1
                for proposal_id in hits:
                    if proposal_by_id[proposal_id].year != year:
                        continue
                    aggs[proposal_id].add(
                        origin=origin,
                        destination=destination,
                        v0=v0_metrics,
                        opt=opt_metrics,
                        example_limit=example_limit,
                    )

            done = row_idx + 1
            if done % progress_every_rows == 0 or done == len(station_order):
                _log(
                    f"{year}: audit progress rows={done}/{len(station_order)} "
                    f"changed_cells={year_changed_cells} attributed_hit_cells={year_hit_cells}"
                )

        summary["years"][year] = {
            "station_count": len(station_order),
            "changed_cells": year_changed_cells,
            "attributed_hit_cells": year_hit_cells,
            "global": global_aggs[year].summary(),
        }

    by_change_rows: list[dict[str, Any]] = []
    example_rows: list[dict[str, Any]] = []
    for proposal in proposals:
        agg = aggs[proposal.proposal_id]
        agg_summary = agg.summary()
        row = {
            "proposal_id": proposal.proposal_id,
            "selection_order": proposal.selection_order,
            "year": proposal.year,
            "line": proposal.line,
            "delta_min": proposal.delta_min,
            "section_from_station": proposal.from_station,
            "section_to_station": proposal.to_station,
            "action": proposal.action,
            "proxy_marginal_net_pax_minutes": proposal.marginal_net_pax_minutes,
            "proxy_marginal_gross_gain_pax_minutes": proposal.marginal_gross_gain_pax_minutes,
            "proxy_marginal_gross_loss_pax_minutes": proposal.marginal_gross_loss_pax_minutes,
            **agg_summary,
        }
        by_change_rows.append(row)
        for kind, heap in (("best_improvement", agg.best_examples), ("worst_regression", agg.worst_examples)):
            for _score, example in sorted(heap, reverse=True):
                example_rows.append(
                    {
                        "proposal_id": proposal.proposal_id,
                        "selection_order": proposal.selection_order,
                        "year": proposal.year,
                        "line": proposal.line,
                        "section_from_station": proposal.from_station,
                        "section_to_station": proposal.to_station,
                        "example_type": kind,
                        **example,
                    }
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "proposal_id",
        "selection_order",
        "year",
        "line",
        "delta_min",
        "section_from_station",
        "section_to_station",
        "action",
        "proxy_marginal_net_pax_minutes",
        "proxy_marginal_gross_gain_pax_minutes",
        "proxy_marginal_gross_loss_pax_minutes",
        "cells",
        "improved_cells",
        "worsened_cells",
        "unchanged_cells",
        "daily_pax_weight",
        "v0_total_min",
        "optimized_total_min",
        "delta_total_min",
        "v0_transfer_min",
        "optimized_transfer_min",
        "delta_transfer_min",
        "v0_dwell_min",
        "optimized_dwell_min",
        "delta_dwell_min",
        "v0_rolling_min",
        "optimized_rolling_min",
        "delta_rolling_min",
        "v0_rolling_pct",
        "optimized_rolling_pct",
        "delta_rolling_pct",
        "total_pax_minutes_delta",
        "transfer_pax_minutes_delta",
        "status",
    ]
    formatted_rows = []
    for row in by_change_rows:
        formatted_rows.append({k: _fmt(v) if isinstance(v, float) else v for k, v in row.items()})
    _write_csv(output_dir / "TimetableAdjustmentEffectAudit_by_change.csv", formatted_rows, fields)

    example_fields = [
        "proposal_id",
        "selection_order",
        "year",
        "line",
        "section_from_station",
        "section_to_station",
        "example_type",
        "origin",
        "destination",
        "daily_pax_weight",
        "delta_total_min",
        "delta_transfer_min",
        "delta_dwell_min",
        "delta_rolling_min",
        "delta_rolling_pct",
        "v0_total_min",
        "opt_total_min",
        "v0_transfer_min",
        "opt_transfer_min",
    ]
    formatted_examples = []
    for row in example_rows:
        formatted_examples.append({k: _fmt(v) if isinstance(v, float) else v for k, v in row.items()})
    _write_csv(output_dir / "TimetableAdjustmentEffectAudit_examples.csv", formatted_examples, example_fields)

    summary["proposal_count"] = len(proposals)
    summary["by_change_csv"] = str(output_dir / "TimetableAdjustmentEffectAudit_by_change.csv")
    summary["examples_csv"] = str(output_dir / "TimetableAdjustmentEffectAudit_examples.csv")
    tmp = output_dir / "TimetableAdjustmentEffectAudit_summary.json.tmp"
    tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(output_dir / "TimetableAdjustmentEffectAudit_summary.json")
    _log(f"Wrote audit outputs in {output_dir}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether accepted timetable adjustments correspond to pax-weighted "
            "OD improvements in final OPTIMIZED row JSONs versus v0 row JSONs."
        )
    )
    parser.add_argument(
        "--proposals",
        type=Path,
        default=Path(__file__).resolve().parent
        / "transfer_optimization_tables"
        / "missed_transfer_station_board_v0"
        / "ProposedTimetableChanges_MissedTransfers_StationBoardStrict.csv",
    )
    parser.add_argument("--v0-row-state-root", type=Path, required=True)
    parser.add_argument("--optimized-row-state-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--years", default="2026,2035")
    parser.add_argument("--progress-every-rows", type=int, default=5)
    parser.add_argument("--example-limit", type=int, default=5)
    args = parser.parse_args()

    years = [year.strip() for year in str(args.years).split(",") if year.strip()]
    proposals = _load_proposals(args.proposals.expanduser().resolve(), set(years))
    audit(
        proposals=proposals,
        v0_root=args.v0_row_state_root,
        optimized_root=args.optimized_row_state_root,
        output_dir=args.output_dir.expanduser().resolve(),
        years=years,
        progress_every_rows=max(1, int(args.progress_every_rows)),
        example_limit=max(0, int(args.example_limit)),
    )


if __name__ == "__main__":
    main()
