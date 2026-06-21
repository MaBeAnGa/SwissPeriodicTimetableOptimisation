#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from build_step3_analysis_from_row_states import (
    DEFAULT_MUNICIPALITIES,
    DEFAULT_STATIONS,
    DEFAULT_V0_STATE,
    _aggregate_variant,
    _as_float,
    _compare_variants,
    _finalize_acc,
    _load_station_meta,
    _log,
    _spatial_rows,
    _station_order,
    _write_csv,
    _now,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build corrected Step 3-only report summaries from row-state JSONs. "
            "The baseline row state is read only as the comparison reference; no Step 2 "
            "variant or Step 2-vs-baseline comparison is computed or written."
        )
    )
    parser.add_argument("--years", nargs="+", default=["2026", "2035"], choices=["2026", "2035"])
    parser.add_argument("--v0-state-root", type=Path, default=DEFAULT_V0_STATE)
    parser.add_argument("--step3-state-root", type=Path, required=True)
    parser.add_argument("--stations-csv", type=Path, default=DEFAULT_STATIONS)
    parser.add_argument("--municipalities-json", type=Path, default=DEFAULT_MUNICIPALITIES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    out = args.output_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = _load_station_meta(
        args.stations_csv.expanduser().resolve(),
        args.municipalities_json.expanduser().resolve(),
    )

    v0_root = args.v0_state_root.expanduser().resolve()
    step3_root = args.step3_state_root.expanduser().resolve()

    payload: dict[str, Any] = {
        "createdAt": _now(),
        "mode": "step3_only_corrected",
        "note": (
            "Only the corrected Step 3 variant and the Step 3-vs-baseline comparison "
            "are written. Baseline row states are read only as a fixed reference."
        ),
        "stateRoots": {
            "v0_reference": str(v0_root),
            "step3_corrected": str(step3_root),
        },
        "years": {},
    }

    national_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    spatial_all: list[dict[str, Any]] = []
    reach_rows: list[dict[str, Any]] = []

    for year in args.years:
        order = _station_order(v0_root, year)
        step3_data = _aggregate_variant(step3_root, year, order)
        step3_fin = _finalize_acc(step3_data["national"])

        year_payload: dict[str, Any] = {
            "stationCount": len(order),
            "variants": {"step3_corrected": step3_fin},
            "comparisons": {},
        }
        national_rows.append({"year": year, "version": "step3_corrected", **step3_fin})
        spatial_all.extend(_spatial_rows(year, "step3_corrected", step3_data["stationStats"], meta))

        comparison_label = "step3_vs_v0"
        comp_payload = _compare_variants(v0_root, step3_root, year, order, comparison_label)
        year_payload["comparisons"][comparison_label] = {
            k: v for k, v in comp_payload.items() if k != "reachabilityByOrigin"
        }
        for metric, stats in comp_payload["metricStats"].items():
            delta_rows.append(
                {
                    "year": year,
                    "comparison": comparison_label,
                    "metric": metric,
                    **stats,
                    "changedCells": comp_payload["changedCells"],
                    "changedCellShare": comp_payload["changedCellShare"],
                    "improvedCells": comp_payload["improvedCells"],
                    "worsenedCells": comp_payload["worsenedCells"],
                }
            )
        for row in comp_payload["reachabilityByOrigin"].values():
            reach_rows.append({"year": year, "comparison": comparison_label, **row})
        payload["years"][year] = year_payload

    (out / "step3_analysis_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(out / "step3_national_summary.csv", national_rows)
    _write_csv(out / "step3_od_delta_stats.csv", delta_rows)
    _write_csv(out / "step3_spatial_breakdown_data.csv", spatial_all)
    _write_csv(out / "step3_reachability_by_origin.csv", reach_rows)

    top_rows: list[dict[str, Any]] = []
    for year in args.years:
        rr = [r for r in reach_rows if r["year"] == year and r["comparison"] == "step3_vs_v0"]
        rankings = (
            ("most_faster_destinations", "fasterDestinations"),
            ("largest_pax_minutes_saved", "paxMinutesSaved"),
            ("most_slower_destinations", "slowerDestinations"),
            ("largest_pax_minutes_lost", "paxMinutesLost"),
        )
        for mode, key in rankings:
            for rank, row in enumerate(sorted(rr, key=lambda x: _as_float(x.get(key)), reverse=True)[:20], start=1):
                top_rows.append({"year": year, "comparison": "step3_vs_v0", "ranking": mode, "rank": rank, **row})
    _write_csv(out / "step3_reachability_top_bottom.csv", top_rows)

    _log(f"Wrote corrected Step 3-only analysis outputs to {out}")


if __name__ == "__main__":
    main()
