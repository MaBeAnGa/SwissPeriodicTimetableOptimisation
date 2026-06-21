from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from build_all_od_pairings import (
    CANONICAL_ALL_OD_COLUMNS,
    DEFAULT_ALL_OD_PATH,
    DEFAULT_AUDIT_OUTPUT_PATH,
    DEFAULT_METHOD_PATH,
    DEFAULT_SUMMARY_PATH,
    DEFAULT_VALIDATION_PATH,
    canonicalize_all_pairings_frame,
    write_method_markdown,
    write_validation_markdown,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Finalize All_OD_Pairings outputs after a full build by archiving the verbose audit table, "
            "rewriting the canonical live CSV, and refreshing the summary/markdown sidecars."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_ALL_OD_PATH, help="Built All_OD_Pairings CSV to finalize.")
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=DEFAULT_AUDIT_OUTPUT_PATH,
        help="Path for the archived verbose audit CSV.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Path for the JSON summary file to update.",
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
        help="Path for the validation markdown sidecar to refresh.",
    )
    parser.add_argument(
        "--method-output",
        type=Path,
        default=DEFAULT_METHOD_PATH,
        help="Path for the method markdown sidecar to refresh.",
    )
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input, low_memory=False)

    # Keep a full audit export whenever the incoming file is wider than the canonical live schema.
    if list(frame.columns) != CANONICAL_ALL_OD_COLUMNS:
        should_write_audit = True
        if args.audit_output.exists():
            existing_audit = pd.read_csv(args.audit_output, nrows=1, low_memory=False)
            if len(existing_audit.columns) > len(frame.columns):
                should_write_audit = False
        if should_write_audit:
            frame.to_csv(args.audit_output, index=False)
        canonical_frame = canonicalize_all_pairings_frame(frame)
    else:
        canonical_frame = frame.copy()
        if not args.audit_output.exists():
            frame.to_csv(args.audit_output, index=False)

    canonical_frame.to_csv(args.input, index=False)

    summary = load_summary(args.summary)
    summary["outputFile"] = str(args.input)
    summary["auditOutputFile"] = str(args.audit_output)
    summary["outputRows"] = int(len(canonical_frame))
    summary["activeRows"] = int(
        ((canonical_frame["exclude_from_analysis"] == 0) & (canonical_frame["pair_weight"] > 0)).sum()
    )
    summary["excludedRows"] = int((canonical_frame["exclude_from_analysis"] == 1).sum())
    summary["inactiveYearRows"] = int((canonical_frame["analysis_status"] == "inactive_station_year").sum())
    summary["weightModels"] = sorted(set(str(item) for item in canonical_frame["weight_model"].unique()))
    summary["canonicalColumns"] = CANONICAL_ALL_OD_COLUMNS
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if summary:
        write_validation_markdown(args.validation_output, summary)
        write_method_markdown(args.method_output, summary)

    print(
        f"Finalized {args.input.name}: {len(canonical_frame)} canonical rows. "
        f"Audit copy: {args.audit_output.name}",
        flush=True,
    )


if __name__ == "__main__":
    main()
