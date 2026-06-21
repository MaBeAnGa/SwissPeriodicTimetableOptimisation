#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from aggregate_adjacent_line_service_pax_from_pathalloc import (  # noqa: E402
    _aggregate_row,
    _connect,
    _export_csv,
    _get_next_row,
    _log,
    _set_meta,
)
from build_od_nondominated_paths_cells_2026_2035 import _load_station_order  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate enriched nondominated OD row-state JSONs to adjacent stop-pair "
            "daily passenger loads by line/service, without creating the giant path-allocated CSV first."
        )
    )
    parser.add_argument("--year", required=True, choices=("2026", "2035"))
    parser.add_argument("--row-state-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--state-db", type=Path, required=True)
    parser.add_argument("--station-scope", choices=("union", "year-specific"), default="year-specific")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--progress-every-rows", type=int, default=5)
    parser.add_argument("--cycle-minutes", type=int, default=120)
    parser.add_argument("--merge-zurich-hb-aliases", action="store_true")
    args = parser.parse_args()

    csv.field_size_limit(sys.maxsize)
    year = str(args.year)
    station_order = _load_station_order(year=year, station_scope=str(args.station_scope))
    rows_dir = args.row_state_root.expanduser().resolve() / year / "rows"
    output_csv = args.output_csv.expanduser().resolve()
    state_db = args.state_db.expanduser().resolve()
    progress_every = max(1, int(args.progress_every_rows))
    cycle_minutes = max(1, int(args.cycle_minutes))

    if not rows_dir.is_dir():
        raise FileNotFoundError(f"Missing row-state directory: {rows_dir}")
    if not args.resume:
        state_db.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            state_db.with_name(state_db.name + suffix).unlink(missing_ok=True)

    conn = _connect(state_db)
    start_row = _get_next_row(conn) if args.resume else 0
    _set_meta(conn, "input_row_state_root", str(args.row_state_root.expanduser().resolve()))
    _set_meta(conn, "rows_dir", str(rows_dir))
    _set_meta(conn, "output_csv", str(output_csv))
    _set_meta(conn, "year", year)
    _set_meta(conn, "cycle_minutes", str(cycle_minutes))
    conn.commit()

    _log(
        f"row-state segment aggregate start year={year} start_row={start_row} "
        f"rows_dir={rows_dir} state_db={state_db} output={output_csv}"
    )

    total_cells = 0
    total_paths = 0
    total_pax = 0.0
    n = len(station_order)
    for row_idx in range(start_row, n):
        row_path = rows_dir / f"row_{row_idx:06d}.json"
        if not row_path.is_file():
            raise FileNotFoundError(f"Missing row-state JSON: {row_path}")
        payload = json.loads(row_path.read_text(encoding="utf-8"))
        origin = str(payload.get("origin", ""))
        if origin != station_order[row_idx]:
            raise ValueError(f"{year}: row {row_idx} origin mismatch {origin!r} != {station_order[row_idx]!r}")
        cells = payload.get("cells")
        if not isinstance(cells, list) or len(cells) != n:
            raise ValueError(f"{year}: row {row_idx} has {len(cells) if isinstance(cells, list) else 'non-list'} cells, expected {n}")
        row = [origin, *cells]
        cells_seen, paths_seen, pax_seen = _aggregate_row(
            conn=conn,
            row=row,
            destinations=station_order,
            cycle_minutes=cycle_minutes,
            merge_zurich_hb_aliases=bool(args.merge_zurich_hb_aliases),
            next_row_after=row_idx + 1,
        )
        total_cells += cells_seen
        total_paths += paths_seen
        total_pax += pax_seen
        done = row_idx + 1
        if done % progress_every == 0 or done == n:
            _log(
                f"{year}: row-state segment progress rows={done}/{n} "
                f"cells={total_cells} paths={total_paths} pax={total_pax:.3f}"
            )

    _log(f"{year}: exporting segment CSV {output_csv}")
    _export_csv(conn, output_csv, cycle_minutes)
    _set_meta(conn, "completed_at", time.strftime("%Y-%m-%d %H:%M:%S %Z"))
    conn.commit()
    conn.close()
    _log(f"{year}: row-state segment aggregation complete output={output_csv}")


if __name__ == "__main__":
    main()
