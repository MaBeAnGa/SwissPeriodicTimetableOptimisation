#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sqlite3
import sys
import time
from functools import lru_cache
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


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except Exception:
        return default


def _hhmm(minute: int | None) -> str:
    if minute is None or minute < 0:
        return "NA"
    minute = int(minute) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


def _match_station(name: str) -> str:
    s = str(name).strip()
    if s.startswith("Zürich HB"):
        return "Zürich HB"
    return s


def _display_station(name: str, *, merge_zurich_hb_aliases: bool) -> str:
    s = str(name).strip()
    if merge_zurich_hb_aliases and s.startswith("Zürich HB"):
        return "Zürich HB"
    return s


@lru_cache(maxsize=50000)
def _parse_canonical_trip(canonical_trip_id: str) -> tuple[tuple[str, int | None, int | None], ...]:
    events: list[dict[str, Any]] = []
    for token in str(canonical_trip_id or "").split("|"):
        parts = token.split("::")
        if len(parts) != 3:
            continue
        station, kind, minute_s = parts
        minute = _as_int(minute_s, None)
        if minute is None:
            continue
        if not events or events[-1]["station"] != station:
            events.append({"station": station, "an": None, "ab": None})
        if kind == "an":
            events[-1]["an"] = minute
        elif kind == "ab":
            events[-1]["ab"] = minute
    return tuple((str(e["station"]), e["an"], e["ab"]) for e in events)


def _find_leg_event(
    leg: dict[str, Any],
    *,
    merge_zurich_hb_aliases: bool,
) -> tuple[str, str, int | None, int | None, int | None, int | None]:
    """Return station/time tuple for the exact adjacent leg.

    The OD encoder normalises Zürich HB terminal origins/destinations to
    "Zürich HB", while canonical trips may still carry platform aliases such
    as "Zürich HB (31-34)".  Matching therefore collapses HB aliases only for
    lookup, but preserves the canonical station label in the output unless the
    caller explicitly requests display-level alias merging.
    """

    from_station = str(leg.get("from_station", "")).strip()
    to_station = str(leg.get("to_station", "")).strip()
    leg_dep = _as_int(leg.get("dep_min"), None)
    leg_arr = _as_int(leg.get("arr_min"), None)
    canonical = str(leg.get("canonical_trip_id", "") or "")
    events = _parse_canonical_trip(canonical)
    from_key = _match_station(from_station)
    to_key = _match_station(to_station)

    candidates: list[tuple[str, str, int | None, int | None, int | None, int | None]] = []
    for i in range(len(events) - 1):
        st1, arr1, dep1 = events[i]
        st2, arr2, dep2 = events[i + 1]
        if _match_station(st1) != from_key or _match_station(st2) != to_key:
            continue
        candidates.append(
            (
                _display_station(st1, merge_zurich_hb_aliases=merge_zurich_hb_aliases),
                _display_station(st2, merge_zurich_hb_aliases=merge_zurich_hb_aliases),
                arr1,
                dep1,
                arr2,
                dep2,
            )
        )

    if candidates:
        if leg_dep is not None or leg_arr is not None:
            for candidate in candidates:
                _, _, _, dep1, arr2, _ = candidate
                dep_ok = leg_dep is None or dep1 == leg_dep
                arr_ok = leg_arr is None or arr2 == leg_arr
                if dep_ok and arr_ok:
                    return candidate
        return candidates[0]

    # Fallback for malformed/older cells with no parseable canonical trip id.
    return (
        _display_station(from_station, merge_zurich_hb_aliases=merge_zurich_hb_aliases),
        _display_station(to_station, merge_zurich_hb_aliases=merge_zurich_hb_aliases),
        None,
        leg_dep,
        leg_arr,
        None,
    )


def _event_mod(minute: int | None, cycle_minutes: int) -> int:
    if minute is None or minute < 0:
        return -1
    return int(minute) % int(cycle_minutes)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-100000")
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS pax (
            station_a TEXT NOT NULL,
            station_b TEXT NOT NULL,
            line TEXT NOT NULL,
            daily_pax REAL NOT NULL DEFAULT 0,
            PRIMARY KEY (station_a, station_b, line)
        );

        CREATE TABLE IF NOT EXISTS events (
            station_a TEXT NOT NULL,
            station_b TEXT NOT NULL,
            line TEXT NOT NULL,
            arr_a_mod INTEGER NOT NULL,
            dep_a_mod INTEGER NOT NULL,
            arr_b_mod INTEGER NOT NULL,
            dep_b_mod INTEGER NOT NULL,
            arr_a_min INTEGER,
            dep_a_min INTEGER,
            arr_b_min INTEGER,
            dep_b_min INTEGER,
            PRIMARY KEY (
                station_a, station_b, line,
                arr_a_mod, dep_a_mod, arr_b_mod, dep_b_mod
            )
        );

        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    conn.commit()
    return conn


def _get_next_row(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM meta WHERE key='next_row'").fetchone()
    if row is None:
        return 0
    return max(0, int(row[0]))


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )


def _aggregate_row(
    *,
    conn: sqlite3.Connection,
    row: list[str],
    destinations: list[str],
    cycle_minutes: int,
    merge_zurich_hb_aliases: bool,
    next_row_after: int,
) -> tuple[int, int, float]:
    segment_pax: dict[tuple[str, str, str], float] = {}
    event_records: dict[tuple[str, str, str, int, int, int, int], tuple[int | None, int | None, int | None, int | None]] = {}
    cells_seen = 0
    paths_seen = 0
    pax_seen = 0.0

    for cell in row[1:]:
        if '"paths":[]' in cell:
            continue
        try:
            obj = json.loads(cell)
        except Exception:
            continue
        paths = obj.get("paths", [])
        if not isinstance(paths, list) or not paths:
            continue
        cells_seen += 1
        for path in paths:
            if not isinstance(path, dict):
                continue
            demand = max(_as_float(path.get("daily_demand_allocated"), 0.0), 0.0)
            if demand <= 0:
                continue
            legs = path.get("legs", [])
            if not isinstance(legs, list) or not legs:
                continue
            paths_seen += 1
            pax_seen += demand
            for leg in legs:
                if not isinstance(leg, dict):
                    continue
                line = str(leg.get("service_name", "") or "").strip()
                if not line:
                    line = "UNKNOWN"
                st1, st2, arr1, dep1, arr2, dep2 = _find_leg_event(
                    leg,
                    merge_zurich_hb_aliases=merge_zurich_hb_aliases,
                )
                if not st1 or not st2 or st1 == st2:
                    continue
                pax_key = (st1, st2, line)
                segment_pax[pax_key] = segment_pax.get(pax_key, 0.0) + demand
                event_key = (
                    st1,
                    st2,
                    line,
                    _event_mod(arr1, cycle_minutes),
                    _event_mod(dep1, cycle_minutes),
                    _event_mod(arr2, cycle_minutes),
                    _event_mod(dep2, cycle_minutes),
                )
                old = event_records.get(event_key)
                if old is None:
                    event_records[event_key] = (arr1, dep1, arr2, dep2)
                else:
                    # Keep the earliest absolute representative for readable output.
                    old_sort = min(x for x in old if x is not None) if any(x is not None for x in old) else 10**9
                    new = (arr1, dep1, arr2, dep2)
                    new_sort = min(x for x in new if x is not None) if any(x is not None for x in new) else 10**9
                    if new_sort < old_sort:
                        event_records[event_key] = new

    cur = conn.cursor()
    cur.execute("BEGIN")
    if segment_pax:
        cur.executemany(
            """
            INSERT INTO pax(station_a, station_b, line, daily_pax)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(station_a, station_b, line)
            DO UPDATE SET daily_pax = daily_pax + excluded.daily_pax
            """,
            [(a, b, line, value) for (a, b, line), value in segment_pax.items()],
        )
    if event_records:
        cur.executemany(
            """
            INSERT INTO events(
                station_a, station_b, line,
                arr_a_mod, dep_a_mod, arr_b_mod, dep_b_mod,
                arr_a_min, dep_a_min, arr_b_min, dep_b_min
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(
                station_a, station_b, line,
                arr_a_mod, dep_a_mod, arr_b_mod, dep_b_mod
            )
            DO UPDATE SET
                arr_a_min = CASE
                    WHEN excluded.arr_a_min IS NOT NULL
                     AND (events.arr_a_min IS NULL OR excluded.arr_a_min < events.arr_a_min)
                    THEN excluded.arr_a_min ELSE events.arr_a_min END,
                dep_a_min = CASE
                    WHEN excluded.dep_a_min IS NOT NULL
                     AND (events.dep_a_min IS NULL OR excluded.dep_a_min < events.dep_a_min)
                    THEN excluded.dep_a_min ELSE events.dep_a_min END,
                arr_b_min = CASE
                    WHEN excluded.arr_b_min IS NOT NULL
                     AND (events.arr_b_min IS NULL OR excluded.arr_b_min < events.arr_b_min)
                    THEN excluded.arr_b_min ELSE events.arr_b_min END,
                dep_b_min = CASE
                    WHEN excluded.dep_b_min IS NOT NULL
                     AND (events.dep_b_min IS NULL OR excluded.dep_b_min < events.dep_b_min)
                    THEN excluded.dep_b_min ELSE events.dep_b_min END
            """,
            [
                (
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    key[5],
                    key[6],
                    vals[0],
                    vals[1],
                    vals[2],
                    vals[3],
                )
                for key, vals in event_records.items()
            ],
        )
    _set_meta(conn, "next_row", str(next_row_after))
    _set_meta(conn, "updated_at", _now())
    conn.commit()
    return cells_seen, paths_seen, pax_seen


def _export_csv(conn: sqlite3.Connection, output_csv: Path, cycle_minutes: int) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    event_rows = conn.execute(
        """
        SELECT
            station_a, station_b, line,
            arr_a_min, dep_a_min, arr_b_min, dep_b_min
        FROM events
        ORDER BY station_a, station_b, line,
                 COALESCE(dep_a_min, arr_a_min, 999999),
                 COALESCE(arr_b_min, dep_b_min, 999999)
        """
    ).fetchall()
    events_by_key: dict[tuple[str, str, str], list[tuple[int | None, int | None, int | None, int | None]]] = {}
    for st1, st2, line, arr1, dep1, arr2, dep2 in event_rows:
        events_by_key.setdefault((st1, st2, line), []).append((arr1, dep1, arr2, dep2))

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "First station",
                "Second station",
                "Line",
                "Arrival time first station",
                "Departure time first station",
                "Arrival time second station",
                "Departure time second station",
                "Hourly frequency",
                "Daily pax",
            ]
        )
        for st1, st2, line, daily_pax in conn.execute(
            """
            SELECT station_a, station_b, line, daily_pax
            FROM pax
            WHERE daily_pax > 0
            ORDER BY station_a, station_b, line
            """
        ):
            events = events_by_key.get((st1, st2, line), [])
            frequency = len(events) * 60.0 / float(cycle_minutes)
            writer.writerow(
                [
                    st1,
                    st2,
                    line,
                    ", ".join(_hhmm(e[0]) for e in events),
                    ", ".join(_hhmm(e[1]) for e in events),
                    ", ".join(_hhmm(e[2]) for e in events),
                    ", ".join(_hhmm(e[3]) for e in events),
                    f"{frequency:.6f}".rstrip("0").rstrip("."),
                    f"{float(daily_pax):.6f}",
                ]
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate a path-allocated nondominated OD CSV to adjacent stop-pair "
            "daily passenger loads by line/service and observed stop timing."
        )
    )
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--state-db", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--progress-every-rows", type=int, default=5)
    p.add_argument("--cycle-minutes", type=int, default=120)
    p.add_argument("--merge-zurich-hb-aliases", action="store_true")
    args = p.parse_args()

    csv.field_size_limit(sys.maxsize)
    input_csv = args.input_csv.expanduser().resolve()
    output_csv = args.output_csv.expanduser().resolve()
    state_db = args.state_db.expanduser().resolve()
    progress_every = max(1, int(args.progress_every_rows))
    cycle_minutes = max(1, int(args.cycle_minutes))

    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {input_csv}")
    if not args.resume:
        state_db.unlink(missing_ok=True)
        for suffix in ("-wal", "-shm"):
            state_db.with_name(state_db.name + suffix).unlink(missing_ok=True)

    conn = _connect(state_db)
    start_row = _get_next_row(conn) if args.resume else 0
    _set_meta(conn, "input_csv", str(input_csv))
    _set_meta(conn, "output_csv", str(output_csv))
    _set_meta(conn, "cycle_minutes", str(cycle_minutes))
    conn.commit()
    _log(
        f"aggregate start input={input_csv.name} start_row={start_row} "
        f"state_db={state_db} output={output_csv}"
    )

    total_cells = 0
    total_paths = 0
    total_pax = 0.0
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        if not header or header[0] != "StationName":
            raise ValueError(f"{input_csv.name}: first column must be StationName")
        destinations = [str(x) for x in header[1:]]
        expected_width = len(header)
        for _ in range(start_row):
            next(reader)

        processed = start_row
        for row in reader:
            if len(row) != expected_width:
                raise ValueError(
                    f"{input_csv.name}: row {processed} width mismatch "
                    f"expected={expected_width} got={len(row)}"
                )
            cells, paths, pax = _aggregate_row(
                conn=conn,
                row=row,
                destinations=destinations,
                cycle_minutes=cycle_minutes,
                merge_zurich_hb_aliases=bool(args.merge_zurich_hb_aliases),
                next_row_after=processed + 1,
            )
            processed += 1
            total_cells += cells
            total_paths += paths
            total_pax += pax
            if processed % progress_every == 0:
                _log(
                    f"progress rows={processed}/{len(destinations)} "
                    f"row_cells={cells} row_paths={paths} "
                    f"cumulative_paths={total_paths}"
                )

    _log("exporting aggregated CSV")
    _export_csv(conn, output_csv, cycle_minutes)
    _set_meta(conn, "completed_at", _now())
    _set_meta(conn, "final_rows", str(processed))
    conn.commit()
    conn.close()
    _log(f"complete rows={processed} output={output_csv}")


if __name__ == "__main__":
    main()
