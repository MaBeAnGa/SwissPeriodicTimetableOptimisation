from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


DASH_VALUES = {"", "-", "—", "–", "―"}
TOKEN_RE = re.compile(r"^\s*(\d{1,2})\s*([inIN]?)\s*([Pp]?)\s*([Qq]?)\s*$")

TIMETABLE_FILE = Path(__file__).resolve().parent / "FullTimetableHistory2026_2035.csv"

# Zurich HB is one physical complex in the new full-timetable model. The alias
# rows keep their exact timetable names, but they inherit metadata from the
# base station. Pairwise transfer logic is handled in the routing layer: the
# complex default is 7 min, and only Zürich HB (31-34) self-transfers later
# get a 4 min override.
ZURICH_HB_BASE_NAME = "Zürich HB"
ZURICH_HB_ALIASES = (
    "Zürich HB",
    "Zürich HB (21-22)",
    "Zürich HB (31-34)",
    "Zürich HB (41-44)",
)
ZURICH_HB_COMPLEX_TRANSFER_MIN = 7

RAW_YEAR_COLUMNS = (
    "2035 alt",
    "2035 (HH)",
    "2035",
    "2026",
    "2026 (HH)",
    "2026 alt",
)

LINE_COUNT_VARIANTS = {
    "2035": ("2035", "2035 (HH)"),
    "2026": ("2026", "2026 (HH)"),
}


@dataclass(frozen=True)
class ParsedToken:
    minute: int
    mode: str
    pflag: bool
    qflag: bool
    raw: str


@dataclass
class ServiceSegment:
    service_name: str
    segment_index: int
    rows: list[dict]


@dataclass
class ServiceBlock:
    service_name: str
    rows: list[dict]


def load_timetable_with_first_header_as_row(filepath: str | Path = TIMETABLE_FILE) -> pd.DataFrame:
    raw = pd.read_csv(filepath, dtype=str).fillna("")
    original_columns = list(raw.columns)

    if len(original_columns) < 3:
        raise ValueError(f"{filepath} does not appear to have the expected timetable structure.")

    renamed_columns = ["station", "kind"] + original_columns[2:]
    raw.columns = renamed_columns

    first_header_row = {"station": original_columns[0], "kind": original_columns[1]}
    for column in original_columns[2:]:
        first_header_row[column] = column

    df = pd.concat([pd.DataFrame([first_header_row]), raw], ignore_index=True)
    df["rownum"] = range(1, len(df) + 1)

    for column in df.columns:
        if df[column].dtype == object:
            df[column] = df[column].astype(str).str.strip()

    return df


def parse_token(cell: str) -> Optional[ParsedToken]:
    cell = str(cell).strip()
    if cell in DASH_VALUES:
        return None

    match = TOKEN_RE.match(cell)
    if not match:
        return None

    minute = int(match.group(1))
    if not (0 <= minute <= 59):
        return None

    suffix = match.group(2).lower()
    if suffix == "":
        mode = "hourly"
    elif suffix == "n":
        mode = "even"
    elif suffix == "i":
        mode = "odd"
    else:
        return None

    return ParsedToken(
        minute=minute,
        mode=mode,
        pflag=bool(match.group(3)),
        qflag=bool(match.group(4)),
        raw=cell,
    )


def parse_service_segments(df: pd.DataFrame) -> list[ServiceSegment]:
    segments: list[ServiceSegment] = []
    current_service: str | None = None
    current_rows: list[dict] = []

    def flush_block(service_name: str | None, rows: list[dict]) -> None:
        if not service_name:
            return

        segment_rows: list[dict] = []
        segment_index = 1
        for row in rows:
            if is_split_row(row):
                if segment_rows:
                    segments.append(
                        ServiceSegment(
                            service_name=service_name,
                            segment_index=segment_index,
                            rows=segment_rows,
                        )
                    )
                    segment_index += 1
                    segment_rows = []
            else:
                segment_rows.append(row)

        if segment_rows:
            segments.append(
                ServiceSegment(
                    service_name=service_name,
                    segment_index=segment_index,
                    rows=segment_rows,
                )
            )

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        kind = str(row_dict["kind"]).strip().lower()
        if kind == "an/ab":
            flush_block(current_service, current_rows)
            current_service = str(row_dict["station"]).strip()
            current_rows = []
        else:
            if current_service is not None:
                current_rows.append(row_dict)

    flush_block(current_service, current_rows)
    return segments


def parse_service_blocks(df: pd.DataFrame) -> list[ServiceBlock]:
    blocks: list[ServiceBlock] = []
    current_service: str | None = None
    current_rows: list[dict] = []

    def flush_block(service_name: str | None, rows: list[dict]) -> None:
        if service_name is not None:
            blocks.append(ServiceBlock(service_name=service_name, rows=list(rows)))

    for _, row in df.iterrows():
        row_dict = row.to_dict()
        kind = str(row_dict["kind"]).strip().lower()
        if kind == "an/ab":
            flush_block(current_service, current_rows)
            current_service = str(row_dict["station"]).strip()
            current_rows = []
        else:
            if current_service is not None:
                current_rows.append(row_dict)

    flush_block(current_service, current_rows)
    return blocks


def is_split_row(row: dict) -> bool:
    return str(row["station"]).strip().lower() == "split" or str(row["kind"]).strip().lower() == "split"


def get_station_names(df: pd.DataFrame) -> list[str]:
    rows = df[df["kind"].astype(str).str.strip().str.lower().isin(["an", "ab"])]
    return sorted(set(rows["station"].astype(str).str.strip()))


def get_active_event_rows_for_column(segment: ServiceSegment, year_col: str) -> list[dict]:
    rows: list[dict] = []
    for row in segment.rows:
        token = parse_token(row.get(year_col, ""))
        if token is None:
            continue
        rows.append(
            {
                "rownum": row["rownum"],
                "station": row["station"],
                "kind": str(row["kind"]).strip().lower(),
                "token": token,
                "raw": row.get(year_col, ""),
            }
        )
    return rows


def trim_invalid_ends(active_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    rows = list(active_rows)
    trimmed: list[dict] = []

    while rows and rows[0]["kind"] == "an":
        trimmed.append({**rows[0], "trimReason": "leading-arrival"})
        rows.pop(0)

    while rows and rows[-1]["kind"] == "ab":
        trimmed.append({**rows[-1], "trimReason": "trailing-departure"})
        rows.pop()

    return rows, trimmed


def validate_ab_an_sequence(active_rows: list[dict]) -> list[dict]:
    issues: list[dict] = []
    if not active_rows:
        return issues

    expected = "ab"
    for row in active_rows:
        if row["kind"] != expected:
            issues.append(
                {
                    **row,
                    "issue": "wrong-kind-order",
                    "expectedKind": expected,
                }
            )
            break
        expected = "an" if expected == "ab" else "ab"

    if active_rows and active_rows[-1]["kind"] != "an":
        issues.append(
            {
                **active_rows[-1],
                "issue": "sequence-does-not-end-with-arrival",
                "expectedKind": "an",
            }
        )

    return issues


def station_metadata_lookup_name(station_name: str) -> str:
    if station_name in ZURICH_HB_ALIASES:
        return ZURICH_HB_BASE_NAME
    return station_name
