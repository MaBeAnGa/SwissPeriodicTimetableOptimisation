from __future__ import annotations

import argparse
import csv
import io
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"
STATION_ASSIGNMENTS_PATH = PROJECT_DIR / "station_population_assignments.csv"
BFS_COMMUNE_LEVELS_PATH = PROJECT_DIR / "bfs_commune_levels_2023.csv"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "station_agglomeration_crosswalk_2023.csv"

BFS_COMMUNE_LEVELS_REFERENCE_DATE = "01-01-2023"
BFS_COMMUNE_LEVELS_REFERENCE_YEAR = "2023"
BFS_COMMUNE_LEVELS_URL = (
    f"https://www.agvchapp.bfs.admin.ch/api/communes/levels?date={BFS_COMMUNE_LEVELS_REFERENCE_DATE}"
)


def normalize_bfs_code(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(character for character in text if character.isdigit())
    if not digits:
        return ""
    return digits.zfill(4)


def ensure_bfs_commune_levels(
    output_path: Path = BFS_COMMUNE_LEVELS_PATH,
    refresh: bool = False,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not refresh:
        return output_path

    try:
        with urllib.request.urlopen(BFS_COMMUNE_LEVELS_URL, timeout=60) as response:
            content = response.read().decode("utf-8-sig")
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            "Could not download the official BFS commune-level typology CSV. "
            f"Tried {BFS_COMMUNE_LEVELS_URL}."
        ) from exc

    if "HistoricalCode,BfsCode,Name" not in content:
        raise RuntimeError(
            "The downloaded BFS commune-level payload did not match the expected CSV header."
        )

    output_path.write_text(content, encoding="utf-8")
    return output_path


def load_bfs_commune_levels(path: Path = BFS_COMMUNE_LEVELS_PATH) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run build_station_agglomeration_crosswalk.py first to cache the official BFS commune-level typology CSV."
        )

    levels: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            code = normalize_bfs_code(row.get("BfsCode"))
            if code:
                levels[code] = {key: str(value or "").strip() for key, value in row.items()}
    return levels


def load_station_agglomeration_crosswalk(
    path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path.name}. Run build_station_agglomeration_crosswalk.py first."
        )

    mapping: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            station = str(row.get("station", "")).strip()
            if station:
                mapping[station] = {key: str(value or "").strip() for key, value in row.items()}
    return mapping


def build_station_agglomeration_rows(
    levels_path: Path = BFS_COMMUNE_LEVELS_PATH,
) -> list[dict[str, object]]:
    filtered = pd.read_csv(FILTERED_STATIONS_PATH, dtype=str).fillna("")
    filtered["StationName"] = filtered["StationName"].astype(str).str.strip()
    filtered = filtered.sort_values("StationName").reset_index(drop=True)

    assignments = pd.read_csv(STATION_ASSIGNMENTS_PATH, dtype=str).fillna("")
    assignment_by_station = {
        str(row["station_name"]).strip(): row
        for _, row in assignments.iterrows()
    }
    levels = load_bfs_commune_levels(levels_path)

    rows: list[dict[str, object]] = []
    for _, station_row in filtered.iterrows():
        station = str(station_row["StationName"]).strip()
        assignment = assignment_by_station.get(station, {})
        country = str(station_row.get("Country", "")).strip() or str(assignment.get("country", "")).strip()
        canton = str(station_row.get("Region", "")).strip() or str(assignment.get("region", "")).strip()

        municipality_code = normalize_bfs_code(
            assignment.get("base_population_municipality_code") or assignment.get("municipality_code")
        )
        municipality_name = (
            str(assignment.get("base_population_municipality_name", "")).strip()
            or str(assignment.get("municipality_name", "")).strip()
        )
        level = levels.get(municipality_code, {})

        aggl2020_code = str(level.get("AGGL2020", "")).strip()
        in_official_agglomeration = country == "CH" and aggl2020_code not in {"", "0"}

        if in_official_agglomeration:
            analysis_cluster_id = f"AGGL2020:{aggl2020_code}"
            analysis_cluster_label = f"Official BFS agglomeration 2020 {aggl2020_code}"
            cluster_scope = "official_bfs_aggl2020"
            notes = (
                "Swiss station mapped to its municipality's official BFS AGGL2020 code "
                f"from the commune-level typology state of {BFS_COMMUNE_LEVELS_REFERENCE_DATE}."
            )
        elif country == "CH":
            analysis_cluster_id = f"STATION:{station}"
            analysis_cluster_label = station
            cluster_scope = "station_singleton_outside_agglomeration"
            notes = (
                "Swiss station lies outside any official BFS AGGL2020 agglomeration and therefore stays a singleton "
                "for the analysis-level OD weighting."
            )
        else:
            analysis_cluster_id = f"STATION:{station}"
            analysis_cluster_label = station
            cluster_scope = "foreign_station_singleton"
            notes = (
                "Foreign station kept as a singleton because the official BFS AGGL2020 typology only covers Swiss municipalities."
            )

        rows.append(
            {
                "station": station,
                "country": country,
                "canton": canton,
                "municipality_code_2023": municipality_code,
                "municipality_name_2023": municipality_name,
                "bfs_level_name_2023": str(level.get("Name", "")).strip(),
                "aggl2020_code": aggl2020_code,
                "rsc2020_code": str(level.get("RSC2020", "")).strip(),
                "stadte2020_code": str(level.get("STADTE2020", "")).strip(),
                "agglgk2020_code": str(level.get("AGGLGK2020", "")).strip(),
                "analysis_cluster_id": analysis_cluster_id,
                "analysis_cluster_label": analysis_cluster_label,
                "cluster_scope": cluster_scope,
                "cluster_source": (
                    "BFS commune-level spatial typologies endpoint "
                    f"({BFS_COMMUNE_LEVELS_REFERENCE_DATE}, AGGL2020)"
                ),
                "notes": notes,
            }
        )

    return rows


def write_crosswalk(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "station",
        "country",
        "canton",
        "municipality_code_2023",
        "municipality_name_2023",
        "bfs_level_name_2023",
        "aggl2020_code",
        "rsc2020_code",
        "stadte2020_code",
        "agglgk2020_code",
        "analysis_cluster_id",
        "analysis_cluster_label",
        "cluster_scope",
        "cluster_source",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the local station-to-agglomeration crosswalk used by the 2023 observed/hybrid OD pipeline. "
            "The Swiss part of the crosswalk uses the official BFS commune-level AGGL2020 membership from the "
            "commune-state endpoint for 01-01-2023."
        )
    )
    parser.add_argument(
        "--levels-output",
        type=Path,
        default=BFS_COMMUNE_LEVELS_PATH,
        help="Path to the cached BFS commune-level typology CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to the generated station-to-agglomeration crosswalk CSV.",
    )
    parser.add_argument(
        "--refresh-levels",
        action="store_true",
        help="Redownload the BFS commune-level typology CSV even if it is already cached locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    levels_path = ensure_bfs_commune_levels(args.levels_output, refresh=args.refresh_levels)
    rows = build_station_agglomeration_rows(levels_path)
    write_crosswalk(rows, args.output)

    official_aggl_rows = sum(1 for row in rows if str(row["cluster_scope"]) == "official_bfs_aggl2020")
    singleton_rows = len(rows) - official_aggl_rows
    print(
        f"Wrote {len(rows)} station crosswalk rows to {args.output} "
        f"({official_aggl_rows} stations inside official AGGL2020 agglomerations, {singleton_rows} singleton stations)."
    )


if __name__ == "__main__":
    main()
