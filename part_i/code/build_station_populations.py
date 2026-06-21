from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


PROJECT_DIR = Path(__file__).resolve().parent
FILTERED_STATIONS_PATH = PROJECT_DIR / "FilteredStations.csv"
ASSIGNMENTS_OUTPUT_PATH = PROJECT_DIR / "station_population_assignments.csv"
SCALING_OUTPUT_PATH = PROJECT_DIR / "canton_population_scaling_factors.csv"

GEOADMIN_IDENTIFY_URL = "https://api3.geo.admin.ch/rest/services/ech/MapServer/identify"
PXWEB_MUNICIPALITY_URL = (
    "https://www.pxweb.bfs.admin.ch/api/v1/en/px-x-0102010000_104/px-x-0102010000_104.px"
)
PXWEB_CANTON_OBSERVED_URL = (
    "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0102020000_104/px-x-0102020000_104.px"
)
PXWEB_CANTON_SCENARIO_URL = (
    "https://www.pxweb.bfs.admin.ch/api/v1/de/px-x-0104020000_106/px-x-0104020000_106.px"
)
HTTP_HEADERS = {
    "User-Agent": "historical-rail-planner-station-population-builder/1.0",
}

CANTON_ABBREVIATIONS = [
    "ZH",
    "BE",
    "LU",
    "UR",
    "SZ",
    "OW",
    "NW",
    "GL",
    "ZG",
    "FR",
    "SO",
    "BS",
    "BL",
    "SH",
    "AR",
    "AI",
    "SG",
    "GR",
    "AG",
    "TG",
    "TI",
    "VD",
    "VS",
    "NE",
    "GE",
    "JU",
]
CANTON_BFS_CODE_BY_ABBREV = {
    abbrev: str(index)
    for index, abbrev in enumerate(CANTON_ABBREVIATIONS, start=1)
}
CANTON_ABBREV_BY_BFS_CODE = {value: key for key, value in CANTON_BFS_CODE_BY_ABBREV.items()}

POPULATION_COLUMN_TO_YEAR_LABEL = {
    "2035_pop": "2035",
    "2026_pop": "2026",
    "2025_pop": "2025",
    "2024_pop": "2024",
    "2023_pop": "2023",
    "2022_pop": "2022",
    "2021_pop": "2021",
    "2020_pop": "2020",
    "2019_pop": "2019",
    "2018_pop": "2018",
    "2017_pop": "2017",
    "2016_pop": "2016",
    "2015_pop": "2015",
    "2014 Q3/4_pop": "2014 Q3/4",
    "2014 Q1/2_pop": "2014 Q1/2",
    "2013_pop": "2013",
    "2012_pop": "2012",
    "2011_pop": "2011",
    "2009_pop": "2009",
    "2008_pop": "2008",
    "2005_pop": "2005",
    "2002_pop": "2002",
    "1982_pop": "1982",
}
SERVICE_COLUMNS_BY_PLANNER_YEAR = {
    "2035": ["2035", "2035alter"],
    "2026": ["2026"],
    "2025": ["2025"],
    "2024": ["2024"],
    "2023": ["2023"],
    "2022": ["2022"],
    "2021": ["2021"],
    "2020": ["2020"],
    "2019": ["2019"],
    "2018": ["2018"],
    "2017": ["2017"],
    "2016": ["2016"],
    "2015": ["2015"],
    "2014 Q3/4": ["2014 Q3/4"],
    "2014 Q1/2": ["2014 Q1/2"],
    "2013": ["2013"],
    "2012": ["2012"],
    "2011": ["2011"],
    "2009": ["2009"],
    "2008": ["2008"],
    "2005": ["2005"],
    "2002": ["2002"],
    "1982": ["1982"],
}
SOURCE_YEAR_BY_PLANNER_YEAR = {
    "2035": "2035",
    "2026": "2026",
    "2025": "2025",
    "2024": "2024",
    "2023": "2023",
    "2022": "2022",
    "2021": "2021",
    "2020": "2020",
    "2019": "2019",
    "2018": "2018",
    "2017": "2017",
    "2016": "2016",
    "2015": "2015",
    "2014 Q3/4": "2014",
    "2014 Q1/2": "2014",
    "2013": "2013",
    "2012": "2012",
    "2011": "2011",
    "2009": "2009",
    "2008": "2008",
    "2005": "2005",
    "2002": "2002",
    "1982": "1982",
}
OBSERVED_SOURCE_YEARS = [
    "1982",
    "2002",
    "2005",
    "2008",
    "2009",
    "2011",
    "2012",
    "2013",
    "2014",
    "2015",
    "2016",
    "2017",
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
]
PROJECTED_SOURCE_YEARS = ["2025", "2026", "2035"]

FOREIGN_STATION_BASES = {
    "Konstanz": {
        "population": 86898,
        "reference_year": "2023",
        "source_title": "Stadt Konstanz: Bevölkerung",
        "source_url": "https://www.konstanz.de/statistik/bevoelkerung",
        "notes": (
            "Official city page citing the Statistisches Landesamt Baden-Württemberg and listing "
            "the official municipal population at 31.12.2023."
        ),
    },
    "Singen (Hohentwiel)": {
        "population": 49705,
        "reference_year": "2023",
        "source_title": "Stadt Singen: Singen in Zahlen",
        "source_url": "https://www.singen.de/informieren/unsere_stadt/singen_in_zahlen",
        "notes": (
            "Official city statistics page listing the population total for the city at 31.12.2023."
        ),
    },
    "Annemasse": {
        "population": 37918,
        "reference_year": "2021",
        "source_title": "Insee: Growth and structure of the population in 2021 - Municipality of Annemasse (74012)",
        "source_url": "https://www.insee.fr/en/statistiques/8201201?geo=COM-74012",
        "notes": (
            "Official Insee municipality profile listing the 2021 municipal population. No compatible "
            "Swiss canton growth series exists for foreign stations in the current build, so the same "
            "official base value is carried across planner years as a documented exception."
        ),
    },
}

BASE_MUNICIPALITY_CODE_OVERRIDES = {
    "0030": {
        "code": "0291",
        "name": "Andelfingen",
        "note": "BFS 2010-2024 municipality table uses the renumbered commune code 0291 for Andelfingen.",
    },
    "0133": {
        "code": "0295",
        "name": "Horgen",
        "note": "BFS 2010-2024 municipality table uses the renumbered commune code 0295 for Horgen.",
    },
    "0142": {
        "code": "0293",
        "name": "Wädenswil",
        "note": "BFS 2010-2024 municipality table uses the renumbered commune code 0293 for Wädenswil.",
    },
    "0174": {
        "code": "0296",
        "name": "Illnau-Effretikon",
        "note": "BFS 2010-2024 municipality table uses the renumbered commune code 0296 for Illnau-Effretikon.",
    },
    "4042": {
        "code": "4021",
        "name": "Baden",
        "note": "Turgi was merged into Baden in the BFS municipality geography used by the 2023 table, so the base population is read from Baden.",
    },
    "5005": {
        "code": "5002",
        "name": "Bellinzona",
        "note": "Giubiasco was merged into Bellinzona in the BFS municipality geography used by the 2023 table, so the base population is read from Bellinzona.",
    },
    "5195": {
        "code": "5240",
        "name": "Val Mara",
        "note": "Maroggia belongs to Val Mara in the BFS municipality geography used by the 2023 table, so the base population is read from Val Mara.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Populate FilteredStations.csv with station population proxies based on official municipality "
            "base populations, official canton growth factors, and service-weighted splitting wherever "
            "multiple modeled stations share the same municipality base."
        )
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.05,
        help="Pause between GeoAdmin municipality lookups.",
    )
    parser.add_argument(
        "--reuse-existing-assignments",
        action="store_true",
        help="Reuse station_population_assignments.csv if it already contains all stations.",
    )
    return parser.parse_args()


def fetch_json(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 60,
) -> Any:
    request = Request(url, data=data, headers={**HTTP_HEADERS, **(headers or {})})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def query_pxweb(url: str, query: list[dict[str, Any]]) -> dict[str, Any]:
    payload = {
        "query": query,
        "response": {"format": "json-stat2"},
    }
    return fetch_json(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )


def ordered_dimension_codes(payload: dict[str, Any], dimension_id: str) -> list[str]:
    index = payload["dimension"][dimension_id]["category"]["index"]
    return [code for code, _ in sorted(index.items(), key=lambda item: item[1])]


def dimension_labels(payload: dict[str, Any], dimension_id: str) -> dict[str, str]:
    labels = payload["dimension"][dimension_id]["category"].get("label", {})
    return {str(key): str(value) for key, value in labels.items()}


def iter_jsonstat_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    dimension_ids = payload["id"]
    sizes = payload["size"]
    codes_by_dimension = [ordered_dimension_codes(payload, dimension_id) for dimension_id in dimension_ids]
    labels_by_dimension = [dimension_labels(payload, dimension_id) for dimension_id in dimension_ids]
    values = payload.get("value", [])
    rows: list[dict[str, Any]] = []

    for linear_index, value in enumerate(values):
        remainder = linear_index
        positions = [0] * len(sizes)
        for offset in range(len(sizes) - 1, -1, -1):
            size = sizes[offset]
            positions[offset] = remainder % size
            remainder //= size

        row: dict[str, Any] = {"value": value}
        for offset, dimension_id in enumerate(dimension_ids):
            code = codes_by_dimension[offset][positions[offset]]
            row[dimension_id] = code
            row[f"{dimension_id}__label"] = labels_by_dimension[offset].get(code, code)
        rows.append(row)

    return rows


def load_filtered_station_rows() -> tuple[list[dict[str, str]], list[str]]:
    with FILTERED_STATIONS_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows or not fieldnames:
        raise ValueError("FilteredStations.csv is empty or unreadable.")
    return rows, fieldnames


def _safe_positive_int(value: Any) -> int | None:
    try:
        text = str(value).strip()
        if not text:
            return None
        numeric = int(round(float(text)))
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _safe_nonnegative_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if not text:
            return 0.0
        numeric = float(text)
    except (TypeError, ValueError):
        return 0.0
    return numeric if numeric > 0 else 0.0


def municipality_query_url(lat: str, lon: str) -> str:
    params = {
        "geometry": f"{lon},{lat}",
        "geometryType": "esriGeometryPoint",
        "sr": "4326",
        "returnGeometry": "false",
        "tolerance": "0",
        "imageDisplay": "1,1,96",
        "mapExtent": f"{lon},{lat},{lon},{lat}",
        "layers": "all:ch.swisstopo.swissboundaries3d-gemeinde-flaeche.fill",
        "lang": "de",
    }
    return f"{GEOADMIN_IDENTIFY_URL}?{urlencode(params)}"


def _filtered_municipality_candidates(
    results: list[dict[str, Any]],
    expected_canton: str,
) -> list[dict[str, Any]]:
    candidates = [
        result.get("attributes", {})
        for result in results
        if str(result.get("attributes", {}).get("objektart_lookup", "")) == "politische_gemeinde"
    ]
    if expected_canton:
        canton_matched = [
            candidate
            for candidate in candidates
            if str(candidate.get("kanton", "")).strip() == expected_canton
        ]
        if canton_matched:
            candidates = canton_matched
    if not candidates:
        raise ValueError("No municipality candidate returned by GeoAdmin.")
    return candidates


def select_current_municipality_candidate(
    results: list[dict[str, Any]],
    expected_canton: str,
) -> dict[str, Any]:
    candidates = _filtered_municipality_candidates(results, expected_canton)

    current_candidates = [
        candidate
        for candidate in candidates
        if bool(candidate.get("is_current_jahr"))
    ]
    preferred = current_candidates or candidates
    preferred.sort(
        key=lambda candidate: (
            int(candidate.get("jahr") or 0),
            int(candidate.get("gde_nr") or 0),
        ),
        reverse=True,
    )
    return preferred[0]


def select_base_year_municipality_candidate(
    results: list[dict[str, Any]],
    expected_canton: str,
    base_year: int,
) -> dict[str, Any]:
    candidates = _filtered_municipality_candidates(results, expected_canton)
    same_year = [
        candidate
        for candidate in candidates
        if int(candidate.get("jahr") or 0) == base_year
    ]
    if same_year:
        same_year.sort(
            key=lambda candidate: int(candidate.get("gde_nr") or 0),
        )
        return same_year[0]

    historic_candidates = [
        candidate
        for candidate in candidates
        if int(candidate.get("jahr") or 0) <= base_year
    ]
    preferred = historic_candidates or candidates
    preferred.sort(
        key=lambda candidate: (
            int(candidate.get("jahr") or 0),
            int(candidate.get("gde_nr") or 0),
        ),
        reverse=True,
    )
    return preferred[0]


def build_station_assignments(
    station_rows: list[dict[str, str]],
    *,
    sleep_seconds: float,
) -> dict[str, dict[str, Any]]:
    assignments: dict[str, dict[str, Any]] = {}
    total = len(station_rows)
    for index, row in enumerate(station_rows, start=1):
        station = str(row.get("StationName", "")).strip()
        if not station:
            continue

        country = str(row.get("Country", "")).strip()
        region = str(row.get("Region", "")).strip()
        lat = str(row.get("lat", "")).strip()
        lon = str(row.get("lon", "")).strip()

        if country != "CH":
            foreign = FOREIGN_STATION_BASES.get(station)
            if not foreign:
                raise ValueError(f"No documented foreign-station population fallback exists for {station}.")
            assignments[station] = {
                "station_name": station,
                "country": country,
                "region": region,
                "lat": lat,
                "lon": lon,
                "assignment_method": "documented_foreign_station_exception",
                "municipality_code": "",
                "municipality_name": station,
                "municipality_canton": region,
                "assignment_year": "",
                "base_population_municipality_code": "",
                "base_population_municipality_name": station,
                "base_population_municipality_year": foreign["reference_year"],
                "candidate_count": 0,
                "base_population_reference_year": foreign["reference_year"],
                "base_population": int(foreign["population"]),
                "base_population_source_title": foreign["source_title"],
                "base_population_source_url": foreign["source_url"],
                "base_population_source_type": "official_foreign_municipal_source",
            "scaling_method": "constant_foreign_exception",
            "shared_base_population_group_size": "1",
            "population_split_method": "constant_foreign_exception",
            "notes": foreign["notes"],
        }
        continue

        payload = fetch_json(municipality_query_url(lat, lon))
        current_candidate = select_current_municipality_candidate(
            payload.get("results", []),
            expected_canton=region,
        )
        base_candidate = select_base_year_municipality_candidate(
            payload.get("results", []),
            expected_canton=region,
            base_year=2023,
        )
        assignments[station] = {
            "station_name": station,
            "country": country,
            "region": region,
            "lat": lat,
            "lon": lon,
            "assignment_method": "geoadmin_current_municipality_identify",
            "municipality_code": f"{int(current_candidate['gde_nr']):04d}",
            "municipality_name": str(current_candidate.get("gemname", "")).strip(),
            "municipality_canton": str(current_candidate.get("kanton", "")).strip(),
            "assignment_year": str(current_candidate.get("jahr", "")).strip(),
            "base_population_municipality_code": f"{int(base_candidate['gde_nr']):04d}",
            "base_population_municipality_name": str(base_candidate.get("gemname", "")).strip(),
            "base_population_municipality_year": str(base_candidate.get("jahr", "")).strip(),
            "candidate_count": len(payload.get("results", [])),
            "base_population_reference_year": "2023",
            "base_population": "",
            "base_population_source_title": (
                "BFS municipality population table 2010-2024 (Permanent and non-permanent resident population "
                "by institutional units, place of birth and citizenship)"
            ),
            "base_population_source_url": "https://www.pxweb.bfs.admin.ch/pxweb/de/px-x-0102010000_104/-/px-x-0102010000_104.px",
            "base_population_source_type": "official_bfs_municipality_population",
            "scaling_method": "canton_growth_vs_2023",
            "shared_base_population_group_size": "",
            "population_split_method": "",
            "notes": (
                "Current municipality selected from the GeoAdmin swissBOUNDARIES3D municipality identify result. "
                "The station's 2023 base population is the municipality's permanent resident population, later "
                "scaled with official canton totals."
            ),
        }
        if index % 25 == 0 or index == total:
            print(f"Mapped {index}/{total} stations to municipality references", flush=True)
        time.sleep(max(sleep_seconds, 0.0))
    return assignments


def apply_base_municipality_overrides(assignments: dict[str, dict[str, Any]]) -> None:
    for assignment in assignments.values():
        original_code = str(assignment.get("base_population_municipality_code", "")).strip()
        override = BASE_MUNICIPALITY_CODE_OVERRIDES.get(original_code)
        if not override:
            continue
        assignment["base_population_municipality_code"] = override["code"]
        assignment["base_population_municipality_name"] = override["name"]
        assignment["base_population_municipality_year"] = "2023"
        note = str(assignment.get("notes", "")).strip()
        override_note = override["note"]
        if override_note not in note:
            assignment["notes"] = f"{note} {override_note}".strip()


def fetch_municipality_base_populations(municipality_codes: list[str]) -> dict[str, int]:
    if not municipality_codes:
        return {}
    populations: dict[str, int] = {}
    chunk_size = 80
    for offset in range(0, len(municipality_codes), chunk_size):
        chunk = municipality_codes[offset : offset + chunk_size]
        payload = query_pxweb(
            PXWEB_MUNICIPALITY_URL,
            [
                {"code": "Jahr", "selection": {"filter": "item", "values": ["2023"]}},
                {
                    "code": "Kanton (-) / Bezirk (>>) / Gemeinde (......)",
                    "selection": {"filter": "item", "values": chunk},
                },
                {"code": "Bevölkerungstyp", "selection": {"filter": "item", "values": ["1"]}},
                {"code": "Geburtsort", "selection": {"filter": "item", "values": ["-99999"]}},
                {"code": "Staatsangehörigkeit", "selection": {"filter": "item", "values": ["-99999"]}},
            ],
        )
        for row in iter_jsonstat_rows(payload):
            code = str(row["Kanton (-) / Bezirk (>>) / Gemeinde (......)"]).strip()
            populations[code] = int(round(float(row["value"])))
    return populations


def fetch_canton_observed_populations() -> dict[tuple[str, str], int]:
    canton_codes = [CANTON_BFS_CODE_BY_ABBREV[abbrev] for abbrev in CANTON_ABBREVIATIONS]
    payload = query_pxweb(
        PXWEB_CANTON_OBSERVED_URL,
        [
            {"code": "Jahr", "selection": {"filter": "item", "values": OBSERVED_SOURCE_YEARS}},
            {"code": "Kanton", "selection": {"filter": "item", "values": canton_codes}},
            {"code": "Staatsangehörigkeit (Kategorie)", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Geschlecht", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Alter", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Demografische Komponente", "selection": {"filter": "item", "values": ["14"]}},
        ],
    )
    populations: dict[tuple[str, str], int] = {}
    for row in iter_jsonstat_rows(payload):
        canton = CANTON_ABBREV_BY_BFS_CODE[str(row["Kanton"])]
        year = str(row["Jahr"])
        populations[(canton, year)] = int(round(float(row["value"])))
    return populations


def fetch_canton_projected_populations() -> dict[tuple[str, str], int]:
    metadata = fetch_json(PXWEB_CANTON_SCENARIO_URL)
    year_dimension = next(
        variable for variable in metadata["variables"] if variable["code"] == "Jahr"
    )
    year_code_by_label = {
        str(label): str(code)
        for code, label in zip(year_dimension["values"], year_dimension["valueTexts"])
    }
    year_codes = [year_code_by_label[year] for year in PROJECTED_SOURCE_YEARS]
    canton_codes = [CANTON_BFS_CODE_BY_ABBREV[abbrev] for abbrev in CANTON_ABBREVIATIONS]
    payload = query_pxweb(
        PXWEB_CANTON_SCENARIO_URL,
        [
            {"code": "Kanton", "selection": {"filter": "item", "values": canton_codes}},
            {"code": "Szenario-Variante", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Staatsangehörigkeit (Kategorie)", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Geschlecht", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Altersklasse", "selection": {"filter": "item", "values": ["0"]}},
            {"code": "Jahr", "selection": {"filter": "item", "values": year_codes}},
            {"code": "Beobachtungseinheit", "selection": {"filter": "item", "values": ["8"]}},
        ],
    )
    populations: dict[tuple[str, str], int] = {}
    for row in iter_jsonstat_rows(payload):
        canton = CANTON_ABBREV_BY_BFS_CODE[str(row["Kanton"])]
        year = str(row["Jahr__label"])
        populations[(canton, year)] = int(round(float(row["value"])))
    return populations


def write_assignments_csv(
    assignments: dict[str, dict[str, Any]],
    *,
    output_path: Path,
) -> None:
    fieldnames = [
        "station_name",
        "country",
        "region",
        "lat",
        "lon",
        "assignment_method",
        "municipality_code",
        "municipality_name",
        "municipality_canton",
        "assignment_year",
        "base_population_municipality_code",
        "base_population_municipality_name",
        "base_population_municipality_year",
        "candidate_count",
        "base_population_reference_year",
        "base_population",
        "base_population_source_type",
        "base_population_source_title",
        "base_population_source_url",
        "scaling_method",
        "shared_base_population_group_size",
        "population_split_method",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for station in sorted(assignments):
            writer.writerow(assignments[station])


def load_existing_assignments(
    *,
    input_path: Path,
    expected_station_names: set[str],
) -> dict[str, dict[str, Any]] | None:
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        return None
    assignments = {
        str(row.get("station_name", "")).strip(): dict(row)
        for row in rows
        if str(row.get("station_name", "")).strip()
    }
    if set(assignments) != expected_station_names:
        return None
    return assignments


def load_existing_scaling_rows(
    *,
    input_path: Path,
) -> list[dict[str, Any]] | None:
    if not input_path.exists():
        return None
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        return None
    expected_pairs = {
        (canton, planner_year)
        for canton in CANTON_ABBREVIATIONS
        for planner_year in SOURCE_YEAR_BY_PLANNER_YEAR
    }
    actual_pairs = {
        (
            str(row.get("canton", "")).strip(),
            str(row.get("planner_year_label", "")).strip(),
        )
        for row in rows
    }
    if actual_pairs != expected_pairs:
        return None
    return rows


def assignments_have_base_populations(assignments: dict[str, dict[str, Any]]) -> bool:
    for assignment in assignments.values():
        if str(assignment.get("country", "")).strip() != "CH":
            continue
        if _safe_positive_int(assignment.get("base_population")) is None:
            return False
    return True


def build_station_groups_by_base_municipality(
    assignments: dict[str, dict[str, Any]],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for station, assignment in assignments.items():
        if str(assignment.get("country", "")).strip() != "CH":
            continue
        municipality_code = str(assignment.get("base_population_municipality_code", "")).strip()
        if not municipality_code:
            raise ValueError(f"Swiss station {station} has no base municipality code.")
        groups.setdefault(municipality_code, []).append(station)
    return {
        municipality_code: sorted(stations)
        for municipality_code, stations in groups.items()
    }


def annotate_assignment_split_metadata(assignments: dict[str, dict[str, Any]]) -> None:
    groups = build_station_groups_by_base_municipality(assignments)
    split_note = (
        " When several modeled stations share this municipality-level base population, the municipality total is "
        "split across planner years in proportion to the stations' service counts in FilteredStations.csv; if a "
        "shared municipality has zero counted services in a given planner year, the split falls back to equal shares."
    )
    for station, assignment in assignments.items():
        if str(assignment.get("country", "")).strip() != "CH":
            assignment["shared_base_population_group_size"] = "1"
            assignment["population_split_method"] = "constant_foreign_exception"
            continue

        municipality_code = str(assignment.get("base_population_municipality_code", "")).strip()
        group_size = len(groups.get(municipality_code, [station]))
        assignment["shared_base_population_group_size"] = str(group_size)
        assignment["population_split_method"] = (
            "service_weighted_split_by_planner_year_with_equal_split_fallback"
            if group_size > 1
            else "single_station_municipality_allocation"
        )
        note = str(assignment.get("notes", "")).strip()
        if group_size > 1 and split_note.strip() not in note:
            assignment["notes"] = f"{note}{split_note}".strip()


def build_canton_scaling_rows(
    observed_populations: dict[tuple[str, str], int],
    projected_populations: dict[tuple[str, str], int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for canton in CANTON_ABBREVIATIONS:
        reference_population = observed_populations[(canton, "2023")]
        for planner_year in SOURCE_YEAR_BY_PLANNER_YEAR:
            source_year = SOURCE_YEAR_BY_PLANNER_YEAR[planner_year]
            key = (canton, source_year)
            if source_year in PROJECTED_SOURCE_YEARS:
                population = projected_populations[key]
                series_type = "projected_reference_scenario"
                scenario_variant = "AR-00-2025"
                source_table = "px-x-0104020000_106"
            else:
                population = observed_populations[key]
                series_type = "observed"
                scenario_variant = ""
                source_table = "px-x-0102020000_104"
            growth_factor = population / reference_population
            rows.append(
                {
                    "canton": canton,
                    "planner_year_label": planner_year,
                    "source_population_year": source_year,
                    "series_type": series_type,
                    "scenario_variant": scenario_variant,
                    "canton_population": population,
                    "reference_population_2023": reference_population,
                    "growth_factor_vs_2023": f"{growth_factor:.12f}",
                    "growth_percent_vs_2023": f"{(growth_factor - 1.0) * 100:.6f}",
                    "source_table": source_table,
                    "source_note": (
                        "Observed canton total population at 31 December."
                        if series_type == "observed"
                        else "Projected canton total population at 31 December from the BFS reference scenario."
                    ),
                }
            )
    rows.sort(key=lambda row: (row["canton"], list(SOURCE_YEAR_BY_PLANNER_YEAR.keys()).index(row["planner_year_label"])))
    return rows


def service_weight_for_station_and_year(row: dict[str, str], planner_year: str) -> float:
    service_columns = SERVICE_COLUMNS_BY_PLANNER_YEAR.get(planner_year, [planner_year])
    return sum(_safe_nonnegative_float(row.get(column_name, "")) for column_name in service_columns)


def allocate_integer_population_by_weight(
    total_population: int,
    station_weights: dict[str, float],
) -> dict[str, int]:
    stations = sorted(station_weights)
    if not stations:
        return {}
    if total_population <= 0:
        return {station: 0 for station in stations}

    positive_weights = {
        station: max(float(weight), 0.0)
        for station, weight in station_weights.items()
    }
    total_weight = sum(positive_weights.values())
    if total_weight <= 0:
        positive_weights = {station: 1.0 for station in stations}
        total_weight = float(len(stations))

    raw_allocations = {
        station: (total_population * positive_weights[station]) / total_weight
        for station in stations
    }
    floored = {
        station: int(math.floor(value))
        for station, value in raw_allocations.items()
    }
    allocated = sum(floored.values())
    remainder = total_population - allocated
    if remainder > 0:
        ranked = sorted(
            stations,
            key=lambda station: (
                raw_allocations[station] - floored[station],
                positive_weights[station],
                station,
            ),
            reverse=True,
        )
        for station in ranked[:remainder]:
            floored[station] += 1
    return floored


def write_scaling_csv(rows: list[dict[str, Any]], *, output_path: Path) -> None:
    fieldnames = [
        "canton",
        "planner_year_label",
        "source_population_year",
        "series_type",
        "scenario_variant",
        "canton_population",
        "reference_population_2023",
        "growth_factor_vs_2023",
        "growth_percent_vs_2023",
        "source_table",
        "source_note",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def populate_station_rows(
    station_rows: list[dict[str, str]],
    assignments: dict[str, dict[str, Any]],
    scaling_rows: list[dict[str, Any]],
) -> list[dict[str, str]]:
    scaling_by_canton_year = {
        (row["canton"], row["planner_year_label"]): float(row["growth_factor_vs_2023"])
        for row in scaling_rows
    }
    station_rows_by_name = {
        str(row.get("StationName", "")).strip(): row
        for row in station_rows
        if str(row.get("StationName", "")).strip()
    }
    groups = build_station_groups_by_base_municipality(assignments)
    station_populations_by_year: dict[tuple[str, str], int] = {}

    for municipality_code, stations in groups.items():
        base_populations = {
            station: _safe_positive_int(assignments[station].get("base_population"))
            for station in stations
        }
        distinct_base_populations = {
            int(population)
            for population in base_populations.values()
            if population is not None
        }
        if len(distinct_base_populations) != 1:
            raise ValueError(
                f"Stations sharing municipality code {municipality_code} do not share one base population: "
                f"{base_populations}"
            )
        municipality_base_population = distinct_base_populations.pop()

        cantons = {
            str(assignments[station].get("region", "")).strip()
            for station in stations
        }
        if len(cantons) != 1:
            raise ValueError(
                f"Stations sharing municipality code {municipality_code} span multiple cantons: {sorted(cantons)}"
            )
        canton = next(iter(cantons))

        for planner_year in POPULATION_COLUMN_TO_YEAR_LABEL.values():
            factor = scaling_by_canton_year[(canton, planner_year)]
            municipality_population_for_year = int(round(municipality_base_population * factor))
            station_weights = {
                station: service_weight_for_station_and_year(
                    station_rows_by_name[station],
                    planner_year,
                )
                for station in stations
            }
            allocations = allocate_integer_population_by_weight(
                municipality_population_for_year,
                station_weights,
            )
            for station, population in allocations.items():
                station_populations_by_year[(station, planner_year)] = population

    populated_rows: list[dict[str, str]] = []
    for row in station_rows:
        station = str(row.get("StationName", "")).strip()
        assignment = assignments.get(station)
        if not assignment:
            raise ValueError(f"No station population assignment exists for {station}.")
        base_population = int(assignment["base_population"])
        country = str(row.get("Country", "")).strip()
        region = str(row.get("Region", "")).strip()

        updated = dict(row)
        for column_name, planner_year in POPULATION_COLUMN_TO_YEAR_LABEL.items():
            if country == "CH":
                updated[column_name] = str(station_populations_by_year[(station, planner_year)])
            else:
                updated[column_name] = str(base_population)
        populated_rows.append(updated)
    return populated_rows


def write_filtered_stations(
    rows: list[dict[str, str]],
    fieldnames: list[str],
    *,
    output_path: Path,
) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    station_rows, fieldnames = load_filtered_station_rows()
    expected_station_names = {
        str(row.get("StationName", "")).strip()
        for row in station_rows
        if str(row.get("StationName", "")).strip()
    }
    assignments = (
        load_existing_assignments(
            input_path=ASSIGNMENTS_OUTPUT_PATH,
            expected_station_names=expected_station_names,
        )
        if args.reuse_existing_assignments
        else None
    )
    if assignments:
        print(
            f"Reusing existing municipality assignments from {ASSIGNMENTS_OUTPUT_PATH.name}",
            flush=True,
        )
    else:
        assignments = build_station_assignments(station_rows, sleep_seconds=args.sleep_seconds)
        write_assignments_csv(assignments, output_path=ASSIGNMENTS_OUTPUT_PATH)
        print(
            f"Wrote intermediate municipality assignments to {ASSIGNMENTS_OUTPUT_PATH.name}",
            flush=True,
        )
    apply_base_municipality_overrides(assignments)
    annotate_assignment_split_metadata(assignments)

    if assignments_have_base_populations(assignments):
        print(
            f"Reusing existing municipality base populations from {ASSIGNMENTS_OUTPUT_PATH.name}",
            flush=True,
        )
    else:
        municipality_codes = sorted(
            {
                assignment["base_population_municipality_code"]
                for assignment in assignments.values()
                if assignment["country"] == "CH" and assignment["base_population_municipality_code"]
            }
        )
        municipality_populations = fetch_municipality_base_populations(municipality_codes)
        for assignment in assignments.values():
            if assignment["country"] == "CH":
                municipality_code = str(assignment["base_population_municipality_code"]).strip()
                if municipality_code not in municipality_populations:
                    raise ValueError(
                        f"No 2023 BFS municipality population was returned for municipality code {municipality_code}."
                    )
                assignment["base_population"] = municipality_populations[municipality_code]
        write_assignments_csv(assignments, output_path=ASSIGNMENTS_OUTPUT_PATH)

    scaling_rows = load_existing_scaling_rows(input_path=SCALING_OUTPUT_PATH)
    if scaling_rows:
        print(
            f"Reusing existing canton scaling factors from {SCALING_OUTPUT_PATH.name}",
            flush=True,
        )
    else:
        observed_populations = fetch_canton_observed_populations()
        projected_populations = fetch_canton_projected_populations()
        scaling_rows = build_canton_scaling_rows(observed_populations, projected_populations)
    populated_station_rows = populate_station_rows(station_rows, assignments, scaling_rows)

    write_assignments_csv(assignments, output_path=ASSIGNMENTS_OUTPUT_PATH)
    write_scaling_csv(scaling_rows, output_path=SCALING_OUTPUT_PATH)
    write_filtered_stations(populated_station_rows, fieldnames, output_path=FILTERED_STATIONS_PATH)

    swiss_station_count = sum(1 for assignment in assignments.values() if assignment["country"] == "CH")
    foreign_station_count = len(assignments) - swiss_station_count
    print(
        f"Updated {len(populated_station_rows)} stations in {FILTERED_STATIONS_PATH.name}; "
        f"{swiss_station_count} use BFS municipality bases plus canton scaling and "
        f"{foreign_station_count} use documented fixed foreign-reference populations."
    )
    print(f"Wrote station assignment audit file to {ASSIGNMENTS_OUTPUT_PATH.name}")
    print(f"Wrote canton scaling audit file to {SCALING_OUTPUT_PATH.name}")


if __name__ == "__main__":
    main()
