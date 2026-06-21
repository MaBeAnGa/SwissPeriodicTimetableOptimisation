from __future__ import annotations

import json
import re
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Tuple


BUILD_MARKER = "\n# -----------------------------\n# Build system\n"
FUTURE_IMPORT = "from __future__ import annotations\n"
TRANSFER_OVERRIDE_SOURCE = (
    Path(__file__).resolve().parent / "station_transfer_overrides.py"
).read_text(encoding="utf-8")
CORRECTED_EFFICIENCY_BREAKDOWN_SOURCE = r'''def calculate_efficiency_breakdown(legs: List[dict]) -> Tuple[float, float, float]:
    rolling = 0.0
    dwell = 0.0
    transfer = 0.0
    if not legs:
        return 0.0, 0.0, 0.0

    def as_float(value) -> float:
        try:
            return float(value)
        except Exception:
            return 0.0

    rolling += max(0.0, as_float(legs[0].get('arr_time')) - as_float(legs[0].get('dep_time')))
    for i in range(1, len(legs)):
        prev_leg, curr_leg = legs[i - 1], legs[i]
        rolling += max(0.0, as_float(curr_leg.get('arr_time')) - as_float(curr_leg.get('dep_time')))
        wait_time = as_float(curr_leg.get('dep_time')) - as_float(prev_leg.get('arr_time'))

        if curr_leg.get('canonical_trip_id') == prev_leg.get('canonical_trip_id'):
            if abs(wait_time) < 1e-9:
                assumed_dwell = 1.0 / 3.0
                dwell += assumed_dwell
                rolling -= assumed_dwell
            else:
                dwell += max(0.0, wait_time)
        else:
            transfer += max(0.0, wait_time)

    return max(0.0, rolling), dwell, transfer
'''


@lru_cache(maxsize=4)
def load_notebook_module(notebook_path: str):
    """
    Load the notebook's timetable definitions as a regular Python module.

    The website reuses the routing logic directly from the thesis notebook so we
    only maintain one source of truth for the timetable model and journey search.
    """
    notebook = Path(notebook_path)
    notebook_json = json.loads(notebook.read_text(encoding="utf-8"))

    code_source = None
    for cell in notebook_json.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        if "class SwissTimetableSystem" in source and "def earliest_arrival_query" in source:
            code_source = source
            break

    if code_source is None:
        raise ValueError(f"Could not locate timetable definitions in {notebook}")

    build_idx = code_source.find(BUILD_MARKER)
    if build_idx != -1:
        code_source = code_source[:build_idx]

    code_source = code_source.replace(FUTURE_IMPORT, "")
    code_source = re.sub(
        r"def calculate_efficiency_breakdown\(legs: List\[dict\]\) -> Tuple\[int, int, int\]:\n"
        r"(?:    .*(?:\n|$))+?"
        r"(?=\ndef plot_structural_efficiency)",
        CORRECTED_EFFICIENCY_BREAKDOWN_SOURCE,
        code_source,
        count=1,
    )
    module_source = FUTURE_IMPORT + code_source + "\n\n" + TRANSFER_OVERRIDE_SOURCE.strip() + "\n"

    module_name = f"_historical_thesis_{notebook.stem.lower().replace(' ', '_')}"
    module = types.ModuleType(module_name)
    module.__file__ = str(notebook)
    sys.modules[module_name] = module

    exec(compile(module_source, str(notebook), "exec"), module.__dict__)
    return module


def build_system(project_dir: str | Path, verbose_validation: bool = False) -> Tuple[object, object]:
    project_path = Path(project_dir)
    notebook_path = project_path / "Master Thesis Notebook.ipynb"
    timetable_path = project_path / "TimetableHistory.csv"
    filtered_stations_path = project_path / "FilteredStations.csv"

    module = load_notebook_module(str(notebook_path))
    system = module.SwissTimetableSystem(
        timetable_file=str(timetable_path),
        filtered_stations_file=str(filtered_stations_path),
        default_transfer_min=module.DEFAULT_TRANSFER_MIN,
        verbose_validation=verbose_validation,
    )
    system.build_all_models()
    return module, system
