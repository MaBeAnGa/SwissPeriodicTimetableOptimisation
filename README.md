# Swiss Periodic Timetable Modelling and Optimization

This repository accompanies the master's thesis:

> Matthias Benjamín Andrews García (2026), *Periodic Timetable Modelling & Matheuristic Event-Retiming Optimization of the Swiss Passenger Rail Network*, ETH Zürich.

The project has two connected parts. Part I reconstructs historical Swiss passenger-rail timetables and measures passenger-weighted journey-time components. Part II applies the same journey and demand-allocation concepts to detailed 2026 and 2035 networks and evaluates a three-step model-based matheuristic for local timetable retiming.

## Repository Structure

- `thesis/`: final LaTeX source, bibliography, the latest successful compiled thesis PDF, and figures used by the thesis. See `docs/final_thesis_completion_report.md` for the current build status.
- `executive_summary/`: final two-page executive summary source, bibliography, and compiled PDF.
- `short_paper/`: final IEEE-style condensed short-paper source, bibliography, compiled PDF, and its included OD-category figure.
- `poster/`: final A0 poster PDF and editable Affinity Designer source.
- `part_i/data/`: compact historical timetable, station, OD-weight, and generalized-cost inputs.
- `part_i/code/`: historical timetable reconstruction, routing, demand weighting, validation, and analysis scripts.
- `part_i/results/`: final compact validation and aggregate result files.
- `part_ii/data/`: 2026/2035 stations, Baseline and Step 1-3 compact timetables, fixed OD demand matrices, and baseline transfer tables.
- `part_ii/code/`: demand construction, routing, optimization Steps 1-3, scenario analysis, and figure-generation scripts.
- `part_ii/results/optimization/`: accepted-edit manifests and compact selection summaries.
- `part_ii/results/final/`: final corrected Step 3 aggregate results used in Results II.
- `docs/`: method notes, revision audit, scenario-size accounting, and the checksummed file manifest.

The Evolution of Swiss Passenger Rail Schedules application is intentionally excluded from this repository.

## Workflow

### Part I

1. Parse the compact historical timetable with `historical_timetable_loader.py`.
2. Construct the station-level 2023 demand anchor with the NPVM allocation and validation scripts.
3. Generate historical OD paths and generalized-cost skims with `build_all_od_pairings.py`.
4. Project and balance annual OD weights with `historical_od_weighting.py` and `build_od_pair_weights.py`.
5. Produce historical aggregates and figures with the scripts in `part_i/code/`.

### Part II

1. Build the year-specific timetable and routing structures with `timetable_parser_2026_2035.py`, `build_od_first_column_metrics_2026_2035.py`, and `build_full_od_matrices_2026_2035.py`.
2. Construct and balance the 2026/2035 OD demand matrices with the gravity, throughput, and transfer-correction scripts.
3. Run Step 1 using the transfer-context tables and station-board missed-transfer selector.
4. Run Step 2 from the validated Step 1 timetable using exact Step 1 reference-path evaluation.
5. Run Step 3 from the completed Step 2 timetable using the lower-threshold candidate context and sequential marginal reference-path rescoring.
6. Compose the final timetable, apply the two documented 2035 reversals, rebuild the national OD path state, and generate the compact final result tables.

The scripts retain their research-workflow interfaces and may require explicit command-line paths for large external scenario states. See `docs/methods/` and individual `--help` output before execution.

## Large Scenario States

The complete routed states are not hosted on GitHub. Their canonical JSON row-state sizes are:

| Scenario | Bytes | Decimal GB | Binary GiB |
|---|---:|---:|---:|
| Baseline | 286,181,932,008 | 286.181932 | 266.527694 |
| Optimized Step 1 | 286,470,205,506 | 286.470206 | 266.796169 |
| Optimized Step 2 | 286,393,095,918 | 286.393096 | 266.724355 |
| Optimized Step 3 | 287,266,858,365 | 287.266858 | 267.538110 |
| **Total** | **1,146,312,091,797** | **1,146.312092** | **1,067.586329** |

Counting the retained monolithic CSV exports alongside the canonical JSON row states would require approximately 2.008 TB (1.827 TiB), but those CSVs duplicate scenario contents. The deduplicated canonical row-state total is exactly 1,146,312,091,797 bytes, or 1.146 TB (1.043 TiB). The full states are available from the author on reasonable request when suitable physical storage is provided.

## Data Provenance

The thesis bibliography and method documentation identify the official NPVM, BFS, SBB, BAV, swisstopo, and timetable sources. The thesis figures are the manually prepared assets from the final Overleaf export `Master_Thesis-10`; only figures referenced by the final TeX source are included. This repository contains the compact research inputs, derived files, and final dissemination deliverables selected for reproducibility. Large raw archives, caches, experimental runs, and duplicate exports are excluded.

## Environment

Python 3.12 or later is recommended. Install the packages listed in `requirements.txt`. Some geospatial and graph-building workflows may also require system libraries used by GeoPandas, Fiona, and PROJ.

## Licensing

Source code is licensed under the MIT License in `LICENSE-CODE`. Original thesis text, figures, documentation, and derived data are licensed under Creative Commons Attribution 4.0 International in `LICENSE-CONTENT`.
