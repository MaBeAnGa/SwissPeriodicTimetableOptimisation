# All_OD_Pairings validation

## Purpose

This note records the implemented historical OD pairing table that now extends the accepted 2023
selected-station analysis layer across all timetable years in the thesis system.

## Implemented model

- Base layer: accepted `historical_od_pair_weights_reference_2023.csv` 2023 analysis table
- Time-varying impedance: representative ready-time timetable skims at `06:00`, `07:00`, and `08:00`, used as the repeating service proxy for the thesis operating day `05:00-00:00`
- Generalized cost used in the pivot: elapsed time from the ready time to arrival, so initial waiting is included
- Growth of marginals: station `*_pop` values from `FilteredStations.csv`
- Balancing: doubly constrained IPF on the 131-station analysis layer
- Applied-method precedent: DfT matrix-building guidance describes both demand-pivot approaches and the Furness process as standard reference-matrix forecasting tools

## Calibration

- Calibrated beta: `1.777396`
- Weighted R^2: `0.44006`
- Rows used in calibration: `17030`

## Skim diagnostics

- Generalized-cost skim model: `representative_ready_times_elapsed_time_06_07_08`
- Ready times: `06:00, 07:00, 08:00`
- Thesis operating-day assumption: `05:00-00:00`
- Representative skim window used in practice: `06:00-08:00`
- Mean ready-time runtime: `1.0126 s`
- Fully reachable pair share: `1.0`
- Fully unreachable pair count: `0`

## Output diagnostics

- Output rows: `391690`
- Distinct years: `23`
- Active projected rows: `391690`
- Explicit zero/excluded rows: `0`
- Inactive station-year zero rows: `0`

## Interpretation

The resulting `All_OD_Pairings.csv` is not a set of observed matrices for every year. It is a
year-specific **modelled OD series** anchored in the accepted 2023 observed/hybrid table, then
updated across years with timetable-based generalized cost and population-scaled marginals.
