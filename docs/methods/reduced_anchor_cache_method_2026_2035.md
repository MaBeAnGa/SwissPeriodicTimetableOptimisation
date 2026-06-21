# Reduced Anchor Cache Build

- Purpose: precompute reduced-anchor OD skims as a lossless acceleration layer for later full-station OD generation.
- Day window assumptions: 05:00-00:00; representative pattern from 05:00-07:00.
- Routing and decomposition logic: identical to `build_od_first_column_metrics_2026_2035.py`.
- Transfer handling: no synthetic transfer penalty is inserted at any anchor in these cached routes; rolling/dwell/transfer values come directly from full-route queries.
- Zero-minute rolling/dwell handling: already embedded in the shared core logic (20-second floor with end-to-end preservation).

- 2026: 52 anchors
- 2035: 52 anchors

## Output files per year

- `<YEAR>_OD_matrix_Frequency_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Fastest_observed_connection_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Wait_time_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Rolling_time_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Dwell_time_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Transfer_time_reduced_anchor.csv`
- `<YEAR>_OD_matrix_PathDetails_reduced_anchor.csv`
- `<YEAR>_OD_matrix_Anchors_reduced_anchor.csv`

