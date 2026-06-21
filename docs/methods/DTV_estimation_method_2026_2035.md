# DTV Estimation for Unmatched Stations (2026/2035 Model)

## Goal

Estimate `DTV` for stations without an official matched SBB station-frequency value, while preserving observed values where exact matches exist.

## Input Data

- Source table: `2nd Half of Project (Comparison 2026 and 2035)/FilteredStations2026_2035.csv`
- Observed target column: `SBB_pf_latest_dtv` (official SBB latest available year)
- Raw predictor columns: `2026_distributedpop`, `2035_distributedpop`, `2026lines`, `2035lines`
- Model predictors (derived): `distributedpop_mean = mean(2026_distributedpop, 2035_distributedpop)` and `lines_mean = mean(2026lines, 2035lines)`

## Model

- Type: log-linear ordinary least squares (OLS)
- Transformations: `log(1 + x)` on both target and predictors
- Fitted equation: `log(1 + DTV) = -1.795817 + (0.691244) * log(1 + distributedpop_mean) + (3.936396) * log(1 + lines_mean) + (-0.197348) * log(1 + distributedpop_mean) * log(1 + lines_mean)`

## Assignment Rule for Final `DTV`

- If `SBB_pf_match_method == exact` and `SBB_pf_latest_dtv` exists: `DTV = SBB_pf_latest_dtv` (observed)
- Otherwise: `DTV = round(model_estimate)`

## Coverage

- Total stations: `1597`
- Stations used to train model (exact observed DTV): `1149`
- Stations receiving model-estimated DTV: `448`

## In-Sample Fit (on training stations)

- `R^2` on log scale: `0.7681`
- `R^2` on raw scale: `0.4756`
- MAE (raw passengers/day): `1651.25`
- RMSE (raw passengers/day): `12326.83`
- MAPE: `75.68%`

## Notes

- This is an explanatory proxy model, not a replacement for observed counts.
- High-traffic hubs are harder to fit exactly with simple station-level predictors.
- The resulting `DTV` column is therefore best interpreted as: observed where available, model-consistent estimate otherwise.
