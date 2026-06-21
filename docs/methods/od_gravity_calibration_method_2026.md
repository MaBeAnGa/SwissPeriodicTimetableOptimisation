# 2026 Station-Level Gravity OD Calibration Method

Last updated: 2026-04-17 09:27 EEST

## Scope

This method builds a calibrated station-to-station demand matrix (currently run for 2026) using:

- 2026 OD service skims (frequency, fastest travel time, wait, rolling, dwell, transfer),
- station attributes (distributed population, line-service count),
- crow-fly station distance,
- and station-level SBB passenger-frequency observations for validation.

The same calibration script now also supports fixed-parameter replay for 2035 once the 2035 gravity-input package is ready.

## Data Preparation

Script:
- `prepare_2026_gravity_inputs.py`

Actions:
- Removes `Cernier Val-de-Ruz` from all 2026 modeling inputs.
- Builds a cleaned station list from `FilteredStations2026_2035.csv` in the same station order as the cleaned 2026 matrices.
- Builds `2026_station_crowfly_distance_km_gravity_input.csv` from station `lat/lon`.
- Writes a reproducibility manifest: `2026_gravity_input_manifest.json`.

Year harmonization (new):
- Exact SBB station counts are first scaled to target year 2026 using canton-level growth factors from `canton_population_scaling_factors.csv`.
- Scaling uses:
  - source year = station `SBB_pf_latest_year`,
  - source canton = station `NPVM_zone_canton`,
  - factor = `growth_factor_vs_2023(canton, 2026) / growth_factor_vs_2023(canton, source_year)`.
- This avoids mixing 2022/2024/2025 observed totals directly in a 2026-target calibration.

Output directory:
- `od_modeling_2026/`

## Calibration Model

Script:
- `calibrate_2026_gravity_model.py`

### Gravity form

For each OD pair `(i, j)`:

- `Flow_ij = exp(LogFlow_ij)`
- `LogFlow_ij = O_i + D_j + Pair_ij`

with:

- `O_i = a1 * log(1 + pop_i) + a2 * log(1 + lines_i)`
- `D_j = b1 * log(1 + pop_j) + b2 * log(1 + lines_j)`

and pair impedance/utility:

- `Pair_ij = c1 * log(1 + freq_ij)`
- `          - c2 * log(1 + crowfly_dist_ij)`
- `          - c3 * fastest_ij`
- `          - c4 * wait_ij`
- `          - c5 * rolling_ij`
- `          - c6 * dwell_ij`
- `          - c7 * transfer_ij`

Station-throughput proxy:

- `T_i = row_sum_i + col_sum_i`
- `T_i_adj = T_i * (1 + g * hub_i_norm)`

where `hub_i_norm` is min-max normalized `2026lines` (simple transfer-intensity proxy).

### Validation target (calibration stage)

Validation uses only exact SBB-matched stations:

- `SBB_pf_match_method == exact`
- and non-empty `SBB_pf_latest_dtv`

Observed counts are year-aligned before validation:
- `obs_i = SBB_pf_latest_dtv_i * scale_to_2026_i`

Model scale is fitted analytically per candidate:

- `k = argmin ||obs - k * pred||^2`
- OD and throughput outputs are multiplied by `k`.

### Post-calibration DTV marginal balancing

After selecting the best gravity candidate, the exported OD matrix is now
bi-proportionally balanced (IPF/RAS) so station marginals match station `DTV`
targets.

- Target vector: `DTV` column from `FilteredStations2026_gravity_input.csv`
  (fallback to year-aligned `SBB_pf_latest_dtv_scaled_to_target_year` if `DTV` is missing).
- For exact SBB-matched stations, target is always replaced by year-aligned
  SBB values before balancing.
- Constraints imposed:
  - `row_sum_i = DTV_i / 2`
  - `col_sum_i = DTV_i / 2`
  - therefore `row_sum_i + col_sum_i = DTV_i`
- Zero diagonal is preserved unless `--allow-self-demand` is used.

This keeps the model-implied OD pattern while forcing standardized station totals.

### Accessibility-induced demand uplift (new, optional)

The script now supports an induced-demand step before IPF balancing:

1. Compute baseline and scenario station accessibility from pair utility terms:
   - `A_i = sum_j exp(Pair_ij)` (diagonal excluded when self-demand is disabled).
2. Compute accessibility change:
   - `ΔlnA_i = ln(A_i_scenario) - ln(A_i_baseline)`.
3. Convert to multiplicative station demand factors:
   - `fO_i = exp(ηO * ΔlnA_i)`, `fD_i = exp(ηD * ΔlnA_i)`.
4. Convert to one throughput factor compatible with equal row/column constraints:
   - `fT_i = sqrt(fO_i * fD_i)`.
5. Adjust station throughput target before IPF:
   - `DTV_i_adj = DTV_i_base * fT_i` (with optional network-total preservation).

Controls:
- `--enable-induced-demand`
- `--induced-base-input-dir`
- `--induced-origin-elasticity`
- `--induced-dest-elasticity`
- `--induced-factor-min`, `--induced-factor-max`
- `--induced-preserve-total-throughput`

This is scenario logic to capture timetable/frequency accessibility improvements as net-new demand on top of base marginals.

### Search strategy

- Random search over parameter ranges (default `n_samples=1200`).
- Candidate ranking uses:
  - `NRMSE = RMSE / mean(obs)`,
  - `MAPE`,
  - `score = NRMSE + 0.5 * (MAPE / 100)`.
- Best candidate is used to produce the exported OD matrix.

## Outputs

From calibration script:

- `2026_gravity_calibration_candidates.csv`
- `2026_gravity_calibration_top_candidates.csv`
- `2026_OD_matrix_Demand_gravity_best_raw.csv` (best gravity matrix before DTV balancing)
- `2026_OD_matrix_Demand_gravity_best.csv`
- `2026_gravity_station_fit.csv`
- `2026_gravity_calibration_summary.json`
- `2026_gravity_calibration_diagnostics.json` (quantiles + parameter diagnostics)
- `2026_gravity_parameter_score_correlations.csv`

File prefixes are now year-driven (`<model_year>_...`) so the same script can write 2035 outputs.

## Latest calibration run

- Timestamp: 2026-04-13 18:10 EEST
- Samples evaluated: `2000`
- Validation stations (exact SBB matches): `1148`
- Population scaling to 2026:
  - stations with canton+year scaling pair: `1145`
  - stations with non-trivial scaling factor: `1145`
- Best gravity-candidate metrics (before DTV balancing):
  - `R² = 0.5083`
  - `MAE = 2844.54`
  - `RMSE = 12042.65`
  - `MAPE = 97.40%`
- DTV marginal fit diagnostics (after IPF balancing):
  - `converged = True`
  - `iterations = 7090`
  - `max_abs_throughput_error = 0.0001614`
  - `mae_throughput_error = 9.10e-07`

## High-detail rerun command (2026)

Use this when you want a broader random search and fully saved diagnostics:

```bash
cd "<USER_HOME>/iCloud Drive (Archive)/Documents/Documents - Matthias’s 16\" MacBook Pro (246)/Paper Pythons/Master Thesis/2nd Half of Project (Comparison 2026 and 2035)"

N_SAMPLES=20000 \
SEED=20260413 \
PROGRESS_EVERY=250 \
./run_2026_gravity_highres.sh
```

This writes:
- candidate table for all samples,
- top-candidate table,
- detailed diagnostics JSON,
- parameter/score correlation CSV,
- station-fit table,
- summary JSON,
- timestamped terminal log.

## 2035-ready projection mode (when 2035 inputs are ready)

The script now supports replaying fixed 2026-calibrated parameters on 2035 matrices:

- `--model-year 2035`
- `--fixed-params-file od_modeling_2026/2026_gravity_calibration_summary.json`
- `--enable-induced-demand` with baseline set to 2026 input package

Wrapper script added:
- `run_2035_gravity_projection_from_2026.sh`

## 2035 projection from 2026 demand + path-count change (new)

For the 2035 run requested in this thread, a second projection path is now implemented to avoid
re-running giant route builds and to keep memory stable with very large files:

- `project_2035_demand_from_pop_and_paths.py`
  - `extract-pathcount-ratio`:
    - streams `2026_OD_matrix_Nondominated_paths_full.csv` and `2035_OD_matrix_Nondominated_paths_full.csv`,
    - extracts only top-level `path_count` from each AB cell,
    - writes compact arrays:
      - `od_modeling_2035/2035_pathcount_ratio_2035_vs_2026.npy`
      - `od_modeling_2035/2026_pathcount_matrix.npy`
      - `od_modeling_2035/2035_pathcount_matrix.npy`
    - supports checkpoint/resume.
  - `build-2035-demand`:
    - starts from `od_modeling_2026/2026_OD_matrix_Demand_gravity_best.csv`,
    - applies station population growth (`2035_distributedpop / 2026_distributedpop`),
    - applies pair-level service multiplier from path-count ratio:
      - `ratio_ij = (count2035_ij + s) / (count2026_ij + s)`
      - `mult_ij = ratio_ij ^ epsilon_service`
    - applies IPF to projected 2035 station targets for direct comparability with 2026 constrained outputs,
    - writes:
      - `od_modeling_2035/2035_OD_matrix_Demand_gravity_best_raw.csv`
      - `od_modeling_2035/2035_OD_matrix_Demand_gravity_best.csv`
      - `od_modeling_2035/2035_station_throughput_projection.csv`
      - `od_modeling_2035/2035_demand_projection_summary.json`

Crash-safe runner wrappers:
- `run_stage1_extract_pathcount_ratio.sh`
- `run_stage2_build_2035_and_attach_demand.sh`

### Frequency/service elasticity interpretation

The script parameterizes induced service-response as:

- `Demand multiplier when service doubles = 2^epsilon_service`

Default `epsilon_service=0.50` implies:
- doubling service -> `~1.414x` pair-level demand multiplier before IPF balancing.

This elasticity is explicitly configurable for sensitivity tests.

### Wait-time induced demand option (new)

Stage-2 projection now also supports an explicit wait-time elasticity term:

- `wait_ratio_ij = (wait2035_ij + s_wait) / (wait2026_ij + s_wait)`
- `wait_mult_ij = wait_ratio_ij ^ epsilon_wait`
- total pair multiplier becomes:
  - `mult_ij = (path_ratio_ij ^ epsilon_service) * wait_mult_ij`

Default in the stage-2 runner:
- `epsilon_wait = -0.40` (configurable via `WAIT_ELASTICITY`).

Important:
- This is a literature-based scenario parameter, not confirmed as one fixed official SBB SIMBA coefficient.
- To avoid overreaction and double-counting with path-count effects, clipping and sensitivity runs are recommended.

## Adding daily demand field to nondominated OD cells (new)

Script:
- `attach_daily_demand_to_nondominated_csv.py`

Purpose:
- stream-adds top-level `daily_demand_total` into each AB cell JSON in the huge nondominated CSVs,
- without loading whole files into memory,
- with checkpoint/resume.

Outputs used in stage runner:
- `2026_OD_matrix_Nondominated_paths_full_with_daily_demand.csv`
- `2035_OD_matrix_Nondominated_paths_full_with_daily_demand.csv`

## Step 3: Path-level demand split within each AB cell (new)

Script:
- `allocate_path_level_demand_from_cell_totals.py`

Runner:
- `run_step3_allocate_path_level_demand.sh`

Logic per AB cell:
1. Read `daily_demand_total` at cell level.
2. Expand each observed nondominated path departure as a repeating service every 120 minutes from 05:00 onward.
3. Keep repeated instances with departure `< 24:00` and (default) arrival `< 24:00`.
4. Compute each path’s **dominance minutes before departure** over the day:
   - dominance for a departure instance = minutes since previous departure instance.
5. Split `daily_demand_total` proportionally by dominance minutes.

Per-path fields written into each cell JSON:
- `dominance_minutes_day`
- `daily_departure_instances`
- `daily_demand_share`
- `daily_demand_allocated`

Cell-level audit fields:
- `allocated_daily_demand_total`
- `allocation_unassigned_demand`
- allocation metadata (`cycle`, `day_start`, `day_end`, method string)

Output files:
- `2026_OD_matrix_Nondominated_paths_full_with_daily_demand_and_path_alloc.csv`
- `2035_OD_matrix_Nondominated_paths_full_with_daily_demand_and_path_alloc.csv`

## Step 4: Service-level load reconstruction from AB path allocations (new)

Script:
- `aggregate_service_loads_from_path_alloc.py`

Runner:
- `run_step4_aggregate_service_loads.sh`

Input:
- `*_OD_matrix_Nondominated_paths_full_with_daily_demand_and_path_alloc.csv`

This stage maps path-level daily demand (from Step 3) onto train/service segments
and trip instances using deterministic flow conservation.

Let:
- `q_p` = `daily_demand_allocated` for path `p` in an AB cell.
- `I_p` = `daily_departure_instances` for path `p`.
- `Legs(p)` = ordered list of legs in path `p`, each with service and from/to stations.

Computed outputs:
1. Service-segment daily load:
   - `L_{s,u,v} = Σ_p q_p * 1[(u,v,s) ∈ Legs(p)]`
2. Network segment daily load (all services merged):
   - `L_{u,v} = Σ_s L_{s,u,v}`
3. Service totals:
   - Daily unique riders per service:
     - `R_s = Σ_p q_p * 1[s used at least once in p]`
   - Daily passenger-leg volume per service:
     - `V_s = Σ_p q_p * (# legs in p that use s)`
4. Trip-instance segment average load:
   - `A_{t,s,u,v} = Σ_p (q_p / I_p) * 1[(t,s,u,v) ∈ Legs(p)]`

Interpretation:
- `R_s` counts a rider once per service used by the rider’s chosen path.
- `V_s` counts rider-leg traversals and is therefore larger/equal to `R_s`.
- `A_{t,s,u,v}` is an average-per-departure-instance load, not total daily volume.

Flow-conservation property:
- For each AB cell, Step 3 guarantees `Σ_p q_p = daily_demand_total`.
- Step 4 redistributes these same `q_p` values across service legs only; no new AB demand is created.

Engineering constraints for large files:
- Streaming CSV read (row-by-row, no full-matrix load in RAM).
- Incremental aggregation persisted in SQLite (`WAL`) with checkpoint/resume.
- Configurable flush interval (`--flush-every-rows`) to bound memory.

Main outputs per year:
- `<year>_service_segment_daily_loads.csv`
- `<year>_network_segment_daily_loads.csv`
- `<year>_service_totals_daily.csv`
- `<year>_trip_instance_segment_avg_loads.csv`
- `<year>_step4_run_stats.csv`

## Step 5: Segment-demand visualization map (2026 vs 2035)

Script:
- `build_segment_demand_map_2026_2035.py`

Purpose:
- Build a zoomable map where line width encodes passenger demand on station-pair segments,
  with separate 2026 and 2035 overlay layers.

Inputs:
- Step-4 outputs:
  - `/tmp/od_modeling_step4/2026/2026_network_segment_daily_loads.csv`
  - `/tmp/od_modeling_step4/2026/2026_service_segment_daily_loads.csv`
  - `/tmp/od_modeling_step4/2035/2035_network_segment_daily_loads.csv`
  - `/tmp/od_modeling_step4/2035/2035_service_segment_daily_loads.csv`
- Station coordinates:
  - `FilteredStations2026_2035.csv` (`StationName`, `lat`, `lon`)

Method:
1. Build station-pair segment set from all year-specific segment loads.
2. Convert directed segment loads to undirected station-pair totals:
   - `daily_pair = daily(a->b) + daily(b->a)`.
3. Normalize to per-hour loads:
   - `pax_per_hour = daily_pair / service_hours`.
   - Default `service_hours = (24:00 - 05:00) = 19h`, matching repeating-pattern assumptions used earlier.
4. Aggregate service contributions on each station pair:
   - For each pair, keep per-service `daily_pax` and `pax_per_hour`.
5. Render map with two overlay layers (2026/2035), dark-red line styles, and hover/popup metadata.

Outputs:
- `adjacent_station_pairs_demand_2026_2035.csv`
- `adjacent_station_pair_service_demand_2026_2035.csv`
- `segment_demand_map_2026_2035.html`
- `segment_demand_map_summary_2026_2035.json`

### Step 5B (2026-04-17): High-accuracy, low-memory rebuild with targeted Hallwil/Hinwil correction

Script:
- `build_segment_demand_map_from_pathalloc_streaming.py`

Runner:
- `run_step5_streaming_high_accuracy.sh`

Why this variant:
- Prior map generation used pre-aggregated segment tables only.
- After discovering the Seon–Boniswil intermediate-stop typo (`Hinwil` vs `Hallwil`), this variant
  recomputes only affected OD cells **on-the-fly** while streaming the huge path-allocation CSVs,
  then immediately aggregates adjacent-segment demand. This avoids full 130+ GB matrix rewrites.

Method specifics:
1. Stream each row of:
   - `2026_OD_matrix_Nondominated_paths_full_with_daily_demand_and_path_alloc.csv`
   - `2035_OD_matrix_Nondominated_paths_full_with_daily_demand_and_path_alloc.csv`
2. For cells affected by target stations (`Hallwil`,`Hinwil` by default), rebuild cell payload using:
   - current model cache (`od_model_cache_2026_2035.pkl`)
   - year-specific demand matrix (`2026_OD_matrix_Demand_gravity_best.csv`, `2035_OD_matrix_Demand_gravity_best.csv`)
   - path allocation logic consistent with Step 3.
3. Expand each leg via `canonical_trip_id` station sequence so all intermediate stations on the same line
   and transfer points are represented in adjacent-pair accumulation.
4. Persist incremental totals in SQLite (WAL mode) with row checkpoints, then export final CSVs + HTML map.

Engineering behavior:
- Memory-bounded (streaming rows, no full matrix load).
- Crash-safe resume (`--resume`) from checkpointed row index.
- Heavy outputs/state are written to `/tmp` by default to avoid iCloud sync overhead during long runs.

## Important interpretation notes

- This is a station-throughput-calibrated gravity model, not a perfect recovery of latent OD truth.
- SBB station-frequency counts include transfer boardings/alightings, so a hub correction term is included, but transfer behavior is still approximated.
- ZVV/VBZ open table extracts used in this step expose `Einsteiger`/`Aussteiger`/`Besetzung` but no explicit `Umsteiger` field in the provided CSV schema; therefore transfer-discount validation currently falls back to internal model diagnostics rather than a direct external transfer count field.
- Estimated `DTV` values are not used in the objective. Only exact observed SBB matches are used for scoring.
- The final exported OD matrix is now DTV-constrained via IPF/RAS; therefore station-throughput error vs those same DTV targets is near-zero by construction.
- Parameter values are empirical fit coefficients for this pipeline; they are not universal transport constants.
- Induced-demand elasticities (`ηO`, `ηD`) are scenario controls, not hard physical constants; they should be stress-tested in low/base/high bands.
