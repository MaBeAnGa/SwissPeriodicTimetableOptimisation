# Station OD Validation 2023

## Scope

- Reference year: `2023`
- Station set: `overall_analysis_131` (131 stations)
- Zone count with allocations: `8688`
- Allocation model: `hybrid_zone_to_station_distance_service_2023_k4_pool6_d2_s0.5_a3_z1.75`

## Allocation Model

- Candidate construction:
  - all stations physically inside a zone polygon are retained
  - nearest stations are then added until at least `4` candidates are present
  - the nearest-station search pool is `6`
- Weight formula:
  - service term = service_2023 ^ `0.5`
  - access term = 1 / (distance_km + `3.0`) ^ `2.0`
  - in-zone stations receive a multiplier of `1.75`

## Allocation Diagnostics

- Zones with at least one in-zone target station: `131` / `8688`
- Average allocation share assigned to in-zone stations: `0.0125`
- Average highest single-station share per zone: `0.5406`
- Median highest single-station share per zone: `0.5030`
- Candidate-count distribution: `{'6': 8688}`

## Matrix Aggregation Diagnostics

- Matrix lines processed: `17026407`
- Distinct zone IDs seen in the matrix: `8509`
- Zone IDs missing an allocation: `9`
- Total zone trips processed: `3927345.829`
- Total station trips allocated: `3927345.830`
- Off-diagonal station trips retained in the final directed OD matrix: `2564451.533`
- Same-station diagonal trips kept only for validation: `1362894.297`
- Trips skipped because a zone allocation was missing: `7879.920`

## Top Origin Stations by Off-Diagonal Production

- `Zürich HB`: `246830.181`
- `Zürich Oerlikon`: `160682.009`
- `Zürich Altstetten`: `121325.023`
- `Bern`: `96210.581`
- `Lausanne`: `91822.858`
- `Zürich Flughafen`: `91626.840`
- `Basel SBB`: `90184.676`
- `Genève`: `67000.987`
- `Thalwil`: `60914.209`
- `Morges`: `54212.772`
- `Winterthur`: `53216.992`
- `Luzern`: `48333.098`
- `Liestal`: `43468.554`
- `Dietikon`: `42849.127`
- `Nyon`: `37081.191`

## Top Destination Stations by Off-Diagonal Attraction

- `Zürich HB`: `247667.124`
- `Zürich Oerlikon`: `161159.112`
- `Zürich Altstetten`: `120693.679`
- `Bern`: `96904.683`
- `Lausanne`: `92307.294`
- `Zürich Flughafen`: `91177.192`
- `Basel SBB`: `90091.407`
- `Genève`: `66929.645`
- `Thalwil`: `60243.587`
- `Winterthur`: `54135.846`
- `Morges`: `53373.661`
- `Luzern`: `48017.811`
- `Liestal`: `43418.825`
- `Dietikon`: `42592.825`
- `Nyon`: `36659.171`

## Top Directed OD Pairs

- `Zürich Oerlikon -> Zürich HB`: `62936.835`
- `Zürich HB -> Zürich Oerlikon`: `62925.184`
- `Zürich Altstetten -> Zürich HB`: `47340.246`
- `Zürich HB -> Zürich Altstetten`: `47112.563`
- `Morges -> Lausanne`: `29296.627`
- `Lausanne -> Morges`: `28698.013`
- `Zürich Flughafen -> Zürich HB`: `26859.802`
- `Liestal -> Basel SBB`: `26769.870`
- `Basel SBB -> Liestal`: `26765.081`
- `Zürich HB -> Zürich Flughafen`: `26741.814`
- `Nyon -> Genève`: `25380.462`
- `Genève -> Nyon`: `25328.973`
- `Zürich Altstetten -> Zürich Oerlikon`: `23344.115`
- `Zürich Oerlikon -> Zürich Altstetten`: `23257.586`
- `Thalwil -> Zürich HB`: `21247.490`
- `Zürich HB -> Thalwil`: `20872.690`
- `Zürich Oerlikon -> Zürich Flughafen`: `17161.425`
- `Zürich Flughafen -> Zürich Oerlikon`: `17134.040`
- `Basel SBB -> Rheinfelden`: `15088.674`
- `Rheinfelden -> Basel SBB`: `15041.702`

## Interpretation

- This file validates the first 2023 station-OD anchor, not yet the historical extrapolation.
- The observed part is the NPVM 2023 zone-to-zone PT matrix itself.
- The hybrid part is the zone-to-station allocation layer, which is fully written out in `npvm_zone_station_allocation_2023.csv` for auditability.
- The top production and attraction rankings above are based on the exported off-diagonal station matrix; same-station demand is reported separately through the diagonal total.
- The strongest off-diagonal pairs are currently short metropolitan links such as `Zürich Oerlikon -> Zürich HB` and `Zürich Altstetten -> Zürich HB`; before live integration, the thesis should decide whether that raw metropolitan concentration is desirable for the network-wide weighting purpose.
- Update: that later step has now been carried through in the live workspace. The accepted 2023 inter-agglomeration layer derived from this anchor is now propagated across all timetable years in `All_OD_Pairings.csv`, while this document remains the stage-specific validation record for the original 2023 anchor.
