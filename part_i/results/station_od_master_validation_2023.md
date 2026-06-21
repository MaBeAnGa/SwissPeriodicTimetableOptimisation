# Station OD Validation 2023

## Scope

- Reference year: `2023`
- Station set: `modeled_station_system_250` (250 stations)
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

- Zones with at least one in-zone target station: `248` / `8688`
- Average allocation share assigned to in-zone stations: `0.0199`
- Average highest single-station share per zone: `0.4834`
- Median highest single-station share per zone: `0.4447`
- Candidate-count distribution: `{'6': 8688}`

## Matrix Aggregation Diagnostics

- Matrix lines processed: `17026407`
- Distinct zone IDs seen in the matrix: `8509`
- Zone IDs missing an allocation: `9`
- Total zone trips processed: `3927345.829`
- Total station trips allocated: `3927345.830`
- Off-diagonal station trips retained in the final directed OD matrix: `2951903.094`
- Same-station diagonal trips kept only for validation: `975442.736`
- Trips skipped because a zone allocation was missing: `7879.920`

## Top Origin Stations by Off-Diagonal Production

- `Zürich HB`: `240300.250`
- `Zürich Oerlikon`: `154092.263`
- `Zürich Altstetten`: `111365.923`
- `Bern`: `96863.898`
- `Genève`: `94558.418`
- `Basel SBB`: `90310.933`
- `Lausanne`: `84721.503`
- `Zürich Flughafen`: `73345.829`
- `Genève-Eaux-Vives`: `55291.901`
- `Lancy-Pont-Rouge`: `51575.452`
- `Renens VD`: `49876.280`
- `Thalwil`: `47548.619`
- `Winterthur`: `47470.731`
- `Genève-Aéroport`: `44406.493`
- `Liestal`: `40684.808`

## Top Destination Stations by Off-Diagonal Attraction

- `Zürich HB`: `240795.622`
- `Zürich Oerlikon`: `154500.292`
- `Zürich Altstetten`: `110761.405`
- `Bern`: `97624.256`
- `Genève`: `94615.690`
- `Basel SBB`: `90214.065`
- `Lausanne`: `85420.086`
- `Zürich Flughafen`: `73211.401`
- `Genève-Eaux-Vives`: `55599.969`
- `Lancy-Pont-Rouge`: `51980.673`
- `Renens VD`: `49614.493`
- `Winterthur`: `48309.555`
- `Thalwil`: `46894.198`
- `Genève-Aéroport`: `43442.291`
- `Liestal`: `40642.786`

## Top Directed OD Pairs

- `Zürich HB -> Zürich Oerlikon`: `59946.632`
- `Zürich Oerlikon -> Zürich HB`: `59902.795`
- `Zürich Altstetten -> Zürich HB`: `43109.766`
- `Zürich HB -> Zürich Altstetten`: `42973.553`
- `Renens VD -> Lausanne`: `26542.119`
- `Lausanne -> Renens VD`: `26415.015`
- `Basel SBB -> Liestal`: `23989.948`
- `Liestal -> Basel SBB`: `23986.724`
- `Genève -> Genève-Eaux-Vives`: `21375.147`
- `Genève-Eaux-Vives -> Genève`: `21297.589`
- `Zürich Altstetten -> Zürich Oerlikon`: `20597.534`
- `Zürich Oerlikon -> Zürich Altstetten`: `20535.683`
- `Genève -> Lancy-Pont-Rouge`: `20329.038`
- `Lancy-Pont-Rouge -> Genève`: `20155.262`
- `Zürich HB -> Zürich Flughafen`: `19092.840`
- `Zürich Flughafen -> Zürich HB`: `19031.135`
- `Thalwil -> Zürich HB`: `16919.979`
- `Zürich HB -> Thalwil`: `16581.974`
- `Genève-Aéroport -> Genève`: `15603.406`
- `Morges -> Lausanne`: `15229.668`

## Interpretation

- This file validates the raw master 2023 station-OD anchor, not yet the historical extrapolation.
- The observed part is the NPVM 2023 zone-to-zone PT matrix itself.
- The hybrid part is the zone-to-station allocation layer, which is fully written out in the allocation CSV for auditability.
- The top production and attraction rankings above are based on the exported off-diagonal station matrix; same-station demand is reported separately through the diagonal total.
- Because this master anchor keeps the full modeled station system, strong metropolitan pairs remain visible here by design. The later website weighting layer can derive an inter-agglomeration analysis view from this master matrix without altering the raw anchor.
- Update: those later steps have now been implemented. The accepted 2023 inter-agglomeration layer has been derived, corridor validation has been recorded, and the resulting base year is now the observed/hybrid anchor behind the live all-years `All_OD_Pairings.csv` series.
