# NPVM Rail-Load Validation 2023

## Scope

- Reference year: `2023`
- OD anchor: `station_od_master_reference_2023.csv`
- Assignment model: exact `best_of_day()` routes on the thesis 2023 timetable model
- Selected OD demand routed exactly: `4,550` directed pairs
- Selected demand share routed exactly: `95.00%`
- Assigned OD weight on successful routes: `2,235,030.7` daily trips
- No-route weight inside the selected set: `0.0` daily trips
- Screenline set: `15` named national corridors

## Why this validation is structured this way

- The official NPVM rail-load layer (`BEL_DWV`) is link-based, while the thesis model is station-based.
- So the comparison is done on **screenlines across named corridors**, not on raw one-to-one link IDs.
- The left-hand network map uses the thesis model's assigned station-to-station movement segments.
- The right-hand map uses the official NPVM 2023 Swiss rail-load network directly.
- This is therefore a **corridor-pattern validation**, which is the scientifically meaningful comparison for a stationized historical rail model.

## Figures

- Network comparison map: `npvm_rail_network_maps_2023.png`
- Screenline scatter: `npvm_rail_screenline_scatter_2023.png`

## Raw fit on the selected corridor set

- Corridor count: `15`
- Pearson `r`: `0.784` (r² = 0.615)
- Mean absolute percentage error: `32.2%`
- Median absolute percentage error: `16.9%`
- Corridors within ±20%: `9` / `15`
- Corridors within ±30%: `11` / `15`
- Corridors within ±50%: `13` / `15`

## Anchor-scaled diagnostic

- Anchor corridors: `Olten-Bern, Arth-Goldau-Bellinzona`
- Fitted multiplicative scale factor: `1.1962`
- Out-of-anchor corridor count: `13`
- Out-of-anchor Pearson `r`: `0.792` (r² = 0.628)
- Out-of-anchor mean absolute percentage error: `56.1%`
- Out-of-anchor median absolute percentage error: `39.0%`
- Out-of-anchor corridors within ±30%: `5` / `13`


## Corridor table

```text
        corridor_label       role predicted_daily_trips official_bel_dwv abs_pct_error scaled_predicted_daily_trips scaled_abs_pct_error
  Zürich HB-Winterthur validation                96,814           99,081          2.3%                      115,807                16.9%
       Aarau-Zürich HB validation               108,258           81,352         33.1%                      129,497                59.2%
            Olten-Bern     anchor                54,421           65,451         16.9%                       65,098                 0.5%
     Basel SBB-Liestal validation               156,104           53,455        192.0%                      186,729               249.3%
         Liestal-Olten validation                31,336           36,493         14.1%                       37,483                 2.7%
 Winterthur-St. Gallen validation                41,805           35,978         16.2%                       50,006                39.0%
             Bern-Thun validation                44,706           35,759         25.0%                       53,476                49.5%
            Luzern-Zug validation                27,669           23,898         15.8%                       33,098                38.5%
       Zug-Arth-Goldau validation                20,372           20,316          0.3%                       24,369                19.9%
Bern-Fribourg/Freiburg validation                23,228           19,815         17.2%                       27,785                40.2%
            Thun-Spiez validation                28,683           16,771         71.0%                       34,310               104.6%
     Bellinzona-Lugano validation                18,259           12,501         46.1%                       21,841                74.7%
        Lugano-Chiasso validation                11,681           11,351          2.9%                       13,972                23.1%
 Biel/Bienne-Neuchâtel validation                 7,845           10,585         25.9%                        9,384                11.3%
Arth-Goldau-Bellinzona     anchor                 8,934            8,535          4.7%                       10,687                25.2%
```

## Interpretation

- The **raw, unscaled** fit is the correct result to carry forward. It is materially stronger than the anchor-scaled diagnostic, so the current 2023 OD anchor should not be globally rescaled before the historical-evolution stage.
- For a stationized national rail model, the raw corridor agreement is reasonably strong: `r² = 0.615`, median absolute corridor error `16.9%`, and `11 / 15` named corridors inside `±30%`.
- The remaining mismatch is concentrated in corridors where the thesis station system still compresses dense approach structure or branch structure more aggressively than the full NPVM rail network.

### Best-matched raw corridors

- `Zug-Arth-Goldau`: `0.3%`
- `Zürich HB-Winterthur`: `2.3%`
- `Lugano-Chiasso`: `2.9%`
- `Arth-Goldau-Bellinzona`: `4.7%`
- `Liestal-Olten`: `14.1%`

### Largest raw corridor mismatches

- `Basel SBB-Liestal`: `192.0%`
- `Thun-Spiez`: `71.0%`
- `Bellinzona-Lugano`: `46.1%`
- `Aarau-Zürich HB`: `33.1%`
- `Biel/Bienne-Neuchâtel`: `25.9%`

## Assignment coverage note

- The validation routed the demand-heavy OD pairs first and stopped at `95.00%` of assignable off-diagonal daily demand.
- This keeps the validation exact where almost all mass sits, while making the run tractable and fully auditable.
- The remaining low-weight tail can be added later by resuming the same route cache rather than starting from scratch.
- The current 2023 timetable model can route about `79.7%` of the raw master-station OD weight directly. The remainder belongs to stations that are not active 2023 timetable nodes in the present modeled station system, so those OD rows are documented but not forced into an artificial assignment.
