# Website OD Pair Weight Validation 2023

## Scope

- Reference year: `2023`
- Weight model: `npvm_2023_observed_hybrid_selected_station_projection_with_same_agglomeration_distance_attenuation`
- Raw master station system: `250` stations
- Website analysis station system: `131` stations
- Distinct analysis clusters represented on the website subset: `91`

## Pair Table Diagnostics

- Same-agglomeration treatment: `attenuate`
- Same-agglomeration attenuation distance scale: `60.0 km`
- Active directed pairs in the website metric: `17030`
- Explicit zero/excluded directed pairs: `0`
- Active same-agglomeration directed pairs retained in the metric: `190`
- Sum of projected active pair weights: `954177.730`
- Raw selected-station inter-agglomeration trips before projection: `842225.431`
- Raw selected-station same-agglomeration trips before attenuation/exclusion: `754128.088`
- Attenuated same-agglomeration trips retained in the overall metric: `111952.299`
- Same-agglomeration share of the raw selected-station trips before attenuation: `47.24%`
- Retained same-agglomeration share of the total selected raw trips after attenuation: `7.01%`
- Distinct inter-cluster directions in the master anchor: `43890`
- Distinct clusters with excluded same-cluster demand: `19`
- Active directed pairs touching at least one foreign station: `260`
- Excluded directed pairs touching at least one foreign station: `0`
- Active weight sum touching at least one foreign station: `5545.174`
- Foreign-station share of the active website weight sum: `0.58%`

## Redistribution Diagnostics

- Origin-share methods used: `{'observed_intercluster_origin_share_on_selected_subset': 131}`
- Destination-share methods used: `{'observed_intercluster_destination_share_on_selected_subset': 131}`
- Cluster scopes touched by active pairs: `{'official_bfs_aggl2020': 25620, 'station_singleton_outside_agglomeration': 7800, 'foreign_station_singleton': 260}`

## Interpretation

- The website's overall historical analysis no longer reads the old gravity-proxy table as its primary source.
- Instead, it now reads an analysis-ready 2023 table derived from the raw observed/hybrid master station anchor.
- Same-agglomeration pairs are retained on the selected-station layer, but only with a calibrated linear distance attenuation applied to their raw observed selected-station demand.
- Inter-agglomeration cluster flows come from the full master station matrix, while their redistribution to the visible website nodes is normalized on the selected 131-station subset.
- This means the website layer is now a projection of the observed/hybrid master demand onto the visible analysis nodes, not a synthetic gravity estimate.
- The gravity formula remains available only as a safety-net fallback for genuinely missing rows outside this canonical table.
