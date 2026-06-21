# Transfer-Time Optimization Method: retained 2026/2035 pipeline

Updated: 2026-05-13 CEST

## Purpose

This document describes only the retained workflow used for the final 2026/2035 baseline and `OPTIMIZED_FILTERED` timetable comparison. The goal is to identify small integer-minute retimings of timetable sections that reduce passenger-weighted generalized in-vehicle plus dwell plus transfer time while preserving rolling time and all minimum-transfer constraints.

The candidate universe is restricted to high-flow transfer patterns in the transfer-discounted transfer summaries:

- `Identified_All_Transfers_2026_transfer_discounted.csv`
- `Identified_All_Transfers_2035_transfer_discounted.csv`

For year \(y\), OD pair \((o,d)\), and retained nondominated path \(p\in\mathcal{P}_{od}^{y}\), let

\[
x_{odp}^{y}=D_{od}^{y}\alpha_{odp}^{y}
\]

be the daily passenger flow assigned to that path, where \(D_{od}^{y}\) is the final transfer-discounted OD demand and \(\alpha_{odp}^{y}\) is the path-allocation share derived from dominance minutes. The timetable-retiming problem is then evaluated on passenger-minutes:

\[
F^{y}(T)=\sum_{(o,d)}\sum_{p\in\mathcal{P}_{od}^{y}}x_{odp}^{y}
\left(r_{odp}^{y}(T)+h_{odp}^{y}(T)+q_{odp}^{y}(T)\right),
\]

where \(r\), \(h\), and \(q\) are rolling, dwell, and transfer minutes respectively. Retiming candidates are accepted only if they preserve the timetable's rolling-time structure; improvements must come from transfer and dwell timing interactions, not from faster trains.

## Arrival/Departure Board Extraction

The script `prepare_transfer_boards_and_baseline.py` reads `FullTimetableHistory2026_2035.csv` directly and uses the same periodic token conventions as the OD routing model:

- plain minute values are hourly,
- `n` means even-hour service,
- `i` means odd-hour service,
- `P` and `Q` preserve the later-hour chaining constraints,
- rolling time is not altered,
- 2026 uses `2026`, `2026 (HH)`, and `2026 alt`,
- 2035 uses `2035`, `2035 (HH)`, and `2035 alt`.

For every transfer station appearing in a high-flow transfer pattern, the script writes board rows between 10:00 and 11:59. This window is chosen to expose both even-hour and odd-hour service patterns. Each board row contains the transfer station, board type, line, previous station, next station, event time, and passenger-weight annotations.

The passenger-weight columns have two meanings:

- `high_flow_transfer_daily_pax_pattern_total_touching_event` is the sum of full daily passenger volumes for high-flow transfer patterns whose repeated timing touches that board event.
- `high_flow_transfer_daily_pax_time_share_touching_event` divides each pattern’s daily passengers across its repeated daily occurrences, giving an approximate daily-passenger share attached to that specific clock event.

Representative morning transfer-summary times are projected into the requested board window with the same 120-minute periodicity used by path allocation. This avoids treating the transfer summary as a literal-only morning extract.

## Baseline Transfer-Share Metric

The same script can stream the two large transfer-discounted path-allocation CSVs to calculate the baseline objective:

\[
\Theta_q^{y}(T)=
\frac{
\sum_{(o,d)}\sum_{p\in\mathcal{P}_{od}^{y}}
x_{odp}^{y}\,
\frac{q_{odp}^{y}(T)}
{r_{odp}^{y}(T)+h_{odp}^{y}(T)+q_{odp}^{y}(T)}
}{
\sum_{(o,d)}\sum_{p\in\mathcal{P}_{od}^{y}}x_{odp}^{y}
}.
\]

Analogous weighted shares are calculated for dwell time and rolling time:

\[
\Theta_h^{y}(T)=
\frac{
\sum_{od,p}x_{odp}^{y}\,
\frac{h_{odp}^{y}(T)}{r_{odp}^{y}(T)+h_{odp}^{y}(T)+q_{odp}^{y}(T)}
}{\sum_{od,p}x_{odp}^{y}},
\qquad
\Theta_r^{y}(T)=
\frac{
\sum_{od,p}x_{odp}^{y}\,
\frac{r_{odp}^{y}(T)}{r_{odp}^{y}(T)+h_{odp}^{y}(T)+q_{odp}^{y}(T)}
}{\sum_{od,p}x_{odp}^{y}}.
\]

Two scopes are written:

- all allocated OD paths,
- only high-flow transfer events with at least 95 daily passengers.

For high-flow events, each transfer event is weighted by the path-level `daily_demand_allocated`, and the denominator is the complete path time (`transfer_min + dwell_min + rolling_min`). This baseline is retained as the comparison target for tiny departure/arrival shifts.

The scan is streaming and resumable through JSON state files under `/tmp/transfer_optimization_baseline_state`, so the 130 GB+ path-allocation files are not loaded into memory.

## Optimization Setup Tables

Updated: 2026-04-28 10:04 CEST

The script `setup_transfer_optimization_tables.py` prepares four compact tables under `transfer_optimization_tables/`:

- `Transfer_Optimization_Event_Ledger_2026_2035.csv` lists every repeated full-day arrival/departure event at transfer stations with at least one high-flow transfer pattern. It includes station-specific minimum transfer time, line, previous station, next station, event time, and high-flow passenger annotations.
- `Transfer_Optimization_Near_Misses_2026_2035.csv` lists arrival-departure pairs around the station-specific minimum transfer time. By default it includes pairs from 5 minutes before feasibility to 10 minutes after feasibility. This table is meant to identify potentially unlockable transfers and fragile already-feasible transfers.
- `Transfer_Optimization_Shift_Feasibility_2026_2035.csv` estimates the feasible earlier/later shift range for each event by treating retiming as a whole-section offset. Rolling time remains unchanged. Earlier shifts are limited by same-train dwell before the section departure; later shifts are limited by same-train dwell after the section arrival. Terminal or section-boundary cases are flagged as `section_start_unbounded_by_same_train` or `section_end_unbounded_by_same_train` because they need later operational review.
- `Transfer_Optimization_Candidates_2026_2035.csv` aggregates screening candidates by year, action, line, delta, section, and 120-minute pattern. The score is a proxy passenger-minute value, not a final decision metric. It is used only to rank candidates for the later station-board strict proposal generator.

The candidate proxy combines two retained sources:

- existing high-flow transfers that could have their transfer time shortened without falling below the station minimum transfer time;
- currently infeasible near-miss transfers that could become feasible with a small shift.

For a transfer event \(a\) at station \(s\), define the observed wait

\[
g_a=t_a^{+}-t_a^{-},
\]

where \(t_a^{-}\) is the arrival event time and \(t_a^{+}\) is the departure event time. A transfer is feasible only if

\[
g_a\ge m_s,
\]

with \(m_s\) taken from `FilteredStations2026_2035.csv` except for the Zürich HB complex rule documented below.

## Net-Impact Screening Layer

Updated: 2026-04-28 10:22 CEST

The script `build_transfer_shift_net_impact.py` adds a conservative net-impact screen on top of the gross candidate table. It does not edit the timetable and does not read the huge path-allocation files. It reads only:

- `Transfer_Optimization_Candidates_2026_2035.csv`,
- `Transfer_Optimization_Event_Ledger_2026_2035.csv`,
- `Transfer_Optimization_Near_Misses_2026_2035.csv`,
- `Identified_All_Transfers_2026_transfer_discounted.csv`,
- `Identified_All_Transfers_2035_transfer_discounted.csv`.

For each proposed line-section shift, it evaluates both endpoints of the shifted section:

- at the section departure station, shifting the departure changes transfer wait by `+delta`;
- at the section arrival station, shifting the arrival changes transfer wait by `-delta`;
- rolling time is held constant because the whole section is shifted by an integer number of minutes.

Existing high-flow transfers are scored with their projected daily occurrence share. If the shifted wait remains above the station minimum, the passenger-minute change is counted as a gain or loss. If the shifted wait falls below the station minimum transfer time, the affected passenger flow is flagged separately as a broken existing high-flow transfer.

Currently infeasible near-miss transfers are scored only as a screening proxy. Their gain equals the passenger proxy multiplied by the amount of transfer shortfall closed by the shift:

\[
G_a(z)=X_a\left[
\max(0,m_s-g_a)-\max(0,m_s-g_a'(z))
\right],
\]

where \(X_a\) is the daily passenger flow attached to the repeated transfer pattern and \(z\) is the candidate section shift.

The main decision table is `Transfer_Optimization_Net_Impact_Physical_Shifts_2026_2035.csv`. It deduplicates the row-level candidate table by actual physical move: year, line, section, shift delta, and 120-minute timing pattern. This prevents the same physical move from appearing twice when both the arrival-side and departure-side candidate logic identify it.

The row-level table `Transfer_Optimization_Net_Impact_Candidates_2026_2035.csv` is kept for auditability. `Transfer_Optimization_Net_Impact_Details_Top.csv` stores the largest component effects for the highest-ranked shifts.

This layer is not by itself a final proof. It is retained as a compact scoring layer that feeds the station-board strict proposal generator described below.

## Minimum-Transfer Routing Correction

Updated: 2026-05-01 18:06 CEST

The all-transfers CSV exposed an upstream routing error rather than a reporting error. Some paths contained transfers below the station-level `MinimumTransferTime`, for example a 1-minute `A / S 8 -> Q / S 15` transfer at `Zürich Oerlikon`. The station metadata was loaded correctly, but the time-expanded graph had a loophole: after arriving at a station, the route could traverse the same-train dwell edge to that train's departure node, then use departure-to-departure `wait` edges to jump to another train before the station minimum transfer time had elapsed.

`build_od_nondominated_paths_cells_2026_2035.py` now protects both route-generation paths:

- The direct cell query and the full-row one-to-many accelerator use the same safe Dijkstra expansion.
- Old prebuilt notebook `transfer` edges are ignored during this safe expansion; transfer edges are regenerated dynamically from arrival nodes using `FilteredStations2026_2035.MinimumTransferTime`.
- Departure-to-departure `wait` edges are still allowed for origin waiting and after a valid transfer, but they can no longer be used as a hidden transfer immediately after a same-train dwell edge unless the previous arrival time already satisfies the relevant minimum transfer.
- `Zürich HB` is treated as one transfer complex for routing: 7 minutes across the complex, except same-alias transfers within `Zürich HB (31-34)`, which use 4 minutes.
- Public OD origins at `Zürich HB` can board departures from any Zürich HB alias. Public OD destinations at `Zürich HB` can arrive at any Zürich HB alias and are still emitted as `Zürich HB` in the JSON terminal payload.

The Zürich HB minimum-transfer rule is therefore

\[
m(a,b)=
\begin{cases}
4, & a=b=\text{Zürich HB (31-34)},\\
7, & a,b\in \mathcal{Z}_{HB}\ \text{and otherwise},\\
m_s, & \text{all other stations},
\end{cases}
\]

where \(\mathcal{Z}_{HB}\) is the set of Zürich HB public/alias nodes and \(m_s\) is the station metadata value.

The new `validate_min_transfer_routing_smoke.py` script checks targeted 2026 and 2035 examples before any large rerun. It covers the former Oerlikon failure class, Zürich HB terminal cases, and representative Zürich HB transfers. A one-row 2026 row-builder smoke test for origin `Aadorf` checked 13,294 generated transfer events and found zero below-minimum transfer waits.

## Station-Board Strict Proposal Generation

Updated: 2026-05-08 09:35 CEST

The retained proposal generator is `generate_missed_transfer_station_board_proposals.py`. It starts from the validated v0 transfer tables and searches for physical section shifts that touch at least one missed-transfer-unlocking endpoint. Because a physical section shift preserves rolling time by moving both the section departure and section arrival by the same integer minute delta, the opposite endpoint side-effects are scored too. For example, moving a section earlier can improve onward transfers at the downstream station while worsening feeder transfers at the upstream station.

A candidate section shift is represented as

\[
z=(y,\ell,u,v,\delta),
\]

where \(y\) is the year, \(\ell\) is the line, \(u\rightarrow v\) is the directed section, and \(\delta\in\mathbb{Z}\) is the minute offset. The timetable edit is

\[
\tau'_{\ell,u}^{\mathrm{dep}}=\tau_{\ell,u}^{\mathrm{dep}}+\delta,\qquad
\tau'_{\ell,v}^{\mathrm{arr}}=\tau_{\ell,v}^{\mathrm{arr}}+\delta.
\]

Thus rolling time is invariant:

\[
\left(\tau'_{\ell,v}^{\mathrm{arr}}-\tau'_{\ell,u}^{\mathrm{dep}}\right)
-
\left(\tau_{\ell,v}^{\mathrm{arr}}-\tau_{\ell,u}^{\mathrm{dep}}\right)=0.
\]

The local transfer wait after already-accepted edits is written

\[
g_a'(z)=g_a+\Delta_a^{\mathrm{dep}}(z)-\Delta_a^{\mathrm{arr}}(z),
\]

where \(\Delta_a^{\mathrm{dep}}\) and \(\Delta_a^{\mathrm{arr}}\) are the cumulative shifts applied to the departure and arrival events of transfer \(a\).

The selector remains cumulative:

- accepted edits are immediately applied to a working timetable;
- each later candidate is scored against all previously accepted event shifts;
- no candidate is accepted if it breaks an existing high-flow transfer below station minimum transfer time;
- the touched timetable segment must preserve trip count, station/kind sequence, rolling durations, and reasonable dwell bounds;
- the output includes both marginal-at-acceptance fields and final-state touched-effect fields after all later compatible edits have also been accepted.

Formally, after \(k\) accepted edits \(Z_k=\{z_1,\ldots,z_k\}\), the next retained edit is selected from the feasible candidate set \(\mathcal{C}(T_k)\):

\[
z_{k+1}=\arg\min_{z\in\mathcal{C}(T_k)}
\widehat{\Delta F}(z\mid T_k),
\]

subject to

\[
\widehat{\Delta F}(z\mid T_k)<0,\qquad
B(z\mid T_k)=0.
\]

\(\widehat{\Delta F}\) is the station-board passenger-minute proxy and \(B\) is the passenger flow of existing high-flow transfers that would be broken below minimum transfer time. If no feasible candidate satisfies both conditions, selection stops.

The new proposal table also stores 10:00-11:59 station-board examples for each accepted/audited candidate. These examples are intended to make the clockface consequences inspectable: unlocked near-misses, worsened existing connections, unchanged effects, and pax-minute deltas are printed directly in the row.

Update: the missed-transfer proposal CSV now also carries `Arrival Line Timings` and `Departure Line Timings`, generated with the same line-profile/timing code used for `Transfers_YEAR_v0.csv`. These columns are built from the current in-memory working timetable. After all accepted edits are chosen, the accepted proposal rows are refreshed once more against the final accepted timetable, so earlier rows show later accepted alterations on the same line. The timing formatter expands displayed runs only for departures in the 05:00-06:59 window; if a `:00` token is shifted to `:59`, the displayed run is placed at `05:59`/`06:59` rather than as a misleading `04:59` departure.

This still does not replace the full OD proof. The proposal stage uses fixed daily-pax weights previously derived from the full v0 nondominated-path and path-allocation CSVs. It therefore measures effects on known transfer flows and near-miss opportunities, but it does not reroute every OD pair under the changed timetable. The retained proof step is: apply the accepted proposal timetable, rebuild the nondominated-path matrix, reattach demand and path allocation, regenerate transfer tables, and compare passenger-weighted total time and component shares against v0.

## Full OD Audit Build

Updated: 2026-05-08 10:05 CEST

`build_optimized_timetable_from_station_board_proposals.py` applies the accepted rows from `transfer_optimization_tables/missed_transfer_station_board_v0/ProposedTimetableChanges_MissedTransfers_StationBoardStrict.csv` to `FullTimetableHistory2026_2035.csv` and writes `FullTimetableHistory2026_2035_OPTIMIZED.csv`.

The application step intentionally reuses the exact cumulative-safe timetable-edit function used during proposal generation. This means the audit timetable is not created by a second looser rewrite path: each row must still match the current timetable token state, preserve rolling-time segment structure, preserve suffix semantics (`i`, `n`, `P`, `Q`), and pass the line-segment compatibility validator. The audit build applied all 31 station-board proposal rows without skips: 9 rows for 2026 and 22 rows for 2035.

`run_optimized_missed_transfer_full_sequence.sh` is the end-to-end validation runner. It:

- rebuilds `FullTimetableHistory2026_2035_OPTIMIZED.csv` from the accepted proposal CSV;
- builds 2026 and 2035 nondominated path row-state JSONs with `OD_TIMETABLE_CSV` pointed to the optimized timetable, `station-scope=year-specific`, and the same minimum-transfer/Zürich-HB routing behavior as v0;
- streams row-state JSONs directly through `attach_and_allocate_demand_from_row_state.py`, avoiding a second raw nondominated CSV intermediate;
- writes the directly comparable large path-allocated files `2026_OPTIMIZED_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv` and `2035_OPTIMIZED_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv`;
- compares those files against the v0 path-allocated files using passenger-weighted OD-cell component metrics for rolling time, transfer time, dwell time, and rolling share.

The comparison script, `compare_pathalloc_component_stats.py`, streams both CSVs row-by-row and asserts identical station order before computing deltas. Metrics are weighted by allocated daily passengers. Therefore the final comparison isolates the timetable-change effect under the same station scope, same demand matrices, same path-allocation method, and same routing/minimum-transfer rules as v0.

## OPTIMIZED_FILTERED Conservative Validation Pipeline

Updated: 2026-05-09 16:25 CEST

After the full `_OPTIMIZED` rebuild, `audit_optimized_timetable_adjustment_effects.py` attributed final-state OD-cell changes back to the 31 accepted station-board timetable edits. This audit is not a full one-edit-at-a-time counterfactual rebuild; it is a conservative final-state attribution check using enriched v0 and optimized row-state JSONs. It is still useful for removing edits that visibly worsened the final timetable outcome on the cells they touched.

`filter_optimized_proposals_from_audit.py` creates `ProposedTimetableChanges_MissedTransfers_StationBoardStrict_FILTERED.csv` from the final OD audit. The retained final set is the 18-row conservative set that improved the aggregate passenger-minute objective after the full OD rebuild: 7 rows for 2026 and 11 rows for 2035. Rolling-share labels are reported as diagnostics, but the retained thesis objective is total passenger-minutes:

\[
z\in Z_{\mathrm{final}}
\quad\Longrightarrow\quad
\Delta F_z<0
\]

for the attributed OD cells touched by proposal \(z\), followed by a full-network comparison of \(F^y(T_{\mathrm{OPTIMIZED\_FILTERED}})\) against \(F^y(T_{v0})\).

`FullTimetableHistory2026_2035_OPTIMIZED_FILTERED.csv` is built from the original v0 timetable, not from the already-optimized timetable. The filtered rows are reapplied through the same cumulative-safe token-shift and segment-compatibility code used for the original proposal selection and the `_OPTIMIZED` build. The filtered timetable build is therefore directly comparable to v0: the routing model, minimum-transfer constraints, station scope, demand matrices, and path-allocation logic are unchanged, and only the accepted timing tokens differ.

The formal mathematical formulation of this optimization target, the binary candidate-selection view, and the full OD audit objective is documented in `METHODS_timetable_optimization_workflow_2026_2035.md`, section 9.

`run_optimized_filtered_full_sequence.sh` is the end-to-end runner for this cleaner version. It:

- regenerates the filtered proposal CSV from the audit table;
- rebuilds `FullTimetableHistory2026_2035_OPTIMIZED_FILTERED.csv` from the original timetable;
- rebuilds 2026 and 2035 nondominated path row-state JSONs with the same year-specific station scope and minimum-transfer/Zürich-HB behavior as v0;
- attaches the existing transfer-discounted demand matrices and path allocation directly from row-state;
- writes the directly comparable `_OPTIMIZED_FILTERED` large path-allocated CSVs;
- compares those CSVs against the v0 path-allocated CSVs using the same passenger-weighted rolling, transfer, dwell, total-time, and rolling-share metrics.
