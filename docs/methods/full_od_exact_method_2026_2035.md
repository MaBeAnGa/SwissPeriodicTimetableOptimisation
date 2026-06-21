# Full OD Builder Method (2026/2035)

- Scope: builds full-station OD skims for frequency, fastest observed in-vehicle travel time, average pre-departure wait, and average rolling/dwell/transfer time components.
- Time window assumptions: pattern search from 05:00 to 07:00 (n+1 query strategy), daily interpretation on 05:00-00:00.
- Route selection rule: identical generalized-cost ordering as notebook `earliest_arrival_query` (arrival time, then later first departure, then fewer transfers, then fewer movement segments).
- Anchor cache rule: reduced-anchor values are used only for anchor->anchor pairs, and only after direct-vs-cache validation.
- One-to-many row acceleration (optional, off by default): destination requests are grouped by origin departure-minute seed (`seed_min`) rather than raw query minute, and one generalized-cost search is run per seed bucket.
- Exactness guard: one-to-many mode is enabled only if sampled OD pairs match the direct per-destination method within numerical tolerance (1e-9) for sampled departure records.
- Fallback behavior: any validation failure disables the shortcut and reverts to direct per-destination queries. Runtime exceptions in one-to-many also fallback per pair to direct querying and are counted.
- Resume idempotency: on startup and per-row execution, already-complete rows are detected from frequency memmap completeness and skipped (no recomputation).
- Row-intent locking (optional): per-row non-blocking advisory locks can be enabled so overlapping workers cannot start the same origin row concurrently.
- Interleaved worker slicing (optional): `--row-modulo M --row-remainder R` restricts a worker to rows where `row_index % M == R`, enabling exact disjoint row assignment such as every fourth row.
- Zero-minute handling: rolling and dwell 0-minute segments are rebalanced with a 20-second minimum while preserving exact end-to-end travel time.

## Output

- `<YEAR>_OD_matrix_Frequency_full*.csv`
- `<YEAR>_OD_matrix_Fastest_observed_connection_full*.csv`
- `<YEAR>_OD_matrix_Wait_time_full*.csv`
- `<YEAR>_OD_matrix_Rolling_time_full*.csv`
- `<YEAR>_OD_matrix_Dwell_time_full*.csv`
- `<YEAR>_OD_matrix_Transfer_time_full*.csv`

## Logging counters

- `pairs_direct_query`: destination pairs solved via direct per-destination querying.
- `pairs_batched_query`: destination pairs solved through one-to-many row batches.
- `pairs_one_to_many_fallback_direct`: destination pairs that attempted one-to-many but were recomputed with direct query due to runtime fallback.
- `pairs_anchor_cache`: destination pairs solved from reduced-anchor cache.
- `one_to_many_batch_runs`: number of unique origin+query-minute one-to-many searches executed.
- `rows_skipped_precomputed`: rows detected as already complete in memmap and therefore not recomputed.
- `rows_skipped_locked`: rows deferred because a row-intent lock was held by another worker at runtime.
