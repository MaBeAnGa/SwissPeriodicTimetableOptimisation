# Transfers v0 Context Tables

Created: 2026-05-02 CEST

This note documents the generation logic for `Transfers_2026_v0.csv` and
`Transfers_2035_v0.csv`. These files extend the earlier
`Identified_All_Transfers_<YEAR>_v0_transfer_discounted.csv` tables with
line-direction context needed for later timetable-optimization work.

## Input files

- `2026_v0_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv`
- `2035_v0_OD_matrix_Nondominated_paths_full_transfer_discounted_with_daily_demand_and_path_alloc.csv`
- `FullTimetableHistory2026_2035.csv`

The large OD CSVs are streamed and are never loaded into memory. The path-level
field `daily_demand_allocated` is used as the passenger weight, so the table
inherits the transfer-discounted v0 demand model and the departure-dominance
path allocation already attached to the cells.

## Output row definition

Each row represents one aggregate line-to-line transfer pattern:

`(Transfer station, Arrival Line, Transfer Time, Departure Line)`

The aggregation retains all observed arrival/departure clock times and sums the
allocated daily passenger demand over all OD paths using that transfer.

`ID` is assigned per year after aggregation, ordered by descending affected
daily passenger flow, then affected OD-pair count, then station/line names. IDs
are stable for a given completed input file and are reused by the prior/following
context columns.

## Added context columns

- `Arrival Line Origin Station`: the directional origin(s) of the train on the
  arrival line for paths contributing to this transfer row. If several line
  patterns contribute, multiple origins are listed with their contributing pax.
- `Arrival Line Prior IDs`: transfer-row IDs at prior stations on the same
  arrival line, upstream of the transfer station in the direction of travel.
  A station is included if it has any identified transfer row where that same
  line is either the arrival or departure line.
- `Arrival Line Prior Transfer Station Names`: the same upstream stations,
  weighted by daily pax who both use the current transfer row and pass that
  upstream station continuously on the current row's arrival line.
- `Arrival Line Prior Dwell Times`: dwell time of the arrival line at those
  upstream identified transfer stations. `TERMINUS` is used when the station is
  a terminus for that line direction or no through dwell exists.
- `Transfer Station Dwell Times`: dwell time of the arrival line and departure
  line at the transfer station of the current row. If either line terminates or
  originates there, that line is marked `TERMINUS`.
- `Departure Line Final Station`: the directional final station(s) of the
  departure line for paths contributing to this transfer row. If several line
  patterns contribute, multiple finals are listed with their contributing pax.
- `Departure Line Following IDs`: transfer-row IDs at downstream stations on
  the same departure line, after the transfer station in the direction of travel.
- `Departure Line Following Transfer Station Names`: downstream stations,
  weighted by daily pax who both use the current transfer row and pass that
  downstream station continuously on the current row's departure line.
- `Departure Line Upcoming Dwell Times`: dwell time of the departure line at
  those downstream identified transfer stations, with `TERMINUS` marking
  termini.

## Line-direction and terminus logic

For each contributing OD path, the script reads the path's actual adjacent
`legs` and each leg's `canonical_trip_id`. The `canonical_trip_id` gives the
ordered stopping pattern and dwell times for the train instance, but passenger
overlap is restricted to the contiguous same-train leg run actually used by the
OD path. This distinction is important: a passenger who transfers onto a train
at Zürich Oerlikon and leaves it at Zürich Flughafen is counted for the
`Zürich Oerlikon-Zürich Flughafen` overlap only, not for the rest of that
train's canonical route toward Winterthur or beyond.

Similarly, on the arrival side, prior-station overlap is restricted to the
portion of the arriving train that the passenger actually used before the
transfer station. Stations that are earlier in the train's full canonical route
but before the passenger boarded that train are not counted.

The timetable file is also parsed to identify line termini from the original
out-and-back line sequences:

- first and last station of a line block are termini;
- an interior station whose predecessor and successor are the same station is
  treated as a turnaround terminus;
- if three such interior turnarounds are detected, the middle one is selected
  as the principal doubled-back terminus;
- if no interior turnaround exists, the end station remains the terminus,
  covering ring-like patterns conservatively.

When a station is a terminus either in the observed canonical trip or in this
line-block terminus map, dwell time is reported as `TERMINUS` rather than as a
turnaround dwell.

## Passenger overlap interpretation

The prior/following station-name columns do not report the total transfer demand
at those other stations. They report the amount of daily passenger flow from the
current transfer row that also passes the named station on the same line and in
the same direction. This is why closer upstream/downstream transfer stations
should generally have equal or higher overlap than farther stations on the same
branch, subject to branching and different stopping patterns.

## Resumability

The generator uses a SQLite state database per year. It has two passes:

1. `aggregate`: identifies all transfer rows, times, OD-pair counts, and total
   affected daily pax.
2. `enrich`: after all IDs are known, streams the OD CSV again to compute
   prior/following overlap, terminal stations, and dwell context.

If interrupted, rerunning the shell script resumes from the saved row counters.
Use `FRESH=1` only when deliberately discarding all prior state. Use
`FRESH_ENRICH=1` to keep the already aggregated transfer rows/IDs but clear and
recompute only the prior/following context columns.
