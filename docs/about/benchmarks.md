# Benchmarks

This page shows what specsolve costs to build a model and hand it to a solver,
beside four other ways of writing the same model, and how far to trust each
number.

Each model is built and handed to a solver. `run()` and `optimize()` are never
called, so no number includes solve time. Every rival model is hand-written in
[`bench/models/`](https://github.com/fluxopt/specsolve/blob/main/bench/models)
beside ours, and CI solves each against ours, so nothing here is fast because it
built a different model.

dispatch
:   Meets a demand from a fleet of units: bounds and one balance. It is the floor.

transport
:   A network of buses and lines: mapping tables, three joins per row.

storage
:   Dispatch plus a state of charge that carries from each snapshot to the next:
    a cyclic recurrence over time.

fleet
:   Twelve declarations rather than one large one: cost per declaration, not per row.

<div id="benchmark-charts" class="bench">
<div class="bench-controls" role="group" aria-label="Controls for the charts and the table">
<div class="bench-row">
<span class="bench-label">Show</span>
<span class="bench-seg" data-key="metric"><button data-value="wall" aria-pressed="true">wall time</button><button data-value="peak" aria-pressed="false">peak RSS</button></span>
<span class="bench-label">Grown by</span>
<span class="bench-seg" data-key="ladder"><button data-value="length" aria-pressed="true">length — more snapshots</button><button data-value="width" aria-pressed="false">width — more entities</button></span>
<span class="bench-label">Axes</span>
<span class="bench-seg" data-key="scale"><button data-value="log" aria-pressed="true">log–log</button><button data-value="linear" aria-pressed="false">linear–linear</button></span>
</div>
<div class="bench-row">
<span class="bench-label">Libraries</span>
<div id="bench-legend"></div>
<span class="bench-hint">click to hide</span>
</div>
</div>
<div id="bench-facets"></div>
<div id="bench-scale" class="bench-scale"></div>
<div class="bench-table"><table id="bench-heat" class="bench-heat"></table></div>
<script type="application/json" id="bench-spec">
{
  "data": "../benchmarks.json",
  "facet": ["model", "sink"],
  "filter": { "ladder": "length" },
  "x": { "field": "variables", "label": "variables", "format": "count" },
  "metrics": {
    "wall": { "label": "wall time", "y": "wall_s", "band": ["wall_q1_s", "wall_q3_s"], "format": "seconds" },
    "peak": { "label": "peak RSS", "y": "peak_gb", "format": "gigabytes" }
  },
  "series": {
    "field": "library",
    "refused": "refused",
    "highlight": "specsolve",
    "floor": { "label": "matrix floor", "members": ["gurobipy-matrix", "highspy-matrix"] },
    "domain": ["specsolve", "specsolve-in-memory", "linopy", "pyomo", "gurobipy-loop", "gurobipy-matrix", "highspy-matrix"],
    "ink": ["var(--s1)", "var(--s1)", "var(--s3)", "var(--s2)", "var(--s4)", "var(--floor)", "var(--floor)"],
    "paint": ["var(--s1)", "var(--p1)", "var(--p3)", "var(--p2)", "var(--p4)", "var(--floor)", "var(--floor)"]
  }
}
</script>
</div>

## Reading the charts

Each panel is one model through one solver, and each line is one library's cost
against the size of the model.

- **specsolve is the thick violet line.** The other libraries are thinner, and
  the legend hides any of them.
- **The grey line is the matrix floor.** It is a hand-written matrix handed
  straight to the solver's bulk API: `gurobipy-matrix` on Gurobi,
  `highspy-matrix` on HiGHS. Nobody writes a model this way. It is the limit a
  modelling library can approach, so its distance from specsolve is what
  specsolve's modelling layer costs.
- **The pale violet line is specsolve on polars' in-memory engine.** The
  specsolve line leaves the engine to specsolve, which builds a small model in
  memory and a large one on polars' default engine. The
  [how-to for small models](../howto/small-models.md) says where the two part.
- **The line is the median of the rounds, and the band is the middle half of
  them.** The band runs from the first quartile to the third. Two lines whose
  bands overlap are two numbers this run cannot tell apart.
- **Grown by length adds snapshots; grown by width adds entities.** Both reach
  the same sizes, so switching holds the size and changes only the shape that
  got there. `transport` is the model where this matters: its bus-by-generator
  join grows with width and never with length. No case on this run reached a
  width rung, so the width view is empty for the reason
  [below](#not-measured-yet).
- **Log–log gives every size equal room, and the slope is the scaling order.**
  A line that rises one decade per decade grows in step with the model.
  Linear–linear draws the gap at the largest size at its real size and puts the
  small sizes at the origin. Both axes switch together, because a linear axis
  over a logarithmic one draws linear growth as a curve.
- **A dashed line is a projection, not a measurement.** It continues a library
  past the size its budget refused, at the growth rate of its last measured
  step and never slower than linear. The axes are scaled to the measurements,
  so a steep projection leaves the top of the panel, and the tooltip gives its
  value.
- **The table divides by specsolve.** The specsolve column is its own cost.
  Every other cell is that library's cost divided by specsolve's at the same
  size: green where specsolve is faster, blue where the other library is, grey
  within ten per cent. The matrix floor's column is not shaded, because a floor
  is meant to be faster. Hover a cell for the library's own number.
- **`>30 s` and `>16 GB` are sizes the harness refused.** It projected the
  measurement past its time or memory budget and skipped it. Hover one for the
  projection. An em dash is a cell with no number for another reason:
  `gurobipy` has no HiGHS.

## Why the median

Every published number is the median of nine rounds. Every measurement gets
the same nine, pinned rather than calibrated by duration, so no cell is a
best-of-nine beside a neighbour's best-of-forty. The median beats
the fastest round because a cell whose nine rounds all ran slow has no clean
round to pick. It beats the mean because one slow round moves a mean and
leaves a median where it was. On this run
([#1950](https://github.com/fluxopt/specsolve/pull/1950)) 31 of 160 published
cells have a mean above 1.10x their median, the worst at 3.04x: `dispatch/xs` on
pyomo, on the `highs` [sink](../reference/glossary.md#how-it-runs). In 23 of
them the first round is the slowest of the nine, at 1.5 to 19.4 times the median
of the other eight, so the median does not move. The other eight are
matrix-floor and gurobipy-loop cells at `xs` and `s` whose rounds fall into two
groups, the slow one about twice the fast one: `dispatch/xs` on gurobipy-loop
on the `gurobi` sink takes either 24 to 26 ms or 49 to 51 ms.

The median decides specsolve against another library in two cells on this run,
one each way, both against gurobipy-loop on the `gurobi` sink. At
`transport/xs` specsolve's fastest round is the faster one, 45.3 ms against
48.1 ms, and its median is the slower, 65.2 ms against 49.5 ms. At `storage/xs`
it is the other way round: 57.0 ms against 50.4 ms fastest, 63.6 ms against
74.4 ms median. No cell flips against linopy.

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py.lock` freezes every version, git commits included:
specsolve itself installs from git, at the commit the numbers were taken on.
`--locked` refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`bench/results`](https://github.com/fluxopt/specsolve/blob/main/bench/results):
one file per sink, case and ladder. Each carries the machine, the versions, the
commit and every round of every measurement. A ladder the box could not finish
leaves no file behind, and the other ladder of that case keeps its own.
`pixi run table` prints the directory as one long CSV and commits nothing; the
JSON stays the archive because it keeps the rounds.
`pixi run refresh` re-takes the numbers, writes the tables into their fences
and the chart rows into `benchmarks.json`.

## A script that solves one model

The tables above are what a process pays once it is running. A script that
imports a library and solves one model also pays the interpreter, the import,
and whatever the library loads the first time it builds. linopy's arrays are
xarray's, and xarray imports `dask.array` on the first array it makes, so
linopy's first model costs more than its import alone. This table times a new
process from launch to exit at `xs`, the rung where starting is a share of the
cost. A run that did not take it leaves the table out.

<!-- bench:fresh -->

### A process started for one solve

Launch to exit: a new interpreter imports the library, builds the model once and hands it to the sink. The tables above run warm and exclude the import, so the gap between this wall and the same rung there is what starting costs. The peak is the new process's own.

| case | sink | variables | wall: specsolve | wall: specsolve-in-memory | wall: linopy | wall: pyomo | wall: gurobipy-loop | wall: gurobipy-matrix | wall: highspy-matrix | peak: specsolve | peak: specsolve-in-memory | peak: linopy | peak: pyomo | peak: gurobipy-loop | peak: gurobipy-matrix | peak: highspy-matrix |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | gurobi | 10k | 0.55 s | 0.55 s | 0.98 s | 0.70 s | 0.25 s | 0.36 s | — | 4.98 GB | 4.98 GB | 4.98 GB | 4.98 GB | 4.98 GB | 4.98 GB | — |
| dispatch | highs | 10k | 0.46 s | 0.48 s | 0.96 s | 0.69 s | — | — | 0.33 s | 2.49 GB | 2.49 GB | 2.49 GB | 2.49 GB | — | — | 2.49 GB |
| fleet | gurobi | 12k | 0.62 s | 0.60 s | 1.15 s | 0.86 s | 0.31 s | 0.35 s | — | 6.01 GB | 6.01 GB | 6.01 GB | 6.01 GB | 6.01 GB | 6.01 GB | — |
| fleet | highs | 12k | 0.53 s | 0.52 s | 1.15 s | 0.99 s | — | — | 0.33 s | 3.21 GB | 3.21 GB | 3.21 GB | 3.21 GB | — | — | 3.21 GB |
| storage | gurobi | 10k | 0.58 s | 0.57 s | 1.05 s | 0.80 s | 0.28 s | 0.34 s | — | 5.72 GB | 5.72 GB | 5.72 GB | 5.72 GB | 5.72 GB | 5.72 GB | — |
| storage | highs | 10k | 0.49 s | 0.49 s | 1.01 s | 0.81 s | — | — | 0.32 s | 2.76 GB | 2.76 GB | 2.76 GB | 2.76 GB | — | — | 2.76 GB |
| transport | gurobi | 9.8k | 0.58 s | 0.57 s | 0.99 s | 0.79 s | 0.30 s | 0.38 s | — | 4.31 GB | 4.31 GB | 4.31 GB | 4.31 GB | 4.31 GB | 4.31 GB | — |
| transport | highs | 9.8k | 0.53 s | 0.50 s | 1.03 s | 0.77 s | — | — | 0.32 s | 5.01 GB | 5.01 GB | 5.01 GB | 5.01 GB | — | — | 5.01 GB |

<!-- bench:/fresh -->

## First model against every model after it

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +16.4 ms on specsolve and +25.7 ms on specsolve-in-memory and +2.1 ms on linopy and +10.5 ms on pyomo and +3.4 ms on gurobipy-loop and +0.1 ms on gurobipy-matrix and +0.9 ms on highspy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | specsolve: first | specsolve: steady | specsolve-in-memory: first | specsolve-in-memory: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady | highspy-matrix: first | highspy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| transport | 9.8k | 40.1 ms | **42.6 ms** | 52.3 ms | 50.4 ms | 96.2 ms | 62.1 ms | 48.9 ms | 47.6 ms | 78.5 ms | 50.8 ms | 23.2 ms | 23.6 ms | 12.2 ms | 12.3 ms |
| dispatch | 10k | 25.6 ms | **24.3 ms** | 26.8 ms | 24.7 ms | 58.2 ms | 28.6 ms | 35.8 ms | 33.6 ms | 29.1 ms | 28.1 ms | 16.0 ms | 16.7 ms | 6.4 ms | 6.3 ms |
| storage | 10k | 45.7 ms | **49.4 ms** | 56.8 ms | 51.7 ms | 93.1 ms | 91.1 ms | 40.3 ms | 40.0 ms | 52.9 ms | 53.5 ms | 21.0 ms | 21.5 ms | 10.3 ms | 10.2 ms |
| fleet | 12k | 78.0 ms | **79.3 ms** | 81.4 ms | 84.6 ms | 189.1 ms | 187.0 ms | 39.1 ms | 37.5 ms | 74.8 ms | 70.9 ms | 23.0 ms | 22.4 ms | 8.6 ms | 7.3 ms |
| transport | 98k | 71.7 ms | **57.4 ms** | 84.7 ms | 59.0 ms | 75.1 ms | 79.7 ms | 583.8 ms | 557.3 ms | 471.6 ms | 444.1 ms | 171.6 ms | 142.7 ms | 24.1 ms | 15.9 ms |
| dispatch | 100k | 37.5 ms | **36.0 ms** | 40.4 ms | 34.6 ms | 31.4 ms | 29.8 ms | 289.5 ms | 274.2 ms | 196.4 ms | 216.0 ms | 87.4 ms | 83.2 ms | 8.7 ms | 8.6 ms |
| storage | 100k | 75.2 ms | **50.1 ms** | 90.3 ms | 58.8 ms | 96.4 ms | 94.3 ms | 660.1 ms | 642.1 ms | 532.0 ms | 528.6 ms | 105.3 ms | 106.3 ms | 14.0 ms | 13.0 ms |
| fleet | 120k | 114.5 ms | **96.0 ms** | 121.7 ms | 92.9 ms | 193.0 ms | 192.3 ms | 854.8 ms | 804.5 ms | 860.7 ms | 869.8 ms | 159.3 ms | 161.0 ms | 12.9 ms | 12.8 ms |
| transport | 980k | 230.9 ms | **234.2 ms** | 287.8 ms | 254.0 ms | 220.0 ms | 208.7 ms | 6255.7 ms | 6211.9 ms | 4760.2 ms | 4769.3 ms | 1050.2 ms | 1064.6 ms | 85.5 ms | 77.3 ms |
| dispatch | 1M | 121.5 ms | **101.8 ms** | 161.4 ms | 146.7 ms | 39.8 ms | 38.2 ms | 4866.1 ms | 4860.5 ms | 2748.9 ms | 2751.2 ms | 852.6 ms | 840.5 ms | 54.0 ms | 52.7 ms |
| storage | 1M | 227.9 ms | **200.8 ms** | 188.1 ms | 210.9 ms | 117.1 ms | 115.5 ms | 5867.6 ms | 5871.6 ms | 5763.5 ms | 5758.1 ms | 1129.0 ms | 1110.8 ms | 108.7 ms | 107.3 ms |
| fleet | 1.2M | 287.2 ms | **275.6 ms** | 249.3 ms | 214.1 ms | 217.1 ms | 215.0 ms | 6242.6 ms | 6138.1 ms | 8370.0 ms | 8348.9 ms | 1638.1 ms | 1655.7 ms | 140.7 ms | 112.7 ms |
| transport | 9.8M | 1107.3 ms | **1067.5 ms** | 1437.6 ms | 1412.0 ms | 1548.5 ms | 1536.5 ms | — | — | — | — | 10298.4 ms | 10265.5 ms | 969.0 ms | 968.2 ms |
| dispatch | 10M | 479.6 ms | **414.9 ms** | 768.3 ms | 725.1 ms | 176.5 ms | 166.6 ms | — | — | 28423.4 ms | 28396.4 ms | 8896.5 ms | 8869.4 ms | 789.3 ms | 792.2 ms |
| storage | 10M | 1206.5 ms | **1063.2 ms** | 1378.5 ms | 1288.6 ms | 415.0 ms | 403.5 ms | — | — | — | — | 11105.8 ms | 11106.1 ms | 1137.5 ms | 1132.9 ms |
| fleet | 12M | 1349.3 ms | **1279.4 ms** | 1601.0 ms | 1463.6 ms | 507.1 ms | 505.0 ms | — | — | — | — | 16707.4 ms | 16690.8 ms | 1406.3 ms | 1408.0 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

This table renders from the `highs` width rungs of `storage` and `transport`,
and no run has published it: both width ladders are killed on both sinks before
they write a file, for the reason [below](#not-measured-yet). `bench.report`
drops a fragment it cannot render rather than blanking the fence, so the fence
stays empty until `pixi run ladder` brings those rungs back.

<!-- bench:sweeps -->
<!-- bench:/sweeps -->

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Sparsity, which is the engine's whole premise.** Every published cell is
  100% live: the `where` in `dispatch` removes nothing, and `transport`,
  `storage` and `fleet` carry no mask at all. A dense coordinate product is the shape an
  array engine is built for. The published ladder therefore compares
  the two lanes only where the relational one has the least to win. `nodal` and
  `sector` are the sparse cases, at 25% and 8.3% of their product. `nodal` has
  a linopy formulation now, so `pixi run density` sweeps it at four densities;
  no published run has taken it.
- **Solve time.** Every number stops at the hand-off. The simplex is the
  solver's work whoever filled the model.
- **The LP-file round trip.** The tables price writing a file, never reading
  one back.
- **Sizes past `l`.** `xl` and `2xl` exist in the harness and no run
  publishes them.
- **The width ladders of `storage` and `transport`.** The memory budget
  projects the next rung off a `w10` or `w100` cell that took under a
  gigabyte. Where the projection comes in under 16 GB the run starts a rung
  that does not fit, and `bench/memory-watchdog.sh` kills it to save the box.
  Four width ladders died that way on this run: `transport/w100` on linopy, on
  `highs` at 26.9 GB and on `gurobi` at 26.8 GB, and `storage/w1000` on the
  matrix floor, on `highs` at 26.9 GB and on `gurobi` at 28.1 GB. Each ladder
  runs in a process and a file of its own, so the kill took only the width
  rungs, and both cases publish their size ladder. The last `highs` numbers
  past `w10` are in
  [#1285](https://github.com/fluxopt/specsolve/pull/1285), taken on a machine that
  could hold them. There specsolve took 0.11 s and 0.59 GB at `transport/w100`,
  against linopy's 53.53 s and 14.26 GB.
- **Anything about expressiveness.** Four models say nothing about a fifth.

## Method

Each measurement runs in a process of its own. Peak memory is `ru_maxrss`
rather than a tracker. Import is excluded from the timing and teardown is
included. A run refuses to start on a machine that is already working. The
rest is in
[`bench/README.md`](https://github.com/fluxopt/specsolve/blob/main/bench/README.md):
every flag, every default switched off and what it costs.

**Peak carries an allocator cost that only the polars arms pay.** polars
ships its own jemalloc settings, so a peak measured through it holds pages
freed and not yet returned. An arm on the system allocator never enters
jemalloc. Our peak moves 12–27% with the decay clock on and off, where
linopy's does not move at three digits
([#896](https://github.com/fluxopt/specsolve/issues/896)). It runs against us and
is left in.

**memray never times anything.** Its tracker slows an allocation-heavy
engine several-fold and overcounts reserved arenas. Peak RSS is the metric;
memray is for attribution.
