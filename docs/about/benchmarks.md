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
- **The pale violet line is specsolve on polars' in-memory engine.** specsolve
  runs on polars' default engine everywhere else on this page. The
  [how-to for small models](../howto/small-models.md) says when to switch.
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
([#1934](https://github.com/fluxopt/specsolve/pull/1934)) 15 published cells have a mean
above 1.10x their median, the worst at 3.00x: `dispatch/xs` on pyomo, on the
`highs` [sink](../reference/glossary.md#how-it-runs). In 11 of them the first
round is the slowest of the nine, at 1.6 to 18.9 times the median of the other
eight, so the median does not move. The other four are matrix-floor and
gurobipy-loop cells at `xs` and `s` whose rounds fall into two groups, the slow
one about twice the fast one: `dispatch/xs` on gurobipy-loop takes either 23 to
26 ms or 48 to 49 ms.

The median flipped two cells on this run, both against specsolve:
`dispatch/xs` and `dispatch/s` on the `highs` sink, against linopy. On each,
specsolve's fastest round is the faster one, 28.0 ms against 28.4 ms and 40.7 ms
against 44.7 ms, and its median is the slower, 39.9 ms against 29.1 ms and
56.3 ms against 46.4 ms.

## How to reproduce it

```bash
uv run --locked bench/reproduce.py
```

`bench/reproduce.py.lock` freezes every version, git commits included: two of
the five libraries install from git and one of those is a branch. `--locked`
refuses to start if the resolution has drifted.

Everything the tables are drawn from is in
[`bench/results`](https://github.com/fluxopt/specsolve/blob/main/bench/results):
one file per sink and case. Each carries the machine, the versions, the commit
and every round of every measurement. A case the box could not finish leaves
no file behind. `pixi run table` prints the directory as one long CSV and
commits nothing; the JSON stays the archive because it keeps the rounds.
`pixi run refresh` re-takes the numbers, writes the tables into their fences
and the chart rows into `benchmarks.json`.

## First model against every model after it

<!-- bench:marginal -->

### Marginal cost per model

Build only, repeated in one process. **first** is the first recorded round and **steady** the best of the rounds after it, so the pair is what a rolling horizon pays for its second window against its first. The harness warms up before it records, so neither column carries the one-time import cost: the median gap between them is +11.8 ms on specsolve and +1.7 ms on linopy and +5.1 ms on pyomo and +1.2 ms on gurobipy-loop and +9.4 ms on gurobipy-matrix and +2.4 ms on highspy-matrix.

**Read down a column, not across the row.** The build is not the same work in every library — one that defers materialising its coefficients to its writer spends almost nothing here and pays it at the seam — so these columns carry no ratios. The tables above measure to a common artifact and are where a comparison belongs.

| case | vars | specsolve: first | specsolve: steady | linopy: first | linopy: steady | pyomo: first | pyomo: steady | gurobipy-loop: first | gurobipy-loop: steady | gurobipy-matrix: first | gurobipy-matrix: steady | highspy-matrix: first | highspy-matrix: steady |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dispatch | 10k | 39.9 ms | **32.1 ms** | 30.6 ms | 28.8 ms | 35.8 ms | 33.8 ms | 27.2 ms | 26.0 ms | 29.2 ms | 15.2 ms | 10.1 ms | 6.3 ms |
| fleet | 12k | 133.6 ms | **130.5 ms** | 221.7 ms | 191.0 ms | 39.2 ms | 37.3 ms | 102.6 ms | 72.5 ms | 24.4 ms | 22.8 ms | 8.0 ms | 7.2 ms |
| dispatch | 100k | 49.0 ms | **39.9 ms** | 30.0 ms | 29.1 ms | 271.4 ms | 263.2 ms | 195.7 ms | 223.6 ms | 85.2 ms | 80.5 ms | 9.2 ms | 8.5 ms |
| fleet | 120k | 188.0 ms | **160.9 ms** | 193.8 ms | 192.2 ms | 726.0 ms | 729.0 ms | 837.8 ms | 836.2 ms | 150.3 ms | 156.3 ms | 12.5 ms | 12.1 ms |
| dispatch | 1M | 138.6 ms | **124.1 ms** | 39.7 ms | 37.9 ms | 4688.5 ms | 4678.1 ms | 2550.6 ms | 2551.3 ms | 820.5 ms | 824.0 ms | 56.9 ms | 32.4 ms |
| fleet | 1.2M | 324.1 ms | **307.5 ms** | 217.5 ms | 217.1 ms | 5849.2 ms | 5814.5 ms | 8317.3 ms | 8272.3 ms | 1681.1 ms | 1650.7 ms | 127.0 ms | 126.0 ms |
| dispatch | 10M | 487.5 ms | **485.9 ms** | 156.2 ms | 148.9 ms | — | — | 27235.1 ms | 27306.0 ms | 8841.1 ms | 8728.3 ms | 792.4 ms | 753.7 ms |
| fleet | 12M | 1776.4 ms | **1743.1 ms** | 528.0 ms | 526.7 ms | — | — | — | — | 16855.1 ms | 16815.2 ms | 1423.1 ms | 1395.6 ms |

<!-- bench:/marginal -->

## The same size, reached by widening

This table renders from `highs` rungs of `storage` and `transport`, and no run
has published it: both cases are killed on both sinks before they write a file,
for the reason [below](#not-measured-yet). `bench.report` drops a fragment it
cannot render rather than blanking the fence, so the fence stays empty until
`pixi run ladder` brings those rungs back.

<!-- bench:sweeps -->
<!-- bench:/sweeps -->

## Not measured yet

Listed so that a claim with no table under it is visible as one.

- **Sparsity, which is the engine's whole premise.** Every published cell is
  100% live: the `where` in `dispatch` removes nothing, and `transport` and
  `fleet` carry no mask at all. A dense coordinate product is the shape an
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
- **`storage` and `transport` are absent from every table.** The memory budget projects the next rung off a `w10` or `w100` cell
  that took under a gigabyte. Where the projection comes in under 16 GB the run
  starts a rung that does not fit, and `bench/memory-watchdog.sh` kills the
  case to save the box. Four cases died that way on this run:
  `transport/w100` on `highs` at 24.3 GB, `storage/w1000` on `highs` at
  24.9 GB, `transport/w100` on `gurobi` at 25.2 GB, `storage/w1000` on
  `gurobi` at 26.5 GB. A killed case writes no
  file, so each loses every rung it had already measured, its rows in the
  marginal table above included
  ([#1498](https://github.com/fluxopt/specsolve/issues/1498)). The last `highs`
  numbers past `w10` are in
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
