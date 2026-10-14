# Benchmarks

What it costs to build a model and hand it to a solver, in specsolve and in
four other ways of writing the same model. No number includes solve time.
Every rival model is hand-written in
[`bench/models/`](https://github.com/fluxopt/specsolve/blob/main/bench/models),
and CI solves each against ours, so nothing here is fast because it built a
different model.

Sizes on this page count **columns** and **rows**: the variables and the
constraints that reach the solver, one for each value. A model declares a
variable once, with its dimensions, such as `p[snapshot, generator]` in
`dispatch`. The solver receives one column for every combination of those
dimensions that exists, and one row for every combination of a constraint's.
Every variable and constraint below also runs over `snapshot`, and a model grows
by its snapshots alone.

<!-- bench:models -->

| | dispatch | network | nodal | sector |
| --- | --- | --- | --- | --- |
| variables | p[generator] | p[bus, tech], f[line], charge[store], discharge[store], soc[store] | p[node, tech] | p[node, tech] |
| constraints | power_balance | balance[bus], soc_balance[store], pool_cap[pool] | balance[node] | balance[node, carrier] |
| grouped sums | — | line → bus, store → bus, bus × tech → pool | — | — |
| dimensions | generator 100 | bus 50, tech 12, line 100, store 50 | node 50, tech 12 | node 50, tech 12, carrier 5 |
| density | 100% | 25% | 25% | 25% |
| snapshots | 2–100k | 2–25k | 2–20k | 2–20k |
| columns per snapshot | 100 | 400 | 150 | 150 |
| rows per snapshot | 1 | 150 | 50 | 125 |
| nonzeros per snapshot | 100 | 800 | 150 | 150 |

<!-- bench:/models -->

From 100k columns up, linopy's peak memory stays within 1.3× of specsolve's
on HiGHS, except on `network`, the one model that sums through a grouping.
There it is 2.8× at 1M columns, 1.76 GB against 0.63 GB, and the memory
budget stops linopy at that size
([#2031](https://github.com/fluxopt/specsolve/pull/2031)).

dispatch
:   Meets a demand from a fleet of units: bounds and one balance. The shape an
    array library is built for, so the comparison least in specsolve's favour.

nodal
:   Dispatch at nodes, where a technology exists only at the nodes it is
    installed at: a quarter of the node and technology pairs exist, and only
    those reach the solver as columns.

network
:   A small PyPSA dispatch model: buses joined by lines, a store at every bus
    that carries its charge from each snapshot to the next, a technology
    portfolio installed at some buses only, and solar and wind availability
    per snapshot. Each plant draws from one of fifty pools, such as a fuel
    contract, and a pool caps what its plants generate together.

sector
:   `nodal`'s portfolio, where each technology delivers one of five carriers,
    such as heat or hydrogen, at its own efficiency. A node has a balance only
    for the carriers its technologies deliver.

<div id="benchmark-charts" class="bench">
<div class="bench-controls" role="group" aria-label="Controls for the charts and the table">
<div class="bench-row">
<span class="bench-label">Solver</span>
<span class="bench-seg" data-key="sink"><button data-value="highs" aria-pressed="true">HiGHS</button><button data-value="gurobi" aria-pressed="false">Gurobi</button></span>
<span class="bench-label">Show</span>
<span class="bench-seg" data-key="metric"><button data-value="wall" aria-pressed="true">wall time</button><button data-value="peak" aria-pressed="false">peak RSS</button></span>
<span class="bench-label">As</span>
<span class="bench-seg" data-key="view"><button data-value="absolute" aria-pressed="true">measured</button><button data-value="relative" aria-pressed="false">÷ specsolve</button></span>
<span class="bench-label">Axes</span>
<span class="bench-seg" data-key="scale"><button data-value="log" aria-pressed="true">log–log</button><button data-value="linear" aria-pressed="false">linear–linear</button></span>
</div>
<div class="bench-row">
<span class="bench-label">Libraries</span>
<div id="bench-legend"></div>
<span class="bench-hint">click to hide</span>
</div>
</div>
<div class="bench-zoombar"><span class="bench-hint">drag on a chart to zoom every chart</span><button data-zoom-reset disabled>Reset zoom</button></div>
<div id="bench-facets"></div>
<div id="bench-scale" class="bench-scale"></div>
<div class="bench-table"><table id="bench-heat" class="bench-heat"></table></div>
<script type="application/json" id="bench-spec">
{
  "data": "../benchmarks.json",
  "facet": ["model"],
  "filter": { "ladder": "length", "sink": "highs" },
  "x": { "field": "variables", "label": "columns", "format": "count" },
  "metrics": {
    "wall": { "label": "wall time", "y": "wall_s", "band": ["wall_q1_s", "wall_q3_s"], "format": "seconds" },
    "peak": { "label": "peak RSS", "y": "peak_gb", "format": "gigabytes" }
  },
  "series": {
    "field": "library",
    "refused": "refused",
    "highlight": "specsolve",
    "floor": { "label": "matrix floor", "members": ["gurobipy-matrix", "highspy-matrix"] },
    "internal": ["specsolve-in-memory"],
    "domain": ["specsolve", "linopy", "pyomo", "gurobipy-loop", "specsolve-in-memory", "gurobipy-matrix", "highspy-matrix"],
    "ink": ["var(--s1)", "var(--s3)", "var(--s2)", "var(--s4)", "var(--s1)", "var(--floor)", "var(--floor)"],
    "paint": ["var(--s1)", "var(--p3)", "var(--p2)", "var(--p4)", "var(--p1)", "var(--floor)", "var(--floor)"]
  }
}
</script>
</div>

## Reading the charts

Each panel is one model, and each line is one library. Every panel has the
same axes, so a height or a size reads the same in each. **Solver** chooses the
solver that every panel and the table run through.

- **specsolve is the thick violet line.**
- **The grey line is the matrix floor**: the time to hand a finished matrix to
  the solver in one bulk call, `passModel` for HiGHS and `loadModel` for
  Gurobi. The floor builds its arrays before the clock starts, so it reads no
  parquet and builds nothing. Its distance from specsolve is everything
  specsolve does before the solver holds the model. Its peak RSS includes the
  arrays as well as the solver's copy of them.
- **The line is the median of nine rounds, or of five where a round takes
  over a second, and the whisker is the middle half.**
  Overlapping whiskers are numbers this run cannot tell apart
  ([why the median](https://github.com/fluxopt/specsolve/blob/main/bench/README.md#why-the-median)).
- **A shaded funnel is a projection** past the time or memory budget. It opens
  from the last measured point. One edge carries the last step's cost per
  column forward, and the other keeps the last step's growth rate. A wide
  funnel is a guess the measurements do not narrow.
- **÷ specsolve divides each point by specsolve's at the same size**, so
  specsolve is the line at 1× and a line above it is that many times slower or
  larger. It applies to wall time and to peak RSS. The window is 0× to 5× on
  linear axes and 0.1× to 50× on log axes, and a point outside it is named with
  its ratio at the edge it crosses.
- **Drag a box on any chart to zoom every panel to it**: its width sets the
  range of model sizes, and its height the range of values. A drag along one
  side zooms only that axis. The table keeps only the sizes inside the box.
  **Reset zoom** above the charts, or a double-click on one, shows everything
  again. Switching between wall time and peak RSS, or to and from ÷ specsolve,
  zooms out too.
- **The table divides by specsolve.** Purple is where specsolve is faster,
  orange where the other library is, grey within ten per cent. The
  specsolve-in-memory and matrix floor columns are not shaded: they are
  references, not rivals. `>30 s` and `>16 GB` are sizes the harness refused.

## A script that solves one model

The charts leave out what a script pays to start: the interpreter, the import
and the first build. This table times a new process from launch to exit, on
HiGHS.

<!-- bench:fresh -->

<div class="bench"><div class="bench-table"><table class="bench-heat">
<thead><tr><th>launch to exit</th><th>dispatch<br>10k columns</th><th>network<br>10k columns</th><th>nodal<br>3k columns</th><th>sector<br>3k columns</th></tr></thead>
<tbody>
<tr><td>specsolve</td><td class="own">0.47 s</td><td class="own">0.52 s</td><td class="own">0.47 s</td><td class="own">0.47 s</td></tr>
<tr><td>specsolve-in-memory</td><td data-ratio="0.974">0.45 s · 1.0x</td><td data-ratio="0.954">0.50 s · 1.0x</td><td data-ratio="0.988">0.47 s · 1.0x</td><td data-ratio="0.994">0.46 s · 1.0x</td></tr>
<tr><td>linopy</td><td data-ratio="2.091">0.97 s · 2.1x</td><td data-ratio="2.112">1.10 s · 2.1x</td><td data-ratio="2.000">0.94 s · 2.0x</td><td data-ratio="2.002">0.93 s · 2.0x</td></tr>
<tr><td>pyomo</td><td data-ratio="1.493">0.69 s · 1.5x</td><td data-ratio="1.710">0.89 s · 1.7x</td><td data-ratio="1.431">0.67 s · 1.4x</td><td data-ratio="1.595">0.74 s · 1.6x</td></tr>
<tr><td>highspy-matrix</td><td data-ratio="0.675">0.31 s · 0.7x</td><td data-ratio="0.612">0.32 s · 0.6x</td><td data-ratio="0.658">0.31 s · 0.7x</td><td data-ratio="0.682">0.32 s · 0.7x</td></tr>
</tbody></table></div></div>

<!-- bench:/fresh -->

A script that solves one small model with specsolve finishes in about half a
second, about half of linopy's time
([#2031](https://github.com/fluxopt/specsolve/pull/2031)).

## Not measured yet

- **Sparsity other than a quarter.** `nodal`, `network` and `sector` are
  sparse at one density, a quarter of the node and technology pairs.
  `pixi run density` sweeps `nodal` from dense to one pair in twelve, and no
  published run has taken that sweep.
- **Solve time.** Every number stops when the model reaches the solver.

## Method

Each measurement runs in its own process, with import excluded and teardown
included, on an idle machine. Peak memory is `ru_maxrss`.
`uv run --locked bench/reproduce.py` runs it again with every version pinned,
and [the harness](https://github.com/fluxopt/specsolve/blob/main/bench/README.md)
has every flag and caveat, such as the allocator cost on polars' peak
([#896](https://github.com/fluxopt/specsolve/issues/896)).
