# Benchmarks

Time and memory to build each model and hand it to the solver, in specsolve and
four other libraries. Nothing is solved, and CI checks that every library builds
the same model
([`bench/models/`](https://github.com/fluxopt/specsolve/blob/main/bench/models)).

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

One panel per model, one line per library. The x axis counts columns: the
variables that reach the solver. **Solver** picks the solver for every panel
and the table.

- **specsolve is the thick violet line.**
- **The grey line is the matrix floor**: a finished matrix handed to the solver
  in one call, `passModel` for HiGHS and `loadModel` for Gurobi. Its gap to
  specsolve is everything specsolve does before the solver holds the model.
  Its memory includes its own copy of the arrays.
- **A point is a median, and its whisker is the middle half of the rounds.**
  Overlapping whiskers are a tie
  ([why the median](https://github.com/fluxopt/specsolve/blob/main/bench/README.md#why-the-median)).
- **A shaded funnel is a projection** past the 30 s or 16 GB budget. A wide
  funnel is a guess.
- **÷ specsolve puts specsolve at 1×**, so a line at 3× is three times slower
  or larger. A point outside the window is named at the edge it crosses.
- **Drag a box on any chart to zoom them all**, and the table with them. A drag
  along one side zooms one axis. Double-click or **Reset zoom** to see
  everything.
- **The table divides by specsolve.** Purple is where specsolve is faster,
  orange where the other library is, and grey within 10%. `>30 s` and `>16 GB`
  are sizes the budget refused.

## The models

Columns and rows count the variables and the constraints that reach the solver,
one per value that exists. Every variable and constraint also runs over
`snapshot`, and a model grows by its snapshots alone.

<!-- bench:models -->

| | dispatch | network | nodal | sector |
| --- | --- | --- | --- | --- |
| for | bounds and one balance: the shape an array library suits best | a small PyPSA model: lines, stores, solar and wind, and pooled caps | dispatch at nodes, where a technology exists only where it is installed | each technology delivers one of five carriers, such as heat or hydrogen |
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
There it is 2.7× at 1M columns, 1.75 GB against 0.65 GB, and the memory
budget stops linopy at that size
([#2039](https://github.com/fluxopt/specsolve/pull/2039)).

## A script that builds one model

The charts leave out what a script pays to start: the interpreter, the import
and the first build. This table times a new process from launch to exit. The
process builds the model, hands it to HiGHS and exits, without a solve.

<!-- bench:fresh -->

<div class="bench"><div class="bench-table"><table class="bench-heat">
<thead><tr><th>launch to exit</th><th>dispatch<br>10k columns</th><th>network<br>10k columns</th><th>nodal<br>3k columns</th><th>sector<br>3k columns</th></tr></thead>
<tbody>
<tr><td>specsolve</td><td class="own">0.46 s</td><td class="own">0.51 s</td><td class="own">0.47 s</td><td class="own">0.49 s</td></tr>
<tr><td>specsolve-in-memory</td><td data-ratio="1.009">0.46 s · 1.0x</td><td data-ratio="0.977">0.50 s · 1.0x</td><td data-ratio="0.994">0.46 s · 1.0x</td><td data-ratio="0.956">0.47 s · 1.0x</td></tr>
<tr><td>linopy</td><td data-ratio="2.088">0.96 s · 2.1x</td><td data-ratio="2.141">1.10 s · 2.1x</td><td data-ratio="1.969">0.92 s · 2.0x</td><td data-ratio="1.935">0.95 s · 1.9x</td></tr>
<tr><td>pyomo</td><td data-ratio="1.500">0.69 s · 1.5x</td><td data-ratio="1.735">0.89 s · 1.7x</td><td data-ratio="1.449">0.68 s · 1.4x</td><td data-ratio="1.507">0.74 s · 1.5x</td></tr>
</tbody></table></div></div>

<!-- bench:/fresh -->

A script that builds one small model with specsolve and hands it to the solver
finishes in about half a second, about half of linopy's time
([#2031](https://github.com/fluxopt/specsolve/pull/2031)).

## Not measured yet

- **Sparsity other than a quarter.** `nodal`, `network` and `sector` are
  sparse at one density, a quarter of the node and technology pairs.
  `pixi run density` sweeps `nodal` from dense to one pair in twelve, and no
  published run has taken that sweep.
- **Solve time.** Every number stops when the model reaches the solver.

## Method

Each measurement runs in its own process, with import excluded and teardown
included, on an idle machine. A point is the median of nine rounds, or of five
where a round takes over a second. Peak memory is `ru_maxrss`. A projection's
two edges carry the last measured step forward, one at its cost per column and
one at its growth rate.
`uv run --locked bench/reproduce.py` runs it again with every version pinned,
and [the harness](https://github.com/fluxopt/specsolve/blob/main/bench/README.md)
has every flag and caveat, such as the allocator cost on polars' peak
([#896](https://github.com/fluxopt/specsolve/issues/896)).
