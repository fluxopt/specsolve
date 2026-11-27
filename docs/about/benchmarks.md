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

From 100k columns up, linopy's peak memory stays within 1.4× of specsolve's
on HiGHS, except on `network`, the one model that sums through a grouping.
There it is 2.6× at 1M columns, 1.80 GB against 0.69 GB, and the memory
budget stops linopy at that size
([#2242](https://github.com/fluxopt/specsolve/pull/2242)).

## A script that builds one model

The charts leave out what a script pays to start: the interpreter, the import
and the first build. This table times a new process from launch to exit. The
process builds the model, hands it to HiGHS and exits, without a solve.

<!-- bench:fresh -->

<div class="bench"><div class="bench-table"><table class="bench-heat">
<thead><tr><th>launch to exit</th><th>dispatch<br>10k columns</th><th>network<br>10k columns</th><th>nodal<br>3k columns</th><th>sector<br>3k columns</th></tr></thead>
<tbody>
<tr><td>specsolve</td><td class="own">0.53 s</td><td class="own">0.55 s</td><td class="own">0.48 s</td><td class="own">0.45 s</td></tr>
<tr><td>specsolve-in-memory</td><td data-ratio="1.020">0.54 s · 1.0x</td><td data-ratio="1.004">0.55 s · 1.0x</td><td data-ratio="1.069">0.51 s · 1.1x</td><td data-ratio="1.111">0.50 s · 1.1x</td></tr>
<tr><td>linopy</td><td data-ratio="2.177">1.16 s · 2.2x</td><td data-ratio="2.317">1.27 s · 2.3x</td><td data-ratio="2.177">1.05 s · 2.2x</td><td data-ratio="2.154">0.97 s · 2.2x</td></tr>
<tr><td>pyomo</td><td data-ratio="1.548">0.82 s · 1.5x</td><td data-ratio="1.794">0.98 s · 1.8x</td><td data-ratio="1.632">0.78 s · 1.6x</td><td data-ratio="1.645">0.74 s · 1.6x</td></tr>
</tbody></table></div></div>

<!-- bench:/fresh -->

A script that builds one small model with specsolve and hands it to the solver
finishes in about half a second, about half of linopy's time
([#2031](https://github.com/fluxopt/specsolve/pull/2031)).

## Where specsolve's build spends its time

Each table follows one model up its size ladder and splits specsolve's time
into the [phases](../reference/api.md#specsolve.types.Metrics) its engine
clocks. A phase that grows faster than the model is where a build slows down
first. A phase a run did not clock reads `—`.

<!-- bench:phases -->

<details markdown="1">
<summary><b>dispatch</b> — specsolve's phases on the highs sink</summary>

| phase | 200 columns | 1k columns | 5k columns | 10k columns | 50k columns | 100k columns | 500k columns | 1M columns | 5M columns | 10M columns |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| load | 0.892 ms | 1.01 ms | 0.971 ms | 1.01 ms | 0.884 ms | 0.931 ms | 0.963 ms | 0.997 ms | 1.04 ms | 1.02 ms |
| tidy | 4.32 ms | 5.22 ms | 4.38 ms | 5.23 ms | 4.93 ms | 6.8 ms | 5.23 ms | 5.43 ms | 9.77 ms | 17.9 ms |
| attach | 1.01 ms | 1.02 ms | 0.79 ms | 1.62 ms | 0.854 ms | 0.68 ms | 0.877 ms | 1.14 ms | 0.888 ms | 0.921 ms |
| build | 7.96 ms | 8.31 ms | 7.81 ms | 15.8 ms | 10.9 ms | 14.4 ms | 42.7 ms | 80.5 ms | 295 ms | 633 ms |

</details>

<details markdown="1">
<summary><b>network</b> — specsolve's phases on the highs sink</summary>

| phase | 800 columns | 4k columns | 10k columns | 50k columns | 100k columns | 500k columns | 1M columns | 5M columns | 10M columns |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| load | 2.1 ms | 2.12 ms | 2.14 ms | 2.34 ms | 2.13 ms | 2.07 ms | 2.24 ms | 2.06 ms | 2.08 ms |
| tidy | 15.7 ms | 16.8 ms | 16.8 ms | 17.6 ms | 18.8 ms | 36.3 ms | 73.3 ms | 485 ms | 967 ms |
| attach | 3.73 ms | 3.42 ms | 2.9 ms | 3.66 ms | 4.29 ms | 7.69 ms | 9.21 ms | 29.8 ms | 37.1 ms |
| build | 27 ms | 29.6 ms | 29.9 ms | 36.8 ms | 41.3 ms | 100 ms | 165 ms | 640 ms | 1.17 s |

</details>

<details markdown="1">
<summary><b>nodal</b> — specsolve's phases on the highs sink</summary>

| phase | 300 columns | 1.5k columns | 3k columns | 15k columns | 30k columns | 150k columns | 300k columns | 1.5M columns | 3M columns |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| load | 0.889 ms | 1.04 ms | 1.05 ms | 0.96 ms | 0.974 ms | 1.05 ms | 1.04 ms | 1.05 ms | 1.09 ms |
| tidy | 5.15 ms | 4.98 ms | 5.22 ms | 5.35 ms | 5.78 ms | 8.65 ms | 11.1 ms | 72.9 ms | 163 ms |
| attach | 0.987 ms | 1.29 ms | 0.862 ms | 0.966 ms | 1.07 ms | 1.62 ms | 2.41 ms | 6.28 ms | 12.4 ms |
| build | 7.62 ms | 8.91 ms | 10.4 ms | 8.36 ms | 11.3 ms | 25.1 ms | 31 ms | 109 ms | 210 ms |

</details>

<details markdown="1">
<summary><b>sector</b> — specsolve's phases on the highs sink</summary>

| phase | 300 columns | 1.5k columns | 3k columns | 15k columns | 30k columns | 150k columns | 300k columns | 1.5M columns | 3M columns |
| --- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| load | 1.01 ms | 1.11 ms | 1.05 ms | 1.12 ms | 1.13 ms | 1.07 ms | 1.16 ms | 1.38 ms | 1.05 ms |
| tidy | 5.57 ms | 5.94 ms | 6.27 ms | 6.92 ms | 8.48 ms | 17.8 ms | 42.1 ms | 179 ms | 599 ms |
| attach | 1.38 ms | 1.13 ms | 1.24 ms | 3.41 ms | 1.87 ms | 3.89 ms | 3.85 ms | 12.5 ms | 21.6 ms |
| build | 10.2 ms | 13.3 ms | 12.9 ms | 18.9 ms | 21.9 ms | 54.8 ms | 86.6 ms | 305 ms | 660 ms |

</details>

<!-- bench:/phases -->

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
