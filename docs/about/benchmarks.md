# Benchmarks

What it costs to build a model and hand it to a solver, in specsolve and in
four other ways of writing the same model. No number includes solve time.
Every rival model is hand-written in
[`bench/models/`](https://github.com/fluxopt/specsolve/blob/main/bench/models),
and CI solves each against ours, so nothing here is fast because it built a
different model.

dispatch
:   Meets a demand from a fleet of units: bounds and one balance.

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

Each panel is one model through one solver, and each line is one library.

- **specsolve is the thick violet line.**
- **The grey line is the matrix floor**: a hand-written matrix handed straight
  to the solver's bulk API. Its distance from specsolve is what the modelling
  layer costs.
- **The line is the median of nine rounds, and the band is the middle half.**
  Overlapping bands are numbers this run cannot tell apart
  ([why the median](https://github.com/fluxopt/specsolve/blob/main/bench/README.md#why-the-median)).
- **A dashed line is a projection** past the time or memory budget.
- **The table divides by specsolve.** Green is where specsolve is faster, blue
  where the other library is, grey within ten per cent. `>30 s` and `>16 GB`
  are sizes the harness refused.

## Not measured yet

- **Sparsity, which is the engine's whole premise.** Every published model is
  dense, the shape an array engine is built for. `pixi run density` sweeps a
  sparse one, and no published run has taken it.
- **Solve time.** Every number stops when the model reaches the solver.

## Method

Each measurement runs in its own process, with import excluded and teardown
included, on an idle machine. Peak memory is `ru_maxrss`.
`uv run --locked bench/reproduce.py` runs it again with every version pinned,
and [the harness](https://github.com/fluxopt/specsolve/blob/main/bench/README.md)
has every flag and caveat, such as the allocator cost on polars' peak
([#896](https://github.com/fluxopt/specsolve/issues/896)).
