# Sweeps and rolling horizons

`solve_over` runs one [model](glossary.md#the-chain) once per slice and folds
the answers together. A slice is one set of [sources](glossary.md#how-it-runs)
under one key. This page is the reference for its axes, keywords and the `Sweep` it
returns. [A horizon too large for one solve](../sweep.md) and
[Solving many scenarios](../howto/scenarios.md) show them in use.

## The axes

An axis says how the sources split into slices:

| Axis | Slices |
|---|---|
| [`EachCoordinate`](api.md#specsolve.EachCoordinate) | one per label of a dimension |
| [`EachWindow`](api.md#specsolve.EachWindow) | one per window of consecutive labels |
| a tuple of them, outer first | one per combination ([several axes](#several-axes)) |
| a sequence of `(key, sources)` pairs | as written. It must pass `key_name=`, and its answer is keyed by slice. |

`EachWindow`'s `into` has no default, and a seam's `where: "t == 0"` matches on it.

**A datetime axis is held in microseconds**, as
[a label is](data.md#where-coordinates-come-from). A key finer than a
microsecond is refused.

**`steps` as a sequence is one block per window**, laid end to end.
`lookahead` is one number, whatever the blocks.

```python
sps.EachWindow('snapshot', steps=[24, 24, 168, 168, 720], lookahead=12, into='t')
sps.EachWindow('snapshot', steps=days_in_each_month, lookahead=48, into='t')
```

**A sequence that stops short of the axis is refused:**

```text
DataError: steps keeps 7 coordinate(s) across 2 window(s), and 'snapshot' has 12 —
the last 5 would be solved by no window. List a block for them, or pass an int to
repeat one size to the end.
```

**`axis.slices(sources)` is the list the axis would run**, as `(key, sources)`
pairs:

```python
slices = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t').slices(sources)
sps.build('window.yaml', slices[37][1]).write('window-37.lp')  # the one that was infeasible
```

**An axis is also plain data.** `axis.to_dict()` writes it as a mapping, and
`solve_over` takes that mapping in the axis's place. An archive's `axis.json`
holds the same mapping. A tuple of axes is `{'each': 'axes', 'axes': [...]}`,
each axis as its `to_dict` writes it. A mapping with a key the axis does not
take, or without one it needs, is refused, and the message names the keys the
axis takes.

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t', carry={'soc_initial': 'soc'})
sps.solve_over('window.yaml', sources, axis.to_dict())
```

**Sources cross a slice in every shape `build` takes.** A parameter or relation
whose table carries the axis is filtered. Every other source passes through. An
index of another dimension that carries the axis is refused. A table carrying
the axis that is short of a coordinate another table has warns, naming both.

## Several axes

**A tuple of axes cuts the sources with each axis in turn, outer first.** A
rolling horizon per scenario is one call:

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    (
        sps.EachCoordinate('scenario'),
        sps.EachWindow('snapshot', steps=24, lookahead=24, into='t', carry={'soc_initial': 'soc'}),
    ),
)
sweep.key_names  # ('scenario', 'snapshot_start')
sweep.primal('soc')  # (scenario, snapshot, value)
```

| Rule | |
|---|---|
| **each chain answers as its own sweep would** | A **chain** is the slices that share the keys of every axis outside the outermost one that carries, such as one scenario's windows. Its answer equals `solve_over` on its own sources, with the same carry and start. |
| **any axis may be windows** | Two axes over one dimension, and two windows over one local index, are refused before a slice is taken. |
| **one key column per axis** | `sweep.key_names` names them outer first, `sweep.key_name` is the last, and `sweep.keys` holds one tuple per slice. On disk, `slice_axis` and `slice` join them with `/`, as `scenario/snapshot_start` and `high/0`. |
| **windows stitch at every level** | The answer is over every dimension a window cut, and holds only what every window owns. Per window, it carries every key column and every local index. |
| **each axis carries its own state** | A carry reaches the slices of its axis's next label. Under an axis inside it, the value is the last inner slice's, and it reaches every inner slice of that label. An inner carry starts again from its seed at each outer label. |
| **an executor runs the chains concurrently** | Where an axis carries, each chain is one task, run in order on one model. Otherwise each slice is one task. |
| **a start is cut by every axis** | An earlier sweep over the same axes starts each slice from its own rows. |
| **a tuple spills, resumes and archives** | `load_archive` gives the tuple back as `archive.axis`. |
| **`key_name=` is refused** | Each axis names its own key column. |

**Windows inside windows are a myopic pathway with foresight.** Each decade
window sees the next decade, solves its hours as a rolling horizon, and hands
on the fleet it ends its own decade with. The spec declares both local
indexes, `y` and `t`:

```python
sweep = sps.solve_over(
    'pathway.yaml',
    sources,
    (
        sps.EachWindow('period', steps=1, lookahead=1, into='y', carry={'existing': 'capacity'}),
        sps.EachWindow('snapshot', steps=24, lookahead=24, into='t', carry={'soc_initial': 'soc'}),
    ),
)
sweep.primal('soc')  # (period, snapshot, value)
```

## Reading a sweep

**Every reader returns the answer**, with the names and shapes of
[`Result`](api.md#specsolve.types.Result): `primal`, `dual`, `evaluate`,
`activity`, `reduced_cost`, `slack`, `variable_basis`, `constraint_basis`,
`scan`, `to_pandas`, `to_dataarray` and `to_dataset`. An `EachWindow`
answer is over the real labels of the sliced dimension, each from the window
that owns it. Any other answer is keyed by slice.

**You name the extra dimension.** `EachCoordinate('scenario')` keys on
`scenario`, so `sweep.to_dataarray('p')` is `(scenario, snapshot, generator)`.

**A sweep holds only the outputs you ask for**, such as
`outputs={'activity', 'reduced_cost'}`. A sweep that did not ask refuses the
reader. A slice with no duals has no reduced costs, and the reader says why. A
`spill_to=` directory solved with other outputs is refused rather than
resumed.

**`per_window=True` reads each window as it was solved**, lookahead rows
included, on every reader:

```python
sweep.primal('soc', per_window=True)  # (snapshot_start, t, value) — every row every window solved
sweep.to_dataarray('balance', 'dual', per_window=True)  # (snapshot_start, t)
```

**A sweep not cut into windows refuses `per_window=True`:**

```text
SpecsolveError: per_window=True reads an EachWindow sweep one window at a time,
and this sweep was not cut into windows: its answer already is one frame per
slice, keyed by 'scenario'. Read it without per_window.
```

**A quantity that is not over the windowed dimension has no answer.**
The reader refuses `sum(p * cost)`, and `to_dataset()` leaves it out. Read it
with `per_window=True`.

**`save` writes a held sweep as a spill**, which `sps.load_sweep` and
`sps.scan_sweep` read back ([`load_sweep`](api.md#specsolve.load_sweep)).
**There is no per-slice reader**: one slice is
`sweep.primal('p').partition_by(sweep.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **duals are keyed, never combined** | A slice with an integer variable contributes no duals, and `sweep.record` says which. |
| **no aggregate objective** | `sweep.record` is keyed by slice, with the objective as one column. |
| **the lookahead is `t >= step`** | What each window owns is `sweep.primal('soc', per_window=True).filter(pl.col('t') < step)`. |
| **a slice that did not solve contributes no rows** | `record` still holds one row per slice, with a null objective where a slice reached none. |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` re-indexes to `into`. Per window, the key column `snapshot_start` holds where each window began. |
| **`key_name=` names the key** | On any one axis. A name that collides with a table column, or starts with `specsolve_`, is refused. |
| **`sweep.metrics` says what each slice took** | One [`Metrics`](api.md#specsolve.types.Metrics) per slice, keyed like `record`. |
| **on disk, the slice is two text columns** | `slice_axis` holds the key name and `slice` the key as text, each joined by `/` over several axes. Two keys of one text are refused. |
| **a slice that fails says which slice** | The engine's error carries a note: `in slice 'bad' (3 of 3)`. |
| **memory grows with the answer, unless it is spilled** | Each slice's model is released; the tables accumulate unless [spilled](#spilling-a-sweep-to-disk). |

## Spilling a sweep to disk

**`spill_to=` writes each slice's tables to a directory**, so memory stays at
one slice:

```python
sweep = sps.solve_over(
    'window.yaml', sources, sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'), spill_to='runs/'
)
sweep.scan('soc')  # a LazyFrame: (snapshot, value), the answer
sweep.scan('balance', 'dual', per_window=True).collect()  # the same readers, the same keywords
```

| Rule | |
|---|---|
| **`scan` reads lazily** | `sweep.scan(name, kind='primal')` returns a `LazyFrame` over the files. |
| **one file per slice and name** | `<kind>/<name>/<position>.parquet`, plus `record/` and `metrics/`. |
| **an interrupted sweep resumes** | Run the same call at the same directory. A finished slice is read back, and a slice interrupted part way is solved again. |
| **a directory holds one sweep** | A different sweep is refused. Changed data or a changed spec is not detected, so delete the directory to solve again. |
| **the parent writes** | Under `executor=`, workers send their answers back to the parent. |

## Archiving a sweep

**`archive=` writes the model, the sources, the axis and the answer**, so the
sweep runs again from the file alone ([archiving](../howto/archiving.md)):

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'),
    archive='roll.zip',
    keep_windows=True,
)
```

| Rule | |
|---|---|
| **an archive holds the answer** | At `answer/<kind>/<name>.parquet`, as for a single solve. A name with no answer is left out, and the reader raises its reason. |
| **`keep_windows=True` keeps each window's frames** | Without them, a `per_window=True` read of the archive is refused. It needs `EachWindow` and `archive=`. |

## Carrying state between slices

**An axis's `carry={parameter: variable}` copies one slice's answer into the
next slice's data.** `EachCoordinate` and `EachWindow` take it. A hand-built
list carries nothing.

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, into='t', carry={'soc_initial': 'soc'}),
)
```

**You name no coordinate.** The row handed on is the last one the slice owns:
label 23 of a window keeping 24, not label 47 of the 48 it solved.

| Rule | |
|---|---|
| **a carry is a copy, never arithmetic** | Accumulation (`existing += built`) is a derived variable in the YAML. |
| **the carry collapses the one dimension the variable has and the parameter does not** | `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t`. Same dimensions move the whole frame. |
| **that dimension must be `EachWindow`'s `into`** | Any other is refused. A carry on `EachCoordinate` collapses nothing. |
| **a carried value is a boundary condition, never a pin** | Write `soc == soc_initial + charge * 0.9 - discharge` at `t == 0`. `soc == soc_initial` drops the first row's dynamics, so the window gets free energy at every seam. |
| **the first slice needs a seed** | It takes the parameter from `sources`. A sweep whose sources lack it is refused. |
| **a carry is checked before any source is read** | Two collapsed dimensions, a parameter over more dimensions than the variable, the wrong dimension and no seed all raise first. `check` cannot see a carry, because it is not part of the spec. |
| **a slice that leaves nothing to carry stops the sweep** | The error names the slice and how it terminated. Without a carry, the sweep goes on. |
| **one axis carries a parameter** | Two axes that write one parameter are refused before a slice is taken. |
| **`carry` excludes `executor` on a sweep of one chain** | Slice *i+1* depends on slice *i*, so the call is refused. Over [several axes](#several-axes), the chains run concurrently. |

## Starting slices

`start=` takes what [`solve`](api.md#specsolve.types.Model.solve) takes, or an
earlier sweep. **A start is cut by the axis, as a source is.**

```python
again = sps.solve_over('dispatch.yaml', new_sources, axis, start=earlier)
cold = sps.solve_over('dispatch.yaml', sources, axis, start=None)
```

| Rule | |
|---|---|
| **an earlier sweep starts each slice from the slice of its key** | Under an executor too. |
| **`'previous'`, the default, starts each slice from the one before** | The first slice, a slice that loads the solver again, and the first slice of each task under an `executor` begin from nothing: a task is a slice, or a chain where an axis carries. |
| **a start is checked before a slice is built** | A table over an `EachWindow` sweep's local index alone is refused. A slice its start leaves no row warns and starts from nothing. |

## Running slices in parallel

**`executor` is any
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)**:
a `submit` that returns a `Future`.

| | Use it when | Notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | Sequential. |
| `ThreadPoolExecutor` | rarely | Slices contend with polars' own thread pool, and peak memory adds up. |
| `ProcessPoolExecutor` | local parallelism | **Must not use `fork`.** Sources cross as parquet. |
| anything remote | a cluster you already run | dask's `Client`, ray, loky. Paths travel as bytes; pass `workers_share_fs=True` if workers mount your filesystem. |

**A forked worker hangs** rather than raising.
[Running a sweep in parallel](../howto/parallel.md) is the recipe.

**Parallel is N × peak.** A four-way pool wants four times the memory of one
slice.

**Pass paths or tables, whichever you have.** A source no slice rewrote is
encoded once. `df.lazy()` pickles larger than the table.

## How a sweep runs

| | |
|---|---|
| **one model, updated per slice** | A serial sweep builds once and [updates](api.md#specsolve.types.Model.update); under `executor=` each task builds its own: a slice, or a chain where an axis carries. A [remote Gurobi](api.md#specsolve.types.Model.solve) opens one session per serial sweep, and a new one at each slice that loads again. |
| **the model is asked before it is sliced** | `EachWindow` refuses a window the model's coupling cannot bear. `EachCoordinate` is not asked. |
| **a slice is total** | A slice supplies what the whole model attaches, not what changed. A hand-built list has to keep this rule. |
