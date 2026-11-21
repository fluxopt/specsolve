# Sweeps and rolling horizons

`solve_over` runs one [model](glossary.md#the-chain) once per slice and folds
the answers together. A slice is one set of [sources](glossary.md#how-it-runs)
under one key. This page is the reference for its axes, keywords and the `Sweep` it
returns. [A horizon too large for one solve](../sweep.md) and
[Solving many scenarios](../howto/scenarios.md) show them in use.

## The axes

`over=` takes an axis, which says how the sources split into slices:

| Axis | Slices |
|---|---|
| [`EachLabel`](api.md#specsolve.EachLabel) | one per label of a dimension the model does not declare |
| [`EachWindow`](api.md#specsolve.EachWindow) | one per window of consecutive labels of a dimension it declares |

**`EachLabel` squeezes a dimension the model does not declare**, and
**`EachWindow` keeps one it does.** A label's slice holds that label's rows
without the column, and the answer lifts the label back as the key. A window
is the model over its own labels, so the spec and the sources are the ones a
single solve takes. A rule for a window's first row is written
`where: "position(snapshot) == 0"`, since a position counts within the window.

**A windowed dimension is declared `ordered: true`.** Any other is refused,
since its windows would follow the row order of the data.

**A datetime axis is held in microseconds**, as
[a label is](data.md#where-coordinates-come-from). A key finer than a
microsecond is refused.

**`steps` as a sequence is one block per window**, laid end to end.
`lookahead` is one number, whatever the blocks.

```python
sps.EachWindow('snapshot', steps=[24, 24, 168, 168, 720], lookahead=12)
sps.EachWindow('snapshot', steps=days_in_each_month, lookahead=48)
```

**A sequence that stops short of the axis is refused:**

```text
DataError: steps keeps 7 label(s) across 2 window(s), and 'snapshot' has 12 —
the last 5 would be solved by no window. List a block for them, or pass an int to
repeat one size to the end.
```

**An axis is also plain data.** `axis.to_dict()` writes it as a mapping, and
`solve_over` takes that mapping in the axis's place. An archive's `axis.json`
holds the same mapping. A mapping with a key the axis does not take, or without
one it needs, is refused, and the message names the keys the axis takes.

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, carry={'soc_initial': 'soc'})
sps.solve_over('window.yaml', sources, axis.to_dict())
```

**Sources cross a slice in every shape `build` takes.** A parameter or relation
whose table carries the axis is filtered. Every other source passes through. An
index of another dimension that carries the axis is refused. A table carrying
the axis that is short of a label another table has warns, naming both.

**Every source carries the axis as one type.** `int` in one table and `str` in
another is refused, naming both tables, and so is a datetime with a time zone
beside one without. A row with no label in that column is refused too.

## Reading a sweep

**Every reader returns the answer**, with the names and shapes of
[`Result`](api.md#specsolve.types.Result): `primal`, `dual`, `evaluate`,
`activity`, `reduced_cost`, `slack`, `variable_basis`, `constraint_basis`,
`scan`, `to_pandas`, `to_dataarray` and `to_dataset`. An `EachWindow`
answer is over the real labels of the sliced dimension, each from the window
that owns it. An `EachLabel` answer is keyed by slice.

**You name the extra dimension.** `EachLabel('scenario')` keys on
`scenario`, so `sweep.to_dataarray('p')` is `(scenario, snapshot, generator)`.
`EachLabel('scenario', key_name='draw')` keys on `draw`.

**A sweep holds only the outputs you ask for**, such as
`outputs={'activity', 'reduced_cost'}`. A sweep that did not ask refuses the
reader. A slice with no duals has no reduced costs, and the reader says why. A
`spill_to=` directory solved with other outputs is refused rather than
resumed.

**`per_window=True` reads each window as it was solved**, lookahead rows
included, on every reader:

```python
sweep.primal('soc', per_window=True)  # (snapshot_start, snapshot, value) — every row every window solved
sweep.to_dataarray('balance', 'dual', per_window=True)  # (snapshot_start, snapshot)
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
`sps.scan_sweep` read back ([`load_sweep`](api.md#specsolve.load_sweep)). It
replaces what an earlier save left in the directory, as a result's `save` does.
**There is no per-slice reader**: one slice is
`sweep.primal('p').partition_by(sweep.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **duals are keyed, never combined** | A slice with an integer variable contributes no duals, and `sweep.records` says which. |
| **no aggregate objective** | `sweep.records` is keyed by slice, with the objective as one column. |
| **a window owns its first `steps` labels** | The rows after them are lookahead. The answer keeps the owned rows, and `per_window=True` reads them all. |
| **a slice that did not solve contributes no rows** | `sweep.records` still holds one row per slice, with a null objective where a slice reached none. |
| **a window keys as `<dim>_start`** | Per window, the key column `snapshot_start` holds the label each window began at, beside the window's own `snapshot`. |
| **an axis's `key_name=` names the key** | On either axis. A name that collides with a table column, or starts with `specsolve_`, is refused. An archive holds it with the axis, so the sweep runs again under the same key. |
| **`sweep.metrics` says what each slice took** | One [`Metrics`](api.md#specsolve.types.Metrics) per slice, keyed like `sweep.records`. |
| **on disk, the slice is two text columns** | `slice_axis` holds the key name and `slice` the key as text. Two keys of one text are refused. |
| **a slice that fails says which slice** | The engine's error carries a note: `in slice 'bad' (3 of 3)`. |
| **memory grows with the answer, unless it is spilled** | Each slice's model is released; the tables accumulate unless [spilled](#spilling-a-sweep-to-disk). |

## Spilling a sweep to disk

**`spill_to=` writes each slice's tables to a directory**, so memory stays at
one slice:

```python
sweep = sps.solve_over('window.yaml', sources, sps.EachWindow('snapshot', steps=24, lookahead=24), spill_to='runs/')
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
    sps.EachWindow('snapshot', steps=24, lookahead=24),
    archive='roll.zip',
)
```

| Rule | |
|---|---|
| **an archive holds a sweep as a spill does** | `answer/` is the layout `spill_to=` and `save` write: `answer/<kind>/<name>/<position>.parquet`, with `specsolve_run` added to each table. |
| **every reader answers off the archive** | `per_window=True` included, since every window is there. |

## Carrying state between slices

**An axis's `carry={parameter: variable}` copies one slice's answer into the
next slice's data.** `EachLabel` and `EachWindow` take it.

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, carry={'soc_initial': 'soc'}),
)
```

**You name no label.** The row handed on is the last one the slice owns:
label 23 of a window keeping 24, not label 47 of the 48 it solved.

| Rule | |
|---|---|
| **a carry is a copy, never arithmetic** | Accumulation (`existing += built`) is a derived variable in the YAML. |
| **the carry collapses the one dimension the variable has and the parameter does not** | `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t`. Same dimensions move the whole table. |
| **a window carries nothing over the dimension it windows** | No two windows hold the same labels, so a parameter over that dimension is refused. |
| **that dimension must be the one `EachWindow` windows** | Any other is refused. A carry on `EachLabel` collapses nothing. |
| **a carried value is a boundary condition, never a pin** | Write `soc == soc_initial + charge * 0.9 - discharge` where `position(t) == 0`, a window's first row. `soc == soc_initial` drops the first row's dynamics, so the window gets free energy at every seam. |
| **the first slice needs a seed** | It takes the parameter from `sources`. A sweep whose sources lack it is refused. |
| **a carry is checked before any source is read** | Two collapsed dimensions, a parameter over more dimensions than the variable, the wrong dimension and no seed all raise first. `check` cannot see a carry, because it is not part of the spec. |
| **a slice that leaves nothing to carry stops the sweep** | The error names the slice and how it terminated. Without a carry, the sweep goes on. |
| **`carry` excludes `executor`** | Slice *i+1* depends on slice *i*, so the call is refused. |

## Building one slice

**[`build_slice`](api.md#specsolve.build_slice) builds the model of one slice**,
as `solve_over` builds it. It takes what the sweep took, and the key of the
slice:

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, carry={'soc_initial': 'soc'})
model = sps.build_slice('window.yaml', sources, axis, 888)
model.write('window-888.lp')
```

**A carried parameter is a source like any other.** The window above starts
from the `soc_initial` in `sources`, which is the seed of the sweep. Overwrite
it to start the window from another level:

```python
model = sps.build_slice('window.yaml', sources | {'soc_initial': 40.0}, axis, 888)
```

**`sweep=` starts the slice from what that sweep handed it.** Each carried
parameter is then what the slice before left in the sweep, so the model is the
one the sweep solved. A sweep that stopped is read back off its spill:

```python
stopped = sps.scan_sweep('runs/')
model = sps.build_slice('window.yaml', sources, axis, stopped.keys[-1], sweep=stopped)
```

| Rule | |
|---|---|
| **a key is where a slice starts** | The label of an `EachLabel` slice, or the first label of an `EachWindow` window. `sweep.keys` lists them. A key that names no slice is refused, and the message names the keys. |
| **`sweep=` reads the slice before** | The first slice has none, so it takes `sources` as given. So does every slice of an axis without a carry. |
| **a slice the sweep never reached is refused** | A carried sweep stops at the first slice that leaves nothing to carry. That slice builds. The slices after it were never built. |
| **the model comes back open** | Close it, or build it in a `with` block, as with [`build`](api.md#specsolve.build). |

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
| **`'previous'`, the default, starts each slice from the one before** | The first slice, a slice that loads the solver again, and every slice under an `executor` begin from nothing. |
| **a start is checked before a slice is built** | A slice its start leaves no row warns and starts from nothing. |

## Running slices in parallel

**`executor` is any
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects)**:
a `submit` that returns a `Future`.

| | Use it when | Notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | Sequential. |
| `ThreadPoolExecutor` | rarely | Slices contend with polars' own thread pool, and peak memory adds up. |
| `ProcessPoolExecutor` | local parallelism | **Must not use `fork`.** Sources cross as parquet. |
| anything remote | a cluster you already run | dask's `Client`, ray, loky. Paths travel as bytes; pass `shared_filesystem=True` if workers mount your filesystem. |

**A forked worker hangs** rather than raising.
[Running a sweep in parallel](../howto/parallel.md) is the recipe.

**Parallel is N × peak.** A four-way pool wants four times the memory of one
slice.

**Pass paths or tables, whichever you have.** A source no slice rewrote is
encoded once. `df.lazy()` pickles larger than the table.

## How a sweep runs

| | |
|---|---|
| **one model, updated per slice** | A serial sweep builds once and [updates](api.md#specsolve.types.Model.update); under `executor=` each slice builds its own. A [remote Gurobi](api.md#specsolve.types.Model.solve) opens one session per serial sweep, and a new one at each slice that loads again. |
| **the model is asked before it is sliced** | `EachWindow` refuses a window the model's coupling cannot bear. `EachLabel` is not asked. |
