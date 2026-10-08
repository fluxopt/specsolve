# Sweeps and rolling horizons

This page is the reference for `solve_over`: the axes it takes, the `Sweep` it
returns, and the `carry`, `executor`, `spill_to=` and `archive=` keywords.

`solve_over` runs one [model](glossary.md#the-chain) once per slice and folds
the answers together. A slice is one set of [sources](glossary.md#how-it-runs)
under one key. Scenarios, rolling horizons and myopic pathways are all the same
fold.

```python
import specsolve as sps

sweep = sps.solve_over('spec.yaml', sources, sps.EachCoordinate('scenario'))
sweep.record  # one Record per slice, the scenario column first
sweep.primal('p')  # (scenario, snapshot, generator, value)
```

## The axes

An axis says how the sources split into slices. `solve_over` accepts three:
[`EachCoordinate`](api.md#specsolve.EachCoordinate), one slice per label of a
dimension; [`EachWindow`](api.md#specsolve.EachWindow), one slice per window of
consecutive labels; and a sequence of `(key, sources)` pairs written by hand. A
hand-built axis must pass `key_name=`. A list names no dimension, so the model
is not asked whether it can be cut that way, and its answer is keyed by slice.

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'),
    carry={'soc_initial': 'soc'},
)
sweep.primal('soc')  # (snapshot, value) — the answer over the real labels
```

`into` has no default, and a seam's `where: "t == 0"` matches on it.

**A datetime axis is held in microseconds**, as
[a label is](data.md#where-coordinates-come-from), and a key finer than a
microsecond is refused.

**`steps` as a sequence is one block per window**, which is a telescoping
horizon, or a month at a time with a few days of overlap:

```python
sps.EachWindow('snapshot', steps=[24, 24, 168, 168, 720], lookahead=12, into='t')
sps.EachWindow('snapshot', steps=days_in_each_month, lookahead=48, into='t')
```

The blocks are taken in order and laid end to end. A sequence that stops short
of the axis is refused. The labels past the last block would be solved by no
window, and the answer would come back short:

```text
DataError: steps keeps 7 coordinate(s) across 2 window(s), and 'snapshot' has 12 —
the last 5 would be solved by no window. List a block for them, or pass an int to
repeat one size to the end.
```

A sequence reaching past the end is not: its trailing blocks simply have
nothing to cover.

**`lookahead` is one number whatever the blocks.** What a model reads ahead is
one integer for the dimension, so no block size enters the check. The same
`lookahead` satisfies uniform windows and unequal ones.

**`axis.slices(sources)` is the list the axis would run**, as the
`(key, sources)` pairs a hand-built axis takes:

```python
slices = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t').slices(sources)
sps.build('window.yaml', slices[37][1]).write('window-37.lp')  # the one that was infeasible
```

Solved as a list, the slices key by `key_name=`, and the answer is keyed by
slice. Two axes compose as a comprehension over the slices of one, each
sliced again by the other.

**Sources cross a slice in every shape `build` takes.** A parameter or a
relation whose table carries the axis is filtered, as a table or as a parquet
path. Every other source passes through as it is. **An index of another
dimension that carries the axis is refused** before a slice is taken, because
an index lists the labels that every slice has. Say which labels a slice has in
a parameter or a relation over the dimension and the axis, where a missing row
reads as absent. A table carrying the axis that is short of a coordinate
another table has raises a `SpecsolveWarning` before a slice is taken, naming
both tables. That slice builds the source empty. An absent row is
how a model masks, so the gap is reported rather than refused.

## Reading a sweep

**Every reader returns the answer.** `primal`, `dual`, `evaluate`, `activity`,
`reduced_cost`, `slack`, `variable_basis`, `constraint_basis`, `scan`,
`to_pandas`, `to_dataarray` and `to_dataset`
keep the names and shapes of [`Result`](api.md#specsolve.types.Result). For `EachWindow` the answer is over the
real labels of the sliced dimension. Each label comes from the window that owns
it, and the final window gives all of its rows. For `EachCoordinate` and a
hand-built axis, each slice is a whole answer, so the table is keyed by slice.

```python
sweep.primal('soc')  # (snapshot, value) — the answer over the real labels
sweep.dual('balance')  # the same, for a price
sweep.evaluate('spend')  # the model's own quantity, over the real labels
```

**A sweep holds the outputs you ask for, as a solve does.**
`sps.solve_over(..., outputs={'activity', 'reduced_cost'})` gives every slice
its activity and its reduced costs, in memory, in the spill and in the
archive. A slice with no duals has no reduced costs, and the reader says why.
A sweep that did not ask refuses the reader. A `spill_to=` directory solved
with other outputs is refused rather than resumed, because its slices hold only
what they were solved with.

**You name the extra dimension, not the library.** `EachCoordinate('scenario')`
keys on `scenario`, so `sweep.to_dataarray('p')` is
`(scenario, snapshot, generator)`.

**`per_window=True` reads each window as it was solved.** It is a keyword on
every reader, beside `kind=` where a reader has one. The table is keyed by
where each window started and is over the index inside the window. It keeps the
lookahead rows:

```python
sweep.primal('soc', per_window=True)  # (snapshot_start, t, value) — every row every window solved
sweep.to_dataarray('balance', 'dual', per_window=True)  # (snapshot_start, t)
```

**A sweep that was not cut into windows refuses `per_window=True`.** Its answer
already is one table per slice:

```text
SpecsolveError: per_window=True reads an EachWindow sweep one window at a time,
and this sweep was not cut into windows: its answer already is one frame per
slice, keyed by 'scenario'. Read it without per_window.
```

**A quantity that is not over the windowed dimension has no answer.** Over a
window, `sum(p * cost)` is one number, so no label of `snapshot` owns it. The
reader refuses it and names the read that works:

```text
SpecsolveError: this has no answer over 'snapshot': the frame has no 't' column,
because the quantity is not over the windowed dimension — each row covers a whole
window, lookahead included under an overlapping window. Read it with
per_window=True for the value of each window, or read a quantity that keeps 't'
and aggregate its answer.
```

**`to_dataset()` with no names reads every name that has an answer.** It
leaves out a quantity that is not over the windowed dimension, as an archive
leaves out its file. A name you give is read or refused as above. With
`per_window=True`, every name is read.

**`save` writes every kind, per window.** `sweep.save('runs/')` writes what
`spill_to=` would have written, so the directory is a spilled sweep. The call
that made the sweep, pointed at it with `spill_to=`, reads it back without
solving; so do `sps.load_sweep('runs/')`, which reads every slice's frames in
and answers `primal`, and `sps.scan_sweep('runs/')`, which leaves them there for
`scan` ([`load_sweep`](api.md#specsolve.load_sweep)).

**There is no per-slice reader.** One slice is a partition of a table you
already hold: `sweep.primal('p').partition_by(sweep.key_name, as_dict=True)`,
with `per_window=True` for a windowed sweep.

| Rule | |
|---|---|
| **everything a slice produced is kept** | Every variable's primals and every constraint's duals come back through `sweep.primal(name)` and `sweep.dual(name)`. Each slice's *model* is released as the loop goes, so build peak stays at one slice. |
| **duals are keyed, never combined** | `sweep.dual(name)` has the shape of `sweep.primal(name)`; averaging, taking the last or reading one slice alone is yours to do. A slice whose model had an integer variable contributes no duals, and `sweep.record` says which slice. |
| **expressions are evaluated per slice** | Every declared `expressions:` name is evaluated at each slice's solution and read through `sweep.evaluate(name)`, and an expression the file never named through the same verb off a sweep archive. In the answer of a windowed sweep only the rows each window owns survive, so summing it cannot double-count the lookahead. A quantity *reduced over* the windowed dimension has no answer, and the error names `per_window=True`. |
| **no aggregate objective** | `sweep.record` is a table keyed by slice, the objective one of its columns. Scenarios are a distribution, not a sum, and summing window objectives double-counts the overlap. |
| **the lookahead is `t >= step`** | Per window, overlapping windows return every row they solved, lookahead included. What each window owns is `sweep.primal('soc', per_window=True).filter(pl.col('t') < step)`. |
| **a slice that did not solve contributes no rows** | A `primal` table can be shorter than the sweep. `record` is always one row per slice and says which did not solve, holding null where a slice reached no objective. |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` drops `snapshot` and re-indexes to `into`. Per window, the key column `snapshot_start` holds where each window began. |
| **a hand-built axis names its own key** | A plain list cannot say what its keys are labels *of*, so it must pass `key_name='draw'`. `key_name` overrides the derived name on any axis. It is refused when it collides with a column the tables already carry: a dimension the spec declares, `value`, or a column of `record` or `metrics`, such as `status`, `objective`, `solves` or `loads`. A name that starts with `specsolve_`, in any letter case, is refused too, as that prefix is reserved. |
| **`sweep.metrics` says what each slice took** | One [`Metrics`](api.md#specsolve.types.Metrics) per slice, keyed like `record`. Each row is that slice's own share, so `solves` is `1`. [`Sweep.metrics`](api.md#specsolve.types.Sweep.metrics) says what `loads` means under a serial fold and under `executor=`. In an **archive** the table carries `specsolve_run` too, so a warehouse of them says which run a slice's cost belongs to. |
| **on disk, the slice is two text columns** | `record` and `metrics` in memory start with the key column, in the key's own type, so they join to the frames. On disk they do not: `slice_axis` holds the key name and `slice` the key as text, and both are null for a single solve. So every `record.parquet` and `metrics.parquet` has the same columns, from a solve or from any sweep. A key therefore names one slice by its text: two keys of one text, such as a repeated key, are refused before a slice is taken, and so is a key that the sweep's one key type would rewrite, such as `True` among integers. |
| **a slice that fails says which slice** | The error is the engine's own, with a note on it: `in slice 'bad' (3 of 3)`. |
| **a sweep's memory grows with its answer, unless it is spilled** | The models are released as the fold goes; the tables accumulate. `spill_to=` writes them out instead ([below](#spilling-a-sweep-to-disk)), and `save` writes a held sweep out the same way, after the fact. |

## Spilling a sweep to disk

`spill_to=` names a directory. Each slice's tables are written there as the fold
goes rather than held, so the sweep's memory stays at one slice:

```python
sweep = sps.solve_over(
    'window.yaml', sources, sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'), spill_to='runs/'
)
sweep.scan('soc')  # a LazyFrame: (snapshot, value), the answer
sweep.scan('balance', 'dual', per_window=True).collect()  # the same readers, the same keywords
```

| Rule | |
|---|---|
| **`scan` reads lazily** | `sweep.scan(name, kind='primal')` returns `primal`, `dual` or `expression` as a `LazyFrame` over the files, `per_window=` included. The answer of a windowed sweep is stitched lazily, at the collect. The frame readers and the exports read the one name they are asked for into memory, as `scan(...).collect()` would. |
| **one file per slice and name** | `<kind>/<name>/<position>.parquet`, per window, with the slice key a column of each, one type across every file a sweep writes. `record/` and `metrics/` hold the record, one row per slice, which names its slice in `slice_axis` and `slice`; `sweep.record` and `sweep.metrics` stay in memory. An **archive** holds those two as one file each, `record.parquet` and `metrics.parquet`. |
| **every file lands whole** | A file is written beside its final name and renamed into place. The record file is written last and marks a slice done, so a slice interrupted part way is solved again rather than read back short. |
| **an interrupted sweep resumes** | Run the same call at the same directory. A slice already there is read back, and under a `carry` its state is read off its file. Only the unfinished slices are built. |
| **a directory holds one sweep** | `sweep.json` records the key name and the keys, and `keys.parquet` holds the keys in their own type. A different sweep pointed at the directory is refused. Changed data or a changed spec is not detected, so delete the directory to solve again. |
| **the parent writes** | Under `executor=` a worker's answer crosses back to the parent, which writes it. |

## Archiving a sweep

`archive=` writes the model, the sources the sweep was cut from, the axis and
the answer, so the sweep runs again from the file alone
([archiving](../howto/archiving.md)):

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
| **an archive holds the answer** | One file per name at `answer/<kind>/<name>.parquet`, the path the archive of a single solve uses. Each holds what the reader returns, and `specsolve_run`, which every file of an archive carries, the windows included. `sps.load_archive` and `sps.scan_archive` read the files back, and the readers return them without `specsolve_run`. |
| **a name with no answer is left out with its reason** | A quantity that is not over the windowed dimension has no file. `answer/reasons.parquet` holds the reason, and the reader raises it. |
| **`keep_windows=True` keeps the windows too** | An `EachWindow` sweep also writes `answer/windows/<kind>/<name>/<position>.parquet`, with `owned.parquet` and a `catalog.parquet` of their own beside them, so `per_window=True` reads off the archive. `answer/sweep.json` records that the windows were kept, so a sweep in which no window wrote a frame reads as the live sweep does. |
| **without the windows, a per-window read is refused** | So is an expression the file never named, which is valued at the solution of each window. The error names the way back. |
| **`keep_windows=True` needs windows and an archive** | On another axis, or without `archive=`, it is refused before a slice is solved. |

```text
SpecsolveError: this archive holds the answer only, because it was written without
keep_windows=True, so it has no per-window frames to read. Solving again from the
archived spec and sources restores them: load_archive gives both, with the axis
and the carry, so sps.solve_over(archive.spec, archive.sources, archive.axis,
carry=archive.carry) runs the sweep again.
```

## Carrying state between slices

`carry` copies one slice's answer into the next slice's data, as a mapping
`{parameter: variable}`.

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'),
    carry={'soc_initial': 'soc'},
)
```

**You name no coordinate.** The two declarations say which dimension is
collapsed. The row handed on is the last one the slice owns: label 23 of a window
keeping 24, not label 47 of the 48 it solved.

| Rule | |
|---|---|
| **a carry is a copy, never arithmetic** | Accumulation (`existing += built`) is a derived variable in the YAML. |
| **the two declarations say what is copied** | The carry collapses the one dimension the *variable* has and the *parameter* does not. Every other dimension rides along. `soc` over `(t, storage)` into `soc_initial` over `(storage)` drops `t` and hands both stores forward. `total` over `(generator)` into `existing` over `(generator)` drops nothing, so the whole frame moves. |
| **the collapsed dimension has to be the one the axis advances along** | It is `EachWindow`'s `into`. Any other dimension has no last-owned row to read, so the carry is refused and the error names the YAML: reduce that dimension in a derived variable, where the typesetter prints it and the oracle checks it. A carry under `EachCoordinate` or a hand-built axis therefore collapses nothing. |
| **a carried value is a boundary condition, never a pin** | The parameter supplies the state *entering* the window, as `soc == soc_initial + charge * 0.9 - discharge` does at `t == 0`. Writing `soc == soc_initial` there replaces the first row's dynamics instead of seeding them, which leaves its `charge` and `discharge` tied to nothing — the window gets free energy at every seam, and the sweep comes out cheaper than full foresight. |
| **the first slice needs a seed** | `carry` supplies the parameter from the second slice on. The first slice takes it from `sources`, and a sweep whose sources lack it is refused before a slice is taken. |
| **a carry is checked before anything is read** | The dims come from the YAML and the axis is an argument, so a carry that cannot line up raises before the axis has scanned a source: collapsing two dimensions at once, a parameter over more dimensions than the variable, a dimension the axis does not advance along, no seed. `check` cannot answer this, because `carry` is an argument to the call, not part of the spec. |
| **the last slice carries nothing** | There is no next slice to read it. |
| **a slice that leaves nothing to carry stops the sweep** | An infeasible window has no level to hand forward. The error names the slice, how it terminated, and the slice left waiting. A sweep without a carry records the slice in `record` and goes on. |
| **`carry` excludes `executor`** | A carried value makes slice *i+1* depend on slice *i*, so the call is refused. |

## Starting slices

`start=` takes what [`solve`](api.md#specsolve.types.Model.solve) takes, an
earlier sweep, or the word `'previous'`. **A start is cut by the axis, as a
source is**: a table that carries the sliced dimension gives each slice its own
rows, and a table without it reaches every slice whole.

```python
again = sps.solve_over('dispatch.yaml', new_sources, axis, start=earlier)
chained = sps.solve_over('dispatch.yaml', sources, axis, start='previous')
```

| Rule | |
|---|---|
| **a table over the sliced dimension gives each slice its rows** | `EachCoordinate('scenario')` gives each slice the rows of its scenario. `EachWindow` gives each window the rows of the coordinates it covers, lookahead included, over its local index. A hand-built axis cuts on its key column. |
| **an earlier sweep is its answer** | Each slice starts from the earlier slice of its key, and each window from the answer over the coordinates it covers. An archive written without `keep_windows=True` starts a sweep too. |
| **`'previous'` starts each slice from the one before it** | As [`solve`](api.md#specsolve.types.Model.solve) does: the solver carries on where the update kept it, and elsewhere the answer is matched by the slice model's own coordinates, so a window takes the window before it by local index. Each slice is solved with its basis for the next to start from, whether or not `outputs=` asks for it; the sweep keeps only what `outputs=` asks for. The first slice, and a slice after one that left no values, starts cold. Each slice depends on the one before, so `'previous'` under an `executor` is refused. |
| **a start is checked before a slice is built** | A table over an `EachWindow` sweep's local index alone is refused: it says nothing about which coordinates it means. So is a start that leaves a slice no row, and a table `solve` refuses. |
| **a start reaches every executor** | Each slice's cut is taken before the slice is sent, so a slice solved in another process takes it as data. |

## Running slices in parallel

`executor` is any
[`concurrent.futures.Executor`](https://docs.python.org/3/library/concurrent.futures.html#executor-objects):
a `submit` that returns a `Future`, and nothing else. The package ships no
remote transport and no vendor integration.

| | Use it when | Notes |
|---|---|---|
| `None` *(default)* | always, until measurement says otherwise | Sequential; nothing is serialised. |
| `ThreadPoolExecutor` | rarely | Sources are **not** encoded. Slices contend with polars' own thread pool, and peak memory is additive rather than per-worker. |
| `ProcessPoolExecutor` | genuine local parallelism | **Must not use `fork`**; see below. Sources cross as parquet. |
| anything remote | a cluster you already run | dask's `Client`, ray's wrappers, loky. Workers are assumed not to share your filesystem, so paths travel as bytes; pass `workers_share_fs=True` if they do mount it. |

**A forked worker hangs.** The polars thread pool does not survive `fork`, and
the failure is a hang rather than an error. `solve_over` cannot enforce the
start method, because a remote executor has none to inspect.
[Running a sweep in parallel](../howto/parallel.md) is the recipe.

**Parallel is N × peak.** Each worker holds its own slice's model, so a
four-way pool wants four times the memory of one slice.

**Pass paths or tables, whichever you already have.** Sources cross a process
boundary as parquet, never as pickled tables. A path the workers can reach
stays a path; one they cannot reach travels as its own bytes. A source no slice
rewrote is encoded once for the whole sweep. `df.lazy()` is not an
optimisation: an eager table is embedded in the plan, so it pickles *larger*
than the table. Only `scan_parquet` is a reference.

## How a sweep runs

| | |
|---|---|
| **a partition is a filter on the sources** | Not a narrower index: the containment check refuses parameter rows outside the declared coordinates, so the axis rewrites the rows and the index they are over in one mapping. |
| **one model, updated per slice** | A serial sweep builds once and [updates](api.md#specsolve.types.Model.update), and a slice whose structure matches the last keeps the loaded solver. A sweep under `executor=` builds per slice, because a built model does not cross a process. The loaded solver holds Gurobi's environment, so a [remote Gurobi](api.md#specsolve.types.Model.solve), such as Instant Cloud, opens one session for a serial sweep. It opens a new one at each slice where `sweep.metrics` counts a load, and at every slice under `executor=`. |
| **each slice begins from nothing, unless `start=` says otherwise** | As [`solve`](api.md#specsolve.types.Model.solve) does. `start='previous'` has something to carry, since consecutive slices differ by one step; whether that pays is a fact about the *model*. Under `executor=` every slice is a first solve. |
| **the model is asked before it is sliced** | The plan says what each axis can bear. [`EachWindow`](api.md#specsolve.EachWindow) says what a window refuses, and reads an offset the data decides off the data. `EachCoordinate` is not asked: the spec must not declare the column it slices, so the model never sees the axis. |
| **the spec is parsed once** | `solve_over` validates it up front, so a spec outside the language fails before the data is touched. Every worker is handed the lowered [program](glossary.md#the-chain), in this process or across one, and none reads the YAML or lowers it again. |
| **a slice is total** | A slice says what the *whole* model attaches, not what changed since the one before it. The class axes always do; a hand-built list has to keep the rule. |
