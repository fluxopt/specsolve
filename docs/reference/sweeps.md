# Sweeps and rolling horizons

This page is the reference for `solve_over`: the axes it takes, the `Sweep` it
returns, and the `carry`, `executor` and `spill_to=` keywords.

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
is not asked whether it can be cut that way and `original_index=` is refused.

```python
sweep = sps.solve_over(
    'window.yaml',
    sources,
    sps.EachWindow('snapshot', steps=24, lookahead=24, into='t'),
    carry={'soc_initial': 'soc'},
)
sweep.primal('soc')  # (snapshot_start, t, value) — the window, and the index inside it
```

`into` has no default, and a seam's `where: "t == 0"` matches on it.

**`steps` as a sequence is one block per window**, which is a telescoping
horizon, or a month at a time with a few days of overlap:

```python
sps.EachWindow('snapshot', steps=[24, 24, 168, 168, 720], lookahead=12, into='t')
sps.EachWindow('snapshot', steps=days_in_each_month, lookahead=48, into='t')
```

The blocks are taken in order and laid end to end. A sequence that stops short
of the axis is refused. The labels past the last block would be solved by no
window, and the stitch would come back short:

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

Solved as a list, the slices key by `key_name=` and `original_index=` is
refused. Two axes compose as a comprehension over the slices of one, each
sliced again by the other.

**Sources cross a slice in every shape `build` takes.** A table carrying the
axis, table or parquet path, is filtered. A number, a `{label: value}` map or a
bare sequence passes through as it is. A table carrying the axis that is short
of a coordinate another table has raises an `SpecsolveWarning` before a slice is
taken, naming both tables. That slice builds the source empty. An absent row is
how a model masks, so the gap is reported rather than refused.

## Reading a sweep

**`Sweep` reads like [`Result`](api.md#specsolve.Result), one dimension wider.**
`primal`, `dual`, `evaluate`, `to_pandas`, `to_dataarray`, `to_dataset` and
`save` keep their names, and every table has the slice key prepended.

**You name the extra dimension, not the library.** `EachCoordinate('scenario')`
keys on `scenario`, so `sweep.to_dataarray('p')` is
`(scenario, snapshot, generator)`.

**`original_index=` asks for the answer over the real labels.** It is a
keyword on the readers, not a reader of its own:

```python
sweep.primal('soc')  # (snapshot_start, t, value) — keyed by slice
sweep.primal('soc', original_index=True)  # (snapshot, value) — the answer
sweep.dual('balance', original_index=True)  # the same, for a price
sweep.evaluate('spend', original_index=True)  # the model's own quantity, over real coordinates
```

For `EachWindow` this is the stitched answer over the global labels. Each
window contributes the labels its block owns, and the final window all of
its rows. For `EachCoordinate` nothing was re-indexed, and its key column
already is a label of the sliced dimension, so the table comes back unchanged.

**A hand-built axis refuses it.** A list of slices does not say what its keys
are labels of, so there is no dimension to read them back over:

```python
sweep = sps.solve_over('window.yaml', sources, windows, key_name='window')
sweep.primal('soc', original_index=True)
```

```text
SpecsolveError: a hand-built axis does not say what its keys are coordinates of,
so this sweep has no dimension to read 'window' back over. Read it keyed, which
is what its slices were solved over, or slice with EachWindow — it keys by where
each window started, records which coordinates each one owns, and stitches.
```

**Keyed is the default, because stitching is lossy.** It drops the lookahead
rows the sweep solved. For the same reason `to_dataset` and `save` have
no `original_index`.

**`original_index` sits beside `kind=` where a reader has one**, so
`sweep.to_dataarray('balance', 'dual', original_index=True)` is the stitched
price over time.

**`save` writes every kind.** `sweep.save('runs/')` writes what
`spill_to=` would have written, so the directory is a spilled sweep. The call
that made the sweep, pointed at it with `spill_to=`, reads it back without
solving; so do `sps.load_sweep('runs/')`, which reads every slice's frames in
and answers `primal`, and `sps.scan_sweep('runs/')`, which leaves them there for
`scan` ([`load_sweep`](api.md#specsolve.load_sweep)).

**There is no per-slice reader.** One slice is a partition of a table you
already hold: `sweep.primal('p').partition_by(sweep.key_name, as_dict=True)`.

| Rule | |
|---|---|
| **everything a slice produced is kept** | Every variable's primals and every constraint's duals come back through `sweep.primal(name)` and `sweep.dual(name)`. Each slice's *model* is released as the loop goes, so build peak stays at one slice. |
| **duals are keyed, never combined** | `sweep.dual(name)` has the shape of `sweep.primal(name)`; averaging, taking the last or reading one slice alone is yours to do. A slice whose model had an integer variable contributes no duals, and `sweep.record` says which slice. |
| **expressions are evaluated per slice** | Every declared `expressions:` name is evaluated at each slice's solution and read through `sweep.evaluate(name)`, and an expression the file never named through the same verb off a sweep archive. Under `original_index=True` only the rows each window owns survive, so summing the stitched table cannot double-count the lookahead. A quantity *reduced over* the sliced dimension is refused there, and the error names the per-slice read. |
| **no aggregate objective** | `sweep.record` is a table keyed by slice, the objective one of its columns. Scenarios are a distribution, not a sum, and summing window objectives double-counts the overlap. |
| **the lookahead is `t >= step`** | Overlapping windows return every row they solved, lookahead included. What each window owns is `sweep.primal('soc').filter(pl.col('t') < step)`. |
| **a slice that did not solve contributes no rows** | A `primal` table can be shorter than the sweep. `record` is always one row per slice and says which did not solve, holding null where a slice reached no objective. |
| **a window keys as `<dim>_start`** | `EachWindow('snapshot', …)` drops `snapshot` and re-indexes to `into`; the key column `snapshot_start` holds where each window began. |
| **a hand-built axis names its own key** | A plain list cannot say what its keys are labels *of*, so it must pass `key_name='draw'`. `key_name` overrides the derived name on any axis. It is refused when it collides with a column the tables already carry: a dimension the spec declares, `value`, or a column of `record` or `metrics`, such as `status`, `objective`, `solves` or `loads`. A name that starts with `specsolve_`, in any letter case, is refused too, as that prefix is reserved. |
| **`sweep.metrics` says what each slice took** | One [`Metrics`](api.md#specsolve.relational.parquet.Metrics) per slice, keyed like `record`. Each row is that slice's own share, so `solves` is `1`. [`Sweep.metrics`](api.md#specsolve.Sweep.metrics) says what `loads` means under a serial fold and under `executor=`. In an **archive** the table carries `specsolve_run` too, so a warehouse of them says which run a slice's cost belongs to. |
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
sweep.scan('soc')  # a LazyFrame: (snapshot_start, t, value), every window, in order
sweep.scan('balance', 'dual', original_index=True).collect()  # the same readers, the same keywords
```

| Rule | |
|---|---|
| **`scan` is the reader** | `sweep.scan(name, kind='primal')` returns `primal`, `dual` or `expression` as a `LazyFrame` over the files, `original_index=` included. On a held sweep it is the same reader made lazy. The frame readers and the exports refuse a spilled sweep and name `scan`. |
| **one file per slice and name** | `<kind>/<name>/<position>.parquet`, with the slice key a column of each, one type across every file a sweep writes. `record/` and `metrics/` hold the record, one row per slice, which names its slice in `slice_axis` and `slice`; `sweep.record` and `sweep.metrics` stay in memory. An **archive** holds those two as one file each, `record.parquet` and `metrics.parquet`. |
| **every file lands whole** | A file is written beside its final name and renamed into place. The record file is written last and marks a slice done, so a slice interrupted part way is solved again rather than read back short. |
| **an interrupted sweep resumes** | Run the same call at the same directory. A slice already there is read back, and under a `carry` its state is read off its file. Only the unfinished slices are built. |
| **a directory holds one sweep** | `sweep.json` records the key name and the keys, and `keys.parquet` holds the keys in their own type. A different sweep pointed at the directory is refused. Changed data or a changed spec is not detected, so delete the directory to solve again. |
| **the parent writes** | Under `executor=` a worker's answer crosses back to the parent, which writes it. |

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
| **one model, updated per slice** | A serial sweep builds once and [updates](api.md#specsolve.Model.update), and a slice whose structure matches the last keeps the loaded solver. A sweep under `executor=` builds per slice, because a built model does not cross a process. The loaded solver holds Gurobi's environment, so a [remote Gurobi](api.md#specsolve.Model.solve), such as Instant Cloud, opens one session for a serial sweep unless `keep='nothing'`. It opens a new one at each slice where `sweep.metrics` counts a load, and at every slice under `executor=`. |
| **`keep=` reaches every slice, and the fold chooses none of them** | It defaults to `'solver'`, as [`solve`](api.md#specsolve.Model.solve) does. `keep='progress'` has something to carry, since consecutive slices differ by one step; whether that pays is a fact about the *model*. Under `executor=` every slice is a first solve and keeps `'nothing'`. |
| **the model is asked before it is sliced** | The plan says what each axis can bear. [`EachWindow`](api.md#specsolve.EachWindow) says what a window refuses, and reads an offset the data decides off the data. `EachCoordinate` is not asked: the spec must not declare the column it slices, so the model never sees the axis. |
| **the spec is parsed once** | `solve_over` validates it up front, so a spec outside the language fails before the data is touched. Every worker is handed the lowered [program](glossary.md#the-chain), in this process or across one, and none reads the YAML or lowers it again. |
| **a slice is total** | A slice says what the *whole* model attaches, not what changed since the one before it. The class axes always do; a hand-built list has to keep the rule. |
