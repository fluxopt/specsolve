# Archiving a solve

How to keep a solve: the spec, the data it was solved with, and what came
back, so you can read the answer later, ask the question again, or hand both
to someone else. The reference is
[`load_archive`](../reference/api.md#specsolve.load_archive) and the entries
beside it.

## Archive as you solve

```python
import specsolve as sps

sps.solve('dispatch.yaml', sources, archive='case/')
```

That writes `spec.yaml`, one `sources/<key>.parquet` per key the file
declares, `catalog.parquet` saying what each file holds, and `answer/` holding
everything the solve produced. Each source is held as the table the solve read,
which [`sps.tidy`](../reference/api.md#specsolve.tidy) returns, with
`specsolve_run` added: a dimension as its labels and `specsolve_position`, a
parameter as its dims and `value`, a relation as its columns:

```text
case/
    spec.yaml
    sources/cost.parquet
    sources/load.parquet
    …
    sources.parquet               (specsolve_run, source, digest) — what each of them is
    catalog.parquet               (specsolve_run, path, …, column, dim) — what each file holds and over which dimensions
    answer/record.parquet      how it terminated, what it reached, when, under what name, and on what
    answer/metrics.parquet        what the build and its solves took
    answer/primal/p.parquet       one file per variable
    answer/dual/power_balance.parquet
```

**The suffix picks the container.** Anything without a `.zip` suffix is a
directory, as above. `.zip` packs the same members into one file:

```python
sps.solve('dispatch.yaml', sources, archive='case.zip')
```

**`sps.solve`, `model.solve` and `sps.solve_over` take `archive=`.** Nothing
else writes one.

**An answer holds what you ask for.** Every answer holds the primal, the duals
and the declared expressions. Anything else is an output, and only a solve that
asks for it holds it, live, saved and archived alike:

```python
sps.solve('dispatch.yaml', sources, outputs={'activity'}, archive='case')
```

This adds `answer/activity/<name>.parquet`, each constraint's left-hand side.
A reader of an output the solve did not ask for refuses, and the message names
the `outputs=` to solve with.

**The record says what produced the answer.** `answer/record.parquet` names
the solver, its version and the options it ran with, and the specsolve and
mathspec versions that built the model. So you can install the same
environment again. The options are one JSON string with sorted keys. An
option that changes the answer, such as a time limit, a gap, a seed or a
tolerance, keeps its value. A value that JSON has no number for, such as an
infinite time limit, is the string `"inf"`, `"-inf"` or `"nan"`. Any other
option keeps its name and has the value `<not recorded>`. An archive often
goes to shared storage, and a Gurobi licence credential such as `WLSSecret` is
passed as an option.
`result.provenance` gives the same five fields without an archive.

To keep the value of an option that is not on the solver's list, name it.
Name no credential:

```python
sps.solve('dispatch.yaml', sources, solver_options={'mip_max_nodes': 1000}, record_options=['mip_max_nodes'])
```

## Read it back

```python
case = sps.load_archive('case/')

case.result.objective  # what it reached
case.result.primal('p')  # the values it came back with
sps.solve(case.spec, case.sources)  # the same question, asked again
```

**`load_archive` reads it whole.** The sources come back as tables, the
answer's frames are in memory, and nothing has to be kept alive afterwards. A
zip unpacks to a scratch directory that is gone when the call returns:

```python
case = sps.load_archive('case.zip')
```

Pass `into=` when you want the extracted tree as well, to query it with an
engine that reads parquet. A directory archive is read where it lies and
refuses an `into=`.

## Read one too big to hold

**`scan_archive` reads nothing until asked.** The sources come back as paths,
and each frame is read off disk at the call that asks for it. A source file
also holds `specsolve_run`, which a build ignores:

```python
archived = sps.scan_archive('sweep.zip', 'sweep/')
archived.sweep.scan('p')  # read at the collect, one name at a time
```

Scan the archive that does not fit in memory, and the one you will read
little of: a load reads every name, a scan only the ones you ask for. What is
scanned has to outlive what it reads off, so a zip needs an `into=` you will
keep.

| | `load_archive` | `scan_archive` |
|---|---|---|
| a source | the table the member holds | the path to it |
| the answer | frames in memory | read at the call that asks |
| a sweep's answer | held: `sweep.primal('p')` | spilled: `sweep.scan('p')` |
| a zip's `into=` | optional, and scratch without one | required, and kept |

## Read what a solve cost

`metrics` is a `Metrics`: how big the model was, how many solves the clocks
cover, and wall-clock seconds in each phase, as one value
([the attributes](../reference/api.md#specsolve.types.Metrics)).

```python
case = sps.load_archive('case/')

case.metrics.rows  # how big the model was
case.metrics.solves  # how many solves the clocks cover
case.metrics.build_seconds  # turning declarations into frames
```

**The row covers the model's whole life, and `solves` says how long that is.**
`sps.solve` builds the model it solves, so its archive reads `solves` of 1. A
model solved more than once before it was archived carries the sum:

```python
with sps.build('dispatch.yaml', sources) as model:
    model.solve()
    model.solve(archive='second/')  # solves: 2, and the clocks cover both
```

A sweep records each slice's own metrics as `sweep.metrics` instead
([sweeps](../reference/sweeps.md)).

## Keep the answer an update produced

An updated model answers the merged data, so archive it where it is
answered:

```python
with sps.build('dispatch.yaml', sources) as model:
    model.update({'p_max': doubled}).solve(archive='case/')
```

## Archive a sweep

`archive=` on `solve_over` writes the sweep's answer at the paths a single
solve uses, one file per name. A rolling horizon's `answer/primal/soc.parquet`
is over `snapshot`, as `sweep.primal('soc')` returns it. Each file carries
`specsolve_run`, as a single solve's does, and the readers drop it:

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t')
sps.solve_over('window.yaml', sources, axis, carry={'soc_initial': 'soc'}, archive='roll/')
```

The archive carries the axis and the carry, so the sweep runs again from the
file alone:

```python
archived = sps.load_archive('roll/')

archived.sweep.primal('soc')  # (snapshot, value), off answer/primal/soc.parquet
sps.solve_over(archived.spec, archived.sources, archived.axis, carry=archived.carry)
```

A source the axis cuts is held uncut, the axis column first. A parameter
given as one number over a window's local index is held as a table over the
axis, because a window of each length reads the number over labels of its own.
Each slice cuts from the archive the tables it attached.

**Keep the windows where you will read them per window.** `keep_windows=True`
also writes each window's frames, lookahead rows included, under
`answer/windows/`, with `owned.parquet` for what each window owns and
`catalog.parquet` for what each of these files holds. Then `per_window=True`
reads off the archive:

```python
sps.solve_over('window.yaml', sources, axis, carry={'soc_initial': 'soc'}, archive='roll/', keep_windows=True)
sps.load_archive('roll/').sweep.primal('soc', per_window=True)  # (snapshot_start, t, value)
```

Without them, `per_window=True` and an expression the file never named are
refused. Solve again from the archive to get them back.

## Archive a sweep too large to hold

`spill_to=` writes each slice's frames as the fold goes, so the sweep holds
one slice at a time. `archive=` packs the whole sweep. Pass both and the
archive reads its answer off the spill, so the sweep is archived without ever
being held:

```python
axis = sps.EachCoordinate('scenario')
sps.solve_over('dispatch.yaml', sources, axis, spill_to='work/', archive='sweep/')
```

```python
archived = sps.scan_archive('sweep/')

archived.sweep.scan('p')  # keyed by scenario, read at the collect
```

`scan_archive` leaves the answer on disk, and `scan` reads it. `load_archive`
reads it into memory, where it fits, and `sweep.primal('p')` answers on that
one.

## Read a directory of them

A directory of archives is a table per glob, and every row of every table
carries `specsolve_run`, the archive's own name:

```python
import polars as pl

pl.read_parquet('runs/*/answer/record.parquet').sort('solved_at')
```

The recipes are [reading a directory of runs](warehouse.md).

## What an archive will not take

**A sweep cut by a hand-built axis.** A list of `(key, sources)` is a set of
sources per slice, and the call is refused before the first slice is solved:

```text
archive= takes a sweep cut by EachCoordinate or EachWindow, which say how one
set of sources was cut and so how the archive can be re-run. A hand-built list
is a set of sources per slice, which are unrelated questions — archive one
solve each.
```

**A directory that already holds something.** A directory archive is written
whole, never merged into what is there. A `.zip` target is replaced.
