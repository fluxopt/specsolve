# Archiving a solve

How to keep the spec, the data and the answer of a solve, to read or solve
again later. The reference is
[`read_archive`](../reference/api.md#specsolve.read_archive).

## Archive as you solve

`sps.solve`, `model.solve` and `sps.solve_over` take `archive=`:

```python
import specsolve as sps

sps.solve('dispatch.yaml', sources, archive='case/')
```

A path without a `.zip` suffix is a directory:

```text
case/
    spec.yaml
    sources/cost.parquet          each source, as the solve read it
    catalog.parquet               what each file holds
    answer/record.parquet         how it ended, what it reached, and on which solver and versions
    answer/metrics.parquet        how long each phase took, from the spec to the archive
    answer/primal/p.parquet       one file per variable
    answer/dual/power_balance.parquet
```

**An answer holds the primal, the duals and the declared expressions.**
Anything else is held only when `outputs=` asks for it.

**The record keeps the solver options that change the answer**, such as a time
limit or a gap. Any other option is `<not recorded>` unless you name it in
`also_record=`.

## Read it back

```python
case = sps.read_archive('case/')

case.result.objective  # what it reached
case.result.primal('p')  # the values it came back with
sps.solve(case.spec, case.sources)  # the same question, asked again
```

**`read_inputs` reads the spec and the data, and never the answer.** It still
reads an archive whose answer layout `read_archive` refuses.

## Read one too big to hold

**`scan_archive` reads nothing until asked:**

```python
archived = sps.scan_archive('sweep.zip', 'sweep/')
archived.sweep.scan('p')  # read at the collect, one name at a time
```

| | `read_archive` | `scan_archive` |
|---|---|---|
| a source | the table the member holds | the path to it |
| the answer | tables in memory | read at the call that asks |
| a sweep's answer | in memory: `sweep.primal('p')` | on disk: `sweep.scan('p')` |
| a zip's `into=` | optional, and scratch without one | required, and kept |

## Archive a sweep

`archive=` on `solve_over` also keeps the axis, carry included, so the sweep
runs again from the archive alone:

```python
axis = sps.EachWindow('snapshot', horizon=24, lookahead=24, carry={'soc_initial': 'soc'})
sps.solve_over('window.yaml', sources, axis, archive='roll/')

archived = sps.read_archive('roll/')
sps.solve_over(archived.spec, archived.sources, archived.axis)
```

A sweep's `answer/` holds one file per slice of each name, so
`primal(..., per_window=True)` reads off the archive too. **A directory archive
is written as the sweep goes**, one slice at a time, so the sweep holds one
slice in memory. Run the same call at the same directory again and the slices
already there are read back rather than solved, which is how a sweep that
stopped resumes. A `.zip` is packed once the sweep ends, and is not resumed.

## What an archive will not take

- **A directory that already holds something**, unless it is the sweep
  archive that sweep resumes. A `.zip` target is replaced.

A directory of archives reads as one set of tables:
[reading a directory of runs](warehouse.md).
