# Archiving a solve

How to keep the spec, the data and the answer of a solve, to read or solve
again later. The reference is
[`load_archive`](../reference/api.md#specsolve.load_archive).

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
    answer/metrics.parquet        what the build and its solves took
    answer/primal/p.parquet       one file per variable
    answer/dual/power_balance.parquet
```

**An answer holds the primal, the duals and the declared expressions.**
Anything else is held only when `outputs=` asks for it.

**The record keeps the solver options that change the answer**, such as a time
limit or a gap. Any other option is `<not recorded>` unless you name it in
`record_options=`.

## Read it back

```python
case = sps.load_archive('case/')

case.result.objective  # what it reached
case.result.primal('p')  # the values it came back with
sps.solve(case.spec, case.sources)  # the same question, asked again
```

**`load_inputs` reads the spec and the data, and never the answer.** It still
reads an archive whose answer layout `load_archive` refuses.

## Read one too big to hold

**`scan_archive` reads nothing until asked:**

```python
archived = sps.scan_archive('sweep.zip', 'sweep/')
archived.sweep.scan('p')  # read at the collect, one name at a time
```

| | `load_archive` | `scan_archive` |
|---|---|---|
| a source | the table the member holds | the path to it |
| the answer | frames in memory | read at the call that asks |
| a sweep's answer | held: `sweep.primal('p')` | spilled: `sweep.scan('p')` |
| a zip's `into=` | optional, and scratch without one | required, and kept |

## Archive a sweep

`archive=` on `solve_over` also keeps the axis and the carry, so the sweep
runs again from the archive alone:

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t')
sps.solve_over('window.yaml', sources, axis, carry={'soc_initial': 'soc'}, archive='roll/')

archived = sps.load_archive('roll/')
sps.solve_over(archived.spec, archived.sources, archived.axis, carry=archived.carry)
```

`keep_windows=True` also writes each window's own frames, which
`primal(..., per_window=True)` reads.

## What an archive will not take

- **A sweep cut by a hand-built axis.** Archive one solve each.
- **A directory that already holds something.** A `.zip` target is replaced.

A directory of archives reads as one set of tables:
[reading a directory of runs](warehouse.md).
