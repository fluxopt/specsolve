# Archiving a solve

How to keep a solve: the spec, the data it was solved with, and what came
back, so you can read the answer later, ask the question again, or hand both
to someone else. The reference is
[`load_archive`](../reference/api.md#specsolve.load_archive) and the entries
beside it.

## Archive as you solve

`sps.solve`, `model.solve` and `sps.solve_over` take `archive=`:

```python
import specsolve as sps

sps.solve('dispatch.yaml', sources, archive='case/')
```

A path without a `.zip` suffix is a directory. `.zip` packs the same files
into one:

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
Anything else is an output, held only when the solve asks for it, such as
`outputs={'activity'}`, `'reduced_cost'`, `'slack'` or `'basis'`.

**The record keeps the solver options that change the answer**, such as a time
limit, a gap, a seed or a tolerance. Any other option keeps its name and the
value `<not recorded>`, because an archive often goes to shared storage and a
licence credential can be an option. Name an option in `record_options=` to
keep its value.

## Read it back

```python
case = sps.load_archive('case/')

case.result.objective  # what it reached
case.result.primal('p')  # the values it came back with
sps.solve(case.spec, case.sources)  # the same question, asked again
```

`load_archive` reads it whole, a zip included. Pass `into=` to keep the
unpacked files, to query them with an engine that reads parquet. Nothing
checks that the files are still the ones the solve wrote.

**`load_inputs` reads the spec and the data, and never the answer.** A release
that changes how an answer is laid out makes `load_archive` refuse an older
archive, and `load_inputs` still reads it, so you can solve it again.

## Read one too big to hold

**`scan_archive` reads nothing until asked.** The sources come back as paths,
and each frame is read at the call that asks for it:

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

`archive=` on `solve_over` writes the sweep's answer at the paths a single
solve uses, one file per name, with the axis and the carry it ran with. So the
sweep runs again from the archive alone:

```python
axis = sps.EachWindow('snapshot', steps=24, lookahead=24, into='t')
sps.solve_over('window.yaml', sources, axis, carry={'soc_initial': 'soc'}, archive='roll/')

archived = sps.load_archive('roll/')
sps.solve_over(archived.spec, archived.sources, archived.axis, carry=archived.carry)
```

`keep_windows=True` also writes each window's own frames, lookahead included,
which `primal(..., per_window=True)` then reads. With `spill_to=` as well, the
archive reads its answer off the spill, so a sweep too large to hold is
archived without being held.

## What an archive will not take

- **A sweep cut by a hand-built axis.** A list of `(key, sources)` is a set of
  unrelated questions, so archive one solve each.
- **A directory that already holds something.** A directory archive is written
  whole, never merged. A `.zip` target is replaced.

A directory of archives reads as one set of tables:
[reading a directory of runs](warehouse.md).
