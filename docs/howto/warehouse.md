# Reading a directory of runs

How to read a directory of archives as one set of tables, and join the values
to their labels. One archive is [archiving a solve](archiving.md). This page is
about the directory that archives pile up in.

## Which runs share a directory

**Put runs in one directory when their value tables should combine.** Each
`archive=` writes one directory, named after the run:

```python
import specsolve as sps

sps.solve(spec, sources, archive='runs/base')
sps.solve(spec, peak, archive='runs/peak')
sps.solve_over(spec, by_scenario, sps.EachCoordinate('scenario'), archive='runs/scenarios')
rolling = sps.EachWindow('snapshot', steps=4, lookahead=0, into='t')
sps.solve_over(spec, by_snapshot, rolling, archive='runs/rolling')
sps.solve_over(spec, by_snapshot, rolling, archive='runs/rolling-kept', keep_windows=True)
```

Here `spec` declares `p` over `t` and `generator`, and `sources` and `peak`
are two sets of its data. `by_scenario` carries a `scenario` column on `load`.
`by_snapshot` gives the data over `snapshot` rather than `t`.

**A sweep adds a column that a single solve does not write.** A scenario sweep
adds its key, here `scenario`, which the spec does not declare. A rolling
horizon writes its answer over the dimension it cuts, `snapshot`, where a
single solve of the same spec writes `t`. So `answer/primal/p.parquet` has
three shapes in this directory. To compare a base run with a sweep, solve the
base as a sweep of one slice, or declare the dimension in the spec.

**Write a directory, not a `.zip`.** No query engine reads inside a zip. Unpack
one with the `into=` of `load_archive` or `scan_archive`.

## The three tables

An archive is a tree of parquet files, so a directory of archives is a table
per glob. Nothing is loaded and no schema is maintained:

```python
import polars as pl

answers = pl.read_parquet('runs/*/answer/record.parquet')
metrics = pl.read_parquet('runs/*/answer/metrics.parquet')
catalog = pl.read_parquet('runs/*/catalog.parquet')
```

| glob | one row per | says |
|---|---|---|
| `answer/record.parquet` | solve, or sweep slice | how it terminated, what it reached, when, under what name, and on which solver and package versions |
| `answer/metrics.parquet` | the same | what the build and its solves spent, and how big the model was |
| `catalog.parquet` | column of labels of each file | what the file holds ([what a file holds](#what-a-file-holds)) |

**Every row says which archive it came from.** `specsolve_run` is the
archive's own name: `runs/base` writes `base`, and `runs/base.zip` writes the
same. Every table in an archive carries it, so nothing has to read the paths.

**These three tables have the same columns, whoever wrote them.** A single
solve, an `EachCoordinate` sweep and an `EachWindow` sweep write one schema
for each, so each glob reads as one table. A sweep writes one row per slice
and names it in two text columns. `slice_axis` is the key name, such as
`scenario` or `snapshot_start`, and `slice` is the key as text. Both are null
for a single solve:

```python
answers.select('specsolve_run', 'slice_axis', 'slice').sort('specsolve_run', 'slice')
# specsolve_run  slice_axis      slice
# base           null            null
# peak           null            null
# rolling        snapshot_start  0
# rolling        snapshot_start  4
# ...
# scenarios      scenario        high
# scenarios      scenario        low
```

**Use `load_archive` and `scan_archive` for one archive, not for many.** They
give back a spec, its sources and an answer, which is what re-running a case
needs. A question about many runs is a query over the parquet.

## The value tables

`answer/primal/p.parquet` holds the model's own dimension columns, `value`,
and `specsolve_run`. A sweep archive writes the same file, with the columns
[a sweep adds](#which-runs-share-a-directory):

```python
pl.read_parquet('runs/base/answer/primal/p.parquet').head(3)
# t  generator  value  specsolve_run
# 0  wind       15.0   base
# 0  gas        0.0    base
# 0  coal       5.0    base
```

**A glob over runs of different shapes is refused.** polars takes the schema
of the first file and refuses a file with another column:

```text
SchemaError: extra column in file outside of expected schema: snapshot
```

Union the files by name instead. A column is null on the rows of a run that
does not write it:

```python
from glob import glob

p = pl.concat([pl.read_parquet(file) for file in sorted(glob('runs/*/answer/primal/p.parquet'))], how='diagonal')
```

**The column `specsolve_run` is the archive's, not the model's.** The
`specsolve_` prefix is reserved for the columns specsolve adds. A spec that
declares a name with that prefix, in any letter case, is refused. Both readers
drop the column from the answer's frames, so a frame read back is the frame
the solve returned. `load_archive` drops it from the sources too. The sources
`scan_archive` gives back are the archive's own files, and reading one gives
the column back.

**Every column has a type that parquet readers agree on.** An archive writes
an unsigned integer as `Int64` and a zoned timestamp in UTC ([the rule](../reference/data.md#column-types-in-an-archive)).

## Keys across runs

**Join a value table to a dimension table on the run and the label.** Each
archive holds its own `sources/<dimension>.parquet`, so the stacked dimension
table repeats each label once per run. The label alone is not unique there,
and a relation can differ from one run to the next. The pair is unique:

```python
generators = pl.read_parquet('runs/*/sources/generator.parquet')
p.join(generators, on=['specsolve_run', 'generator'])
```

**A tool that relates tables on one column needs the pair as one column.**
Build it the same way on both sides when you read the tables:

```python
generators.with_columns(key=pl.concat_str('specsolve_run', 'generator', separator='|'))
```

A rolling horizon holds no table for `t`, because each window numbers its own
`t` from 0. Its answer is over `snapshot`, which the spec does not declare.

## Relations

**Every archive holds each relation as `sources/<relation>.parquet`**, one
column per column it declares, and `catalog.parquet` names the dimension of
each column. A join through a relation is a join on the run and the shared
columns:

```python
sited = pl.read_parquet('runs/*/sources/sited.parquet')
p.join(sited, on=['specsolve_run', 'generator']).group_by('specsolve_run', 't', 'bus').agg(pl.col('value').sum())
```

Each kind of relation takes one join pattern:

| relation | example | join |
|---|---|---|
| keyed by one dimension | `sited: {key: generator, values: bus}` | on the key |
| keyed by a pair | `zone_of: {key: [generator, t], values: zone}` | on both key columns |
| bare | `connection: {key: [generator, bus]}` | on the columns the value table shares |
| roles | `ends: {key: line, values: {bus0: bus, bus1: bus}}` | once per role, on the role's column |
| onto itself | `{key: snapshot, values: {rep: snapshot}}` | as roles |
| a partition | `shift`, `sum_back` or `position` with `within=` | none |

## Totals over a bare relation

**A sum through a bare relation counts a member once per row it is related
to.** `reach` is `sum(p, by=connection, over=generator, into=bus)`, and
`wind` is connected to both buses. So the total of `reach` over the buses
counts the output of `wind` twice:

```python
reach = pl.read_parquet('runs/base/answer/expression/reach.parquet')
reach['value'].sum() - p.filter(specsolve_run='base')['value'].sum()  # the output of wind, counted again
```

**Declare a total that must keep the model's meaning as a named expression
in the spec**, and read it from `answer/expression/` rather than computing it
again. The same holds for a quantity the spec defines through `within=`.

```sql
select bus, sum(value) as reached
from read_parquet('runs/base/answer/expression/reach.parquet')
group by bus
order by bus;
```

## What a file holds

`catalog.parquet` says what each file in the archive holds, so a reader
needs no `spec.yaml`. It describes each file as it is written. It has one row
per column of labels of each file under `sources/` and `answer/`:

| column | holds |
|---|---|
| `specsolve_run` | the archive's name, as on every other table it holds |
| `path` | the file's path inside the archive, as `sources/load.parquet` or `answer/dual/load.parquet` |
| `name` | the name the spec declares |
| `kind` | `dimension`, `relation`, `parameter`, `variable`, `constraint` or `expression` |
| `description` | the spec's `description:`, or null |
| `dtype` | the declared type of a dimension's labels or a parameter's `value`, else null |
| `column` | a column of the file that holds labels |
| `dim` | the dimension of those labels, or null for a file over no dimension |

**Join it on the path, not the name.** A constraint can have the name of a
parameter, so `name = 'load'` can match the parameter's source and the
constraint's dual. `path` and `column` identify one row. In DuckDB:

```sql
select name, kind, description, "column", dim
from read_parquet('runs/base/catalog.parquet')
where path = 'answer/primal/p.parquet'
order by "column";
```

**The catalog lists the files the archive holds, and no other.** A name the
spec declares has no row where it has no file: every answer of a solve that
left no values, the duals of a model that has none, and a named expression the
data cannot evaluate. `answer/record.parquet` and `answer/reasons.parquet` say
why.

**A sweep archive lists the columns the sweep wrote.** An `EachCoordinate`
sweep adds its key column, such as `scenario`, to the answer and to each
source it cuts. An `EachWindow` sweep holds its answer and the sources it cuts
over the dimension it slices, such as `snapshot`, where the spec declares the
local index `t`. Each of these columns has a row, with the dimension the axis
slices as its `dim`.

**Kept windows have a catalog of their own.** With `keep_windows=True`,
`answer/windows/catalog.parquet` lists the per-window frames and
`answer/windows/owned.parquet`, which says what coordinate each window owns.
It has the same columns. `catalog.parquet` never lists them, so it is the same
whether the windows are kept or not.

The catalog has no units, because the spec declares none. It has no row for
`specsolve_run`, which every file carries, or for a dimension's
`specsolve_position`. Neither holds labels.

## Compare cases solved apart

`solved_at` is when the solver returned, in UTC, so runs solved on different
machines still order:

```python
answers.sort('solved_at').select('specsolve_run', 'slice', 'status', 'objective')
```

**Check the spec before you read the numbers.** Every archive holds the spec
it answered as `spec.yaml`. One distinct text across the directory is the claim
that every run answered the same document:

```python
from pathlib import Path

specs = {path.read_text() for path in Path('runs').glob('*/spec.yaml')}
assert len(specs) == 1, 'one spec, or this compares nothing'
```

## Find which input changed between two runs

Two runs of one spec over different data hold the same `spec.yaml`. Their
sources separate them. Compare each table, and the ones that differ are the
inputs that moved:

```python
base = sps.load_archive('runs/base')
other = sps.load_archive('runs/peak')

moved = [name for name in sorted(base.sources) if not base.sources[name].equals(other.sources[name])]
moved  # ['load']
```

The tables are the ones the solve read, so a value compares exactly, whichever
version of polars wrote either archive.

## See what the runs cost

`answer/metrics.parquet` carries the model's size beside the clocks, so one
query says which cases are growing and where the time goes:

```python
metrics.select('specsolve_run', 'slice', 'rows', 'nonzeros', 'build_seconds', 'solve_seconds').sort(
    pl.col('build_seconds') + pl.col('solve_seconds'), descending=True
)
```

The columns are [the metrics](../reference/api.md#specsolve.types.Metrics). A sweep
writes one row per slice, with the slice's own share of the clocks and
`solves` of `1`
([reading a sweep](../reference/sweeps.md#reading-a-sweep)).

## Query it from a database

A directory archive is parquet where it lies, so a query engine such as DuckDB
reads it without polars in the way. The SQL on this page runs in DuckDB. The
same glob reads solves and sweeps together:

```sql
select specsolve_run, slice, rows, nonzeros, build_seconds, solve_seconds
from read_parquet('runs/*/answer/metrics.parquet')
order by build_seconds + solve_seconds desc;
```

**Read a value table with `union_by_name`**, which keeps the `scenario` of a
sweep and the `snapshot` of a rolling horizon:

<!-- warehouse: not run, union_by_name is DuckDB's own -->
```sql
select *
from read_parquet('runs/*/answer/primal/p.parquet', union_by_name = true)
order by specsolve_run, t, snapshot;
```

Without `union_by_name`, DuckDB takes the columns of the first file it reads.
When that file is a single solve, the rows of a scenario sweep lose `scenario`
and no error occurs.

Inside one archive the values join to the sources they were solved from on the
columns both carry:

```sql
select p.t, p.generator, p.value, s.bus
from read_parquet('runs/base/answer/primal/p.parquet') p
join read_parquet('runs/base/sources/sited.parquet') s using (generator, specsolve_run);
```

## Delta Lake

Fabric Direct Lake and Unity Catalog read Delta Lake tables rather than a
folder of parquet. Write one table per glob, with `deltalake` installed:

<!-- warehouse: not run, deltalake is not a dependency here -->
```python
for table in ('answer/record', 'answer/metrics', 'sources', 'catalog'):
    pl.read_parquet(f'runs/*/{table}.parquet').write_delta(f'lake/{table.replace("/", "_")}', mode='overwrite')
```
