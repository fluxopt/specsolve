# Reading a directory of runs

How to read a directory of archives as one set of tables, and join the values
to their labels. One archive is [archiving a solve](archiving.md).

## Which runs share a directory

**Put runs in one directory when their tables should combine:**

```python
import specsolve as sps

sps.solve(spec, sources, archive='runs/base')
sps.solve(spec, peak, archive='runs/peak')
sps.solve_over(spec, by_scenario, sps.EachCoordinate('scenario'), archive='runs/scenarios')
rolling = sps.EachWindow('snapshot', steps=4, lookahead=0, into='t')
sps.solve_over(spec, by_snapshot, rolling, archive='runs/rolling')
sps.solve_over(spec, by_snapshot, rolling, archive='runs/rolling-kept', keep_windows=True)
```

**A sweep writes columns a single solve does not**: its key, or the dimension
a rolling horizon cuts. To compare a base run with a sweep, solve the base as
a sweep of one slice.

**Write a directory, not a `.zip`.** No query engine reads inside a zip.

## The three tables

Read one table per glob:

```python
import polars as pl

answers = pl.read_parquet('runs/*/answer/record.parquet')
metrics = pl.read_parquet('runs/*/answer/metrics.parquet')
catalog = pl.read_parquet('runs/*/catalog.parquet')
```

| glob | one row per | says |
|---|---|---|
| `answer/record.parquet` | solve, or sweep slice | how it ended, and on which solver |
| `answer/metrics.parquet` | the same | what the build and its solves spent |
| `catalog.parquet` | column of labels of each file | [what a file holds](#what-a-file-holds) |

**Every row names its archive in `specsolve_run`.** A sweep writes one row per
slice, named in `slice_axis` and `slice`:

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

## The value tables

A value table holds the dimension columns, `value` and `specsolve_run`:

```python
pl.read_parquet('runs/base/answer/primal/p.parquet').head(3)
# t  generator  value  specsolve_run
# 0  wind       15.0   base
# 0  gas        0.0    base
# 0  coal       5.0    base
```

**A glob over runs of different shapes is refused:**

```text
SchemaError: extra column in file outside of expected schema: snapshot
```

Union the files by name instead:

```python
from glob import glob

p = pl.concat([pl.read_parquet(file) for file in sorted(glob('runs/*/answer/primal/p.parquet'))], how='diagonal')
```

Every column has a type that parquet readers agree on
([the rule](../reference/data.md#column-types-in-an-archive)).

## Keys across runs

**Join a value table to a dimension table on the run and the label:**

```python
generators = pl.read_parquet('runs/*/sources/generator.parquet')
p.join(generators, on=['specsolve_run', 'generator'])
```

**A relation joins on the run and the columns it shares:**

```python
sited = pl.read_parquet('runs/*/sources/sited.parquet')
p.join(sited, on=['specsolve_run', 'generator']).group_by('specsolve_run', 't', 'bus').agg(pl.col('value').sum())
```

| relation | example | join |
|---|---|---|
| keyed by one dimension | `sited: {key: generator, values: bus}` | on the key |
| keyed by a pair | `zone_of: {key: [generator, t], values: zone}` | on both key columns |
| bare | `connection: {key: [generator, bus]}` | on the columns the value table shares |
| roles | `ends: {key: line, values: {bus0: bus, bus1: bus}}` | once per role, on the role's column |
| a partition | `shift`, `sum_back` or `position` with `within=` | none |

**A sum through a bare relation counts a member once per row it is related
to.** `wind` is connected to both buses, so it counts twice:

```python
reach = pl.read_parquet('runs/base/answer/expression/reach.parquet')
reach['value'].sum() - p.filter(specsolve_run='base')['value'].sum()  # the output of wind, counted again
```

Declare such a total as a named expression and read it from
`answer/expression/`:

```sql
select bus, sum(value) as reached
from read_parquet('runs/base/answer/expression/reach.parquet')
group by bus
order by bus;
```

## What a file holds

`catalog.parquet` describes each file without `spec.yaml`. **Join it on the
path, not the name**, because a constraint can share a parameter's name:

```sql
select name, kind, description, "column", dim
from read_parquet('runs/base/catalog.parquet')
where path = 'answer/primal/p.parquet'
order by "column";
```

## Compare runs

**Check that every run answered the same spec:**

```python
from pathlib import Path

from mathspec import to_spec

specs = {to_spec(path).to_yaml(canonical=True) for path in Path('runs').glob('*/spec.yaml')}
assert len(specs) == 1, 'one spec, or this compares nothing'
```

**Find the sources that differ:**

```python
base = sps.load_archive('runs/base')
other = sps.load_archive('runs/peak')

moved = [name for name in sorted(base.sources) if not base.sources[name].equals(other.sources[name])]
moved  # ['load']
```

**Find where the time goes**
([the metrics](../reference/api.md#specsolve.types.Metrics)):

```python
metrics.select('specsolve_run', 'slice', 'rows', 'nonzeros', 'build_seconds', 'solve_seconds').sort(
    pl.col('build_seconds') + pl.col('solve_seconds'), descending=True
)
```

## Query it from a database

DuckDB reads a directory archive as it is:

```sql
select specsolve_run, slice, rows, nonzeros, build_seconds, solve_seconds
from read_parquet('runs/*/answer/metrics.parquet')
order by build_seconds + solve_seconds desc;
```

Read a value table with `union_by_name`, or a sweep's rows lose their key
with no error:

<!-- warehouse: not run, union_by_name is DuckDB's own -->
```sql
select *
from read_parquet('runs/*/answer/primal/p.parquet', union_by_name = true)
order by specsolve_run, t, snapshot;
```

Join values to their sources on the columns both carry:

```sql
select p.t, p.generator, p.value, s.bus
from read_parquet('runs/base/answer/primal/p.parquet') p
join read_parquet('runs/base/sources/sited.parquet') s using (generator, specsolve_run);
```

For Fabric Direct Lake or Unity Catalog, write one Delta Lake table per glob:

<!-- warehouse: not run, deltalake is not a dependency here -->
```python
for table in ('answer/record', 'answer/metrics', 'sources', 'catalog'):
    pl.read_parquet(f'runs/*/{table}.parquet').write_delta(f'lake/{table.replace("/", "_")}', mode='overwrite')
```
