# Reading a directory of runs

How to read a directory of archives as one set of tables, and join the values
to their labels. One archive is [archiving a solve](archiving.md).

## Which runs share a directory

**Put runs in one directory when their tables should combine:**

```python
import specsolve as sps

sps.solve(spec, sources, archive='runs/base')
sps.solve(spec, peak, archive='runs/peak')
sps.solve_over(spec, by_scenario, sps.EachLabel('scenario'), archive='runs/scenarios')
sps.solve_over(spec, longer, sps.EachWindow('t', horizon=4, lookahead=0), archive='runs/rolling')
```

**A sweep writes one file per slice, and each carries the sweep's key**:
`scenario`, or `t_start` for a rolling horizon. To compare a base run with a
sweep, solve the base as a sweep of one slice.

**Write a directory, not a `.zip`.** No query engine reads inside a zip.

## The three tables

A single solve writes one file per table. A sweep writes one file per slice,
in a directory of the same name, as `spill_to=` and `Sweep.save` do. So a
table of the answer is two globs. polars refuses a glob that matches no file,
so expand them first, and a folder of only solves or only sweeps reads too:

```python
from glob import glob

import polars as pl


def files(*patterns):
    return sorted(file for pattern in patterns for file in glob(pattern))


answers = pl.read_parquet(files('runs/*/answer/record.parquet', 'runs/*/answer/record/*.parquet'))
metrics = pl.read_parquet(files('runs/*/answer/metrics.parquet', 'runs/*/answer/metrics/*.parquet'))
catalog = pl.read_parquet('runs/*/catalog.parquet')
```

| glob | one row per | says |
|---|---|---|
| `answer/record.parquet`, `answer/record/*.parquet` | solve, or sweep slice | how it ended, and on which solver |
| `answer/metrics.parquet`, `answer/metrics/*.parquet` | the same | what the build and its solves spent |
| `catalog.parquet` | column of labels of each file | [what a file holds](#what-a-file-holds) |

**Every row names its archive in `specsolve_run`.** A sweep writes one row per
slice, named in `slice_axis` and `slice`:

```python
answers.select('specsolve_run', 'slice_axis', 'slice').sort('specsolve_run', 'slice')
# specsolve_run  slice_axis      slice
# base           null            null
# peak           null            null
# rolling        t_start         0
# rolling        t_start         4
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

**A sweep's files add its key column**, so union solves and sweeps by name:

```python
held = files('runs/*/answer/primal/p.parquet', 'runs/*/answer/primal/p/*.parquet')
p = pl.concat([pl.read_parquet(file) for file in held], how='diagonal')
```

**A rolling horizon's files hold every row each window solved.** Where the
windows look ahead, `answer/horizon.parquet` lists the rows of each window's
horizon: join on `specsolve_run`, the key and the windowed dimension to keep
those.

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
| bare | `connection: {key: [generator, bus], missing: absent}` | on the columns the value table shares |
| roles | `ends: {key: line, values: {bus0: bus, bus1: bus}}` | once per role, on the role's column |
| a partition | `shift`, `sum_back` or `position` with `within=` | none |

**A sum through a bare relation counts a label once per row it is related
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
base = sps.read_archive('runs/base')
other = sps.read_archive('runs/peak')

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
from (
    select * from read_parquet('runs/*/answer/metrics.parquet')
    union all
    select * from read_parquet('runs/*/answer/metrics/*.parquet')
) as metrics
order by build_seconds + solve_seconds desc;
```

Read a value table with `union_by_name`, or a sweep's rows lose their key
with no error:

<!-- warehouse: not run, union_by_name is DuckDB's own -->
```sql
select *
from read_parquet(['runs/*/answer/primal/p.parquet', 'runs/*/answer/primal/p/*.parquet'], union_by_name = true)
order by specsolve_run, t;
```

Join values to their sources on the columns both carry:

```sql
select p.t, p.generator, p.value, s.bus
from read_parquet('runs/base/answer/primal/p.parquet') p
join read_parquet('runs/base/sources/sited.parquet') s using (generator, specsolve_run);
```

For Fabric Direct Lake or Unity Catalog, write one Delta Lake table per table:

<!-- warehouse: not run, deltalake is not a dependency here -->
```python
for name, table in (('record', answers), ('metrics', metrics), ('catalog', catalog)):
    table.write_delta(f'lake/{name}', mode='overwrite')
```
