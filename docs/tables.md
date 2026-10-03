# Tables in, tables out

The dispatch model from [Run a model](run.md), fed from parquet files and
read back as tables. The last two steps keep the solve as an **archive**
([glossary](reference/glossary.md#the-chain)), a directory that holds the
spec and parquet files, and query it.

Every block on this page runs when the site is built, and what you see under it
is what it printed on this commit. A block that raises fails the build. The
files go to a temporary directory, `work`.

```python exec="true" source="material-block" session="tables"
import tempfile
from pathlib import Path

import polars as pl

import specsolve as sps

SPEC = 'examples/dispatch.yaml'
work = Path(tempfile.mkdtemp())
```

## 1. Tables in

Start from two tables: one row per generator, and the load over four
snapshots. Write one parquet file per parameter, with the parameter's
dimension columns and a `value` column:

```python exec="true" source="material-block" result="text" session="tables"
generators = pl.DataFrame(
    {'generator': ['wind', 'solar', 'gas'], 'p_max': [80.0, 0.0, 200.0], 'cost': [10.0, 25.0, 50.0]}
)
load = pl.DataFrame({'snapshot': range(4), 'value': [60.0, 120.0, 180.0, 90.0]})

generators.select('generator', value='p_max').write_parquet(work / 'p_max.parquet')
generators.select('generator', value='cost').write_parquet(work / 'cost.parquet')
load.write_parquet(work / 'load.parquet')

print(pl.read_parquet(work / 'cost.parquet'))
```

`sources` names a table for each key the file declares
([the data contract](reference/data.md)). A value is a parquet path or a table
in memory. The `generator` labels come from the polars frame, and the
`snapshot` labels from the column of that name in `load.parquet`:

```python exec="true" source="material-block" result="text" session="tables"
sources = {
    'generator': generators,
    'snapshot': work / 'load.parquet',
    'p_max': work / 'p_max.parquet',
    'cost': work / 'cost.parquet',
    'load': work / 'load.parquet',
}

result = sps.solve(SPEC, sources)
print(result.objective)
```

## 2. Tables out

[`primal`](reference/api.md#specsolve.Result.primal) gives the value of a variable
as a polars table: one row per coordinate, keyed by the dimension labels, with
a `value` column. Solar has no capacity, so `p` has no rows for it:

```python exec="true" source="material-block" result="text" session="tables"
p = result.primal('p')
print(p)
```

`dual` gives the price of each constraint row in the same shape:

```python exec="true" source="material-block" result="text" session="tables"
print(result.dual('power_balance'))
```

The result is an ordinary table, so polars works on it directly. The cost per
generator is a join and a sum:

```python exec="true" source="material-block" result="text" session="tables"
spend = (
    p.join(generators, on='generator')
    .group_by('generator')
    .agg(spend=(pl.col('value') * pl.col('cost')).sum())
    .sort('generator')
)
print(spend)
```

The two rows add up to the objective.

## 3. Archive

`archive=` writes the spec, the sources and the answer to one directory
([archiving a solve](howto/archiving.md)):

```python exec="true" source="material-block" result="text" session="tables"
case = work / 'case'
sps.solve(SPEC, sources, archive=case)

for file in sorted(case.rglob('*')):
    if file.is_file():
        print(file.relative_to(case))
```

`spec.yaml` is the spec. `sources/` holds one parquet file per key, the table
the solve read ([`sps.tidy`](reference/api.md#specsolve.tidy)), and
`sources.parquet` a digest of each. `catalog.parquet` says what each file holds.
`answer/` holds the record of the solve,
its metrics, and one parquet file per variable and constraint. The record is a
table too:

```python exec="true" source="material-block" result="text" session="tables"
print(pl.read_parquet(case / 'answer' / 'record.parquet').select('specsolve_run', 'status', 'objective'))
```

[`load_archive`](reference/api.md#specsolve.load_archive) reads the directory
back. The spec and the sources in it ask the same question again:

```python exec="true" source="material-block" result="text" session="tables"
archived = sps.load_archive(case)
print(sps.solve(archived.spec, archived.sources).objective)
```

## 4. Query the archive

The archive is parquet on disk, so a query needs no specsolve. This scan
joins the answer to the costs it was solved with, and gives the cost per
generator of step 2:

```python exec="true" source="material-block" result="text" session="tables"
solved = pl.scan_parquet(case / 'answer' / 'primal' / 'p.parquet')
costs = pl.scan_parquet(case / 'sources' / 'cost.parquet')

query = (
    solved.join(costs, on='generator', suffix='_cost')
    .group_by('generator')
    .agg(spend=(pl.col('value') * pl.col('value_cost')).sum())
    .sort('generator')
)
print(query.collect())
```

A database that reads parquet runs the same query. The SQL for DuckDB is on
[reading a directory of runs](howto/warehouse.md#query-it-from-a-database).

## Where next

| | |
|---|---|
| [Change a model](change.md) | the next tutorial: new numbers, more rows, new math |
| [The data contract](reference/data.md) | what each key in `sources` accepts and refuses |
| [Archiving a solve](howto/archiving.md) | zips, sweeps and archives too big to hold |
| [Reading a directory of runs](howto/warehouse.md) | many archives as one table |
