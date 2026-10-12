# Your data

Attach the model from [Your first model](run.md) to data in files, and keep
what a solve produces. Only the data changes. Every block runs from the
`examples/` folder of the repository.

## The files

[`generators.csv`](https://github.com/fluxopt/specsolve/blob/main/examples/tutorial/data/generators.csv)
has one row for each generator:

```text title="tutorial/data/generators.csv"
--8<-- "examples/tutorial/data/generators.csv"
```

[`load.csv`](https://github.com/fluxopt/specsolve/blob/main/examples/tutorial/data/load.csv)
has the load in each snapshot:

```text title="tutorial/data/load.csv"
--8<-- "examples/tutorial/data/load.csv"
```

The columns `p_min` and `no_load_cost` wait for a later tutorial.

## From files to sources

Read both files with polars:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
import polars as pl

import specsolve as sps

generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
print(generators)
```

Each parameter needs its own table with a `value` column, so you pick it out
of `generators` with `select`. `load.csv` already has that shape:

```python exec="true" source="material-block" session="data" workdir="examples"
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
    'load': load,
}
```

A source can also be a pandas frame or a parquet path; see
[the data contract](reference/data.md).

## Solve

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
result = sps.solve('dispatch.yaml', sources)
print(result.objective)
```

The data is the same as before, so the cost is 10500 again.

## Work with the answer

The answer is a polars table. Join it to the cost to see what each generator
costs over the four snapshots:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
spend = (
    result.primal('p')
    .join(generators, on='generator')
    .group_by('generator')
    .agg(spend=(pl.col('value') * pl.col('cost')).sum())
    .sort('generator')
)
print(spend)
```

Gas costs 7500 and wind 3000, which add up to the objective.

## Keep the run

`archive=` writes the spec, the sources and the answer to a folder:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
import tempfile
from pathlib import Path

run = Path(tempfile.mkdtemp()) / 'first_run'
sps.solve('dispatch.yaml', sources, archive=run)

for file in sorted(run.rglob('*.parquet')):
    print(file.relative_to(run))
```

`sources/` holds each table as the solve read it, and `answer/` holds one
file for each variable and constraint.

[`load_archive`](reference/api.md#specsolve.load_archive) reads the answer
back without solving again:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
archived = sps.load_archive(run)
print(archived.result.objective)
```

To zip an archive, see [Archiving a solve](howto/archiving.md). To query many
as one table, see [Reading a directory of runs](howto/warehouse.md).

## Next

In [the next tutorial](change.md) you change the data and solve again.
