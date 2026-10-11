# Your data

Attach the model from [Your first model](run.md) to data that comes in files,
and keep what a solve produces. The model does not change. Only the data does.

Every block on this page runs from the `examples/` folder of the repository,
where the model and the data are.

## The files

The data is two CSV files in `examples/tutorial/data/`.
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

`generators.csv` has two columns that `dispatch.yaml` does not use yet,
`p_min` and `no_load_cost`. A later tutorial uses them. Real data often holds
more than one model needs.

## From files to sources

Read both files with polars:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
import polars as pl

import specsolve as sps

generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
print(generators)
```

`dispatch.yaml` wants one table for each parameter, with a `value` column. You
pick each one out of `generators` with `select`. `load.csv` already has the
right shape. The labels of each dimension come from a column:

```python exec="true" source="material-block" session="data" workdir="examples"
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
    'load': load,
}
```

A source can also be a pandas frame or a parquet path.
[The data contract](reference/data.md) lists every form, and what
specsolve refuses.

## Solve

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
result = sps.solve('dispatch.yaml', sources)
print(result.objective)
```

The data is the same as in the first tutorial, so the cost is the same: 10500.

## Work with the answer

The answer is a polars table, so you continue with polars. Join the output of
each generator to its cost, and you get what each generator costs over the
four snapshots:

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

Gas costs 7500 and wind 3000. Together they make the objective.

## Keep the run

`archive=` writes the spec, the sources and the answer to a folder. We write
it to a temporary folder here:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
import tempfile
from pathlib import Path

run = Path(tempfile.mkdtemp()) / 'first_run'
sps.solve('dispatch.yaml', sources, archive=run)

for file in sorted(run.rglob('*.parquet')):
    print(file.relative_to(run))
```

Every file is a parquet file. `sources/` holds each table as the solve read
it, and `answer/` holds the record of the solve and one file for each variable
and constraint.

[`load_archive`](reference/api.md#specsolve.load_archive) reads the folder
back. You get the answer without solving again:

```python exec="true" source="material-block" result="text" session="data" workdir="examples"
archived = sps.load_archive(run)
print(archived.result.objective)
```

[Archiving a solve](howto/archiving.md) shows how to zip an archive, and
[Reading a directory of runs](howto/warehouse.md) how to query many of them
as one table.

## Next

You have attached your model to data from files, and kept the run. In
[the next tutorial](change.md) you change the data and solve again.
