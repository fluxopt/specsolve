# Solving many scenarios

How to solve one model for several sets of data in one call, and read the
answers as one table. The reference is
[Sweeps and rolling horizons](../reference/sweeps.md).

## Put the scenarios in one table

Give the table that differs a `scenario` column that the spec does not
declare. [`EachCoordinate`](../reference/api.md#specsolve.EachCoordinate)
solves once for each label:

```python exec="true" source="material-block" result="text" session="scenarios"
import polars as pl

import specsolve as sps

generators = pl.read_csv('examples/tutorial/data/generators.csv')
load = pl.read_csv('examples/tutorial/data/load.csv')
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
}

loads = pl.concat(
    [
        load.with_columns(scenario=pl.lit('mild')),
        load.with_columns(scenario=pl.lit('cold'), value=1.3 * pl.col('value')),
    ]
)
scenarios = sps.solve_over('examples/dispatch.yaml', sources | {'load': loads}, sps.EachCoordinate('scenario'))
print(scenarios.record.select('scenario', 'objective'))
```

A table without the column is the same in every scenario.

## Read the answers

Every reader adds the `scenario` column:

```python exec="true" source="material-block" result="text" session="scenarios"
gas = scenarios.primal('p').filter(pl.col('generator') == 'gas')
print(gas.pivot(on='scenario', index='snapshot', values='value'))
```

`record` has one row for each scenario. There is no total objective, and a
scenario with an integer variable has no duals.

## Vary more than one table

Give each table that differs the same column. Here gas also costs 80 in the
cold scenario:

```python exec="true" source="material-block" result="text" session="scenarios"
costs = pl.concat(
    [
        sources['cost'].with_columns(scenario=pl.lit('mild')),
        sources['cost'].with_columns(
            scenario=pl.lit('cold'),
            value=pl.when(pl.col('generator') == 'gas').then(80.0).otherwise(pl.col('value')),
        ),
    ]
)
both = sps.solve_over(
    'examples/dispatch.yaml', sources | {'load': loads, 'cost': costs}, sps.EachCoordinate('scenario')
)
print(both.record.select('scenario', 'objective'))
```

## Solve them in parallel

The scenarios do not depend on each other, so they can run on a pool of
workers. Each worker needs the memory of one scenario.
[Running a sweep in parallel](parallel.md) is the recipe.
