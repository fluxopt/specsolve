# Solving many scenarios

How to solve one model for several sets of data in one call, and read the
answers as one table. The reference is
[Sweeps and rolling horizons](../reference/sweeps.md).

## Put the scenarios in one table

Give the table that differs a `scenario` column that the spec does not
declare. [`EachLabel`](../reference/api.md#specsolve.EachLabel)
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
scenarios = sps.solve_over('examples/dispatch.yaml', sources | {'load': loads}, sps.EachLabel('scenario'))
print(scenarios.records.select('scenario', 'objective'))
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
both = sps.solve_over('examples/dispatch.yaml', sources | {'load': loads, 'cost': costs}, sps.EachLabel('scenario'))
print(both.records.select('scenario', 'objective'))
```

## Draw each table from its own pool

To combine the years of one table with the years of another, give each pool
its own column instead of `scenario`. Then write one table that names a label
of each pool for every scenario, and join it to each pool:

```python exec="true" source="material-block" result="text" session="scenarios"
weather = pl.concat(
    [
        load.with_columns(weather=pl.lit('mild')),
        load.with_columns(weather=pl.lit('cold'), value=1.3 * pl.col('value')),
    ]
)
fuel = pl.concat(
    [
        sources['cost'].with_columns(fuel=pl.lit('cheap')),
        sources['cost'].with_columns(
            fuel=pl.lit('dear'),
            value=pl.when(pl.col('generator') == 'gas').then(80.0).otherwise(pl.col('value')),
        ),
    ]
)
draws = pl.DataFrame(
    {'scenario': ['s1', 's2', 's3'], 'weather': ['mild', 'cold', 'cold'], 'fuel': ['dear', 'cheap', 'dear']}
)
drawn = {
    'load': draws.join(weather, on='weather').drop('weather', 'fuel'),
    'cost': draws.join(fuel, on='fuel').drop('weather', 'fuel'),
}
pooled = sps.solve_over('examples/dispatch.yaml', sources | drawn, sps.EachLabel('scenario'))
print(pooled.records.select('scenario', 'objective'))
```

Scenarios `s2` and `s3` share the cold load, and only `s3` pays 80 for gas, so
`s3` costs the most. The pools stay as they are, and a new scenario is one more
row of `draws`.

## Solve them in parallel

The scenarios do not depend on each other, so they can run on a pool of
workers. Each worker needs the memory of one scenario.
[Running a sweep in parallel](parallel.md) is the recipe.
