# Preparing the data

From the files an instance arrives in to the `sources` mapping the functions
take, one table per parameter. What that mapping may contain is
[the data contract](../reference/data.md).

## The files you start from

Entity tables, attributes side by side as a PyPSA-style CSV folder holds
them, and tidy time series. The committed instance for
[dispatch](../examples/dispatch.md):

`examples/ports/data/dispatch/generators.csv`

```csv
generator,p_max,cost
wind,80.0,10.0
solar,0.0,25.0
gas,200.0,50.0
```

`examples/ports/data/dispatch/load.csv`

```csv
snapshot,value
0,60.0
1,120.0
2,180.0
3,90.0
```

## One table per parameter

One `select` per parameter: its dimension columns and a `value` column. The
time series passes through untouched:

```python
import polars as pl

generators = pl.read_csv('examples/ports/data/dispatch/generators.csv')

load = pl.read_csv('examples/ports/data/dispatch/load.csv')

sources = {
    'snapshot': load.select('snapshot').unique(maintain_order=True),
    'generator': generators.select('generator'),
    'p_max': generators.select('generator', pl.col('p_max').alias('value')),
    'cost': generators.select('generator', pl.col('cost').alias('value')),
    'load': load,
}
```

With one parquet file per parameter, pass the paths instead:
`sources = {'p_max': 'p_max.parquet', ...}`.

## From linopy's shapes

Pass a pandas Series over one dimension as it is; its index attaches to the
dimension it is named after. A Series over two dimensions is a table:
`reset_index()` it, since a `MultiIndex` is refused. A `DataArray` becomes one with `.to_series()`. The
[dispatch](../examples/dispatch.md) instance, linopy-style:

```python
import pandas as pd

p_max = pd.Series({'wind': 80.0, 'solar': 0.0, 'gas': 200.0}).rename_axis('generator')
cost = pd.Series({'wind': 10.0, 'solar': 25.0, 'gas': 50.0}).rename_axis('generator')
load = pd.Series([60.0, 120.0, 180.0, 90.0]).rename_axis('snapshot')

sources = {'snapshot': load.index, 'generator': p_max.index, 'p_max': p_max, 'cost': cost, 'load': load}
```

## From PyPSA's shapes

A static attribute is an indexed Series already: rename its index. A wide
time series needs `stack()` and `reset_index()`, since a parameter over two
dimensions is a table with both as columns. Here the load is mapped onto
buses on the way, the shape [transport](../examples/transport.md) attaches:

```python
load = (
    n.loads_t.p_set.rename(columns=n.loads.bus)
    .rename_axis(index='snapshot', columns='bus')
    .stack()
    .rename('value')
    .reset_index()
)

sources = {
    'snapshot': load['snapshot'].unique(),
    'bus': load['bus'].unique(),
    'generator': n.generators.index.rename('generator'),
    'p_max': n.generators['p_nom'].rename_axis('generator'),
    'cost': n.generators['marginal_cost'].rename_axis('generator'),
    'gen_bus': n.generators['bus'].rename_axis('generator').reset_index(),
    'load': load,
}
```

`gen_bus` is a
[relation](https://mathspec.readthedocs.io/en/latest/reference/language/relations/),
so it arrives under its own name as a table, one column per column it declares:
PyPSA's `bus` column as it stands, not merged into an index.
