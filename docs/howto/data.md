# Preparing the data

How to turn an instance's files into the `sources` mapping, one table per
parameter. The reference is [the data contract](../reference/data.md).

## The files you start from

The instance for [dispatch](../examples/dispatch.md):

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

One `select` per parameter: its dimension columns and a `value` column:

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

## From a large file

Read only the rows and columns the model uses. Specsolve holds every source
whole in memory while the model lives, so what you read is what it holds.

- **Store the data as parquet.** Parquet keeps each column's type, and polars
  can skip the rows and columns you do not ask for. A path in `sources` is
  always read as parquet, so convert a CSV once:
  `pl.read_csv('load.csv').write_parquet('load.parquet')`.
- **Scan the file, then filter and select before you collect.**
  `pl.scan_parquet` reads nothing yet. Polars pushes the `filter` and the
  `select` into the scan, so `collect` reads only the slice for one year:

  ```python
  import polars as pl

  year = pl.scan_parquet('timeseries.parquet').filter(pl.col('year') == 2030).select('snapshot', 'load').collect()
  generators = pl.read_parquet('generators.parquet', columns=['generator', 'p_max', 'cost'])

  sources = {
      'snapshot': year.select('snapshot'),
      'load': year.select('snapshot', value='load'),
      'generator': generators.select('generator'),
      'p_max': generators.select('generator', value='p_max'),
      'cost': generators.select('generator', value='cost'),
  }
  ```

- **A lazy frame is a source too.** Specsolve collects it when the data
  attaches. Collect once yourself when several sources come from one file, as
  above. Each lazy source scans the file again.

## From linopy's shapes

Pass a pandas Series over one dimension as it is, its index named after the
dimension. `reset_index()` a Series over two dimensions, since a `MultiIndex`
is refused. A `DataArray` becomes a Series with `.to_series()`:

```python
import pandas as pd

p_max = pd.Series({'wind': 80.0, 'solar': 0.0, 'gas': 200.0}).rename_axis('generator')
cost = pd.Series({'wind': 10.0, 'solar': 25.0, 'gas': 50.0}).rename_axis('generator')
load = pd.Series([60.0, 120.0, 180.0, 90.0]).rename_axis('snapshot')

sources = {'snapshot': load.index, 'generator': p_max.index, 'p_max': p_max, 'cost': cost, 'load': load}
```

## From PyPSA's shapes

Rename the index of a static attribute. `stack()` and `reset_index()` a wide
time series. Here the load is mapped onto buses, the shape
[transport](../examples/transport.md) attaches:

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
    'generator': n.generators[['bus']].rename_axis('generator').reset_index(),
    'p_max': n.generators['p_nom'].rename_axis('generator'),
    'cost': n.generators['marginal_cost'].rename_axis('generator'),
    'load': load,
}
```

`bus` is a
[property](https://mathspec.readthedocs.io/en/latest/reference/language/sets/)
of `generator`, so the table under `generator` carries it: the `generator`
column is the index, and the `bus` column is the property. The same table may
go under `generator.bus` instead, without the other columns.
