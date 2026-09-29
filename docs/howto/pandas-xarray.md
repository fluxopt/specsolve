# Reading an answer into pandas or xarray

How to hand an answer to code that works in pandas or xarray. A
[result](../reference/api.md#specsolve.Result) reads as polars tables, and the
bridges below convert one name at a time. specsolve installs neither library,
so install the one you need:

```bash
pip install pandas xarray
```

The examples solve [dispatch](../examples/dispatch.md) on its committed
instance: three generators over four snapshots.

## As a pandas table

```python
import specsolve as sps

result = sps.solve('dispatch.yaml', sources)
print(result.to_pandas('p').head(3))
```

```text
   snapshot generator  value
0         0      wind   60.0
1         0       gas    0.0
2         1      wind   80.0
```

The table has the shape [`primal`](../reference/api.md#specsolve.Result.primal)
gives: one column per dimension, a `value` column, and one row per coordinate
the model built. `kind='dual'` reads a constraint's duals, and
`kind='expression'` a named expression:

```python
print(result.to_pandas('power_balance', kind='dual'))
```

```text
   snapshot  value
0         0   10.0
1         1   50.0
2         2   50.0
3         3   50.0
```

## As a labelled xarray array

```python
print(result.to_dataarray('p'))
```

```text
<xarray.DataArray 'p' (snapshot: 4, generator: 2)> Size: 64B
array([[  0.,  60.],
       [ 40.,  80.],
       [100.,  80.],
       [ 10.,  80.]])
Coordinates:
  * snapshot   (snapshot) int64 32B 0 1 2 3
  * generator  (generator) object 16B 'gas' 'wind'
```

The array is dense over the variable's dimensions. A coordinate that a
`where:` removed comes back `NaN`. A label that no row holds is not on the
axis: solar has no capacity, so `p` has no `solar` column.
[`to_dataarray`](../reference/api.md#specsolve.Result.to_dataarray) takes
`kind=` as `to_pandas` does.

## Every variable as one xarray dataset

```python
result.to_dataset()  # every variable
result.to_dataset('power_balance', kind='dual')  # the duals you name
```

One call reads one kind, since a dual and a variable can share a name. On a
large model, name the few you need: each arrives dense.

## From a sweep

A [sweep](../reference/sweeps.md) has the same three bridges. The slice key
becomes a dimension, such as `scenario`. `original_index=True` reads a rolling
horizon over the dimension it sliced instead, so the answer comes back
indexed by time:

```python
sweep.to_dataarray('p')  # (scenario, snapshot, generator)
rolling.to_pandas('soc', original_index=True)  # over hour, not hour_start
```
