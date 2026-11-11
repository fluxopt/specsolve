# Your first model

Solve three generators that meet a load at least cost. The next four
tutorials grow this same model.

## Install

```bash
pip install specsolve
```

This also installs polars, the HiGHS solver and
[mathspec](https://mathspec.readthedocs.io/en/latest/), the language a model
is written in.

## The model

A model is one YAML file, called a **spec**. Save this one as `dispatch.yaml`:

<!-- The fence includes examples/dispatch.yaml, which tests/test_tutorial_models.py solves. -->
<!-- doctest: skip -->
```yaml title="dispatch.yaml"
--8<-- "examples/dispatch.yaml"
```

The file holds no numbers, only five parts:

- **`dimensions`** are the sets the model runs over: four snapshots (hours)
  and the generators.
  [Dimensions →](https://mathspec.readthedocs.io/en/latest/reference/language/dimensions/)
- **`parameters`** are the data you attach later.
  [Parameters →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#parameters)
- **`variables`** are what the solver decides: the output `p` of each
  generator in each snapshot.
  [Variables →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#variables)
- **`constraints`** are the rules: in each snapshot, the generators meet the
  load.
  [Constraints →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#constraints)
- **`objective`** is what the solver minimises: the cost of the output.
  [Objective →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#objective)

## Check the file

[`check`](reference/api.md#run-a-spec) tells you whether the file is a model.
It needs no data:

```python exec="true" source="material-block" session="first" workdir="examples"
import specsolve as sps

program = sps.check('dispatch.yaml')
```

The check passes, so the block prints nothing. Now change the line of
`power_balance` to `expression: p == load`, and check again. The message says
how to fix it:

```text
DimensionError: Constraint 'power_balance': the expression carries dims ['generator'] that are not in its dims: ['snapshot']. Add them to dims:, or sum them out.
```

The `sum` over `generator` fits `p` to one row per snapshot. Put it back.

## Attach the data

The numbers come in a `dict` called **`sources`**, one entry for each name
the file declares. A parameter is a table with its dimensions and a `value`
column. A dimension is a list of labels:

```python exec="true" source="material-block" session="first" workdir="examples"
import polars as pl

generators = ['wind', 'solar', 'gas']
sources = {
    'snapshot': range(4),
    'generator': generators,
    'p_max': pl.DataFrame({'generator': generators, 'value': [80.0, 0.0, 200.0]}),
    'cost': pl.DataFrame({'generator': generators, 'value': [10.0, 25.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': range(4), 'value': [60.0, 120.0, 180.0, 90.0]}),
}
```

Wind is cheap but small, gas large but expensive, and solar has no capacity.
[The data contract](reference/data.md) lists every form a source can take.

## Solve

[`solve`](reference/api.md#run-a-spec) attaches the data, builds the model
and hands it to HiGHS:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
result = sps.solve('dispatch.yaml', sources)
print(result.objective)
```

The load is met for 10500.

## Read the answer

`primal` gives the value of a variable as a polars table:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
print(result.primal('p'))
```

Wind runs first because it is cheapest, and gas covers the rest. Solar has no
rows: `where: "p_max > 0"` left it out.

`dual` gives the price of a constraint. For `power_balance`, that is the cost
of one more unit of load:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
print(result.dual('power_balance'))
```

In snapshot 0, wind alone meets the load, so one more unit costs 10. In the
other snapshots gas is the last generator on, and one more unit costs 50.
[What a change is worth](howto/prices.md) reads the other prices an answer
holds.

## Next

In [the next tutorial](tables.md) you attach the same model to data from
files.
