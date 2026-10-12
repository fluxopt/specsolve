# Your first model

Solve your first model with specsolve: three generators that meet a load at
least cost. The next four tutorials grow this same model, one step at a time.

## Install

```bash
pip install specsolve
```

This installs specsolve with polars, the HiGHS solver and
[mathspec](https://mathspec.readthedocs.io/en/latest/), the language a model
is written in.

## The model

A model is one YAML file, called a **spec**. Save this one as `dispatch.yaml`:

<!-- The fence includes examples/dispatch.yaml, which tests/test_tutorial_models.py solves. -->
<!-- doctest: skip -->
```yaml title="dispatch.yaml"
--8<-- "examples/dispatch.yaml"
```

The file holds no numbers. It says what the model is, in five parts:

- **`dimensions`** are the sets the model runs over: four snapshots, which
  are hours, and the generators.
  [Dimensions →](https://mathspec.readthedocs.io/en/latest/reference/language/dimensions/)
- **`parameters`** are the data you attach later: the capacity and cost of
  each generator, and the load in each snapshot.
  [Parameters →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#parameters)
- **`variables`** are what the solver decides: the output `p` of each
  generator in each snapshot. `where` leaves out a generator with no capacity.
  [Variables →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#variables)
- **`constraints`** are the rules: in each snapshot, the generators together
  meet the load.
  [Constraints →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#constraints)
- **`objective`** is what the solver minimises: the cost of what the
  generators produce.
  [Objective →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#objective)

You write a spec once. You solve it with as many sets of data as you like.

## Check the file

[`check`](reference/api.md#run-a-spec) reads the file and tells you whether
it is a model. It needs no data:

```python exec="true" source="material-block" session="first" workdir="examples"
import specsolve as sps

program = sps.check('dispatch.yaml')
```

The check passes, so the block prints nothing. Now we break the file on
purpose. Change the line of `power_balance` to `expression: p == load`, and
check again. The check refuses the file, and the message says how to fix it:

```text
DimensionError: Constraint 'power_balance': the expression carries dims ['generator'] that are not in its dims: ['snapshot'] — every stray dim multiplies the rows this constraint builds; add it to dims: if that is intended, or sum it out.
```

`p` has one value per snapshot and generator, and `power_balance` holds once
per snapshot. The `sum` over `generator` is what makes the two fit. Put it
back before the next step.

## Attach the data

The numbers come in a `dict` called **`sources`**, with one entry for each
name the file declares. A parameter is a table with one column for each of
its dimensions and a `value` column. A dimension takes a list of its labels:

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

Wind is cheap but small. Gas is large but expensive. Solar has no capacity
here. [The data contract](reference/data.md) lists every form a source can
take.

## Solve

[`solve`](reference/api.md#run-a-spec) attaches the data to the spec, builds
the model and hands it to HiGHS:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
result = sps.solve('dispatch.yaml', sources)
print(result.objective)
```

The cheapest way to meet the load costs 10500.

## Read the answer

`primal` gives you the value of a variable, as a polars table with one row
for each snapshot and generator:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
print(result.primal('p'))
```

Wind runs first because it is cheapest, and gas covers the rest. Solar has no
rows: `where: "p_max > 0"` left it out of the model.

`dual` gives you the price of a constraint. For `power_balance`, that is what
one more unit of load would cost in each snapshot:

```python exec="true" source="material-block" result="text" session="first" workdir="examples"
print(result.dual('power_balance'))
```

In snapshot 0, wind alone meets the load, so one more unit costs 10. In the
other snapshots gas is the last generator on, and one more unit costs 50.

## Next

You have written a spec, checked it, solved it and read the answer. In
[the next tutorial](tables.md) you attach the same model to data from files.
