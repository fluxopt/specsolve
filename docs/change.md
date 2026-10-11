# Change it and solve again

Ask the model from [Your data](tables.md) a series of what-if questions, and
let the solver keep its model between them. You change numbers first, and
then the length of the horizon.

Every block on this page runs from the `examples/` folder of the repository.
We start from the sources of the last tutorial:

```python exec="true" source="material-block" session="change" workdir="examples"
import polars as pl

import specsolve as sps

generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
    'load': load,
}
```

## Build once

[`solve`](reference/api.md#run-a-spec) builds a model and throws it away.
[`build`](reference/api.md#run-a-spec) keeps it, so you can solve it more
than once:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
model = sps.build('dispatch.yaml', sources)
print(model.solve().objective)
```

The answer is the 10500 of the first two tutorials.

## New numbers

How much would more wind save? `update` puts new numbers on the model that
the solver holds, and `solve` answers again. We try four wind capacities:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
rows = []
for wind in (80.0, 120.0, 160.0, 200.0):
    p_max = pl.DataFrame({'generator': ['wind', 'solar', 'gas'], 'value': [wind, 0.0, 200.0]})
    rows.append({'wind': wind, 'objective': model.update({'p_max': p_max}).solve().objective})

print(pl.DataFrame(rows))
```

Each step of 40 saves less than the one before. With more wind, more of it is
left over in the snapshots where the load is low.

`diagnostics` counts what the model did:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
after_numbers = model.diagnostics()
print(f'{after_numbers.loads} load, {after_numbers.solves} solves')
```

The solver loaded the model once and solved it five times. An update changes
only the numbers, so the solver keeps what it has. The answer is always the
one a new `solve` gives:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
fresh = sps.solve('dispatch.yaml', sources | {'p_max': p_max})
print(fresh.objective)
```

## A longer horizon

Now solve two days instead of one. The `snapshot` labels and the `load` both
grow to eight rows. We put the capacity back to the data, too:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
two_days = pl.DataFrame({'snapshot': range(8), 'value': 2 * load['value'].to_list()})

longer = model.update({'snapshot': two_days['snapshot'], 'load': two_days, 'p_max': sources['p_max']}).solve()
after_rows = model.diagnostics()

print(longer.objective)
print(f'{after_rows.loads} loads, {after_rows.solves} solves')
```

Two days cost twice one day. The model was loaded a second time: new
snapshots add new columns, and the solver has to start again.
[Making a loop of solves faster](howto/warm-start.md#start-from-an-earlier-answer) starts the new solve from an earlier answer.

## Next

You have changed the numbers and the size of your model without writing it
again. In [the next tutorial](commitment.md) you change the model itself:
generators that are either on or off.
