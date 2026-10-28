# Change it and solve again

Ask the model from [Your data](tables.md) what-if questions, and let the
solver keep its model between them. Every block runs from the `examples/`
folder. We start from the sources of the last tutorial:

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

[`build`](reference/api.md#run-a-spec) keeps the model that `solve` throws
away, so you can solve it more than once:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
model = sps.build('dispatch.yaml', sources)
print(model.solve().objective)
```

The answer is 10500, as before.

## New numbers

How much would more wind save? `update` puts new numbers on the model the
solver holds. We try four wind capacities:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
rows = []
for wind in (80.0, 120.0, 160.0, 200.0):
    p_max = pl.DataFrame({'generator': ['wind', 'solar', 'gas'], 'value': [wind, 0.0, 200.0]})
    rows.append({'wind': wind, 'objective': model.update({'p_max': p_max}).solve().objective})

print(pl.DataFrame(rows))
```

Each step of 40 saves less than the one before, because more of the wind is
left over when the load is low.

`diagnostics` counts what the model did:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
after_numbers = model.diagnostics()
print(f'{after_numbers.loads} load, {after_numbers.solves} solves')
```

The solver loaded the model once and solved it five times. The answer is the
one a new `solve` gives:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
fresh = sps.solve('dispatch.yaml', sources | {'p_max': p_max})
print(fresh.objective)
```

## How far a number can move

A loop answers one what-if at a time. Ranging answers one question for every
number at once: how far can it move, alone, before a different generator sets
the price? Ask for it with `outputs={'ranging'}`. We read how far the load of
each snapshot can move:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
ranged = sps.solve('dispatch.yaml', sources, outputs={'ranging'})

lower = ranged.rhs_lower('power_balance').rename({'value': 'lower'})
upper = ranged.rhs_upper('power_balance').rename({'value': 'upper'})
price = ranged.dual('power_balance').rename({'value': 'price'})
print(load.rename({'value': 'load'}).join(lower, on='snapshot').join(upper, on='snapshot').join(price, on='snapshot'))
```

In snapshot 0, wind alone serves the load at a price of 10. The price stays 10
while the load stays at or below 80, wind's capacity. In the other snapshots,
gas is the marginal generator at a price of 50. The price stays 50 while the
load stays between 80 and 280, the capacity of both generators together.

`cost_lower` and `cost_upper` do the same for each cost. Wind keeps its share
of the load while it costs no more than 50, the cost of gas
([`Result.cost_lower`](reference/api.md#specsolve.types.Result.cost_lower)).
Only an LP solved to optimality has ranges.

## A longer horizon

Now solve two days. The `snapshot` labels and the `load` grow to eight rows,
and the capacity goes back to the data:

```python exec="true" source="material-block" result="text" session="change" workdir="examples"
two_days = pl.DataFrame({'snapshot': range(8), 'value': 2 * load['value'].to_list()})

longer = model.update({'snapshot': two_days['snapshot'], 'load': two_days, 'p_max': sources['p_max']}).solve()
after_rows = model.diagnostics()

print(longer.objective)
print(f'{after_rows.loads} loads, {after_rows.solves} solves')
```

Two days cost twice one day. New snapshots add new columns, so the solver
loaded the model a second time.
[Starting from an earlier answer](howto/warm-start.md#start-from-an-earlier-answer) starts the new solve from an earlier answer.

## Next

In [the next tutorial](sweep.md) you solve a horizon too large for one solve,
a window at a time.
