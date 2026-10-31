# What a change is worth

How to read from one solve what a small change would be worth: to a
constraint, to a variable's bound, or to a value in your data. Each answer is
a rate, in objective units per unit of the thing you change.

| You ask | The change is to | Read |
|---|---|---|
| What does one more unit of load cost in each snapshot? | a constraint | [`dual`](../reference/api.md#specsolve.types.Result.dual) |
| How much cheaper would gas have to be to run? | a variable at its bound | [`reduced_cost`](../reference/api.md#specsolve.types.Result.reduced_cost) |
| Where is more capacity worth most? Which input matters? | a value in your data | [`parameter_dual`](../reference/api.md#specsolve.types.Result.parameter_dual) |

The examples solve [dispatch](../examples/dispatch.md) with the data of
[Your data](../tables.md). A reduced cost is read only where the solve asks for
it:

```python exec="true" source="material-block" result="text" session="prices"
import polars as pl

import specsolve as sps

generators = pl.read_csv('examples/tutorial/data/generators.csv')
load = pl.read_csv('examples/tutorial/data/load.csv')
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
    'load': load,
}
result = sps.solve('examples/dispatch.yaml', sources, outputs={'reduced_cost'})
print(result.objective)
```

## The price of a constraint

`dual` gives the rate at which the objective rises as a constraint's right side
rises. For `power_balance`, that is the price of one more unit of load:

```python exec="true" source="material-block" result="text" session="prices"
print(result.dual('power_balance'))
```

Wind alone meets the load in snapshot 0, so a unit there costs 10. Gas sets
the price at 50 in the other snapshots.

## What a variable's bound costs

`reduced_cost` gives the rate at which the objective rises as a variable's
binding bound rises. A variable between its bounds has zero:

```python exec="true" source="material-block" result="text" session="prices"
print(result.reduced_cost('p'))
```

Gas sits at zero in snapshot 0 with a reduced cost of 40. It runs there only
once it is 40 cheaper. Wind sits at its capacity in the other snapshots, and
each unit of room there saves 40.

## What a value in your data is worth

`parameter_dual` gives the dual of each value of a parameter: the rate at
which the objective rises with it, in the shape of the parameter's own table:

```python exec="true" source="material-block" result="text" session="prices"
print(result.parameter_dual('p_max'))
print(result.parameter_dual('cost'))
```

One more unit of wind capacity saves 120: 40 in each of the three snapshots
where it runs at its limit. A rise of one in a generator's cost adds its
output: 300 for wind and 150 for gas.

A parameter dual reads every place the model uses the value: an objective
coefficient, a bound, a coefficient in a row, a right side. Where the value
stands alone on a right side, it is the dual: `parameter_dual('load')` equals
`dual('power_balance')`.

Solar has no capacity, so the file builds no output for it, yet its rate is
-75. Its first unit replaces gas in the same three snapshots, at 25 less each.
A parameter dual prices what a rise would switch on against the prices this
solve already has, and needs no second solve.

A rise can also switch on something that moves the objective at once, such
as a generator whose bound is a fixed 100 rather than its `p_max`. The rate is
then `-inf` or `inf`, and a warning names the rows. One solve cannot decide
every threshold. Where it cannot, the rate is that of the model as built, and
a warning names the threshold. Ask that one with
[`update`](../reference/api.md#specsolve.types.Model.update) and a re-solve.

## Which input matters most

A rate is per unit of its own input, so a capacity and a cost cannot be
ranked by their rates. Multiply each rate by its value: the sum is what a 1%
rise of every value of that input changes the objective by.

```python exec="true" source="material-block" result="text" session="prices"
rows = []
for name in ('cost', 'load', 'p_max'):
    rate = result.parameter_dual(name)
    keys = [c for c in rate.columns if c != 'value']
    data = sources[name].rename({'value': 'data'})
    one_percent = rate.join(data, on=keys).select(pl.col('value') * pl.col('data')).sum().item() / 100
    rows.append({'input': name, 'one_percent': one_percent})

print(pl.DataFrame(rows).sort(pl.col('one_percent').abs(), descending=True))
```

The load forecast moves the cost most: 1% more load costs 201. 1% higher
costs add 105, and 1% more capacity saves 96.

## A rate or a sweep

A rate holds while the change leaves the same generators at the same limits.
A larger change moves them, and the rate with them: in
[Change it and solve again](../change.md#new-numbers), each step of 40 more
wind saves less than the one before. For a change that large, solve each case
with [Solving many scenarios](scenarios.md). Rank the inputs as above first,
and sweep the ones that matter.
