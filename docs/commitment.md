# On or off

Change the model itself: a generator is now either on or off. Gas cannot run
below 50, and costs 500 for each snapshot it is on. Every block runs from the
`examples/` folder.

## The new rules

Copy `dispatch.yaml` to `tutorial/commitment.yaml`, and make these changes:

```python exec="true" result="diff" workdir="examples"
import difflib
from pathlib import Path

before = Path('dispatch.yaml').read_text().splitlines(keepends=True)
after = Path('tutorial/commitment.yaml').read_text().splitlines(keepends=True)
print(''.join(difflib.unified_diff(before, after, 'dispatch.yaml', 'tutorial/commitment.yaml', n=1)))
```

- **Two new parameters**, `p_min` and `no_load_cost`: the least output of a
  generator that is on, and its cost for each snapshot on.
- **A new variable `on`**, with `domain: binary`, so it is 0 or 1.
  [Variables →](https://mathspec.readthedocs.io/en/latest/reference/language/declarations/#variables)
- **Two new constraints.** `p_upper` keeps a generator that is off at zero,
  and `p_lower` keeps a generator that is on at its minimum or above.
- **The objective** adds the no-load cost of each generator that is on.

Check the file:

```python exec="true" source="material-block" session="commitment" workdir="examples"
import polars as pl

import specsolve as sps

program = sps.check('tutorial/commitment.yaml')
```

## The new data

`generators.csv` already has the two new columns, so the sources gain two
lines:

```python exec="true" source="material-block" session="commitment" workdir="examples"
generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
sources = {
    'generator': generators['generator'],
    'snapshot': load['snapshot'],
    'p_max': generators.select('generator', value='p_max'),
    'cost': generators.select('generator', value='cost'),
    'p_min': generators.select('generator', value='p_min'),
    'no_load_cost': generators.select('generator', value='no_load_cost'),
    'load': load,
}
```

## Solve

A model with a binary variable is a mixed-integer program (MIP). You solve it
the same way:

```python exec="true" source="material-block" result="text" session="commitment" workdir="examples"
result = sps.solve('tutorial/commitment.yaml', sources)
print(result.objective)
```

The cost goes up from 10500 to 14000. To see why, pivot the output:

```python exec="true" source="material-block" result="text" session="commitment" workdir="examples"
print(result.primal('p').pivot(on='generator', index='snapshot', values='value'))
```

In snapshot 0, wind meets the load alone and gas stays off. In snapshots 1
and 3, the load needs less than 50 of gas, but gas runs at its minimum of 50
and wind gives way. In snapshot 2, gas runs above its minimum.

## No prices from a MIP

A MIP has no dual prices, and specsolve says so:

```python exec="true" source="material-block" result="text" session="commitment" workdir="examples"
try:
    result.dual('power_balance')
except sps.errors.SpecsolveError as error:
    print(error)
```

To get prices, fix the on-off decisions and solve again, as
[Fixing, relaxing and removing](howto/fix-relax-remove.md) shows.

## Next

In the next tutorial you add a battery, and solve over a horizon
longer than one solve can see.
