# Fixing, relaxing and removing

How to do what linopy's `fix`, `relax` and `remove_constraints` do. A fix is
data, as in [Change it and solve again](../change.md). The other two are an
edit to the spec.

| linopy | here | rebuild |
|---|---|---|
| `x.fix(v)` | both bounds read a parameter; write the same number into both | no, `update` |
| `x.relax()` | `domain:` in the declaration | yes |
| `remove_constraints` | drop the key from the spec | yes |

The spec is `examples/dispatch.yaml`.

```python exec="true" source="material-block" session="lifecycle"
import polars as pl
from mathspec import to_spec

import specsolve as sps

MODEL = 'examples/dispatch.yaml'
GENERATORS = ['wind', 'solar', 'gas']

sources = {
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}),
    'snapshot': pl.DataFrame({'snapshot': range(6)}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 60.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
}
```

## Fix a variable

Give the variable two bound parameters, each total over its coordinates.
Keep `p_max` as the `where` mask. Then write the pinned value into both bounds
with `update`:

```python exec="true" source="material-block" result="text" session="lifecycle"
pinnable = to_spec(MODEL).to_dict()
pinnable['parameters']['p_lo'] = {'dims': ['snapshot', 'generator']}
pinnable['parameters']['p_hi'] = {'dims': ['snapshot', 'generator']}
pinnable['variables']['p']['bounds'] = {'lower': 'p_lo', 'upper': 'p_hi'}

grid = pl.DataFrame({'snapshot': range(6)}).join(pl.DataFrame({'generator': GENERATORS}), how='cross')
p_lo = grid.with_columns(value=pl.lit(0.0))
p_hi = grid.join(sources['p_max'], on='generator')

pinned = sps.build(pinnable, sources | {'p_lo': p_lo, 'p_hi': p_hi})
unpinned = pinned.solve().objective

hold = pl.when(pl.col('generator') == 'gas').then(60.0).otherwise(pl.col('value'))
held = pinned.update({'p_lo': p_lo.with_columns(value=hold), 'p_hi': p_hi.with_columns(value=hold)}).solve().objective

pinning = pinned.diagnostics()
print(f'{pinning.loads} loads over {pinning.solves} solves — a pin moves bounds, not labels')
print(pl.DataFrame({'gas': ['free to dispatch', 'held at 60'], 'objective': [unpinned, held]}))
```

One load for both answers. To fix a combination of variables, write a
constraint instead.

## Relax integrality

Patch `domain:` and build again. An integer model has no duals.

```python exec="true" source="material-block" result="text" session="lifecycle"
integral = to_spec(MODEL).to_dict()
integral['variables']['p']['domain'] = 'integer'

milp = sps.solve(integral, sources)
print(f'integer objective {milp.objective:,.1f}, has_primal {milp.has_primal}')

try:
    milp.dual('power_balance')
except sps.errors.SpecsolveError as exc:
    print(exc)

relaxed = sps.solve(MODEL, sources)  # the same file, continuous as declared
print(relaxed.dual('power_balance'))
```

## Prices from a MIP

To price a MIP, fix its integer decisions at the answer and solve again. What
is left is an LP with the same optimum, and its duals are the prices for those
decisions. The model is `examples/tutorial/commitment.yaml`, where `on` says
whether a generator is on. `on` becomes continuous, and two bounds hold it at
the MIP's answer:

```python exec="true" source="material-block" result="text" session="lifecycle"
UNITS = 'examples/tutorial/commitment.yaml'

generators = pl.read_csv('examples/tutorial/data/generators.csv')
load = pl.read_csv('examples/tutorial/data/load.csv')
units = {'generator': generators['generator'], 'snapshot': load['snapshot'], 'load': load} | {
    name: generators.select('generator', value=name) for name in ('p_max', 'cost', 'p_min', 'no_load_cost')
}
mip = sps.solve(UNITS, units)

fixable = to_spec(UNITS).to_dict()
fixable['parameters']['on_fixed'] = {'dims': ['snapshot', 'generator'], 'missing': 'absent'}
fixable['variables']['on'] |= {'domain': 'continuous', 'bounds': {'lower': 'on_fixed', 'upper': 'on_fixed'}}

priced = sps.solve(fixable, units | {'on_fixed': mip.primal('on')})
print(f'MIP {mip.objective}, fixed LP {priced.objective}')
print(priced.dual('power_balance'))
```

Gas sets the price in snapshot 2. In the other snapshots wind is at the margin.
The prices hold the on-off decisions fixed, so they do not include the cost of
keeping gas on.

## Remove a constraint

`pop` the constraint's key from the spec:

```python exec="true" source="material-block" result="text" session="lifecycle"
ramped = to_spec(MODEL).to_dict()
ramped['dimensions']['snapshot']['ordered'] = True
ramped['parameters']['ramp_max'] = {'dims': ['generator']}
ramped['constraints']['ramp_up'] = {
    'dims': ['snapshot', 'generator'],
    'expression': 'p - shift(p, along=snapshot, offset=1) <= ramp_max',
}
data = sources | {'ramp_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 100.0, 20.0]})}

with_ramp = sps.solve(ramped, data).objective
ramped['constraints'].pop('ramp_up')
without_ramp = sps.solve(ramped, data).objective

print(pl.DataFrame({'model': ['with ramp_up', 'ramp_up removed'], 'objective': [with_ramp, without_ramp]}))
```

To switch rows off with data instead, put a `where` on the constraint. An
`update` that changes the mask loads the model again.
