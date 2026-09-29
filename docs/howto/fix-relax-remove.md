# Fixing, relaxing and removing

How to do what linopy's `fix`, `relax` and `remove_constraints` do. None is a
method here: a fix is data, and the other two are an edit to the spec
([Change a model](../change.md) teaches both).

| linopy | here | rebuild |
|---|---|---|
| `x.fix(v)` | both bounds read a parameter; write the same number into both | no, `update` |
| `x.relax()` | `domain:` in the declaration | yes |
| `remove_constraints` | drop the key from the spec | yes |

The spec is `examples/dispatch.yaml`. Every block runs when the site is
built, and a block that raises fails the build.

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

Give the variable two bound parameters, each total over its coordinates: a
frame holding only the pinned rows is a load error. Keep `p_max` as the
`where` mask, so a pin does not renumber the labels. Then write the pinned
value into both bounds with `update`:

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

One load for both answers. To fix a combination of variables, such as
`sum(p, over=generator) == target`, write a constraint instead: a combination
is not a bound.

## Relax integrality

Patch `domain:` and build again. An integer variable makes duals undefined,
and asking for one says so.

```python exec="true" source="material-block" result="text" session="lifecycle"
integral = to_spec(MODEL).to_dict()
integral['variables']['p']['domain'] = 'integer'

milp = sps.solve(integral, sources)
print(f'integer objective {milp.objective:,.1f}, has_primal {milp.has_primal}')

try:
    milp.dual('power_balance')
except sps.SpecsolveError as exc:
    print(exc)

relaxed = sps.solve(MODEL, sources)  # the same file, continuous as declared
print(relaxed.dual('power_balance'))
```

## Remove a constraint

`pop` the constraint's key from the spec. Below, the ramp limit from
[Change a model](../change.md#3-new-math), added and taken away.

```python exec="true" source="material-block" result="text" session="lifecycle"
ramped = to_spec(MODEL).to_dict()
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

To switch rows off with data instead, put a `where` on the constraint. The
declaration stays and builds no rows where the mask is false. An `update` that
changes the mask loads the model again, since the rows are renumbered.
`diagnostics().loads` counts the loads, and `omissions` counts the rows not
built.
