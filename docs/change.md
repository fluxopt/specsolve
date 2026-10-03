# Change a model

The dispatch model from [Run a model](run.md), changed three ways, cheapest
first. Every block is a function of a **spec** (the YAML, as a `dict`) and its
**sources**, so blocks re-run in any order mean the same thing.

1. **New numbers**: `update`, and the solver keeps the model it has loaded.
2. **More rows**: the same math over a longer axis.
3. **New math**: patch the `dict` and re-run.

Every block on this page runs when the site is built, and what you see under it
is what it printed on this commit. A block that raises fails the build.

```python exec="true" source="material-block" session="loops"
import polars as pl
from mathspec import to_markdown, to_spec

import specsolve as sps

SPEC = 'examples/dispatch.yaml'
GENERATORS = ['wind', 'solar', 'gas']

sources = {
    'snapshot': pl.DataFrame({'snapshot': range(6)}),
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 60.0]}),
    'load': pl.DataFrame({'snapshot': range(6), 'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0]}),
}

print(to_markdown(SPEC))
```

## 1. New numbers

Build once, then `update` per run. New costs go onto the model HiGHS holds,
and the matrix is never handed over twice.

```python exec="true" source="material-block" result="text" session="loops"
model = sps.build(SPEC, sources)

rows = []
for gas_cost in (40.0, 60.0, 90.0):
    costs = pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, gas_cost]})
    rows.append({'gas_cost': gas_cost, 'objective': model.update({'cost': costs}).solve().objective})

sweep = pl.DataFrame(rows)
reused = model.diagnostics()

print(f'{reused.loads} model loaded, {reused.solves} solves')
print(sweep)
```

`loads` is 1 against `solves` of 3: three answers, one load.
`model.update(x).solve()` gives what `sps.solve(SPEC, sources | x)` gives,
always. The next block solves the last cost from scratch to show it.

```python exec="true" source="material-block" result="text" session="loops"
fresh = sps.solve(SPEC, sources | {'cost': costs}).objective
updated = sweep.filter(pl.col('gas_cost') == 90.0).item(0, 'objective')

print(f'updated {updated:,.1f} — fresh build {fresh:,.1f}')
```

## 2. More rows

A longer horizon is a longer table plus the index to match.

```python exec="true" source="material-block" result="text" session="loops"
horizon = pl.DataFrame(
    {
        'snapshot': range(12),
        'value': [90.0, 120.0, 150.0, 180.0, 140.0, 100.0, 95.0, 130.0, 160.0, 190.0, 150.0, 110.0],
    }
)

index = pl.DataFrame({'snapshot': range(12)})
schedule = model.update({'snapshot': index, 'load': horizon}).solve().primal('p')
grown = model.diagnostics()

print(f'{schedule.height} rows of p now, and {grown.loads} loads over {grown.solves} solves')
print(schedule.head())
```

`loads` is 2 now: new coordinates renumber the columns, so this model was
loaded from scratch. The answer is the same either way.
`schedule.pivot(on='generator', index='snapshot', values='value')` is the
wide view.

## 3. New math

`to_dict()` is the spec as data, and every verb takes a `dict`. An edit is a
key, and `to_spec` validates it again. Below, a ramp limit on gas, which
needs a parameter as well as a constraint.

```python exec="true" source="material-block" result="text" session="loops"
spec = to_spec(SPEC).to_dict()
spec['parameters']['ramp_max'] = {'dims': ['generator']}
spec['constraints']['ramp_up'] = {
    'dims': ['snapshot', 'generator'],
    'expression': 'p - shift(p, along=snapshot, offset=1) <= ramp_max',
}

ramp_max = pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 100.0, 20.0]})
base = sps.solve(SPEC, sources).objective
ramped = sps.solve(spec, sources | {'ramp_max': ramp_max}).objective

print(pl.DataFrame({'model': ['dispatch', 'dispatch + ramp limit'], 'objective': [base, ramped]}))
```

The limit binds: gas starts climbing early, and free wind is curtailed to
make room. The math re-renders from the patched spec:

```python exec="true" source="material-block" session="loops"
print(to_markdown(spec, legend=False, numbered=False))
```

An edit the language refuses is refused before any data is attached:

```python exec="true" source="material-block" result="text" session="loops"
typo = {
    **spec,
    'constraints': {
        **spec['constraints'],
        'peak': {'dims': ['snapshot'], 'expression': 'sum(p, over=generators) <= load'},
    },
}

try:
    sps.check(typo)
except sps.errors.LanguageError as exc:
    print(exc)
```

## What leaves the session

The spec, as a file you can diff and commit:

```python exec="true" source="material-block" result="yaml" session="loops"
print(to_spec(spec).to_yaml())
```

## Where next

| | |
|---|---|
| [Sweep a model](sweep.md) | the next tutorial: one model once per scenario, then window by window |
| [Warm-starting a re-solve](howto/warm-start.md) | keep the solver's work between solves, not only the model |
| [Fixing, relaxing and removing](howto/fix-relax-remove.md) | linopy's `fix`, `relax` and `remove_constraints`, as these three loops |
| [Debugging a wrong answer](howto/debug.md) | read the row a build produced, when the file looks right and the answer is not |
