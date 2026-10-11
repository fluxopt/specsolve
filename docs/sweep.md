# A battery, over time

Add a battery to the model from [On or off](commitment.md), so that one
snapshot depends on the one before. Then solve it many times with one call:
once for each scenario, and window by window. Every block runs from the
`examples/` folder.

## The battery

Copy `tutorial/commitment.yaml` to `tutorial/storage.yaml`, and make these
changes:

```python exec="true" result="diff" workdir="examples"
import difflib
from pathlib import Path

before = Path('tutorial/commitment.yaml').read_text().splitlines(keepends=True)
after = Path('tutorial/storage.yaml').read_text().splitlines(keepends=True)
print(''.join(difflib.unified_diff(before, after, 'tutorial/commitment.yaml', 'tutorial/storage.yaml', n=1)))
```

- **Three new variables**: `charge`, `discharge` and `soc`, the level of the
  battery at the end of a snapshot.
- **The power balance** counts what the battery gives back and what it takes.
- **`soc_next`** says that the level is the level of the snapshot before
  (read with `shift`), plus 90% of the charge, minus the discharge.
  [shift →](https://mathspec.readthedocs.io/en/latest/reference/language/operators/#shift)
- **`soc_first`** does the same for snapshot 0, from the new parameter
  `soc_initial`.

## One day

The sources gain `soc_initial`. We keep the generator data in its own `dict`
for reuse:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
import polars as pl

import specsolve as sps

generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
fleet = {'generator': generators['generator']} | {
    name: generators.select('generator', value=name) for name in ('p_max', 'cost', 'p_min', 'no_load_cost')
}
sources = fleet | {'snapshot': load['snapshot'], 'load': load, 'soc_initial': 0.0}

result = sps.solve('tutorial/storage.yaml', sources)
print(result.objective)
```

The battery brings the cost down from 14000 to 10850. Here is what it does:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
battery = pl.concat(
    [result.primal(name).rename({'value': name}) for name in ('charge', 'discharge', 'soc')], how='align'
)
print(battery)
```

The battery stores spare wind in snapshots 0 and 1 and gives it back in 2
and 3, enough to keep gas off in snapshot 3.

## One solve for each scenario

What if the load is 30% higher? Put both loads in one table, with a
`scenario` column the spec does not declare.
[`solve_over`](reference/sweeps.md) with
[`EachCoordinate`](reference/sweeps.md#the-axes) solves once for each label:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
loads = pl.concat(
    [
        load.with_columns(scenario=pl.lit('mild')),
        load.with_columns(scenario=pl.lit('cold'), value=1.3 * pl.col('value')),
    ]
)
scenarios = sps.solve_over('tutorial/storage.yaml', sources | {'load': loads}, sps.EachCoordinate('scenario'))
print(scenarios.record.select('scenario', 'objective'))
```

The answers come back as one table, with the `scenario` column added. Here is
when gas is on:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
gas_on = scenarios.primal('on').filter(pl.col('generator') == 'gas')
print(gas_on.pivot(on='scenario', index='snapshot', values='value'))
```

In the cold scenario the battery cannot cover snapshot 3, and gas stays on.

## Window by window

A rolling horizon solves a long horizon in windows, and hands the battery level
from each window to the next. Here the horizon is three days of the same four
hours. The load is over `hour` now.
[`EachWindow`](reference/sweeps.md#the-axes) cuts `hour` into windows, and
numbers the hours of each window from zero into `snapshot`:

```python exec="true" source="material-block" session="time" workdir="examples"
days = pl.DataFrame({'hour': range(12), 'value': 3 * load['value'].to_list()})
hourly = fleet | {'load': days, 'soc_initial': 0.0}
```

Each window keeps three hours, and `lookahead=1` lets it see one more.
[`carry`](reference/sweeps.md#carrying-state-between-slices) copies the last
`soc` a window keeps into the `soc_initial` of the next:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
rolling = sps.solve_over(
    'tutorial/storage.yaml',
    hourly,
    sps.EachWindow('hour', steps=3, lookahead=1, into='snapshot'),
    carry={'soc_initial': 'soc'},
)
print(rolling.record.select('hour_start', 'objective'))
print(rolling.primal('soc').filter(pl.col('hour') % 3 == 2))
```

`record` has one row for each window. The second table is the battery level
that each window hands on. The first window ends with 10 in the battery, and
the next window spends it in hour 3, so gas stays off.

## What a window cannot see

Without lookahead, the first window empties the battery in hour 2, and the
next window needs gas in hour 3. We compare three lookaheads with one solve
over all twelve hours:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
def schedule_cost(sweep: sps.types.Sweep) -> float:
    output = sweep.primal('p').join(generators, on='generator')
    on = sweep.primal('on').join(generators, on='generator')
    return (output['value'] * output['cost']).sum() + (on['value'] * on['no_load_cost']).sum()


rows = []
for lookahead in (0, 1, 3):
    sweep = sps.solve_over(
        'tutorial/storage.yaml',
        hourly,
        sps.EachWindow('hour', steps=3, lookahead=lookahead, into='snapshot'),
        carry={'soc_initial': 'soc'},
    )
    rows.append({'lookahead': lookahead, 'cost': round(schedule_cost(sweep), 1)})

whole = sps.solve(
    'tutorial/storage.yaml',
    fleet | {'snapshot': range(12), 'load': days.rename({'hour': 'snapshot'}), 'soc_initial': 0.0},
)
print(pl.DataFrame(rows))
print(whole.objective)
```

Without lookahead, the schedule costs 12% more than one solve over the whole
horizon. One hour closes most of that gap. Three hours close all of it,
because every window then sees to the end of its day.

## Next

These pages go further:

| | |
|---|---|
| [Sweeps and rolling horizons](reference/sweeps.md) | every axis, `carry`, and how a sweep is read |
| [Running a sweep in parallel](howto/parallel.md) | one window or scenario for each worker |
| [Debugging a wrong answer](howto/debug.md) | when a model solves and the number is wrong |
| [Examples](examples/index.md) | every model in the repository |
