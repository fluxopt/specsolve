# A battery, over time

Add a battery to the model from [On or off](commitment.md), so that one
snapshot depends on the snapshot before it. Then solve the model many times
with one call: once for each scenario of the load, and window by window over a
longer horizon.

Every block on this page runs from the `examples/` folder of the repository.

## The battery

Copy `tutorial/commitment.yaml` to `tutorial/storage.yaml`, and make the
changes below:

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
- **`soc_next`** says that the level of a snapshot is the level of the
  snapshot before, plus 90% of the charge, minus the discharge. `shift` reads
  the snapshot before.
  [shift →](https://mathspec.readthedocs.io/en/latest/reference/language/operators/#shift)
- **`soc_first`** does the same for snapshot 0, which has no snapshot before
  it. It starts from the new parameter `soc_initial`.

## One day

The sources are those of the last tutorial, plus `soc_initial`. We keep the
generator data in its own `dict`, because the rest of the page reuses it:

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

The battery brings the cost down from 14000 to 10850. Look at what it does:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
battery = pl.concat(
    [result.primal(name).rename({'value': name}) for name in ('charge', 'discharge', 'soc')], how='align'
)
print(battery)
```

In snapshots 0 and 1, the battery stores wind that would otherwise go unused.
In snapshots 2 and 3 it gives the energy back. In snapshot 3 that is enough to
keep gas off.

## One solve for each scenario

What if the load is 30% higher? Put both loads in one table, with a
`scenario` column that the spec does not declare.
[`solve_over`](reference/sweeps.md) with
[`EachCoordinate`](reference/sweeps.md#the-axes) solves the model once for
each label of that column:

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

`record` has one row for each solve. The answers come back as one table, with
the `scenario` column added. Here is when gas is on:

```python exec="true" source="material-block" result="text" session="time" workdir="examples"
gas_on = scenarios.primal('on').filter(pl.col('generator') == 'gas')
print(gas_on.pivot(on='scenario', index='snapshot', values='value'))
```

In the cold scenario the battery cannot cover snapshot 3, and gas stays on.

## Window by window

A real horizon is often too long to solve at once. A rolling horizon solves it
in windows, one after the other, and hands the battery level from each window
to the next. Here the horizon is three days of the same four hours.

The load is over `hour` now, not `snapshot`.
[`EachWindow`](reference/sweeps.md#the-axes) cuts `hour` into windows, and
numbers the hours of each window from zero into `snapshot`. So snapshot 0 of
each window is the first hour of that window:

```python exec="true" source="material-block" session="time" workdir="examples"
days = pl.DataFrame({'hour': range(12), 'value': 3 * load['value'].to_list()})
hourly = fleet | {'load': days, 'soc_initial': 0.0}
```

Each window keeps three hours, and `lookahead=1` lets it see one hour more.
[`carry`](reference/sweeps.md#carrying-state-between-slices) copies the last
`soc` that a window keeps into the `soc_initial` of the next window:

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

`record` has one row for each window, keyed by the hour it starts at. The
answers come back over `hour`, as one schedule for all twelve hours. The
second table is the battery level at the last hour each window keeps, which is
what the next window starts from. The first window ends with 10 in the
battery. The window that starts at hour 3 spends those 10 in hour 3, and gas
stays off.

## What a window cannot see

A window does not know the hours after the ones it sees. Without lookahead,
the first window empties the battery in hour 2, and the next window needs gas
in hour 3. We compare the cost of the schedule for three lookaheads with the
cost of one solve over all twelve hours:

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
horizon. One hour of lookahead closes most of that gap. Three hours close all
of it, because every window then sees to the end of the day it is in.

## Next

You have grown one model from five parts to on-off generators and a battery.
You have solved it once, once for each scenario, and window by window. These
pages go further:

| | |
|---|---|
| [Sweeps and rolling horizons](reference/sweeps.md) | every axis, `carry`, and how a sweep is read |
| [Running a sweep in parallel](howto/parallel.md) | one window or scenario for each worker |
| [Debugging a wrong answer](howto/debug.md) | when a model solves and the number is wrong |
| [Examples](examples/index.md) | every model in the repository |
