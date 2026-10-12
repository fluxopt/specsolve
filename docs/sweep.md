# A horizon too large for one solve

A model over a year can be too large to solve at once. Solve it a window at a
time instead, and see what that costs. Every block runs from the `examples/`
folder.

## The model

`tutorial/storage.yaml` is the first model with two additions. A generator is
either on or off, and gas cannot run below 50 while it is on. A battery
carries its level from one snapshot to the next.

??? note "tutorial/storage.yaml"

    ```yaml
    --8<-- "examples/tutorial/storage.yaml"
    ```

Solve one day. The sources gain `p_min`, `no_load_cost` and the battery's
level at the start, `soc_initial`:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
import polars as pl

import specsolve as sps

generators = pl.read_csv('tutorial/data/generators.csv')
load = pl.read_csv('tutorial/data/load.csv')
fleet = {'generator': generators['generator']} | {
    name: generators.select('generator', value=name) for name in ('p_max', 'cost', 'p_min', 'no_load_cost')
}

result = sps.solve('tutorial/storage.yaml', fleet | {'snapshot': load['snapshot'], 'load': load, 'soc_initial': 0.0})
print(result.objective)
```

## A longer horizon is a larger model

Repeat the day to make a longer horizon, and build the model without solving
it. [`diagnostics`](reference/api.md#specsolve.types.Model.diagnostics) gives
its size:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
def horizon(days: int) -> dict:
    hours = days * load.height
    repeated = pl.DataFrame({'snapshot': range(hours), 'value': days * load['value'].to_list()})
    return fleet | {'snapshot': range(hours), 'load': repeated, 'soc_initial': 0.0}


sizes = []
for days in (1, 3, 30, 365):
    with sps.build('tutorial/storage.yaml', horizon(days)) as model:
        size = model.diagnostics()
    sizes.append({'days': days, 'columns': size.columns, 'rows': size.rows})
print(pl.DataFrame(sizes))
```

Each day adds 28 columns and 24 rows. A real model has many more generators
and hours than this one, and its solve gets slower as it grows.

## Window by window

A rolling horizon solves the horizon in windows. Here three days stand in for
a year, and the load is over `hour`:

```python exec="true" source="material-block" session="horizon" workdir="examples"
days = pl.DataFrame({'hour': range(12), 'value': 3 * load['value'].to_list()})
hourly = fleet | {'load': days, 'soc_initial': 0.0}
```

[`EachWindow`](reference/sweeps.md#the-axes) cuts `hour` into windows of
three hours, and numbers the hours of each window from zero into `snapshot`.
`lookahead=0` means a window sees only the hours it keeps.
[`carry`](reference/sweeps.md#carrying-state-between-slices) copies the last
`soc` of a window into the `soc_initial` of the next:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
windows = sps.solve_over(
    'tutorial/storage.yaml',
    hourly,
    sps.EachWindow('hour', steps=3, lookahead=0, into='snapshot', carry={'soc_initial': 'soc'}),
)
print(windows.metrics.select('hour_start', 'columns', 'rows'))
```

Each window has 21 columns, however long the horizon is.

## What a window cannot see

Solve the three days at once, and compare the cost of the two schedules. The
objective of a window counts only that window, so add up what the windows
kept:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
def schedule_cost(sweep: sps.types.Sweep) -> float:
    output = sweep.primal('p').join(generators, on='generator')
    on = sweep.primal('on').join(generators, on='generator')
    return (output['value'] * output['cost']).sum() + (on['value'] * on['no_load_cost']).sum()


whole = sps.solve('tutorial/storage.yaml', horizon(3))
print(f'one solve {whole.objective:.1f}, windows {schedule_cost(windows):.1f}')
```

The windows cost 12% more. To see why, look at the battery level at the end of
each window:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
print(windows.primal('soc').filter(pl.col('hour') % 3 == 2))
```

Every window ends with the battery empty. A window cannot see the hours after
it, so it spends all the stored energy, and the next window needs gas sooner.

## Lookahead

`lookahead` lets each window see more hours than it keeps, and solve them
too. Try none, one and three:

```python exec="true" source="material-block" result="text" session="horizon" workdir="examples"
rows = []
for lookahead in (0, 1, 3):
    sweep = sps.solve_over(
        'tutorial/storage.yaml',
        hourly,
        sps.EachWindow('hour', steps=3, lookahead=lookahead, into='snapshot', carry={'soc_initial': 'soc'}),
    )
    rows.append(
        {
            'lookahead': lookahead,
            'columns': sweep.metrics['columns'][0],
            'cost': round(schedule_cost(sweep), 1),
        }
    )
print(pl.DataFrame(rows))
```

One hour of lookahead closes most of the gap, and three hours close all of it.
Each hour also makes every window larger. The lookahead that is enough depends
on the model: here, it is how far ahead the battery has to plan.

## Next

These pages go further:

| | |
|---|---|
| [Sweeps and rolling horizons](reference/sweeps.md) | every axis, `carry`, and how a sweep is read |
| [Making a loop of solves faster](howto/warm-start.md) | each window starts from the answer of the window before |
| [Debugging a wrong answer](howto/debug.md) | when a model solves and the number is wrong |
| [Examples](examples/index.md) | every model in the repository |
