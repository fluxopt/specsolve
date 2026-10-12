# Starting from an earlier answer

How to start each solve in a loop from the one before, or from an answer you
kept. The reference is
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve).

## Carry on from the solve before

Build once, then update and solve:

```python
import specsolve as sps

model = sps.build('dispatch.yaml', sources)
model.solve()
for numbers in steps:
    result = model.update(numbers).solve()
```

Each solve carries on from the last while `update` keeps the loaded solver. An
`update` that moves a mask or a coordinate set loads the solver again
([`Model.update`](../reference/api.md#specsolve.types.Model.update)).
`start=None` begins from nothing every time. The answer is the same either way.

## Check that it pays

Run the loop once each way and read the clock:

```python
for start in ('previous', None):
    model = sps.build('dispatch.yaml', sources)
    model.solve()
    for numbers in steps:
        model.update(numbers).solve(start=start)
    print(start, model.diagnostics().seconds['solve'], model.diagnostics().loads)
```

Take the faster one. `loads` equal to the number of solves means each update
moved a mask.

**Carrying on usually pays, and can lose by an order of magnitude.** It cut
wall time by 1.5× to 5.7× on the bench's dispatch and storage models
([#1884](https://github.com/fluxopt/specsolve/pull/1884)). On a model whose
presolve cracks the problem, it cost **76.6 s against 4.3 s** on HiGHS, because
a solver that carries on skips its presolve
([#815](https://github.com/fluxopt/specsolve/pull/815)).

To time a cold solve, close the model and build it again: `start=None` alone
leaves solver state behind.

## A sweep

[`solve_over`](../reference/sweeps.md) carries each slice on from the one
before it, except under an executor. `start=None` turns it off. Pass an earlier
sweep as `start=` to start each slice from the earlier slice of its key:

```python
monday = sps.load_archive('runs/monday.zip').sweep
tuesday = sps.solve_over('dispatch.yaml', tuesday_sources, axis, start=monday)
```

## Start from an earlier answer

A model that gains a row, such as a cutting-plane master that gains a cut,
loads again. Start it from an earlier answer that carries its basis:

```python
previous = master.solve(outputs={'basis'})
for cut in cuts:
    previous = master.update(cut).solve(start=previous, outputs={'basis'})
```

Over a Benders run with HiGHS, this saved 66% to 73% of the simplex iterations
([#1877](https://github.com/fluxopt/specsolve/pull/1877)).

## Start a MIP from values

The solver takes the values as its first incumbent. Pass an earlier answer, or
values per variable under `'primal'`:

```python
yesterday = sps.load_result('runs/monday')
today = sps.solve('commitment.yaml', sources, start=yesterday)
today = sps.solve('commitment.yaml', sources, start={'primal': {'on': 1.0}})
```

A start never changes the optimum.
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve) lists what each
solver takes.
