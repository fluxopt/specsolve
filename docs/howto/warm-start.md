# Making a loop of solves faster

How to make each solve in a loop start from the work of the one before, and
how to build many small models faster. The reference for every start is
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

While an `update` keeps the loaded solver, each solve carries on from where the
last one ended. An `update` that moves a mask or a coordinate set loads the
solver again ([`Model.update`](../reference/api.md#specsolve.types.Model.update)),
and that solve begins from nothing. `start=None` begins from nothing every
time. The answer is the same either way; only the time changes.

## Check that it pays

Run the loop once each way and read the clock the package keeps:

```python
for start in ('previous', None):
    model = sps.build('dispatch.yaml', sources)
    model.solve()
    for numbers in steps:
        model.update(numbers).solve(start=start)
    print(start, model.diagnostics().seconds['solve'], model.diagnostics().loads)
```

Take the faster one. `loads` equal to the number of solves means each update
moved a mask, so the loop pays for loading the solver, not for the solve.

**Carrying on usually pays, and can lose by an order of magnitude.** Over six
updates on the bench's dispatch and storage models, it cut the wall time by
1.5× to 5.7× on HiGHS and Gurobi
([#1884](https://github.com/fluxopt/specsolve/pull/1884)). On a dispatch model
whose presolve cracks the problem outright, it cost **76.6 s against 4.3 s** on
HiGHS, because a solver that carries on skips its presolve
([#815](https://github.com/fluxopt/specsolve/pull/815)). No solver option gives
the presolve back.

To time a cold solve, as a benchmark or a comparison of `solver_options` needs,
close the model and build it again: `start=None` alone leaves solver state
behind.

## A sweep

[`solve_over`](../reference/sweeps.md) carries each slice on from the one
before it in the same way. Under an executor each slice builds its own model,
so each begins from nothing. `start=None` turns it off. To start a sweep from
an earlier one, pass that sweep as `start=`, and each slice starts from the
earlier slice of its key:

```python
monday = sps.load_archive('runs/monday.zip').sweep
tuesday = sps.solve_over('dispatch.yaml', tuesday_sources, axis, start=monday)
```

## Start from an earlier answer

The solver carries on only while the model's structure stays the same. A
cutting-plane master that gains a cut has gained a row, so it loads again.
Start it from an earlier answer that carries its basis instead:

```python
previous = master.solve(outputs={'basis'})
for cut in cuts:
    previous = master.update(cut).solve(start=previous, outputs={'basis'})
```

The engine lays the basis onto the new build by coordinate, so a cut enters
without moving the vertex. Over a Benders run with HiGHS, this saved 66% to
73% of the simplex iterations at every size measured
([#1877](https://github.com/fluxopt/specsolve/pull/1877)). On a small master,
matching the basis can cost more than the iterations it saves.

## Start a MIP from values

A mixed-integer model starts from values, which the solver takes as its first
incumbent. Pass an earlier answer, or values per variable under `'primal'`, in
any shape a parameter's source takes:

```python
yesterday = sps.load_result('runs/monday')
today = sps.solve('commitment.yaml', sources, start=yesterday)
today = sps.solve('commitment.yaml', sources, start={'primal': {'on': 1.0}})
```

A start may leave out declarations and coordinates, and never changes the
optimum. A basis you build yourself goes under `'variable_basis'` and
`'constraint_basis'`. [`Model.solve`](../reference/api.md#specsolve.types.Model.solve)
lists what each solver takes, and what a start may not hold.

## Build many small models faster

specsolve builds a model of at most 250,000 columns and 250,000 rows on
polars' in-memory engine, and a larger one on polars' default engine. A model
of 10,000 to 170,000 variables built in up to 32% less time in memory. At 10
million variables, in-memory took up to 35% more time and 36% more memory
([#1920](https://github.com/fluxopt/specsolve/pull/1920)).

To move the rest of a process onto the in-memory engine too, such as a sweep
of many small models and your own queries, set it for the process:

```python
import polars as pl

pl.Config.set_engine_affinity('in-memory')
```

`POLARS_ENGINE_AFFINITY=in-memory` in the environment does the same, and the
workers of a [parallel sweep](parallel.md) inherit it.
