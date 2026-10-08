# Choosing a solver

How to solve on a solver other than HiGHS, pass it options, or write the model
to a file for a solver specsolve does not run. HiGHS ships with specsolve and
is the default. Nothing in the spec names a solver, so the call chooses one.

## See which solvers this build has

Name a solver that does not exist, and the refusal lists the ones that do.
This block runs when the site is built, so the list is this commit's:

```python exec="true" source="material-block" result="text"
import specsolve as sps

try:
    sps.solve('examples/dispatch.yaml', {}, solver_name='cplex')
except sps.errors.SpecsolveError as exc:
    print(exc)
```

The name is checked before any data is attached.

## Install the solver

Every solver except HiGHS comes with the extra of the same name:

```bash
pip install "specsolve[gurobi]"
```

A name the build knows, but whose package is not installed, raises
`ModuleNotFoundError` with the `pip install` line to run.

## Name it in the call

`solver_name=` goes on [`solve`](../reference/api.md#specsolve.solve),
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve) and
[`solve_over`](../reference/sweeps.md):

```python
result = sps.solve('dispatch.yaml', sources, solver_name='gurobi')
```

## Pass solver options

`solver_options=` goes to the solver verbatim, in its own vocabulary. A time
limit is `time_limit` on HiGHS and `TimeLimit` on Gurobi:

```python
result = sps.solve('dispatch.yaml', sources, solver_name='gurobi', solver_options={'TimeLimit': 60})
```

## Solve an LP by interior point without crossover

An LP can be solved by interior point and left there, without the crossover
that moves the answer to a vertex:

| solver | `solver_options=` |
|---|---|
| HiGHS | `{'solver': 'ipm', 'run_crossover': 'off'}` |
| Gurobi | `{'Method': 2, 'Crossover': 0}` |
| Xpress | `{'lpflags': 4, 'crossover': 0}` |

The answer is optimal to the solver's tolerance and carries duals. It ends on
no basis, so `outputs={'basis'}` gives none, and an LP
[started from it](warm-start.md#start-from-an-earlier-answer) starts from its
values rather than a basis.

**On HiGHS, measure before you turn it on.** On the benchmark models at rung
`m`, it was the fastest of the three ways only on `transport`, by 13%. Simplex
was the fastest on `dispatch` and `storage`. On `nodal`, interior point with
crossover was the fastest, and without crossover it took 20 times as long.
Every objective was within 7e-9, relative, of the simplex one
([#NNNN](https://github.com/fluxopt/specsolve/pull/NNNN)). Time each way on
your own model before you keep one.

## A solver that cannot take the model

Solvers take different constructs: HiGHS has no SOS sets, for example. `solve`
refuses a built model the solver cannot take, before the load, and the
refusal names the solvers that do take the construct. To ask without solving,
build and check:

```python
import specsolve as sps

model = sps.build('dispatch.yaml', sources)
model.check('gurobi')
```

The full table is
[what each sink takes](../reference/api.md#what-each-sink-takes).

The answer is read off the model the build produced, not off the file. A file
may declare a quadratic cost that the data prices at zero everywhere, or an
integer variable that a `where:` leaves without a column; that model is an LP,
and every solver takes it.

## Write the model to a file

For a solver or tool that specsolve does not run, write the model. The suffix
picks the format, and this build writes `.lp` and `.mps`:

```python
sps.write('dispatch.yaml', sources, 'dispatch.lp')
```

The two formats carry different constructs: MPS has no quadratic objective or
indicator, for example. The table above covers both.
