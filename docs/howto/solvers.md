# Choosing a solver

How to solve on a solver other than HiGHS, pass it options, or write the model
to a file. HiGHS ships with specsolve and is the default.

## See which solvers this build has

Name a solver that does not exist, and the refusal lists the ones that do:

```python exec="true" source="material-block" result="text"
import specsolve as sps

try:
    sps.solve('examples/dispatch.yaml', {}, solver_name='cplex')
except sps.errors.SpecsolveError as exc:
    print(exc)
```

## Install the solver

Every solver except HiGHS comes with the extra of the same name:

```bash
pip install "specsolve[gurobi]"
```

## Name it in the call

`solver_name=` goes on [`solve`](../reference/api.md#specsolve.solve),
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve) and
[`solve_over`](../reference/sweeps.md):

```python
result = sps.solve('dispatch.yaml', sources, solver_name='gurobi')
```

## Pass solver options

`solver_options=` goes to the solver verbatim, in its own vocabulary:

```python
result = sps.solve('dispatch.yaml', sources, solver_name='gurobi', solver_options={'TimeLimit': 60})
```

## A solver that cannot take the model

`solve` refuses a model the solver cannot take, such as SOS sets on HiGHS, and
names the solvers that do. To ask without solving, build and check:

```python
import specsolve as sps

model = sps.build('dispatch.yaml', sources)
model.check('gurobi')
```

The full table is
[what each sink takes](../reference/api.md#what-each-sink-takes).

## Write the model to a file

For a solver specsolve does not run, write the model. The suffix picks the
format, `.lp` or `.mps`:

```python
sps.write('dispatch.yaml', sources, 'dispatch.lp')
```
