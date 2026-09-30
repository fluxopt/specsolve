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
except sps.SpecsolveError as exc:
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
[`Model.solve`](../reference/api.md#specsolve.Model.solve) and
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

## Check a spec against a solver before solving

Solvers take different constructs: HiGHS has no SOS sets, for example.
`check` with `sink=` refuses a spec the solver cannot take, and needs no data:

```python
sps.check('dispatch.yaml', sink='highs')
```

The refusal names the solvers that do take the construct. The full table is
[what each sink takes](../reference/api.md#what-each-sink-takes).

## Write the model to a file

For a solver or tool that specsolve does not run, write the model. The suffix
picks the format, and this build writes `.lp` and `.mps`:

```python
sps.write('dispatch.yaml', sources, 'dispatch.lp')
```

The two formats carry different constructs: MPS has no quadratic objective or
indicator, for example. The table above covers both.

A written column is `x0` and a row is `c0` unless you ask for names. With
`names=True`, each column and row is named by its declaration and coordinate,
such as `p(0,wind)` and `balance(0)`. Use it when the tool that reads the file
finds a variable by name, as JuMP's `variable_by_name` does:

```python
sps.write('dispatch.yaml', sources, 'dispatch.lp', names=True)
```

A label keeps its letters, digits and ``!"#$%&'.;?@`{|}~``. Every other
character is written as `_`, because an LP reader refuses it, so `north sea`
becomes `north_sea`. A declaration with no dims is written `total()`. When two
labels of one declaration become the same name, the write is refused and the
error names both labels.

## Hand the model to pyomo

To extend the model in pyomo, or solve it with a solver pyomo runs, export the
built model. It needs the `[pyomo]` extra:

```python
import pyomo.environ as pyo

with sps.build('dispatch.yaml', sources) as model:
    m = model.to_pyomo()
pyo.SolverFactory('appsi_highs').solve(m)
m.p[0, 'wind'].value
```

Each variable is a `Var` and each constraint a `Constraint`, named as the spec
declares them and indexed by their labels. A coordinate that a `where` removed
has no entry. A row holds the numbers the build computed, so new data means a
new export.

A spec may give a variable and a constraint the same name, and a pyomo model
has one namespace. The later one takes its kind as a suffix, such as
`m.start_up_constraint` beside `m.start_up`. So does a declaration named
`dual`, `rc` or `slack`, because pyomo's solvers write their results there.
