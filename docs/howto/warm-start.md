# Warm-starting a re-solve

How to start each solve in a loop from the work the solve before it did. It
suits a loop whose solves differ by a small step: a rolling horizon, a myopic
pathway, a search that inches. The reference is
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve).

## Start each solve from the one before

Build once, then pass each answer to the next solve as `start=`:

```python
import specsolve as sps

model = sps.build('dispatch.yaml', sources)
result = model.solve()
for numbers in steps:
    result = model.update(numbers).solve(start=result)
```

While an `update` keeps the loaded solver, the solver carries on from where
the last solve ended, and nothing is read or matched. An `update` that moves a
mask or a coordinate set loads the solver again
([`Model.update`](../reference/api.md#specsolve.types.Model.update)), and the
answer is then matched onto the new build by coordinate, as the sections below
describe. Without `start=`, each solve begins from nothing on the solver still
loaded. The answer is the same either way; only the time changes.

## Check that it pays

Run the loop once with each and read the clock the package keeps:

```python
for warm in (False, True):
    model = sps.build('dispatch.yaml', sources)
    result = model.solve()
    for numbers in steps:
        result = model.update(numbers).solve(start=result if warm else None)
    print(warm, model.diagnostics().seconds['solve'], model.diagnostics().loads)
```

Take the faster one. `loads` equal to the number of solves means each update
moved a mask and the model was loaded again, so the loop is paying for the
load, not the solve.

**Carrying on can lose by an order of magnitude and win by a factor of two**,
so measure rather than guess. Over six updates on HiGHS
([#815](https://github.com/fluxopt/specsolve/pull/815)), carrying the solver's
work cost **76.6 s against 4.3 s** on a dispatch model whose presolve cracks
the problem outright, an 18× loss, and **111.2 s against 213.9 s** on a
storage model whose cyclic recurrence presolve cannot crack, a 1.9× win. It
pays where the model is hard for its solver's preprocessing *and* consecutive
solves differ by a small step. The answer does not change either way: across
both models the objectives agreed to 2e-15 relative. No solver option reaches
the same thing; on both solvers that ship, an option asking for it did not
produce it (#815).

## Warm-start a sweep

[`solve_over`](../reference/sweeps.md) takes `start='previous'`, which starts
each slice from the one before it. Where the update between them keeps the
solver, the solver carries on:

```python
axis = sps.EachWindow('hour', steps=24, lookahead=24, into='t')
sweep = sps.solve_over('horizon.yaml', sources, axis, start='previous')
```

Each slice then depends on the one before, so `start='previous'` runs only
without an executor
([running slices in parallel](../reference/sweeps.md#running-slices-in-parallel)).

To start a sweep from an earlier one, pass that sweep as `start=`. Each slice
starts from the earlier slice of its key, under an executor too:

```python
monday = sps.load_archive('runs/monday.zip').sweep
tuesday = sps.solve_over('dispatch.yaml', tuesday_sources, axis, start=monday)
```

A table passed as `start=` is cut by the axis, as a source is. Write it over
the sliced dimension, such as `snapshot` for `EachWindow`, and each window
takes the rows of the coordinates it covers.

## Time a cold solve

Pass no `start=`. The loaded solver forgets the work the last solve did, so
the solve begins from nothing. To discard the loaded solver as well, so that
no solver-internal state survives, close the model and build it again. A
benchmark needs that, and so does comparing two sets of `solver_options`.

## Start from an earlier answer

The solver carries on only while the model's structure stays the same. A
cutting-plane master re-solved after gaining a cut has gained a *row*, and the
update loads a fresh solver. Start an LP from an earlier answer that carries
its basis:

```python
previous = master.solve(outputs={'basis'})
for cut in cuts:
    previous = master.update(cut).solve(start=previous, outputs={'basis'})
```

The engine lays the answer's basis onto the new build by coordinate. A
coordinate both builds hold keeps its status. A variable only the new build
holds starts at a bound, and a constraint only the new build holds starts not
binding, so a cut enters without moving the vertex. The answer can be live,
loaded with [`load_result`](../reference/api.md#specsolve.load_result) or read
from an archive, and it can come from another solver. An answer solved without
`outputs={'basis'}`, or by interior point with crossover off, carries no basis,
and an LP then starts from its values, as the next section describes.

**It pays most where the model changes least.** An unchanged model started
from its own answer does no simplex work. Over a Benders run on the master of
`bench/warm_payoff.py`, with HiGHS, a start from the previous master saved 66%
to 73% of the simplex iterations at every size measured
([#1877](https://github.com/fluxopt/specsolve/pull/1877)). Whether it saves
time too depends on the model: on a small master, reading and matching the
basis costs more than the iterations it saves. Measure before relying on it.

## Start from values

A mixed-integer model starts from values: the solver takes them as its first
incumbent, which bounds the search from the start. Pass an earlier answer, and
its primal is the start:

```python
yesterday = sps.load_result('runs/monday')
today = sps.solve('commitment.yaml', sources, start=yesterday)
```

Or pass values per variable under `'primal'`, from a heuristic or a rule of
thumb, in any shape a parameter's source takes over the variable's dims: a
table in the shape [`primal`](../reference/api.md#specsolve.types.Result.primal)
returns, a parquet path, a `{label: value}` map, or one number for every
coordinate:

```python
on = pl.DataFrame({'unit': ['coal', 'gas'], 'hour': [0, 0], 'value': [1.0, 1.0]})
result = sps.solve('commitment.yaml', sources, start={'primal': {'status': on}})
result = sps.solve('commitment.yaml', sources, start={'primal': {'status': 1.0}})
```

**An LP takes values only where the solver uses them.** HiGHS starts an LP
from values that give every column one. Gurobi takes values for an LP, and the
solve warns, because no gain from them is known. Xpress takes values for an LP
only where they give every column one, and refuses values that leave a column
out.

## Start from a basis you give

A basis is a status per coordinate, so it can come from anywhere: a file, a
rule, another tool. Pass a table per variable under `'variable_basis'` and per
constraint under `'constraint_basis'`, in the shape
[`variable_basis`](../reference/api.md#specsolve.types.Result.variable_basis)
returns, or a parquet path to one. The status is one of `basic`, `at_lower`,
`at_upper`, `fixed` and `superbasic`:

```python
loose = pl.DataFrame({'snapshot': [0, 1], 'value': ['basic', 'basic']})
result = sps.solve('dispatch.yaml', sources, start={'constraint_basis': {'line_limit': loose}})
```

A basis starts an LP only. Given both a basis and values, an LP starts from
the basis, and a mixed-integer model from the values. The engine repairs a
basis that does not fit: a status at a bound the variable lacks moves to one it
has, and the count of basic entries is made one per constraint.

## What a start may leave out

A start can name some declarations and some coordinates and leave out the
rest. The solver fills in what is missing, and repairs what does not fit, as
far as it can. A start is a hint: it never changes the optimum, only how soon
a good solution is found. Each table is read and checked as a parameter's
source is. The solve refuses, before the solver loads:

- **A key other than the three readers.** `start=` takes `'primal'`,
  `'variable_basis'` and `'constraint_basis'`.
- **A table for a declaration the model lacks**, or one a parameter over the
  same dims would be refused for: a missing column, a label the dimension
  lacks, a coordinate given twice.
- **A basis status outside the five words.**
- **A basis alone for a mixed-integer model**, which starts from values.
- **A start that lands on no coordinate the model holds.**
