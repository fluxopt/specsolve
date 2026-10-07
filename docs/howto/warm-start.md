# Warm-starting a re-solve

How to start each solve in a loop from the work the solve before it did. It
suits a loop whose solves differ by a small step: a rolling horizon, a myopic
pathway, a search that inches. The reference is
[`Model.solve`](../reference/api.md#specsolve.types.Model.solve).

## Keep the solver's progress

Build once, then solve each `update` with `keep='progress'`:

```python
import specsolve as sps

model = sps.build('dispatch.yaml', sources)
model.solve()
for numbers in steps:
    result = model.update(numbers).solve(keep='progress')
    print(result.kept)  # progress
```

[`kept`](../reference/api.md#specsolve.types.Result.kept) says
what the solve actually kept. The first solve of a model keeps `'nothing'`,
because no work came before it. An `update` that moves a mask or a coordinate
set also gives `'nothing'`: the columns change, so the model is loaded again
([`Model.update`](../reference/api.md#specsolve.types.Model.update)).

The default, `keep='solver'`, reuses the loaded model and discards the work.
The answer is the same under every `keep`; only the time changes.

## Check that it pays

Run the loop once with each `keep=` and read the clock the package keeps:

```python
for keep in ('solver', 'progress'):
    model = sps.build('dispatch.yaml', sources)
    for numbers in steps:
        assert model.update(numbers).solve(keep=keep).kept in {keep, 'nothing'}
    print(keep, model.diagnostics().seconds['solve'])
```

Take the faster one. `'nothing'` on every iteration means each update moved a
mask and the model was rebuilt, so the loop is paying for the build, not the
solve.

**`keep='progress'` can lose by an order of magnitude and win by a factor of
two**, so measure rather than guess. Over six updates on HiGHS
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

[`solve_over`](../reference/sweeps.md) takes the same `keep=`, and carries
from one slice to the next:

```python
axis = sps.EachWindow('hour', steps=24, lookahead=24, into='t')
sweep = sps.solve_over('horizon.yaml', sources, axis, keep='progress')
```

Under `executor=`, every slice is a first solve and keeps `'nothing'`
([running slices in parallel](../reference/sweeps.md#running-slices-in-parallel)).

## Time a cold solve

Pass `keep='nothing'`. It discards the held solver before the load, so no
basis, incumbent or solver-internal state survives. A benchmark needs that, and
so does comparing two sets of `solver_options`.

## Start from an earlier answer

`keep='progress'` carries the work only while the model's structure stays the
same. A cutting-plane master re-solved after gaining a cut has gained a *row*,
and a rebuild loads a fresh solver. Start an LP from an earlier answer instead:

```python
previous = master.solve(outputs={'basis'})
for cut in cuts:
    previous = master.update(cut).solve(start=previous, outputs={'basis'})
```

The engine lays the answer's basis onto the new build by coordinate. A
coordinate both builds hold keeps its status. A variable only the new build
holds starts at a bound, and a constraint only the new build holds starts not
binding, so a cut enters without moving the vertex. The answer to start from
must be solved with `outputs={'basis'}`. It can be live, loaded with
[`load_result`](../reference/api.md#specsolve.load_result) or read from an
archive, and it can come from another solver.

**It pays most where the model changes least.** An unchanged model started
from its own answer does no simplex work. Over a Benders run on the master of
`bench/warm_payoff.py`, with HiGHS, a start from the previous master saved 66%
to 73% of the simplex iterations at every size measured
([#1877](https://github.com/fluxopt/specsolve/pull/1877)). Whether it saves
time too depends on the model: on a small master, reading and matching the
basis costs more than the iterations it saves. Measure before relying on it.

## Start a mixed-integer solve from values

A mixed-integer model has no basis to start from, so `start=` gives it values
instead. The solver takes them as its first incumbent, which bounds the search
from the start. Pass an earlier answer, and its primal is the start:

```python
yesterday = sps.load_result('runs/monday')
today = sps.solve('commitment.yaml', sources, start=yesterday)
```

Or pass a table per variable, in the shape
[`primal`](../reference/api.md#specsolve.types.Result.primal) returns, from a
heuristic or a rule of thumb:

```python
on = pl.DataFrame({'unit': ['coal', 'gas'], 'hour': [0, 0], 'value': [1.0, 1.0]})
result = sps.solve('commitment.yaml', sources, start={'status': on})
```

A start can name some variables and some coordinates and leave out the rest.
The solver fills in what is missing, and repairs what does not fit, as far as
it can. A start is a hint: it never changes the optimum, only how soon a good
solution is found. A table that names no variable of the model, or whose
columns are not the variable's dims and `value`, is refused. So is one that
holds no coordinate the model has, and a table for an LP, which starts from a
basis.
