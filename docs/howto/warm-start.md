# Warm-starting a re-solve

How to start each solve in a loop from the work the solve before it did. It
suits a loop whose solves differ by a small step: a rolling horizon, a myopic
pathway, a search that inches. The reference is
[`Model.solve`](../reference/api.md#specsolve.Model.solve).

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

[`kept`](../reference/api.md#specsolve.Result.kept) says
what the solve actually kept. The first solve of a model keeps `'nothing'`,
because no work came before it. An `update` that moves a mask or a coordinate
set also gives `'nothing'`: the columns change, so the model is loaded again
([`Model.update`](../reference/api.md#specsolve.Model.update)).

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

## What a rebuild loses

**A rebuild carries no progress.** A cutting-plane master re-solved after
gaining a cut has gained a *row*, and a basis spans the model it was read
from. [#382](https://github.com/fluxopt/specsolve/issues/382) tracks that case.
