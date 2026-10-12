# Decomposition, as evidence

This page shows that the language can express a Benders decomposition and reach
the right answer, for anyone decomposing a model in specsolve or asking for a
driver that does it.

**specsolve ships no decomposition driver, and
[#596](https://github.com/fluxopt/specsolve/issues/596) settled that it will not
own one.** The loop is the caller's, and so are its failure modes. Every block
below is validated against
[`examples/benders/`](https://github.com/fluxopt/specsolve/blob/main/examples/benders/run.py).

## Why anyone wants it

A model too large to solve is usually made smaller: representative days instead
of a year, forty nodes instead of three hundred, one weather year instead of
many. Each answers a *different question*, and none bounds how wrong it is for
the one you asked. Decomposition answers the question you asked, with a
**gap**: stop at 1% and you are within 1%.

## The problem, whole

Choose generator capacity, then dispatch it, with investment and operation
decided together. The capacity choice is small and the dispatch is large, which
is what makes it worth decomposing.

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  invest: {dims: [generator]}
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
variables:
  cap:
    dims: [generator]
    bounds: {lower: 0, upper: 100}
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    dims: [snapshot, generator]
    expression: p <= cap * avail
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(cap * invest) + sum(p * cost)
```

## The split is one substitution

The subproblem is the same dispatch at a capacity someone else chose: **`cap`
stops being a variable and becomes a parameter.**

```yaml
dimensions:
  snapshot: {dtype: int}
  generator: {dtype: str}
parameters:
  cost: {dims: [generator]}
  load: {dims: [snapshot]}
  avail: {dims: [snapshot, generator]}
  cap_hat: {dims: [generator]}          # was `cap`, a variable
variables:
  p:
    dims: [snapshot, generator]
    bounds: {lower: 0}
constraints:
  capacity:
    dims: [snapshot, generator]
    expression: p <= cap_hat * avail
  balance:
    dims: [snapshot]
    expression: sum(p, over=generator) >= load
objective:
  sense: minimize
  expression: sum(p * cost)
```

Three things are *not* here: no `invest`, no `cap` bounds, no investment term.
The subproblem does not know it is part of anything.

## The master, where a cut is data

The master keeps the capacity decision. It stands in `theta` for the dispatch
it can no longer see: one variable holding what operating that capacity will
cost. Cuts teach the master what `theta` is:

```yaml
dimensions:
  generator: {dtype: str}
  cut: {dtype: int}
  fcut: {dtype: int}
parameters:
  invest: {dims: [generator]}
  cut_const: {dims: [cut]}
  cut_slope: {dims: [cut, generator]}
  fcut_const: {dims: [fcut]}
  fcut_slope: {dims: [fcut, generator]}
variables:
  cap:
    dims: [generator]
    bounds: {lower: 0, upper: 100}
  theta:
    dims: []
    bounds: {lower: 0}
constraints:
  optimality_cut:
    dims: [cut]
    expression: theta >= cut_const + sum(cut_slope * cap, over=generator)
  feasibility_cut:
    dims: [fcut]
    expression: sum(fcut_slope * cap, over=generator) <= fcut_const
objective:
  sense: minimize
  expression: sum(cap * invest) + theta
```

**`cut` and `fcut` take their members from data**
([the data contract](../reference/data.md)). An iteration appends rows to their
parameter tables and generates no YAML, so the spec a reviewer reads is the
spec that runs.

`theta` is a scalar variable, `dims: []`, and its `lower: 0` is the only
thing keeping the first master bounded before any cut exists.

## Reading a cut out of an answer

A cut is the value and the slope of the subproblem at the capacity that was
tried. The slope is the shadow price of the capacity constraint, weighted by
availability and summed over snapshots. `sources` is the data attached to
the spec ([glossary](../reference/glossary.md#how-it-runs)):

```python
import specsolve as sps
import polars as pl

with sps.solve('examples/benders/sub.yaml', sources) as sub:
    slope = (
        sub.dual('capacity')
        .join(avail, on=['snapshot', 'generator'], suffix='_avail')
        .with_columns((pl.col('value') * pl.col('value_avail')).alias('term'))
        .group_by('generator')
        .agg(pl.col('term').sum().alias('slope'))
    )
```

The whole interface with the [engine](../reference/glossary.md#how-it-runs) is
`dual` and a join against the model's own `avail` table. Appending the cut is
two `pl.concat` calls onto the parameter tables the master already declares.

## When the subproblem is infeasible

Below some capacity there is no dispatch at all, and the subproblem is
infeasible. There is nothing to price: `dual()` raises rather than returning a
vector of zeros that looks like an answer, and correctly so.

What such a solve does have is a **certificate that no dispatch exists**, and
`dual_ray` is it. It weights the subproblem's rows so that together they
demand more than the generators can deliver:

```python
with sps.build(sub_spec, dispatch) as sub_model:
    answer = sub_model.update({'cap_hat': capacity}).solve()
    if not answer.has_primal:
        u = answer.dual_ray('capacity')  # one weight per (snapshot, generator)
        v = answer.dual_ray('balance')  # one weight per snapshot
```

Weighting each row by its own weight and adding them gives
`Σ u·cap_hat·avail + Σ v·load > 0`, and that is the proof. Every term in it is
linear in capacity, so **asking the master for a capacity where the same
combination is not positive is exactly one row** — slope `Σₛ u·avail` per
generator, against `−Σₛ v·load`, which is the `fcut_slope` and `fcut_const`
the master already declares.

Three properties make this the cut to use rather than a fallback:

- **It needs no second model.** Until specsolve read a ray, this page carried a
  fourth YAML file — the subproblem with a slack variable and an objective
  asking *how far from dispatchable* a capacity was — and a second solve for
  every capacity that failed.
- **It is a stronger cut.** The example converges in **2 steps with one
  feasibility cut** where the elastic model took 4 and three of them.
- **The sign is the row's own**, one convention across every sink, so the
  arithmetic above does not ask which solver ran.

A sink computes a certificate only if it was asked to: HiGHS always does,
Gurobi needs `solver_options={'InfUnbdInfo': 1}` and Xpress
`solver_options={'presolve': 0}`. Reading a ray without them raises, and the
message names the option.

## The loop

```python
sub_spec, master_spec = (to_spec(path) for path in paths)

with (
    sps.build(sub_spec, {**dispatch, 'cap_hat': capacity}) as sub_model,
    sps.build(master_spec, {**master_sources, **empty}) as master,
):
    for step in range(25):
        sub = sub_model.update({'cap_hat': capacity}).solve()
        dispatchable = sub.has_primal
        if dispatchable:
            slope, here_value = slope_at(sub, capacity)
            upper = min(upper, spent(capacity) + sub.objective)
            appended(tables, 'cut', sub.objective - here_value, slope)
        else:
            slope, against = cut_from_ray(sub)
            appended(tables, 'fcut', against, slope)

        answer = master.update({**tables, **coordinates}).solve()
        lower = answer.objective
        capacity = answer.primal('cap').select('generator', 'value')

        if upper < float('inf') and upper - lower <= 1e-6 * abs(upper):
            break
```

Twenty lines, two built models and a growing pair of tables. **A reader could
write this**, which is what [#596](https://github.com/fluxopt/specsolve/issues/596)
settled on.

Each spec is read once above the loop **and built once**, because a cut is a row
in a parameter table rather than an edit to a file. `sps.build` attaches the data
and `update` puts the next iteration's numbers on the model that is already
there ([glossary](../reference/glossary.md#the-chain)), so parsing, validation
and the build are paid once per run rather than three times an iteration. The
subproblem's `cap_hat` reaches its rows as a right-hand side, so the solver keeps
the model it holds and re-solves from the last basis. The master gains a row a
step, so it is loaded again — which is what the last two lines below count.

Both cut families now come out of the *same* subproblem, its prices for one and
its ray for the other, which is why there are two models here and not three.

## Running it

```bash
pixi run python examples/benders/run.py
```

```text
the whole problem, in one plan: 9600.00

  step 0  feasibility  lower  2850.00   upper none yet
  step 1  optimality   lower  9600.00   upper 9600.00

decomposed: 9600.00 in 2 steps
monolithic: 9600.00
difference: 0.0e+00
cuts: 1 optimality, 1 feasibility
the subproblem loaded the solver 1 time(s) in 2 solves
the master loaded the solver 2 time(s) in 2 solves
```

One capacity is excluded as undispatchable, and the first optimality cut then
closes the gap exactly. The elastic model this page used to carry took four
steps over the same data, which is the difference between a cut that says *how
far from dispatchable* a capacity was and one that says *why no dispatch
exists*.

## The check is the algorithm's own

specsolve can always build the monolith from the same sources, so the example
solves both and prints the difference: `0.0e+00` above, asserted in
`tests/test_benders_example.py`. That is the two-lane differential test aimed
at an algorithm instead of an engine. It is always available because the
undecomposed form is another file over the same data.

## What is deliberately absent

Missing is everything that makes a decomposition survive a real model: cut
management as the master grows, stabilisation, multi-cut, tolerances that hold
when duals are degenerate, and an answer for when convergence does not happen.
[#596](https://github.com/fluxopt/specsolve/issues/596) asked whether specsolve
should own that surface and answered no, so all of it stays the caller's. What a
caller still lacks *from specsolve* is collected in
[#1677](https://github.com/fluxopt/specsolve/issues/1677). This page settles only
that the *language* is not the obstacle.
