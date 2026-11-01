# Making a solve faster

How to solve a model that takes too long, and what each way costs. Try them in
this order: the first ones keep the answer, the later ones change it.

| Way | Keeps the optimum | What it costs |
|---|---|---|
| [Keep the model loaded](#keep-the-model-loaded) | yes | only data that moves no mask or label |
| [Start from an earlier answer](#start-from-an-earlier-answer) | yes | can make a solve slower |
| [Build many small models faster](#build-many-small-models-faster) | yes | only below 250,000 columns and rows |
| [Build on one polars thread](#build-on-one-polars-thread) | yes | slows a large model |
| [Accept a gap](#accept-a-gap) | within the gap | the answer is close, not optimal |
| [Relax the integers](#relax-the-integers) | no, a bound | the schedule is fractional |
| [Solve in windows](#solve-in-windows) | no | the windows cannot see past their end |
| [Decompose](#decompose) | within the gap | you write and own the loop |
| [Make the model smaller](#make-the-model-smaller) | no | it answers a different question |

## Keep the model loaded

Build once, then `update` and solve. The solver keeps the model, and the next
solve skips the build:

```python
model = sps.build('dispatch.yaml', sources)
for numbers in steps:
    result = model.update(numbers).solve()
```

An `update` that moves a mask or adds a label builds and loads again. An
`update` that gives a parameter new numbers at the coordinates it already
has, where the parameter is only a bound, a right-hand side, a cost or a
coefficient of a linear constraint, also skips most of the build: only the
declarations it feeds are read again. A new coefficient still loads the
solver again. [Change it and solve again](../change.md) shows the
loop and how to check that it stays loaded.

## Start from an earlier answer

Each solve in a loop starts from the one before, by default. That is usually
faster, and on some models much slower, because a solver that carries on skips
its presolve. [Starting from an earlier answer](warm-start.md) has the
measurements and shows how to time both ways.

## Build many small models faster

specsolve builds a model of at most 250,000 columns and 250,000 rows on
polars' in-memory engine. To move the rest of a process onto it too, such as a
sweep of many small models and your own queries:

```python
import polars as pl

pl.Config.set_engine_affinity('in-memory')
```

`POLARS_ENGINE_AFFINITY=in-memory` in the environment does the same, and the
workers of a [parallel sweep](parallel.md) inherit it.

## Build on one polars thread

A small model can build faster on one polars thread, and a large model builds
slower. Polars reads the setting when it is imported, so set it in the
environment, or in `os.environ` before the first import of polars or
specsolve:

```bash
POLARS_MAX_THREADS=1 python solve.py
```

The numbers that follow are rough estimates from one 4-core machine. One
thread saved about a quarter of the build time up to 20,000 variables, and
about a tenth from 100,000 to 200,000. At 1,000,000 variables the build took
half as long again
([#2016](https://github.com/fluxopt/specsolve/pull/2016)). On 2 cores, one
thread saved nothing
([#2017](https://github.com/fluxopt/specsolve/pull/2017)). For the best
setting for your model, time both settings on the machine that runs it, each
in a process of its own.

The workers of a [parallel sweep](parallel.md) inherit the setting. It changes
polars only, and the solver keeps its own threads.

## Accept a gap

A MIP solver proves its answer optimal by closing a gap between the answer and
a bound. Most of the time can go into the last part. Stop at 1%:

```python
result = sps.solve('commitment.yaml', sources, solver_options={'mip_rel_gap': 0.01})
```

The option names are the solver's own; [Choosing a solver](solvers.md) shows
how to pass them. The answer is within 1% of the optimum, and it can be a
different schedule.

## Relax the integers

Make the integer variables continuous. The model is then an LP, which solves
faster and has prices:

```python
from mathspec import to_spec

relaxed = to_spec('commitment.yaml').to_dict()
relaxed['variables']['on'] |= {'domain': 'continuous', 'bounds': {'lower': 0, 'upper': 1}}
```

The answer is a lower bound on the cost, not a schedule: a generator can be a
third on. To get prices for the integer answer instead, see
[Prices from a MIP](fix-relax-remove.md#prices-from-a-mip).

## Solve in windows

Solve a long horizon one window at a time. Each window stays the same size,
however long the horizon is:

```python
sweep = sps.solve_over(
    'storage.yaml',
    sources,
    sps.EachWindow('hour', steps=24, lookahead=12, into='snapshot', carry={'soc_initial': 'soc'}),
)
```

A window cannot see past its end, so the schedule costs more than one solve.
A lookahead buys that back, and makes each window larger. The windows run one
after the other, because each one starts where the one before stopped.
[A horizon too large for one solve](../sweep.md) shows both.

## Decompose

A model that is large because of one part, such as dispatch under an
investment choice, can be split into a small master and a large subproblem.
The answer has a gap that you choose, so stop at 1% and the answer is within
1%. specsolve gives no driver for it, so you write the loop.
[The Benders example](https://github.com/fluxopt/specsolve/blob/main/examples/benders/README.md)
is one, checked against the whole model.

## Make the model smaller

Representative days instead of a year, fewer nodes or fewer weather years
make a smaller model. That is a different model: it answers a different
question, and nothing tells you how far its answer is from the one you asked
for.
