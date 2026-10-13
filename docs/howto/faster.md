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

An `update` that moves a mask or adds a label builds and loads again.
[Change it and solve again](../change.md) shows the loop and how to check
that it stays loaded.

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

A small model builds faster on one polars thread, and a large model builds
slower. The size where it turns depends on the model and the machine. Polars
reads the thread count once, when it is imported, so set it in the
environment:

```bash
POLARS_MAX_THREADS=1 python solve.py
```

In a notebook, set it before the first import of polars or specsolve:

```python
import os

os.environ['POLARS_MAX_THREADS'] = '1'

import specsolve as sps
```

This page timed it when it was built. It built a dispatch model of 10,000
variables ten times on each setting, each setting in a process of its own. A
box spans the middle half of the builds, and the line in it is the median:

```python exec="true" html="true"
import math
import os
import statistics
import subprocess
import sys

BUILDS = """
import time

import polars as pl

import specsolve as sps

generators = [f'g{i:02d}' for i in range(50)]
sources = {
    'generator': generators,
    'snapshot': list(range(200)),
    'p_max': {g: 10.0 + i for i, g in enumerate(generators)},
    'cost': {g: float(i) for i, g in enumerate(generators)},
    'load': [300.0] * 200,
}
print(pl.thread_pool_size())
for _ in range(11):
    started = time.perf_counter()
    sps.build('examples/dispatch.yaml', sources).close()
    print(time.perf_counter() - started)
"""


def builds(threads: str | None) -> tuple[int, list[float]]:
    env = {k: v for k, v in os.environ.items() if k != 'POLARS_MAX_THREADS'}
    if threads is not None:
        env['POLARS_MAX_THREADS'] = threads
    out = subprocess.run([sys.executable, '-c', BUILDS], env=env, capture_output=True, text=True, check=True)
    pool, _warm, *seconds = out.stdout.split()
    return int(pool), sorted(float(s) * 1000 for s in seconds)


pool, default = builds(None)
_, one = builds('1')
rows = [(f'{pool} threads', default, 'var(--s2)'), ('1 thread', one, 'var(--s1)')]
share = [o / d for o, d in zip(one, default, strict=True)]

left, right, top, step = 84, 464, 28, 52
top_ms = math.ceil(max(default + one) / 10) * 10
x = lambda ms: left + (right - left) * ms / top_ms
bottom = top + step * len(rows)
ink = 'var(--md-default-fg-color)'
light = 'var(--md-default-fg-color--light)'
svg = [
    f'<svg viewBox="0 0 480 {bottom + 40}" role="img" style="width:100%;max-width:480px;font-size:13px" '
    f'aria-label="Ten build times on each thread setting, in milliseconds">'
]
for ms in range(0, top_ms + 1, 10):
    svg.append(
        f'<line x1="{x(ms):.1f}" x2="{x(ms):.1f}" y1="{top - 8}" y2="{bottom}" '
        f'style="stroke:var(--md-default-fg-color--lightest)"/>'
        f'<text x="{x(ms):.1f}" y="{bottom + 16}" text-anchor="middle" style="fill:{light}">{ms}</text>'
    )
svg.append(
    f'<text x="{(left + right) / 2}" y="{bottom + 34}" text-anchor="middle" style="fill:{light}">build time, ms</text>'
)
for i, (label, times, colour) in enumerate(rows):
    y = top + step * i + step / 2 - 8
    q1, median, q3 = statistics.quantiles(times, n=4, method='inclusive')
    svg.append(
        f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" style="fill:{ink}">{label}</text>'
        f'<line x1="{x(times[0]):.1f}" x2="{x(times[-1]):.1f}" y1="{y}" y2="{y}" style="stroke:{light};stroke-width:1.5"/>'
        f'<rect x="{x(q1):.1f}" y="{y - 12}" width="{x(q3) - x(q1):.1f}" height="24" rx="4" '
        f'style="fill:{colour};fill-opacity:.18;stroke:{colour};stroke-width:1.5"/>'
        f'<line x1="{x(median):.1f}" x2="{x(median):.1f}" y1="{y - 12}" y2="{y + 12}" style="stroke:{colour};stroke-width:2.5"/>'
    )
    for ms in times:
        svg.append(
            f'<circle cx="{x(ms):.1f}" cy="{y}" r="4" style="fill:{colour};stroke:var(--md-default-bg-color);stroke-width:2">'
            f'<title>{label}: {ms:.1f} ms</title></circle>'
        )
svg.append('</svg>')
summary = '. '.join(
    f'{label}: median {statistics.median(t):.1f} ms, from {t[0]:.1f} to {t[-1]:.1f}' for label, t, _ in rows
)
print(f'<div class="bench">{"".join(svg)}</div>')
print(f'<p>{summary}. One thread took {min(share):.0%} to {max(share):.0%} of the time.</p>')
```

Time your own model on both settings, in a process of its own for each, to
find which side of the turn it is on.

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
