"""Attribute build wall time to the queries that spend it.

Wraps ``LazyFrame.collect`` and tags every collection with the build step that
issued it, so the output is a ranked list of queries.

    pixi run -e bench python -m bench.profile_build dispatch l
    pixi run -e bench python -m bench.profile_build transport m

The wrapper adds overhead per call, so read the shares, not the seconds.
"""

from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path
from typing import Any

from bench import cases as bench_cases

#: What a collection is attributed to, as ``(module path, owner or None, name)``
#: — a method on the class that owns it, or a module-level function.
STEPS = (
    ('specsolve.relational.engines.polars.assembly', 'Assembly', '_build_variable'),
    ('specsolve.relational.engines.polars.assembly', 'Assembly', '_build_constraint'),
    ('specsolve.relational.engines.polars.assembly', 'Assembly', '_build_objective'),
    ('specsolve.relational.engines.polars.labels', None, 'frame'),
    ('specsolve.relational.engines.polars.attaching', None, 'attach'),
)


def _instrument(timings: dict[Any, list[float]], phase: dict[str, str]) -> None:
    """Tag each collection with the build step that issued it.

    A query is identified by its flattened plan — the same role the SQL text
    plays in a statement-level profiler.
    """
    import importlib

    import polars as pl

    original_collect = pl.LazyFrame.collect

    def collect(self, *args, **kwargs):
        started = time.perf_counter()
        try:
            return original_collect(self, *args, **kwargs)
        finally:
            elapsed = time.perf_counter() - started
            key = (phase['now'], ' '.join(self.explain(optimized=False).split())[:88])
            entry = timings.setdefault(key, [0.0, 0])
            entry[0] += elapsed
            entry[1] += 1

    pl.LazyFrame.collect = collect

    for module_path, class_name, name in STEPS:
        module = importlib.import_module(module_path)
        owner = module if class_name is None else getattr(module, class_name)
        original_step = getattr(owner, name)

        def wrap(step, label):
            def wrapper(*args, **kwargs):
                previous, phase['now'] = phase['now'], label
                try:
                    return step(*args, **kwargs)
                finally:
                    phase['now'] = previous

            return wrapper

        setattr(owner, name, wrap(original_step, f'{class_name}.{name}' if class_name else name))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=sorted(bench_cases.CASES))
    parser.add_argument('size', help='a rung of the case ladder, e.g. xs s m l')
    parser.add_argument('--top', type=int, default=12, help='queries to list')
    args = parser.parse_args()

    timings: dict[Any, list[float]] = {}
    phase = {'now': 'setup'}
    _instrument(timings, phase)

    import specsolve as sps

    case = bench_cases.CASES[args.case]
    shape = case.shape(args.size)
    sources = case.data(shape)

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        started = time.perf_counter()
        with sps.build(case.spec_path(shape), sources) as model:
            build = time.perf_counter() - started
            phase['now'] = 'emit'
            started = time.perf_counter()
            model.write(Path(tmp) / 'model.lp')
            emit = time.perf_counter() - started

    print(f'\n{args.case}/{args.size}: build {build:.2f}s, emit {emit:.2f}s')
    print('(instrumented — read the shares, not the seconds)\n')

    by_step: dict[str, list[float]] = collections.defaultdict(lambda: [0.0, 0])
    for (step, _), (elapsed, calls) in timings.items():
        by_step[step][0] += elapsed
        by_step[step][1] += calls
    total = sum(v[0] for v in by_step.values()) or 1.0

    print(f'{"step":24} {"seconds":>8} {"share":>7} {"calls":>7}')
    for step, (elapsed, calls) in sorted(by_step.items(), key=lambda kv: -kv[1][0]):
        print(f'{step:24} {elapsed:8.2f} {100 * elapsed / total:6.0f}% {calls:7d}')

    print(f'\ntop {args.top} queries')
    ranked = sorted(timings.items(), key=lambda kv: -kv[1][0])[: args.top]
    for (step, sql), (elapsed, calls) in ranked:
        print(f'  {elapsed:6.2f}s {100 * elapsed / total:4.0f}%  n={calls:<4d} [{step}]\n      {sql}')


if __name__ == '__main__':
    main()
