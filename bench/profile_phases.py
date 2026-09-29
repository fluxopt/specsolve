"""Attribute build wall time to phases, in seconds you can compare to a real run.

``profile_build.py`` answers which query; this answers which phase. It wraps
three methods per build, so its seconds are comparable to ``bench/``.

    pixi run -e bench python -m bench.profile_phases profiled l
    pixi run -e bench python -m bench.profile_phases transport l --rounds 15

Parsing and lowering happen once, outside the loop. The first pass attaches
the sources every round; the second reuses one frozen ``AttachedSources``, so
the difference between the two minima is what attaching costs.
"""

from __future__ import annotations

import argparse
import collections
import statistics
import time
from typing import Any

from bench import cases as bench_cases

#: Wrapped per build rather than per collect, so the overhead is three calls.
PHASES = ('_build_variable', '_build_constraint', '_build_objective')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog='python -m bench.profile_phases')
    parser.add_argument('case', choices=sorted(bench_cases.CASES))
    parser.add_argument('size')
    parser.add_argument('--rounds', type=int, default=9, help='timed builds per arm; the minimum is reported')
    args = parser.parse_args(argv)

    from mathspec import to_spec

    from specsolve.relational.engines.polars import engine as executor_module
    from specsolve.relational.engines.polars.assembly import Assembly
    from specsolve.relational.engines.polars.engine import PolarsEngine
    from specsolve.sources import tidy_sources

    spent: dict[str, list[float]] = collections.defaultdict(list)
    for name in PHASES:
        setattr(Assembly, name, _timed(name, getattr(Assembly, name), spent))

    case = bench_cases.CASES[args.case]
    shape = case.shape(args.size)
    schema = to_spec(str(case.spec_path(shape)))
    program = schema.program
    sources = tidy_sources(program, dict(case.data(shape)))

    real_attach = executor_module.attach
    cached: list[Any] = []

    def attach_once(program_: Any, sources_: Any) -> Any:
        if not cached:
            cached.append(real_attach(program_, sources_))
        return cached[0]

    def one() -> float:
        """One build, timed. Called once untimed to fill the attach cache."""
        engine = PolarsEngine()
        started = time.perf_counter()
        engine.build(program, sources)
        elapsed = time.perf_counter() - started
        engine.close()
        return elapsed

    full = [one() for _ in range(args.rounds)]

    executor_module.attach = attach_once
    one()
    spent.clear()
    assembly = [one() for _ in range(args.rounds)]
    executor_module.attach = real_attach

    print(f'\n{args.case}/{args.size}: {args.rounds} rounds, minimum reported\n')
    _line('build, attaching included', full)
    _line('build, attaching reused', assembly)
    print(f'  {"attaching":28} {(min(full) - min(assembly)) * 1000:8.1f} ms')
    print()
    for phase, times in sorted(spent.items(), key=lambda kv: -sum(kv[1])):
        calls = len(times) // args.rounds
        print(f'  {phase:28} {sum(times) / args.rounds * 1000:8.1f} ms   over {calls} call(s)')
    return 0


def _timed(name: str, method: Any, spent: dict[str, list[float]]) -> Any:
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return method(self, *args, **kwargs)
        finally:
            spent[name].append(time.perf_counter() - started)

    return wrapped


def _line(label: str, times: list[float]) -> None:
    low = min(times)
    spread = (max(times) / low - 1) * 100
    print(f'  {label:28} {low * 1000:8.1f} ms   median {statistics.median(times) * 1000:7.1f}   spread {spread:4.1f}%')


if __name__ == '__main__':
    raise SystemExit(main())
