"""The ladder: one model, two lanes, one seam — measured by whichever plugin is loaded.

    pixi run refresh    # every rung, then both writers, in order — or by hand:
    pixi run -e bench pytest bench --benchmark-memory --benchmark-json=bench/results/latest.json \\
        --sizes xs s m l
    pixi run -e bench python -m bench.report bench/results/latest.json    # -> markdown
    pixi run -e bench python -m bench.plot                                # -> the chart page

Selection is `--cases / --sizes / --arms / --sinks` (see `conftest.py`), and
`-k` narrows further.

Peak RSS is the published metric, and it needs `isolate=True`: a fresh process
per pass. The memray peak is recorded beside it, but only `rss` compares across
libraries, because memray counts polars' reserved arenas and not the
interpreter.

Solve time is not measured.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from functools import partial
from typing import Any

import pytest

from bench.arms import ARMS, unmeasurable
from bench.conftest import shape_of


def _rounds(benchmark: Any, request: pytest.FixtureRequest, fn: Any, *args: Any, setup: Any = None) -> Any:
    """*fn* once per round, with a full garbage collection before each clock starts.

    Without it a round inherits the last one's garbage: on `dispatch/m` the
    gurobi sink put 0.45 s and 0.67 s in one distribution (#1288).

    Pedantic mode takes its rounds from the caller, so `--benchmark-min-rounds`
    is read here. Under CodSpeed there is no such option, and the plain call
    stands.
    """
    rounds = getattr(request.config.option, 'benchmark_min_rounds', None)
    if rounds is None:
        if setup is not None:
            args, _ = setup()
        return benchmark(fn, *args)
    return benchmark.pedantic(
        fn,
        args=() if setup is not None else args,
        setup=_collected if setup is None else _CollectedSetup(setup),
        rounds=rounds,
        iterations=1,
        warmup_rounds=0,
    )


def _collected() -> None:
    gc.collect()


@dataclass
class _CollectedSetup:
    """The collection above, in front of a setup that also supplies the arguments.

    A dataclass rather than a closure, because `benchmem(isolate=True)` pickles
    setup and action to a spawned child (#1617).
    """

    setup: Any

    def __call__(self) -> Any:
        gc.collect()
        return self.setup()


def _record(benchmark: Any, counts: dict[str, Any], case_name: str, size: str) -> None:
    """Attach the dims the published tables read, and check the model is the right one.

    Written only when the fixture carries `extra_info`, which CodSpeed's does
    not. ``live_fraction`` is measured, not declared. ``variables`` is the
    numeric x of a scaling curve; ``size`` is a label.
    """
    shape = shape_of(case_name, size)
    assert 0 < counts['columns'] <= shape.nominal_variables
    info = getattr(benchmark, 'extra_info', None)
    if info is None:
        return
    info['columns'] = counts['columns']
    info['rows'] = counts['rows']
    info['nonzeros'] = counts['nonzeros']
    info['live_fraction'] = counts['columns'] / shape.nominal_variables
    info['variables'] = shape.nominal_variables


def _measured(benchmark: Any) -> float | None:
    """The fastest round, in seconds — or None under CodSpeed, where the budget does not apply."""
    stats = getattr(getattr(benchmark, 'stats', None), 'stats', None)
    return float(stats.min) if stats is not None else None


def _peak(benchmark: Any) -> float | None:
    """Whole-process high-water of the isolated pass, or None where there was none."""
    blob = (getattr(benchmark, 'extra_info', None) or {}).get('benchmem') or {}
    rss = blob.get('rss_bytes')
    if isinstance(rss, list):
        return min(rss) if rss else None
    return rss


@pytest.mark.benchmem(isolate=True)
def test_emit(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    sink: str,
) -> None:
    """Build the model and hand it over — an LP file on disk, or a populated solver.

    Every arm starts from the same parquet and stops at the same seam, and pays
    for its own data ingestion.
    """
    missing = unmeasurable(arm, case_name, sink) or ceiling.reached(arm, case_name, size, sink)
    if missing:
        pytest.skip(missing)

    module = ARMS[arm]
    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    counts = _rounds(benchmark, request, module.build_and_emit, sink, prepared)
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, sink, _measured(benchmark), _peak(benchmark))


@pytest.mark.benchmem(isolate=True)
def test_window(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    sink: str,
) -> None:
    """What the *second* window of a rolling horizon costs, and every one after.

    `test_emit` prices the first window. An arm carries between windows whatever
    its library has a verb for: specsolve re-attaches and pushes onto the loaded
    solver, linopy builds a new model. Each arm's `window_setup` runs untracked
    in the spawned child before every sample (#1617). An arm with no `window`
    verb is skipped.
    """
    missing = unmeasurable(arm, case_name, sink) or ceiling.reached(arm, case_name, size, sink)
    if missing:
        pytest.skip(missing)

    module = ARMS[arm]
    if not hasattr(module, 'window'):
        pytest.skip(f'{arm} has no rolling-horizon verb — nothing here says what its second window costs')
    if sink == 'lp':
        pytest.skip('a file is written whole every window — there is no loaded artifact to re-attach to')

    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    counts = _rounds(benchmark, request, module.window, setup=partial(module.window_setup, sink, prepared))
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, sink, _measured(benchmark), _peak(benchmark))


def test_rebuild(benchmark: Any, paths: Any, ceiling: Any, builds: int, case_name: str, size: str, arm: str) -> None:
    """First build against every later one, in one process.

    First is what a fresh interpreter pays for one build; steady is what a
    rolling horizon pays for every build after it. Not `isolate=True` and
    sink-free, because repeated builds in one process are the question.
    `conftest.py` deselects it under CodSpeed.
    """
    if builds < 1:
        pytest.skip('--builds 0')
    missing = unmeasurable(arm, case_name, ARMS[arm].SINKS[0]) or ceiling.reached(arm, case_name, size, '')
    if missing:
        pytest.skip(missing)
    module = ARMS[arm]
    counts = benchmark.pedantic(
        module.build_only,
        args=(module.prepare(case_name, size, paths(case_name, size), {}),),
        setup=_collected,
        rounds=builds,
        iterations=1,
        warmup_rounds=0,
    )
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, '', _measured(benchmark))
