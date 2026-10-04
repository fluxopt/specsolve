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
import json
import pickle
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pytest

from bench.arms import ARMS, WRITERS, unmeasurable
from bench.cases import CASES, rescaled, shortened
from bench.conftest import shape_of

#: The rungs a sweep is measured at. Above them the solver is the sweep's cost.
SWEPT_SIZES = ('2xs', 'xs', 's')

#: The rungs a fresh process is measured at: the sizes a process started per solve would build.
FRESH_SIZES = ('2xs', 'xs')

#: Each window change, and whether it should load the solver from scratch.
RELOADS = {'values': False, 'shape': True, 'one': False, 'coefficient': True, 'cold': True}


def _rounds(benchmark: Any, request: pytest.FixtureRequest, fn: Any, *args: Any, setup: Any = None) -> Any:
    """*fn* once per round, with a full garbage collection before each clock starts.

    Without it a round inherits the last one's garbage: on `dispatch/m` the
    gurobi sink put 0.45 s and 0.67 s in one distribution (#1288).

    Pedantic mode takes its rounds from the caller, so `--benchmark-min-rounds`
    is read here. Under CodSpeed there is no such option. A setup still goes
    in pedantic form there: its instruments call the target more than once —
    a warm-up, then the measured call — and only the pedantic form runs the
    setup before each, so a window that consumes its model starts from it
    every time.
    """
    rounds = getattr(request.config.option, 'benchmark_min_rounds', None)
    if rounds is None:
        if setup is not None:
            return benchmark.pedantic(fn, setup=setup)
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
    numeric x of a scaling curve; ``size`` is a label. ``phase_seconds`` is
    the last round's split, not a minimum: it attributes the wall time, it
    does not replace it.
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
    if counts.get('phases'):
        info['phase_seconds'] = counts['phases']


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
@pytest.mark.parametrize('change', tuple(RELOADS))
def test_window(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    sink: str,
    change: str,
) -> None:
    """What the *second* window of a rolling horizon costs, up to the solve.

    `test_emit` prices the first window. *change* is what the second one moves:
    ``values`` re-attaches the same data, which a loaded solver takes by value;
    ``shape`` attaches the rung one snapshot shorter, which it cannot; ``one``
    updates one parameter alone, which costs a whole rebuild all the same;
    ``coefficient`` updates the case's parameter inside the matrix, which moves
    the digest and loads the solver again, so only a case that names one has
    it; and ``cold`` re-attaches the same data under ``keep='nothing'``. An arm carries
    between windows whatever its library has a verb for: specsolve updates and
    loads, linopy builds a new model. Each arm's `window_setup`
    runs untracked in the spawned child before every sample (#1617). An arm
    with no `window` verb is skipped.
    """
    missing = unmeasurable(arm, case_name, sink) or ceiling.reached(arm, case_name, size, sink)
    if missing:
        pytest.skip(missing)

    module = ARMS[arm]
    if not hasattr(module, 'window'):
        pytest.skip(f'{arm} has no rolling-horizon verb — nothing here says what its second window costs')
    if sink in WRITERS:
        pytest.skip('a file is written whole every window — there is no loaded artifact to re-attach to')
    if change not in getattr(module, 'WINDOW_CHANGES', RELOADS):
        pytest.skip(f'{arm} rebuilds every window, so a {change} window is its values window measured again')

    case = CASES[case_name]
    if change == 'coefficient' and case.coefficient is None:
        pytest.skip(f'{case_name} has no parameter inside the matrix, so no update of its values moves the digest')

    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    following = prepared
    if change == 'shape':
        following = module.prepare(case_name, size, case.data(shortened(case.shape(size))), {})
    if change == 'coefficient':
        following = (prepared[0], {case.coefficient: rescaled(prepared[1][case.coefficient])})
    setup = partial(module.window_setup, sink, prepared, following, change)
    counts = _rounds(benchmark, request, module.window, setup=setup)
    if 'reloaded' in counts:
        assert counts['reloaded'] == RELOADS[change], (
            f'a {change} window {"reloaded" if counts["reloaded"] else "pushed onto"} the solver, '
            f'so this rung measured the other path'
        )
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, sink, _measured(benchmark), _peak(benchmark))


@pytest.mark.benchmem(isolate=True)
def test_sweep(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    sink: str,
) -> None:
    """What ``solve_over`` costs across a few slices of one rung, solves included.

    The solver runs on every slice, so the wall time is not the sweep's alone;
    its own clocks say what the solver took. An arm with no `sweep` verb is
    skipped.
    """
    module = ARMS[arm]
    if not hasattr(module, 'sweep'):
        pytest.skip(f'{arm} has no sweep verb — nothing here says what its loop over slices costs')
    if sink in WRITERS:
        pytest.skip('a sweep solves every slice, and a file is not a solver')
    if size not in SWEPT_SIZES:
        pytest.skip(f'a sweep is measured at {", ".join(SWEPT_SIZES)}; above them it measures the solver')
    missing = unmeasurable(arm, case_name, sink) or ceiling.reached(arm, case_name, size, f'sweep-{sink}')
    if missing:
        pytest.skip(missing)

    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    counts = _rounds(benchmark, request, module.sweep, sink, prepared)
    assert counts['loads'] == 1, 'a sweep of unchanged values loads its solver once and pushes onto it after'
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, f'sweep-{sink}', _measured(benchmark), _peak(benchmark))


@pytest.mark.benchmem(isolate=True)
@pytest.mark.parametrize('into', ('frames', 'parquet'))
def test_read(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    into: str,
) -> None:
    """What reading a solve's answer back costs: every value laid out against the build, *into* frames or onto disk.

    The answer is a full-length one built in `read_setup`, so no solver runs and
    the cost is the rung's rather than the solve's, at every rung. Sink-free,
    because the vectors are the same whoever produced them. An arm with no
    `read` verb is skipped.
    """
    module = ARMS[arm]
    if not hasattr(module, 'read'):
        pytest.skip(f'{arm} reads its answer back inside its solve — nothing here can time the read alone')
    missing = unmeasurable(arm, case_name, module.SINKS[0]) or ceiling.reached(arm, case_name, size, 'read')
    if missing:
        pytest.skip(missing)

    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    counts = _rounds(benchmark, request, module.read, setup=partial(module.read_setup, prepared, into))
    _record(benchmark, counts, case_name, size)
    ceiling.record(arm, case_name, size, 'read', _measured(benchmark), _peak(benchmark))


def _fresh(payload: Path) -> dict[str, Any]:
    """Run `bench.fresh` on *payload* in a new interpreter, and return the answer it wrote beside it."""
    answer = payload.with_suffix('.json')
    done = subprocess.run(
        [sys.executable, '-m', 'bench.fresh', str(payload), str(answer)], capture_output=True, text=True
    )
    if done.returncode:
        raise RuntimeError(f'bench.fresh exited {done.returncode}:\n{done.stderr[-4000:]}')
    return json.loads(answer.read_text())


def test_fresh(
    benchmark: Any,
    request: pytest.FixtureRequest,
    paths: Any,
    ceiling: Any,
    case_name: str,
    size: str,
    arm: str,
    sink: str,
) -> None:
    """The first window in a new process, launch to exit: what a process started for each solve pays.

    `test_emit` excludes the import and the interpreter, and `test_rebuild`'s
    first round runs after both. Here the wall time includes them, and
    ``first_window`` is the arm's own ``build_and_emit`` inside the process, so
    the gap between the two is what starting costs. ``process_rss_bytes`` is the
    child's own peak: `benchmem` would measure this process, not that one.
    """
    if size not in FRESH_SIZES:
        pytest.skip(f'a fresh process is measured at {", ".join(FRESH_SIZES)}; above them the build is the cost')
    missing = unmeasurable(arm, case_name, sink) or ceiling.reached(arm, case_name, size, f'fresh-{sink}')
    if missing:
        pytest.skip(missing)

    module = ARMS[arm]
    prepared = module.prepare(case_name, size, paths(case_name, size), {})
    with tempfile.TemporaryDirectory(prefix='specsolve-fresh-') as tmp:
        payload = Path(tmp) / 'payload.pickle'
        payload.write_bytes(pickle.dumps((arm, sink, prepared)))
        counts = _rounds(benchmark, request, _fresh, payload)
    _record(benchmark, counts, case_name, size)
    info = getattr(benchmark, 'extra_info', None)
    if info is not None:
        info['process_rss_bytes'] = counts['peak_rss_bytes']
    ceiling.record(arm, case_name, size, f'fresh-{sink}', _measured(benchmark), counts['peak_rss_bytes'])


def test_rebuild(benchmark: Any, paths: Any, ceiling: Any, builds: int, case_name: str, size: str, arm: str) -> None:
    """First build against every later one, in one process.

    First is the first build in a process that has imported its library already
    (`test_fresh` adds the import); steady is what a rolling horizon pays for
    every build after it. Not `isolate=True` and
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
