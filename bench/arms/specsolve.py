"""The relational lane: the YAML, the parquet, and a sink it never runs."""

from __future__ import annotations

import importlib
import inspect
import itertools
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

#: Every sink the relational lane can hand a model to.
SINKS = ('lp', 'mps', 'highs', 'gurobi', 'xpress')

#: What has to be importable for this arm to run; an absent library skips the cell.
REQUIRES = ()

if TYPE_CHECKING:
    from collections.abc import Mapping

    from bench.arms import Counts
    from bench.cases import Case


def checked_sources(case: Case, size: str, paths: dict[str, str]) -> dict[str, str]:
    """Every generated parquet, checked against what the model declares, before the clock.

    A path the model declares nothing for raises; it is usually a stale parquet
    in the case's cache directory.
    """
    import yaml as pyyaml

    spec = case.spec_path(case.shape(size))
    schema = pyyaml.safe_load(spec.read_text())
    declared = set().union(*(schema.get(block, {}) for block in ('parameters', 'dimensions', 'relations')))
    undeclared = sorted(set(paths) - declared)
    if undeclared:
        raise ValueError(
            f'{case.name}: {undeclared} declared as neither parameter, dimension nor relation in '
            f'{spec} — the build would not see it. Stale files under bench/.cache/?'
        )
    return dict(paths)


def prepare(
    case_name: str, size: str, paths: dict[str, str], options: Mapping[str, Any]
) -> tuple[Path, dict[str, str]]:
    """The spec to build and the sources to build it from, both already checked."""
    from bench.cases import CASES

    del options
    case = CASES[case_name]
    return case.spec_path(case.shape(size)), checked_sources(case, size, paths)


def _tables(handle: Any) -> Any:
    """The built model's tables, wherever the checkout under test keeps them.

    The ladder runs across checkouts, so this reaches the handle, the engine and
    the tables (``handoff`` or ``tables``) in each shape an older checkout had.
    """
    engine = getattr(handle, '_engine', handle)
    built = getattr(engine, '_built', None) or getattr(engine, '_model', None)
    if built is None:
        return engine._tables()
    tables = getattr(built, 'tables', None) or built.handoff
    return tables() if callable(tables) else tables


def _counts(tables: Any, *, nonzeros: bool) -> Counts:
    """The dims the published tables read; nonzeros is None where no matrix is assembled or reachable."""
    matrix = getattr(tables, 'matrix', None) if nonzeros else None
    return {
        'columns': tables.column_count,
        'rows': tables.row_count,
        'nonzeros': getattr(matrix, 'height', None),
    }


def _clocks(model: Any) -> dict[str, float]:
    """The engine's own seconds per phase that ran, summed over the model's life; empty on a checkout that kept none.

    A checkout with `Model.metrics` reads it; one from before it (#2188) kept
    the phases in the engine's `_seconds`, which `bench.ab` meets on an older base.
    """
    if callable(getattr(model, 'metrics', None)):
        metrics = model.metrics()
        return {
            field.removesuffix('_seconds'): seconds
            for field, seconds in metrics._asdict().items()
            if field.endswith('_seconds') and seconds > 0
        }
    return dict(getattr(getattr(model, '_engine', None), '_seconds', None) or {})


def _laps(model: Any, since: int = 0) -> list[dict[str, Any]]:
    """Every lap from the *since*-th on, as rows of `Model.timings`; empty on a checkout from before it (#2207).

    Copied off the engine's log rather than read through ``timings()``: this
    runs inside the timed call, and building a polars frame there is charged
    to the wall time.
    """
    log = getattr(model, '_clock', None) or getattr(getattr(model, '_engine', None), '_log', None)
    if log is None:
        return []
    steps = list(log._steps)
    return [
        {'run': run, 'phase': steps[code][0], 'step': steps[code][1], 'entry': steps[code][2], 'seconds': seconds}
        for run, code, seconds in itertools.islice(zip(log._runs, log._codes, log._seconds, strict=True), since, None)
    ]


def build_and_emit(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Build relationally and hand the model over — an LP file, or a solver; the solver never runs.

    A solver is constructed straight off the tables, so its load is in no
    phase: it is the wall time less the phases.
    """
    import specsolve as sps

    spec, sources = prepared
    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp, sps.build(spec, sources) as model:
        if sink in ('lp', 'mps'):
            model.write(Path(tmp) / f'model.{sink}')
        else:
            _loaded(sink, model).close()

        return _counts(_tables(model), nonzeros=True) | {'phases': _clocks(model), 'laps': _laps(model)}


def _loaded(sink: str, model: Any) -> Any:
    """A solver holding *model*, for a sink that has one: its class, or the ``build_<sink>`` an older checkout loads through."""
    module = importlib.import_module(f'specsolve._sinks.{sink}')
    load = getattr(module, f'build_{sink}', None) or getattr(module, sink.capitalize())
    return load(_tables(model))


#: The private engine method each verb was timed through while an `Engine` ran the lifecycle, which a checkout older than the verb lacks.
TIMED_THROUGH = {'window': 'handoff', 'read': '_answered'}


def unsupported(verb: str) -> str | None:
    """Why the checkout under test cannot take *verb*, or None when it can.

    A run of this harness against an older checkout's `src/` meets a base
    without the method a verb is timed through. Its cells are skipped there, so
    a comparison holds only what both runs measured. A checkout whose `Model`
    runs the lifecycle itself has every verb.
    """
    method = TIMED_THROUGH.get(verb)
    try:
        from specsolve._engine.engine import Engine
    except ModuleNotFoundError:
        return None
    if method is None or hasattr(Engine, method):
        return None
    return f'this checkout has no Engine.{method}, which the {verb} is timed through'


def window_setup(
    sink: str, prepared: tuple[Path, dict[str, str]], following: tuple[Path, dict[str, str]], change: str
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Window one, built and loaded before the clock, and held for the one that is timed.

    Nothing is released: the model and its solver stay resident, so the
    measured peak includes them. A ``one`` window updates the first declared
    parameter alone, as ``update`` is usually called; a ``coefficient`` window
    updates whatever *following* carries, which is one parameter inside the
    matrix; a ``cold`` window drops the held solver, so the next is loaded fresh.
    """
    import specsolve as sps

    spec, sources = prepared
    model = sps.build(spec, sources)
    _handed_off(model, sink)
    updated = following[1]
    if change == 'one':
        first = next(name for name in model._program.parameters if name in updated)
        updated = {first: updated[first]}
    return (model, sink, updated, change == 'cold'), {}


def window(model: Any, sink: str, sources: dict[str, str], cold: bool) -> Counts:
    """What the second window of a rolling horizon costs, up to the solve.

    ``update`` rebuilds, and ``handoff`` is what ``solve`` does before the run: it
    digests the new build against the one the solver holds, then pushes the
    bounds, costs and right-hand sides onto it, or loads it from scratch where
    the digest moved. A later window pays one digest fewer, because the held
    one is kept. A *cold* window closes the held solver first, so the model
    is loaded from scratch.
    """
    before, lapped = _clocks(model), len(_laps(model))
    model.update(sources)
    reloaded = _handed_off(model, sink, cold=cold)
    phases = {phase: seconds - before.get(phase, 0.0) for phase, seconds in _clocks(model).items()}
    return _counts(_tables(model), nonzeros=True) | {
        'reloaded': reloaded,
        'phases': phases,
        'laps': _laps(model, since=lapped),
    }


def _handed_off(model: Any, sink: str, *, cold: bool = False) -> bool:
    """The handoff, and whether it loaded the solver from scratch; a *cold* window closes the held solver first.

    ``Model._handoff``, or ``Engine.handoff`` on a checkout whose engine holds the solver.
    """
    holder = getattr(model, '_engine', model)
    if cold and holder._solver is not None:
        holder._solver.close()
        holder._solver = None
    handoff = getattr(holder, '_handoff', None) or holder.handoff
    return handoff(sink, None)[1]


#: Slices in the measured sweep — enough that every one after the first is the push path.
SWEEP_SLICES = 4


def _under_each_scenario(spec: Path, sources: dict[str, str]) -> dict[str, Any]:
    """*sources* with one parameter repeated under each of ``SWEEP_SLICES`` scenarios.

    The parameter over the fewest dimensions carries the column, so cutting it
    is the least of what a slice costs.
    """
    import polars as pl

    import specsolve as sps

    parameters = sps.check(spec).parameters
    carrier = min((name for name in parameters if name in sources), key=lambda name: len(parameters[name].dims))
    scenarios = pl.LazyFrame({'scenario': range(SWEEP_SLICES)})
    return {**sources, carrier: pl.scan_parquet(sources[carrier]).join(scenarios, how='cross')}


def sweep(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """``solve_over`` across scenarios of the rung's own data, folded in order on one model.

    Every scenario carries the same values, so the first loads the solver and
    the rest push. The solver runs on every slice, and ``phases`` sums the
    sweep's own per-slice clocks, so its ``solve`` is what the wall time owes
    the solver and the rest is the sweep's.
    """
    import specsolve as sps

    spec, sources = prepared
    swept = sps.solve_over(spec, _under_each_scenario(spec, sources), sps.EachLabel('scenario'), solver_name=sink)
    metrics = swept.metrics
    return {
        'columns': metrics['columns'][0],
        'rows': metrics['rows'][0],
        'nonzeros': metrics['nonzeros'][0],
        'loads': int(metrics['solver_instances'].sum()),
        'phases': {
            column.removesuffix('_seconds'): float(metrics[column].sum())
            for column in metrics.columns
            if column.endswith('_seconds')
        },
    }


def read_setup(prepared: tuple[Path, dict[str, str]], into: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """A build and an answer to it, before the clock — every column and row zero, reported optimal.

    What reading an answer back costs is set by how many values it lays out,
    not by what they are, so no solver runs. A model with an integer variable
    gets no duals, as a real solve leaves it.
    """
    import polars as pl

    import specsolve as sps
    from specsolve._sinks.solver import SolveAnswer
    from specsolve._status import SolveStatus

    spec, sources = prepared
    model = sps.build(spec, sources)
    tables = _tables(model)
    rows = pl.zeros(tables.row_count, dtype=pl.Float64, eager=True)
    answer = SolveAnswer(
        SolveStatus('optimal'),
        0.0,
        primal=pl.zeros(tables.column_count, dtype=pl.Float64, eager=True),
        dual=None if _discrete(model) else rows,
        activity=rows,
    )
    return (model, answer, into), {}


def read(model: Any, answer: Any, into: str) -> Counts:
    """Lay *answer* out against the build, then read every value back — *into* tidy frames, or onto disk.

    ``frames`` collects every variable's primal and every constraint's dual;
    ``parquet`` writes the answer as an archive holds it, through
    ``Result._save`` where a checkout has no ``Result.save``. Both read the default answer, which holds
    no optional output. The answer passes through
    ``Model._with_record_fields`` as a solve's does, so what `solve` attaches
    is timed too; a checkout older than it reads the engine's answer as it is.
    """
    result = _laid_out(model, answer)
    if (with_record_fields := getattr(model, '_with_record_fields', None)) is not None:
        result = with_record_fields(result, 'highs', None, ())
    if into == 'frames':
        for name in model._program.variables:
            result.primal(name)
        for name in model._program.constraints:
            if answer.dual is not None:
                result.dual(name)
    else:
        with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp:
            _write_answer(result, Path(tmp) / 'answer')
    return _counts(_tables(model), nonzeros=False)


def _discrete(model: Any) -> list[str]:
    """The variables *model* declares as anything but continuous, as the checkout under test names them."""
    if (engine := getattr(model, '_engine', None)) is not None:
        return engine._discrete()
    from specsolve._engine import readback

    return readback.discrete(model._built)


def _laid_out(model: Any, answer: Any) -> Any:
    """*answer* laid out against *model*'s build as a result: ``readback.result``, or ``Engine._answered`` before it."""
    if (engine := getattr(model, '_engine', None)) is None:
        from specsolve._engine import readback

        return readback.result(model._built, answer, 'highs', None)
    answered = engine._answered
    kept = ('nothing',) if 'kept' in inspect.signature(answered).parameters else ()
    return answered(answer, 'highs', *kept, None)


def _write_answer(result: Any, out: Path) -> None:
    """*result*'s answer at *out*, as the checkout under test writes one."""
    if (save := getattr(result, 'save', None)) is not None:
        save(out)
        return
    from specsolve._answer_layout import write_whole

    result._save(out, write_whole)


def build_only(prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Just the build — no sink, nothing to release."""
    import specsolve as sps

    spec, sources = prepared
    with sps.build(spec, sources) as model:
        return _counts(_tables(model), nonzeros=False) | {'phases': _clocks(model), 'laps': _laps(model)}


def written(prepared: tuple[Path, dict[str, str]], path: Path) -> None:
    """Build and write the model to *path* as an LP file, unsolved — what the whole-model comparison reads."""
    import specsolve as sps

    spec, sources = prepared
    with sps.build(spec, sources) as model:
        model.write(path)


def objective(prepared: tuple[Path, dict[str, str]]) -> float:
    """Solve, and return the objective the parity gate compares."""
    import specsolve as sps

    spec, sources = prepared
    with sps.solve(spec, sources) as sol:
        if sol.termination_condition != 'optimal':
            raise RuntimeError(f'specsolve solve terminated {sol.termination_condition!r}, not optimal')
        return float(sol.objective)
