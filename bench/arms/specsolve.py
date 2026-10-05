"""The relational lane: the YAML, the parquet, and a sink it never runs."""

from __future__ import annotations

import importlib
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


def _handoff(handle: Any) -> Any:
    """The built model's frames, wherever the checkout under test keeps them.

    The ladder runs across checkouts, so this reaches the handle, the engine and
    the frames (``handoff`` or ``tables``) in each shape an older checkout had.
    """
    engine = getattr(handle, '_engine', handle)
    built = getattr(engine, '_model', None)
    if built is None:
        return engine._tables()
    handoff = getattr(built, 'handoff', None) or built.tables
    return handoff() if callable(handoff) else handoff


def _counts(tables: Any, *, nonzeros: bool) -> Counts:
    """The dims the published tables read; nonzeros is None where no matrix is assembled or reachable."""
    matrix = getattr(tables, 'matrix', None) if nonzeros else None
    return {
        'columns': tables.column_count,
        'rows': tables.row_count,
        'nonzeros': getattr(matrix, 'height', None),
    }


def _clocks(model: Any) -> dict[str, float]:
    """The engine's own seconds per phase, summed over the model's life; empty on a checkout that kept none."""
    return dict(getattr(getattr(model, '_engine', None), '_seconds', None) or {})


def build_and_emit(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Build relationally and hand the model over — an LP file, or a solver; the solver never runs.

    A solver is constructed straight off the frames, so its load is in no
    phase: it is the wall time less the phases.
    """
    import specsolve as sps

    spec, sources = prepared
    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp, sps.build(spec, sources) as model:
        if sink in ('lp', 'mps'):
            model.write(Path(tmp) / f'model.{sink}')
        else:
            _loaded(sink, model).close()

        return _counts(_handoff(model), nonzeros=True) | {'phases': _clocks(model)}


def _loaded(sink: str, model: Any) -> Any:
    """A solver holding *model*, for a sink that has one: its class, or the ``build_<sink>`` an older checkout loads through."""
    module = importlib.import_module(f'specsolve.relational.sinks.solvers.{sink}')
    load = getattr(module, f'build_{sink}', None) or getattr(module, sink.capitalize())
    return load(_handoff(model))


#: The private engine method each verb is timed through, which a checkout older than the verb lacks.
TIMED_THROUGH = {'window': '_hand_off', 'read': '_answered'}


def unsupported(verb: str) -> str | None:
    """Why the checkout under test cannot take *verb*, or None when it can.

    `bench.yml` measures the base branch's `src/` under this harness, and a base
    older than the method a verb is timed through has none of it. Its cells are
    skipped there, and the gate compares only what both runs measured.
    """
    from specsolve.relational.engine.engine import Engine

    method = TIMED_THROUGH.get(verb)
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
    matrix; a ``cold`` window asks for ``keep='nothing'``.
    """
    import specsolve as sps

    spec, sources = prepared
    model = sps.build(spec, sources)
    model._engine._hand_off(sink, None, 'solver')
    updated = following[1]
    if change == 'one':
        first = next(name for name in model._program.parameters if name in updated)
        updated = {first: updated[first]}
    return (model, sink, updated, 'nothing' if change == 'cold' else 'solver'), {}


def window(model: Any, sink: str, sources: dict[str, str], keep: str) -> Counts:
    """What the second window of a rolling horizon costs, up to the solve.

    ``update`` rebuilds, and ``_hand_off`` is what ``solve`` does before the run: it
    digests the new build against the one the solver holds, then pushes the
    bounds, costs and right-hand sides onto it, or loads it from scratch where
    the digest moved. A later window pays one digest fewer, because the held
    one is kept.
    """
    before = _clocks(model)
    model.update(sources)
    _, kept = model._engine._hand_off(sink, None, keep)
    phases = {phase: seconds - before.get(phase, 0.0) for phase, seconds in _clocks(model).items()}
    return _counts(_handoff(model), nonzeros=True) | {'reloaded': kept == 'nothing', 'phases': phases}


#: Slices in the measured sweep — enough that every one after the first is the push path.
SWEEP_SLICES = 4


def sweep(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """``solve_over`` across hand-built slices of the rung's own data, folded in order on one model.

    Every slice carries the same values, so the first loads the solver and the
    rest push. The solver runs on every slice, and ``phases`` sums the sweep's
    own per-slice clocks, so its ``solve`` is what the wall time owes the
    solver and the rest is the sweep's.
    """
    import specsolve as sps

    spec, sources = prepared
    swept = sps.solve_over(
        spec, sources, [(i, sources) for i in range(SWEEP_SLICES)], key_name='scenario', solver_name=sink
    )
    metrics = swept.metrics
    return {
        'columns': metrics['columns'][0],
        'rows': metrics['rows'][0],
        'nonzeros': metrics['nonzeros'][0],
        'loads': int(metrics['loads'].sum()),
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
    from specsolve.relational.sinks.solvers.base import SolveAnswer
    from specsolve.relational.status import SolveStatus

    spec, sources = prepared
    model = sps.build(spec, sources)
    handoff = _handoff(model)
    rows = pl.zeros(handoff.row_count, dtype=pl.Float64, eager=True)
    answer = SolveAnswer(
        SolveStatus('optimal'),
        0.0,
        primal=pl.zeros(handoff.column_count, dtype=pl.Float64, eager=True),
        dual=None if model._engine._discrete() else rows,
        activity=rows,
    )
    return (model, answer, into), {}


def read(model: Any, answer: Any, into: str) -> Counts:
    """Lay *answer* out against the build, then read every value back — *into* tidy frames, or onto disk.

    ``frames`` collects every variable's primal and every constraint's dual
    and activity; ``parquet`` is ``Result.save``.
    """
    result = model._engine._answered(answer, 'highs', 'nothing', None)
    if into == 'frames':
        for name in model._program.variables:
            result.primal(name)
        for name in model._program.constraints:
            if answer.dual is not None:
                result.dual(name)
            result.activity(name)
    else:
        with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp:
            result.save(Path(tmp) / 'answer')
    return _counts(_handoff(model), nonzeros=False)


def build_only(prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Just the build — no sink, nothing to release."""
    import specsolve as sps

    spec, sources = prepared
    with sps.build(spec, sources) as model:
        return _counts(_handoff(model), nonzeros=False) | {'phases': _clocks(model)}


def objective(prepared: tuple[Path, dict[str, str]]) -> float:
    """Solve, and return the objective the parity gate compares."""
    import specsolve as sps

    spec, sources = prepared
    with sps.solve(spec, sources) as sol:
        if sol.termination_condition != 'optimal':
            raise RuntimeError(f'specsolve solve terminated {sol.termination_condition!r}, not optimal')
        return float(sol.objective)
