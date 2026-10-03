"""The relational lane: the YAML, the parquet, and a sink it never runs."""

from __future__ import annotations

import importlib
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bench.cases import CASES

#: Every sink the relational lane can hand a model to.
SINKS = ('lp', 'highs', 'gurobi')

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


def build_and_emit(sink: str, prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Build relationally and hand the model over — an LP file, or a solver; the solver never runs."""
    import specsolve as sps

    spec, sources = prepared
    with tempfile.TemporaryDirectory(prefix='specsolve-bench-') as tmp, sps.build(spec, sources) as model:
        if sink == 'lp':
            model.write(Path(tmp) / 'model.lp')
        else:
            _loaded(sink, model).close()

        return _counts(_handoff(model), nonzeros=True)


def _loaded(sink: str, model: Any) -> Any:
    """A solver holding *model*, for a sink that has one: its class, or the ``build_<sink>`` an older checkout loads through."""
    module = importlib.import_module(f'specsolve.relational.sinks.solvers.{sink}')
    load = getattr(module, f'build_{sink}', None) or getattr(module, sink.capitalize())
    return load(_handoff(model))


def window_setup(
    sink: str, prepared: tuple[Path, dict[str, str]], following: tuple[Path, dict[str, str]]
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Window one, built and loaded before the clock, and held for the one that is timed.

    Nothing is released: the model and its solver stay resident, so the
    measured peak includes them.
    """
    import specsolve as sps

    spec, sources = prepared
    model = sps.build(spec, sources)
    model._engine._hand_off(sink, None, 'solver')
    return (model, sink, following[1]), {}


def window(model: Any, sink: str, sources: dict[str, str]) -> Counts:
    """What the second window of a rolling horizon costs, up to the solve.

    ``update`` rebuilds, and ``_hand_off`` is what ``solve`` does before the run: it
    digests the new build against the one the solver holds, then pushes the
    bounds, costs and right-hand sides onto it, or loads it from scratch where
    the digest moved. A later window pays one digest fewer, because the held
    one is kept.
    """
    model.update(sources)
    _, kept = model._engine._hand_off(sink, None, 'solver')
    return _counts(_handoff(model), nonzeros=True) | {'reloaded': kept == 'nothing'}


def build_only(prepared: tuple[Path, dict[str, str]]) -> Counts:
    """Just the build — no sink, nothing to release."""
    import specsolve as sps

    spec, sources = prepared
    with sps.build(spec, sources) as model:
        return _counts(_handoff(model), nonzeros=False)


def objective(prepared: tuple[Path, dict[str, str]]) -> float:
    """Solve, and return the objective the parity gate compares."""
    import specsolve as sps

    spec, sources = prepared
    with sps.solve(spec, sources) as sol:
        if sol.termination_condition != 'optimal':
            raise RuntimeError(f'specsolve solve terminated {sol.termination_condition!r}, not optimal')
        return float(sol.objective)
