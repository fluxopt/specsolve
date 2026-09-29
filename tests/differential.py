"""The both-lanes harness: one model, two backends, one answer.

The same YAML means the same thing on the linopy lane and on the streaming
relational one (docs/about/architecture.md, hard rule 3). Importing this module
is the oracle's guard: a bare install skips every module that uses it at
collection.

The engine stays open for the length of the ``with`` block, so per-variable
primal checks live inside it::

    with differential(NONCONVEX_YAML, sources, lp=True) as run:
        assert run.result.to_pandas('op_cost') ...
"""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DataError
from specsolve.lanes import lowered
from specsolve.relational.engines.polars.engine import PolarsEngine
from specsolve.sources import tidy_sources
from tests.conftest import schema_of, solve_written_file
from tests.oracle import linopy, specsolve_linopy, xr

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from specsolve.relational.engines.polars.engine import Result

#: Both lanes hand the same numbers to the same solver, so they agree to solver precision.
RTOL = 1e-9


class NoFiniteAnswerError(AssertionError):
    """The fixture admits no finite optimum, so neither lane is on trial.

    A class of its own so a caller generating its models can tell "this data
    has no answer" from "the lanes disagree".
    """


@dataclass
class Agreement:
    """What the two lanes produced, for tests that assert past the objective."""

    oracle: float
    """The linopy objective — the number both lanes had to reach."""

    model: linopy.Model
    """The linopy model, for structural assertions (labels, masks, solution)."""

    result: Result
    """The relational solution; live until the ``with`` block exits."""

    engine: PolarsEngine
    lp: Path | None = None
    """The written LP file, when ``lp=True`` — already checked to agree."""


@contextmanager
def differential(
    spec: str | Path | dict[str, Any],
    sources: Mapping[str, Any],
    *,
    lp: bool = False,
) -> Iterator[Agreement]:
    """Build ``spec`` on both lanes with the same inputs; assert they agree.

    ``spec`` is a ``Path`` to a file, the YAML text itself, or a raw dict.
    Both lanes are handed it with every formulation written out
    (``Spec.expand()``).

    Duals are not compared: an LP with alternative optima has many optimal
    duals, and the two lanes hand HiGHS the same rows in a different order.

    Set ``lp=True`` to also write and re-solve the LP file, the third opinion.
    """
    model = schema_of(spec).expand()

    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)

        m = specsolve_linopy.build(model, dict(sources))
        m.solve(solver_name='highs', output_flag=False)
        oracle = float(m.objective.value)
        if not np.isfinite(oracle):
            raise NoFiniteAnswerError('the linopy oracle is infeasible or unbounded — fix the data, not the tolerance')

        program = lowered(model)
        with PolarsEngine() as engine:
            engine.build(program, tidy_sources(program, dict(sources)))
            result = engine.solve()
            assert result.is_ok, f'the relational lane reached no solution: {result.status}'
            assert result.objective == pytest.approx(oracle, rel=RTOL), (
                f'the lanes disagree on the objective — relational {result.objective}, linopy {oracle}'
            )
            _same_shape(engine.diagnostics(), m)

            lp_path = None
            if lp:
                lp_path = work / 'model.lp'
                engine.write(lp_path)
                assert solve_written_file(lp_path) == pytest.approx(oracle, rel=RTOL), (
                    f'the written {lp_path.name} re-solves to a different objective than both lanes reached'
                )

            yield Agreement(oracle=oracle, model=m, result=result, engine=engine, lp=lp_path)


@contextmanager
def at_a_point(
    spec: str | Path | dict[str, Any], sources: Mapping[str, Any]
) -> Iterator[Callable[[str | Mapping[str, Any]], pl.DataFrame]]:
    """Both lanes with every variable held at one chosen point; yield a reader that asserts they agree.

    No solver runs. Each variable takes a seeded value at every coordinate it
    exists at, the linopy lane reads it as ``.solution`` and the relational
    lane through ``Model.evaluator``, so ``read(expression)`` values the
    expression on both and returns the relational frame once the two are the
    same frame: the same coordinates, the same values to ``RTOL``.

    That is the claim :func:`differential` cannot make. An objective only sees
    a row that binds, and a coefficient only a column that moves; a read at a
    point sees every coordinate of every term, at any degree, and needs no data
    that keeps a model feasible.
    """
    model = schema_of(spec).expand()
    m = specsolve_linopy.build(model, dict(sources))
    point = _hold(m)
    with sps.build(model, dict(sources)) as built:
        relational = built.evaluator(point, None, 'a chosen point has no duals')

        def read(expression: str | Mapping[str, Any]) -> pl.DataFrame:
            ours = relational(expression)
            theirs = _frame(specsolve_linopy.evaluate(m, model, expression, dict(sources)))
            assert by_coordinate(ours) == pytest.approx(by_coordinate(theirs), rel=RTOL), (
                f'the lanes read `{expression}` differently at {_pointby_coordinate(point)} — '
                f'relational {by_coordinate(ours)}, linopy {by_coordinate(theirs)}'
            )
            return ours

        yield read


def _hold(m: linopy.Model) -> dict[str, pl.DataFrame]:
    """Set a seeded value on every variable of *m* where it exists, and return the same point as frames.

    A value on the quarter grid in ``[-2, 2]``, never zero: a failure prints
    short numbers, and a zero only arises where an expression makes one, which
    is the case a quotient has to answer. The masked coordinates stay NaN, as a
    solve leaves them, and have no row in the frame, as a saved solution has
    none. Marking *m* solved is what lets the lane's ``.solution`` be read.
    """
    rng = np.random.default_rng(1203)
    point = {}
    for name, variable in m.variables.items():
        labels = variable.labels
        values = rng.integers(1, 9, labels.shape) * rng.choice([-1, 1], labels.shape) / 4
        held = labels.copy(data=values).where(labels != -1)
        variable.solution = held
        point[name] = _frame(held)
    m.status = 'ok'
    m.termination_condition = 'optimal'
    return point


def _frame(values: xr.DataArray) -> pl.DataFrame:
    """A linopy-lane value as the relational lane's ``(dims…, value)`` frame, with no row where it is absent."""
    if not values.dims:
        return pl.DataFrame({'value': [float(values)]}).filter(pl.col('value').is_not_nan())
    table = values.rename('value').to_dataframe().reset_index()[[*values.dims, 'value']]
    return pl.from_pandas(table).filter(pl.col('value').is_not_nan())


def by_coordinate(frame: pl.DataFrame) -> dict[tuple[str, ...], float]:
    """*frame*'s values keyed by coordinate, dims sorted by name and labels as text.

    The lanes order a result's dims differently and type an ``int`` label
    differently, and neither is a difference in the value.
    """
    dims = sorted(c for c in frame.columns if c != 'value')
    return {tuple(str(label) for label in row[:-1]): row[-1] for row in frame.select(*dims, 'value').iter_rows()}


def _pointby_coordinate(point: Mapping[str, pl.DataFrame]) -> dict[str, dict[tuple[str, ...], float]]:
    return {name: by_coordinate(frame) for name, frame in point.items()}


def both_lanes_refuse(spec: str | Path | dict[str, Any], sources: Mapping[str, Any], match: str) -> str:
    """Both doors refuse *sources* with one sentence, returned for the cases that pin more of it.

    The two are built apart: a ``pytest.raises`` around :func:`differential`
    is satisfied by the linopy build alone.
    """
    model = schema_of(spec).expand()
    with pytest.raises(DataError, match=match) as relational:
        sps.build(model, dict(sources)).close()
    with pytest.raises(DataError, match=match) as linopy_lane:
        specsolve_linopy.build(model, dict(sources))
    assert str(relational.value) == str(linopy_lane.value), 'one defect, one sentence'
    return str(relational.value)


def _same_shape(diagnostics: Any, linopy_lane: Any) -> None:
    """The two lanes built the same *model*, not merely the same answer.

    An objective and a re-solved LP file are invariant to a column pinned to
    ``[0, 0]`` or a row that always holds, so the counts are compared. Counts
    rather than sets: the two lanes name their columns differently.
    """
    assert diagnostics.columns == linopy_lane.nvars, (
        f'the lanes disagree on how many columns this model has — relational {diagnostics.columns}, linopy {linopy_lane.nvars}'
    )
    assert diagnostics.rows == linopy_lane.ncons, (
        f'the lanes disagree on how many rows this model has — relational {diagnostics.rows}, linopy {linopy_lane.ncons}'
    )
