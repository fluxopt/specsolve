"""Slack: how far each constraint is from binding, non-negative wherever it holds.

Computed from the activity and the right-hand side, so it is readable wherever
the activity is, a mixed-integer incumbent included. The oracle is arithmetic
done by hand on rows written both ways round: the value must not depend on
which side a term is written on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.engine import readback

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output, Sweep

SLACK: frozenset[Output] = frozenset({'slack'})

#: ``x`` is pushed down onto its floor of 1, ``y`` is fixed at 2, the cap of 4
#: is 3 away and the sum's floor of 0 is 3 away.
BY_HAND = {'floor': 0.0, 'cap': 3.0, 'fixed': 0.0, 'loose': 3.0}


def spec(rows: dict[str, str], domain: str = 'continuous') -> dict:
    return {
        'variables': {
            'x': {'dims': [], 'bounds': {'lower': -10, 'upper': 10}, 'domain': domain},
            'y': {'dims': [], 'bounds': {'lower': -10, 'upper': 10}},
        },
        'constraints': {name: {'dims': [], 'expression': written} for name, written in rows.items()},
        'objective': {'sense': 'minimize', 'expression': 'x + y'},
    }


WRITTEN = [
    pytest.param(
        {'floor': 'x >= 1', 'cap': 'x <= 4', 'fixed': 'y == 2', 'loose': 'x + y >= 0'},
        id='variables-on-the-left',
    ),
    pytest.param(
        {'floor': '1 <= x', 'cap': '4 >= x', 'fixed': '2 == y', 'loose': '0 <= x + y'},
        id='variables-on-the-right',
    ),
    pytest.param(
        {'floor': 'x - 1 >= 0', 'cap': '0 >= x - 4', 'fixed': 'y - 2 == 0', 'loose': '-x <= y'},
        id='constants-moved-across',
    ),
]


@pytest.mark.parametrize('rows', WRITTEN)
def test_the_slack_is_the_distance_to_binding_however_the_row_is_written(solver_name: str, rows) -> None:
    with sps.solve(spec(rows), {}, solver_name=solver_name, outputs=SLACK) as answer:
        got = {name: answer.slack(name)['value'].item() for name in rows}
    assert got == pytest.approx(BY_HAND, abs=1e-9), 'a binding row reads zero, a loose one how far it is from binding'


def test_an_equality_off_its_right_hand_side_reads_below_zero_either_way() -> None:
    """``==`` holds only at zero, so a residual of either sign is a violation of that size."""
    with sps.build(spec({'fixed': 'y == 2'}), {}) as model:
        handoff = model._engine._model.handoff
        over = readback.slacks(handoff, pl.Series('value', [2.5]))
        under = readback.slacks(handoff, pl.Series('value', [1.5]))
    assert (over.item(), under.item()) == pytest.approx((-0.5, -0.5)), 'the residual, signed as a violation'


def test_an_integer_answer_has_a_slack() -> None:
    """Like the activity, and unlike a dual, the slack exists at any incumbent."""
    rows = {'floor': 'x >= 1.5', 'cap': 'x <= 4'}
    with sps.solve(spec(rows, domain='integer'), {}, outputs=SLACK) as answer:
        assert answer.primal('x')['value'].item() == pytest.approx(2.0), 'the first integer above the floor'
        assert answer.slack('floor')['value'].item() == pytest.approx(0.5), 'half a unit above a floor of 1.5'
        assert answer.slack('cap')['value'].item() == pytest.approx(2.0), 'two below a cap of 4'


def test_a_saved_answer_reads_back_its_slack(tmp_path: Path) -> None:
    with sps.solve(spec({'cap': 'x <= 4', 'floor': 'x >= 1'}), {}, outputs=SLACK) as answer:
        live = answer.slack('cap')
        answer.save(tmp_path)
    assert sps.load_result(tmp_path).slack('cap').equals(live)


def test_a_solve_not_asked_for_its_slack_names_the_output() -> None:
    with (
        sps.solve(spec({'cap': 'x <= 4', 'floor': 'x >= 1'}), {}) as answer,
        pytest.raises(SpecsolveError, match=r"outputs=\{'slack'\}"),
    ):
        answer.slack('cap')


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

#: The dispatch model of ``test_strategy``: ``balance`` is an equality, met at
#: every snapshot, so its slack is zero in every scenario.
SHAPES = [pytest.param(how, id=how) for how in ('held', 'spilled', 'archived')]


def _swept(tmp_path: Path, how: str) -> Sweep:
    from tests.test_strategy import DISPATCH, scenario_sources

    axis = sps.EachCoordinate('scenario')
    if how == 'archived':
        sps.solve_over(DISPATCH, scenario_sources(), axis, outputs=SLACK, archive=tmp_path / 'run.zip')
        archive = sps.load_archive(tmp_path / 'run.zip')
        assert isinstance(archive, sps.archive.SweepArchive)
        return archive.sweep
    spill = {'spill_to': tmp_path / 'spill'} if how == 'spilled' else {}
    return sps.solve_over(DISPATCH, scenario_sources(), axis, outputs=SLACK, **spill)


@pytest.mark.parametrize('how', SHAPES)
def test_a_sweep_reads_each_slices_slack(tmp_path: Path, how: str) -> None:
    frame = _swept(tmp_path, how).slack('balance')
    assert frame.height == 12, 'three scenarios of four snapshots'
    assert np.abs(frame['value'].to_numpy()).max() < 1e-9, 'an equality met at every snapshot is zero away from binding'
