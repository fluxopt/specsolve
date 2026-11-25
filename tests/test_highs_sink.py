"""The ``highs`` sink keeps a coefficient that ``sps.write`` keeps, or says it cannot.

HiGHS drops a matrix or Hessian entry at or below its ``small_matrix_value``
when a model is passed. Its default is 1e-9, and 1e-12 is the lowest it
accepts.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve._sinks.highs import Highs
from specsolve.errors import SpecsolveWarning

ROW = {
    'parameters': {'a': {'dims': []}},
    'variables': {'x': {'dims': [], 'bounds': {'lower': 0}}},
    'constraints': {'c': {'dims': [], 'expression': 'a * x >= 0.001'}},
    'objective': {'sense': 'minimize', 'expression': 'x'},
}

SQUARE = {
    'parameters': {'a': {'dims': []}},
    'variables': {'x': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}}},
    'objective': {'sense': 'minimize', 'expression': 'a * x * x - x'},
}


def _a(value: float) -> dict[str, pl.DataFrame]:
    return {'a': pl.DataFrame({'value': [value]})}


def test_a_coefficient_below_the_highs_default_reaches_the_solver():
    """5e-10 is under HiGHS's default cut of 1e-9.

    The sink passed that default, so the row read ``0 >= 0.001`` and the solve
    came back infeasible, while ``sps.write`` kept the coefficient and the
    model solved elsewhere to 2e6.
    """
    with sps.solve(ROW, _a(5e-10)) as result:
        assert result.termination_condition == 'optimal'
        assert result.objective == pytest.approx(2e6), 'the 5e-10 coefficient took part in the solve'


@pytest.mark.parametrize(
    ('spec', 'held'),
    [
        pytest.param(ROW, lambda model: model.lp_.a_matrix_.value_, id='constraint'),
        pytest.param(SQUARE, lambda model: model.hessian_.value_, id='quadratic-objective'),
    ],
)
def test_highs_holds_a_coefficient_below_its_default(spec, held):
    """A square term is loaded as ``2a``, 1e-9, which the default cut drops too."""
    with sps.build(spec, _a(5e-10)) as model, Highs(model._engine._model.tables) as solver:
        assert len(held(solver.handle.getModel())) == 1, 'the 5e-10 coefficient is in the loaded model'


@pytest.mark.parametrize(
    ('spec', 'what'),
    [
        pytest.param(ROW, 'constraint', id='constraint'),
        pytest.param(SQUARE, 'quadratic objective', id='quadratic-objective'),
    ],
)
def test_a_coefficient_highs_cannot_take_warns(spec, what):
    """5e-13 is under the lowest cut HiGHS accepts, so it is dropped, and the solve says so."""
    with pytest.warns(SpecsolveWarning, match=rf'drops 1 {what} coefficient.*5e-13.*1e-12'):
        sps.solve(spec, _a(5e-13)).close()


def test_the_callers_small_matrix_value_decides_what_is_dropped():
    with pytest.warns(SpecsolveWarning, match=r'5e-10.*1e-09'):
        result = sps.solve(ROW, _a(5e-10), solver_options={'small_matrix_value': 1e-9})
    with result:
        assert result.termination_condition == 'infeasible'
