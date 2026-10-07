"""The irreducible infeasible subsystem of the last solve, read back by declaration and coordinate.

The model is a dispatch whose second snapshot asks for more than both
technologies can deliver, so the one minimal conflict is that snapshot's
balance row against the two upper bounds it pushes on. ``ramp`` and the first
snapshot are feasible, and must stay out of it.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.sinks import SOLVERS

SHORT = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'tech': {'dtype': 'str'}},
    'parameters': {'demand': {'dims': ['snapshot']}, 'cap': {'dims': ['tech']}},
    'variables': {'p': {'dims': ['snapshot', 'tech'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {
        'balance': {'dims': ['snapshot'], 'expression': 'sum(p, over=tech) == demand'},
        'ramp': {'dims': ['tech'], 'expression': 'sum(p, over=snapshot) <= 1000'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

SOURCES = {
    'snapshot': pl.DataFrame({'snapshot': [0, 1]}),
    'tech': pl.DataFrame({'tech': ['gas', 'wind']}),
    'demand': pl.DataFrame({'snapshot': [0, 1], 'value': [60.0, 200.0]}),
    'cap': pl.DataFrame({'tech': ['gas', 'wind'], 'value': [100.0, 50.0]}),
}

FEASIBLE = {**SOURCES, 'demand': pl.DataFrame({'snapshot': [0, 1], 'value': [60.0, 120.0]})}

#: *SHORT* with whole units: ``2 * sum(p) == 3`` has a real solution and no
#: integer one.
INTEGER = {
    'dimensions': {'tech': {'dtype': 'str'}},
    'variables': {'u': {'dims': ['tech'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'odd': {'dims': [], 'expression': 'sum(2 * u, over=tech) == 3'},
        'spare': {'dims': ['tech'], 'expression': 'u <= 9'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(u)'},
}

INTEGER_SOURCES = {'tech': pl.DataFrame({'tech': ['gas', 'wind']})}


def test_the_subsystem_is_the_row_and_the_bounds_that_conflict(solver_name: str) -> None:
    """Snapshot 1 asks 200 of two technologies capped at 100 and 50: that row, and those two caps."""
    with sps.build(SHORT, SOURCES) as model:
        assert model.solve(solver_name).termination_condition == 'infeasible'
        found = model.infeasible_subsystem()

    assert list(found.constraints) == ['balance'], 'ramp and its 1000 are slack, so only the balance row conflicts'
    assert found.constraints['balance'].to_dicts() == [{'snapshot': 1, 'sense': '==', 'rhs': 200.0}], (
        'snapshot 0 asks 60, which either technology delivers, so it is not in the subsystem'
    )
    assert list(found.bounds) == ['p'], 'p is the only variable'
    assert found.bounds['p'].to_dicts() == [
        {'snapshot': 1, 'tech': 'gas', 'bound': 'upper', 'value': 100.0},
        {'snapshot': 1, 'tech': 'wind', 'bound': 'upper', 'value': 50.0},
    ], 'both caps at snapshot 1, in label order, and no lower bound: the shortfall is on the upper side'


#: Two rows that cannot both hold at snapshot 0, with no bound in the argument.
TWO_ROWS = {
    'dimensions': {'snapshot': {'dtype': 'int'}},
    'parameters': {'need': {'dims': ['snapshot']}, 'cap': {'dims': ['snapshot']}},
    'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0}}},
    'constraints': {
        'demand': {'dims': ['snapshot'], 'expression': 'p >= need'},
        'limit': {'dims': ['snapshot'], 'expression': 'p <= cap'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

TWO_ROWS_SOURCES = {
    'snapshot': pl.DataFrame({'snapshot': [0, 1]}),
    'need': pl.DataFrame({'snapshot': [0, 1], 'value': [3.0, 0.0]}),
    'cap': pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 5.0]}),
}


def test_two_rows_that_conflict_with_no_bound_between_them(solver_name: str) -> None:
    """A conflict no bound takes part in, which HiGHS's default search, a check of bounds alone, misses."""
    with sps.build(TWO_ROWS, TWO_ROWS_SOURCES) as model:
        model.solve(solver_name)
        found = model.infeasible_subsystem()
    assert {name: frame.to_dicts() for name, frame in found.constraints.items()} == {
        'demand': [{'snapshot': 0, 'sense': '>=', 'rhs': 3.0}],
        'limit': [{'snapshot': 0, 'sense': '<=', 'rhs': 1.0}],
    }, 'snapshot 0 asks at least 3 and at most 1; snapshot 1 is slack'
    assert not found.bounds, "p's lower bound of zero is not needed for the conflict"


def test_the_subsystem_prints_one_line_per_member(solver_name: str) -> None:
    """What a caller reads in a terminal: the row, then each bound, at its coordinate."""
    with sps.build(SHORT, SOURCES) as model:
        model.solve(solver_name)
        printed = str(model.infeasible_subsystem())
    assert printed.splitlines() == [
        'balance[snapshot=1] == 200',
        'p[snapshot=1, tech=gas] <= 100 (upper bound)',
        'p[snapshot=1, tech=wind] <= 50 (upper bound)',
    ], 'constraints first, then bounds, each at its coordinate'


def test_the_last_solve_is_the_one_explained(solver_name: str) -> None:
    """New numbers pushed onto the held solver move the conflict, and the subsystem moves with it."""
    with sps.build(SHORT, SOURCES) as model:
        model.solve(solver_name)
        model.update({'demand': pl.DataFrame({'snapshot': [0, 1], 'value': [200.0, 60.0]})})
        assert model.solve(solver_name).kept == 'solver', 'only numbers moved, so the solver kept the model'
        found = model.infeasible_subsystem()
    assert found.constraints['balance'].to_dicts() == [{'snapshot': 0, 'sense': '==', 'rhs': 200.0}], (
        'the shortfall is at snapshot 0 now'
    )


def test_a_discrete_model_names_the_row_where_the_solver_finds_one(solver_name: str) -> None:
    """Integrality is what makes ``odd`` fail, and no sink reports it: the row is the whole report.

    HiGHS searches without integrality, where ``odd`` holds, so it finds none
    and the refusal names the sinks that do.
    """
    with sps.build(INTEGER, INTEGER_SOURCES) as model:
        assert model.solve(solver_name).termination_condition == 'infeasible'
        if solver_name == 'highs':
            with pytest.raises(SpecsolveError, match='integer') as refused:
                model.infeasible_subsystem()
            assert 'gurobi' in str(refused.value), 'the refusal names a sink that does find one'
            return
        found = model.infeasible_subsystem()
    assert list(found.constraints) == ['odd'], 'spare is slack'
    assert found.constraints['odd'].to_dicts() == [{'sense': '==', 'rhs': 3.0}], 'a scalar row has no dims'


def _unsolved(model: Any) -> None:
    pass


def _solved_then_updated(model: Any) -> None:
    model.solve()
    model.update({'demand': SOURCES['demand']})


def _solved_then_closed(model: Any) -> None:
    model.solve()
    model.close()


def _solved_then_broken(model: Any) -> None:
    """A second solve whose run raises after its load: the solver no longer holds what the first solved."""
    from specsolve.relational.sinks.solvers.highs import Highs

    def broken(self: Highs, handoff: object) -> None:
        raise RuntimeError('the run broke')

    model.solve()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(Highs, '_run', broken)
        with pytest.raises(RuntimeError, match='the run broke'):
            model.solve(keep='nothing')


@pytest.mark.parametrize(
    'since',
    [
        pytest.param(_unsolved, id='never-solved'),
        pytest.param(_solved_then_updated, id='updated'),
        pytest.param(_solved_then_closed, id='closed'),
        pytest.param(_solved_then_broken, id='a-solve-that-raised'),
    ],
)
def test_no_solve_to_explain_is_refused(since: Any) -> None:
    model = sps.build(SHORT, SOURCES)
    since(model)
    with pytest.raises(SpecsolveError, match='not been solved since it was built, updated or closed'):
        model.infeasible_subsystem()
    model.close()


def test_a_feasible_solve_has_no_subsystem() -> None:
    with sps.build(SHORT, FEASIBLE) as model:
        assert model.solve().termination_condition == 'optimal'
        with pytest.raises(SpecsolveError, match="'optimal'"):
            model.infeasible_subsystem()


class _StoppedShort:
    """A gurobi model whose IIS search a limit stopped: *minimal* is what ``IISMinimal`` reads, or ``None`` for unreadable."""

    def __init__(self, held: Any, minimal: int | None) -> None:
        self._held, self._minimal = held, minimal

    def __getattr__(self, name: str) -> Any:
        if name != 'IISMinimal':
            return getattr(self._held, name)
        if self._minimal is None:
            raise AttributeError("Unable to retrieve attribute 'IISMinimal'")
        return self._minimal


@pytest.mark.parametrize('minimal', [pytest.param(0, id='not-minimal'), pytest.param(None, id='nothing-readable')])
def test_a_gurobi_search_stopped_short_is_refused(minimal: int | None) -> None:
    """A time limit can stop ``computeIIS`` early, which no small model does reliably, so the model is stood in for."""
    if not SOLVERS['gurobi'].is_available():
        pytest.skip('gurobi is not installed here')
    with sps.build(SHORT, SOURCES) as model:
        model.solve('gurobi')
        held = model._engine._solver
        held._m = _StoppedShort(held._m, minimal)
        with pytest.raises(SpecsolveError, match='found no subsystem'):
            model.infeasible_subsystem()
