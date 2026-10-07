"""The irreducible infeasible subsystem a live solver finds, read back by declaration and coordinate.

The model is a dispatch whose second snapshot asks for more than both
technologies can deliver, so the one minimal conflict is that snapshot's
balance row against the two upper bounds it pushes on. ``ramp`` and the first
snapshot are feasible, and must stay out of it.
"""

from __future__ import annotations

import gc
from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError

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
        answer = model.solve(solver_name)
        assert answer.termination_condition == 'infeasible'
        found = answer.iis()

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
        found = model.solve(solver_name).iis()
    assert {name: frame.to_dicts() for name, frame in found.constraints.items()} == {
        'demand': [{'snapshot': 0, 'sense': '>=', 'rhs': 3.0}],
        'limit': [{'snapshot': 0, 'sense': '<=', 'rhs': 1.0}],
    }, 'snapshot 0 asks at least 3 and at most 1; snapshot 1 is slack'
    assert not found.bounds, "p's lower bound of zero is not needed for the conflict"


def test_the_subsystem_prints_one_line_per_member(solver_name: str) -> None:
    """What a caller reads in a terminal: the row, then each bound, at its coordinate."""
    with sps.build(SHORT, SOURCES) as model:
        printed = str(model.solve(solver_name).iis())
    assert printed.splitlines() == [
        'balance[snapshot=1] == 200',
        'p[snapshot=1, tech=gas] <= 100 (upper bound)',
        'p[snapshot=1, tech=wind] <= 50 (upper bound)',
    ], 'constraints first, then bounds, each at its coordinate'


def test_a_discrete_model_names_the_row_where_the_solver_finds_one(solver_name: str) -> None:
    """Integrality is what makes ``odd`` fail, and no sink reports it: the row is the whole report.

    HiGHS searches without integrality, where ``odd`` holds, so it finds none
    and the refusal names the sinks that do.
    """
    with sps.build(INTEGER, INTEGER_SOURCES) as model:
        answer = model.solve(solver_name)
        assert answer.termination_condition == 'infeasible'
        if solver_name == 'highs':
            with pytest.raises(SpecsolveError, match='integer') as refused:
                answer.iis()
            assert 'gurobi' in str(refused.value), 'the refusal names a sink that does find one'
            return
        found = answer.iis()
    assert list(found.constraints) == ['odd'], 'spare is slack'
    assert found.constraints['odd'].to_dicts() == [{'sense': '==', 'rhs': 3.0}], 'a scalar row has no dims'


def _solved_again(model: Any, solver_name: str) -> None:
    model.solve(solver_name)


def _updated(model: Any, solver_name: str) -> None:
    model.update({'demand': SOURCES['demand']})


def _closed(model: Any, solver_name: str) -> None:
    model.close()


@pytest.mark.parametrize(
    'moved_on',
    [
        pytest.param(_solved_again, id='solved-again'),
        pytest.param(_updated, id='updated'),
        pytest.param(_closed, id='closed'),
    ],
)
def test_a_solver_that_moved_on_is_refused(solver_name: str, moved_on: Any) -> None:
    """The solver now holds another model, or none: what it would find is not this answer's."""
    model = sps.build(SHORT, SOURCES)
    answer = model.solve(solver_name)
    moved_on(model, solver_name)
    with pytest.raises(SpecsolveError, match='ask iis'):
        answer.iis()
    model.close()


def test_a_model_dropped_without_closing_takes_its_solver_with_it() -> None:
    """The result holds the engine weakly, so it never keeps a solver, or a licence, alive."""
    model = sps.build(SHORT, SOURCES)
    answer = model.solve()
    del model
    gc.collect()
    with pytest.raises(SpecsolveError, match='ask iis'):
        answer.iis()


def test_the_one_call_solve_has_no_solver_left() -> None:
    """``sps.solve`` closes the model before it returns, and the refusal says how to keep one."""
    answer = sps.solve(SHORT, SOURCES)
    assert answer.termination_condition == 'infeasible'
    with pytest.raises(SpecsolveError, match=r'sps\.build'):
        answer.iis()


def test_a_feasible_solve_has_no_subsystem() -> None:
    with sps.build(SHORT, FEASIBLE) as model:
        answer = model.solve()
        assert answer.termination_condition == 'optimal'
        with pytest.raises(SpecsolveError, match="'optimal'"):
            answer.iis()


def test_a_closed_result_is_refused() -> None:
    with sps.build(SHORT, SOURCES) as model:
        answer = model.solve()
        answer.close()
        with pytest.raises(SpecsolveError, match='closed'):
            answer.iis()
