"""Ranging: how far each cost and each right-hand side can move with the optimal basis still optimal, on every sink.

The oracle is a one-snapshot dispatch worked out by hand. Wind runs at its cap,
gas at the floor ``must_run`` holds it to, and solar is the marginal generator
that serves the rest, so the price is solar's cost. Two rows never bind,
``ceiling`` from above and ``floor`` from below. Every column and row is in
one basis only, so each range is the same on every sink.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve._answer_layout import NO_RANGING
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output

RANGING: frozenset[Output] = frozenset({'ranging'})

GENERATORS = ['wind', 'solar', 'gas']

SPEC = {
    'dimensions': {'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'floor': {'dims': ['generator']},
    },
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {
        'balance': {'dims': [], 'expression': 'sum(p, over=generator) == 60'},
        'must_run': {'dims': ['generator'], 'where': 'floor > 0', 'expression': 'p >= floor'},
        'ceiling': {'dims': [], 'expression': 'sum(p, over=generator) <= 1000'},
        'floor_total': {'dims': [], 'expression': 'sum(p, over=generator) >= 10'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

SOURCES = {
    'generator': GENERATORS,
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [50.0, 30.0, 200.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 50.0]}),
    'floor': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 5.0]}),
}

#: Wind stays at its cap while it is cheaper than solar, solar stays marginal
#: between wind's cost and gas's, and gas stays held at its floor while it
#: costs more than the price.
BY_HAND_COSTS = {'wind': (-math.inf, 2.0), 'solar': (1.0, 50.0), 'gas': (2.0, math.inf)}

#: Solar serves the load between 0 and its cap of 30 on top of wind's 50 and
#: gas's 5, gas's floor moves solar the other way until solar or gas reaches
#: zero, and a row that does not bind ranges from its activity, 60, without
#: limit on its open side.
BY_HAND_RHS = {
    'balance': (55.0, 85.0),
    'must_run': (0.0, 10.0),
    'ceiling': (60.0, math.inf),
    'floor_total': (-math.inf, 60.0),
}

#: Xpress widens each finite end by its feasibility and optimality
#: tolerances, which are ``1e-6`` by default.
TOLERANCE = 1e-5


def _ends(lower: pl.DataFrame, upper: pl.DataFrame) -> list[tuple[float, float]]:
    return list(zip(lower['value'].to_list(), upper['value'].to_list(), strict=True))


def _close(found: dict[str, tuple[float, float]], expected: dict[str, tuple[float, float]]) -> bool:
    return found.keys() == expected.keys() and all(
        np.allclose(found[key], expected[key], rtol=0.0, atol=TOLERANCE) for key in expected
    )


def test_every_sink_reads_the_ranges_worked_out_by_hand(solver_name: str) -> None:
    with sps.solve(SPEC, SOURCES, solver_name=solver_name, outputs=RANGING) as answer:
        costs = dict(zip(GENERATORS, _ends(answer.cost_lower('p'), answer.cost_upper('p')), strict=True))
        rhs = {name: _ends(answer.rhs_lower(name), answer.rhs_upper(name))[0] for name in BY_HAND_RHS}
    assert _close(costs, BY_HAND_COSTS), f'the cost ranges of the vertex worked out by hand, not {costs}'
    assert _close(rhs, BY_HAND_RHS), f'the right-hand side ranges worked out by hand, not {rhs}'


def test_a_maximised_objective_ranges_its_costs_in_its_own_sense(solver_name: str) -> None:
    """The same dispatch, maximising the negated cost: each range is the minimised one negated, its ends swapped."""
    spec = {**SPEC, 'objective': {'sense': 'maximize', 'expression': 'sum(p * cost)'}}
    sources = {**SOURCES, 'cost': SOURCES['cost'].with_columns(-pl.col('value'))}
    with sps.solve(spec, sources, solver_name=solver_name, outputs=RANGING) as answer:
        costs = dict(zip(GENERATORS, _ends(answer.cost_lower('p'), answer.cost_upper('p')), strict=True))
    negated = {name: (-upper, -lower) for name, (lower, upper) in BY_HAND_COSTS.items()}
    assert _close(costs, negated), f'each range negated and its ends swapped, not {costs}'


def test_a_range_reads_back_with_its_coordinates() -> None:
    with sps.solve(SPEC, SOURCES, outputs=RANGING) as answer:
        lower = answer.cost_lower('p')
        rows = answer.rhs_upper('must_run')
    assert lower.columns == ['generator', 'value'], 'the primal shape: the dims, then the value'
    assert lower['generator'].to_list() == GENERATORS, 'in the order the primal comes back in'
    assert rows['generator'].to_list() == ['gas'], "only the row the constraint's where builds"


def test_a_range_converts_like_any_other_output() -> None:
    pytest.importorskip('xarray')
    with sps.solve(SPEC, SOURCES, outputs=RANGING) as answer:
        array = answer.to_dataarray('p', kind='cost_upper')
    assert array.sel(generator='solar').item() == pytest.approx(50.0), "solar's upper cost end, labelled by generator"


def test_a_saved_and_an_archived_answer_read_back_their_ranges(tmp_path: Path) -> None:
    readers = (('cost_lower', 'p'), ('cost_upper', 'p'), ('rhs_lower', 'balance'), ('rhs_upper', 'ceiling'))
    with sps.solve(SPEC, SOURCES, outputs=RANGING, archive=tmp_path / 'case') as answer:
        live = {reader: getattr(answer, reader)(name) for reader, name in readers}
        answer.save(tmp_path / 'saved')
    archived = sps.load_archive(tmp_path / 'case')
    assert isinstance(archived, sps.types.ResultArchive)
    for read in (sps.load_result(tmp_path / 'saved'), archived.result):
        for reader, name in readers:
            assert getattr(read, reader)(name).equals(live[reader]), f'{reader} of {name!r} reads back as solved'


@pytest.mark.parametrize('reader', ['cost_lower', 'cost_upper', 'rhs_lower', 'rhs_upper'])
def test_a_solve_not_asked_for_its_ranging_names_the_output(reader: str) -> None:
    name = 'p' if reader.startswith('cost') else 'balance'
    with (
        sps.solve(SPEC, SOURCES) as answer,
        pytest.raises(SpecsolveError, match=r"outputs=\{'ranging'\}"),
    ):
        getattr(answer, reader)(name)


INTEGER = {
    **SPEC,
    'variables': {'p': {**SPEC['variables']['p'], 'domain': 'integer'}},
}

QUADRATIC = {**SPEC, 'objective': {'sense': 'minimize', 'expression': 'sum(p * cost) + sum(p * p)'}}

#: Each sink's own way to an LP answer that is not a vertex, as in ``test_basis``.
NO_CROSSOVER = {
    'highs': {'solver': 'ipm', 'run_crossover': 'off', 'presolve': 'off'},
    'gurobi': {'Method': 2, 'Crossover': 0},
    'xpress': {'lpflags': 4, 'crossover': 0},
}


@pytest.mark.parametrize(
    ('spec', 'how'),
    [
        pytest.param(INTEGER, None, id='mixed-integer'),
        pytest.param(SPEC, 'no-crossover', id='interior-point-without-crossover'),
    ],
)
def test_a_solve_that_left_no_range_says_why(solver_name: str, spec: dict, how: str | None) -> None:
    options = NO_CROSSOVER[solver_name] if how else None
    with (
        sps.solve(spec, SOURCES, solver_name=solver_name, solver_options=options, outputs=RANGING) as answer,
        pytest.raises(SpecsolveError, match=NO_RANGING[:40]),
    ):
        answer.cost_lower('p')


@pytest.mark.parametrize('solver', ['highs', 'gurobi'])
def test_a_quadratic_objective_leaves_no_range(solver: str) -> None:
    """A range is around an LP's basis; the two sinks that take a quadratic objective both leave none."""
    if solver == 'gurobi':
        pytest.importorskip('gurobipy')
    with (
        sps.solve(QUADRATIC, SOURCES, solver_name=solver, outputs=RANGING) as answer,
        pytest.raises(SpecsolveError, match=NO_RANGING[:40]),
    ):
        answer.rhs_lower('balance')


def test_a_run_a_limit_stopped_leaves_no_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """A range is read only off an optimal solve, whatever the sink offers.

    HiGHS's primal simplex stops after one iteration at a feasible point that
    is not optimal. HiGHS itself has no ranging there, so the double offers
    some: the guard is the base's `run`, which must not ask.
    """
    from specsolve._sinks import SOLVERS
    from specsolve._sinks.highs import Highs

    class Eager(Highs):
        def _ranging(self):
            columns, rows = self._columns, self._rows
            return np.zeros(columns), np.zeros(columns), np.zeros(rows), np.zeros(rows)

    monkeypatch.setitem(SOLVERS, 'highs', Eager)
    options = {'simplex_strategy': 4, 'simplex_iteration_limit': 1, 'presolve': 'off'}
    with sps.solve(SPEC, SOURCES, solver_options=options, outputs=RANGING) as answer:
        assert answer.termination_condition == 'iteration_limit', 'the run stopped short of optimal'
        assert answer.has_primal, 'at a feasible point, so there is a primal to read'
        with pytest.raises(SpecsolveError, match=NO_RANGING[:40]):
            answer.cost_lower('p')


@pytest.mark.parametrize('side', ['columns', 'rows'])
def test_a_range_that_does_not_span_the_model_is_refused(monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    """A range is positional, as a primal is, so one a member reads short is a different model's."""
    from specsolve._sinks import SOLVERS
    from specsolve._sinks.highs import Highs

    class Crooked(Highs):
        def _ranging(self):
            low, up, rhs_low, rhs_up = super()._ranging()
            return (low[:-1], up, rhs_low, rhs_up) if side == 'columns' else (low, up, rhs_low[:-1], rhs_up)

    monkeypatch.setitem(SOLVERS, 'highs', Crooked)
    which = {
        'columns': 'cost lower values for a model with 3',
        'rows': 'right-hand side lower values for a model with 4',
    }
    with pytest.raises(SpecsolveError, match=which[side]):
        sps.solve(SPEC, SOURCES, outputs=RANGING)


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------


def test_a_sweep_reads_each_slices_ranges(tmp_path: Path) -> None:
    from tests.test_strategy import DISPATCH, scenario_sources

    sources = scenario_sources()
    axis = sps.EachCoordinate('scenario')
    sps.solve_over(DISPATCH, sources, axis, outputs=RANGING, archive=tmp_path / 'run.zip')
    archive = sps.load_archive(tmp_path / 'run.zip')
    assert isinstance(archive, sps.types.SweepArchive)
    held = sps.solve_over(DISPATCH, sources, axis, outputs=RANGING)
    for sweep in (held, archive.sweep):
        upper = sweep.rhs_upper('balance')
        assert upper.height == 12, 'three scenarios of four snapshots'
        first = upper['scenario'][0]
        alone = dict(sources)
        alone['load'] = sources['load'].filter(pl.col('scenario') == first).drop('scenario')
        with sps.solve(DISPATCH, alone, outputs=RANGING) as answer:
            expected = answer.rhs_upper('balance')
        assert upper.filter(pl.col('scenario') == first).drop('scenario').equals(expected), (
            "a slice's ranges are its own solve's"
        )


def test_a_sweep_of_slices_that_left_no_range_says_why() -> None:
    from tests.test_strategy import DISPATCH, scenario_sources

    sweep = sps.solve_over(
        DISPATCH,
        scenario_sources(),
        sps.EachCoordinate('scenario'),
        solver_options=NO_CROSSOVER['highs'],
        outputs=RANGING,
    )
    with pytest.raises(SpecsolveError, match=NO_RANGING[:40]):
        sweep.cost_lower('p')
