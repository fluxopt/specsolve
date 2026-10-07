"""The ``xpress`` sink, with HiGHS as its oracle.

Three sinks loading one :class:`Handoff` must produce the same model, so the
assertions are agreements. Where a value *is* asserted it comes from
``examples/ports/references.json``.

Every test skips without ``xpress``. Its wheel carries a Community licence,
active on import, that refuses a model whose rows *plus* columns exceed 5000;
the one port over it is named in ``OVER_THE_XPRESS_LIMIT``.

Where this sink differs from the other two: the objective's constant is a
*column*, discarding a solve is a control rather than a call, and an unsolved
problem hands back a trivial basis where Gurobi refuses one.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.solvers.xpress import Xpress
from tests.conftest import (
    CASES,
    assert_agrees_with_highs,
    assert_infeasible_reports_both_axes,
    expanded,
    port_sources,
)

xpress = pytest.importorskip('xpress', reason='the xpress sink needs the [xpress] extra')


@pytest.mark.parametrize(
    ('name', 'variable', 'constraint', 'has_duals'),
    [
        pytest.param('LP', 'p', 'meet', True, id='lp'),
        pytest.param('MAX', 'p', 'lim', True, id='max'),
        pytest.param('MIP', 'x', 'budget', False, id='mip'),
    ],
)
def test_xpress_and_highs_agree(name: str, variable: str, constraint: str, has_duals: bool) -> None:
    """The shared cases: a maximisation, an objective constant and an integrality, each stated outside the frames."""
    assert_agrees_with_highs('xpress', name, variable, constraint, has_duals=has_duals)


#: `osemosys_utopia` builds 10,857 rows plus columns against the Community
#: licence's 5,000.
OVER_THE_XPRESS_LIMIT = {'osemosys_utopia'}


def test_every_port_reaches_its_reference_optimum_on_xpress(port: dict[str, Any]) -> None:
    """``test_ports.py``'s corpus, solved by the third solver.

    The one assertion here no part of this package produced. A sink that
    mis-loads the matrix — a block boundary off by a row, a sense inverted —
    still reaches *a* number; this is what that number is checked against.
    """
    if port['name'] in OVER_THE_XPRESS_LIMIT:
        pytest.skip(f'{port["name"]} exceeds the bundled xpress licence — see OVER_THE_XPRESS_LIMIT')
    with sps.solve(expanded(port['spec'], 'piecewise'), port_sources(port['name']), solver_name='xpress') as solution:
        assert solution.is_ok, f'{port["name"]} did not solve: {solution.status}'
        assert solution.objective == pytest.approx(port['objective'], rel=port['rtol'])


@pytest.mark.parametrize('batch_rows', [None, 1, 2, 10_000], ids=lambda n: f'batch-{n}')
def test_block_boundaries_do_not_move_the_answer(batch_rows: int | None) -> None:
    """The matrix goes in a block at a time, and a block is a slice of rows.

    A boundary that dropped or repeated a row's entries would still solve, so
    the budget is varied against a fixed answer rather than asserted about.
    """
    spec, data = CASES['LP']
    with sps.build(spec, data) as model:
        tables = model._engine._model.handoff
        reference = model.solve().objective
    problem = Xpress(tables, batch_rows=batch_rows).handle
    problem.optimize()
    assert float(problem.attributes.objval) == pytest.approx(reference), f'batch_rows={batch_rows} moved the answer'


def test_an_infeasible_solve_reports_both_axes_in_xpress_wording() -> None:
    assert_infeasible_reports_both_axes('xpress')


def test_a_mixed_integer_model_has_no_duals() -> None:
    """Xpress refuses the read rather than handing back zeros, and the refusal
    is the answer — a zero vector would be indistinguishable from real prices."""
    with (
        sps.solve(*CASES['MIP'], solver_name='xpress') as solution,
        pytest.raises(SpecsolveError, match='mixed-integer'),
    ):
        solution.dual('budget')


def test_the_objective_constant_rides_on_the_model_not_the_answer() -> None:
    """The one quantity this sink spells as a *column*.

    Xpress carries it as the objective coefficient of column ``-1``, negated,
    where the other two sinks set an attribute — so a sign error here is a
    model that solves and answers wrong by a constant.
    """
    with sps.solve(*CASES['MAX'], solver_name='xpress') as solution:
        assert solution.objective == pytest.approx(12.0), 'cap 3 + cap 4, plus the declared 5'


def test_forgetting_makes_the_next_solve_start_cold() -> None:
    """``keepbasis``, not ``problem.reset()``, which clears the problem itself.

    A re-solve that kept the basis does no simplex work, and one that forgot it
    does the same work as the first.
    """
    from tests.test_warm_start import DISPATCH, SNAPSHOTS, dispatch_sources

    with sps.build(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}) as model:
        tables = model._engine._model.handoff
    session = Xpress(tables)
    try:
        session.run(tables)
        cold = int(session._p.attributes.simplexiter)
        assert cold > 0, 'the model must make the simplex work, or this is unobservable'

        session.run(tables)
        assert int(session._p.attributes.simplexiter) == 0, 'a re-solve that kept the basis has nothing to do'

        session.forget()
        session.run(tables)
        assert int(session._p.attributes.simplexiter) == cold, 'forget() must send the next solve back to scratch'
    finally:
        session.close()


def test_solver_options_reach_xpress() -> None:
    """Forwarded verbatim, in the solver's own vocabulary — a control name here."""
    spec, data = CASES['LP']
    with sps.build(spec, data) as model:
        tables = model._engine._model.handoff
    problem = Xpress(tables, solver_options={'timelimit': 42}).handle
    assert int(problem.controls.timelimit) == 42, 'the option did not reach the problem'


def test_constructing_xpress_loads_the_model_and_stops() -> None:
    """The seam `bench/` measures: a loaded problem, unsolved."""
    spec, data = CASES['LP']
    with sps.build(spec, data) as model:
        tables = model._engine._model.handoff
    problem = Xpress(tables).handle
    assert (problem.attributes.rows, problem.attributes.cols) == (tables.row_count, tables.column_count)
    assert int(problem.attributes.solvestatus) == 0, 'constructing Xpress loads the model and does not solve it'


def test_a_set_reaches_the_solver_natively() -> None:
    """``sos`` is listed, so the family hands the sets over as sets — asserted
    on the enumerated optimum, plus the count the solver itself reports."""
    from tests.test_sos import DATA, best, spec

    with sps.build(spec(2), DATA) as model:
        tables = model._engine._model.handoff
    problem = Xpress(tables).handle
    assert int(problem.attributes.sets) == 2, 'both declared sets reached the solver as sets'
    problem.optimize()
    assert float(problem.attributes.objval) == pytest.approx(best(2))


#: A model whose optimum leaves a row **slack**. Every case in ``CASES`` binds
#: every constraint, so ``rhs - slack`` and ``rhs`` agree there and the
#: subtraction is invisible — this is the model that separates them.
SLACK = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['t']}, 'price': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'lim': {'dims': ['t'], 'expression': 'p <= cap'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * price, over=t) + 0'},
}

SLACK_DATA = {
    't': [0, 1],
    'cap': pl.DataFrame({'t': [0, 1], 'value': [10.0, 20.0]}),
    'price': pl.DataFrame({'t': [0, 1], 'value': [1.0, 2.0]}),
}


def test_activity_is_the_row_value_and_not_its_right_hand_side() -> None:
    """A non-binding row, which no case in ``CASES`` has.

    Xpress reports a *slack* and this sink subtracts it from the right-hand
    side to recover the row's own value. Here the minimum parks both variables
    at 0 against caps of 10 and 20, so activity and rhs differ by the whole of
    each bound.
    """
    with sps.solve(SLACK, SLACK_DATA, solver_name='xpress', outputs={'activity'}) as solution:
        assert solution.activity('lim')['value'].to_list() == pytest.approx([0.0, 0.0]), (
            'activity is the row value at the solution, not the bound it was compared against'
        )


def test_a_solve_that_errored_is_not_reported_as_unknown() -> None:
    """The second axis, which linopy's map does not read.

    A unit probe rather than a solve, since the suite cannot make the Optimizer
    *fail* reliably. A failure and a model nobody solved do not arrive as the
    same word.
    """
    from types import SimpleNamespace

    from specsolve.relational.sinks.solvers.xpress import _status_of

    failed = _status_of(SimpleNamespace(attributes=SimpleNamespace(solvestatus=2, solstatus=0)))
    assert failed.termination_condition == 'internal_solver_error'
    assert failed.status == 'error'
    assert not failed.has_primal

    unsolved = _status_of(SimpleNamespace(attributes=SimpleNamespace(solvestatus=0, solstatus=0)))
    assert unsolved.termination_condition == 'unknown', 'nothing solved yet is not a solver failure'


def test_the_missing_extra_is_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """A caller without the package is told which extra to install, once —
    the same sentence whether the refusal lands early or at the import."""
    monkeypatch.setattr(Xpress, 'requires', ('xpress_not_installed',))
    assert not Xpress.is_available()
    with pytest.raises(ModuleNotFoundError, match=r'\[xpress\] extra'):
        sps.solve(*CASES['LP'], solver_name='xpress')


def test_a_basis_is_loaded_in_the_row_statuses_xpress_itself_reports() -> None:
    """A basis read from Xpress and loaded back is the one Xpress holds, row by row and column by column.

    Which side of its slack a nonbasic row names is not observable here: Xpress
    puts the slack at its one finite bound on load, whichever side it was given.
    """
    from tests.test_basis import SPEC

    with sps.build(SPEC, {}) as model:
        tables = model._engine._model.handoff
    first = Xpress(tables)
    try:
        basis = first.run(tables, basis=True).basis
        native_rows, native_columns = first._p.getBasis()
    finally:
        first.close()
    assert basis is not None

    fresh = Xpress(tables)
    try:
        fresh.warm(basis)
        loaded_rows, loaded_columns = fresh._p.getBasis()
    finally:
        fresh.close()
    assert list(loaded_rows) == list(native_rows), 'the rows go back in the statuses they were read in'
    assert list(loaded_columns) == list(native_columns), 'and the columns'
