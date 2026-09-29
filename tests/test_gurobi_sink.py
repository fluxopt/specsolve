"""The ``gurobi`` sink, against the ``highs`` sink.

Two sinks loading one :class:`Handoff` must produce the same model, so the
assertions here are agreements. Where a value is asserted it comes from
``examples/ports/references.json``, somebody else's published optimum.

Every test skips without ``gurobipy``. Its wheel ships a licence limited to
2000 columns, so the models here stay small. A port that outgrows it is named in
``OVER_THE_GUROBI_LIMIT`` and skipped rather than shrunk, since half a model
reaches no published number.
"""

from __future__ import annotations

import builtins
import gc
from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.solvers.gurobi import build_gurobi
from tests.conftest import (
    CASES,
    QP,
    QP_SOURCES,
    assert_agrees_with_highs,
    assert_infeasible_reports_both_axes,
    expanded,
    port_sources,
)

gurobipy = pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')


# ---------------------------------------------------------------------------
# the two sinks answer the same question the same way
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('name', 'variable', 'constraint', 'has_duals'),
    [
        pytest.param('LP', 'p', 'meet', True, id='lp'),
        pytest.param('MAX', 'p', 'lim', True, id='max'),
        pytest.param('MIP', 'x', 'budget', False, id='mip'),
        pytest.param('QP', 'p', 'meet', True, id='qp'),
    ],
)
def test_gurobi_and_highs_agree(name: str, variable: str, constraint: str, has_duals: bool) -> None:
    assert_agrees_with_highs('gurobi', name, variable, constraint, has_duals=has_duals)


#: `osemosys_utopia` builds 5,733 columns against the bundled licence's 2,000.
#: On an unrestricted licence it reaches 29446.862694340936, against
#: 29446.86269434094 from highs and OSeMOSYS's own 29446.86269.
OVER_THE_GUROBI_LIMIT = {'osemosys_utopia'}


def test_every_port_reaches_its_reference_optimum_on_gurobi(port: dict[str, Any]) -> None:
    """``test_ports.py``'s corpus, solved by the other solver.

    A sink that mis-loads the matrix still reaches *a* number; this checks it
    against a published one.
    """
    if port['name'] in OVER_THE_GUROBI_LIMIT:
        pytest.skip(f'{port["name"]} exceeds the bundled gurobi licence — see OVER_THE_GUROBI_LIMIT')
    with sps.solve(expanded(port['spec'], 'piecewise'), port_sources(port['name']), solver_name='gurobi') as solution:
        assert solution.is_ok, f'{port["name"]} did not solve: {solution.status}'
        assert solution.objective == pytest.approx(port['objective'], rel=port['rtol'])


def test_block_boundaries_do_not_move_the_answer() -> None:
    """``batch_rows=1`` forces one block per row, so every CSR view is built at
    a boundary — where an off-by-one in ``indptr`` shifts coefficients into the
    neighbouring row rather than dropping them."""
    with sps.build(*CASES['LP']) as model:
        whole = model.solve(solver_name='gurobi')
        tables = model._engine._model.handoff
        with build_gurobi(tables, batch_rows=1) as sink:
            ragged = sink.run(tables)
        assert ragged.objective == pytest.approx(whole.objective)
        held = model._engine._model.variables['p']
        assert held.share(ragged.primal).to_list() == pytest.approx(whole.primal('p')['value'].to_list())


# ---------------------------------------------------------------------------
# what the sink says when there is nothing to read
# ---------------------------------------------------------------------------


def test_an_infeasible_solve_reports_both_axes_in_gurobis_wording() -> None:
    assert_infeasible_reports_both_axes('gurobi')


def test_gurobi_takes_the_two_quadratic_models_highs_refuses() -> None:
    """The capability axis from the side that has the capability: naming this
    sink in a refusal is only true if it actually solves what HiGHS will not."""
    need = QP_SOURCES

    nonconvex = QP | {'objective': {'sense': 'minimize', 'expression': '-sum(p * p, over=g)'}}
    with pytest.raises(SpecsolveError, match='not positive semidefinite'):
        sps.solve(nonconvex, need)
    assert sps.solve(nonconvex, need, solver_name='gurobi').objective == pytest.approx(-200.0), (
        'the concave objective is driven to the bound on both columns, which only a spatial branch-and-bound finds'
    )

    integral = QP | {'variables': {**QP['variables'], 'p': {**QP['variables']['p'], 'domain': 'integer'}}}
    with pytest.raises(SpecsolveError, match='separately and refuses them together'):
        sps.solve(integral, need)
    with sps.solve(integral, need, solver_name='gurobi') as mixed, sps.solve(QP, need, solver_name='gurobi') as lp:
        assert mixed.is_ok
        assert mixed.objective == pytest.approx(lp.objective), (
            'the integral optimum is integral here, so the MIQP reaches the relaxation exactly — '
            'what is under test is that the pair loads at all'
        )


def test_a_pushed_quadratic_objective_replaces_rather_than_accumulates() -> None:
    """What an update must not do twice. ``setMObjective`` replaces the whole
    objective, which is why the linear cost is passed to it again; accumulating
    would answer twice the curvature on the second solve."""
    scaled = QP | {
        'parameters': {'need': {'dims': []}, 'toll': {'dims': ['g']}, 'wear': {'dims': []}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p * p * wear + p * q + q * q + q * toll, over=g)'},
    }
    soft = QP_SOURCES | {'wear': pl.DataFrame({'value': [1.0]})}
    stiff = QP_SOURCES | {'wear': pl.DataFrame({'value': [4.0]})}

    with sps.build(scaled, soft) as model:
        first = model.solve(solver_name='gurobi').objective
        model.update(stiff)
        pushed = model.solve(solver_name='gurobi').objective
        assert model.diagnostics().loads == 1, 'the pattern did not move, so the coefficients are pushed'

    assert pushed != pytest.approx(first), 'a stiffer model is a different answer'
    with sps.solve(scaled, stiff, solver_name='gurobi') as fresh:
        assert pushed == pytest.approx(fresh.objective), (
            'an update answers what a fresh build answers — a quadratic part left unreplaced would '
            'report the old curvature, and one accumulated would report both'
        )


def test_a_mixed_integer_model_has_no_duals() -> None:
    """Gurobi refuses ``Pi`` rather than returning zeros; the sink passes the
    refusal on as the ``None`` that makes ``dual`` explain itself."""
    with sps.solve(*CASES['MIP'], solver_name='gurobi') as solution:
        assert solution.has_primal
        with pytest.raises(SpecsolveError, match='mixed-integer'):
            solution.dual('budget')


def test_solver_options_reach_gurobi() -> None:
    """Verbatim, in Gurobi's own vocabulary — ``TimeLimit``, not HiGHS' ``time_limit``."""
    with sps.solve(*CASES['MIP'], solver_options={'TimeLimit': 0.0}, solver_name='gurobi') as solution:
        assert solution.termination_condition == 'time_limit'
    with pytest.raises(gurobipy.GurobiError, match='no_such_parameter'):
        sps.solve(*CASES['MIP'], solver_options={'no_such_parameter': 1}, solver_name='gurobi')


# ---------------------------------------------------------------------------
# the seams: the build without the search, and choosing a sink at all
# ---------------------------------------------------------------------------


def test_solver_options_land_on_the_environment() -> None:
    """Options land on the environment, where a licence parameter has to go.

    ``WLSAccessID``, ``ComputeServer`` and ``TokenServer`` can only be set
    before an environment starts. Asserted through an ordinary parameter, which
    the model sees as its default.
    """
    with (
        sps.build(*CASES['MIP']) as model,
        build_gurobi(model._engine._model.handoff, solver_options={'TimeLimit': 5.0}) as solver,
    ):
        assert solver.handle.Params.TimeLimit == 5.0


def test_build_gurobi_loads_the_model_and_stops() -> None:
    """`bench/`'s seam: the hand-off with no search behind it, so what it
    reports is what was loaded rather than what was solved."""
    with sps.build(*CASES['MIP']) as model:
        tables = model._engine._model.handoff
        with build_gurobi(tables) as solver:
            m = solver.handle
            assert (m.NumVars, m.NumConstrs) == (tables.column_count, tables.row_count)
            assert m.NumIntVars == tables.cols.filter(pl.col('vtype') != 'continuous').height
            assert m.ModelSense == gurobipy.GRB.MAXIMIZE
            assert m.SolCount == 0


def test_a_dropped_solver_disposes_the_model_it_holds() -> None:
    """The licence a loaded model holds is released when its holder goes.

    Gurobi keeps an environment until its last model is gone, so dropping a
    :class:`Gurobi` without ``close()`` disposes the model too. Asserted through
    a model the caller still holds, where a refcount could not release it.
    """
    with sps.build(*CASES['MIP']) as model:
        solver = build_gurobi(model._engine._model.handoff)
        m = solver.handle
        del solver
        gc.collect()
        with pytest.raises(gurobipy.GurobiError, match='freed'):
            _ = m.NumVars


def test_close_disposes_a_model_the_caller_still_holds() -> None:
    """``close()`` is the release, not a hint to the collector — and it is idempotent."""
    with sps.build(*CASES['MIP']) as model:
        solver = build_gurobi(model._engine._model.handoff)
        m = solver.handle
        solver.close()
        solver.close()
        with pytest.raises(gurobipy.GurobiError, match='freed'):
            _ = m.NumVars


def test_a_load_that_fails_releases_its_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """An environment started for a load that raises is disposed before the error leaves.

    Otherwise a model the sink refuses part way — a bad option value, a
    matrix Gurobi rejects — would hold a licence until the collector found the
    half-built model.
    """
    from specsolve.relational.sinks.solvers import gurobi as sink

    events: list[str] = []
    real_env = gurobipy.Env

    class Env(real_env):  # type: ignore[misc, valid-type]
        def dispose(self) -> None:
            events.append('env disposed')
            super().dispose()

    monkeypatch.setattr(gurobipy, 'Env', Env)
    monkeypatch.setattr(sink, '_filled', lambda *args: (_ for _ in ()).throw(RuntimeError('mid-load')))
    with sps.build(*CASES['MIP']) as model:
        try:
            build_gurobi(model._engine._model.handoff)
        except RuntimeError:
            events.append('error left')
    assert events[:2] == ['env disposed', 'error left'], (
        'the environment is disposed before the error reaches the caller; gurobipy disposes again at dealloc'
    )


def test_the_objective_constant_rides_on_the_model_not_the_answer() -> None:
    """Gurobi has ``ObjCon``, so the constant is part of the model it holds —
    which makes the build seam a complete hand-off rather than a model plus a
    number to remember."""
    with sps.build(*CASES['MAX']) as model, build_gurobi(model._engine._model.handoff) as solver:
        assert solver.handle.ObjCon == pytest.approx(5.0)


def test_the_missing_extra_is_named() -> None:
    """What a caller without gurobipy meets — both halves named, since the
    absent one is as often scipy."""
    real_import = builtins.__import__

    def refuse(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {'gurobipy', 'scipy.sparse'}:
            raise ModuleNotFoundError(f'No module named {name!r}')
        return real_import(name, *args, **kwargs)

    with sps.build(*CASES['LP']) as model, pytest.MonkeyPatch.context() as patch:
        patch.setattr(builtins, '__import__', refuse)
        with pytest.raises(ModuleNotFoundError, match=r'\[gurobi\] extra \(gurobipy, scipy\)'):
            model.solve(solver_name='gurobi')
