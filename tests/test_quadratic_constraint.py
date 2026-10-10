"""A quadratic constraint — the construct with no two-lane oracle.

`linopy.Model` refuses a `QuadraticExpression` in a constraint, so two weaker
oracles replace the differential one:

1. **Two independent encodings** — the direct `addMQConstr` hand-off against
   the same model as LP text read back by Gurobi's parser. They share the
   frames and nothing below, so an encoding error shows up as two numbers.
2. **A residual at the primal** — `xᵀQx + Ax` recomputed from the built frames
   (`conftest.recomputed_row_values`) against the activity the solver reports.

Neither catches a *shared misreading*, so one test here is an optimum done by
hand.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from tests.conftest import recomputed_row_values

gurobipy = pytest.importorskip('gurobipy', reason='a quadratic constraint has no other solver sink')

RTOL = 1e-6

#: What a *spatial* branch-and-bound proves is a gap, not a point: it stops on
#: ``MIPGap`` (1e-4 relative by default), so two searches of one model may end
#: on different vertices of the same optimal face. Wide enough for that, and
#: still two orders inside the ~30% that separates the models under test.
SEARCHED = 1e-2

#: ``p·q >= floor`` per generator, with a linear cap over the pair. Four
#: columns, so it runs under the bundled licence — and the optimum is
#: arithmetic: at the bound ``p = q = 2`` on each generator.
SPEC = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'floor': {'dims': []}},
    'variables': {
        'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
        'q': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {
        'cap': {'dims': [], 'expression': 'sum(p, over=g) <= 9'},
        'coupled': {'dims': ['g'], 'expression': 'p * q >= floor'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p + q, over=g)'},
}

SOURCES = {'g': ['a', 'b'], 'floor': pl.DataFrame({'value': [4.0]})}


def spec(**patch) -> dict:
    return {**SPEC, **patch}


def _read_back(path) -> float:
    """The written LP file, solved by Gurobi's own parser — the second encoding."""
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        found = gurobipy.read(str(path), env=env)
        try:
            found.optimize()
            assert found.Status == gurobipy.GRB.OPTIMAL, f'the written file solved to status {found.Status}'
            assert found.NumQConstrs, 'the file carried no quadratic constraint at all'
            return float(found.ObjVal)
        finally:
            found.dispose()


# ---------------------------------------------------------------------------
# the replacement oracle
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('p * q >= floor', id='a-bare-product'),
        pytest.param('p * q + p >= floor', id='a-product-beside-a-linear-term'),
        pytest.param('p * q >= floor - p', id='terms-on-both-sides'),
        pytest.param('2 * p * q >= floor', id='a-coefficient-on-the-product'),
        pytest.param('p * p + q * q <= 40', id='a-convex-region'),
    ],
)
def test_the_two_encodings_reach_one_optimum(expression, tmp_path):
    """One path goes through ``addMQConstr`` and numpy, the other through the
    writer's text and Gurobi's parser: a coefficient doubled, a pair
    transposed or a linear half dropped shows up as two different numbers."""
    varied = spec(constraints={**SPEC['constraints'], 'coupled': {'dims': ['g'], 'expression': expression}})
    with sps.solve(varied, SOURCES, solver_name='gurobi') as direct:
        assert direct.is_ok
        path = tmp_path / 'model.lp'
        sps.write(varied, SOURCES, path)
        assert _read_back(path) == pytest.approx(direct.objective, rel=RTOL), (
            'the written file and the direct hand-off are the same model by two routes; a '
            'disagreement is an encoding error in one of them'
        )


def test_the_activity_is_the_whole_left_hand_side():
    """`xᵀQx + Ax`, recomputed — the residual half of the oracle.

    Not ``Ax``: for ``p·q >= 4`` it is 4 at the optimum where
    ``Ax`` is 0, the row owning no linear entry at all.
    """
    with sps.build(SPEC, SOURCES) as model:
        result = model.solve(solver_name='gurobi', outputs={'activity'})
        recomputed = recomputed_row_values(model._engine, result)
        block = model._engine._model.constraints['coupled']
        reported = result.activity('coupled')['value'].to_numpy()
        assert reported == pytest.approx(recomputed[block.start : block.start + block.height], rel=RTOL)
        assert reported == pytest.approx([4.0, 4.0], rel=RTOL), 'the row binds, and its value is the product'


def test_the_optimum_is_the_one_done_by_hand():
    """The check neither weak oracle can make.

    Minimise ``p + q`` subject to ``p·q >= 4``: the product is fixed, so the
    sum is least where the factors are equal. Two encodings agreeing on a wrong
    number is what a shared misreading looks like, and arithmetic is what is
    left to catch it.
    """
    with sps.solve(SPEC, SOURCES, solver_name='gurobi') as result:
        assert result.objective == pytest.approx(8.0, rel=RTOL)
        # looser than the objective's: the search closes a gap, not a point
        assert result.primal('p')['value'].to_list() == pytest.approx([2.0, 2.0], rel=1e-3)
        assert result.primal('q')['value'].to_list() == pytest.approx([2.0, 2.0], rel=1e-3)


# ---------------------------------------------------------------------------
# where the rows land, and what that buys
# ---------------------------------------------------------------------------


def test_quadratic_declarations_take_the_tail_of_the_label_space():
    """Quadratic is a property of a *declaration*, so its rows are contiguous —
    which is what keeps every read-back a slice. Declared first in the file
    here, deliberately: the ordering is the engine's, not the author's."""
    first = spec(
        constraints={
            'coupled': {'dims': ['g'], 'expression': 'p * q >= floor'},
            'cap': {'dims': [], 'expression': 'sum(p, over=g) <= 9'},
        }
    )
    with sps.build(first, SOURCES) as model:
        tables = model._engine._model.handoff
        assert model._engine._model.constraints['cap'].start == 0, 'the linear declaration is built first'
        assert [row for row, _ in tables.quadratic_blocks()] == [1, 2], (
            'the quadratic rows are the tail, however the file was written'
        )
        assert tables.linear_row_count == 1


def test_a_quadratic_row_has_no_price_unless_the_caller_asks():
    """The default is the *answer*, not the extra information.

    ``QCPDual`` puts the solve on the convex path, so a nonconvex row that
    solves without it fails outright with it (*Constraint Q not PSD*). A caller
    whose model is convex asks.
    """
    with sps.solve(SPEC, SOURCES, solver_name='gurobi') as silent:
        with pytest.raises(SpecsolveError, match='QCPDual'):
            silent.dual('coupled')
        assert silent.is_ok, 'and the answer itself is unaffected'

    with sps.solve(SPEC, SOURCES, solver_name='gurobi', solver_options={'QCPDual': 1}) as priced:
        assert priced.dual('coupled')['value'].to_list() == pytest.approx([0.5, 0.5], rel=RTOL), (
            'the price of relaxing p·q >= 4 at p = q = 2'
        )


def test_asking_for_prices_on_a_nonconvex_row_says_which_option_did_it():
    """The one error a caller's own option can provoke, translated — left
    alone it arrives as a ``GurobiError`` naming a parameter they set for an
    unrelated reason. Convexity is data, so nothing could refuse it earlier."""
    nonconvex = spec(
        constraints={'coupled': {'dims': ['g'], 'expression': 'p * p + q * q >= floor'}},
    )
    assert sps.solve(nonconvex, SOURCES, solver_name='gurobi').objective == pytest.approx(4.0, rel=RTOL), (
        'the nonconvex region solves by default — spatial branch-and-bound needs no parameter'
    )
    with pytest.raises(SpecsolveError, match=r'not convex.*QCPDual'):
        sps.solve(nonconvex, SOURCES, solver_name='gurobi', solver_options={'QCPDual': 1})


def _entries(expression: str, sources=None) -> pl.DataFrame:
    """The quadratic stream of a model whose row is *expression*."""
    varied = spec(constraints={'coupled': {'dims': ['g'], 'expression': expression}})
    with sps.build(varied, dict(sources or SOURCES)) as model:
        return model._engine._model.handoff.qmatrix


def test_a_pair_in_a_row_is_stored_once_whichever_order_it_was_written():
    """``q · p`` and ``p · q`` are one entry of one row.

    Sharper here than in the objective: a pair written the other way round
    lands in ``Q``'s *lower* triangle, still solves — ``xᵀQx`` reads both — and
    stops matching the LP file, where the two orders are two different lines.
    """
    written = _entries('p * q >= floor')
    reversed_ = _entries('q * p >= floor')
    assert written.equals(reversed_), 'the order the factors were written in is not part of the model'
    assert (written['col_l'] <= written['col_r']).all(), 'a pair is ordered by column index'


def test_a_quadratic_row_is_structure_whole_and_a_update_reloads():
    """The digest rule, and the *opposite* of the objective's.

    An objective's quadratic part is replaced by one call, so its coefficients
    are pushed and only the pattern is structure. A constraint's has no such
    call, so pushing half would leave a stale coefficient answering a model
    nobody wrote: it is structure whole.
    """
    weighted = spec(
        parameters={'floor': {'dims': []}, 'weight': {'dims': ['g']}},
        constraints={'coupled': {'dims': ['g'], 'expression': 'p * q * weight >= floor'}},
    )
    both = {**SOURCES, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 1.0]})}
    heavier = {**SOURCES, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [2.0, 2.0]})}
    dropped = {**SOURCES, 'weight': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 0.0]})}

    for moved in (heavier, dropped, {**both, 'floor': pl.DataFrame({'value': [9.0]})}):
        with sps.build(weighted, both) as model:
            first = model.solve(solver_name='gurobi').objective
            model.update(moved)
            again = model.solve(solver_name='gurobi').objective
            assert model.diagnostics().loads == 2, 'anything about a quadratic row is a model to load again'
            with sps.solve(weighted, moved, solver_name='gurobi') as fresh:
                assert again == pytest.approx(fresh.objective, rel=SEARCHED), (
                    'an update answers what a fresh build answers — a coefficient or a right-hand '
                    'side left behind would answer the model before it'
                )
            assert again != pytest.approx(first, rel=SEARCHED), 'and the answer really did move'


def test_a_linear_right_hand_side_beside_a_quadratic_row_is_pushed():
    """The digest rule's other half: the linear rows of a model that has a
    quadratic one take a new right-hand side by value, and Gurobi is handed
    one per linear row — it refuses a vector that also spans the quadratic tail."""
    capped = spec(
        parameters={'floor': {'dims': []}, 'limit': {'dims': []}},
        constraints={**SPEC['constraints'], 'cap': {'dims': [], 'expression': 'sum(p, over=g) <= limit'}},
        objective={'sense': 'maximize', 'expression': 'sum(p + q, over=g)'},
    )
    loose = {**SOURCES, 'limit': pl.DataFrame({'value': [9.0]})}
    tight = {**SOURCES, 'limit': pl.DataFrame({'value': [7.0]})}
    with sps.build(capped, loose) as model:
        model.solve(solver_name='gurobi').close()
        model.update(tight)
        pushed = model.solve(solver_name='gurobi')
        assert model.diagnostics().loads == 1, 'only a linear right-hand side moved, so it is pushed'
        with sps.solve(capped, tight, solver_name='gurobi') as fresh:
            assert pushed.objective == pytest.approx(fresh.objective, rel=SEARCHED), (
                'a pushed right-hand side answers what a fresh build answers'
            )
        pushed.close()


def test_the_pair_a_row_holds_is_structure_even_at_the_same_coefficient():
    """A purpose-built probe: no data change can reach this one.

    The digest reads the pattern *and* the coefficients, and every update a
    model can express moves both — labels are dense, so a mask that changes
    which pair a row holds also changes how many there are. Asked of the tables
    directly instead: one entry at a different column, same coefficient.
    """
    with sps.build(SPEC, SOURCES) as model:
        tables = model._engine._model.handoff
        assert tables.qmatrix.height, 'the model under test carries a quadratic row'
        moved = replace(tables, qmatrix=tables.qmatrix.with_columns(pl.col('col_r') + 1))
        assert moved.structure != tables.structure, (
            'the same coefficient on a different pair is a different row, and a digest that '
            'missed it would push new numbers onto a solver holding the old constraint'
        )


# ---------------------------------------------------------------------------
# what cannot build one, and how it says so
# ---------------------------------------------------------------------------


def test_the_linopy_lane_refuses_it_in_the_languages_own_words(tmp_path):
    """Hard rule 3's amendment where it bites. The oracle still *accepts* the
    model — one ``inputs.lowered`` gate — and refuses it before linopy is asked,
    where linopy's ``NotImplementedError`` names nothing."""
    import yaml as pyyaml

    from tests.linopy_lane.loader import OracleCannotBuildError
    from tests.oracle import specsolve_linopy

    path = tmp_path / 'spec.yaml'
    path.write_text(pyyaml.safe_dump(SPEC))
    with pytest.raises(OracleCannotBuildError, match='linopy lane cannot build'):
        specsolve_linopy.build(path, SOURCES)


def test_highs_refuses_it_before_the_load_and_names_who_takes_it():
    with pytest.raises(SpecsolveError, match='no such concept') as refused:
        sps.solve(SPEC, SOURCES)
    assert 'gurobi' in str(refused.value), 'the refusal names the sink that takes a quadratic row'


def test_the_highs_hand_off_refuses_one_even_when_reached_directly():
    """The backstop for the hand-off `bench/` uses: constructing `Highs` is past the
    capability check, and the linear rows of a quadratic model load perfectly
    well — as a different model, answering a number nothing would question."""
    from specsolve.relational.sinks.solvers.highs import Highs

    with sps.build(SPEC, SOURCES) as model, pytest.raises(SpecsolveError, match='no quadratic-constraint concept'):
        Highs(model._engine._model.handoff)


def test_a_bare_check_stays_silent_about_all_of_it():
    """Whether a model is sayable is solver-independent, and this one is."""
    sps.check(SPEC)
