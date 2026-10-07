"""``start=``: an LP solved from an earlier answer's basis, a mixed-integer model from values, both matched by coordinate.

Warmth is read off each solver's own simplex iteration counter, which is
deterministic, so none of this needs an idle box. A mixed-integer start is read
off the incumbent a solve stopped at its first solution returns. The answer is
the oracle for correctness: a start moves the route, never the optimum.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.answer_layout import NO_BASIS
from specsolve.relational.engine.readback import _counted
from specsolve.relational.sinks.handoff import SENSE_CODES
from specsolve.relational.sinks.solvers.base import (
    AT_LOWER,
    AT_UPPER,
    BASIC,
    BASIS_STATUSES,
    FIXED,
    SUPERBASIC,
    settled,
)
from tests.conftest import ITEMS, KNAPSACK, knapsack_sources
from tests.test_warm_start import DISPATCH, DISPATCH_CAPPED, GENERATORS, SIMPLEX_ITERATIONS, dispatch_sources

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output, Result

BASIS: frozenset[Output] = frozenset({'variable_basis', 'constraint_basis'})

#: One set of numbers over 44 snapshots, so a build over fewer is a subset of it.
FULL = dispatch_sources(list(range(44))) | {'snapshot': list(range(44))}


def snapshots(n: int) -> dict[str, Any]:
    """The first *n* snapshots of [`FULL`][], the same numbers at each."""
    kept = {name: FULL[name].filter(pl.col('snapshot') < n) for name in ('cost', 'load')}
    return {**FULL, **kept, 'snapshot': list(range(n))}


def solved(spec: dict, given: dict, solver_name: str, **solve: Any) -> tuple[Result, int]:
    """One solve of *spec* on a fresh model, and the simplex iterations it took."""
    with sps.build(spec, given) as model:
        answer = model.solve(solver_name=solver_name, **solve)
        return answer, SIMPLEX_ITERATIONS[solver_name](model._engine._solver)


def test_an_unchanged_model_started_from_its_own_answer_does_no_simplex_work(solver_name: str) -> None:
    before, cold = solved(DISPATCH, snapshots(40), solver_name, outputs=BASIS)
    after, warm = solved(DISPATCH, snapshots(40), solver_name, start=before)
    assert cold > 0, 'the model must make the simplex work, or warmth would be unobservable'
    assert warm == 0, 'the basis an optimum ended on starts the same model at that optimum'
    assert after.objective == pytest.approx(before.objective), 'a start moves the route, never the optimum'


@pytest.mark.parametrize(
    'now',
    [pytest.param(44, id='snapshots-gained'), pytest.param(36, id='snapshots-lost')],
)
def test_a_model_that_gained_or_lost_coordinates_starts_from_the_ones_it_shares(solver_name: str, now: int) -> None:
    """Each coordinate both builds hold keeps its status; a column gained starts at a bound, a row gained basic."""
    before, _ = solved(DISPATCH, snapshots(40), solver_name, outputs=BASIS)
    cold_answer, cold = solved(DISPATCH, snapshots(now), solver_name)
    warm_answer, warm = solved(DISPATCH, snapshots(now), solver_name, start=before)
    assert warm < cold, 'the shared coordinates carry the work done on them'
    assert warm_answer.objective == pytest.approx(cold_answer.objective), 'a start moves the route, never the optimum'


def test_a_constraint_gained_enters_without_moving_the_optimum(solver_name: str) -> None:
    """A cutting-plane master re-solved after gaining a cut: the case #382 left open."""
    before, _ = solved(DISPATCH, snapshots(40), solver_name, outputs=BASIS)
    capped = snapshots(40) | {'cap': pl.DataFrame({'generator': GENERATORS, 'value': [1500.0] * len(GENERATORS)})}
    cold_answer, cold = solved(DISPATCH_CAPPED, capped, solver_name)
    warm_answer, warm = solved(DISPATCH_CAPPED, capped, solver_name, start=before)
    assert warm < cold, 'the rows the cut leaves alone carry their statuses'
    assert warm_answer.objective == pytest.approx(cold_answer.objective), 'the cut binds the same way however it starts'


def test_one_solvers_basis_starts_another(solver_name: str) -> None:
    """The statuses cross in one vocabulary, so the solver that read them need not be the one that takes them."""
    before, _ = solved(DISPATCH, snapshots(40), 'highs', outputs=BASIS)
    _, warm = solved(DISPATCH, snapshots(40), solver_name, start=before)
    assert warm == 0, "HiGHS's optimal basis is every solver's"


@pytest.mark.parametrize('kept', ['saved', 'archived'])
def test_an_answer_off_disk_starts_a_solve(tmp_path: Path, kept: str) -> None:
    if kept == 'saved':
        with sps.solve(DISPATCH, snapshots(40), outputs=BASIS) as answer:
            answer.save(tmp_path / 'answer')
        before = sps.load_result(tmp_path / 'answer')
    else:
        sps.solve(DISPATCH, snapshots(40), outputs=BASIS, archive=tmp_path / 'run.zip')
        archive = sps.load_archive(tmp_path / 'run.zip')
        assert isinstance(archive, sps.archive.ResultArchive)
        before = archive.result
    _, warm = solved(DISPATCH, snapshots(40), 'highs', start=before)
    assert warm == 0, 'the frames on disk are the frames the live answer held'


def test_a_declaration_whose_dims_changed_starts_as_new() -> None:
    """Its coordinates are not the earlier ones, so none of its statuses carry, and the rest still do."""
    scalar = {
        'variables': {'x': {'dims': [], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'cap': {'dims': [], 'expression': 'x <= 3'}},
        'objective': {'sense': 'maximize', 'expression': 'x'},
    }
    indexed = {
        'dimensions': {'i': {}},
        'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'cap': {'dims': [], 'expression': 'sum(x, over=i) <= 3'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=i)'},
    }
    before = sps.solve(scalar, {}, outputs=BASIS)
    with sps.solve(indexed, {'i': ['a', 'b']}, start=before) as after:
        assert after.objective == pytest.approx(3.0), 'the cap binds however the sum is split'


# ---------------------------------------------------------------------------
# refusals
# ---------------------------------------------------------------------------


def test_an_answer_solved_without_its_basis_is_refused_before_the_solver_loads(monkeypatch: pytest.MonkeyPatch) -> None:
    from specsolve.relational import sinks

    with sps.solve(DISPATCH, snapshots(40), outputs={'variable_basis'}) as before:
        monkeypatch.setattr(sinks, 'loaded', lambda *_: pytest.fail('the solver loaded before the refusal'))
        with pytest.raises(SpecsolveError, match=r"without 'constraint_basis'.*outputs=\{'variable_basis', "):
            sps.solve(DISPATCH, snapshots(40), start=before)


def test_an_answer_that_ended_on_no_basis_says_why() -> None:
    before = sps.solve(KNAPSACK, knapsack_sources(), outputs=BASIS)
    with pytest.raises(SpecsolveError, match=NO_BASIS[:40]):
        sps.solve(DISPATCH, snapshots(40), start=before)


def test_an_lp_takes_no_table_of_values() -> None:
    values = {'p': sps.solve(DISPATCH, snapshots(40)).primal('p')}
    with pytest.raises(SpecsolveError, match='starts from a basis'):
        sps.solve(DISPATCH, snapshots(40), start=values)


def test_a_closed_answer_is_refused() -> None:
    before = sps.solve(DISPATCH, snapshots(40), outputs=BASIS)
    before.close()
    with pytest.raises(SpecsolveError, match='closed'):
        sps.solve(DISPATCH, snapshots(40), start=before)


# ---------------------------------------------------------------------------
# a mixed-integer model starts from values
# ---------------------------------------------------------------------------

#: Each sink stopped at the first solution it holds, with nothing that would
#: find a better one before it: what it returns is the start it was given.
FIRST_SOLUTION = {
    'highs': {'mip_max_improving_sols': 1, 'presolve': 'off'},
    'gurobi': {'SolutionLimit': 1, 'Presolve': 0, 'Heuristics': 0},
    'xpress': {'maxmipsol': 1, 'presolve': 0},
}

#: Two items that fit together, worth 2 + 3, far short of the optimum of 37.
TWO_ITEMS = pl.DataFrame({'item': ['item1', 'item2'], 'value': [1.0, 1.0]})


def _first(solver_name: str, **solve: Any) -> float:
    return sps.solve(KNAPSACK, knapsack_sources(), solver_name=solver_name, solver_options=FIRST_SOLUTION[solver_name], **solve).objective


def test_a_mixed_integer_solve_stops_at_the_values_it_starts_from(solver_name: str) -> None:
    every = pl.DataFrame({'item': ITEMS, 'value': [1.0 if item in ('item1', 'item2') else 0.0 for item in ITEMS]})
    assert _first(solver_name, start={'take': every}) == pytest.approx(16.0), 'the start is the first solution'


def test_a_partial_start_is_completed_by_the_solver(solver_name: str) -> None:
    assert _first(solver_name, start={'take': TWO_ITEMS}) == pytest.approx(16.0), 'the items not named stay out'


def test_an_earlier_answer_starts_a_mixed_integer_solve_at_its_optimum(solver_name: str) -> None:
    before = sps.solve(KNAPSACK, knapsack_sources(), solver_name=solver_name)
    assert _first(solver_name, start=before) == pytest.approx(before.objective), 'its primal is the first solution'


@pytest.mark.parametrize(
    ('start', 'match'),
    [
        pytest.param({'tkae': TWO_ITEMS}, "unknown variable 'tkae'.*take", id='a-misspelled-variable'),
        pytest.param({'take': TWO_ITEMS.rename({'item': 'items'})}, r"\['item', 'value'\]", id='a-column-not-its-dims'),
        pytest.param({'take': TWO_ITEMS.with_columns(item=pl.lit('nothing'))}, 'no value at any', id='no-coordinate-held'),
    ],
)
def test_a_table_that_cannot_start_the_model_is_refused(start: dict, match: str) -> None:
    with pytest.raises(SpecsolveError, match=match):
        sps.solve(KNAPSACK, knapsack_sources(), start=start)


# ---------------------------------------------------------------------------
# the two steps that make a carried basis one a solver takes
# ---------------------------------------------------------------------------

INF = np.inf


def _handoff(lb: list[float], ub: list[float], senses: list[str]) -> Any:
    """The two projections [`settled`][] reads, and nothing else."""
    return SimpleNamespace(
        dense_columns=lambda _: SimpleNamespace(lb=np.asarray(lb), ub=np.asarray(ub)),
        dense_rows=lambda _: SimpleNamespace(sense=np.asarray([SENSE_CODES[s] for s in senses], dtype=np.int64)),
    )


@pytest.mark.parametrize(
    ('status', 'lb', 'ub', 'expected'),
    [
        pytest.param(AT_LOWER, 0.0, 5.0, 'at_lower', id='a-lower-bound-it-has'),
        pytest.param(AT_UPPER, 0.0, INF, 'at_lower', id='an-upper-bound-it-lost'),
        pytest.param(AT_LOWER, -INF, 5.0, 'at_upper', id='a-lower-bound-it-lost'),
        pytest.param(AT_LOWER, -INF, INF, 'superbasic', id='free-now'),
        pytest.param(AT_UPPER, 3.0, 3.0, 'fixed', id='bounds-now-equal'),
        pytest.param(FIXED, 0.0, 5.0, 'at_lower', id='bounds-no-longer-equal'),
        pytest.param(BASIC, -INF, INF, 'basic', id='basic-stays'),
        pytest.param(SUPERBASIC, 0.0, 5.0, 'superbasic', id='superbasic-stays'),
    ],
)
def test_a_nonbasic_column_is_put_at_a_bound_the_model_has(status: int, lb: float, ub: float, expected: str) -> None:
    columns = settled(_handoff([lb], [ub], []), np.asarray([status]), np.asarray([], dtype=np.int8)).columns
    assert BASIS_STATUSES[columns[0]] == expected


def test_a_nonbasic_row_is_at_the_bound_its_sense_gives() -> None:
    nonbasic = np.asarray([AT_LOWER, AT_LOWER, AT_UPPER, BASIC])
    rows = settled(_handoff([], [], ['<=', '>=', '==', '<=']), np.asarray([], dtype=np.int8), nonbasic).rows
    assert [BASIS_STATUSES[code] for code in rows] == ['at_upper', 'at_lower', 'fixed', 'basic'], (
        'a <= row binds at its upper side, a >= row at its lower, an == row at both, and a basic row stays'
    )


@pytest.mark.parametrize(
    ('columns', 'rows', 'expected'),
    [
        pytest.param([BASIC, AT_LOWER], [AT_UPPER, BASIC], ([BASIC, AT_LOWER], [AT_UPPER, BASIC]), id='one-per-row'),
        pytest.param([BASIC, BASIC], [AT_UPPER], ([BASIC, AT_LOWER], [AT_UPPER]), id='a-nonbasic-row-dropped'),
        pytest.param([AT_LOWER], [AT_UPPER, BASIC], ([AT_LOWER], [BASIC, BASIC]), id='a-basic-column-dropped'),
    ],
)
def test_a_carried_basis_has_one_basic_entry_per_row(columns: list[int], rows: list[int], expected: tuple) -> None:
    counted = _counted(np.asarray(columns, dtype=np.int8), np.asarray(rows, dtype=np.int8))
    assert ([*counted[0]], [*counted[1]]) == expected, 'a surplus leaves from the columns, a shortfall enters as rows'
