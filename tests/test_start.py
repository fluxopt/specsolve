"""``start=``: an LP solved from an earlier answer's basis, a mixed-integer model from values, both matched by coordinate.

Warmth is read off each solver's own simplex iteration counter, which is
deterministic, so none of this needs an idle box. A mixed-integer start is read
off the incumbent a solve stopped at its first solution returns. The answer is
the oracle for correctness: a start moves the route, never the optimum.
"""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError, SpecsolveWarning
from specsolve.relational.answer_layout import AT_LOWER, AT_UPPER, BASIC, BASIS_STATUSES, FIXED, SUPERBASIC
from specsolve.relational.engine.readback import _counted
from specsolve.relational.sinks.handoff import SENSE_CODES
from specsolve.relational.sinks.solvers.base import settled
from tests.conftest import ITEMS, KNAPSACK, knapsack_sources
from tests.test_warm_start import DISPATCH, DISPATCH_CAPPED, GENERATORS, SIMPLEX_ITERATIONS, dispatch_sources

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output, Result

BASIS: frozenset[Output] = frozenset({'basis'})

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


def _as_tables(answer: Result, reader: str) -> dict[str, pl.DataFrame]:
    """Every table *reader* returns off *answer*, a status as plain text: a start as one written elsewhere would be."""
    names = DISPATCH['constraints'] if reader == 'constraint_basis' else DISPATCH['variables']
    return {name: getattr(answer, reader)(name).with_columns(pl.col('value').cast(pl.String)) for name in names}


def test_a_basis_given_as_tables_starts_an_lp_as_an_answer_does(solver_name: str) -> None:
    before, _ = solved(DISPATCH, snapshots(40), solver_name, outputs=BASIS)
    start = {reader: _as_tables(before, reader) for reader in ('variable_basis', 'constraint_basis')}
    _, warm = solved(DISPATCH, snapshots(40), solver_name, start=start)
    assert warm == 0, 'the tables carry the basis the optimum ended on'


def test_half_a_basis_is_completed_and_reaches_the_optimum() -> None:
    """The rows left out start basic and the count is made right, which a solver takes, though it need not pay.

    On this model it does not: the columns alone took more iterations than a
    cold start, so the claim here is only that the start is taken.
    """
    before, _ = solved(DISPATCH, snapshots(40), 'highs', outputs=BASIS)
    after, _ = solved(DISPATCH, snapshots(40), 'highs', start={'variable_basis': _as_tables(before, 'variable_basis')})
    assert after.objective == pytest.approx(before.objective), 'a start moves the route, never the optimum'


def test_an_lp_answer_without_its_basis_starts_from_its_values() -> None:
    before, cold = solved(DISPATCH, snapshots(40), 'highs')
    _, warm = solved(DISPATCH, snapshots(40), 'highs', start=before)
    assert warm < cold, 'HiGHS uses a primal that gives every column a value'


#: What each sink does with a start of values for an LP, complete and partial.
#: This is the claim each sink's ``lp_values`` makes, checked against the solve.
LP_VALUES = {
    'highs': {'complete': 'used', 'partial': 'no_gain'},
    'gurobi': {'complete': 'no_gain', 'partial': 'no_gain'},
    'xpress': {'complete': 'no_gain', 'partial': 'refused'},
}


@pytest.mark.parametrize('given', ['complete', 'partial'])
def test_a_start_of_values_for_an_lp_is_used_warned_of_or_refused_as_its_sink_says(
    solver_name: str, given: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specsolve.relational import sinks

    before = sps.solve(DISPATCH, snapshots(40), solver_name=solver_name)
    primal = before.primal('p')
    start = {'primal': {'p': primal if given == 'complete' else primal.head(primal.height // 2)}}
    expected = LP_VALUES[solver_name][given]
    assert sinks.solver(solver_name).lp_values[given] == expected, 'the sink declares what this test observes'
    if expected == 'refused':
        monkeypatch.setattr(sinks, 'loaded', lambda *_: pytest.fail('the solver loaded before the refusal'))
        with pytest.raises(SpecsolveError, match='leave a column out'):
            sps.solve(DISPATCH, snapshots(40), solver_name=solver_name, start=start)
        return
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        after = sps.solve(DISPATCH, snapshots(40), solver_name=solver_name, start=start)
    warned = [w for w in caught if issubclass(w.category, SpecsolveWarning) and 'no gain' in str(w.message)]
    assert bool(warned) == (expected == 'no_gain'), 'the solve warns exactly where no gain is known'
    assert after.objective == pytest.approx(before.objective), 'the sink took the values and still solved'


def test_a_start_from_another_model_lands_nowhere_and_is_refused() -> None:
    before = sps.solve(KNAPSACK, knapsack_sources(), outputs=BASIS)
    with pytest.raises(SpecsolveError, match='no value at any coordinate'):
        sps.solve(DISPATCH, snapshots(40), start=before)


def test_a_closed_answer_is_refused() -> None:
    before = sps.solve(DISPATCH, snapshots(40), outputs=BASIS)
    before.close()
    with pytest.raises(SpecsolveError, match='closed'):
        sps.solve(DISPATCH, snapshots(40), start=before)


# ---------------------------------------------------------------------------
# a mixed-integer model starts from values
# ---------------------------------------------------------------------------

#: Each sink stopped at its root node with nothing that finds a solution of its
#: own, so that any incumbent it returns is one it was started from.
ROOT_ONLY = {
    'highs': {'mip_max_nodes': 0, 'mip_heuristic_effort': 0.0, 'presolve': 'off'},
    'gurobi': {'NodeLimit': 0, 'Heuristics': 0, 'Presolve': 0, 'Cuts': 0},
    'xpress': {'maxnode': 0, 'heuremphasis': 0, 'presolve': 0, 'cutstrategy': 0},
}

#: Two items that fit together, worth 8 + 2, far short of the optimum of 56.
TWO_ITEMS = pl.DataFrame({'item': ['item1', 'item2'], 'value': [1.0, 1.0]})

#: The knapsack's one row, not binding.
FITS = pl.DataFrame({'value': ['basic']})


def _at_the_root(solver_name: str, **solve: Any) -> Result:
    return sps.solve(
        KNAPSACK, knapsack_sources(), solver_name=solver_name, solver_options=ROOT_ONLY[solver_name], **solve
    )


def test_a_mixed_integer_solve_holds_no_incumbent_of_its_own_at_the_root(solver_name: str) -> None:
    """The control for the three below: without a start, the root leaves nothing."""
    assert not _at_the_root(solver_name).has_primal, 'with heuristics off, no incumbent is found at the root'


#: The two items worth 8 + 2 in, every other out: the knapsack's start as each
#: shape a parameter's source takes.
TWO_IN = {item: 1.0 if item in ('item1', 'item2') else 0.0 for item in ITEMS}


def _shaped(shape: str, tmp_path: Path) -> object:
    table = pl.DataFrame({'item': list(TWO_IN), 'value': list(TWO_IN.values())})
    if shape == 'parquet':
        table.write_parquet(tmp_path / 'take.parquet')
        return str(tmp_path / 'take.parquet')
    if shape == 'pandas':
        pytest.importorskip('pandas')
        pytest.importorskip('pyarrow')
        return table.to_pandas()
    return {'polars': table, 'mapping': TWO_IN, 'sequence': list(TWO_IN.values())}[shape]


@pytest.mark.parametrize('shape', ['polars', 'pandas', 'parquet', 'mapping', 'sequence'])
def test_a_mixed_integer_solve_returns_the_values_it_starts_from(solver_name: str, shape: str, tmp_path: Path) -> None:
    """A start takes every shape a parameter's source takes, read by the same reader."""
    assert _at_the_root(solver_name, start={'primal': {'take': _shaped(shape, tmp_path)}}).objective == pytest.approx(
        10.0
    ), 'the start, worth 8 + 2, is the incumbent'


def test_one_number_starts_every_coordinate(solver_name: str) -> None:
    assert _at_the_root(solver_name, start={'primal': {'take': 0.0}}).objective == pytest.approx(0.0), (
        'every item out is the incumbent'
    )


def test_a_partial_start_is_completed_by_the_solver(solver_name: str) -> None:
    answer = _at_the_root(solver_name, start={'primal': {'take': TWO_ITEMS}})
    taken = answer.primal('take').filter(pl.col('item').is_in(['item1', 'item2']))['value']
    assert taken.to_list() == pytest.approx([1.0, 1.0]), 'the items the start names stay in'
    assert answer.objective >= 10.0, 'the solver fills in the items the start leaves out'


def test_an_earlier_answer_starts_a_mixed_integer_solve_at_its_optimum(solver_name: str) -> None:
    before = sps.solve(KNAPSACK, knapsack_sources(), solver_name=solver_name)
    assert _at_the_root(solver_name, start=before).objective == pytest.approx(before.objective), (
        'its primal is the incumbent'
    )


@pytest.mark.parametrize(
    ('start', 'match'),
    [
        pytest.param(
            {'take': TWO_ITEMS},
            r"'primal', 'variable_basis', 'constraint_basis', and not 'take'",
            id='a-key-that-is-no-reader',
        ),
        pytest.param({'primal': {'tkae': TWO_ITEMS}}, "unknown variable 'tkae'.*take", id='a-misspelled-variable'),
        pytest.param(
            {'constraint_basis': {'fist': FITS}},
            "under 'constraint_basis' an unknown constraint 'fist'",
            id='a-misspelled-constraint',
        ),
        pytest.param(
            {'primal': {'take': TWO_ITEMS.rename({'item': 'items'})}},
            r"missing columns \['item'\]",
            id='a-column-not-its-dims',
        ),
        pytest.param(
            {'primal': {'take': TWO_ITEMS}, 'constraint_basis': {'fits': FITS.with_columns(value=pl.lit('loose'))}},
            "the basis status 'loose'",
            id='a-status-that-is-no-word',
        ),
        pytest.param(
            {'constraint_basis': {'fits': FITS}}, 'a basis alone', id='a-basis-alone-for-a-mixed-integer-model'
        ),
        pytest.param(
            {'primal': {'take': TWO_ITEMS.with_columns(item=pl.lit('itme1'))}},
            "'itme1'",
            id='a-label-the-dimension-lacks',
        ),
        pytest.param(
            {'primal': {'take': pl.concat([TWO_ITEMS, TWO_ITEMS])}}, 'more than one row', id='a-coordinate-twice'
        ),
        pytest.param({'primal': {'take': [1.0, 0.0]}}, '2 values against 12', id='a-sequence-of-the-wrong-length'),
    ],
)
def test_a_start_that_cannot_start_the_model_is_refused(start: dict, match: str) -> None:
    with pytest.raises(SpecsolveError, match=match):
        sps.solve(KNAPSACK, knapsack_sources(), start=start)


def test_an_answer_of_another_model_places_nothing_and_is_refused() -> None:
    """Its names match no variable here, so it would start nothing; that is a mistake, not a hint."""
    before = sps.solve(DISPATCH, snapshots(40))
    with pytest.raises(SpecsolveError, match='no value at any coordinate'):
        sps.solve(KNAPSACK, knapsack_sources(), start=before)


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


# ---------------------------------------------------------------------------
# start= beside keep=, and a sweep's start
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('keep', ['progress', 'nothing'])
@pytest.mark.parametrize('verb', ['model-solve', 'solve_over'])
def test_a_start_beside_a_keep_that_also_says_where_to_begin_is_refused(
    keep: str, verb: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from specsolve.relational import sinks

    before = sps.solve(KNAPSACK, knapsack_sources())
    monkeypatch.setattr(sinks, 'loaded', lambda *_: pytest.fail('the solver loaded before the refusal'))
    with pytest.raises(SpecsolveError, match=rf"start= and keep='{keep}' both say"):
        if verb == 'model-solve':
            sps.build(KNAPSACK, knapsack_sources()).solve(keep=keep, start=before)
        else:
            sps.solve_over(KNAPSACK, knapsack_sources(), DRAWS, key_name='draw', keep=keep, start=before)


#: Two draws of the knapsack, each a whole model.
DRAWS = [('a', knapsack_sources()), ('b', knapsack_sources())]


def _swept_at_the_root(solver_name: str, start: Any, executor: Any = None) -> sps.types.Sweep:
    return sps.solve_over(
        KNAPSACK,
        {},
        DRAWS,
        key_name='draw',
        solver_name=solver_name,
        solver_options=ROOT_ONLY[solver_name],
        start=start,
        executor=executor,
    )


EXECUTORS = [
    pytest.param(None, id='serial'),
    pytest.param('threads', id='threads'),
    pytest.param('processes', id='processes'),
]


@pytest.mark.parametrize('how', EXECUTORS)
def test_each_slice_starts_from_its_slice_of_an_earlier_sweep(how: str | None) -> None:
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    earlier = sps.solve_over(KNAPSACK, {}, DRAWS, key_name='draw')
    if how is None:
        sweep = _swept_at_the_root('highs', earlier)
    else:
        pool = (
            ThreadPoolExecutor(2)
            if how == 'threads'
            else ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'))
        )
        with pool:
            sweep = _swept_at_the_root('highs', earlier, pool)
    assert sweep.record['objective'].to_list() == pytest.approx(earlier.record['objective'].to_list()), (
        'each slice, stopped at its root, returns its slice of the earlier sweep as its incumbent'
    )


def test_one_answer_starts_every_slice() -> None:
    before = sps.solve(KNAPSACK, knapsack_sources())
    sweep = _swept_at_the_root('highs', before)
    assert sweep.record['objective'].to_list() == pytest.approx([before.objective] * 2), (
        'every slice, stopped at its root, returns the one answer as its incumbent'
    )


def test_each_window_starts_from_the_same_window_of_an_earlier_sweep(monkeypatch: pytest.MonkeyPatch) -> None:
    from specsolve.relational.sinks.solvers.highs import Highs
    from tests.test_strategy import WINDOW, WINDOW_AXIS, horizon_sources

    earlier = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, outputs=BASIS)
    warmed: list[object] = []
    original = Highs.warm
    monkeypatch.setattr(Highs, 'warm', lambda self, basis: (warmed.append(basis), original(self, basis))[1])
    again = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, start=earlier)
    assert len(warmed) == len(earlier.keys), 'every window is warmed from a basis'
    assert again.record['objective'].to_list() == pytest.approx(earlier.record['objective'].to_list()), (
        'a start moves the route, never the optimum'
    )


def test_a_table_over_the_windows_local_index_alone_is_refused() -> None:
    from tests.test_strategy import WINDOW, WINDOW_AXIS, horizon_sources

    earlier = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS)
    first = earlier.primal('soc', per_window=True).filter(pl.col('snapshot_start') == 0).drop('snapshot_start')
    with pytest.raises(SpecsolveError, match="over the windows' local index 't' and not over 'snapshot'"):
        sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, start={'primal': {'soc': first}})


def test_a_table_over_the_sliced_dimension_starts_each_window_from_the_hours_it_covers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from specsolve.relational.sinks.solvers.highs import Highs
    from tests.test_strategy import WINDOW, WINDOW_AXIS, horizon_sources

    earlier = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS)
    tables = {name: earlier.primal(name) for name in WINDOW['variables']}
    given: list[np.ndarray] = []
    original = Highs.start
    monkeypatch.setattr(Highs, 'start', lambda self, values: (given.append(values), original(self, values))[1])
    again = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, start={'primal': tables})
    assert len(given) == len(earlier.keys), 'every window is started'
    assert all(not np.isnan(values).any() for values in given), 'each window takes a value for every hour it covers'
    assert again.record['objective'].to_list() == pytest.approx(earlier.record['objective'].to_list()), (
        'a start moves the route, never the optimum'
    )


def test_an_archived_window_sweep_starts_a_sweep_from_its_answer(tmp_path: Path) -> None:
    from tests.test_strategy import WINDOW, WINDOW_AXIS, horizon_sources

    earlier = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, archive=tmp_path / 'run.zip')
    archived = sps.load_archive(tmp_path / 'run.zip')
    assert isinstance(archived, sps.archive.SweepArchive)
    again = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, start=archived.sweep)
    assert again.record['objective'].to_list() == pytest.approx(earlier.record['objective'].to_list()), (
        'the answer over the hours is all a window needs, kept windows or not'
    )


#: The second draw cannot be packed at all.
EMPTY_HANDED = {**knapsack_sources(), 'capacity': pl.DataFrame({'value': [-1.0]})}


@pytest.mark.parametrize(
    'start',
    [
        pytest.param(lambda: sps.solve_over(KNAPSACK, {}, DRAWS[:1], key_name='draw'), id='a-sweep-short-of-a-key'),
        pytest.param(
            lambda: sps.solve_over(KNAPSACK, {}, [DRAWS[0], ('b', EMPTY_HANDED)], key_name='draw'),
            id='a-sweep-with-a-slice-that-left-no-values',
        ),
        pytest.param(
            lambda: {'primal': {'take': TWO_ITEMS.with_columns(draw=pl.lit('a'))}}, id='a-table-short-of-a-key'
        ),
    ],
)
def test_a_slice_its_start_leaves_no_row_starts_from_nothing_with_a_warning(start: Any) -> None:
    """A missing start costs the slice its head start, not the sweep its answer."""
    given = start()
    with pytest.warns(SpecsolveWarning, match=r"start= gives the slices \['b'\] no row"):
        sweep = sps.solve_over(KNAPSACK, {}, DRAWS, key_name='draw', start=given)
    assert sweep.record['has_primal'].to_list() == [True, True], 'both draws solve, the second from nothing'


# ---------------------------------------------------------------------------
# start='previous'
# ---------------------------------------------------------------------------


def _warmed(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Every basis HiGHS is warmed from, in order."""
    from specsolve.relational.sinks.solvers.highs import Highs

    warmed: list[object] = []
    original = Highs.warm
    monkeypatch.setattr(Highs, 'warm', lambda self, basis: (warmed.append(basis), original(self, basis))[1])
    return warmed


SCENARIOS = sps.EachCoordinate('scenario')


def test_each_slice_starts_from_the_one_before_it_and_the_first_cold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from tests.test_strategy import DISPATCH as SWEPT
    from tests.test_strategy import scenario_sources

    warmed = _warmed(monkeypatch)
    cold = sps.solve_over(SWEPT, scenario_sources(), SCENARIOS)
    chained = sps.solve_over(SWEPT, scenario_sources(), SCENARIOS, start='previous', spill_to=tmp_path)
    assert not [path.name for path in tmp_path.iterdir() if path.name.endswith('_basis')], (
        'the basis read to chain the slices is not spilled, since the sweep did not ask for it'
    )
    assert len(warmed) == len(chained.keys) - 1, 'every slice but the first starts from the basis before it'
    assert chained.record['objective'].to_list() == pytest.approx(cold.record['objective'].to_list()), (
        'a start moves the route, never the optimum'
    )
    with pytest.raises(SpecsolveError, match=r"outputs=\{'basis'\}"):
        chained.variable_basis('p')


def test_each_window_starts_from_the_window_before_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from tests.test_strategy import WINDOW, WINDOW_AXIS, horizon_sources

    warmed = _warmed(monkeypatch)
    sweep = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, start='previous')
    assert len(warmed) == len(sweep.keys) - 1, 'a window takes the basis before it, matched by its local index'


def test_a_slice_after_one_that_left_no_values_starts_cold(monkeypatch: pytest.MonkeyPatch) -> None:
    from specsolve.relational.sinks.solvers.highs import Highs

    started: list[object] = []
    original = Highs.start
    monkeypatch.setattr(Highs, 'start', lambda self, values: (started.append(values), original(self, values))[1])
    draws = [DRAWS[0], ('b', EMPTY_HANDED), ('c', knapsack_sources())]
    sweep = sps.solve_over(KNAPSACK, {}, draws, key_name='draw', start='previous')
    assert sweep.record['has_primal'].to_list() == [True, False, True], 'the middle draw cannot be packed'
    assert len(started) == 1, "only 'b' is started, from 'a'; 'c' follows a slice with no values and starts cold"


def test_a_spilled_slice_read_back_starts_the_one_after_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The slice before the one solved again lies on disk, and its frames there are the start."""
    from tests.test_strategy import DISPATCH as SWEPT
    from tests.test_strategy import scenario_sources

    call = {'start': 'previous', 'outputs': BASIS, 'spill_to': tmp_path / 'spill'}
    sps.solve_over(SWEPT, scenario_sources(), SCENARIOS, **call)
    last = max((tmp_path / 'spill' / 'record').glob('*.parquet'))
    last.unlink()
    warmed = _warmed(monkeypatch)
    sps.solve_over(SWEPT, scenario_sources(), SCENARIOS, **call)
    assert len(warmed) == 1, 'only the last slice is solved again, warmed from the one before it on disk'


@pytest.mark.parametrize(
    ('call', 'match'),
    [
        pytest.param(
            lambda: sps.solve_over(KNAPSACK, {}, DRAWS, key_name='draw', start='prev'),
            "takes 'previous' as a word, and not 'prev'",
            id='another-word',
        ),
        pytest.param(
            lambda: sps.solve_over(
                KNAPSACK,
                {},
                DRAWS,
                key_name='draw',
                start='previous',
                executor=ThreadPoolExecutor(2),
            ),
            'cannot run concurrently',
            id='previous-under-an-executor',
        ),
        pytest.param(
            lambda: sps.solve(KNAPSACK, knapsack_sources(), start='previous'),
            'a word only solve_over takes',
            id='previous-on-one-solve',
        ),
    ],
)
def test_a_start_word_that_cannot_hold_is_refused(call: Any, match: str) -> None:
    with pytest.raises(SpecsolveError, match=match):
        call()


def test_a_start_table_solve_refuses_stops_the_sweep_before_a_slice_is_built(monkeypatch: pytest.MonkeyPatch) -> None:
    from specsolve import strategy

    monkeypatch.setattr(strategy, 'build', lambda *_: pytest.fail('a slice was built before the refusal'))
    with pytest.raises(SpecsolveError, match="unknown variable 'tkae'"):
        sps.solve_over(KNAPSACK, {}, DRAWS, key_name='draw', start={'primal': {'tkae': TWO_ITEMS}})
