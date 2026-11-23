"""The fold: one plan per slice, and the answers stitched back together.

Every claim here is about slicing, coupling and folding, not the language: the
windowed model ``WINDOW`` uses only constructs that ship.
"""

from __future__ import annotations

import contextlib
import datetime
import importlib.metadata
import json
import math
import multiprocessing
import re
import shutil
import sys
import warnings
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace
from unittest import mock

import polars as pl
import pytest
from mathspec import to_spec

import specsolve as sps
from specsolve import _solve_over as driver
from specsolve import _sweep as sweep_module
from specsolve._axes import CarryRule, Slice, as_built, axis_from
from specsolve._inputs import as_program
from specsolve._records import Metrics, Provenance, Record
from tests.conftest import DISPATCH_SPEC, override

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

GENERATORS = ['wind', 'gas']
STORES = ['battery', 'pumped']

#: Dispatch, declaring no scenario: the slice column never appears
#: in the model.
DISPATCH = DISPATCH_SPEC

#: Storage over a window's *own* labels, with the initial condition split out by a `where` on a dimension
#: literal. `soc_step` carries no `edge=`, so its vacated row drops and the
#: narrowed `soc_open` supplies it from a carried parameter.
WINDOW = {
    'dimensions': {'snapshot': {'dtype': 'int', 'ordered': True}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
        'soc_initial': {'dims': []},
    },
    'variables': {
        'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 30}},
        'discharge': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 30}},
        'soc': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'dims': ['snapshot'],
            'expression': 'sum(p, over=generator) + discharge - charge == load',
        },
        'soc_open': {
            'dims': ['snapshot'],
            'where': 'position(snapshot) == 0',
            'expression': 'soc == soc_initial + charge * 0.9 - discharge',
        },
        'soc_step': {
            'dims': ['snapshot'],
            'where': 'position(snapshot) > 0',
            'expression': 'soc == shift(soc, along=snapshot, offset=1) + charge * 0.9 - discharge',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

#: The same storage, but two of them — so `soc` is over `(t, storage)` and the
#: carried `soc_initial` over `(storage)`. The carry drops `t` and `storage`
#: rides along.
#:
#: `charge` and `discharge` are capped well below a window's worth so a store
#: cannot empty itself before its horizon end; otherwise every window ends at zero and
#: carrying the state is indistinguishable from not carrying it.
MULTI_STORE = {
    'dimensions': {
        'snapshot': {'dtype': 'int', 'ordered': True},
        'generator': {'dtype': 'str'},
        'storage': {'dtype': 'str'},
    },
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['snapshot']},
        'soc_initial': {'dims': ['storage']},
        'efficiency': {'dims': ['storage']},
    },
    'variables': {
        'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'dims': ['snapshot', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'discharge': {'dims': ['snapshot', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'soc': {'dims': ['snapshot', 'storage'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'dims': ['snapshot'],
            'expression': 'sum(p, over=generator) + sum(discharge, over=storage) - sum(charge, over=storage) == load',
        },
        'soc_open': {
            'dims': ['snapshot', 'storage'],
            'where': 'position(snapshot) == 0',
            'expression': 'soc == soc_initial + charge * efficiency - discharge',
        },
        'soc_step': {
            'dims': ['snapshot', 'storage'],
            'where': 'position(snapshot) > 0',
            'expression': 'soc == shift(soc, along=snapshot, offset=1) + charge * efficiency - discharge',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

#: A myopic pathway: what a period builds is what the next period already has.
#: `total` and `existing` are both over `(generator)`, so the carry drops
#: nothing and the whole vector moves.
MYOPIC = {
    'dimensions': {'generator': {'dtype': 'str'}},
    'parameters': {
        'existing': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'demand': {'dims': []},
    },
    'variables': {
        'build': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 50}},
        'total': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 200}},
    },
    'constraints': {
        'accumulate': {'dims': ['generator'], 'expression': 'total == existing + build'},
        'meet': {'dims': [], 'expression': 'sum(total, over=generator) >= demand'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(build * cost, over=generator)'},
}

STATIC = {
    'generator': pl.DataFrame({'generator': GENERATORS}),
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 100.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
}


def scenario_sources() -> dict[str, object]:
    """Three scenarios differing only in load — `load` carries the slice key."""
    rows = []
    for scenario, scale in (('low', 1.0), ('mid', 2.0), ('high', 3.0)):
        rows += [{'scenario': scenario, 'snapshot': t, 'value': 5.0 * scale + t} for t in range(4)]
    return {**STATIC, 'snapshot': pl.DataFrame({'snapshot': range(4)}), 'load': pl.DataFrame(rows)}


def _draw(base: dict, scenario: str, snapshots: int = 4) -> pl.DataFrame:
    """One scenario's load, cut to its first *snapshots* coordinates."""
    return base['load'].filter(pl.col('scenario') == scenario).drop('scenario').head(snapshots)


def multi_store_sources() -> dict[str, object]:
    """Two stores, each with a real starting level worth handing on at a horizon end."""
    return {
        **span_sources(12),
        'storage': pl.DataFrame({'storage': STORES}),
        'soc_initial': pl.DataFrame({'storage': STORES, 'value': [40.0, 20.0]}),
        'efficiency': pl.DataFrame({'storage': STORES, 'value': [0.9, 0.75]}),
    }


def myopic_sources() -> dict[str, object]:
    """Three periods of rising demand — `demand` carries the slice key."""
    return {
        'generator': pl.DataFrame({'generator': GENERATORS}),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
        'existing': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0]}),
        'demand': pl.DataFrame({'period': [1, 2, 3], 'value': [10.0, 25.0, 40.0]}),
    }


def span_sources(periods: int = 12) -> dict[str, object]:
    """A load profile of any length — the pattern repeats past twelve."""
    load = [5.0, 9.0, 30.0, 40.0, 6.0, 8.0, 35.0, 45.0, 7.0, 10.0, 25.0, 50.0]
    return {
        **STATIC,
        'snapshot': range(periods),
        'load': pl.DataFrame({'snapshot': range(periods), 'value': [load[t % len(load)] for t in range(periods)]}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }


def coordinate_sources(coordinates: list, load: float = 5.0) -> dict[str, object]:
    """Six coordinates of flat load, whatever the coordinates are."""
    return {
        **STATIC,
        'snapshot': pl.DataFrame({'snapshot': coordinates}),
        'load': pl.DataFrame({'snapshot': coordinates, 'value': [load] * 6}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }


#: Window geometries whose *tail* differs: a final window of one, a final
#: window of ``horizon``, a windowed dimension shorter than a single window, and a tail that
#: divides exactly.
GEOMETRIES = [
    pytest.param(periods, horizon, lookahead, id=f'n{periods}-h{horizon}-la{lookahead}')
    for periods in (1, 2, 5, 7, 12)
    for horizon in (1, 2, 3, 6)
    for lookahead in range(7 - horizon)
]

#: The one contiguous geometry most window tests share — frozen, so sharing is safe.
WINDOW_AXIS = sps.EachWindow('snapshot', horizon=4, lookahead=0)


@pytest.fixture(scope='module')
def sweep() -> sweep_module.Sweep:
    """The scenario sweep, solved once for every test that only reads it."""
    return sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'))


@pytest.fixture(scope='module')
def overlapping() -> sweep_module.Sweep:
    """The overlapping-window sweep, solved once for every test that only reads it."""
    return sps.solve_over(
        WINDOW,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'}),
    )


@pytest.fixture
def builds(monkeypatch):
    """`spy(module)` — the models `module.build` produces from here on, live."""

    def spy(module) -> list:
        built: list = []
        original = module.build
        monkeypatch.setattr(module, 'build', lambda *a, **k: built.append(original(*a, **k)) or built[-1])
        return built

    return spy


# ---------------------------------------------------------------------------
# EachLabel — the independent case
# ---------------------------------------------------------------------------


def answer_of(runs: sweep_module.Sweep) -> pl.DataFrame:
    """A sweep's record without `solved_at` and `specsolve_run`, which belong to a *run* rather than an answer."""
    return runs.records.drop('solved_at', 'specsolve_run')


def test_a_scenario_sweep_solves_each_slice_and_keys_the_answers(sweep):
    """The model never mentions `scenario`; the driver filters and drops it."""
    runs = sweep

    assert len(runs) == 3
    assert runs.keys == ['high', 'low', 'mid'], 'keys come back sorted, not in data order'
    assert runs.records.columns == [
        'scenario',
        'status',
        'termination_condition',
        'objective',
        'has_primal',
        'solved_at',
        'specsolve_run',
        'slice_axis',
        'slice',
        'solver',
        'solver_version',
        'solver_options',
        'specsolve_version',
        'mathspec_version',
    ], 'the record, keyed'
    assert set(runs.primal('p').columns) == {'scenario', 'snapshot', 'generator', 'value'}
    assert runs.primal('p').height == 3 * 4 * 2

    assert runs.records['solver'].to_list() == ['highs'] * 3, 'every slice names the solver that answered it'
    by_key = dict(zip(runs.records['scenario'], runs.records['objective'], strict=True))
    assert by_key['low'] < by_key['mid'] < by_key['high'], 'a bigger load is a costlier dispatch'


_OPTIONS = {'time_limit': math.inf, 'presolve': 'on'}


def _serial(tmp_path) -> pl.DataFrame:
    return sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), solver_options=_OPTIONS).records


def _pooled(tmp_path) -> pl.DataFrame:
    with ThreadPoolExecutor(2) as pool:
        axis = sps.EachLabel('scenario')
        return sps.solve_over(DISPATCH, scenario_sources(), axis, executor=pool, solver_options=_OPTIONS).records


def _rolling(tmp_path) -> pl.DataFrame:
    carry = {'soc_initial': 'soc'}
    return sps.solve_over(WINDOW, span_sources(), replace(WINDOW_AXIS, carry=carry), solver_options=_OPTIONS).records


def _scanned(tmp_path) -> pl.DataFrame:
    axis = sps.EachLabel('scenario')
    sps.solve_over(DISPATCH, scenario_sources(), axis, spill_to=tmp_path / 'spill', solver_options=_OPTIONS)
    return sps.scan_sweep(tmp_path / 'spill').records


def _read(tmp_path) -> pl.DataFrame:
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), solver_options=_OPTIONS).save(
        tmp_path / 'saved'
    )
    return sps.read_sweep(tmp_path / 'saved').records


def _archived(tmp_path) -> pl.DataFrame:
    axis = sps.EachLabel('scenario')
    sps.solve_over(DISPATCH, scenario_sources(), axis, archive=tmp_path / 'case', solver_options=_OPTIONS)
    return pl.read_parquet(sorted((tmp_path / 'case' / 'answer' / 'record').glob('*.parquet')))


@pytest.mark.parametrize(
    'swept',
    [
        pytest.param(_serial, id='serial'),
        pytest.param(_pooled, id='pooled'),
        pytest.param(_rolling, id='rolling'),
        pytest.param(_scanned, id='scanned-spill'),
        pytest.param(_read, id='read-save'),
        pytest.param(_archived, id='archive-file'),
    ],
)
def test_every_slice_of_every_kind_of_sweep_names_what_produced_it(swept, tmp_path) -> None:
    """The provenance columns sit after the slice columns, and every path a sweep's record takes keeps them."""
    record = swept(tmp_path)
    produced = (
        'highs',
        importlib.metadata.version('highspy'),
        '{"presolve": "on", "time_limit": "inf"}',
        sps.__version__,
        importlib.metadata.version('mathspec'),
    )
    assert record.columns[-7:] == ['slice_axis', 'slice', *Provenance._fields], (
        'the slice columns, then the provenance columns'
    )
    assert set(record.select(Provenance._fields).rows()) == {produced}, (
        'every slice names the solver, its options and the versions that answered it'
    )


def test_a_serial_fold_builds_once_and_updates(builds):
    """The fold is an update loop: nothing after the first slice pays for the YAML, the program or a fresh solver.

    The difference is invisible in the answer, so it is counted at `build`.
    """
    built = builds(driver)

    runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'))

    assert len(runs) == 3, 'three slices'
    assert len(built) == 1, f'{len(built)} builds for three slices — the fold stopped updating'
    seen = built[0].metrics()
    assert (seen.solver_instances, seen.solves) == (1, 3), 'one solver load, the slices differing only in numbers'


def test_a_carried_fold_still_builds_once(builds):
    """A carry writes a parameter the first slice already attached, so no slice names new sources."""
    built = builds(driver)

    runs = sps.solve_over(WINDOW, span_sources(), replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))

    assert runs.keys == [0, 4, 8]
    assert len(built) == 1, f'{len(built)} builds for three windows — the carry cost the fold its fast path'


def test_a_pooled_fold_builds_per_slice(builds):
    """The exception: a built model cannot be pickled, so a pooled slice builds its own."""
    built = builds(driver)

    with ThreadPoolExecutor(2) as pool:
        runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=pool)

    assert len(runs) == 3
    assert len(built) == 3, 'a slice that may run in another process builds its own model'


def test_each_slice_matches_solving_that_slice_alone(sweep):
    """The fold must not change the answer — the oracle is `solve` itself."""
    folded = dict(zip(sweep.records['scenario'], sweep.records['objective'], strict=True))

    for scenario, expected in folded.items():
        one = scenario_sources()
        one['load'] = _draw(one, scenario)
        with sps.solve(DISPATCH, one) as result:
            assert result.objective == pytest.approx(expected)


def test_an_axis_naming_a_column_no_source_carries_says_so():
    with pytest.raises(sps.errors.DataError, match="no source carries a 'draw' column"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('draw'))


def test_a_name_the_sweep_does_not_hold_says_what_it_does_hold(sweep):
    """Everything a slice produced is kept, so a miss is a name, not a flag."""
    with pytest.raises(sps.errors.SpecsolveError, match="no variable 'q' in this sweep"):
        sweep.primal('q')
    with pytest.raises(sps.errors.SpecsolveError, match="no constraint 'nope' in this sweep"):
        sweep.dual('nope')


def test_a_sweep_that_solved_nothing_blames_the_solve():
    """An absent table has one cause, and the message says which.

    The load is pushed past total capacity, so every slice is infeasible.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'))

    assert len(runs) == 3, 'an unsolvable slice is still a row of the record'
    assert runs.records['objective'].null_count() == 3, 'no slice reached one, and none is written as nan'
    with pytest.raises(sps.errors.SpecsolveError, match='holds no variable tables at all') as raised:
        runs.primal('p')
    assert 'infeasible' in str(raised.value), 'the message names what the slices actually did'


def test_a_sweep_that_solved_and_holds_no_expression_does_not_blame_the_solve(sweep):
    """A kind the sweep holds no table of was read as a sweep that solved nothing.

    The dispatch spec declares no expression, so every slice is optimal and
    the sweep holds none. The refusal said the slices terminated optimal and
    that the models did not solve, in one breath.
    """
    assert sweep.records['has_primal'].all()
    with pytest.raises(sps.errors.SpecsolveError, match='holds no named expression tables') as raised:
        sweep.evaluate('2 * p')
    assert 'did not solve' not in str(raised.value), 'every slice solved, and the refusal does not say otherwise'
    assert '3 of its 3 slices left values' in str(raised.value), 'the refusal says what is true of them'


def test_a_slice_that_reached_no_objective_does_not_poison_the_sweep():
    """`objective` is a table, and in a table an absent number is null.

    nan is a *number* to every aggregate that meets it, so one infeasible
    slice makes the mean over the sweep nan.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(
        pl.when(pl.col('scenario') == 'high').then(pl.col('value') + 1_000).otherwise(pl.col('value'))
    )
    runs = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'))

    assert runs.records['objective'].null_count() == 1, 'the one slice that came back infeasible'
    assert runs.records['objective'].is_nan().sum() == 0, 'written as no value rather than as nan'
    assert runs.records['objective'].mean() == pytest.approx(runs.records.filter('has_primal')['objective'].mean()), (
        'so the mean over the sweep is the mean over the slices that solved'
    )


# ---------------------------------------------------------------------------
# EachWindow — the coupled case
# ---------------------------------------------------------------------------


def test_a_rolling_horizon_carries_state_on_from_each_horizon_end():
    """Three contiguous windows, the store's level handed forward.

    `soc_initial` is updated per window from the previous window's last `soc`,
    which is the carry doing its one job — a copy, at a named index.
    """
    runs = sps.solve_over(WINDOW, span_sources(), replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))

    assert runs.keys == [0, 4, 8]
    assert runs.primal('p', per_window=True).height == 3 * 4 * 2
    assert set(runs.primal('soc', per_window=True).columns) == {'snapshot_start', 'snapshot', 'value'}, (
        'per window the rows carry their own `snapshot`, so the key column cannot be called `snapshot`'
    )
    assert runs.records['objective'].to_list() == pytest.approx([2270.0, 2770.0, 2655.0])


def test_overlapping_windows_advance_by_step_and_look_ahead_by_length(overlapping):
    runs = overlapping
    assert runs.keys == [0, 3, 6, 9]
    assert runs.primal('soc', per_window=True).filter(pl.col('snapshot_start') == 9).height == 3, (
        'the tail window is short rather than padded: 9..11 is three rows, not six'
    )


def test_stitch_drops_the_overlap_and_restores_the_global_coordinate(overlapping):
    """The answer a rolling horizon is for, without the caller doing arithmetic."""
    runs = overlapping
    stitched = runs.primal('soc')
    assert stitched.columns == ['snapshot', 'value'], 'the slice bookkeeping is gone'
    assert stitched['snapshot'].to_list() == list(range(12)), 'and every coordinate is present once'
    assert runs.primal('soc', per_window=True).height == 21, 'every window solved its lookahead too'


@pytest.mark.parametrize(('periods', 'horizon', 'lookahead'), GEOMETRIES)
def test_a_window_geometry_covers_every_coordinate_exactly_once(periods, horizon, lookahead):
    """A stitched sweep reproduces the coordinate list, whatever the tail."""
    runs = sps.solve_over(
        WINDOW,
        span_sources(periods),
        sps.EachWindow('snapshot', horizon=horizon, lookahead=lookahead),
    )
    assert runs.primal('soc')['snapshot'].to_list() == list(range(periods)), (
        'the answer must reproduce the coordinate list, whatever the tail'
    )
    assert runs.primal('soc', per_window=True)['snapshot_start'].n_unique() == len(range(0, periods, horizon)), (
        'one slice per window start'
    )


@pytest.mark.parametrize(('periods', 'horizon', 'lookahead'), GEOMETRIES)
def test_a_carry_finds_the_horizon_end_in_every_geometry(periods, horizon, lookahead):
    """The coordinate a carry hands on is the window's horizon end.

    A non-final window's horizon is exactly ``horizon=`` labels; a final one's is
    whatever is left, which can be one. Both are in range by construction,
    because a window's horizon is rows it solved.
    """
    runs = sps.solve_over(
        WINDOW,
        span_sources(periods),
        sps.EachWindow('snapshot', horizon=horizon, lookahead=lookahead, carry={'soc_initial': 'soc'}),
    )
    assert runs.primal('soc')['snapshot'].to_list() == list(range(periods)), (
        'the horizon end is in range for every geometry, so the sweep completes'
    )


def test_stitch_keeps_the_whole_of_the_final_short_window():
    """A tail window holds at most `horizon`, so its horizon is all of it.

    12 coordinates kept 5 at a time leaves a final window of two.
    """
    runs = sps.solve_over(
        WINDOW,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=5, lookahead=1),
    )
    assert runs.keys == [0, 5, 10], 'three windows, the last of two coordinates'
    assert runs.primal('soc')['snapshot'].to_list() == list(range(12)), (
        'the short tail window is kept whole, not dropped for being short'
    )


def _spilled_by_label(tmp_path):
    return sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path / 'runs')


@pytest.mark.parametrize(
    'read',
    [
        pytest.param(lambda sweep, tmp_path: sweep.primal('p', per_window=True), id='a-coordinate-primal'),
        pytest.param(lambda sweep, tmp_path: sweep.dual('balance', per_window=True), id='a-coordinate-dual'),
        pytest.param(lambda sweep, tmp_path: sweep.to_dataset(per_window=True), id='a-coordinate-dataset'),
        pytest.param(
            lambda sweep, tmp_path: _spilled_by_label(tmp_path).scan('p', per_window=True), id='a-spilled-scan'
        ),
    ],
)
def test_per_window_is_refused_on_a_sweep_that_was_not_cut_into_windows(read, sweep, tmp_path):
    """`per_window=True` on a sweep whose answer already is one table per slice is refused rather than ignored."""
    with pytest.raises(
        sps.errors.SpecsolveError, match='not cut into windows: its answer already is one table per slice'
    ):
        read(sweep, tmp_path)


def test_duals_stitch_the_same_way_primals_do(overlapping):
    """A price at a coordinate is the one of the window whose horizon holds it, not a blend.

    What has to be undone is a property of the *axis*, so a dual stitches as a
    primal does.
    """
    runs = overlapping
    windows, answer = runs.dual('balance', per_window=True), runs.dual('balance')
    assert windows.columns == ['snapshot_start', 'snapshot', 'value']
    assert answer.columns == ['snapshot', 'value']
    assert answer['snapshot'].to_list() == list(range(12)), 'one price per coordinate'
    assert windows.height > answer.height, 'the overlap is priced twice before the answer collapses it'


def test_per_window_keeps_every_row_the_answer_drops(overlapping):
    """The answer drops the lookahead rows; `per_window=True` is where they are still read."""
    runs = overlapping
    assert runs.primal('soc').height == 12, "the answer holds only the rows of each window's horizon"
    assert runs.primal('soc', per_window=True).height == 21, 'per window keeps every row every window solved'
    assert runs.records.join(runs.primal('soc', per_window=True), on=runs.key_name).height == 21, (
        'per window is keyed by the same column as `objective`, so the two still join'
    )


# ---------------------------------------------------------------------------
# named expressions across a sweep
# ---------------------------------------------------------------------------

#: The window model with its cost named twice: `spend` keeps the window's labels
#: (pointwise in `t`, so it can be stitched) and `window_spend` reduces over them
#: (one number per window, so it cannot).
SPENDING = override(
    WINDOW,
    **{
        'expressions.spend': 'sum(p * cost, over=generator)',
        'expressions.window_spend': 'sum(sum(p * cost, over=generator), over=snapshot)',
    },
)


@pytest.fixture(scope='module')
def priced() -> sweep_module.Sweep:
    """The overlapping-window sweep of the expression-bearing model, solved once."""
    return sps.solve_over(
        SPENDING,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'}),
    )


def test_a_stitched_expression_prices_only_the_rows_of_each_horizon(priced):
    """The answer of an expression drops the lookahead double-count.

    The oracle is the answer's dispatch priced by hand. The per-window sum
    exceeds it, since the per-window tables carry the overlap.
    """
    stitched = priced.evaluate('spend')
    assert stitched.columns == ['snapshot', 'value']
    assert stitched['snapshot'].to_list() == list(range(12)), 'one value per coordinate, like a stitched primal'

    by_hand = (
        priced.primal('p')
        .join(STATIC['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum())
        .sort('snapshot')
    )
    assert stitched['value'].to_list() == pytest.approx(by_hand['value'].to_list())
    assert priced.evaluate('spend', per_window=True)['value'].sum() > stitched['value'].sum(), (
        'the per-window tables still carry the lookahead rows, so their sum double-counts'
    )


def test_a_quantity_reduced_over_the_sliced_dimension_has_no_way_back(priced):
    """Per window it reads; as an answer the refusal says why not, and names the read that works."""
    keyed = priced.evaluate('window_spend', per_window=True)
    assert keyed.columns == ['snapshot_start', 'value']
    assert keyed.height == len(priced), 'one total per window, keyed like objective'
    with pytest.raises(sps.errors.SpecsolveError, match=r'not over the windowed dimension.*per_window=True'):
        priced.evaluate('window_spend')


def test_each_slice_expression_matches_solving_that_slice_alone():
    """The fold must not change an expression's value — the oracle is `solve`."""
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    runs = sps.solve_over(spec, scenario_sources(), sps.EachLabel('scenario'))

    for scenario in runs.keys:
        one = scenario_sources()
        one['load'] = _draw(one, scenario)
        with sps.solve(spec, one) as result:
            alone = result.evaluate('spend')
            folded = runs.evaluate('spend').filter(pl.col('scenario') == scenario).drop('scenario')
            assert folded['value'].to_list() == pytest.approx(alone['value'].to_list()), (
                'a slice read out of the sweep is the slice solved alone'
            )


def test_an_expression_the_sweep_does_not_hold_says_what_it_does_hold(priced):
    with pytest.raises(sps.errors.SpecsolveError, match="no named expression 'nope' in this sweep"):
        priced.evaluate('nope')


def test_an_expression_no_slice_could_evaluate_carries_its_reason():
    """A failing evaluation is carried per name, never raised mid-fold.

    `ratio` divides by a parameter with one row, so every slice's evaluation
    fails on the sparse divisor — and the sweep must still complete, hold every
    other table, and hand the caller the divisor's own sentence on read.
    """
    spec = override(
        SPENDING,
        **{'parameters.scale': {'dims': ['snapshot'], 'missing': 'neutral'}, 'expressions.ratio': 'load / scale'},
    )
    sources = {**span_sources(12), 'scale': pl.DataFrame({'snapshot': [0], 'value': [2.0]})}
    runs = sps.solve_over(spec, sources, sps.EachWindow('snapshot', horizon=6, lookahead=0))

    assert runs.primal('p').height > 0, 'the failing expression must not fail the sweep'
    assert runs.evaluate('spend').height > 0, 'nor take the healthy expression with it'
    with pytest.raises(sps.errors.SpecsolveError, match='scale'):
        runs.evaluate('ratio')


#: Six coordinates, three windows of two, whatever the coordinates *are*:
#: `horizon` and `lookahead` count coordinates rather than coordinate values.
COORDINATE_TYPES = [
    pytest.param(list(range(6)), id='dense-ints'),
    pytest.param([0, 10, 20, 30, 40, 50], id='gapped-ints'),
    pytest.param(list(range(100, 106)), id='ints-not-from-zero'),
    pytest.param([datetime.datetime(2030, 1, 1, h) for h in range(6)], id='datetimes'),
    pytest.param([f's{i}' for i in range(6)], id='strings'),
]


@pytest.mark.parametrize('coordinates', COORDINATE_TYPES)
def test_a_window_spans_coordinates_whatever_they_are_numbered(coordinates):
    """The only requirement on a windowed dimension is that it is orderable.

    The initial condition's `where: "position(snapshot) == 0"` counts within the window, so
    it matches each window's first label whatever the labels are.
    """
    runs = sps.solve_over(WINDOW, coordinate_sources(coordinates), sps.EachWindow('snapshot', horizon=2, lookahead=0))

    assert len(runs) == 3
    assert runs.keys == coordinates[::2], 'a window is keyed by its first coordinate'
    soc = runs.primal('soc', per_window=True)
    assert soc.height == 6
    assert soc['snapshot'].to_list() == coordinates, 'each window holds its own labels, and no local index'


@pytest.mark.parametrize('coordinates', COORDINATE_TYPES)
def test_stitch_recovers_coordinates_no_arithmetic_could(coordinates):
    """`snapshot_start + t` is meaningless for a datetime or a string axis.

    The window→coordinate mapping is the axis's to keep, and stitching is the
    only way back to it: nothing the caller holds could reconstruct these.
    """
    runs = sps.solve_over(
        WINDOW, coordinate_sources(coordinates, load=10.0), sps.EachWindow('snapshot', horizon=2, lookahead=0)
    )
    assert runs.primal('soc')['snapshot'].to_list() == coordinates


HOURS = [datetime.datetime(2030, 1, 1, h) for h in range(6)]


def _in_unit(unit: str, *, nanoseconds: int = 0) -> pl.Series:
    """``HOURS`` held in *unit*, the first *nanoseconds* past the hour."""
    return (
        pl.Series(HOURS).cast(pl.Datetime('ns')) + pl.Series([nanoseconds, *[0] * 5]).cast(pl.Duration('ns'))
    ).dt.cast_time_unit(unit)


def _keyed_by_datetime(keys: pl.Series) -> dict[str, object]:
    """``scenario_sources`` with its first three keys keyed by *keys* rather than by name."""
    sources = scenario_sources()
    named = pl.DataFrame({'scenario': ['low', 'mid', 'high'], 'key': keys[:3]})
    load = pl.DataFrame(sources['load']).join(named, on='scenario').drop('scenario').rename({'key': 'scenario'})
    return {**sources, 'load': load}


@pytest.mark.parametrize(
    ('unit', 'kind'),
    [
        pytest.param('ns', 'window', id='window-ns'),
        pytest.param('ms', 'window', id='window-ms'),
        pytest.param('ns', 'label', id='label-ns'),
        pytest.param('ms', 'label', id='label-ms'),
    ],
)
def test_a_sweep_over_a_datetime_axis_in_any_unit_solves(unit, kind):
    """A window over a nanosecond or millisecond axis crashed in its first slice.

    The coordinates were read out as python datetimes, which are microseconds,
    and met a column in another unit.
    """
    keys = _in_unit(unit)
    if kind == 'window':
        runs = sps.solve_over(WINDOW, coordinate_sources(keys), sps.EachWindow('snapshot', horizon=2, lookahead=0))
        assert runs.primal('soc')['snapshot'].to_list() == HOURS, 'the stitch gives back every hour'
    else:
        runs = sps.solve_over(DISPATCH, _keyed_by_datetime(keys), sps.EachLabel('scenario'))
        assert runs.keys == HOURS[:3], 'one slice per key, keyed by the instant'


@pytest.mark.parametrize('kind', ['window', 'label'])
def test_a_sweep_over_a_datetime_axis_finer_than_a_microsecond_is_refused(kind):
    """A key read out as a python datetime dropped its nanoseconds and no longer matched its own rows.

    A window crashed, and a label sweep built that slice empty.
    """
    keys = _in_unit('ns', nanoseconds=1)
    with pytest.raises(sps.errors.DataError, match='finer than a microsecond'):
        if kind == 'window':
            sps.solve_over(WINDOW, coordinate_sources(keys), sps.EachWindow('snapshot', horizon=2, lookahead=0))
        else:
            sps.solve_over(DISPATCH, _keyed_by_datetime(keys), sps.EachLabel('scenario'))


def test_a_window_key_column_never_shadows_the_dimension_it_cut(sweep):
    """`snapshot_start` holds window starts, beside the `snapshot` every row of a window carries.

    A key column called `snapshot` would collide with the dimension itself.
    """
    runs = sps.solve_over(WINDOW, span_sources(), WINDOW_AXIS)
    soc = runs.primal('soc', per_window=True)
    assert soc.columns == ['snapshot_start', 'snapshot', 'value'], 'the key, then the window in its own labels'
    assert runs.records.columns[0] == 'snapshot_start', 'both tables key the same way'
    assert sorted(soc['snapshot_start'].unique().to_list()) == [0, 4, 8]

    assert sweep.records.columns[0] == 'scenario', (
        'EachLabel keeps the plain name: there the key really is a coordinate of it'
    )


@pytest.mark.parametrize(
    ('geometry', 'expected'),
    [
        pytest.param({'horizon': 0, 'lookahead': 0}, 'at least one label', id='a-window-keeping-nothing'),
        pytest.param(
            {'horizon': [4, 0], 'lookahead': 0},
            r'at least one label.*\[0\]',
            id='a-horizon-keeping-nothing',
        ),
        pytest.param({'horizon': [], 'lookahead': 0}, 'horizon is empty', id='no-horizons-at-all'),
        pytest.param({'horizon': 4, 'lookahead': -1}, 'is negative', id='a-negative-lookahead'),
    ],
)
def test_the_window_geometry_is_checked_at_construction(geometry, expected):
    """A bad geometry is refused when the axis is constructed."""
    with pytest.raises(ValueError, match=expected):
        sps.EachWindow('snapshot', **geometry)


#: Window horizons that are not all the same size. Between them: a telescoping
#: sequence that coarsens, one that refines, months of unequal length, and a
#: sequence overshooting the axis so its tail horizons have nothing to cover.
HORIZONS = [
    pytest.param([1, 2, 3, 6], id='coarsening'),
    pytest.param([6, 3, 2, 1], id='refining'),
    pytest.param([4, 4, 4], id='uniform-spelled-as-a-sequence'),
    pytest.param([5, 7], id='two-unequal-months'),
    pytest.param([5, 7, 99], id='a-horizon-past-the-end-of-the-axis'),
]


@pytest.mark.parametrize('horizons', HORIZONS)
def test_windows_of_unequal_size_cover_every_coordinate_exactly_once(horizons):
    """`horizon` as a sequence keeps those numbers in order, one window each."""
    runs = sps.solve_over(
        WINDOW,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=horizons, lookahead=2, carry={'soc_initial': 'soc'}),
    )
    stitched = runs.primal('soc')

    assert stitched['snapshot'].to_list() == list(range(12)), 'every coordinate, once, whatever the horizon sizes'
    assert runs.keys == _starts(horizons, 12), 'one window per horizon, keyed by the coordinate it starts on'
    assert runs.records['termination_condition'].to_list() == ['optimal'] * len(runs), 'every window solved'


def _starts(horizons: list[int], total: int) -> list[int]:
    """Where each window starts — the keys a horizon list implies, a horizon past the end contributing none."""
    starts, at = [], 0
    for horizon in horizons:
        if at >= total:
            break
        starts.append(at)
        at += horizon
    return starts


#: Horizon lists whose last element overshoots what the axis has left — an int that
#: does not divide, and a sequence whose tail reaches past the end.
OVERSHOOTING = [
    pytest.param(5, 12, [5, 5, 2], id='an-int-that-does-not-divide'),
    pytest.param(7, 12, [7, 5], id='an-int-larger-than-the-remainder'),
    pytest.param(20, 12, [12], id='an-int-larger-than-the-axis'),
    pytest.param([5, 7, 99], 12, [5, 7], id='a-sequence-reaching-past-the-end'),
    pytest.param([5, 20], 12, [5, 7], id='a-sequence-whose-last-horizon-overshoots'),
]


@pytest.mark.parametrize(('horizon', 'periods', 'expected'), OVERSHOOTING)
def test_the_horizons_partition_the_axis_and_never_claim_more_than_is_left(horizon, periods, expected):
    """A probe, because no solved sweep can tell `min(horizon, left)` from `horizon`.

    Only the *last* horizon can overshoot, and its carry is never read, so
    trimming it changes no answer. It keeps each window's horizon true, which the
    stitch and the carry both read.
    """
    axis = sps.EachWindow('snapshot', horizon=horizon, lookahead=0)

    assert axis._horizon_lengths(periods) == expected, 'each horizon is trimmed to what the axis has left'
    assert sum(axis._horizon_lengths(periods)) == periods, 'and together they cover it exactly once'

    slices, stitch = axis._slice(as_program(WINDOW), span_sources(periods), 'snapshot_start')
    horizons = stitch.horizon.group_by('snapshot_start', maintain_order=True).len()['len'].to_list()
    assert horizons == expected, 'which is what each slice records as its horizon'
    for current in slices:
        assert current.horizon_end in current.sources['snapshot'], 'no horizon end is a label its window does not hold'


def test_a_horizon_list_that_stops_short_of_the_axis_is_refused():
    """The coordinates past the last horizon would be solved by no window.

    Trimming them silently is the one outcome a sweep must not have: the stitch
    would come back short and read as a complete schedule.
    """
    with pytest.raises(sps.errors.DataError, match=r'keeps 7 label\(s\) across 2 window\(s\).*has 12'):
        sps.solve_over(
            WINDOW,
            span_sources(12),
            sps.EachWindow('snapshot', horizon=[3, 4], lookahead=0),
        )


def test_the_lookahead_the_model_needs_is_one_number_whatever_the_horizons():
    """`Separability.ahead` is one integer for the dimension, so the gate is one check.

    `shift(soc, along=snapshot, offset=-1)` reads one label ahead, so a lookahead
    of zero is refused and one is enough — for uniform horizons and unequal ones
    alike, since no horizon size enters the arithmetic.
    """
    ahead = 'soc == shift(soc, along=snapshot, offset=-1) + charge * 0.9 - discharge'
    reaching = override(WINDOW, **{'constraints.soc_step.expression': ahead})

    for horizon in (3, [1, 2, 3, 6]):
        with pytest.raises(
            sps.errors.SpecsolveError, match=r'lookahead=0\) looks ahead by 0 label\(s\).*reads 1 ahead'
        ):
            sps.solve_over(reaching, span_sources(12), sps.EachWindow('snapshot', horizon=horizon, lookahead=0))

        runs = sps.solve_over(reaching, span_sources(12), sps.EachWindow('snapshot', horizon=horizon, lookahead=1))
        assert runs.records['termination_condition'].to_list() == ['optimal'] * len(runs), (
            'one label of lookahead is what the model reads, so every window is whole'
        )


def test_a_short_tail_window_carries_off_its_own_last_row():
    """A final window's horizon is shorter than ``horizon``, and its horizon end is its own last row.

    12 coordinates kept 5 at a time leaves a final window of two, which
    holds no fifth row. Nothing reads the last slice's carry, so the value is
    never computed — but a window short of ``horizon`` in the *middle* of a sweep
    cannot happen, which is what keeps every horizon end in range.
    """
    runs = sps.solve_over(
        WINDOW,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=5, lookahead=1, carry={'soc_initial': 'soc'}),
    )
    assert runs.keys == [0, 5, 10]
    assert runs.records['termination_condition'].to_list() == ['optimal'] * 3
    assert runs.primal('soc', per_window=True).filter(pl.col('snapshot_start') == 10).height == 2


def test_a_carry_collapses_one_dimension_and_every_other_rides_along():
    """`soc` is over `(snapshot, storage)` and `soc_initial` over `(storage)`.

    The two entries say what is copied: `snapshot` is what the parameter
    lacks, so it is the one the carry collapses, and `storage` passes through —
    both stores are handed forward, each its own level.
    """
    runs = sps.solve_over(MULTI_STORE, multi_store_sources(), replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))

    assert runs.keys == [0, 4, 8]
    assert set(runs.primal('soc', per_window=True).columns) == {'snapshot_start', 'snapshot', 'storage', 'value'}

    def at(name: str, start: int, t: int, store: str) -> float:
        rows = runs.primal(name, per_window=True).filter(
            (pl.col('snapshot_start') == start) & (pl.col('snapshot') == start + t) & (pl.col('storage') == store)
        )
        return rows['value'].item()

    efficiency = dict(zip(STORES, [0.9, 0.75], strict=True))
    for previous, start in ((0, 4), (4, 8)):
        for store in STORES:
            opened = at('soc', start, 0, store)
            expected = (
                at('soc', previous, 3, store)
                + at('charge', start, 0, store) * efficiency[store]
                - at('discharge', start, 0, store)
            )
            assert opened == pytest.approx(expected, abs=1e-6), (
                'the opening row is the previous window at its fourth row, for this same store'
            )

    fresh = sps.solve_over(MULTI_STORE, multi_store_sources(), WINDOW_AXIS)
    assert not fresh.primal('soc', per_window=True).equals(runs.primal('soc', per_window=True)), (
        'the carry changed nothing'
    )


def test_the_carried_row_is_the_horizon_end_and_not_the_last_one_solved():
    """Under overlap the two differ, and only the horizon end holds the state to hand on.

    At `length=6, step=3` a window solves six rows, 0..5 from its start, and its horizon is 0..2. The lookahead
    rows 3..5 are solved against a window that ends at 5, so the store empties
    into them; the next window recomputes those coordinates as its own
    horizon. Handing row 5 forward would seed it with a level that was never
    going to happen.

    The first assertion is what makes the rest discriminating: where the two
    rows hold the same level, reading either passes.
    """
    runs = sps.solve_over(
        WINDOW,
        span_sources(12),
        sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'}),
    )

    def at(name: str, start: int, t: int) -> float:
        on = (pl.col('snapshot_start') == start) & (pl.col('snapshot') == start + t)
        table = runs.primal(name, per_window=True).filter(on)
        return table['value'].item()

    for start in (0, 3, 6):
        assert at('soc', start, 2) != pytest.approx(at('soc', start, 5), abs=1e-6), (
            'the horizon end and the last solved row must differ, or this test cannot tell them apart'
        )

    for previous, start in ((0, 3), (3, 6), (6, 9)):
        opened = at('soc', start, 0) - at('charge', start, 0) * 0.9 + at('discharge', start, 0)
        assert opened == pytest.approx(at('soc', previous, 2), abs=1e-6), (
            "the window opens on the previous one's horizon end"
        )
        assert opened != pytest.approx(at('soc', previous, 5), abs=1e-6), (
            'and not on the last row it solved, which is lookahead the next window recomputes'
        )


def test_a_myopic_pathway_carries_a_whole_vector():
    """Capacity per generator, handed forward as a table rather than a number.

    `total` and `existing` are both over `(generator)`, so nothing is dropped
    and the table *is* the carry.
    """
    runs = sps.solve_over(
        MYOPIC,
        myopic_sources(),
        sps.EachLabel('period', carry={'existing': 'total'}),
    )

    assert runs.keys == [1, 2, 3]
    built = runs.primal('build').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    total = runs.primal('total').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    assert built == pytest.approx([10.0, 15.0, 15.0]), (
        'each period builds only the increment: what the last one built came back as `existing`'
    )
    assert total == pytest.approx([10.0, 25.0, 40.0]), 'demand 10 -> 25 -> 40 is met exactly'


#: The five ways a carry cannot line up.
_PERIOD_AXIS = sps.EachLabel('period')
#: The stores with a size to choose, which is over no snapshot, and a budget for it.
BUILDING_STORES = override(
    MULTI_STORE,
    **{
        'parameters.budget': {'dims': []},
        'variables.built': {'dims': ['storage'], 'bounds': {'lower': 0, 'upper': 100}},
    },
)

UNSOUND_CARRIES = [
    pytest.param(
        WINDOW, span_sources, WINDOW_AXIS, {'soc_initial': 'p'},
        r'would collapse .*at once', "['snapshot', 'generator']",
        id='collapses-two-dimensions',
    ),
    pytest.param(
        WINDOW, span_sources, WINDOW_AXIS, {'p_max': 'soc'},
        'cannot line up', None,
        id='parameter-over-more-than-the-variable',
    ),
    pytest.param(
        WINDOW, span_sources, WINDOW_AXIS, {'soc_initial': 'nope'},
        'does not declare', None,
        id='a-name-neither-side-declares',
    ),
    pytest.param(
        WINDOW, span_sources, WINDOW_AXIS, {'load': 'soc'},
        r"carry 'load' <- 'soc' keeps 'snapshot'", 'no two windows hold the same labels',
        id='a-window-carrying-a-whole-table-over-the-dimension-it-windows',
    ),
    pytest.param(
        WINDOW, span_sources, sps.EachLabel('scenario'), {'soc_initial': 'soc'},
        r"collapses 'snapshot', and this axis advances along none", 'Reduce',
        id='a-label-sweep-collapsing-a-dimension-it-does-not-advance-along',
    ),
    pytest.param(
        BUILDING_STORES, lambda: {**multi_store_sources(), 'budget': pl.DataFrame({'value': [1.0]})},
        WINDOW_AXIS, {'budget': 'built'},
        r"collapses 'storage', and this axis advances along 'snapshot'", 'Reduce',
        id='a-window-collapsing-a-dimension-that-is-not-its-own',
    ),
]  # fmt: skip


@pytest.mark.parametrize(('spec', 'sources', 'axis', 'carry', 'expected', 'names'), UNSOUND_CARRIES)
def test_a_carry_that_cannot_line_up_says_so_before_anything_solves(spec, sources, axis, carry, expected, names):
    """Every one of these is answerable from the two entries and the axis alone.

    The axis decides the last two: the coordinate handed on is a slice's
    horizon end, so only the dimension the axis advances along can be the one a
    carry collapses. Any other and there is no coordinate to choose without
    doing the model's arithmetic here.
    """
    with pytest.raises(sps.errors.SpecsolveError, match=expected) as raised:
        sps.solve_over(spec, sources(), replace(axis, carry=carry))
    if names is not None:
        assert names in str(raised.value), 'the message names the dimensions it could not choose between'


#: The window model over hours, a datetime dimension.
BY_THE_HOUR = override(WINDOW, **{'dimensions.snapshot': {'dtype': 'datetime', 'ordered': True}})


def _dated_sources(labels: list[datetime.datetime], load_labels: pl.Series) -> dict[str, object]:
    """The full span over *labels*, its load keyed by *load_labels*."""
    sources = span_sources(len(labels))
    return {
        **sources,
        'snapshot': pl.DataFrame({'snapshot': labels}),
        'load': sources['load'].with_columns(load_labels.alias('snapshot')),
    }


def test_a_window_refuses_a_label_finer_than_a_microsecond_as_a_single_solve_does():
    """A window cast every label to microseconds before matching it, so a row a nanosecond past a label took that label's place.

    A single solve refuses the row. The window read its value at the label
    beside it.
    """
    hours = [datetime.datetime(2030, 1, 1, h) for h in range(8)]
    nanoseconds = pl.Series(hours).dt.cast_time_unit('ns').to_physical()
    a_nanosecond_past = (nanoseconds + pl.Series([0, 1, 0, 0, 0, 0, 0, 0])).cast(pl.Datetime('ns'))
    sources = _dated_sources(hours, a_nanosecond_past)
    finer = r"parameter 'load' holds 'snapshot' label\(s\) finer than a microsecond"
    with pytest.raises(sps.errors.DataError, match=finer):
        sps.solve(BY_THE_HOUR, sources)
    with pytest.raises(sps.errors.DataError, match=finer):
        sps.solve_over(BY_THE_HOUR, sources, WINDOW_AXIS)


def test_a_window_takes_a_date_over_a_datetime_dimension_as_a_single_solve_does():
    """A date is the microsecond its day starts on, and a window compared it with a datetime as it came.

    polars refused the comparison, so a table a single solve attaches stopped
    the sweep.
    """
    days = [datetime.datetime(2030, 1, d) for d in range(1, 9)]
    by_datetime = sps.solve_over(BY_THE_HOUR, _dated_sources(days, pl.Series(days)), WINDOW_AXIS)
    by_date = sps.solve_over(BY_THE_HOUR, _dated_sources(days, pl.Series(days).cast(pl.Date)), WINDOW_AXIS)
    assert by_date.records['objective'].to_list() == by_datetime.records['objective'].to_list()
    assert by_date.primal('soc').equals(by_datetime.primal('soc'))


@pytest.mark.filterwarnings('ignore:.*a partial start of values for an LP')
def test_a_window_cuts_a_start_keyed_by_date_as_a_single_solve_reads_it():
    """A start table is cut as a source is, and one keyed by date stopped the sweep on the same comparison."""
    days = [datetime.datetime(2030, 1, d) for d in range(1, 9)]
    sources = _dated_sources(days, pl.Series(days))
    whole = sps.solve(BY_THE_HOUR, sources)
    by_date = {'primal': {'soc': whole.primal('soc').with_columns(pl.col('snapshot').cast(pl.Date))}}
    assert sps.solve(BY_THE_HOUR, sources, start=by_date).objective == pytest.approx(whole.objective), (
        'a single solve reads each date as the midnight it starts on'
    )
    swept = sps.solve_over(BY_THE_HOUR, sources, WINDOW_AXIS, start=by_date)
    assert swept.records['has_primal'].all(), 'every window solved from the rows of the start it covers'


def test_a_window_refuses_labels_that_carry_a_relation_as_a_single_solve_does():
    """A window replaced the labels table with its own labels, so a column `tidy_sources` refuses there went unseen."""
    spec = override(
        WINDOW,
        **{'dimensions.season': {'dtype': 'str'}, 'relations.season_of': {'key': 'snapshot', 'values': 'season'}},
    )
    seasons = ['winter'] * 4 + ['summer'] * 4
    sources = {
        **span_sources(8),
        'season': ['winter', 'summer'],
        'season_of': pl.DataFrame({'snapshot': range(8), 'season': seasons}),
        'snapshot': pl.DataFrame({'snapshot': range(8), 'season_of': seasons}),
    }
    carried = "the labels table of dimension 'snapshot' carries a 'season_of' column"
    with pytest.raises(sps.errors.DataError, match=carried):
        sps.solve(spec, sources)
    with pytest.raises(sps.errors.DataError, match=carried):
        sps.solve_over(spec, sources, WINDOW_AXIS)


def test_a_start_with_a_status_that_is_none_is_refused_before_a_slice_is_solved(tmp_path):
    """A basis status was read when its slice was solved, so a bad one in the second window left the first on disk."""
    sources = span_sources(8)
    basis = (
        sps.solve(WINDOW, sources, outputs={'basis'})
        .variable_basis('soc')
        .with_columns(pl.col('value').cast(pl.String))
    )
    bogus = basis.with_columns(
        pl.when(pl.col('snapshot') >= 4).then(pl.lit('bogus')).otherwise(pl.col('value')).alias('value')
    )
    out = tmp_path / 'spill'
    with pytest.raises(sps.errors.SpecsolveError, match="gives the basis status 'bogus'"):
        sps.solve_over(WINDOW, sources, WINDOW_AXIS, start={'variable_basis': {'soc': bogus}}, spill_to=out)
    assert not out.exists(), 'refused before the spill is opened, so before any slice is solved'


def test_a_sweep_copied_halfway_is_solved_again_rather_than_read_back(monkeypatch, tmp_path):
    """A copied sweep landed its records before its metrics, and a record is what says a slice is done.

    A copy that stopped between the two left slices the next run read back,
    and the read failed on the metrics that never landed.
    """
    spilled = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path / 'spill')
    write = sweep_module.write_whole

    def stops_at_the_metrics(table, path) -> None:
        if path.parent.name == 'metrics':
            raise _CrashError
        write(table, path)

    with monkeypatch.context() as patched:
        patched.setattr(sweep_module, 'write_whole', stops_at_the_metrics)
        with pytest.raises(_CrashError):
            spilled.save(tmp_path / 'copy')

    again = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path / 'copy')
    assert again.records['objective'].to_list() == spilled.records['objective'].to_list()


def test_a_sweep_whose_first_record_never_landed_is_no_sweep_to_read(tmp_path):
    """A `record/` with no file in it reached polars, which raised a `ValueError` about concatenating nothing."""
    out = tmp_path / 'spill'
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=out)
    for written in (out / 'record').glob('*.parquet'):
        written.unlink()
    with pytest.raises(sps.errors.LayoutError, match="holds no slice under 'record'"):
        sps.read_sweep(out)


def test_a_carry_is_refused_before_a_single_source_is_read(tmp_path):
    """ "Early" has to mean before the data, not merely before the solve.

    The unreadable path is the assertion: reaching it at all means the check
    ran after the data was read.
    """
    missing = tmp_path / 'not-written-yet.parquet'
    sources = {**span_sources(), 'load': str(missing)}

    with pytest.raises(sps.errors.SpecsolveError, match='does not declare'):
        sps.solve_over(WINDOW, sources, replace(WINDOW_AXIS, carry={'soc_initial': 'nope'}))

    with pytest.raises(Exception, match='not-written-yet') as raised:
        sps.solve_over(WINDOW, sources, replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))
    assert not isinstance(raised.value, sps.errors.SpecsolveError), 'the file, not the carry, is what failed'


# ---------------------------------------------------------------------------
# the fold's own rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    'axis',
    [
        pytest.param(sps.EachLabel('period', carry={'existing': 'total'}), id='label'),
        pytest.param(replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}), id='window'),
    ],
)
def test_an_axis_that_carries_is_still_hashable(axis):
    """An axis is a frozen value, so a carry, which is a mapping, stays out of its hash."""
    assert hash(axis) == hash(replace(axis, carry={})), 'the carry is left out of the hash'
    assert axis != replace(axis, carry={}), 'and kept in the equality'


@pytest.mark.parametrize(
    'axis',
    [
        pytest.param(sps.EachLabel('period'), id='label'),
        pytest.param(sps.EachLabel('period', carry={'existing': 'total'}), id='label-carrying'),
        pytest.param(sps.EachLabel('period', key_name='year'), id='label-naming-its-key'),
        pytest.param(replace(WINDOW_AXIS, key_name='window'), id='window-naming-its-key'),
        pytest.param(WINDOW_AXIS, id='window'),
        pytest.param(sps.EachWindow('snapshot', horizon=(2, 4), lookahead=1), id='window-of-listed-horizons'),
        pytest.param(replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}), id='window-carrying'),
    ],
)
def test_an_axis_reads_back_from_the_json_of_its_dict(axis):
    """``to_dict`` is plain data: through JSON and back, it is the axis it was."""
    assert axis_from(json.loads(json.dumps(axis.to_dict()))) == axis


def test_an_axis_that_carries_nothing_writes_no_carry_key():
    """The ``axis.json`` of an archive whose sweep carried nothing holds no ``carry``, as ``INPUTS_LAYOUT`` has it."""
    assert sps.EachLabel('period').to_dict() == {'each': 'label', 'dim': 'period'}, (
        'no carry key where nothing is carried'
    )


def test_solve_over_takes_an_axis_as_its_dict():
    """The dict stands where the axis does, carry included."""
    axis = replace(WINDOW_AXIS, carry={'soc_initial': 'soc'})
    given = sps.solve_over(WINDOW, span_sources(), json.loads(json.dumps(axis.to_dict())))
    expected = sps.solve_over(WINDOW, span_sources(), axis)
    assert given.records['objective'].to_list() == expected.records['objective'].to_list()
    assert given.primal('soc').equals(expected.primal('soc')), 'each window started from the one before it'


def test_an_archive_holds_the_dict_of_its_axis(tmp_path):
    """``axis.json`` is ``to_dict``, so the archive's file and the public form are one."""
    axis = replace(WINDOW_AXIS, carry={'soc_initial': 'soc'})
    sps.solve_over(WINDOW, span_sources(), axis, archive=tmp_path / 'roll')
    assert json.loads((tmp_path / 'roll' / 'axis.json').read_text()) == axis.to_dict()


@pytest.mark.parametrize(
    ('given', 'expected'),
    [
        pytest.param({'dim': 'snapshot'}, "under 'each', one of \\['label', 'window'\\] \\(got None\\)", id='no-each'),
        pytest.param({'each': 'slice', 'dim': 'snapshot'}, "\\(got 'slice'\\)", id='an-each-that-names-no-axis'),
        pytest.param(
            {'each': 'label', 'dim': 'period', 'carries': {}},
            "takes 'each' and \\['dim', 'carry', 'key_name'\\].*\\['carries'\\] is not one of them",
            id='a-key-the-axis-does-not-take',
        ),
        pytest.param(
            {'each': 'window', 'dim': 'snapshot', 'horizon': 4},
            "\\['dim', 'horizon', 'lookahead'\\] are required: \\['lookahead'\\] is missing",
            id='a-key-the-axis-needs',
        ),
    ],
)
def test_an_axis_dict_that_names_no_axis_is_refused_with_the_keys_it_takes(given, expected):
    with pytest.raises(sps.errors.SpecsolveError, match=expected):
        sps.solve_over(WINDOW, span_sources(), given)


def test_carry_and_executor_are_refused_together():
    """Sequential by definition, so the combination is a call-time error rather
    than something discovered at slice two."""
    with pytest.raises(sps.errors.SpecsolveError, match='mutually exclusive'):
        sps.solve_over(
            WINDOW,
            span_sources(),
            replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}),
            executor=object(),
        )


class Inline:
    """The whole protocol `solve_over` needs, in nine lines.

    ``executor=`` takes any :class:`concurrent.futures.Executor`, not only a
    `ProcessPoolExecutor`, so a dask ``Client`` or any other pool plugs in.
    """

    def submit(self, fn, /, *args, **kwargs):
        future: Future = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:  # a pool reports through the future, never raises
            future.set_exception(exc)
        return future


def _process_pool(method: str):
    return ProcessPoolExecutor(2, mp_context=multiprocessing.get_context(method))


@contextlib.contextmanager
def _entered(pool):
    """The live executor: entered where it is a real pool, taken as-is for `Inline`, which has no ``__enter__``."""
    if hasattr(pool, '__enter__'):
        with pool as live:
            yield live
    else:
        yield pool


#: Every executor shape the docs name. `fork` is absent: polars' thread pool
#: does not survive it, and a forked worker *hangs* rather than failing.
EXECUTORS = [
    pytest.param(Inline, id='inline-protocol'),
    pytest.param(lambda: ThreadPoolExecutor(2), id='threads'),
    pytest.param(lambda: _process_pool('spawn'), id='processes-spawn'),
    pytest.param(
        lambda: _process_pool('forkserver'),
        id='processes-forkserver',
        marks=pytest.mark.skipif(
            'forkserver' not in multiprocessing.get_all_start_methods(),
            reason='forkserver is not available on this platform',
        ),
    ),
]


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_gives_the_same_answers_in_the_same_order(make_executor):
    """One fold, four pools, one answer — and the sequential run is the oracle.

    Same numbers *and* the same order. Futures complete out of order, so a
    sweep that read them by completion would reorder itself run to run.
    """
    sources = scenario_sources()
    sequential = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'))

    with _entered(make_executor()) as live:
        parallel = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'), executor=live)

    assert parallel.keys == sequential.keys
    assert answer_of(parallel).equals(answer_of(sequential))
    assert parallel.primal('p').equals(sequential.primal('p'))


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_carries_expressions_the_same(make_executor):
    """Expression tables cross the wire the way primals do — encoded and back.

    The process pools are the point: a thread pool never encodes, so only they
    exercise `_encode`/`_decode` on the expression tables a worker returns.
    """
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    sources = scenario_sources()
    sequential = sps.solve_over(spec, sources, sps.EachLabel('scenario'))

    with _entered(make_executor()) as live:
        parallel = sps.solve_over(spec, sources, sps.EachLabel('scenario'), executor=live)

    assert parallel.evaluate('spend').equals(sequential.evaluate('spend')), (
        'a sweep reads the same named expression under any executor'
    )


def test_a_thread_pool_does_not_encode_for_a_boundary_it_never_crosses(monkeypatch):
    """In-process, so a parquet round trip would be paid for nothing: 31% of a thread-pool sweep, measured.

    Every other executor is assumed to cross, because none of them can be asked.
    """
    seen: list[str] = []
    original = driver._encode

    def spy(sources, memo, **kwargs):
        seen.append('encoded')
        return original(sources, memo, **kwargs)

    monkeypatch.setattr(driver, '_encode', spy)
    with ThreadPoolExecutor(2) as pool:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=pool)
    assert seen == [], 'a thread pool encoded its sources'

    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=pool)
    assert seen, 'a process pool did not encode its sources'


def test_the_model_and_its_plan_both_cross_a_process():
    """A worker is handed the document and the compiled program, under every executor.

    A slice reads no file: it is handed the `Spec`.
    """
    spec = to_spec(DISPATCH)
    serial = sps.solve_over(spec, scenario_sources(), sps.EachLabel('scenario'))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        pooled = sps.solve_over(spec, scenario_sources(), sps.EachLabel('scenario'), executor=pool)
    assert answer_of(pooled).equals(answer_of(serial))
    assert pooled.primal('p').equals(serial.primal('p'))


def test_a_failing_slice_reports_the_real_error_across_a_process_boundary():
    """An exception has to survive pickling or the cause is lost.

    A custom ``__init__`` signature is the classic way this breaks, and it
    shows up as an unrelated ``TypeError`` raised while *unpickling* — so the
    worker's real complaint never arrives.
    """
    broken = {**scenario_sources()}
    broken.pop('cost')
    with (
        ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool,
        pytest.raises(sps.errors.DataError, match="no data provided for parameter 'cost'"),
    ):
        sps.solve_over(DISPATCH, broken, sps.EachLabel('scenario'), executor=pool)


def test_a_parquet_path_slices_without_being_read_whole(tmp_path):
    """A path source is scanned, so the per-slice filter pushes into the file."""
    sources = scenario_sources()
    path = tmp_path / 'load.parquet'
    table = sources.pop('load')
    assert isinstance(table, pl.DataFrame)
    table.write_parquet(path)

    runs = sps.solve_over(DISPATCH, {**sources, 'load': str(path)}, sps.EachLabel('scenario'))
    assert runs.keys == ['high', 'low', 'mid']
    assert runs.primal('p').height == 3 * 4 * 2


def test_a_path_stays_a_path_for_a_local_pool_and_travels_as_bytes_for_a_remote_one(tmp_path, monkeypatch):
    """`shared_filesystem` is inferred from the pool, and only paths are affected.

    A `ProcessPoolExecutor`'s workers are this machine's, so a path stays a
    path. An executor this package did not ship could be anywhere, so its paths
    travel as their own bytes. `shared_filesystem=` says it outright when the
    guess is wrong.
    """
    sources = scenario_sources()
    path = tmp_path / 'p_max.parquet'
    table = sources.pop('p_max')
    assert isinstance(table, pl.DataFrame)
    table.write_parquet(path)
    sources['p_max'] = str(path)

    crossed: list[object] = []
    original = driver._encode

    def spy(sliced, memo, **kwargs):
        """Record what `p_max` crossed as; the same helper also encodes answers back."""
        encoded = original(sliced, memo, **kwargs)
        if 'p_max' in encoded:
            crossed.append(encoded['p_max'])
        return encoded

    monkeypatch.setattr(driver, '_encode', spy)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        local = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'), executor=pool)
        assert all(v == str(path) for v in crossed), 'a local pool shipped a file it could have opened'

        crossed.clear()
        remote = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'), executor=pool, shared_filesystem=False)
        assert all(v == path.read_bytes() for v in crossed), 'the file did not travel as its own bytes'

    assert answer_of(remote).equals(answer_of(local)), 'the path and the bytes are the same numbers'
    assert remote.primal('p').equals(local.primal('p'))

    crossed.clear()
    sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'), executor=Inline())
    assert all(v == path.read_bytes() for v in crossed), 'an executor we did not ship was assumed local'


# ---------------------------------------------------------------------------
# reading a sweep back — Result's readers, one dimension wider
# ---------------------------------------------------------------------------


def test_the_readers_mirror_result_with_the_slice_key_as_one_more_dimension(sweep):
    """Every reader here is `Result`'s under the same name, with the slice key as one more dimension."""
    pytest.importorskip('xarray')
    runs = sweep

    pandas_table = runs.to_pandas('p')
    assert list(pandas_table.columns) == ['scenario', 'snapshot', 'generator', 'value']
    assert len(pandas_table) == 3 * 4 * 2

    array = runs.to_dataarray('p')
    assert array.name == 'p'
    assert array.dims == ('scenario', 'snapshot', 'generator')
    assert array.shape == (3, 4, 2)
    assert array.sel(scenario='low', generator='wind').shape == (4,), (
        'the slice key is an ordinary coordinate, which is the whole point'
    )

    dataset = runs.to_dataset()
    assert set(dataset.data_vars) == {'p'}
    assert dataset['p'].dims == ('scenario', 'snapshot', 'generator')


def test_every_bridge_takes_a_kind_on_a_sweep(priced):
    """The same `kind=` on a sweep's bridges, `per_window` included: a
    stitched price comes back as an array over time, and a dataset of every
    expression per window is one call."""
    pytest.importorskip('xarray')
    price = priced.to_dataarray('balance', 'dual')
    assert price.dims == ('snapshot',), 'the answer is over the dimension the axis sliced'
    assert price.name == 'balance'
    spent = priced.to_dataset(kind='expression', per_window=True)
    assert set(spent.data_vars) == {'spend', 'window_spend'}, 'every expression the slices evaluated'
    assert spent['spend'].dims == ('snapshot_start', 'snapshot'), 'per window, keyed by where each window started'
    assert priced.to_dataset('spend', kind='expression')['spend'].dims == ('snapshot',), (
        'and the answer of one that keeps the windowed dimension is over the dimension the axis sliced'
    )
    with pytest.raises(sps.errors.SpecsolveError, match='primal, dual, expression'):
        priced.to_pandas('soc', 'objective')


def test_save_writes_what_a_spill_writes_and_the_directory_reads_back_as_one(priced, builds, tmp_path):
    """`save` is the spill after the fact: the same layout, all three
    kinds and the record, so `scan` reads it and the same call pointed at it
    with `spill_to=` reads it back without solving a slice."""
    out = priced.save(tmp_path / 'sweep')
    assert out == tmp_path / 'sweep', 'the directory comes back, not a dict nobody indexes'
    assert sorted(p.name for p in out.iterdir()) == [
        'dual',
        'expression',
        'format.json',
        'horizon.parquet',
        'keys.parquet',
        'metrics',
        'primal',
        'record',
        'spec.yaml',
        'sweep.json',
    ], 'the three kinds, the record, the spec, the manifest, the keys, the layout it is in, and the way back'
    assert sorted(p.name for p in (out / 'expression').iterdir()) == ['spend', 'window_spend'], (
        'every declared expression the slices evaluated'
    )

    built = builds(driver)
    reopened = _spilled(out)
    assert built == [], 'a directory the export wrote is a sweep already done'
    assert reopened.records.equals(priced.records)
    assert reopened.scan('soc').collect().equals(priced.primal('soc'))
    assert reopened.scan('balance', 'dual').collect().equals(priced.dual('balance'))
    assert reopened.scan('spend', 'expression').collect().equals(priced.evaluate('spend'))


def test_a_sweep_keys_every_file_it_writes_with_one_type(priced, tmp_path):
    """One key, one dtype, or the files a sweep writes are not one table.

    `pl.lit` reads an int as `Int32` where inference from a Python value gives
    `Int64`; polars joins across the two, but a concatenation of them is
    refused.
    """
    out = priced.save(tmp_path / 'sweep')
    keyed = {
        str(file.relative_to(out)): pl.read_parquet_schema(file)[priced.key_name]
        for file in sorted(out.rglob('*.parquet'))
        if priced.key_name in pl.read_parquet_schema(file)
    }
    assert len(set(keyed.values())) == 1, f'one type for {priced.key_name!r}, and these files disagree: {keyed}'
    assert set(keyed.values()) == {pl.Int64}, 'the type a Python int infers to everywhere else here'


def test_a_resume_checks_the_layout_it_is_extending_rather_than_restamping_it(tmp_path) -> None:
    """`spill_to=` at a directory an earlier build wrote is the one place the stamp has to hold."""
    out = tmp_path / 'sweep'
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=out)
    (out / 'format.json').write_text(json.dumps({'layout': 99}))

    with pytest.raises(sps.errors.LayoutError, match='layout 99'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=out)
    assert json.loads((out / 'format.json').read_text()) == {'layout': 99}, (
        'and the stamp it was refused over is left as it was found'
    )


class _CrashError(Exception):
    """The process dying while the spill directory is being stamped."""


@pytest.mark.parametrize('lost', ['keys.parquet', 'horizon.parquet'])
def test_a_spill_stamped_halfway_is_stamped_again_rather_than_resumed(monkeypatch, tmp_path, lost) -> None:
    """``sweep.json`` lands after the files a scan reads beside it.

    It was written first, so a crash before ``keys.parquet`` or
    ``horizon.parquet`` left a directory the next run took for a stamped one and
    resumed, and ``scan_sweep`` of it failed on the missing file.
    """
    write_whole = sweep_module.write_whole

    def crashing(table: pl.DataFrame, path) -> None:
        if path.name == lost:
            raise _CrashError
        write_whole(table, path)

    out = tmp_path / 'sweep'
    with monkeypatch.context() as patched:
        patched.setattr(sweep_module, 'write_whole', crashing)
        with pytest.raises(_CrashError):
            sps.solve_over(WINDOW, span_sources(8), WINDOW_AXIS, spill_to=out)

    ran = sps.solve_over(WINDOW, span_sources(8), WINDOW_AXIS, spill_to=out)
    assert sps.scan_sweep(out).records.equals(ran.records)


def _typed_apart(in_load: object, in_cost: object) -> dict[str, object]:
    """Dispatch sources whose `load` and `cost` each carry one `draw`, typed as the two labels are.

    A label given as a one-row series keeps that series' type.
    """
    load = pl.DataFrame({'draw': [in_load] * 4, 'snapshot': range(4), 'value': [5.0, 6.0, 7.0, 8.0]})
    drawn = in_cost.extend_constant(in_cost[0], 1) if isinstance(in_cost, pl.Series) else [in_cost] * 2
    cost = pl.DataFrame({'draw': drawn, 'generator': GENERATORS, 'value': [1.0, 50.0]})
    return {**STATIC, 'snapshot': range(4), 'load': load, 'cost': cost}


def test_a_row_with_no_label_in_the_sliced_column_is_refused_by_name():
    """A null among the labels reached the sort that orders the keys, which raised a bare `TypeError`."""
    sources = scenario_sources()
    unlabelled = sources['load'].with_columns(
        pl.when(pl.col('scenario') == 'mid').then(None).otherwise(pl.col('scenario')).alias('scenario')
    )
    with pytest.raises(sps.errors.DataError, match="source 'load' holds a null 'scenario'") as refused:
        sps.solve_over(DISPATCH, {**sources, 'load': unlabelled}, sps.EachLabel('scenario'))
    assert "drop_nulls('scenario')" in str(refused.value), 'the refusal names the rewrite'


AWARE = datetime.datetime(2020, 1, 1, tzinfo=datetime.UTC)


@pytest.mark.parametrize(
    ('in_load', 'in_cost', 'kinds'),
    [
        pytest.param(1, 'a', 'as str and int', id='an-int-and-a-str'),
        pytest.param(AWARE, datetime.datetime(2021, 1, 1), 'as datetime and datetime in UTC', id='aware-and-naive'),
        pytest.param(datetime.date(2020, 1, 1), datetime.datetime(2021, 1, 1), 'as datetime and Date', id='a-date'),
        pytest.param(1, 2.5, 'as float and int', id='an-int-and-a-float'),
        pytest.param(0, True, 'as bool and int', id='an-int-and-a-bool'),
    ],
)
def test_two_sources_that_type_the_axis_apart_are_refused_by_name(in_load, in_cost, kinds):
    """Two sources carried the sliced column as two types, and the sweep failed wherever the pair first clashed.

    An int beside a str, a naive datetime beside an aware one and a date
    beside a datetime raised a bare `TypeError` from sorting the labels. An
    int beside a float or a bool reached a refusal about keys that named
    neither source. The schemas say it before a label is read.
    """
    with pytest.raises(sps.errors.DataError, match=rf"sources 'cost' and 'load' carry 'draw' {kinds}") as refused:
        sps.solve_over(DISPATCH, _typed_apart(in_load, in_cost), sps.EachLabel('draw'))
    assert "Cast 'draw' in one of them" in str(refused.value), 'the refusal names the rewrite'


@pytest.mark.filterwarnings('ignore:.* has no rows for draw')
@pytest.mark.parametrize(
    ('in_load', 'in_cost', 'match'),
    [
        pytest.param(float('nan'), float('nan'), "'nan' names more than one slice", id='two nan, unequal yet one text'),
        pytest.param(
            -1,
            pl.Series([2**63 + 1], dtype=pl.UInt64),
            'fit no one column type',
            id='an int below what unsigned holds beside one above what signed does',
        ),
    ],
)
def test_keys_that_cannot_key_one_sweep_are_refused_before_a_slice_is_taken(tmp_path, in_load, in_cost, match):
    """The record and metrics name a slice by its key's text, and every file writes the key as one type.

    Two sources that each carry a NaN give two keys of one text, which failed
    in polars' ``replace_strict`` when the fold ended — after every slice had
    solved — and again on every ``scan_sweep`` of the spill. Integers that
    span more than any one integer type holds have no type to be written as.
    """
    out = tmp_path / 'spill'
    with pytest.raises(sps.errors.SpecsolveError, match=match):
        sps.solve_over(DISPATCH, _typed_apart(in_load, in_cost), sps.EachLabel('draw'), spill_to=out)
    assert not out.exists(), 'refused before the spill is opened, so before any slice is solved'


def test_a_saved_result_carries_the_row_a_sweep_keys(sweep, tmp_path):
    """One solve's record is one slice's, so cases solved apart concatenate."""
    sources = scenario_sources()
    low = {**sources, 'load': sources['load'].filter(pl.col('scenario') == 'low').drop('scenario')}
    with sps.solve(DISPATCH, low) as alone:
        one = pl.read_parquet(alone.save(tmp_path / 'low') / 'record.parquet')

    assert one.columns == [column for column in sweep.records.columns if column != sweep.key_name], (
        'the fold keys the record it writes; a lone solve writes the same columns unkeyed'
    )
    row = one.row(0, named=True)
    slice_of_the_fold = sweep.records.filter(pl.col('scenario') == 'low').drop('scenario').row(0, named=True)
    assert (row['status'], row['termination_condition']) == (
        slice_of_the_fold['status'],
        slice_of_the_fold['termination_condition'],
    ), 'the lone solve and the slice of the fold terminated the same way'
    assert row['objective'] == pytest.approx(slice_of_the_fold['objective']), (
        'and reached the same number, the two being the same model over the same numbers'
    )


def test_a_bulk_export_of_a_sweep_that_solved_nothing_is_refused():
    """`to_dataset` writes no empty answer: a sweep every slice of which was
    infeasible holds no variable tables, and it refuses with the sentence
    `primal` gives. The names are resolved before xarray is reached, so a bare
    install gets the sentence rather than an ImportError."""
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'))

    with pytest.raises(sps.errors.SpecsolveError, match='holds no variable tables at all'):
        runs.to_dataset()


def test_a_sweep_that_solved_nothing_still_saves_its_records(tmp_path):
    """`save` is not an export, and an infeasible study is an answer.

    A single solve that left no values writes its record and no tables, and a
    sweep of them does the same.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario'))

    out = runs.save(tmp_path / 'sweep')
    records = pl.read_parquet(sorted((out / 'record').glob('*.parquet')))
    assert records['termination_condition'].unique().to_list() == ['infeasible'], (
        'every slice terminated infeasible, and the record says so'
    )
    assert not (out / 'primal').exists(), 'and no tables are written, there being none'
    assert sps.read_sweep(out).records.height == records.height, 'the saved study reads back'


@pytest.mark.parametrize('lost', ['record', 'metrics'], ids=str)
def test_a_sweep_directory_missing_its_record_is_refused_by_name(lost: str, tmp_path) -> None:
    """A manifest with no record beside it is not a sweep this package wrote.

    Both are written per slice as the fold goes, so a directory holding one
    and not the other was edited or interrupted.
    """
    out = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path / 'sweep')
    assert sps.read_sweep(out._spill.directory).records.height == 3, 'the whole one reads back first'
    shutil.rmtree(out._spill.directory / lost)

    with pytest.raises(sps.errors.LayoutError, match=f"holds no slice under '{lost}'"):
        sps.read_sweep(out._spill.directory)


def test_a_sweep_read_whole_is_held_and_a_scanned_one_is_spilled(tmp_path):
    """The two verbs give the same study and differ in where its tables are.

    `scan_sweep` leaves the tables on disk, so it owes the directory every
    read. `read_sweep` reads them in, so the directory is free afterwards.
    """
    spilled = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path / 'sweep')
    expected = spilled.scan('p').collect()
    loaded = sps.read_sweep(tmp_path / 'sweep')
    scanned = sps.scan_sweep(tmp_path / 'sweep')

    assert scanned.primal('p').equals(expected), 'both read the study the spill wrote'
    shutil.rmtree(tmp_path / 'sweep')

    assert loaded.primal('p').equals(expected), 'the held sweep answers off no directory at all'
    with pytest.raises(FileNotFoundError):
        scanned.primal('p')
    assert loaded.keys == scanned.keys, 'and is keyed as the sweep was solved either way'


def test_a_reader_for_a_name_the_sweep_lacks_fails_the_way_primal_does(sweep):
    """One explanation, reached through every reader."""
    for read in (sweep.to_pandas, sweep.to_dataarray):
        with pytest.raises(sps.errors.SpecsolveError, match="no variable 'q' in this sweep"):
            read('q')


def test_an_axis_key_name_overrides_what_it_derived_and_refuses_a_collision():
    """The derived name is right by default and the caller's word wins.

    A key that is a declared dimension would collide with a column the tables
    carry.
    """
    runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario', key_name='case'))
    assert runs.records.columns[0] == 'case'
    assert set(runs.primal('p').columns) == {'case', 'snapshot', 'generator', 'value'}

    with pytest.raises(sps.errors.SpecsolveError, match=r"key_name='generator' is a dimension the spec declares"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario', key_name='generator'))


def test_a_window_key_name_names_the_column_each_window_is_read_under():
    """A window's key is where it started, called `<dim>_start` unless the axis names it."""
    named = sps.solve_over(WINDOW, span_sources(8), replace(WINDOW_AXIS, key_name='window'))
    assert named.primal('soc', per_window=True).columns == ['window', 'snapshot', 'value'], (
        'the key column takes the name the axis gives it'
    )
    assert named.primal('soc').columns == ['snapshot', 'value'], 'and the stitched answer carries no key'


def test_an_archive_runs_again_under_the_key_name_the_sweep_ran_with(tmp_path):
    """The key name was an argument of the call, so the axis an archive held ran again keyed by its default."""
    axis = sps.EachLabel('scenario', key_name='draw')
    sps.solve_over(DISPATCH, scenario_sources(), axis, archive=tmp_path / 'study')
    study = sps.read_archive(tmp_path / 'study')
    assert study.axis == axis, 'the axis reads back with the key name it was given'
    assert sps.solve_over(study.spec, study.sources, study.axis).key_name == 'draw'


@pytest.mark.parametrize(
    'over',
    [
        pytest.param([('low', {}), ('high', {})], id='a-list-of-pairs'),
        pytest.param('scenario', id='the-name-of-a-dimension'),
    ],
)
def test_an_over_that_is_no_axis_is_refused_before_anything_is_read(over):
    """Anything else reached the first attribute an axis has, and raised an `AttributeError` from inside."""
    with pytest.raises(sps.errors.SpecsolveError, match=r'over= takes EachLabel or EachWindow') as raised:
        sps.solve_over(DISPATCH, scenario_sources(), over)
    assert f'(got {type(over).__name__})' in str(raised.value), 'the refusal names what it was given'


@pytest.mark.parametrize('key_name', ['specsolve_case', 'Specsolve_case', 'SPECSOLVE_CASE'], ids=str)
def test_a_slice_key_with_the_reserved_prefix_is_refused_in_any_letter_case(key_name):
    """A capital passed the reserved prefix, though a query engine reads `Specsolve_run` as `specsolve_run`."""
    with pytest.raises(sps.errors.SpecsolveError, match=rf"key_name='{key_name}' starts with 'specsolve_'"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario', key_name=key_name))


def test_duals_come_back_keyed_by_slice_and_are_never_combined(sweep):
    """A shadow price belongs to the slice that priced it; the reduction is the caller's."""
    runs = sweep

    prices = runs.dual('balance')
    assert prices.columns[0] == runs.key_name, 'the key comes first, as it does for a primal'
    assert set(prices[runs.key_name].unique()) == set(runs.keys)
    assert prices.height == runs.primal('p').height // 2, 'one price per row, not per column'


def test_a_slice_without_duals_does_not_fail_the_sweep():
    """An integer variable leaves duals undefined, and that is one slice's news.

    The sweep still returns, and asked for a price it says what a single solve
    says: which variable is not continuous.
    """
    integral = override(DISPATCH, **{'variables.p.domain': 'integer'})
    runs = sps.solve_over(integral, scenario_sources(), sps.EachLabel('scenario'))

    assert len(runs) == 3, 'every slice is still a row of the record'
    assert runs.primal('p').height > 0, 'primals are unaffected'
    with pytest.raises(sps.errors.SpecsolveError, match='duals are undefined for a mixed-integer model') as raised:
        runs.dual('balance')
    assert "'p' is not continuous" in str(raised.value), 'the sweep names the variable, as one solve does'


def test_a_bad_name_is_reported_without_the_optional_dependency(sweep):
    """`to_pandas` answers about the model before it asks about the environment.

    A bad name reads the same on every install; a name the sweep does hold
    still needs the dependency, and says which package to install.
    """
    with mock.patch.dict(sys.modules, {'pandas': None}):
        with pytest.raises(sps.errors.SpecsolveError, match="no variable 'q' in this sweep"):
            sweep.to_pandas('q')
        with pytest.raises(ModuleNotFoundError, match='pip install pandas'):
            sweep.to_pandas('p')


# ---------------------------------------------------------------------------
# the model is asked before it is sliced
# ---------------------------------------------------------------------------


def _window_with(constraint: dict, **parameters: dict) -> dict:
    """`WINDOW` with one more constraint over `snapshot`, and any parameter it reads."""
    return override(
        WINDOW,
        parameters={**WINDOW['parameters'], **parameters},
        constraints={**WINDOW['constraints'], 'extra': constraint},
    )


def test_a_budget_over_the_whole_windowed_dimension_is_refused_with_the_change_that_would_lift_it():
    spec = _window_with({'dims': [], 'expression': 'sum(discharge, over=snapshot) <= 100'})
    with pytest.raises(sps.errors.SpecsolveError, match=r"constraint 'extra': sums over snapshot") as refused:
        sps.solve_over(spec, span_sources(8), WINDOW_AXIS)
    assert 'sum_back(window=n)' in str(refused.value), 'the refusal names the rolling form that windows'


def test_a_window_must_look_ahead_as_far_as_the_rows_read():
    """`shift(load, along=snapshot, offset=-2)` reads two rows ahead; a contiguous
    window would read past its end, an overlap of two covers it."""
    spec = _window_with(
        {'dims': ['snapshot'], 'expression': 'sum(p, over=generator) >= shift(load, along=snapshot, offset=-2, edge=0)'}
    )
    with pytest.raises(sps.errors.SpecsolveError, match=r'looks ahead by 0 label\(s\), and the model reads 2 ahead'):
        sps.solve_over(spec, span_sources(8), sps.EachWindow('snapshot', horizon=4, lookahead=0))
    runs = sps.solve_over(spec, span_sources(8), sps.EachWindow('snapshot', horizon=4, lookahead=2))
    assert runs.keys == [0, 4], 'with the lookahead covered, every window solves'


@pytest.mark.parametrize(
    ('delays', 'axis', 'refused'),
    [
        pytest.param(
            [1, 2],
            sps.EachWindow('snapshot', horizon=4, lookahead=0),
            False,
            id='a-delay-behind-needs-no-overlap',
        ),
        pytest.param(
            [-1, -3],
            sps.EachWindow('snapshot', horizon=4, lookahead=0),
            True,
            id='a-delay-ahead-needs-the-overlap',
        ),
        pytest.param(
            [-1, -3],
            sps.EachWindow('snapshot', horizon=4, lookahead=3),
            False,
            id='and-an-overlap-of-three-covers-it',
        ),
    ],
)
def test_an_offset_the_data_decides_is_read_off_the_data(delays, axis, refused):
    """`shift(..., offset=delay)` names a parameter, so the language cannot say
    how far a row reads; the driver reads the values, whose sign says which way."""
    spec = _window_with(
        {'dims': ['snapshot', 'generator'], 'expression': 'p >= shift(p, along=snapshot, offset=delay, edge=0) - 100'},
        delay={'dims': ['generator'], 'dtype': 'int'},
    )
    sources = {**span_sources(8), 'delay': pl.DataFrame({'generator': GENERATORS, 'value': delays})}
    if refused:
        with pytest.raises(sps.errors.SpecsolveError, match='the model reads 3 ahead'):
            sps.solve_over(spec, sources, axis)
    else:
        assert len(sps.solve_over(spec, sources, axis)) == 2, 'every window solved'


def test_a_reach_a_relation_decides_is_refused_with_the_relation_named():
    """`shift(..., within=day_of[day])` reaches within the groups the relation makes, and
    whether a window cuts a group is nothing the driver computes."""
    spec = _window_with(
        {
            'dims': ['snapshot', 'generator'],
            'expression': 'p >= shift(p, along=snapshot, offset=1, within=day_of[day], edge=0) - at(day_cap, by=day_of[day])',
        },
        day_cap={'dims': ['day']},
    )
    spec['dimensions'] = {**spec['dimensions'], 'day': {'dtype': 'int', 'ordered': True}}
    spec['relations'] = {'day_of': {'key': 'snapshot', 'values': 'day'}}
    with pytest.raises(sps.errors.SpecsolveError, match=r"constraint 'extra': through the relation 'day_of'"):
        sps.solve_over(spec, span_sources(8), WINDOW_AXIS)


def test_a_position_counts_within_each_window_and_nothing_warns():
    """`position(snapshot) == 0` is each window's first row, which is how an initial condition is written."""
    spec = _window_with({'dims': ['snapshot'], 'where': 'position(snapshot) == 0', 'expression': 'soc <= 50'})
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        runs = sps.solve_over(spec, span_sources(8), WINDOW_AXIS)
    assert runs.dual('extra', per_window=True)['snapshot'].to_list() == [0, 4], 'one row at the start of each window'


@pytest.mark.parametrize(
    'delay',
    [
        pytest.param(-3.0, id='one-number-for-every-generator'),
        pytest.param({'wind': -3, 'gas': -1}, id='a-map-from-label-to-value'),
        pytest.param([-3, -1], id='a-sequence-in-label-order'),
        pytest.param(pl.DataFrame({'generator': GENERATORS, 'value': [-3, -1]}), id='a-tidy-table'),
    ],
)
def test_an_offset_is_read_off_every_shape_a_source_may_arrive_in(delay):
    """The reach is the same whatever the caller wrote, because the least value
    of a source does not depend on the labels it is spread over."""
    spec = _window_with(
        {'dims': ['snapshot', 'generator'], 'expression': 'p >= shift(p, along=snapshot, offset=delay, edge=0) - 100'},
        delay={'dims': ['generator'], 'dtype': 'int'},
    )
    with pytest.raises(sps.errors.SpecsolveError, match='the model reads 3 ahead'):
        sps.solve_over(spec, {**span_sources(8), 'delay': delay}, WINDOW_AXIS)


def test_a_window_over_a_dimension_the_spec_does_not_declare_is_refused_by_name():
    with pytest.raises(
        sps.errors.SpecsolveError, match=r"EachWindow\('snapshots'\).*Did you mean 'snapshot'"
    ) as refused:
        sps.solve_over(WINDOW, span_sources(8), sps.EachWindow('snapshots', horizon=4, lookahead=0))
    assert 'is sliced with EachLabel' in str(refused.value), (
        'the refusal names the axis for a dimension the spec does not declare'
    )


def test_a_window_over_a_dimension_not_declared_ordered_is_refused():
    """A window is a run of consecutive labels, and only an ordered dimension says which labels are consecutive.

    Nothing in the dispatch model steps along `snapshot`, so the coupling
    check passes it, and the windows would follow the row order of its labels.
    """
    unordered = override(DISPATCH, **{'dimensions.snapshot': {'dtype': 'int'}})
    sources = {
        **STATIC,
        'snapshot': range(4),
        'load': pl.DataFrame({'snapshot': range(4), 'value': [5.0, 6.0, 7.0, 8.0]}),
    }
    with pytest.raises(
        sps.errors.SpecsolveError,
        match=r"EachWindow\('snapshot'\) reads the order of 'snapshot', which is not declared ordered",
    ) as refused:
        sps.solve_over(unordered, sources, sps.EachWindow('snapshot', horizon=2, lookahead=0))
    assert "'snapshot: {ordered: true}'" in str(refused.value), 'the refusal names the rewrite'


def test_a_label_sweep_over_a_dimension_the_spec_declares_is_refused():
    """`EachLabel` squeezes its dimension out of every source, so a declared dimension would be left with no data."""
    with pytest.raises(sps.errors.SpecsolveError, match=r"EachLabel\('generator'\) squeezes 'generator'"):
        sps.solve_over(WINDOW, span_sources(8), sps.EachLabel('generator', key_name='g'))


# ---------------------------------------------------------------------------
# what a first non-toy sweep runs into
# ---------------------------------------------------------------------------


#: Every shape `build` takes for a source that does not carry the axis: a
#: number for a scalar parameter, a bare sequence for a dimension's labels, a
#: `{label: value}` map. None of them is a table: the axis has nothing to filter
#: in them.
NOT_A_TABLE = [
    pytest.param(WINDOW, span_sources, WINDOW_AXIS, {'soc_initial': 0.0}, id='a-number'),
    pytest.param(DISPATCH, scenario_sources, sps.EachLabel('scenario'), {'snapshot': range(4)}, id='bare-labels'),
    pytest.param(DISPATCH, scenario_sources, sps.EachLabel('scenario'), {'cost': {'wind': 1.0, 'gas': 50.0}}, id='a-map'),
]  # fmt: skip


@pytest.mark.parametrize(('spec', 'sources', 'axis', 'plain'), NOT_A_TABLE)
def test_a_sweep_takes_every_source_shape_solve_takes(spec, sources, axis, plain):
    """The same `sources` dict moves from `solve` to `solve_over` unchanged.

    A source that is not a table cannot carry the axis, so it passes through
    untouched — and under a process pool it crosses as itself, since a number
    pickles.
    """
    with_tables = sources()
    as_plain = {**with_tables, **plain}
    runs = sps.solve_over(spec, as_plain, axis)
    assert (
        runs.records['objective'].to_list() == sps.solve_over(spec, with_tables, axis).records['objective'].to_list()
    ), 'a number, a sequence and a map attach exactly as the tables they stand for'
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        pooled = sps.solve_over(spec, as_plain, axis, executor=pool)
    assert answer_of(pooled).equals(answer_of(runs)), 'the plain shapes cross a process as themselves'


def test_a_source_short_of_a_coordinate_of_the_axis_is_reported():
    """`cost`, declared neutral, stops at period 2 while `demand` runs to 3, so period 3 builds with no cost at all.

    The warning comes before a slice is taken, naming the source, the
    coordinate it lacks, and a source that has it.
    """
    sources = myopic_sources()
    sources['cost'] = pl.DataFrame({'period': [1, 1, 2, 2], 'generator': GENERATORS * 2, 'value': [1.0, 50.0] * 2})
    with pytest.warns(sps.errors.SpecsolveWarning, match=r"'cost' has no rows for period 3, which 'demand' has"):
        runs = sps.solve_over(
            override(MYOPIC, **{'parameters.cost.missing': 'neutral'}),
            sources,
            sps.EachLabel('period', carry={'existing': 'total'}),
        )
    assert runs.records['objective'].to_list()[-1] == 0.0, 'the sweep still runs, and period 3 is free'


def test_a_carry_with_no_seed_says_the_first_slice_needs_one():
    """`carry` supplies `soc_initial` from the second slice on; the first has
    nothing to start from, and the error says so rather than reporting a
    parameter with no data as if the carry did not exist.
    """
    sources = span_sources(12)
    del sources['soc_initial']
    with pytest.raises(sps.errors.SpecsolveError, match=r"carry writes 'soc_initial' from the second slice on"):
        sps.solve_over(WINDOW, sources, replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))


def test_a_slice_that_leaves_nothing_to_carry_stops_the_sweep_by_name():
    """One infeasible window under a carry: the next window has no level to
    start from, so the sweep cannot go on — and the error names the slice
    that terminated, how, and the slice left waiting.
    """
    sources = span_sources(12)
    sources['load'] = sources['load'].with_columns(
        pl.when(pl.col('snapshot') == 5).then(10_000.0).otherwise(pl.col('value')).alias('value')
    )
    with pytest.raises(sps.errors.SpecsolveError, match=r'slice 4 .*infeasible') as raised:
        sps.solve_over(WINDOW, sources, replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}))
    assert 'slice 8' in str(raised.value), 'the message names the slice that had nothing to start from'


def _with_a_short_last_scenario() -> tuple[dict[str, object], dict[str, object]]:
    """Scenario sources whose last scenario in key order is `top`, whole and then short of two snapshots."""
    base = scenario_sources()
    whole = base['load'].with_columns(pl.col('scenario').replace('high', 'top'))
    rest = whole.filter(pl.col('scenario') != 'top')
    short = pl.DataFrame({'scenario': ['top', 'top'], 'snapshot': [0, 1], 'value': [1.0, 2.0]})
    return {**base, 'load': whole}, {**base, 'load': pl.concat([rest, short])}


@pytest.mark.parametrize('make_executor', [pytest.param(None, id='serial'), *EXECUTORS[:2]])
def test_a_failing_slice_is_named(make_executor):
    """Slice three of three fails to build, and the traceback says so.

    The error is the engine's own, with a note added, so a caller matching on
    the message still matches.
    """
    with (
        _entered(make_executor() if make_executor else None) as executor,
        pytest.raises(sps.errors.DataError) as raised,
    ):
        sps.solve_over(DISPATCH, _with_a_short_last_scenario()[1], sps.EachLabel('scenario'), executor=executor)
    assert any("slice 'top'" in note and '3 of 3' in note for note in raised.value.__notes__), (
        'the note names the slice by key and by position'
    )


@pytest.mark.parametrize('key_name', ['value', 'status', 'termination_condition', 'objective'])
def test_a_key_that_collides_with_a_fixed_column_is_refused(key_name):
    """`value` collides in every table a reader returns, and the other three
    in `objective` — where the key would silently replace the column rather
    than join it."""
    with pytest.raises(sps.errors.SpecsolveError, match=f'key_name={key_name!r} .* column'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario', key_name=key_name))


@pytest.mark.parametrize('key_name', ['columns', 'rows', 'solves', 'solver_instances', 'build_seconds'], ids=str)
def test_a_key_named_like_a_metrics_column_is_refused_before_a_slice_is_solved(key_name):
    """`Sweep.metrics` carries the key beside every `Metrics` column.

    The guard named only the `Record` columns, so a key named like a metric
    passed it: every slice solved, then polars refused the table with a
    `DuplicateError`, and again on every read of a spilled sweep.
    """
    with pytest.raises(sps.errors.SpecsolveError, match=f'key_name={key_name!r} .* column'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario', key_name=key_name))


@pytest.mark.parametrize('make_executor', EXECUTORS[:2])
def test_a_pooled_sweep_parses_the_spec_once(make_executor, monkeypatch):
    """The spec is parsed once per call, whichever executor runs the slices.

    What a worker receives is the document already read, so no slice reads
    the YAML again. Counted at `to_spec`, where the language reads a file.
    """
    from mathspec import _validation as validation
    from mathspec.spec import Spec

    from specsolve import _inputs as inputs

    parsed: list[object] = []
    original = validation.to_spec

    def spy(spec):
        if not isinstance(spec, Spec):
            parsed.append(spec)
        return original(spec)

    monkeypatch.setattr(validation, 'to_spec', spy)
    monkeypatch.setattr(inputs, 'to_spec', spy)
    with _entered(make_executor()) as executor:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=executor)
    assert len(parsed) == 1, f'the model was parsed {len(parsed)} times for three slices'


def _objective_alone(spec, sources, axis, key, **given) -> float:
    """The objective of one slice, built by `build_slice` and solved alone."""
    with sps.build_slice(spec, sources, axis, key, **given) as model:
        return model.solve().objective


def test_one_slice_of_a_sweep_builds_alone():
    """`build_slice` is the model of one slice the sweep would have run.

    One slice builds alone, and each solved alone reaches the objective its
    window did.
    """
    sources = span_sources(12)
    axis = sps.EachWindow('snapshot', horizon=3, lookahead=3)
    sweep = sps.solve_over(WINDOW, sources, axis)
    assert sweep.keys == [0, 3, 6, 9], 'one slice per window, keyed by where it starts'

    with sps.build_slice(WINDOW, sources, axis.to_dict(), 3) as model:
        assert str(model.row('soc_open', snapshot=3)).startswith('soc_open[snapshot=3]'), (
            'one window builds alone, its axis given as the plain data solve_over takes too'
        )

    alone = [_objective_alone(WINDOW, sources, axis, key) for key in sweep.keys]
    assert alone == pytest.approx(sweep.records['objective'].to_list()), (
        'each window solved alone is the window the sweep solved'
    )


@pytest.mark.parametrize(
    ('over', 'without', 'refusal'),
    [
        pytest.param('snapshot', None, 'over= takes EachLabel or EachWindow', id='no-axis'),
        pytest.param(
            sps.EachLabel('snapshot'),
            None,
            "EachLabel('snapshot') squeezes 'snapshot' out of every source, and the spec declares it",
            id='an-axis-the-spec-does-not-allow',
        ),
        pytest.param(
            sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'}),
            'soc_initial',
            "supply 'soc_initial' in sources as the seed",
            id='a-carry-with-no-seed',
        ),
    ],
)
def test_build_slice_refuses_the_axis_and_the_carry_solve_over_refuses(over, without, refusal):
    """One slice is built as the sweep builds it, so what the sweep refuses of an axis is refused here too."""
    sources = {name: source for name, source in span_sources().items() if name != without}
    with pytest.raises(sps.errors.SpecsolveError, match=re.escape(refusal)):
        sps.build_slice(WINDOW, sources, over, 0)


@pytest.mark.parametrize(
    ('horizon', 'keyed'),
    [
        pytest.param(3, '0, 3, 6, 9', id='every-key-where-they-are-few'),
        pytest.param(1, '0, 1, 2, 3, 4 and 7 more', id='the-first-five-where-they-are-many'),
    ],
)
def test_a_key_that_is_no_slice_is_refused_naming_the_keys(horizon, keyed):
    """A key is where a window starts, so a label inside one names no slice."""
    axis = sps.EachWindow('snapshot', horizon=horizon, lookahead=0)
    refusal = f'key=99 is no slice of this axis, whose {12 // horizon} slices are keyed {keyed}.'
    with pytest.raises(sps.errors.SpecsolveError, match=re.escape(refusal)):
        sps.build_slice(WINDOW, span_sources(), axis, 99)


@pytest.mark.parametrize('make_executor', [pytest.param(None, id='serial'), *EXECUTORS])
def test_a_sweep_reports_what_each_slice_cost(make_executor):
    """`runs.metrics` is one row per slice: the model's size, whether the
    solver was loaded from scratch, and the seconds each phase took —
    `Model.diagnostics()` one dimension wider, the same way the readers are.

    A serial sweep updates one model, so after the first slice the solver is
    pushed values rather than loaded; a pooled sweep builds each slice alone,
    so every one loads.
    """
    with _entered(make_executor() if make_executor else None) as executor:
        runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=executor)
    table = runs.metrics
    assert table.columns == ['scenario', *Metrics._fields], 'the key, then the columns a single solve writes'
    assert table['scenario'].to_list() == runs.keys
    assert table['slice'].to_list() == ['high', 'low', 'mid'], 'the key again, as text, in slice order'
    assert table['slice_axis'].unique().to_list() == ['scenario'], 'every row names the axis it is a slice of'
    assert table['columns'].unique().to_list() == [8], 'every slice is the same model over different numbers'
    assert (table.select(pl.col('attach_seconds', 'build_seconds', 'solve_seconds') >= 0).to_numpy()).all(), (
        'a clock is never negative'
    )
    assert table['solves'].to_list() == [1, 1, 1], "each row is one slice's solve, not the model's running total"
    assert table['solver_instances'].to_list() == ([1, 0, 0] if executor is None else [1] * 3), (
        'a serial sweep loads the solver once and pushes values after; a pooled one builds every slice cold'
    )
    assert table['write_seconds'].to_list() == [0.0] * 3, 'a sweep writes no model file'


# ---------------------------------------------------------------------------
# spilling to disk
# ---------------------------------------------------------------------------

PRICED_AXIS = sps.EachWindow('snapshot', horizon=3, lookahead=3)
PRICED_CARRY = {'soc_initial': 'soc'}


def _spilled(directory, **kwargs) -> sweep_module.Sweep:
    return sps.solve_over(
        SPENDING, span_sources(12), replace(PRICED_AXIS, carry=PRICED_CARRY), spill_to=directory, **kwargs
    )


@pytest.mark.parametrize(
    ('table', 'reader', 'row_type'),
    [pytest.param('record', 'records', Record, id='record'), pytest.param('metrics', 'metrics', Metrics, id='metrics')],
)
def test_a_slices_rows_are_written_in_the_columns_a_single_solve_writes(tmp_path, table, reader, row_type):
    """The spill writes a slice's rows as `Record` and `Metrics` declare them, the slice named in
    `slice_axis` and `slice` rather than in a column of the key's own name, so a glob over sweeps
    and single solves reads one schema."""
    runs = _spilled(tmp_path / 'sweep')
    written = pl.read_parquet(sorted((tmp_path / 'sweep' / table).glob('*.parquet')))

    assert written.columns == list(row_type._fields), 'the columns the type declares, in its order, and no key column'
    assert written['slice_axis'].unique().to_list() == [runs.key_name], 'the axis the sweep is cut on'
    assert written['slice'].to_list() == [str(key) for key in runs.keys], 'each slice key as text'
    held = getattr(runs, reader)
    assert held.columns == [runs.key_name, *written.columns], 'the held table is the written one, keyed'


def test_a_slice_written_in_another_layout_is_refused_by_name(tmp_path):
    """A resume reads a slice's record back as values, so a file short of a column is refused by name."""
    _spilled(tmp_path / 'sweep')
    first = min((tmp_path / 'sweep' / 'metrics').glob('*.parquet'))
    pl.read_parquet(first).drop('solver_instances').write_parquet(first)

    with pytest.raises(sps.errors.LayoutError, match=r"Metrics row that is short of \['solver_instances'\]"):
        _spilled(tmp_path / 'sweep')


def test_a_spilled_sweep_holds_nothing_and_scans_back_what_it_wrote(priced, tmp_path):
    """`spill_to=` writes each slice's tables as the fold goes and keeps none of them.

    What comes back through `scan` is the table the in-memory reader would
    have returned — primal, dual and expression, the answer or per window —
    so the two ways of running a sweep cannot answer differently.
    """
    runs = _spilled(tmp_path / 'sweep')
    assert answer_of(runs).equals(answer_of(priced))
    assert runs.scan('soc').collect().equals(priced.primal('soc'))
    assert runs.scan('balance', 'dual').collect().equals(priced.dual('balance'))
    assert runs.scan('spend', 'expression').collect().equals(priced.evaluate('spend'))
    assert runs.scan('soc', per_window=True).collect().equals(priced.primal('soc', per_window=True))
    assert not list(tmp_path.rglob('*.part')), 'every file landed under its final name'
    shutil.rmtree(tmp_path / 'sweep')
    with pytest.raises(FileNotFoundError):
        runs.primal('soc')


@pytest.mark.parametrize(
    'read',
    [
        pytest.param(lambda runs: runs.primal('soc'), id='primal'),
        pytest.param(lambda runs: runs.dual('balance'), id='dual'),
        pytest.param(lambda runs: runs.evaluate('spend'), id='expression'),
    ],
)
def test_the_table_readers_answer_off_a_spilled_sweep_as_off_a_held_one(read, priced, tmp_path):
    """A spilled sweep's tables lie on disk, and a table reader reads the one
    name it is asked for, so it answers what the sweep held in memory answers."""
    assert read(_spilled(tmp_path)).equals(read(priced))


def test_an_export_reads_a_spilled_sweep_as_a_held_one(priced, tmp_path):
    """`to_dataset` reads each name off the files, as the table readers do."""
    pytest.importorskip('xarray')
    assert _spilled(tmp_path).to_dataset().equals(priced.to_dataset())


def test_a_spilled_sweep_saves_as_the_sweep_it_is(priced, tmp_path):
    """`save` copies a spilled sweep's slices to another directory, as it writes a held one's."""
    saved = _spilled(tmp_path / 'sweep').save(tmp_path / 'copy')
    assert sps.read_sweep(saved).primal('soc').equals(priced.primal('soc'))


def test_scan_reads_an_in_memory_sweep_too(sweep):
    """`scan` means the same thing on both: code written for a spilled sweep
    runs unchanged on one that fit in memory."""
    assert sweep.scan('p').collect().equals(sweep.primal('p'))
    with pytest.raises(sps.errors.SpecsolveError, match="no variable 'nope'"):
        sweep.scan('nope')
    with pytest.raises(sps.errors.SpecsolveError, match='primal, dual, expression'):
        sweep.scan('p', 'objective')


def test_a_spilled_sweep_resumes_after_the_slice_that_failed(builds, tmp_path):
    """The slices that solved before the failure are not solved again.

    Three scenarios, the third of which cannot build; the second run with the
    same directory builds one model, and comes back identical to a sweep that
    never failed.
    """
    good, bad = _with_a_short_last_scenario()
    axis = sps.EachLabel('scenario')
    with pytest.raises(sps.errors.DataError):
        sps.solve_over(DISPATCH, bad, axis, spill_to=tmp_path)

    built = builds(driver)
    resumed = sps.solve_over(DISPATCH, good, axis, spill_to=tmp_path)
    assert len(built) == 1, 'only the slice that failed is built again'

    fresh = sps.solve_over(DISPATCH, good, axis)
    assert answer_of(resumed).equals(answer_of(fresh))
    assert resumed.scan('p').collect().equals(fresh.primal('p'))


#: A knapsack of whole items with a price on its budget, which a mixed-integer solve cannot give.
PRICED_KNAPSACK = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {'x': {'dims': ['f'], 'domain': 'integer', 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'budget_row': {'dims': [], 'expression': 'sum(x, over=f) <= budget'}},
    'expressions': {'price': {'dims': [], 'expression': 'dual(budget_row)'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


def test_a_resumed_sweep_keeps_the_reasons_of_the_slices_it_reads_back(monkeypatch, tmp_path):
    """Why an output is missing was written once, when the fold ended, and a slice read back gives no reason.

    A sweep that stopped after its last record and before that file was
    resumed without solving a slice, so nothing gave the reason again, and a
    reader answered with another sentence. A slice's reasons now land before
    the record that marks it done.
    """
    sources = {
        'f': ['a', 'b'],
        'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}),
        'budget': pl.DataFrame({'draw': [1, 2], 'value': [5.0, 7.0]}),
    }
    out = tmp_path / 'spill'
    write = sweep_module.write_whole

    def stops_after_the_last_record(table, path) -> None:
        write(table, path)
        if (path.parent.name, path.name) == ('record', '000001.parquet'):
            raise _CrashError

    with monkeypatch.context() as patched:
        patched.setattr(sweep_module, 'write_whole', stops_after_the_last_record)
        with pytest.raises(_CrashError):
            sps.solve_over(PRICED_KNAPSACK, sources, sps.EachLabel('draw'), spill_to=out)

    resumed = sps.solve_over(PRICED_KNAPSACK, sources, sps.EachLabel('draw'), spill_to=out)
    for read in (resumed, sps.scan_sweep(out)):
        with pytest.raises(sps.errors.SpecsolveError, match='duals are undefined for a mixed-integer model'):
            read.evaluate('price')


def test_a_resumed_carry_reads_its_state_off_the_disk(monkeypatch, tmp_path):
    """A rolling horizon interrupted after two windows continues from the
    second window's file, and ends where an uninterrupted one does.

    Both start each window from nothing: the resumed run builds the third
    window afresh, so a carried-on solve could end on another vertex of the
    same optimum.
    """
    answered = driver._answers
    seen: list[int] = []

    def two_then_fail(*args):
        if len(seen) == 2:
            raise RuntimeError('the box went away')
        seen.append(1)
        return answered(*args)

    monkeypatch.setattr(driver, '_answers', two_then_fail)
    with pytest.raises(RuntimeError, match='went away'):
        _spilled(tmp_path, start=None)
    monkeypatch.setattr(driver, '_answers', answered)
    assert sps.scan_sweep(tmp_path).keys == [0, 3], 'the interrupted spill reads back keyed by the two it finished'

    resumed = _spilled(tmp_path, start=None)
    cold = sps.solve_over(SPENDING, span_sources(12), replace(PRICED_AXIS, carry=PRICED_CARRY), start=None)
    assert answer_of(resumed).equals(answer_of(cold))
    assert resumed.scan('soc').collect().equals(cold.primal('soc'))
    assert resumed.scan('soc', per_window=True).collect().equals(cold.primal('soc', per_window=True))
    loads = cold.metrics['solver_instances'].to_list()
    loads[2] = 1
    assert resumed.metrics['solver_instances'].to_list() == loads, (
        'the two read back are the record they left, and the third loads where the uninterrupted run updated'
    )


def test_a_slice_written_part_way_is_solved_again(builds, tmp_path):
    """The objective file is written last and is what marks a slice done, so
    a slice whose tables landed but whose record did not is solved again."""
    _spilled(tmp_path)
    (tmp_path / 'record' / '000001.parquet').unlink()
    built = builds(driver)
    resumed = _spilled(tmp_path)
    assert len(built) == 1, 'the slice without its record is the one built'
    assert resumed.keys == [0, 3, 6, 9], 'the sweep comes back whole'


def test_a_sweep_saved_over_another_leaves_none_of_its_files(tmp_path):
    """A save wrote into what the directory held, so a file the new sweep lacks survived from the old one.

    Here the first sweep solves every scenario and the second leaves one
    infeasible. Read back, that slice said it had no values and still
    returned the first sweep's.
    """
    sources = scenario_sources()
    out = sps.solve_over(DISPATCH, sources, sps.EachLabel('scenario')).save(tmp_path / 'study')
    beyond = sources['load'].with_columns(
        pl.when(pl.col('scenario') == 'high').then(pl.col('value') + 1_000).otherwise(pl.col('value'))
    )
    again = sps.solve_over(DISPATCH, {**sources, 'load': beyond}, sps.EachLabel('scenario'))
    again.save(out)

    saved = sps.read_sweep(out)
    assert saved.records['has_primal'].to_list() == again.records['has_primal'].to_list()
    assert saved.primal('p')['scenario'].unique().sort().to_list() == ['low', 'mid'], (
        'the scenario that has no values now has none on disk either'
    )


def test_a_directory_holding_another_sweep_is_refused(tmp_path):
    """A directory answers for one sweep. Another one pointed at it would read
    the first one's slices back as its own, so the mismatch is refused."""
    _spilled(tmp_path)
    with pytest.raises(sps.errors.SpecsolveError, match='holds a sweep keyed by'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path)


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_spills_the_same_files(make_executor, sweep, tmp_path):
    """Under a pool the answers still land in the directory, in slice order."""
    with _entered(make_executor()) as executor:
        runs = sps.solve_over(
            DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=executor, spill_to=tmp_path
        )
    assert runs.scan('p').collect().equals(sweep.primal('p'))
    assert sorted(p.name for p in (tmp_path / 'primal' / 'p').iterdir()) == [
        '000000.parquet',
        '000001.parquet',
        '000002.parquet',
    ], 'one file per slice, numbered by position'


def test_a_pooled_sweep_resumes_too(builds, tmp_path):
    """A slice the directory holds is never submitted; the pool only sees the
    ones still to solve, and the fold reads the rest back in order."""
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), spill_to=tmp_path)
    (tmp_path / 'record' / '000001.parquet').unlink()
    built = builds(driver)
    with ThreadPoolExecutor(2) as pool:
        resumed = sps.solve_over(
            DISPATCH, scenario_sources(), sps.EachLabel('scenario'), executor=pool, spill_to=tmp_path
        )
    assert len(built) == 1, 'the slice without its record is the one submitted'
    assert resumed.keys == ['high', 'low', 'mid'], 'the sweep comes back whole and in order'


def test_scan_on_a_spilled_sweep_says_what_it_does_hold(tmp_path):
    """A name no slice wrote has no directory, and the message lists the
    names that do — the same sentence the in-memory reader gives."""
    runs = _spilled(tmp_path)
    with pytest.raises(
        sps.errors.SpecsolveError, match=r"no variable 'nope' in this sweep — it holds 'charge', 'discharge'"
    ):
        runs.scan('nope')


def test_an_export_reads_the_key_off_each_row_and_skips_a_slice_with_none(sweep):
    """A name is held per slice that produced it, not per slice, so an export
    cannot count positions: it reads the key off the rows, and a slice with no
    rows of a name — a variable every row of which the slice left absent — has no
    key, which is what the spill writes for it."""
    index = sweep._by_slice('primal')['p']
    assert list(index) == ['high', 'low', 'mid'], 'one table per slice, keyed by its own key'
    assert all(sweep.key_name not in table.columns for table in index.values()), 'the key column is dropped'

    held = sweep._slices['primal']['p'].filter(pl.col(sweep.key_name) != 'low')
    short = replace(sweep, _slices={'primal': {'p': held}})
    assert list(short._by_slice('primal')['p']) == ['high', 'mid'], 'a slice with no rows is left out'


def test_a_sweep_archive_carries_its_carry(tmp_path):
    """The carry is config the tables do not hold, so the archive stores it beside the axis."""
    sps.solve_over(
        WINDOW, span_sources(), replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}), archive=tmp_path / 'roll'
    )
    assert sps.read_archive(tmp_path / 'roll').axis.carry == {'soc_initial': 'soc'}, (
        'the carry reads back as it was given'
    )


def test_a_sweep_archive_with_no_carry_reads_an_empty_carry(tmp_path):
    """A sweep that chained nothing carries nothing — the manifest omits the key and the reader defaults it."""
    sps.solve_over(WINDOW, span_sources(), WINDOW_AXIS, archive=tmp_path / 'plain')
    assert sps.read_archive(tmp_path / 'plain').axis.carry == {}, (
        'no carry given, none stored, an empty mapping read back'
    )


def test_a_carried_sweep_reruns_from_its_archive_with_the_stored_carry(tmp_path):
    """The stored carry is what makes a re-run the same sweep: with it the chained answer is reproduced."""
    original = sps.solve_over(
        WINDOW, span_sources(), replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}), archive=tmp_path / 'roll'
    )
    packed = sps.read_archive(tmp_path / 'roll')
    rerun = sps.solve_over(packed.spec, packed.sources, packed.axis)
    assert rerun.primal('soc').equals(original.primal('soc')), 'the re-run with the stored carry matches the archive'


def test_a_sweep_archive_evaluates_a_quantity_the_file_never_named_per_slice(tmp_path):
    """`Sweep.evaluate` reads an undeclared quantity at each slice's own solution — matches solving that slice alone."""
    axis = sps.EachLabel('scenario')
    sps.solve_over(DISPATCH, scenario_sources(), axis, archive=tmp_path / 'study.zip')
    sweep = sps.read_archive(tmp_path / 'study.zip', tmp_path / 'out')
    expr = 'sum(p * cost, over=generator)'
    swept = sweep.sweep.evaluate(expr)
    for key in sweep.sweep.keys:
        with sps.build_slice(DISPATCH, scenario_sources(), axis, key) as model:
            live = model.solve().evaluate(expr)
        got = swept.filter(pl.col(sweep.sweep.key_name) == key).drop(sweep.sweep.key_name)
        columns = live.columns[:-1]
        assert got.sort(columns).equals(live.sort(columns)), f'slice {key!r} evaluates at its own primal, no re-solve'


def test_a_scanned_sweep_archive_evaluates_the_same(tmp_path):
    """A sweep left on disk (`scan_archive`) evaluates against those tables, the same values held reads."""
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachLabel('scenario'), archive=tmp_path / 'study.zip')
    expr = 'sum(p * cost, over=generator)'
    whole = sps.read_archive(tmp_path / 'study.zip', tmp_path / 'whole').sweep.evaluate(expr)
    scanned = sps.scan_archive(tmp_path / 'study.zip', tmp_path / 'scan').sweep.evaluate(expr)
    assert scanned.equals(whole), 'a scanned sweep evaluates against the tables on disk, the same answer'


def test_a_live_sweep_has_no_model_to_evaluate_against():
    """A Sweep a live solve returned retains no model, so an undeclared expression says why — the archive is what carries one — while a declared name is stitched from what the sweep holds."""
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    runs = sps.solve_over(spec, scenario_sources(), sps.EachLabel('scenario'))
    with pytest.raises(sps.errors.SpecsolveError, match='no model behind it') as refused:
        runs.evaluate('sum(p, over=generator)')
    assert 'archive=' in str(refused.value), 'the refusal names what gives a sweep its models back'
    assert 'read_archive' in str(refused.value), 'and where the sweep that reads any expression comes from'
    assert 'sps.solve(' not in str(refused.value), (
        "one solve of a sweep's sources is another question, and here it is refused: they hold a row per scenario"
    )
    assert runs.evaluate('spend')['scenario'].n_unique() == len(runs), (
        'the declared name answers without a model, stitched from every slice'
    )


@pytest.mark.parametrize(
    'read',
    [
        pytest.param(lambda sweep, spilled: sweep, id='the-sweep-as-it-came-back'),
        pytest.param(lambda sweep, spilled: sps.scan_sweep(spilled), id='its-spill-scanned'),
    ],
)
def test_a_slice_of_a_carried_sweep_is_built_as_the_sweep_built_it_given_the_sweep(read, tmp_path):
    """Under a carry a window after the first is built from what the one before it left, which the sweep holds.

    From the sources alone it is built from the seed, which is no model the
    sweep solved.
    """
    axis = sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'})
    sources = span_sources()
    sweep = sps.solve_over(WINDOW, sources, axis, spill_to=tmp_path / 'runs')
    solved = sweep.records['objective'].to_list()

    from_the_seed = [_objective_alone(WINDOW, sources, axis, key) for key in sweep.keys]
    assert from_the_seed[0] == pytest.approx(solved[0]), 'the first window is built from the seed either way'
    assert from_the_seed[1] != pytest.approx(solved[1]), 'and the second, from the seed, is another model'

    held = read(sweep, tmp_path / 'runs')
    as_the_sweep_built = [_objective_alone(WINDOW, sources, axis, key, sweep=held) for key in sweep.keys]
    assert as_the_sweep_built == pytest.approx(solved), 'each window built alone is the window the sweep solved'

    full = {**sources, 'soc_initial': pl.DataFrame({'value': [100.0]})}
    assert _objective_alone(WINDOW, full, axis, 3, sweep=held) == pytest.approx(solved[1]), (
        'the sweep says what the window started from, whatever the sources hold'
    )


def test_a_slice_starts_from_the_carried_parameter_its_sources_give():
    """A carried parameter is a source like any other, so a slice built without the sweep starts from the value given.

    Given the level the window before ended at, it is the window the sweep
    solved. Given another level, it is another model.
    """
    axis = sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'})
    sources = span_sources()
    sweep = sps.solve_over(WINDOW, sources, axis)
    solved = sweep.records['objective'][1]

    ended_at = sweep.primal('soc').filter(pl.col('snapshot') == 2).select('value')
    assert _objective_alone(WINDOW, {**sources, 'soc_initial': ended_at}, axis, 3) == pytest.approx(solved), (
        'the second window, from the level the first ended at'
    )
    full = pl.DataFrame({'value': [100.0]})
    assert _objective_alone(WINDOW, {**sources, 'soc_initial': full}, axis, 3) < solved - 1.0, (
        'and from a full store it is a cheaper window than the sweep solved'
    )


def _stopped_at_the_second_window(spilled) -> tuple[sps.EachWindow, dict[str, object]]:
    """A carried sweep run to where it stops: its second window cannot meet a load beyond every generator."""
    axis = sps.EachWindow('snapshot', horizon=3, lookahead=3, carry={'soc_initial': 'soc'})
    sources = span_sources()
    beyond = sources['load'].with_columns(
        pl.when(pl.col('snapshot').is_between(6, 8)).then(pl.col('value') + 1_000).otherwise(pl.col('value'))
    )
    stopped = {**sources, 'load': beyond}
    with pytest.raises(sps.errors.SpecsolveError, match='leaves nothing to carry'):
        sps.solve_over(WINDOW, stopped, axis, spill_to=spilled)
    return axis, stopped


def test_the_window_a_carried_sweep_stopped_at_builds_alone(tmp_path):
    """A carried sweep stops at a window that leaves nothing to hand on, and that window is the one to build alone."""
    axis, stopped = _stopped_at_the_second_window(tmp_path / 'runs')
    ran = sps.scan_sweep(tmp_path / 'runs')
    assert ran.keys == [0, 3], 'the window that solved, and the one the sweep stopped at'
    with sps.build_slice(WINDOW, stopped, axis, 3, sweep=ran) as model:
        assert model.solve().termination_condition == 'infeasible', 'built alone, it is the window that failed'


@pytest.mark.parametrize(
    ('key', 'refusal'),
    [
        pytest.param(
            6,
            "slice 3 terminated infeasible in sweep=, so it left no ['soc'] to build slice 6 from.",
            id='the-slice-before-left-nothing',
        ),
        pytest.param(
            9,
            "sweep= did not run slice 6, the one before 9, so it holds no ['soc'] to build slice 9 from. "
            'It ran 2 slice(s), keyed 0, 3.',
            id='the-slice-before-never-ran',
        ),
    ],
)
def test_a_slice_the_sweep_never_reached_is_refused(key, refusal, tmp_path):
    """Past where a carried sweep stopped, no slice was built, so there is nothing to build one from.

    Read anyway, the tables would be empty and the slice built from a carried
    parameter with no rows.
    """
    axis, stopped = _stopped_at_the_second_window(tmp_path / 'runs')
    with pytest.raises(sps.errors.SpecsolveError, match=re.escape(refusal)):
        sps.build_slice(WINDOW, stopped, axis, key, sweep=sps.scan_sweep(tmp_path / 'runs'))


def test_a_slice_is_built_from_its_cut_and_what_the_one_before_handed_on():
    """`as_built` threads a carry for every caller, and asks for a slice's primal only when the next slice is asked for.

    A caller that solves as it goes has the primal by then. The last slice
    hands nothing on, so it is never asked.
    """
    slices = [Slice(key, {'seed': f'the seed of {key}', 'own': key}, cut=lambda table: table) for key in 'abc']
    asked: list[object] = []

    def primals_of(before: Slice) -> dict[str, pl.DataFrame]:
        asked.append(before.key)
        return {'state': pl.DataFrame({'value': [float(len(asked))]})}

    walk = as_built(slices, {'seed': CarryRule('state', None)}, primals_of)
    first = next(walk)
    assert (first.key, first.sources['seed'], asked) == ('a', 'the seed of a', []), (
        'the first slice is its own cut, and no primal is asked for yet'
    )
    rest = list(walk)
    assert asked == ['a', 'b'], 'each slice is asked once the next one is, and the last never'
    assert [built.sources['seed'].item() for built in rest] == [1.0, 2.0], 'each takes what the one before it handed on'
    assert [built.sources['own'] for built in rest] == ['b', 'c'], 'beside its own cut of everything else'


#: Two draws chained by a carry that switches a variable off: `p` exists only while `seed` is positive.
SWITCHED_OFF = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'seed': {'dims': []}, 'demand': {'dims': []}},
    'variables': {
        'state': {'dims': [], 'bounds': {'lower': 0, 'upper': 0}},
        'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}, 'where': 'seed > 0'},
        'q': {'dims': [], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {'meet': {'dims': [], 'expression': 'q >= demand'}},
    'objective': {'sense': 'minimize', 'expression': 'q + state + sum(p)'},
}


def test_an_archived_sweep_rebuilds_each_slice_with_the_carry_it_was_solved_with(tmp_path):
    """A slice was rebuilt from the seed in the sources, whatever the slice before it had handed on.

    Here the first draw leaves `seed` at 0, so the second builds no `p`. The
    rebuild read `seed` as 1, expected a `p` table the archive does not hold,
    and refused its own answer.
    """
    sources = {
        'g': ['a', 'b'],
        'seed': pl.DataFrame({'value': [1.0]}),
        'demand': pl.DataFrame({'draw': [1, 2], 'value': [3.0, 4.0]}),
    }
    axis = sps.EachLabel('draw', carry={'seed': 'state'})
    live = sps.solve_over(SWITCHED_OFF, sources, axis, archive=tmp_path / 'study')
    assert live.primal('p')['draw'].unique().to_list() == [1], 'the second draw builds no p'

    read = sps.read_archive(tmp_path / 'study').sweep.evaluate('state + q').sort('draw')
    assert read['value'].to_list() == [3.0, 4.0], 'each draw read against the model it was solved as'


def test_evaluate_across_a_sweep_refuses_an_expression_that_reads_a_carried_parameter(tmp_path):
    """The narrow gap: a carried value is a previous slice's answer, not stored data, so evaluate refuses it."""
    sps.solve_over(
        WINDOW,
        span_sources(),
        replace(WINDOW_AXIS, carry={'soc_initial': 'soc'}),
        archive=tmp_path / 'roll.zip',
    )
    sweep = sps.read_archive(tmp_path / 'roll.zip', tmp_path / 'roll')
    assert sweep.sweep.evaluate('sum(p * cost)', per_window=True).height, (
        'an expression over static data evaluates per slice'
    )
    with pytest.raises(sps.errors.SpecsolveError, match='carried'):
        sweep.sweep.evaluate('soc_initial')


def test_evaluate_reads_an_undeclared_expression_as_an_answer_like_primal(tmp_path):
    """`evaluate` stitches as `primal` does — the sliced dimension back, the slice key gone."""
    sps.solve_over(WINDOW, span_sources(), WINDOW_AXIS, archive=tmp_path / 'roll.zip')
    answer = sps.read_archive(tmp_path / 'roll.zip', tmp_path / 'roll').sweep
    reindexed = answer.evaluate('sum(p, over=generator)')
    assert reindexed.columns == ['snapshot', 'value'], 'the sliced dimension is restored and the slice key dropped'
    by_hand = answer.primal('p').group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    assert reindexed.sort('snapshot').equals(by_hand.select('snapshot', 'value')), (
        'the evaluated expression equals the answer of the primal summed by hand'
    )
    assert answer.evaluate('sum(p * cost)', per_window=True).height == len(answer), 'one total per window'


def test_evaluate_refuses_a_quantity_reduced_over_the_sliced_dim_and_names_per_window(tmp_path):
    """A scalar-per-window quantity has no labels of the sliced dimension to restore, so its answer is refused — as `expression` is."""
    sps.solve_over(WINDOW, span_sources(), WINDOW_AXIS, archive=tmp_path / 'roll.zip')
    answer = sps.read_archive(tmp_path / 'roll.zip', tmp_path / 'roll').sweep
    with pytest.raises(sps.errors.SpecsolveError, match=r"no answer over 'snapshot'.*per_window=True"):
        answer.evaluate('sum(p * cost)')


#: The labels table of another dimension, carrying the axis column, under each axis.
CUT_LABELS = [
    pytest.param(
        MYOPIC,
        lambda: {
            **myopic_sources(),
            'generator': pl.DataFrame({'generator': ['wind', 'gas', 'wind', 'gas'], 'period': [1, 1, 2, 2]}),
        },
        sps.EachLabel('period'),
        "the labels table of dimension 'generator' carries a 'period' column, and EachLabel('period')",
        id='each-label',
    ),
]


@pytest.mark.parametrize(('spec', 'sources', 'axis', 'match'), CUT_LABELS)
def test_the_labels_of_another_dimension_carrying_the_axis_are_refused_before_a_slice(spec, sources, axis, match):
    """A labels table lists the labels every slice has; which of them a slice has is a parameter or a relation.

    Cut by the axis, the labels table would make each slice a model over other labels.
    """
    with (
        mock.patch.object(type(axis), '_slice', side_effect=AssertionError('the axis cut the sources')),
        pytest.raises(sps.errors.DataError, match=re.escape(match)) as refused,
    ):
        sps.solve_over(spec, sources(), axis)
    assert 'in a parameter or a relation over' in str(refused.value), 'the message names the rewrite'


def test_a_window_cuts_what_the_spec_declares_over_the_dimension_and_nothing_else():
    """A window cuts what the spec declares, so the labels table of another dimension that happens to carry a column
    of the windowed name is read as the labels table it is."""
    sources = {**span_sources(8), 'generator': pl.DataFrame({'generator': GENERATORS, 'snapshot': [0, 1]})}
    runs = sps.solve_over(WINDOW, sources, WINDOW_AXIS)
    assert runs.primal('p')['generator'].unique().sort().to_list() == sorted(GENERATORS), 'every window has both'


def test_one_window_over_the_whole_windowed_dimension_is_the_single_solve():
    """A window takes the spec and the sources a single solve takes, so one window of everything is that solve."""
    sources = span_sources(12)
    whole = sps.solve(WINDOW, sources)
    windowed = sps.solve_over(WINDOW, sources, sps.EachWindow('snapshot', horizon=12, lookahead=0))
    assert windowed.records['objective'].to_list() == pytest.approx([whole.objective]), 'one window, one objective'
    assert windowed.primal('soc').equals(whole.primal('soc')), "the answer is the single solve's table"
    assert windowed.primal('p').columns == whole.primal('p').columns, 'with its dimensions in the same order'


def test_a_label_the_dimension_does_not_have_is_refused_under_windows_as_for_a_single_solve():
    """A row whose label the dimension does not have falls in no window, so cutting alone would drop it without a word.

    Every window keeps such rows, and `tidy_sources` refuses them as it does for a
    single solve.
    """
    sources = span_sources(8)
    sources['load'] = pl.concat([sources['load'], pl.DataFrame({'snapshot': [99], 'value': [1.0]})])
    with pytest.raises(sps.errors.DataError, match=r"label\(s\) in dimension 'snapshot' that are not labels"):
        sps.solve_over(WINDOW, sources, WINDOW_AXIS)


def test_a_window_needs_the_labels_a_single_solve_needs():
    """The labels a window cuts are the dimension's own, in label order, so they have to be there."""
    sources = span_sources(8)
    del sources['snapshot']
    with pytest.raises(sps.errors.DataError, match="dimension 'snapshot' has no labels"):
        sps.solve_over(WINDOW, sources, WINDOW_AXIS)
