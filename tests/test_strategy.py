"""The fold: one plan per slice, and the answers stitched back together.

Every claim here is about slicing, coupling and folding, not the language: the
windowed model ``WINDOW`` uses only constructs that ship.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import multiprocessing
import re
import shutil
import sys
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from unittest import mock

import polars as pl
import pytest
from mathspec import to_spec

import specsolve as sps
from specsolve import strategy
from specsolve.api import Model
from specsolve.relational.parquet import SliceMetrics
from tests.conftest import DISPATCH_SPEC, override

# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------

GENERATORS = ['wind', 'gas']
STORES = ['battery', 'pumped']

#: Dispatch, with a scenario-free declaration: the slice column never appears
#: in the model.
DISPATCH = DISPATCH_SPEC

#: Storage over a *local* index, with the seam split out by a `where` on a dim
#: literal. `soc_step` carries no `edge=`, so its vacated row drops and the
#: masked `soc_open` supplies it from a carried parameter.
WINDOW = {
    'dimensions': {'t': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['t']},
        'soc_initial': {'dims': []},
    },
    'variables': {
        'p': {'dims': ['t', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'discharge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'soc': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'dims': ['t'],
            'expression': 'sum(p, over=generator) + discharge - charge == load',
        },
        'soc_open': {
            'dims': ['t'],
            'where': 't == 0',
            'expression': 'soc == soc_initial + charge * 0.9 - discharge',
        },
        'soc_step': {
            'dims': ['t'],
            'where': 't > 0',
            'expression': 'soc == shift(soc, along=t, offset=1) + charge * 0.9 - discharge',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}

#: The same storage, but two of them — so `soc` is over `(t, storage)` and the
#: carried `soc_initial` over `(storage)`. The carry drops `t` and `storage`
#: rides along.
#:
#: `charge` and `discharge` are capped well below a window's worth so a store
#: cannot empty itself before the seam; otherwise every window ends at zero and
#: carrying the state is indistinguishable from not carrying it.
MULTI_STORE = {
    'dimensions': {'t': {'dtype': 'int'}, 'generator': {'dtype': 'str'}, 'storage': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['t']},
        'soc_initial': {'dims': ['storage']},
        'efficiency': {'dims': ['storage']},
    },
    'variables': {
        'p': {'dims': ['t', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'charge': {'dims': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'discharge': {'dims': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 5}},
        'soc': {'dims': ['t', 'storage'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'balance': {
            'dims': ['t'],
            'expression': 'sum(p, over=generator) + sum(discharge, over=storage) - sum(charge, over=storage) == load',
        },
        'soc_open': {
            'dims': ['t', 'storage'],
            'where': 't == 0',
            'expression': 'soc == soc_initial + charge * efficiency - discharge',
        },
        'soc_step': {
            'dims': ['t', 'storage'],
            'where': 't > 0',
            'expression': 'soc == shift(soc, along=t, offset=1) + charge * efficiency - discharge',
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
    """Two stores, each with a real starting level worth handing across a seam."""
    return {
        **horizon_sources(12),
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


def horizon_sources(periods: int = 12) -> dict[str, object]:
    """A load profile of any length — the pattern repeats past twelve."""
    load = [5.0, 9.0, 30.0, 40.0, 6.0, 8.0, 35.0, 45.0, 7.0, 10.0, 25.0, 50.0]
    return {
        **STATIC,
        'load': pl.DataFrame({'snapshot': range(periods), 'value': [load[t % len(load)] for t in range(periods)]}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }


def coordinate_sources(coordinates: list, load: float = 5.0) -> dict[str, object]:
    """Six coordinates of flat load, whatever the coordinates are."""
    return {
        **STATIC,
        'load': pl.DataFrame({'snapshot': coordinates, 'value': [load] * 6}),
        'soc_initial': pl.DataFrame({'value': [0.0]}),
    }


#: Window geometries whose *tail* differs: a final window of one, a final
#: window of ``steps``, a horizon shorter than a single window, and a tail that
#: divides exactly.
GEOMETRIES = [
    pytest.param(periods, steps, lookahead, id=f'n{periods}-s{steps}-la{lookahead}')
    for periods in (1, 2, 5, 7, 12)
    for steps in (1, 2, 3, 6)
    for lookahead in range(7 - steps)
]

#: The one contiguous geometry most window tests share — frozen, so sharing is safe.
WINDOW_AXIS = sps.EachWindow('snapshot', steps=4, lookahead=0, into='t')


@pytest.fixture(scope='module')
def sweep() -> strategy.Sweep:
    """The scenario sweep, solved once for every test that only reads it."""
    return sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'))


@pytest.fixture(scope='module')
def overlapping() -> strategy.Sweep:
    """The overlapping-window sweep, solved once for every test that only reads it."""
    return sps.solve_over(
        WINDOW,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=3, lookahead=3, into='t'),
        carry={'soc_initial': 'soc'},
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
# EachCoordinate — the independent case
# ---------------------------------------------------------------------------


def answer_of(runs: strategy.Sweep) -> pl.DataFrame:
    """A sweep's record without `solved_at` and `specsolve_run`, which belong to a *run* rather than an answer."""
    return runs.record.drop('solved_at', 'specsolve_run')


def test_a_scenario_sweep_solves_each_slice_and_keys_the_answers(sweep):
    """The model never mentions `scenario`; the driver filters and drops it."""
    runs = sweep

    assert len(runs) == 3
    assert runs.keys == ['high', 'low', 'mid'], 'keys come back sorted, not in data order'
    assert runs.record.columns == [
        'scenario',
        'status',
        'termination_condition',
        'objective',
        'has_primal',
        'spec_digest',
        'solved_at',
        'specsolve_run',
        'model_digest',
    ], 'the record, keyed'
    assert set(runs.primal('p').columns) == {'scenario', 'snapshot', 'generator', 'value'}
    assert runs.primal('p').height == 3 * 4 * 2

    by_key = dict(zip(runs.record['scenario'], runs.record['objective'], strict=True))
    assert by_key['low'] < by_key['mid'] < by_key['high'], 'a bigger load is a costlier dispatch'


def test_a_fold_passes_its_keep_to_every_slice_and_chooses_none(monkeypatch):
    """`keep` reaches each slice as asked, and the default is `solve`'s.

    The request is invisible in the answer and in `loads`, so it is read off
    the call. `kept` would test the data instead: a slice whose labels moved is
    loaded again and keeps `nothing`.
    """
    asked: list[object] = []
    original = Model.solve

    def recording(self, *args, **kwargs):
        asked.append(kwargs.get('keep'))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Model, 'solve', recording)

    sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'))
    assert asked == ['solver'] * 3, f'the fold defaulted to {asked}, not solve()s own default'

    asked.clear()
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), keep='progress')
    assert asked == ['progress'] * 3, f'the fold asked for {asked}, not what the caller chose'


def test_a_serial_fold_builds_once_and_updates(builds):
    """The fold is an update loop: nothing after the first slice pays for the YAML, the plan or a fresh solver.

    The difference is invisible in the answer, so it is counted at `build`.
    """
    built = builds(strategy)

    runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'))

    assert len(runs) == 3, 'three slices'
    assert len(built) == 1, f'{len(built)} builds for three slices — the fold stopped updating'
    seen = built[0].diagnostics()
    assert (seen.loads, seen.solves) == (1, 3), 'one solver load, the slices differing only in numbers'


def test_a_carried_fold_still_builds_once(builds):
    """A carry writes a parameter the first slice already attached, so no slice names new sources."""
    built = builds(strategy)

    runs = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'})

    assert runs.keys == [0, 4, 8]
    assert len(built) == 1, f'{len(built)} builds for three windows — the carry cost the fold its fast path'


def test_a_pooled_fold_builds_per_slice(builds):
    """The exception: a built model cannot be pickled, so a pooled slice builds its own."""
    built = builds(strategy)

    with ThreadPoolExecutor(2) as pool:
        runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=pool)

    assert len(runs) == 3
    assert len(built) == 3, 'a slice that may run in another process builds its own model'


def test_each_slice_matches_solving_that_slice_alone(sweep):
    """The fold must not change the answer — the oracle is `solve` itself."""
    folded = dict(zip(sweep.record['scenario'], sweep.record['objective'], strict=True))

    for scenario, expected in folded.items():
        one = scenario_sources()
        one['load'] = _draw(one, scenario)
        with sps.solve(DISPATCH, one) as result:
            assert result.objective == pytest.approx(expected)


def test_an_axis_naming_a_column_no_source_carries_says_so():
    with pytest.raises(sps.DataError, match="no source carries a 'draw' column"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('draw'))


def test_a_name_the_sweep_does_not_hold_says_what_it_does_hold(sweep):
    """Everything a slice produced is kept, so a miss is a name, not a flag."""
    with pytest.raises(sps.SpecsolveError, match="no variable 'q' in this sweep"):
        sweep.primal('q')
    with pytest.raises(sps.SpecsolveError, match="no constraint 'nope' in this sweep"):
        sweep.dual('nope')


def test_a_sweep_that_solved_nothing_blames_the_solve():
    """An absent frame has one cause, and the message says which.

    The load is pushed past total capacity, so every slice is infeasible.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'))

    assert len(runs) == 3, 'an unsolvable slice is still a row of the record'
    assert runs.record['objective'].null_count() == 3, 'no slice reached one, and none is written as nan'
    with pytest.raises(sps.SpecsolveError, match='holds no variable frames at all') as raised:
        runs.primal('p')
    assert 'infeasible' in str(raised.value), 'the message names what the slices actually did'


def test_a_slice_that_reached_no_objective_does_not_poison_the_sweep():
    """`objective` is a table, and in a table an absent number is null.

    nan is a *number* to every aggregate that meets it, so one infeasible
    slice makes the mean over the sweep nan.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(
        pl.when(pl.col('scenario') == 'high').then(pl.col('value') + 1_000).otherwise(pl.col('value'))
    )
    runs = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'))

    assert runs.record['objective'].null_count() == 1, 'the one slice that came back infeasible'
    assert runs.record['objective'].is_nan().sum() == 0, 'written as no value rather than as nan'
    assert runs.record['objective'].mean() == pytest.approx(runs.record.filter('has_primal')['objective'].mean()), (
        'so the mean over the sweep is the mean over the slices that solved'
    )


# ---------------------------------------------------------------------------
# EachWindow — the coupled case
# ---------------------------------------------------------------------------


def test_a_rolling_horizon_carries_state_across_the_seam():
    """Three contiguous windows, the store's level handed forward.

    `soc_initial` is updated per window from the previous window's last `soc`,
    which is the carry doing its one job — a copy, at a named index.
    """
    runs = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'})

    assert runs.keys == [0, 4, 8]
    assert runs.primal('p').height == 3 * 4 * 2
    assert set(runs.primal('soc').columns) == {'snapshot_start', 't', 'value'}, (
        'the rows are indexed by `t`, so the key column cannot be called `snapshot`'
    )
    assert runs.record['objective'].to_list() == pytest.approx([2270.0, 2770.0, 2655.0])


def test_overlapping_windows_advance_by_step_and_look_ahead_by_length(overlapping):
    runs = overlapping
    assert runs.keys == [0, 3, 6, 9]
    assert runs.primal('soc').filter(pl.col('snapshot_start') == 9).height == 3, (
        'the tail window is short rather than padded: 9..11 is three rows, not six'
    )


def test_stitch_drops_the_overlap_and_restores_the_global_coordinate(overlapping):
    """The answer a rolling horizon is for, without the caller doing arithmetic."""
    runs = overlapping
    stitched = runs.primal('soc', original_index=True)
    assert stitched.columns == ['snapshot', 'value'], 'the slice bookkeeping is gone'
    assert stitched['snapshot'].to_list() == list(range(12)), 'and every coordinate is present once'
    assert runs.primal('soc').height == 21, 'every window kept the `steps` coordinates it owns'


@pytest.mark.parametrize(('periods', 'steps', 'lookahead'), GEOMETRIES)
def test_a_window_geometry_covers_every_coordinate_exactly_once(periods, steps, lookahead):
    """A stitched sweep reproduces the coordinate list, whatever the tail."""
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(periods),
        sps.EachWindow('snapshot', steps=steps, lookahead=lookahead, into='t'),
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(periods)), (
        'the original index must reproduce the coordinate list, whatever the tail'
    )
    assert runs.primal('soc')['snapshot_start'].n_unique() == len(range(0, periods, steps)), (
        'one slice per window start'
    )


@pytest.mark.parametrize(('periods', 'steps', 'lookahead'), GEOMETRIES)
def test_a_carry_finds_the_seam_in_every_geometry(periods, steps, lookahead):
    """The coordinate a carry hands on is the last one the window owns.

    A non-final window owns exactly ``steps``; a final one owns whatever is
    left, which can be one. Both are in range by construction, because a
    window owns rows it solved.
    """
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(periods),
        sps.EachWindow('snapshot', steps=steps, lookahead=lookahead, into='t'),
        carry={'soc_initial': 'soc'},
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(periods)), (
        'the seam is in range for every geometry, so the sweep completes'
    )


def test_stitch_keeps_the_whole_of_the_final_short_window():
    """A tail window holds at most `steps`, so the owning rule keeps all of it.

    12 coordinates kept 5 at a time leaves a final window of two.
    """
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=5, lookahead=1, into='t'),
    )
    assert runs.keys == [0, 5, 10], 'three windows, the last of two coordinates'
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == list(range(12)), (
        'the short tail window is kept whole, not dropped for being short'
    )


def test_a_hand_built_axis_refuses_to_read_over_a_dimension_it_never_named(tmp_path):
    """`original_index=True` on a hand-built axis is refused rather than ignored.

    A list of slices carries no `into`, no sliced dimension and no record of
    what each window owns, so there is no way back to `snapshot`. `scan`
    reaches the same guard by its own route.
    """
    sources = horizon_sources(12)
    windows = sps.EachWindow('snapshot', steps=3, lookahead=3, into='t').slices(sources)

    runs = sps.solve_over(WINDOW, sources, windows, key_name='window')
    assert runs.primal('soc').columns == ['window', 't', 'value'], 'a hand-built axis keys by what it was told'
    with pytest.raises(sps.SpecsolveError, match='does not say what its keys are coordinates of'):
        runs.primal('soc', original_index=True)

    spilled = sps.solve_over(WINDOW, sources, windows, key_name='window', spill_to=tmp_path / 'runs')
    with pytest.raises(sps.SpecsolveError, match='does not say what its keys are coordinates of'):
        spilled.scan('soc', original_index=True)


def test_stitching_an_axis_that_re_indexed_nothing_changes_nothing(sweep):
    """A caller handed an axis should not have to ask which kind it is."""
    assert sweep.primal('p', original_index=True).equals(sweep.primal('p')), (
        'an axis that re-indexed nothing has nothing to restore'
    )
    assert sweep.dual('balance', original_index=True).equals(sweep.dual('balance')), (
        'and that holds for duals too, since it is a property of the axis'
    )


def test_duals_stitch_the_same_way_primals_do(overlapping):
    """A window's price at a coordinate is the owning window's, not a blend.

    What has to be undone is a property of the *axis*, so a dual stitches as a
    primal does.
    """
    runs = overlapping
    keyed, stitched = runs.dual('balance'), runs.dual('balance', original_index=True)
    assert keyed.columns == ['snapshot_start', 't', 'value']
    assert stitched.columns == ['snapshot', 'value']
    assert stitched['snapshot'].to_list() == list(range(12)), 'one price per coordinate'
    assert keyed.height > stitched.height, 'the overlap is priced twice before the index collapses it'


def test_keyed_is_the_default_because_stitching_drops_rows(overlapping):
    """The default may not silently discard answers the sweep computed."""
    runs = overlapping
    assert runs.primal('soc').height == 21, 'keyed keeps every row every window solved'
    assert runs.primal('soc', original_index=True).height == 12, 'only the rows each window owns'
    assert runs.record.join(runs.primal('soc'), on=runs.key_name).height == 21, (
        'keyed by the same column as `objective`, so the two still join'
    )


# ---------------------------------------------------------------------------
# named expressions across a sweep
# ---------------------------------------------------------------------------

#: The window model with its cost named twice: `spend` keeps the local index
#: (pointwise in `t`, so it can be stitched) and `window_spend` reduces over it
#: (one number per window, so it cannot).
SPENDING = override(
    WINDOW,
    **{
        'expressions.spend': 'sum(p * cost, over=generator)',
        'expressions.window_spend': 'sum(sum(p * cost, over=generator), over=t)',
    },
)


@pytest.fixture(scope='module')
def priced() -> strategy.Sweep:
    """The overlapping-window sweep of the expression-bearing model, solved once."""
    return sps.solve_over(
        SPENDING,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=3, lookahead=3, into='t'),
        carry={'soc_initial': 'soc'},
    )


def test_a_stitched_expression_prices_only_the_rows_a_window_owns(priced):
    """`expression(original_index=True)` drops the lookahead double-count.

    The oracle is the stitched dispatch priced by hand. The keyed sum exceeds
    it, since the keyed frames carry the overlap.
    """
    stitched = priced.evaluate('spend', original_index=True)
    assert stitched.columns == ['snapshot', 'value']
    assert stitched['snapshot'].to_list() == list(range(12)), 'one value per coordinate, like a stitched primal'

    by_hand = (
        priced.primal('p', original_index=True)
        .join(STATIC['cost'].rename({'value': 'cost'}), on='generator')
        .group_by('snapshot')
        .agg((pl.col('value') * pl.col('cost')).sum())
        .sort('snapshot')
    )
    assert stitched['value'].to_list() == pytest.approx(by_hand['value'].to_list())
    assert priced.evaluate('spend')['value'].sum() > stitched['value'].sum(), (
        'the keyed frames still carry the lookahead rows, so their sum double-counts'
    )


def test_a_quantity_reduced_over_the_sliced_dimension_has_no_way_back(priced):
    """Per window it reads; over the original index the refusal says why not."""
    keyed = priced.evaluate('window_spend')
    assert keyed.columns == ['snapshot_start', 'value']
    assert keyed.height == len(priced), 'one total per window, keyed like objective'
    with pytest.raises(sps.SpecsolveError, match='reduced over the sliced dimension'):
        priced.evaluate('window_spend', original_index=True)


def test_each_slice_expression_matches_solving_that_slice_alone():
    """The fold must not change an expression's value — the oracle is `solve`."""
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    runs = sps.solve_over(spec, scenario_sources(), sps.EachCoordinate('scenario'))

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
    with pytest.raises(sps.SpecsolveError, match="no named expression 'nope' in this sweep"):
        priced.evaluate('nope')


def test_an_expression_no_slice_could_evaluate_carries_its_reason():
    """A failing evaluation is carried per name, never raised mid-fold.

    `ratio` divides by a parameter with one row, so every slice's evaluation
    fails on the sparse divisor — and the sweep must still complete, hold every
    other frame, and hand the caller the divisor's own sentence on read.
    """
    spec = override(
        SPENDING,
        **{'parameters.scale': {'dims': ['t']}, 'expressions.ratio': 'load / scale'},
    )
    sources = {**horizon_sources(12), 'scale': pl.DataFrame({'snapshot': [0], 'value': [2.0]})}
    with pytest.warns(sps.SpecsolveWarning, match="'scale' has no rows for snapshot 1"):
        runs = sps.solve_over(spec, sources, sps.EachWindow('snapshot', steps=6, lookahead=0, into='t'))

    assert runs.primal('p').height > 0, 'the failing expression must not fail the sweep'
    assert runs.evaluate('spend').height > 0, 'nor take the healthy expression with it'
    with pytest.raises(sps.SpecsolveError, match='scale'):
        runs.evaluate('ratio')


#: Six coordinates, three windows of two, whatever the coordinates *are*:
#: `steps` and `lookahead` count coordinates rather than coordinate values.
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

    The local index is dense `0..n-1`, so the seam's `where: "t == 0"` matches
    on a dimension with gaps in it.
    """
    runs = sps.solve_over(
        WINDOW, coordinate_sources(coordinates), sps.EachWindow('snapshot', steps=2, lookahead=0, into='t')
    )

    assert len(runs) == 3
    assert runs.keys == coordinates[::2], 'a window is keyed by its first coordinate'
    soc = runs.primal('soc')
    assert soc.height == 6
    assert sorted(soc['t'].unique().to_list()) == [0, 1], 'the local index is dense per window'


@pytest.mark.parametrize('coordinates', COORDINATE_TYPES)
def test_stitch_recovers_coordinates_no_arithmetic_could(coordinates):
    """`snapshot_start + t` is meaningless for a datetime or a string axis.

    The window→coordinate mapping is the axis's to keep, and stitching is the
    only way back to it: nothing the caller holds could reconstruct these.
    """
    runs = sps.solve_over(
        WINDOW, coordinate_sources(coordinates, load=10.0), sps.EachWindow('snapshot', steps=2, lookahead=0, into='t')
    )
    assert runs.primal('soc', original_index=True)['snapshot'].to_list() == coordinates


def test_a_window_key_column_never_shadows_the_dimension_it_replaced(sweep):
    """`snapshot_start` holds window starts, and there are no snapshots left.

    `EachWindow` drops the global dimension and re-indexes to `into`, so a key
    column called `snapshot` would be window starts sitting under the name of
    the coordinate they are *not* — one that joins cleanly against real
    snapshot-indexed data and silently keeps a twelfth of it.
    """
    runs = sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS)
    soc = runs.primal('soc')
    assert 'snapshot' not in soc.columns
    assert soc.columns[0] == 'snapshot_start'
    assert runs.record.columns[0] == 'snapshot_start', 'both frames key the same way'
    assert sorted(soc['snapshot_start'].unique().to_list()) == [0, 4, 8]

    assert sweep.record.columns[0] == 'scenario', (
        'EachCoordinate keeps the plain name: there the key really is a coordinate of it'
    )


@pytest.mark.parametrize(
    ('geometry', 'expected'),
    [
        pytest.param(
            {'steps': 0, 'lookahead': 0, 'into': 't'}, 'at least one coordinate', id='a-window-keeping-nothing'
        ),
        pytest.param(
            {'steps': [4, 0], 'lookahead': 0, 'into': 't'},
            r'at least one coordinate.*\[0\]',
            id='a-block-keeping-nothing',
        ),
        pytest.param({'steps': [], 'lookahead': 0, 'into': 't'}, 'steps is empty', id='no-blocks-at-all'),
        pytest.param({'steps': 4, 'lookahead': -1, 'into': 't'}, 'is negative', id='a-negative-lookahead'),
        pytest.param({'steps': 4, 'lookahead': 0, 'into': 'snapshot'}, 'must differ from dim', id='into-is-the-dim'),
        pytest.param({'steps': 4, 'lookahead': 0, 'into': ''}, 'no default', id='into-is-empty'),
    ],
)
def test_the_window_geometry_is_checked_at_construction(geometry, expected):
    """A bad geometry is refused when the axis is constructed."""
    with pytest.raises(ValueError, match=expected):
        sps.EachWindow('snapshot', **geometry)


#: Window blocks that are not all the same size. Between them: a telescoping
#: horizon that coarsens, one that refines, months of unequal length, and a
#: sequence overshooting the axis so its tail blocks have nothing to cover.
BLOCKS = [
    pytest.param([1, 2, 3, 6], id='coarsening'),
    pytest.param([6, 3, 2, 1], id='refining'),
    pytest.param([4, 4, 4], id='uniform-spelled-as-a-sequence'),
    pytest.param([5, 7], id='two-unequal-months'),
    pytest.param([5, 7, 99], id='a-block-past-the-end-of-the-axis'),
]


@pytest.mark.parametrize('blocks', BLOCKS)
def test_windows_of_unequal_size_cover_every_coordinate_exactly_once(blocks):
    """`steps` as a sequence keeps those numbers in order, one window each."""
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=blocks, lookahead=2, into='t'),
        carry={'soc_initial': 'soc'},
    )
    stitched = runs.primal('soc', original_index=True)

    assert stitched['snapshot'].to_list() == list(range(12)), 'every coordinate, once, whatever the block sizes'
    assert runs.keys == _starts(blocks, 12), 'one window per block, keyed by the coordinate it starts on'
    assert runs.record['termination_condition'].to_list() == ['optimal'] * len(runs), 'every window solved'


def _starts(blocks: list[int], total: int) -> list[int]:
    """Where each window starts — the keys a block list implies, a block past the end contributing none."""
    starts, at = [], 0
    for block in blocks:
        if at >= total:
            break
        starts.append(at)
        at += block
    return starts


#: Block lists whose last entry overshoots what the axis has left — an int that
#: does not divide, and a sequence whose tail reaches past the end.
OVERSHOOTING = [
    pytest.param(5, 12, [5, 5, 2], id='an-int-that-does-not-divide'),
    pytest.param(7, 12, [7, 5], id='an-int-larger-than-the-remainder'),
    pytest.param(20, 12, [12], id='an-int-larger-than-the-axis'),
    pytest.param([5, 7, 99], 12, [5, 7], id='a-sequence-reaching-past-the-end'),
    pytest.param([5, 20], 12, [5, 7], id='a-sequence-whose-last-block-overshoots'),
]


@pytest.mark.parametrize(('steps', 'periods', 'expected'), OVERSHOOTING)
def test_the_blocks_partition_the_axis_and_never_claim_more_than_is_left(steps, periods, expected):
    """A probe, because no solved sweep can tell `min(block, left)` from `block`.

    Only the *last* block can overshoot, and its carry is never read, so
    trimming it changes no answer. It keeps `_Slice.owns` true, which the
    stitch and the seam both read.
    """
    axis = sps.EachWindow('snapshot', steps=steps, lookahead=0, into='t')

    assert axis._blocks(periods) == expected, 'each block is trimmed to what the axis has left'
    assert sum(axis._blocks(periods)) == periods, 'and together they cover it exactly once'

    slices, _ = axis._slice(horizon_sources(periods), 'snapshot_start')
    assert [current.owns for current in slices] == expected, 'which is what each slice records owning'
    for current in slices:
        assert current.owns <= len(current.sources['t']), 'no window owns more coordinates than it holds'


def test_a_block_list_that_stops_short_of_the_axis_is_refused():
    """The coordinates past the last block would be solved by no window.

    Trimming them silently is the one outcome a sweep must not have: the stitch
    would come back short and read as a complete schedule.
    """
    with pytest.raises(sps.DataError, match=r'keeps 7 coordinate\(s\) across 2 window\(s\).*has 12'):
        sps.solve_over(
            WINDOW,
            horizon_sources(12),
            sps.EachWindow('snapshot', steps=[3, 4], lookahead=0, into='t'),
        )


def test_the_lookahead_the_model_needs_is_one_number_whatever_the_blocks():
    """`Separability.ahead` is one integer for the dimension, so the gate is one check.

    `shift(soc, along=t, offset=-1)` reads one coordinate ahead, so a lookahead
    of zero is refused and one is enough — for uniform blocks and unequal ones
    alike, since no block size enters the arithmetic.
    """
    ahead = 'soc == shift(soc, along=t, offset=-1) + charge * 0.9 - discharge'
    reaching = override(WINDOW, **{'constraints.soc_step.expression': ahead})

    for steps in (3, [1, 2, 3, 6]):
        with pytest.raises(sps.SpecsolveError, match=r'lookahead=0\) looks ahead by 0 coordinate\(s\).*reads 1 ahead'):
            sps.solve_over(
                reaching, horizon_sources(12), sps.EachWindow('snapshot', steps=steps, lookahead=0, into='t')
            )

        runs = sps.solve_over(
            reaching, horizon_sources(12), sps.EachWindow('snapshot', steps=steps, lookahead=1, into='t')
        )
        assert runs.record['termination_condition'].to_list() == ['optimal'] * len(runs), (
            'one coordinate of lookahead is what the model reads, so every window is whole'
        )


def test_a_short_tail_window_carries_off_its_own_last_row():
    """A final window owns fewer rows than ``steps``, and its seam is its own last.

    12 coordinates kept 5 at a time leaves a final window of two, which
    holds no `t == 4`. Nothing reads the last slice's carry, so the value is
    never computed — but a window short of ``steps`` in the *middle* of a sweep
    cannot happen, which is what makes the owned count always in range.
    """
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=5, lookahead=1, into='t'),
        carry={'soc_initial': 'soc'},
    )
    assert runs.keys == [0, 5, 10]
    assert runs.record['termination_condition'].to_list() == ['optimal'] * 3
    assert runs.primal('soc').filter(pl.col('snapshot_start') == 10).height == 2


def test_a_carry_collapses_one_dimension_and_every_other_rides_along():
    """`soc` is over `(t, storage)` and `soc_initial` over `(storage)`.

    The two declarations say what is copied: `t` is what the parameter lacks,
    so `t` is the one the carry collapses, and `storage` passes through — both
    stores are handed forward, each its own level.
    """
    runs = sps.solve_over(MULTI_STORE, multi_store_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'})

    assert runs.keys == [0, 4, 8]
    assert set(runs.primal('soc').columns) == {'snapshot_start', 't', 'storage', 'value'}

    def at(name: str, start: int, t: int, store: str) -> float:
        rows = runs.primal(name).filter(
            (pl.col('snapshot_start') == start) & (pl.col('t') == t) & (pl.col('storage') == store)
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
                'the opening row is the previous window at t == 3, for this same store'
            )

    fresh = sps.solve_over(MULTI_STORE, multi_store_sources(), WINDOW_AXIS)
    assert not fresh.primal('soc').equals(runs.primal('soc')), 'the carry changed nothing'


def test_the_carried_row_is_the_last_one_owned_and_not_the_last_one_solved():
    """Under overlap the two differ, and only the owned one is the state at the seam.

    At `length=6, step=3` a window solves `t` 0..5 and owns 0..2. The lookahead
    rows 3..5 are solved against a horizon that ends at 5, so the store empties
    into them; the next window recomputes those coordinates from its own
    horizon. Handing row 5 forward would seed it with a level that was never
    going to happen.

    The first assertion is what makes the rest discriminating: where the two
    rows hold the same level, reading either passes.
    """
    runs = sps.solve_over(
        WINDOW,
        horizon_sources(12),
        sps.EachWindow('snapshot', steps=3, lookahead=3, into='t'),
        carry={'soc_initial': 'soc'},
    )

    def at(name: str, start: int, t: int) -> float:
        frame = runs.primal(name).filter((pl.col('snapshot_start') == start) & (pl.col('t') == t))
        return frame['value'].item()

    for start in (0, 3, 6):
        assert at('soc', start, 2) != pytest.approx(at('soc', start, 5), abs=1e-6), (
            'the owned row and the last solved row must differ, or this test cannot tell them apart'
        )

    for previous, start in ((0, 3), (3, 6), (6, 9)):
        opened = at('soc', start, 0) - at('charge', start, 0) * 0.9 + at('discharge', start, 0)
        assert opened == pytest.approx(at('soc', previous, 2), abs=1e-6), (
            'the window opens on the last row the previous one owned'
        )
        assert opened != pytest.approx(at('soc', previous, 5), abs=1e-6), (
            'and not on the last row it solved, which is lookahead the next window recomputes'
        )


def test_a_myopic_pathway_carries_a_whole_vector():
    """Capacity per generator, handed forward as a frame rather than a number.

    `total` and `existing` are both over `(generator)`, so nothing is dropped
    and the frame *is* the carry.
    """
    runs = sps.solve_over(
        MYOPIC,
        myopic_sources(),
        sps.EachCoordinate('period'),
        carry={'existing': 'total'},
    )

    assert runs.keys == [1, 2, 3]
    built = runs.primal('build').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    total = runs.primal('total').filter(pl.col('generator') == 'wind').sort('period')['value'].to_list()
    assert built == pytest.approx([10.0, 15.0, 15.0]), (
        'each period builds only the increment: what the last one built came back as `existing`'
    )
    assert total == pytest.approx([10.0, 25.0, 40.0]), 'demand 10 -> 25 -> 40 is met exactly'


#: The five ways a carry cannot line up.
_PERIOD_AXIS = sps.EachCoordinate('period')
UNSOUND_CARRIES = [
    pytest.param(
        WINDOW, horizon_sources, WINDOW_AXIS, {'soc_initial': 'p'},
        r'would collapse .*at once', "['t', 'generator']",
        id='collapses-two-dimensions',
    ),
    pytest.param(
        WINDOW, horizon_sources, WINDOW_AXIS, {'p_max': 'soc'},
        'cannot line up', None,
        id='parameter-over-more-than-the-variable',
    ),
    pytest.param(
        WINDOW, horizon_sources, WINDOW_AXIS, {'soc_initial': 'nope'},
        'does not declare', None,
        id='a-name-neither-side-declares',
    ),
    pytest.param(
        WINDOW, horizon_sources, sps.EachCoordinate('scenario'), {'soc_initial': 'soc'},
        r"collapses 't', and this axis owns none", 'Reduce',
        id='a-coordinate-sweep-collapsing-a-dimension-it-does-not-advance-along',
    ),
    pytest.param(
        MULTI_STORE, multi_store_sources, WINDOW_AXIS, {'load': 'soc'},
        r"collapses 'storage', and this axis advances along 't'", 'Reduce',
        id='a-window-collapsing-a-dimension-that-is-not-its-own',
    ),
]  # fmt: skip


@pytest.mark.parametrize(('spec', 'sources', 'axis', 'carry', 'expected', 'names'), UNSOUND_CARRIES)
def test_a_carry_that_cannot_line_up_says_so_before_anything_solves(spec, sources, axis, carry, expected, names):
    """Every one of these is answerable from the two declarations and the axis alone.

    The axis decides the last two: the coordinate handed on is the last one a
    slice owns, so only the dimension the axis advances along can be the one a
    carry collapses. Any other and there is no coordinate to choose without
    doing the model's arithmetic here.
    """
    with pytest.raises(sps.SpecsolveError, match=expected) as raised:
        sps.solve_over(spec, sources(), axis, carry=carry)
    if names is not None:
        assert names in str(raised.value), 'the message names the dimensions it could not choose between'


def test_a_carry_is_refused_before_a_single_source_is_read(tmp_path):
    """ "Early" has to mean before the data, not merely before the solve.

    The unreadable path is the assertion: reaching it at all means the check
    ran after the data was read.
    """
    missing = tmp_path / 'not-written-yet.parquet'
    sources = {**horizon_sources(), 'load': str(missing)}

    with pytest.raises(sps.SpecsolveError, match='does not declare'):
        sps.solve_over(WINDOW, sources, WINDOW_AXIS, carry={'soc_initial': 'nope'})

    with pytest.raises(Exception, match='not-written-yet') as raised:
        sps.solve_over(WINDOW, sources, WINDOW_AXIS, carry={'soc_initial': 'soc'})
    assert not isinstance(raised.value, sps.SpecsolveError), 'the file, not the carry, is what failed'


# ---------------------------------------------------------------------------
# the fold's own rules
# ---------------------------------------------------------------------------


def test_carry_and_executor_are_refused_together():
    """Sequential by definition, so the combination is a call-time error rather
    than something discovered at slice two."""
    with pytest.raises(sps.SpecsolveError, match='mutually exclusive'):
        sps.solve_over(
            WINDOW,
            horizon_sources(),
            WINDOW_AXIS,
            carry={'soc_initial': 'soc'},
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
    sequential = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'))

    with _entered(make_executor()) as live:
        parallel = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'), executor=live)

    assert parallel.keys == sequential.keys
    assert answer_of(parallel).equals(answer_of(sequential))
    assert parallel.primal('p').equals(sequential.primal('p'))


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_carries_expressions_the_same(make_executor):
    """Expression frames cross the wire the way primals do — encoded and back.

    The process pools are the point: a thread pool never encodes, so only they
    exercise `_encode`/`_decode` on the expression frames a worker returns.
    """
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    sources = scenario_sources()
    sequential = sps.solve_over(spec, sources, sps.EachCoordinate('scenario'))

    with _entered(make_executor()) as live:
        parallel = sps.solve_over(spec, sources, sps.EachCoordinate('scenario'), executor=live)

    assert parallel.evaluate('spend').equals(sequential.evaluate('spend')), (
        'a sweep reads the same named expression under any executor'
    )


def test_a_thread_pool_does_not_encode_for_a_boundary_it_never_crosses(monkeypatch):
    """In-process, so a parquet round trip would be paid for nothing: 31% of a thread-pool sweep, measured.

    Every other executor is assumed to cross, because none of them can be asked.
    """
    seen: list[str] = []
    original = strategy._encode

    def spy(sources, memo, **kwargs):
        seen.append('encoded')
        return original(sources, memo, **kwargs)

    monkeypatch.setattr(strategy, '_encode', spy)
    with ThreadPoolExecutor(2) as pool:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=pool)
    assert seen == [], 'a thread pool encoded its sources'

    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=pool)
    assert seen, 'a process pool did not encode its sources'


def test_the_model_and_its_plan_both_cross_a_process():
    """A worker is handed the document and the lowered plan, under every executor.

    A slice reads no file: it is handed the `Spec`.
    """
    spec = to_spec(DISPATCH)
    serial = sps.solve_over(spec, scenario_sources(), sps.EachCoordinate('scenario'))
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        pooled = sps.solve_over(spec, scenario_sources(), sps.EachCoordinate('scenario'), executor=pool)
    assert answer_of(pooled).equals(answer_of(serial))
    assert pooled.primal('p').equals(serial.primal('p'))


def test_a_failing_slice_reports_the_real_error_across_a_process_boundary():
    """An exception has to survive pickling or the cause is lost.

    A custom ``__init__`` signature is the classic way this breaks, and it
    surfaces as an unrelated ``TypeError`` raised while *unpickling* — so the
    worker's real complaint never arrives.
    """
    broken = {**scenario_sources()}
    broken.pop('cost')
    with (
        ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool,
        pytest.raises(sps.DataError, match="no data provided for parameter 'cost'"),
    ):
        sps.solve_over(DISPATCH, broken, sps.EachCoordinate('scenario'), executor=pool)


def test_a_parquet_path_slices_without_being_read_whole(tmp_path):
    """A path source is scanned, so the per-slice filter pushes into the file."""
    sources = scenario_sources()
    path = tmp_path / 'load.parquet'
    frame = sources.pop('load')
    assert isinstance(frame, pl.DataFrame)
    frame.write_parquet(path)

    runs = sps.solve_over(DISPATCH, {**sources, 'load': str(path)}, sps.EachCoordinate('scenario'))
    assert runs.keys == ['high', 'low', 'mid']
    assert runs.primal('p').height == 3 * 4 * 2


def test_a_path_stays_a_path_for_a_local_pool_and_travels_as_bytes_for_a_remote_one(tmp_path, monkeypatch):
    """`workers_share_fs` is inferred from the pool, and only paths are affected.

    A `ProcessPoolExecutor`'s workers are this machine's, so a path stays a
    path. An executor this package did not ship could be anywhere, so its paths
    travel as their own bytes. `workers_share_fs=` says it outright when the
    guess is wrong.
    """
    sources = scenario_sources()
    path = tmp_path / 'p_max.parquet'
    frame = sources.pop('p_max')
    assert isinstance(frame, pl.DataFrame)
    frame.write_parquet(path)
    sources['p_max'] = str(path)

    crossed: list[object] = []
    original = strategy._encode

    def spy(sliced, memo, **kwargs):
        """Record what `p_max` crossed as; the same helper also encodes answers back."""
        encoded = original(sliced, memo, **kwargs)
        if 'p_max' in encoded:
            crossed.append(encoded['p_max'])
        return encoded

    monkeypatch.setattr(strategy, '_encode', spy)
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        local = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'), executor=pool)
        assert all(v == str(path) for v in crossed), 'a local pool shipped a file it could have opened'

        crossed.clear()
        remote = sps.solve_over(
            DISPATCH, sources, sps.EachCoordinate('scenario'), executor=pool, workers_share_fs=False
        )
        assert all(v == path.read_bytes() for v in crossed), 'the file did not travel as its own bytes'

    assert answer_of(remote).equals(answer_of(local)), 'the path and the bytes are the same numbers'
    assert remote.primal('p').equals(local.primal('p'))

    crossed.clear()
    sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'), executor=Inline())
    assert all(v == path.read_bytes() for v in crossed), 'an executor we did not ship was assumed local'


# ---------------------------------------------------------------------------
# reading a sweep back — Result's readers, one dimension wider
# ---------------------------------------------------------------------------


def test_the_readers_mirror_result_with_the_slice_key_as_one_more_dimension(sweep):
    """Every reader here is `Result`'s under the same name, with the slice key as one more dimension."""
    pytest.importorskip('xarray')
    runs = sweep

    pandas_frame = runs.to_pandas('p')
    assert list(pandas_frame.columns) == ['scenario', 'snapshot', 'generator', 'value']
    assert len(pandas_frame) == 3 * 4 * 2

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
    """The same `kind=` on a sweep's bridges, `original_index` included: a
    stitched price comes back as an array over time, and a dataset of every
    expression is one call."""
    pytest.importorskip('xarray')
    price = priced.to_dataarray('balance', 'dual', original_index=True)
    assert price.dims == ('snapshot',), 'the stitched price is over the dimension the axis sliced'
    assert price.name == 'balance'
    spent = priced.to_dataset(kind='expression')
    assert set(spent.data_vars) == {'spend', 'window_spend'}, 'every expression the slices evaluated'
    assert spent['spend'].dims == ('snapshot_start', 't'), 'keyed by slice, as every bulk export is'
    with pytest.raises(sps.SpecsolveError, match='primal, dual, expression'):
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
        'metrics',
        'owned.parquet',
        'primal',
        'record',
        'sweep.json',
    ], 'the three kinds, the record, the manifest, the layout it is in, and the way back'
    assert sorted(p.name for p in (out / 'expression').iterdir()) == ['spend', 'window_spend'], (
        'every declared expression the slices evaluated'
    )

    built = builds(strategy)
    reopened = _spilled(out)
    assert built == [], 'a directory the export wrote is a sweep already done'
    assert reopened.record.equals(priced.record)
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
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=out)
    (out / 'format.json').write_text(json.dumps({'layout': 99}))

    with pytest.raises(sps.LayoutError, match='layout 99'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=out)
    assert json.loads((out / 'format.json').read_text()) == {'layout': 99}, (
        'and the stamp it was refused over is left as it was found'
    )


def test_a_sweep_keyed_in_more_than_one_type_is_refused(tmp_path) -> None:
    """Refused rather than widened: the caller's own labels are not ours to change."""
    sources = scenario_sources()
    with pytest.raises(sps.SpecsolveError, match='more than one type'):
        sps.solve_over(DISPATCH, sources, [(1, sources), (2.5, sources)], key_name='draw')


def test_a_saved_result_carries_the_row_a_sweep_keys(sweep, tmp_path):
    """One solve's record is one slice's, so cases solved apart concatenate."""
    sources = scenario_sources()
    low = {**sources, 'load': sources['load'].filter(pl.col('scenario') == 'low').drop('scenario')}
    with sps.solve(DISPATCH, low) as alone:
        one = pl.read_parquet(alone.save(tmp_path / 'low') / 'record.parquet')

    assert one.columns == [column for column in sweep.record.columns if column != sweep.key_name], (
        'the fold keys the record it writes; a lone solve writes the same columns unkeyed'
    )
    row = one.row(0, named=True)
    slice_of_the_fold = sweep.record.filter(pl.col('scenario') == 'low').drop('scenario').row(0, named=True)
    assert (row['status'], row['termination_condition']) == (
        slice_of_the_fold['status'],
        slice_of_the_fold['termination_condition'],
    ), 'the lone solve and the slice of the fold terminated the same way'
    assert row['objective'] == pytest.approx(slice_of_the_fold['objective']), (
        'and reached the same number, the two being the same model over the same numbers'
    )


def test_a_bulk_export_of_a_sweep_that_solved_nothing_is_refused():
    """`to_dataset` writes no empty answer: a sweep every slice of which was
    infeasible holds no variable frames, and it refuses with the sentence
    `primal` gives. The names are resolved before xarray is reached, so a bare
    install gets the sentence rather than an ImportError."""
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'))

    with pytest.raises(sps.SpecsolveError, match='holds no variable frames at all'):
        runs.to_dataset()


def test_a_sweep_that_solved_nothing_still_saves_its_records(tmp_path):
    """`save` is not an export, and an infeasible study is an answer.

    A single solve that left no values writes its record and no frames, and a
    sweep of them does the same.
    """
    sources = scenario_sources()
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    runs = sps.solve_over(DISPATCH, sources, sps.EachCoordinate('scenario'))

    out = runs.save(tmp_path / 'sweep')
    records = pl.read_parquet(sorted((out / 'record').glob('*.parquet')))
    assert records['termination_condition'].unique().to_list() == ['infeasible'], (
        'every slice terminated infeasible, and the record says so'
    )
    assert not (out / 'primal').exists(), 'and no frames are written, there being none'
    assert sps.load_sweep(out).record.height == records.height, 'the saved study reads back'


@pytest.mark.parametrize('lost', ['record', 'metrics'], ids=str)
def test_a_sweep_directory_missing_its_record_is_refused_by_name(lost: str, tmp_path) -> None:
    """A manifest with no record beside it is not a sweep this package wrote.

    Both are written per slice as the fold goes, so a directory holding one
    and not the other was edited or interrupted.
    """
    out = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=tmp_path / 'sweep')
    assert sps.load_sweep(out._spill.directory).record.height == 3, 'the whole one reads back first'
    shutil.rmtree(out._spill.directory / lost)

    with pytest.raises(sps.LayoutError, match=f"no '{lost}.parquet'"):
        sps.load_sweep(out._spill.directory)


def test_a_loaded_sweep_is_held_and_a_scanned_one_is_spilled(tmp_path):
    """The two verbs give the same study and differ in where its frames are.

    `scan_sweep` leaves the frames on disk: `scan` reads them, and the readers
    that return a frame refuse. `load_sweep` reads them in, so every reader
    answers and the directory is free afterwards.
    """
    spilled = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=tmp_path / 'sweep')
    expected = spilled.scan('p').collect()
    loaded = sps.load_sweep(tmp_path / 'sweep')
    scanned = sps.scan_sweep(tmp_path / 'sweep')

    assert scanned.scan('p').collect().equals(expected), 'both read the study the spill wrote'
    with pytest.raises(sps.SpecsolveError, match=r'sweep\.scan'):
        scanned.primal('p')
    shutil.rmtree(tmp_path / 'sweep')

    assert loaded.primal('p').equals(expected), 'the held sweep answers the frame readers, off no directory at all'
    assert loaded.keys == scanned.keys, 'and is keyed as the sweep was solved either way'


def test_a_reader_for_a_name_the_sweep_lacks_fails_the_way_primal_does(sweep):
    """One explanation, reached through every reader."""
    for read in (sweep.to_pandas, sweep.to_dataarray):
        with pytest.raises(sps.SpecsolveError, match="no variable 'q' in this sweep"):
            read('q')


def test_a_hand_built_axis_needs_no_class_but_must_name_its_own_key():
    """`axis` also takes a plain list of `(key, sources)`.

    The list cannot say what its keys are coordinates *of*, so `key=` is
    required there.
    """
    base = scenario_sources()
    slices = [(name, {**base, 'load': _draw(base, name)}) for name in ('low', 'high')]

    with pytest.raises(sps.SpecsolveError, match='hand-built axis needs key_name='):
        sps.solve_over(DISPATCH, base, slices)

    runs = sps.solve_over(DISPATCH, base, slices, key_name='draw')
    assert runs.keys == ['low', 'high']
    assert runs.record.columns[0] == 'draw'
    assert runs.primal('p').columns[0] == 'draw', 'both frames key the same way, or they stop joining'


#: The second slice of a two-slice hand-built axis, each naming *less* than the
#: first. Neither class axis can produce one.
NARROWED = [
    pytest.param(lambda base: {'load': _draw(base, 'high'), 'snapshot': range(4)}, id='fewer sources'),
    pytest.param(lambda base: {**base, 'load': _draw(base, 'high', 2)}, id='no index'),
]


@pytest.mark.parametrize('second', NARROWED)
def test_a_hand_built_slice_that_names_less_does_not_inherit_the_last_one(second):
    """A slice says what the whole model attaches, whichever way the sweep runs.

    A serial fold updates and keeps what the last slice attached, where a
    pooled fold builds each slice alone. The failure is a disagreement between
    the two: either outcome on its own reads as an answer.
    """
    base = scenario_sources()
    slices = [('low', {**base, 'load': _draw(base, 'low'), 'snapshot': range(4)}), ('high', second(base))]

    def fold(executor: object) -> object:
        try:
            return sps.solve_over(DISPATCH, base, slices, key_name='draw', executor=executor).objective.to_dicts()
        except sps.DataError as exc:
            return str(exc)

    with ThreadPoolExecutor(2) as pool:
        assert fold(None) == fold(pool), 'a sweep answers the slice it was given, not the one before it'


def test_key_overrides_what_an_axis_derived_and_refuses_a_collision():
    """The derived name is right by default and the caller's word wins.

    A key that is a declared dimension would collide with a column the frames
    carry.
    """
    runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), key_name='case')
    assert runs.record.columns[0] == 'case'
    assert set(runs.primal('p').columns) == {'case', 'snapshot', 'generator', 'value'}

    with pytest.raises(sps.SpecsolveError, match=r"key_name='generator' is a dimension the spec declares"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), key_name='generator')


@pytest.mark.parametrize('key_name', ['specsolve_case', 'Specsolve_case', 'SPECSOLVE_CASE'], ids=str)
def test_a_slice_key_with_the_reserved_prefix_is_refused_in_any_letter_case(key_name):
    """A capital passed the reserved prefix, though a query engine reads `Specsolve_run` as `specsolve_run`."""
    with pytest.raises(sps.SpecsolveError, match=rf"key_name='{key_name}' starts with 'specsolve_'"):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), key_name=key_name)


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
    runs = sps.solve_over(integral, scenario_sources(), sps.EachCoordinate('scenario'))

    assert len(runs) == 3, 'every slice is still a row of the record'
    assert runs.primal('p').height > 0, 'primals are unaffected'
    with pytest.raises(sps.SpecsolveError, match='duals are undefined for a mixed-integer model') as raised:
        runs.dual('balance')
    assert "'p' is not continuous" in str(raised.value), 'the sweep names the variable, as one solve does'


def test_a_bad_name_is_reported_without_the_optional_dependency(sweep):
    """`to_pandas` answers about the model before it asks about the environment.

    A bad name reads the same on every install; a name the sweep does hold
    still needs the dependency, and says which package to install.
    """
    with mock.patch.dict(sys.modules, {'pandas': None}):
        with pytest.raises(sps.SpecsolveError, match="no variable 'q' in this sweep"):
            sweep.to_pandas('q')
        with pytest.raises(ModuleNotFoundError, match='pip install pandas'):
            sweep.to_pandas('p')


# ---------------------------------------------------------------------------
# the model is asked before it is sliced
# ---------------------------------------------------------------------------


def _horizon(constraint: dict, **parameters: dict) -> dict:
    """`WINDOW` with one more constraint over `t`, and any parameter it reads."""
    return override(
        WINDOW,
        parameters={**WINDOW['parameters'], **parameters},
        constraints={**WINDOW['constraints'], 'extra': constraint},
    )


def test_a_window_over_a_horizon_budget_is_refused_with_the_change_that_would_lift_it():
    spec = _horizon({'dims': [], 'expression': 'sum(discharge, over=t) <= 100'})
    with pytest.raises(sps.SpecsolveError, match=r"constraint 'extra': sums over t") as refused:
        sps.solve_over(spec, horizon_sources(8), WINDOW_AXIS)
    assert 'sum_back(window=n)' in str(refused.value), 'the refusal names the rolling form that windows'


def test_a_window_must_look_ahead_as_far_as_the_rows_read():
    """`shift(load, along=t, offset=-2)` reads two rows ahead; a contiguous
    window would read past its end, an overlap of two covers it."""
    spec = _horizon({'dims': ['t'], 'expression': 'sum(p, over=generator) >= shift(load, along=t, offset=-2, edge=0)'})
    with pytest.raises(sps.SpecsolveError, match=r'looks ahead by 0 coordinate\(s\), and the model reads 2 ahead'):
        sps.solve_over(spec, horizon_sources(8), sps.EachWindow('snapshot', steps=4, lookahead=0, into='t'))
    runs = sps.solve_over(spec, horizon_sources(8), sps.EachWindow('snapshot', steps=4, lookahead=2, into='t'))
    assert runs.keys == [0, 4], 'with the lookahead covered, every window solves'


@pytest.mark.parametrize(
    ('delays', 'axis', 'refused'),
    [
        pytest.param(
            [1, 2],
            sps.EachWindow('snapshot', steps=4, lookahead=0, into='t'),
            False,
            id='a-delay-behind-needs-no-overlap',
        ),
        pytest.param(
            [-1, -3],
            sps.EachWindow('snapshot', steps=4, lookahead=0, into='t'),
            True,
            id='a-delay-ahead-needs-the-overlap',
        ),
        pytest.param(
            [-1, -3],
            sps.EachWindow('snapshot', steps=4, lookahead=3, into='t'),
            False,
            id='and-an-overlap-of-three-covers-it',
        ),
    ],
)
def test_an_offset_the_data_decides_is_read_off_the_data(delays, axis, refused):
    """`shift(..., offset=delay)` names a parameter, so the language cannot say
    how far a row reads; the driver reads the values, whose sign says which way."""
    spec = _horizon(
        {'dims': ['t', 'generator'], 'expression': 'p >= shift(p, along=t, offset=delay, edge=0) - 100'},
        delay={'dims': ['generator'], 'dtype': 'int'},
    )
    sources = {**horizon_sources(8), 'delay': pl.DataFrame({'generator': GENERATORS, 'value': delays})}
    if refused:
        with pytest.raises(sps.SpecsolveError, match='the model reads 3 ahead'):
            sps.solve_over(spec, sources, axis)
    else:
        assert len(sps.solve_over(spec, sources, axis)) == 2, 'every window solved'


def test_a_reach_a_relation_decides_is_refused_with_the_relation_named():
    """`shift(..., by=day_of, within=day)` reaches within the groups the relation makes, and
    whether a window cuts a group is nothing the driver computes."""
    spec = _horizon(
        {
            'dims': ['t', 'generator'],
            'expression': 'p >= shift(p, along=t, offset=1, by=day_of, within=day, edge=0) - at(day_cap, by=day_of, over=day, into=t)',
        },
        day_cap={'dims': ['day']},
    )
    spec['dimensions'] = {**spec['dimensions'], 'day': {'dtype': 'int'}}
    spec['relations'] = {'day_of': {'key': 't', 'values': 'day'}}
    with pytest.raises(sps.SpecsolveError, match=r"constraint 'extra': through the relation 'day_of'"):
        sps.solve_over(spec, horizon_sources(8), WINDOW_AXIS)


def test_a_position_the_model_counts_is_a_warning_and_the_windows_still_solve():
    spec = _horizon({'dims': ['t'], 'where': 'position(t) == 0', 'expression': 'soc <= 50'})
    with pytest.warns(sps.SpecsolveWarning, match=r"constraint 'extra': counts a position along t"):
        runs = sps.solve_over(spec, horizon_sources(8), WINDOW_AXIS)
    assert len(runs) == 2, 'a restart is reported, not refused'


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
    spec = _horizon(
        {'dims': ['t', 'generator'], 'expression': 'p >= shift(p, along=t, offset=delay, edge=0) - 100'},
        delay={'dims': ['generator'], 'dtype': 'int'},
    )
    with pytest.raises(sps.SpecsolveError, match='the model reads 3 ahead'):
        sps.solve_over(spec, {**horizon_sources(8), 'delay': delay}, WINDOW_AXIS)


def test_a_window_whose_local_index_the_spec_does_not_declare_is_refused_by_name():
    with pytest.raises(sps.SpecsolveError, match=r"EachWindow\(into='tt'\).*Did you mean 't'") as refused:
        sps.solve_over(WINDOW, horizon_sources(8), sps.EachWindow('snapshot', steps=4, lookahead=0, into='tt'))
    assert 'no such dimension' in str(refused.value), 'the refusal says the spec declares nothing by that name'


def test_a_coordinate_sweep_over_a_dimension_the_spec_declares_is_refused():
    """`EachCoordinate` drops its column, so a declared dimension would be left with no data."""
    with pytest.raises(sps.SpecsolveError, match=r"EachCoordinate\('generator'\) drops 'generator'"):
        sps.solve_over(WINDOW, horizon_sources(8), sps.EachCoordinate('generator'), key_name='g')


# ---------------------------------------------------------------------------
# what a first non-toy sweep runs into
# ---------------------------------------------------------------------------


#: Every shape `build` takes for a source that does not carry the axis: a
#: number for a scalar parameter, a bare sequence for an index, a
#: `{label: value}` map. None of them is a table: the axis has nothing to filter
#: in them.
NOT_A_TABLE = [
    pytest.param(WINDOW, horizon_sources, WINDOW_AXIS, {'soc_initial': 0.0}, id='a-number'),
    pytest.param(DISPATCH, scenario_sources, sps.EachCoordinate('scenario'), {'snapshot': range(4)}, id='a-bare-index'),
    pytest.param(DISPATCH, scenario_sources, sps.EachCoordinate('scenario'), {'cost': {'wind': 1.0, 'gas': 50.0}}, id='a-map'),
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
        runs.record['objective'].to_list() == sps.solve_over(spec, with_tables, axis).record['objective'].to_list()
    ), 'a number, a sequence and a map attach exactly as the tables they stand for'
    with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
        pooled = sps.solve_over(spec, as_plain, axis, executor=pool)
    assert answer_of(pooled).equals(answer_of(runs)), 'the plain shapes cross a process as themselves'


def test_a_source_short_of_a_coordinate_of_the_axis_is_reported():
    """`cost` stops at period 2 while `demand` runs to 3, so period 3 builds with no cost at all.

    The warning comes before a slice is taken, naming the source, the
    coordinate it lacks, and a source that has it.
    """
    sources = myopic_sources()
    sources['cost'] = pl.DataFrame({'period': [1, 1, 2, 2], 'generator': GENERATORS * 2, 'value': [1.0, 50.0] * 2})
    with pytest.warns(sps.SpecsolveWarning, match=r"'cost' has no rows for period 3, which 'demand' has"):
        runs = sps.solve_over(MYOPIC, sources, sps.EachCoordinate('period'), carry={'existing': 'total'})
    assert runs.record['objective'].to_list()[-1] == 0.0, 'the sweep still runs, and period 3 is free'


def test_a_carry_with_no_seed_says_the_first_slice_needs_one():
    """`carry` supplies `soc_initial` from the second slice on; the first has
    nothing to start from, and the error says so rather than reporting a
    parameter with no data as if the carry did not exist.
    """
    sources = horizon_sources(12)
    del sources['soc_initial']
    with pytest.raises(sps.SpecsolveError, match=r"carry writes 'soc_initial' from the second slice on"):
        sps.solve_over(WINDOW, sources, WINDOW_AXIS, carry={'soc_initial': 'soc'})


def test_a_slice_that_leaves_nothing_to_carry_stops_the_sweep_by_name():
    """One infeasible window under a carry: the next window has no level to
    start from, so the sweep cannot go on — and the error names the slice
    that terminated, how, and the slice left waiting.
    """
    sources = horizon_sources(12)
    sources['load'] = sources['load'].with_columns(
        pl.when(pl.col('snapshot') == 5).then(10_000.0).otherwise(pl.col('value')).alias('value')
    )
    with pytest.raises(sps.SpecsolveError, match=r'slice 4 .*infeasible') as raised:
        sps.solve_over(WINDOW, sources, WINDOW_AXIS, carry={'soc_initial': 'soc'})
    assert 'slice 8' in str(raised.value), 'the message names the slice that had nothing to start from'


@pytest.mark.parametrize('make_executor', [pytest.param(None, id='serial'), *EXECUTORS[:2]])
def test_a_failing_slice_is_named(make_executor):
    """Slice three of three fails to build, and the traceback says so.

    The error is the engine's own, with a note added, so a caller matching on
    the message still matches.
    """
    base = scenario_sources()
    slices = [(k, {**base, 'load': _draw(base, k)}) for k in ('low', 'mid')]
    slices.append(('bad', {**slices[0][1], 'load': pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]})}))
    with _entered(make_executor() if make_executor else None) as executor, pytest.raises(sps.DataError) as raised:
        sps.solve_over(DISPATCH, base, slices, key_name='draw', executor=executor)
    assert any("slice 'bad'" in note and '3 of 3' in note for note in raised.value.__notes__), (
        'the note names the slice by key and by position'
    )


@pytest.mark.parametrize('key_name', ['value', 'status', 'termination_condition', 'objective'])
def test_a_key_that_collides_with_a_fixed_column_is_refused(key_name):
    """`value` collides in every frame a reader returns, and the other three
    in `objective` — where the key would silently replace the column rather
    than join it."""
    with pytest.raises(sps.SpecsolveError, match=f'key_name={key_name!r} .* column'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), key_name=key_name)


@pytest.mark.parametrize('make_executor', EXECUTORS[:2])
def test_a_pooled_sweep_parses_the_spec_once(make_executor, monkeypatch):
    """The spec is parsed once per call, whichever executor runs the slices.

    What a worker receives is the document already read, so no slice reads
    the YAML again. Counted at the language's own front door.
    """
    from mathspec import Spec, validation

    from specsolve import lanes

    parsed: list[object] = []
    original = validation.to_spec

    def spy(spec):
        if not isinstance(spec, Spec):
            parsed.append(spec)
        return original(spec)

    monkeypatch.setattr(validation, 'to_spec', spy)
    monkeypatch.setattr(lanes, 'to_spec', spy)
    with _entered(make_executor()) as executor:
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=executor)
    assert len(parsed) == 1, f'the model was parsed {len(parsed)} times for three slices'


def test_an_axis_hands_out_its_slices_so_one_can_be_built_alone():
    """`axis.slices(sources)` is the hand-built list the sweep would have run.

    One slice builds alone, and the list solved hand-built gives the same
    answers under the axis's own key.
    """
    sources = horizon_sources(12)
    axis = sps.EachWindow('snapshot', steps=3, lookahead=3, into='t')
    slices = axis.slices(sources)
    assert [key for key, _ in slices] == [0, 3, 6, 9], 'one slice per window, keyed by where it starts'

    with sps.build(WINDOW, slices[1][1]) as model:
        assert str(model.row('soc_open', t=0)).startswith('soc_open[t=0]'), 'one window builds alone'

    by_axis = sps.solve_over(WINDOW, sources, axis)
    by_hand = sps.solve_over(WINDOW, sources, slices, key_name='snapshot_start')
    assert answer_of(by_hand).equals(answer_of(by_axis))
    assert by_hand.primal('soc').equals(by_axis.primal('soc'))


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
        runs = sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=executor)
    frame = runs.metrics
    assert frame.columns == [
        'scenario',
        'columns',
        'rows',
        'nonzeros',
        'loaded',
        'attach_seconds',
        'build_seconds',
        'handoff_seconds',
        'solve_seconds',
    ], 'the key, then the size, then the one flag, then the clocks in the order the phases run'
    assert frame['scenario'].to_list() == runs.keys
    assert frame['columns'].unique().to_list() == [8], 'every slice is the same model over different numbers'
    assert (frame.select(pl.col('attach_seconds', 'build_seconds', 'solve_seconds') >= 0).to_numpy()).all(), (
        'a clock is never negative'
    )
    assert frame['loaded'].to_list() == ([True, False, False] if executor is None else [True] * 3), (
        'a serial sweep loads the solver once and pushes values after; a pooled one builds every slice cold'
    )


# ---------------------------------------------------------------------------
# spilling to disk
# ---------------------------------------------------------------------------

PRICED_AXIS = sps.EachWindow('snapshot', steps=3, lookahead=3, into='t')
PRICED_CARRY = {'soc_initial': 'soc'}


def _spilled(directory, **kwargs) -> strategy.Sweep:
    return sps.solve_over(SPENDING, horizon_sources(12), PRICED_AXIS, carry=PRICED_CARRY, spill_to=directory, **kwargs)


def test_a_slices_metrics_are_written_in_the_columns_its_type_declares(tmp_path):
    """The fold and the spill both write a slice's metrics as `SliceMetrics` declares them."""
    runs = _spilled(tmp_path / 'sweep')
    written = pl.read_parquet(sorted((tmp_path / 'sweep' / 'metrics').glob('*.parquet')))

    assert written.columns == [runs.key_name, *SliceMetrics._fields], (
        'the key the sweep is cut on, then the metrics in the order the type declares them'
    )
    assert runs.metrics.columns == written.columns, 'the held table is the spilled one, column for column'


def test_a_slice_written_in_another_layout_is_refused_by_name(tmp_path):
    """A resume reads a slice's record back as values, so a file short of a column is refused by name."""
    _spilled(tmp_path / 'sweep')
    first = min((tmp_path / 'sweep' / 'metrics').glob('*.parquet'))
    pl.read_parquet(first).drop('loaded').write_parquet(first)

    with pytest.raises(sps.LayoutError, match=r"SliceMetrics row that is short of \['loaded'\]"):
        _spilled(tmp_path / 'sweep')


def test_a_spilled_sweep_holds_nothing_and_scans_back_what_it_wrote(priced, tmp_path):
    """`spill_to=` writes each slice's frames as the fold goes and keeps none of them.

    What comes back through `scan` is the frame the in-memory reader would
    have returned — primal, dual and expression, keyed or over the original
    index — so the two ways of running a sweep cannot answer differently.
    """
    runs = _spilled(tmp_path)
    assert answer_of(runs).equals(answer_of(priced))
    assert not runs._primals and not runs._duals and not runs._expressions, 'a spilled sweep holds no frame'
    assert runs.scan('soc').collect().equals(priced.primal('soc'))
    assert runs.scan('balance', 'dual').collect().equals(priced.dual('balance'))
    assert runs.scan('spend', 'expression').collect().equals(priced.evaluate('spend'))
    assert runs.scan('soc', original_index=True).collect().equals(priced.primal('soc', original_index=True))
    assert not list(tmp_path.rglob('*.part')), 'every file landed under its final name'


@pytest.mark.parametrize(
    'read',
    [
        pytest.param(lambda runs: runs.primal('soc'), id='primal'),
        pytest.param(lambda runs: runs.dual('balance'), id='dual'),
        pytest.param(lambda runs: runs.evaluate('spend'), id='expression'),
        pytest.param(lambda runs: runs.save('elsewhere'), id='save'),
        pytest.param(lambda runs: runs.to_dataset(), id='to_dataset'),
    ],
)
def test_the_frame_readers_refuse_a_spilled_sweep_and_name_scan(read, tmp_path):
    """One meaning per name: `primal` returns a frame in memory or raises,
    never a frame it would have to read off disk first. The message names `scan`."""
    runs = _spilled(tmp_path)
    with pytest.raises(sps.SpecsolveError, match=r'sweep\.scan'):
        read(runs)


def test_scan_reads_an_in_memory_sweep_too(sweep):
    """`scan` means the same thing on both: code written for a spilled sweep
    runs unchanged on one that fit in memory."""
    assert sweep.scan('p').collect().equals(sweep.primal('p'))
    with pytest.raises(sps.SpecsolveError, match="no variable 'nope'"):
        sweep.scan('nope')
    with pytest.raises(sps.SpecsolveError, match='primal, dual, expression'):
        sweep.scan('p', 'objective')


def test_a_spilled_sweep_resumes_after_the_slice_that_failed(builds, tmp_path):
    """The slices that solved before the failure are not solved again.

    Three hand-built slices, the third of which cannot build; the second run
    with the same directory builds one model, and comes back identical to a
    sweep that never failed.
    """
    base = scenario_sources()
    good = [(k, {**base, 'load': _draw(base, k)}) for k in ('low', 'mid', 'high')]
    bad = [*good[:2], ('high', {**good[0][1], 'load': pl.DataFrame({'snapshot': [0, 1], 'value': [1.0, 2.0]})})]
    with pytest.raises(sps.DataError):
        sps.solve_over(DISPATCH, base, bad, key_name='draw', spill_to=tmp_path)

    built = builds(strategy)
    resumed = sps.solve_over(DISPATCH, base, good, key_name='draw', spill_to=tmp_path)
    assert len(built) == 1, 'only the slice that failed is built again'

    fresh = sps.solve_over(DISPATCH, base, good, key_name='draw')
    assert answer_of(resumed).equals(answer_of(fresh))
    assert resumed.scan('p').collect().equals(fresh.primal('p'))


def test_a_resumed_carry_reads_its_state_off_the_disk(priced, monkeypatch, tmp_path):
    """A rolling horizon interrupted after two windows continues from the
    second window's file, and ends where an uninterrupted one does."""
    answered = strategy._answers
    seen: list[int] = []

    def two_then_fail(*args):
        if len(seen) == 2:
            raise RuntimeError('the box went away')
        seen.append(1)
        return answered(*args)

    monkeypatch.setattr(strategy, '_answers', two_then_fail)
    with pytest.raises(RuntimeError, match='went away'):
        _spilled(tmp_path)
    monkeypatch.setattr(strategy, '_answers', answered)

    resumed = _spilled(tmp_path)
    assert answer_of(resumed).equals(answer_of(priced))
    assert resumed.scan('soc', original_index=True).collect().equals(priced.primal('soc', original_index=True))
    loaded = priced.metrics['loaded'].to_list()
    loaded[2] = True
    assert resumed.metrics['loaded'].to_list() == loaded, (
        'the two read back are the record they left, and the third loads where the uninterrupted run updated'
    )


def test_a_slice_written_part_way_is_solved_again(builds, tmp_path):
    """The objective file is written last and is what marks a slice done, so
    a slice whose frames landed but whose record did not is solved again."""
    _spilled(tmp_path)
    (tmp_path / 'record' / '000001.parquet').unlink()
    built = builds(strategy)
    resumed = _spilled(tmp_path)
    assert len(built) == 1, 'the slice without its record is the one built'
    assert resumed.keys == [0, 3, 6, 9], 'the sweep comes back whole'


def test_a_directory_holding_another_sweep_is_refused(tmp_path):
    """A directory answers for one sweep. Another one pointed at it would read
    the first one's slices back as its own, so the mismatch is refused."""
    _spilled(tmp_path)
    with pytest.raises(sps.SpecsolveError, match='holds a sweep keyed by'):
        sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=tmp_path)


@pytest.mark.parametrize('make_executor', EXECUTORS)
def test_every_executor_spills_the_same_files(make_executor, sweep, tmp_path):
    """Under a pool the answers still land in the directory, in slice order."""
    with _entered(make_executor()) as executor:
        runs = sps.solve_over(
            DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=executor, spill_to=tmp_path
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
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), spill_to=tmp_path)
    (tmp_path / 'record' / '000001.parquet').unlink()
    built = builds(strategy)
    with ThreadPoolExecutor(2) as pool:
        resumed = sps.solve_over(
            DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), executor=pool, spill_to=tmp_path
        )
    assert len(built) == 1, 'the slice without its record is the one submitted'
    assert resumed.keys == ['high', 'low', 'mid'], 'the sweep comes back whole and in order'


def test_scan_on_a_spilled_sweep_says_what_it_does_hold(tmp_path):
    """A name no slice wrote has no directory, and the message lists the
    names that do — the same sentence the in-memory reader gives."""
    runs = _spilled(tmp_path)
    with pytest.raises(sps.SpecsolveError, match=r"no variable 'nope' in this sweep — it holds 'charge', 'discharge'"):
        runs.scan('nope')


def test_an_export_reads_the_key_off_each_frame_and_skips_an_empty_one(sweep):
    """A name is held per slice that produced it, not per slice, so an export
    cannot count positions: it reads the key off each frame, and a frame with
    no rows — a variable every row of which a slice masked — carries none and
    is left out, which is what the spill writes for it."""
    frames = list(sweep._primals['p'])
    empty = frames[0].clear()
    by_key = strategy._by_key([empty, *frames], sweep.key_name)
    assert list(by_key) == ['high', 'low', 'mid'], 'one entry per frame that has rows, keyed by its own key'
    assert all(sweep.key_name not in frame.columns for frame in by_key.values()), 'the key column is dropped'


def test_a_sweep_archive_carries_its_carry(tmp_path):
    """The carry is config the frames do not hold, so the archive stores it beside the axis."""
    sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll')
    assert sps.load_archive(tmp_path / 'roll').carry == {'soc_initial': 'soc'}, 'the carry reads back as it was given'


def test_a_sweep_archive_with_no_carry_reads_an_empty_carry(tmp_path):
    """A sweep that chained nothing carries nothing — the manifest omits the key and the reader defaults it."""
    sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, archive=tmp_path / 'plain')
    assert sps.load_archive(tmp_path / 'plain').carry == {}, 'no carry given, none stored, an empty mapping read back'


def test_a_carried_sweep_reruns_from_its_archive_with_the_stored_carry(tmp_path):
    """The stored carry is what makes a re-run the same sweep: with it the chained answer is reproduced."""
    original = sps.solve_over(
        WINDOW, horizon_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll'
    )
    packed = sps.load_archive(tmp_path / 'roll')
    rerun = sps.solve_over(packed.spec, packed.sources, packed.axis, carry=packed.carry)
    assert rerun.primal('soc').equals(original.primal('soc')), 'the re-run with the stored carry matches the archive'


def test_a_sweep_archive_evaluates_a_quantity_the_file_never_named_per_slice(tmp_path):
    """`Sweep.evaluate` reads an undeclared quantity at each slice's own solution — matches solving that slice alone."""
    axis = sps.EachCoordinate('scenario')
    sps.solve_over(DISPATCH, scenario_sources(), axis, archive=tmp_path / 'study.zip')
    sweep = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'out')
    expr = 'sum(p * cost, over=generator)'
    swept = sweep.answer.evaluate(expr)
    for key, slice_sources in axis.slices(scenario_sources()):
        live = sps.solve(DISPATCH, slice_sources).evaluate(expr)
        got = swept.filter(pl.col(sweep.answer.key_name) == key).drop(sweep.answer.key_name)
        columns = live.columns[:-1]
        assert got.sort(columns).equals(live.sort(columns)), f'slice {key!r} evaluates at its own primal, no re-solve'


def test_a_scanned_sweep_archive_evaluates_the_same(tmp_path):
    """A sweep left on disk (`scan_archive`) evaluates against those frames, the same values held reads."""
    sps.solve_over(DISPATCH, scenario_sources(), sps.EachCoordinate('scenario'), archive=tmp_path / 'study.zip')
    expr = 'sum(p * cost, over=generator)'
    whole = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'whole').answer.evaluate(expr)
    scanned = sps.scan_archive(tmp_path / 'study.zip', tmp_path / 'scan').answer.evaluate(expr)
    assert scanned.equals(whole), 'a scanned sweep evaluates against the frames on disk, the same answer'


def test_a_live_sweep_has_no_model_to_evaluate_against():
    """A Sweep a live solve returned retains no model, so an undeclared expression says why — the archive is what carries one — while a declared name is stitched from what the sweep holds."""
    spec = override(DISPATCH, **{'expressions.spend': 'sum(p * cost, over=generator)'})
    runs = sps.solve_over(spec, scenario_sources(), sps.EachCoordinate('scenario'))
    with pytest.raises(sps.SpecsolveError, match='no model behind it'):
        runs.evaluate('sum(p, over=generator)')
    assert runs.evaluate('spend')['scenario'].n_unique() == len(runs), (
        'the declared name answers without a model, stitched from every slice'
    )


def test_evaluate_across_a_sweep_refuses_an_expression_that_reads_a_carried_parameter(tmp_path):
    """The narrow gap: a carried value is a previous slice's answer, not stored data, so evaluate refuses it."""
    sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll.zip')
    sweep = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll')
    assert sweep.answer.evaluate('sum(p * cost)').height, 'an expression over static data evaluates per slice'
    with pytest.raises(sps.SpecsolveError, match='carried'):
        sweep.answer.evaluate('soc_initial')


def test_evaluate_over_the_original_index_reindexes_like_primal(tmp_path):
    """`evaluate(original_index=True)` reuses the reindex `primal` does — the sliced dim back, the slice key gone."""
    sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, archive=tmp_path / 'roll.zip')
    answer = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll').answer
    reindexed = answer.evaluate('sum(p, over=generator)', original_index=True)
    assert reindexed.columns == ['snapshot', 'value'], 'the sliced dim is restored and the slice key dropped'
    by_hand = answer.primal('p', original_index=True).group_by('snapshot').agg(pl.col('value').sum()).sort('snapshot')
    assert reindexed.sort('snapshot').equals(by_hand.select('snapshot', 'value')), (
        'the evaluated expression reindexed equals the primal reindexed and summed by hand'
    )


def test_evaluate_over_the_original_index_refuses_a_quantity_reduced_over_the_sliced_dim(tmp_path):
    """A scalar-per-window quantity has no local index to restore, so original_index refuses it — as `expression` does."""
    sps.solve_over(WINDOW, horizon_sources(), WINDOW_AXIS, archive=tmp_path / 'roll.zip')
    answer = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll').answer
    with pytest.raises(sps.SpecsolveError, match="over 'snapshot'"):
        answer.evaluate('sum(p * cost)', original_index=True)


#: An index of another dimension that carries the axis column, under each axis.
CUT_INDEXES = [
    pytest.param(
        MYOPIC,
        lambda: {
            **myopic_sources(),
            'generator': pl.DataFrame({'generator': ['wind', 'gas', 'wind', 'gas'], 'period': [1, 1, 2, 2]}),
        },
        sps.EachCoordinate('period'),
        "index for dimension 'generator' carries a 'period' column, and EachCoordinate('period')",
        id='each-coordinate',
    ),
    pytest.param(
        WINDOW,
        lambda: {**horizon_sources(8), 'generator': pl.DataFrame({'generator': GENERATORS, 'snapshot': [0, 1]})},
        sps.EachWindow('snapshot', steps=4, lookahead=0, into='t'),
        "index for dimension 'generator' carries a 'snapshot' column, and EachWindow('snapshot')",
        id='each-window',
    ),
]


@pytest.mark.parametrize(('spec', 'sources', 'axis', 'match'), CUT_INDEXES)
def test_an_index_of_another_dimension_that_carries_the_axis_is_refused_before_a_slice(spec, sources, axis, match):
    """An index lists the labels every slice has; which of them a slice has is a parameter or a relation.

    Cut by the axis, the index would make each slice a model over other labels.
    """
    with (
        mock.patch.object(type(axis), '_slice', side_effect=AssertionError('the axis cut the sources')),
        pytest.raises(sps.DataError, match=re.escape(match)) as refused,
    ):
        sps.solve_over(spec, sources(), axis)
    assert 'in a parameter or a relation over' in str(refused.value), 'the message names the rewrite'
