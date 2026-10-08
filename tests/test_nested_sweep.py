"""A sweep cut by several axes, outer first: one chain of slices per combination of the outer keys.

The oracle is the loop it replaces. Each chain of a nested sweep answers as
the sweep of that chain's sources alone would, carry and start included, so
every test here compares against `solve_over` run once per outer key.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import replace

import polars as pl
import pytest

import specsolve as sps
from specsolve import api, strategy
from tests.test_strategy import MYOPIC, WINDOW, WINDOW_AXIS, horizon_sources, myopic_sources

AXES = (sps.EachCoordinate('scenario'), WINDOW_AXIS)
CARRY = {'soc_initial': 'soc'}
#: The windows of `WINDOW_AXIS`, handing the store's level on to the next.
CARRIED_WINDOW = replace(WINDOW_AXIS, carry=CARRY)
#: A rolling horizon per scenario: the windows carry, the scenarios do not.
CARRIED = (sps.EachCoordinate('scenario'), CARRIED_WINDOW)
SCENARIOS = {'low': 1.0, 'high': 1.5}


def scenario_horizons() -> dict[str, object]:
    """Twelve snapshots of load under each scenario, `high` half as large again as `low`.

    The store starts each scenario at 50 and every window empties it, so a
    window that took the seed rather than the level carried to it would solve
    another problem.
    """
    base = horizon_sources(12)
    load = pl.concat(
        [
            base['load'].with_columns(pl.lit(name).alias('scenario'), pl.col('value') * scale)
            for name, scale in SCENARIOS.items()
        ]
    )
    return {**base, 'load': load, 'soc_initial': pl.DataFrame({'value': [50.0]})}


def alone(scenario: str, **options) -> sps.types.Sweep:
    """The sweep of one scenario's sources, run by itself: what each chain of the nested sweep must equal."""
    sources = scenario_horizons()
    load = sources['load'].filter(pl.col('scenario') == scenario).drop('scenario')
    return sps.solve_over(WINDOW, {**sources, 'load': load}, CARRIED_WINDOW, **options)


def same(got: pl.DataFrame, expected: pl.DataFrame) -> None:
    """The two frames hold one answer: a solver may write zero as ``-0.0``, which ``equals`` tells apart."""
    assert got.drop('value').equals(expected.drop('value')), 'the same coordinates, in the same order'
    assert got['value'].to_list() == pytest.approx(expected['value'].to_list(), abs=1e-9), 'the same values'


def chain_of(frame: pl.DataFrame, scenario: str) -> pl.DataFrame:
    return frame.filter(pl.col('scenario') == scenario).drop('scenario')


@pytest.fixture(scope='module')
def nested() -> sps.types.Sweep:
    return sps.solve_over(WINDOW, scenario_horizons(), CARRIED)


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_each_chain_answers_as_its_own_sweep_would(nested, scenario):
    """The carry hands the level on within a scenario and restarts from the seed at the next one."""
    own = alone(scenario)
    same(chain_of(nested.primal('soc'), scenario), own.primal('soc'))
    objectives = chain_of(nested.record, scenario)['objective'].to_list()
    assert objectives == pytest.approx(own.record['objective'].to_list()), 'every window solves as it would alone'


def test_the_keys_name_every_axis_outer_first(nested):
    assert nested.key_names == ('scenario', 'snapshot_start'), 'one key column per axis, outer first'
    assert nested.keys == [('high', 0), ('high', 4), ('high', 8), ('low', 0), ('low', 4), ('low', 8)], (
        'a key per slice, outer keys sorted, then the windows in order'
    )
    assert nested.record.columns[:2] == ['scenario', 'snapshot_start'], 'the record leads with the key columns'
    assert nested.record['slice'].to_list()[:2] == ['high/0', 'high/4'], 'the record names a slice by its labels'
    assert nested.primal('soc').columns == ['scenario', 'snapshot', 'value'], 'the answer keeps the outer key'
    assert nested.primal('soc', per_window=True).columns == ['scenario', 'snapshot_start', 't', 'value'], (
        'per window, every key column and the local index'
    )


def test_previous_starts_each_chain_cold(monkeypatch):
    """`start='previous'` follows the chain: the first window of every scenario starts from nothing."""
    given = []
    original = api.Model.solve
    monkeypatch.setattr(api.Model, 'solve', lambda self, **kw: given.append(kw.get('start')) or original(self, **kw))
    runs = sps.solve_over(WINDOW, scenario_horizons(), CARRIED, start='previous')

    assert [start is None for start in given] == [True, False, False, True, False, False], (
        'cold at the first window of each scenario, and from the window before everywhere else'
    )
    for scenario in SCENARIOS:
        own = alone(scenario, start='previous').record['objective'].to_list()
        assert chain_of(runs.record, scenario)['objective'].to_list() == pytest.approx(own), (
            'each window reaches the optimum it reaches alone; a start may end the LP on another optimal vertex'
        )


def _threads():
    return ThreadPoolExecutor(2)


def _processes():
    return ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'))


@pytest.mark.parametrize('pool', [_threads, _processes], ids=['threads', 'processes'])
def test_an_executor_runs_the_chains_concurrently_and_each_in_order(nested, pool):
    """A carry no longer rules an executor out where there is more than one chain."""
    with pool() as executor:
        runs = sps.solve_over(WINDOW, scenario_horizons(), CARRIED, executor=executor)
    same(runs.primal('soc'), nested.primal('soc'))
    assert runs.keys == nested.keys


def test_under_an_executor_previous_starts_each_chain_cold_once(monkeypatch):
    """Each chain is one task, so only its first slice starts from nothing."""
    given = []
    original = api.Model.solve
    monkeypatch.setattr(api.Model, 'solve', lambda self, **kw: given.append(kw.get('start')) or original(self, **kw))
    with ThreadPoolExecutor(2) as executor:
        sps.solve_over(WINDOW, scenario_horizons(), CARRIED, start='previous', executor=executor)
    assert sum(start is None for start in given) == len(SCENARIOS), 'one cold start per chain, whatever the order'


def test_an_executor_on_a_sweep_of_one_chain_still_refuses_a_carry():
    with ThreadPoolExecutor(2) as executor, pytest.raises(sps.errors.SpecsolveError, match='one chain'):
        sps.solve_over(WINDOW, horizon_sources(12), CARRIED_WINDOW, executor=executor)


def test_a_spilled_nested_sweep_resumes_without_solving(nested, tmp_path, monkeypatch):
    sps.solve_over(WINDOW, scenario_horizons(), CARRIED, spill_to=tmp_path / 'spill')
    built = []
    monkeypatch.setattr(strategy, 'build', lambda *a, **k: built.append(a))
    again = sps.solve_over(WINDOW, scenario_horizons(), CARRIED, spill_to=tmp_path / 'spill')

    assert not built, 'every slice is on disk, so none is built again'
    same(again.primal('soc'), nested.primal('soc'))
    assert sps.load_sweep(tmp_path / 'spill').key_names == nested.key_names


def test_a_nested_sweep_archive_holds_its_axes_and_runs_again(nested, tmp_path):
    sps.solve_over(WINDOW, scenario_horizons(), CARRIED, archive=tmp_path / 'run', keep_windows=True)
    archive = sps.load_archive(tmp_path / 'run')

    assert archive.axis == CARRIED, 'the axes read back with their carry'
    same(archive.sweep.primal('soc'), nested.primal('soc'))
    same(archive.sweep.primal('soc', per_window=True), nested.primal('soc', per_window=True))
    again = sps.solve_over(archive.spec, archive.sources, archive.axis)
    same(again.primal('soc'), nested.primal('soc'))


def test_an_undeclared_expression_reads_off_a_nested_archive(nested, tmp_path):
    sps.solve_over(WINDOW, scenario_horizons(), CARRIED, archive=tmp_path / 'run', keep_windows=True)
    doubled = sps.load_archive(tmp_path / 'run').sweep.evaluate('soc * 2')
    expected = nested.primal('soc').with_columns(pl.col('value') * 2)
    same(doubled, expected)


def test_a_nested_sweep_starts_from_an_earlier_one(nested, monkeypatch):
    """Each window takes its own scenario's rows of the earlier answer, cut by both axes, over its local index."""
    given = []
    original = api.Model.solve
    monkeypatch.setattr(api.Model, 'solve', lambda self, **kw: given.append(kw.get('start')) or original(self, **kw))
    again = sps.solve_over(WINDOW, scenario_horizons(), CARRIED, start=nested)

    windows = nested.primal('soc', per_window=True)
    for key, start in zip(nested.keys, given, strict=True):
        own = windows.filter(pl.col('scenario') == key[0], pl.col('snapshot_start') == key[1])
        same(start['primal']['soc'].select('t', 'value').sort('t'), own.select('t', 'value').sort('t'))
    assert again.record['objective'].to_list() == pytest.approx(nested.record['objective'].to_list()), (
        'the same optimum, though the LP may end on another of its optimal vertices'
    )


def test_three_axes_cut_in_turn():
    sources = scenario_horizons()
    years = pl.concat([sources['load'].with_columns(pl.lit(year).alias('year')) for year in (2030, 2040)])
    runs = sps.solve_over(WINDOW, {**sources, 'load': years}, (sps.EachCoordinate('year'), *CARRIED))

    assert runs.key_names == ('year', 'scenario', 'snapshot_start'), 'one key column per axis, outer first'
    assert len(runs) == 2 * 2 * 3, 'two years, two scenarios, three windows each'
    one = runs.primal('soc').filter(pl.col('year') == 2040).drop('year')
    same(chain_of(one, 'high'), alone('high').primal('soc'))


def test_two_coordinate_axes_chain_the_inner_one():
    """A myopic pathway per scenario: the build of one period is the next period's existing fleet."""
    base = myopic_sources()
    demand = pl.concat(
        [
            base['demand'].with_columns(pl.lit(name).alias('scenario'), pl.col('value') * scale)
            for name, scale in SCENARIOS.items()
        ]
    )
    carry = {'existing': 'total'}
    runs = sps.solve_over(
        MYOPIC, {**base, 'demand': demand}, (sps.EachCoordinate('scenario'), sps.EachCoordinate('period', carry=carry))
    )
    for scenario in SCENARIOS:
        mine = demand.filter(pl.col('scenario') == scenario).drop('scenario')
        own = sps.solve_over(MYOPIC, {**base, 'demand': mine}, sps.EachCoordinate('period', carry=carry))
        same(chain_of(runs.primal('total'), scenario), own.primal('total'))


@pytest.mark.parametrize(
    ('axis', 'options', 'match'),
    [
        pytest.param((WINDOW_AXIS, sps.EachCoordinate('scenario')), {}, 'only be the last axis', id='window-outside'),
        pytest.param(
            (sps.EachCoordinate('scenario'), sps.EachCoordinate('scenario')), {}, 'Cut each dimension once', id='twice'
        ),
        pytest.param((sps.EachCoordinate('scenario'), [('a', {})]), {}, 'neither', id='not-an-axis'),
        pytest.param(AXES, {'key_name': 'case'}, 'names its own key column', id='key-name'),
    ],
)
def test_a_tuple_of_axes_that_cannot_cut_is_refused_before_a_slice_is_solved(axis, options, match, monkeypatch):
    built = []
    monkeypatch.setattr(strategy, 'build', lambda *a, **k: built.append(a))
    with pytest.raises(sps.errors.SpecsolveError, match=match):
        sps.solve_over(WINDOW, scenario_horizons(), axis, **options)
    assert not built, 'refused before any slice is built'


#: A rolling horizon that also builds: each window may add capacity on top of
#: what `existing` says is there, and the store carries its level as `WINDOW`'s does.
BUILDING = {
    'dimensions': {'t': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'cost': {'dims': ['generator']},
        'build_cost': {'dims': ['generator']},
        'existing': {'dims': ['generator']},
        'load': {'dims': ['t']},
        'soc_initial': {'dims': []},
    },
    'variables': {
        'build': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 100}},
        'capacity': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 200}},
        'p': {'dims': ['t', 'generator'], 'bounds': {'lower': 0}},
        'charge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'discharge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 30}},
        'soc': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'accumulate': {'dims': ['generator'], 'expression': 'capacity == existing + build'},
        'limit': {'dims': ['t', 'generator'], 'expression': 'p <= capacity'},
        'balance': {'dims': ['t'], 'expression': 'sum(p, over=generator) + discharge - charge == load'},
        'soc_open': {'dims': ['t'], 'where': 't == 0', 'expression': 'soc == soc_initial + charge * 0.9 - discharge'},
        'soc_step': {
            'dims': ['t'],
            'where': 't > 0',
            'expression': 'soc == shift(soc, along=t, offset=1) + charge * 0.9 - discharge',
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost) + sum(build * build_cost)'},
}
PERIODS = {2030: 1.0, 2040: 2.0}
FLEET = ['gas', 'solar']


def building_sources(period: int | None = None) -> dict[str, object]:
    """Eight hours of load per period, doubling from 2030 to 2040; one period's alone when *period* is given."""
    load = horizon_sources(8)['load']
    rows = pl.concat(
        [load.with_columns(pl.lit(year).alias('period'), pl.col('value') * k) for year, k in PERIODS.items()]
    )
    if period is not None:
        rows = rows.filter(pl.col('period') == period).drop('period')
    return {
        'generator': pl.DataFrame({'generator': FLEET}),
        'cost': pl.DataFrame({'generator': FLEET, 'value': [50.0, 1.0]}),
        'build_cost': pl.DataFrame({'generator': FLEET, 'value': [10.0, 30.0]}),
        'existing': pl.DataFrame({'generator': FLEET, 'value': [20.0, 0.0]}),
        'load': rows,
        'soc_initial': pl.DataFrame({'value': [50.0]}),
    }


def test_each_axis_carries_its_own_state():
    """Periods hand on the fleet, windows hand on the store.

    The fleet a period leaves is the capacity of its last window, and it is
    every window of the next period's `existing`. The store restarts from its
    seed at each period, as each window's level reaches only the next window.
    The oracle is the loop the nested sweep replaces: one carried horizon per
    period, the fleet copied across by hand.
    """
    windows = replace(WINDOW_AXIS, carry=CARRY)
    runs = sps.solve_over(
        BUILDING, building_sources(), (sps.EachCoordinate('period', carry={'existing': 'capacity'}), windows)
    )

    existing = building_sources()['existing']
    for period in PERIODS:
        own = sps.solve_over(BUILDING, {**building_sources(period), 'existing': existing}, windows)
        mine = runs.record.filter(pl.col('period') == period)['objective'].to_list()
        assert mine == pytest.approx(own.record['objective'].to_list()), f'{period} solves as its own horizon would'
        last = own.keys[-1]
        existing = (
            own.primal('capacity', per_window=True).filter(pl.col('snapshot_start') == last).drop('snapshot_start')
        )
    assert runs.record['objective'].to_list()[2] != pytest.approx(runs.record['objective'].to_list()[0]), (
        "2040 starts from 2030's fleet, not the seed, so its first window solves another problem"
    )


def test_a_parameter_two_axes_carry_is_refused():
    """Each carry alone would line up; together a slice would take two values for `existing`."""
    carry = {'existing': 'total'}
    base = myopic_sources()
    demand = pl.concat([base['demand'].with_columns(pl.lit(name).alias('scenario')) for name in SCENARIOS])
    axes = (sps.EachCoordinate('scenario', carry=carry), sps.EachCoordinate('period', carry=carry))
    with pytest.raises(sps.errors.SpecsolveError, match='Carry each parameter on one axis'):
        sps.solve_over(MYOPIC, {**base, 'demand': demand}, axes)


def test_an_outer_carry_makes_its_axis_part_of_the_chain():
    """Periods hand the fleet on, so they cannot run apart: the whole pathway is one chain, and an executor is refused."""
    axes = (
        sps.EachCoordinate('period', carry={'existing': 'capacity'}),
        replace(WINDOW_AXIS, carry=CARRY),
    )
    with ThreadPoolExecutor(2) as executor, pytest.raises(sps.errors.SpecsolveError, match='one chain'):
        sps.solve_over(BUILDING, building_sources(), axes, executor=executor)
