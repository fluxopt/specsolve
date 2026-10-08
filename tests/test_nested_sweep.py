"""A sweep cut by several axes, outer first: one chain of slices per combination of the outer keys.

The oracle is the loop it replaces. Each chain of a nested sweep answers as
the sweep of that chain's sources alone would, carry and start included, so
every test here compares against `solve_over` run once per outer key.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import polars as pl
import pytest

import specsolve as sps
from specsolve import api, strategy
from tests.test_strategy import MYOPIC, WINDOW, WINDOW_AXIS, horizon_sources, myopic_sources

AXES = (sps.EachCoordinate('scenario'), WINDOW_AXIS)
CARRY = {'soc_initial': 'soc'}
SCENARIOS = {'low': 1.0, 'high': 1.5}


def scenario_horizons() -> dict[str, object]:
    """Twelve snapshots of load under each scenario, `high` half as large again as `low`."""
    base = horizon_sources(12)
    load = pl.concat(
        [
            base['load'].with_columns(pl.lit(name).alias('scenario'), pl.col('value') * scale)
            for name, scale in SCENARIOS.items()
        ]
    )
    return {**base, 'load': load}


def alone(scenario: str, **options) -> sps.types.Sweep:
    """The sweep of one scenario's sources, run by itself: what each chain of the nested sweep must equal."""
    sources = scenario_horizons()
    load = sources['load'].filter(pl.col('scenario') == scenario).drop('scenario')
    return sps.solve_over(WINDOW, {**sources, 'load': load}, WINDOW_AXIS, **options)


def chain_of(frame: pl.DataFrame, scenario: str) -> pl.DataFrame:
    return frame.filter(pl.col('scenario') == scenario).drop('scenario')


@pytest.fixture(scope='module')
def nested() -> sps.types.Sweep:
    return sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY)


@pytest.mark.parametrize('scenario', sorted(SCENARIOS))
def test_each_chain_answers_as_its_own_sweep_would(nested, scenario):
    """The carry hands the level on within a scenario and restarts from the seed at the next one."""
    own = alone(scenario, carry=CARRY)
    assert chain_of(nested.primal('soc'), scenario).equals(own.primal('soc'))
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
    runs = sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, start='previous')

    assert [start is None for start in given] == [True, False, False, True, False, False], (
        'cold at the first window of each scenario, and from the window before everywhere else'
    )
    for scenario in SCENARIOS:
        assert chain_of(runs.primal('soc'), scenario).equals(alone(scenario, carry=CARRY).primal('soc'))


def _threads():
    return ThreadPoolExecutor(2)


def _processes():
    return ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn'))


@pytest.mark.parametrize('pool', [_threads, _processes], ids=['threads', 'processes'])
def test_an_executor_runs_the_chains_concurrently_and_each_in_order(nested, pool):
    """A carry no longer rules an executor out where there is more than one chain."""
    with pool() as executor:
        runs = sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, start='previous', executor=executor)
    assert runs.primal('soc').equals(nested.primal('soc'))
    assert runs.keys == nested.keys


def test_an_executor_on_a_sweep_of_one_chain_still_refuses_a_carry():
    with ThreadPoolExecutor(2) as executor, pytest.raises(sps.errors.SpecsolveError, match='one chain'):
        sps.solve_over(WINDOW, horizon_sources(12), WINDOW_AXIS, carry=CARRY, executor=executor)


def test_a_spilled_nested_sweep_resumes_without_solving(nested, tmp_path, monkeypatch):
    sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, spill_to=tmp_path / 'spill')
    built = []
    monkeypatch.setattr(strategy, 'build', lambda *a, **k: built.append(a))
    again = sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, spill_to=tmp_path / 'spill')

    assert not built, 'every slice is on disk, so none is built again'
    assert again.primal('soc').equals(nested.primal('soc'))
    assert sps.load_sweep(tmp_path / 'spill').key_names == nested.key_names


def test_a_nested_sweep_archive_holds_its_axes_and_runs_again(nested, tmp_path):
    sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, archive=tmp_path / 'run', keep_windows=True)
    archive = sps.load_archive(tmp_path / 'run')

    assert archive.axis == AXES
    assert archive.sweep.primal('soc').equals(nested.primal('soc'))
    assert archive.sweep.primal('soc', per_window=True).equals(nested.primal('soc', per_window=True))
    again = sps.solve_over(archive.spec, archive.sources, archive.axis, carry=archive.carry)
    assert again.primal('soc').equals(nested.primal('soc'))


def test_an_undeclared_expression_reads_off_a_nested_archive(nested, tmp_path):
    sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, archive=tmp_path / 'run', keep_windows=True)
    doubled = sps.load_archive(tmp_path / 'run').sweep.evaluate('soc * 2')
    expected = nested.primal('soc').with_columns(pl.col('value') * 2)
    assert doubled.equals(expected)


def test_a_nested_sweep_starts_from_an_earlier_one(nested, monkeypatch):
    """Each window takes its own scenario's rows of the earlier answer, cut by both axes, over its local index."""
    given = []
    original = api.Model.solve
    monkeypatch.setattr(api.Model, 'solve', lambda self, **kw: given.append(kw.get('start')) or original(self, **kw))
    again = sps.solve_over(WINDOW, scenario_horizons(), AXES, carry=CARRY, start=nested)

    windows = nested.primal('soc', per_window=True)
    for key, start in zip(nested.keys, given, strict=True):
        own = windows.filter(pl.col('scenario') == key[0], pl.col('snapshot_start') == key[1])
        assert start['primal']['soc'].select('t', 'value').sort('t').equals(own.select('t', 'value').sort('t')), key
    assert again.record['objective'].to_list() == pytest.approx(nested.record['objective'].to_list()), (
        'the same optimum, though the LP may end on another of its optimal vertices'
    )


def test_three_axes_cut_in_turn():
    sources = scenario_horizons()
    years = pl.concat([sources['load'].with_columns(pl.lit(year).alias('year')) for year in (2030, 2040)])
    runs = sps.solve_over(WINDOW, {**sources, 'load': years}, (sps.EachCoordinate('year'), *AXES), carry=CARRY)

    assert runs.key_names == ('year', 'scenario', 'snapshot_start'), 'one key column per axis, outer first'
    assert len(runs) == 2 * 2 * 3, 'two years, two scenarios, three windows each'
    one = runs.primal('soc').filter(pl.col('year') == 2040).drop('year')
    assert chain_of(one, 'high').equals(alone('high', carry=CARRY).primal('soc'))


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
        MYOPIC, {**base, 'demand': demand}, (sps.EachCoordinate('scenario'), sps.EachCoordinate('period')), carry=carry
    )
    for scenario in SCENARIOS:
        mine = demand.filter(pl.col('scenario') == scenario).drop('scenario')
        own = sps.solve_over(MYOPIC, {**base, 'demand': mine}, sps.EachCoordinate('period'), carry=carry)
        assert chain_of(runs.primal('total'), scenario).equals(own.primal('total'))


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
        sps.solve_over(WINDOW, scenario_horizons(), axis, carry=CARRY, **options)
    assert not built, 'refused before any slice is built'
