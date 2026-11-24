"""Every lap of a model's life, one row each, in `Model.timings`.

The laps tile each call, so a phase's rows add up to its clock in the metrics,
and a run's rows to what the model took between handing back two results.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

import specsolve as sps
from specsolve._engine import assembly, engine
from specsolve._records import PHASES

SPEC = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}},
    'parameters': {'load': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0}}, 'q': {'dims': ['t'], 'bounds': {'lower': 0}}},
    'constraints': {
        'balance': {'dims': ['t'], 'expression': 'p + q == load'},
        'cap': {'dims': ['t'], 'expression': 'p <= 3'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p + 2 * q, over=t)'},
}
DATA = {'t': [0, 1, 2], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [5.0, 4.0, 6.0]})}
GROWN = {'t': [0, 1, 2, 3], 'load': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [5.0, 4.0, 6.0, 1.0]})}

BUILD = [
    ('model', 'setup'),
    ('variable', 'p'),
    ('variable', 'q'),
    ('constraint', 'balance'),
    ('constraint', 'cap'),
    ('objective', 'objective'),
    ('model', 'matrix'),
    ('model', 'tables'),
]


def _by_phase(laps: pl.DataFrame) -> dict[str, float]:
    """*laps* summed by phase, every phase present, in the order the phases run."""
    summed = dict(laps.group_by('phase').agg(pl.col('seconds').sum()).iter_rows())
    return {phase: summed.get(phase, 0.0) for phase in PHASES}


def _clocks(metrics: sps.types.Metrics) -> dict[str, float]:
    return {phase: getattr(metrics, f'{phase}_seconds') for phase in PHASES}


def test_a_build_logs_its_setup_each_entry_and_its_stacking_in_build_order():
    """The build phase's rows name the step: the setup, each entry, then the two stacking steps."""
    with sps.build(SPEC, DATA) as model:
        build = model.timings().filter(phase='build')
    assert build.select('kind', 'entry').rows() == BUILD, 'every step once, in the order the build takes them'
    assert build.get_column('seconds').min() > 0, 'every step took some time'


def test_the_laps_add_up_to_the_metrics_exactly():
    """The whole log, summed by phase, is the model's whole-life metrics, to the last bits."""
    with sps.build(SPEC, DATA) as model:
        model.solve()
        model.update(GROWN)
        model.solve()
        assert _by_phase(model.timings()) == pytest.approx(_clocks(model.metrics()), rel=1e-12), (
            'the metrics are the log summed'
        )


def test_each_run_adds_up_to_the_result_it_handed_back():
    """A run is what the model took between handing back two results: that result's row."""
    with sps.build(SPEC, DATA) as model:
        results = [model.solve()]
        model.update(GROWN)
        results.append(model.solve())
        model.update(DATA)
        model.update(GROWN)
        results.append(model.solve())
        timings = model.timings()
    assert timings.get_column('run').unique().sort().to_list() == [0, 1, 2], 'one run per result handed back'
    for run, result in enumerate(results):
        assert _by_phase(timings.filter(run=run)) == pytest.approx(_clocks(result.metrics), rel=1e-12), (
            f'run {run} is its result'
        )
    assert timings.filter(run=2, phase='build', entry='setup').height == 2, (
        'two builds before the third solve, both logged'
    )


def test_an_archive_landing_is_in_its_run_and_not_in_the_result(tmp_path):
    """The result's row is written into the archive, so it is frozen before the archive lands; its run is not."""
    with sps.build(SPEC, DATA) as model:
        result = model.solve(archive=tmp_path / 'one.zip')
        run = _by_phase(model.timings().filter(run=0))
        whole = _clocks(model.metrics())
    assert run['archive'] > result.metrics.archive_seconds, 'the landing comes after the row it carries'
    assert run == pytest.approx(whole, rel=1e-12), (
        'and the run, with its landing, is the whole life of a model solved once'
    )
    others = {k: v for k, v in _clocks(result.metrics).items() if k != 'archive'}
    assert {k: v for k, v in run.items() if k != 'archive'} == pytest.approx(others, rel=1e-12), (
        'every other phase is the result'
    )


def test_laps_after_the_last_result_are_the_next_run():
    """An update after the last solve is no result's yet: it opens the next run."""
    with sps.build(SPEC, DATA) as model:
        model.solve()
        model.update(GROWN)
        pending = model.timings().filter(run=1)
    assert set(pending.get_column('phase')) >= {'tidy', 'attach', 'build'}, 'the update, logged under run 1'


def test_a_slow_entry_shows_in_its_own_row(monkeypatch):
    """Ten seconds added to one constraint's build land in that constraint's row and in no other."""
    added = [0.0]
    monkeypatch.setattr(engine, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = assembly.Assembly._build_constraint

    def slow(self, name, c):
        if name == 'cap':
            added[0] += 10.0
        return real(self, name, c)

    monkeypatch.setattr(assembly.Assembly, '_build_constraint', slow)
    with sps.build(SPEC, DATA) as model:
        took = dict(model.timings().filter(phase='build').select('entry', 'seconds').iter_rows())
    assert took['cap'] >= 10.0, 'the slowed constraint carries its delay'
    assert max(s for entry, s in took.items() if entry != 'cap') < 10.0, 'and no other entry does'


@pytest.mark.parametrize(
    ('module', 'slowed', 'row'),
    [
        pytest.param('engine', 'short_parameters', 'setup', id='setup-reading-the-short-parameters'),
        pytest.param('assembly', '_row_starts', 'matrix', id='matrix'),
        pytest.param('assembly', '_stack', 'tables', id='tables-stacking-the-columns'),
    ],
)
def test_a_slow_step_outside_any_entry_shows_in_its_model_row(monkeypatch, module, slowed, row):
    """What the build does around its entries is timed too: ten seconds added to a step land in its row."""
    owner = {'engine': engine, 'assembly': assembly}[module]
    added = [0.0]
    monkeypatch.setattr(engine, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = getattr(owner, slowed)

    def slow(*args, **kwargs):
        if slowed != '_stack' or args[1] is assembly._COLS:
            added[0] += 10.0
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, slowed, slow)
    with sps.build(SPEC, DATA) as model:
        took = dict(model.timings().filter(phase='build').select('entry', 'seconds').iter_rows())
    assert took[row] >= 10.0, f'the slowed step lands in the {row} row'
    assert max(s for entry, s in took.items() if entry != row) < 10.0, 'and in no other row'


def test_a_build_that_raises_keeps_the_laps_it_reached(monkeypatch):
    """The entry that raised has its row; nothing after it does."""
    real = assembly.Assembly._build_constraint

    def fails(self, name, c):
        if name == 'cap':
            raise RuntimeError('the build fails here')
        return real(self, name, c)

    with sps.build(SPEC, DATA) as model:
        monkeypatch.setattr(assembly.Assembly, '_build_constraint', fails)
        with pytest.raises(RuntimeError, match='the build fails here'):
            model.update(DATA)
        rebuilt = model.timings().filter(phase='build', run=0).select('kind', 'entry').rows()[len(BUILD) :]
    assert rebuilt == BUILD[:5], 'the second build logs up to the entry that raised, it included, and nothing after'


def test_a_model_with_no_objective_logs_no_objective_row():
    """Nothing was built for an objective the spec does not declare."""
    spec = {k: v for k, v in SPEC.items() if k != 'objective'}
    with sps.build(spec, DATA) as model:
        kinds = model.timings().filter(phase='build').get_column('kind').to_list()
    assert 'objective' not in kinds, 'no row for an objective that was never built'


def test_a_lap_outside_a_step_names_none():
    """A phase that has no steps logs its laps with ``kind`` and ``entry`` null."""
    with sps.build(SPEC, DATA) as model:
        model.solve()
        solve = model.timings().filter(phase='solve')
    assert solve.height == 1, 'one solve, one lap'
    assert solve.select(pl.col('kind', 'entry').is_null().all()).row(0) == (True, True), 'no step inside the solve'
