"""Every lap of a model's life, one row each, in `Model.timings`.

The laps tile each call, so a phase's rows add up to its clock in the metrics,
and a run's rows to what the model took between handing back two results.
"""

from __future__ import annotations

import time

import polars as pl
import pytest

import specsolve as sps
from specsolve import _clock, _encoding, _sources
from specsolve._answer.records import PHASES
from specsolve._engine import assembly, attaching, readback
from specsolve._sinks.highs import Highs
from specsolve.types import Model

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
    ('setup', None),
    ('variable', 'p'),
    ('variable', 'q'),
    ('constraint', 'balance'),
    ('constraint', 'cap'),
    ('objective', None),
    ('matrix', None),
    ('tables', None),
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
    assert build.select('step', 'entry').rows() == BUILD, 'every step once, in the order the build takes them'
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
    assert timings.filter(run=2, phase='build', step='setup').height == 2, (
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


RELOADED = {'load': pl.DataFrame({'t': [0, 1, 2], 'value': [7.0, 2.0, 3.0]})}


def test_an_update_of_numbers_alone_logs_each_entry_it_reads_again():
    """New numbers at the old coordinates read `balance` again and nothing else, so the build logs that row alone.

    The splice keeps the matrix, so it logs no `matrix` row; it ends in
    `tables`, as a full build does, so nothing it does lands in the next lap.
    """
    with sps.build(SPEC, DATA) as model:
        model.solve()
        model.update(RELOADED)
        model.solve()
        build = model.timings().filter(phase='build', run=1)
        assert _by_phase(model.timings()) == pytest.approx(_clocks(model.metrics()), rel=1e-12), (
            'the metrics are the log summed, the splice included'
        )
    assert build.select('step', 'entry').rows() == [('setup', None), ('constraint', 'balance'), ('tables', None)], (
        'the setup, the one entry the load feeds, then the tables'
    )


def test_a_slow_entry_read_again_shows_in_its_own_row(monkeypatch):
    """Ten seconds added to `balance`'s constant side, read again on the old indices, land in its row alone."""
    with sps.build(SPEC, DATA) as model:
        added = [0.0]
        monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
        real = assembly.Assembly._constant_side

        def slow(self, name, *args):
            if name == 'balance':
                added[0] += 10.0
            return real(self, name, *args)

        monkeypatch.setattr(assembly.Assembly, '_constant_side', slow)
        model.update(RELOADED)
        took = model.timings().filter(phase='build', run=0).tail(3)
    rows = dict(took.select(pl.coalesce('entry', 'step'), 'seconds').iter_rows())
    assert rows['balance'] >= 10.0, 'the slowed constraint carries its delay'
    assert max(s for entry, s in rows.items() if entry != 'balance') < 10.0, 'and neither the setup nor the tables do'


def test_a_coefficient_read_again_lands_in_its_constraint_row(monkeypatch):
    """Ten seconds added to re-reading `balance`'s coefficients land in its row, not in a `matrix` row of their own."""
    spec = {
        **SPEC,
        'parameters': {**SPEC['parameters'], 'eff': {'dims': ['t']}},
        'constraints': {**SPEC['constraints'], 'balance': {'dims': ['t'], 'expression': 'eff * p + q == load'}},
    }
    eff = pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 0.9, 0.8]})
    with sps.build(spec, DATA | {'eff': eff}) as model:
        added = [0.0]
        monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
        real = assembly.Assembly._linear_share

        def slow(self, name, *args, **kwargs):
            if name == 'balance':
                added[0] += 10.0
            return real(self, name, *args, **kwargs)

        monkeypatch.setattr(assembly.Assembly, '_linear_share', slow)
        model.update({'eff': eff.with_columns(pl.col('value') * 0.5)})
        took = model.timings().filter(phase='build', run=0).tail(3)
    assert took.select('step', 'entry').rows() == [('setup', None), ('constraint', 'balance'), ('tables', None)], (
        'the coefficients are read again within the one constraint they scale'
    )
    rows = dict(took.select(pl.coalesce('entry', 'step'), 'seconds').iter_rows())
    assert rows['balance'] >= 10.0, 'the slowed coefficients carry their delay in their constraint row'


def test_a_slow_entry_shows_in_its_own_row(monkeypatch):
    """Ten seconds added to one constraint's build land in that constraint's row and in no other."""
    added = [0.0]
    monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = assembly.Assembly._build_constraint

    def slow(self, name, c):
        if name == 'cap':
            added[0] += 10.0
        return real(self, name, c)

    monkeypatch.setattr(assembly.Assembly, '_build_constraint', slow)
    with sps.build(SPEC, DATA) as model:
        took = dict(model.timings().filter(phase='build').select(pl.coalesce('entry', 'step'), 'seconds').iter_rows())
    assert took['cap'] >= 10.0, 'the slowed constraint carries its delay'
    assert max(s for entry, s in took.items() if entry != 'cap') < 10.0, 'and no other entry does'


@pytest.mark.parametrize(
    ('module', 'slowed', 'row'),
    [
        pytest.param('assembly', 'short_parameters', 'setup', id='setup-reading-the-short-parameters'),
        pytest.param('assembly', '_row_starts', 'matrix', id='matrix'),
        pytest.param('assembly', '_stack', 'tables', id='tables-stacking-the-columns'),
        pytest.param('assembly', 'BuiltModel', 'tables', id='tables-making-the-built-model'),
    ],
)
def test_a_slow_step_outside_any_entry_shows_in_its_model_row(monkeypatch, module, slowed, row):
    """What the build does around its entries is timed too: ten seconds added to a step land in its row."""
    owner = {'assembly': assembly}[module]
    added = [0.0]
    monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = getattr(owner, slowed)

    def slow(*args, **kwargs):
        if slowed != '_stack' or args[1] is assembly._COLS:
            added[0] += 10.0
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, slowed, slow)
    with sps.build(SPEC, DATA) as model:
        took = dict(model.timings().filter(phase='build').select(pl.coalesce('entry', 'step'), 'seconds').iter_rows())
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
        rebuilt = model.timings().filter(phase='build', run=0).select('step', 'entry').rows()[len(BUILD) :]
    assert rebuilt == BUILD[:5], 'the second build logs up to the entry that raised, it included, and nothing after'


def test_a_model_with_no_objective_logs_no_objective_row():
    """Nothing was built for an objective the spec does not declare."""
    spec = {k: v for k, v in SPEC.items() if k != 'objective'}
    with sps.build(spec, DATA) as model:
        steps = model.timings().filter(phase='build').get_column('step').to_list()
    assert 'objective' not in steps, 'no row for an objective that was never built'


def test_a_lap_outside_a_step_names_none():
    """A phase that has no steps logs its laps with ``step`` and ``entry`` null."""
    with sps.build(SPEC, DATA) as model:
        model.solve()
        solve = model.timings().filter(phase='solve')
    assert solve.height == 1, 'one solve, one lap'
    assert solve.select(pl.col('step', 'entry').is_null().all()).row(0) == (True, True), 'no step inside the solve'


# ---------------------------------------------------------------------------
# the handoff's steps
# ---------------------------------------------------------------------------

NUMBERS = {'load': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 2.0, 3.0]})}


def _handoff_steps(model: sps.types.Model, run: int) -> list[str]:
    return model.timings().filter(phase='handoff', run=run).get_column('step').to_list()


@pytest.mark.parametrize(
    ('update', 'expected'),
    [
        pytest.param(None, ['start', 'check', 'instance', 'warm'], id='a-first-load'),
        pytest.param(NUMBERS, ['digest', 'start', 'check', 'push', 'warm'], id='new-numbers-pushed'),
        pytest.param(GROWN, ['digest', 'start', 'check', 'instance', 'warm'], id='a-reload'),
    ],
)
def test_a_handoff_logs_the_same_steps_on_every_solver(solver_name, update, expected):
    """A handoff that loads logs one new ``instance``; one that keeps the solver pushes. Every solver alike."""
    with sps.build(SPEC, DATA) as model:
        model.solve(solver_name=solver_name)
        run = 0
        if update is not None:
            model.update(update)
            model.solve(solver_name=solver_name)
            run = 1
        steps = _handoff_steps(model, run)
        handoff = model.timings().filter(phase='handoff').get_column('seconds').sum()
        whole = model.metrics().handoff_seconds
    assert steps == expected, 'every step of the handoff, in the order it took them'
    assert handoff == pytest.approx(whole, rel=1e-12), 'and the handoff still adds up'


@pytest.mark.parametrize(
    ('owner', 'slowed', 'step'),
    [
        pytest.param(readback, 'matched_start', 'start', id='matching-a-start'),
        pytest.param(Model, 'check_sink', 'check', id='checking-the-solver'),
        pytest.param(Highs, '_load', 'instance', id='loading-a-new-solver'),
    ],
)
def test_a_slow_step_of_a_handoff_lands_in_its_own_row(monkeypatch, owner, slowed, step):
    """Ten seconds added to one step of a handoff land in that step's row and in no other."""
    added = [0.0]
    monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = getattr(owner, slowed)

    def slow(*args, **kwargs):
        added[0] += 10.0
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, slowed, slow)
    with sps.build(SPEC, DATA) as model:
        model.solve(start=model.solve() if step == 'start' else 'previous')
        took = model.timings().filter(phase='handoff').group_by('step').agg(pl.col('seconds').sum())
    took = dict(took.iter_rows())
    assert took[step] >= 10.0, f'the slowed step lands in {step}'
    assert max(s for name, s in took.items() if name != step) < 10.0, 'and in no other step'


# ---------------------------------------------------------------------------
# the tidy's steps
# ---------------------------------------------------------------------------

GROUPED = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
    'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'p_max': {'dims': ['generator']}, 'load': {'dims': ['bus']}},
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'dims': ['bus'], 'expression': 'sum(p, over=generator, by=gen_bus[bus]) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p, over=generator)'},
}
GROUPED_DATA = {
    'bus': ['north', 'south'],
    'generator': ['g1', 'g2'],
    'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'south']}),
    'p_max': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [10.0, 10.0]}),
    'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [4.0, 6.0]}),
}


def _tidy_steps(model: sps.types.Model) -> list[tuple[str, str | None]]:
    return model.timings().filter(phase='tidy').select('step', 'entry').rows()


def _attach_steps(model: sps.types.Model) -> list[tuple[str, str | None]]:
    return model.timings().filter(phase='attach').select('step', 'entry').rows()


def test_a_tidy_logs_the_read_and_each_source_held_to_its_shape_in_order():
    """The paths read at once, each dimension, relation and parameter held to its shape, then the check over them all."""
    with sps.build(GROUPED, GROUPED_DATA) as model:
        steps = _tidy_steps(model)
        tidy = model.timings().filter(phase='tidy').get_column('seconds').sum()
        whole = model.metrics().tidy_seconds
    assert steps == [
        ('read', None),
        ('shape', 'bus'),
        ('shape', 'generator'),
        ('shape', 'gen_bus'),
        ('shape', 'load'),
        ('shape', 'p_max'),
        ('shape', None),
    ], 'every step once, in the order the tidy takes them: each entry by name'
    assert tidy == pytest.approx(whole, rel=1e-12), 'and the tidy still adds up'


def test_an_attach_logs_each_table_encoded_and_checked_then_what_the_model_makes_of_it():
    """Each relation and parameter encoded and held to its content, then the fill, the settling and the assumptions."""
    with sps.build(GROUPED, GROUPED_DATA) as model:
        steps = _attach_steps(model)
        attach = model.timings().filter(phase='attach').get_column('seconds').sum()
        whole = model.metrics().attach_seconds
    assert steps == [
        ('encode', 'gen_bus'),
        ('content', 'gen_bus'),
        ('encode', 'load'),
        ('content', 'load'),
        ('encode', 'p_max'),
        ('content', 'p_max'),
        ('fill', None),
        ('settle', None),
        ('assume', None),
    ], 'every step once, in the order the attach takes them: each entry by name'
    assert attach == pytest.approx(whole, rel=1e-12), 'and the attach adds up'


@pytest.mark.parametrize(
    ('owner', 'slowed', 'row'),
    [
        pytest.param(_sources, '_read_paths_at_once', ('tidy', 'read', None), id='reading-the-paths'),
        pytest.param(_sources, '_labels_table', ('tidy', 'shape', 'bus'), id='a-dimension-s-labels'),
        pytest.param(_sources, '_read_relation', ('tidy', 'shape', 'gen_bus'), id='a-relation'),
        pytest.param(_sources, 'checked_parameter', ('tidy', 'shape', 'load'), id='a-parameter-s-shape'),
        pytest.param(_encoding, '_in_dimension_types', ('attach', 'encode', 'gen_bus'), id='encoding-a-table'),
        pytest.param(
            _encoding,
            '_refuse_unknown_or_repeated_coordinates',
            ('attach', 'content', 'load'),
            id='a-parameter-s-content',
        ),
        pytest.param(attaching, 'filled_parameters', ('attach', 'fill', None), id='the-fill'),
        pytest.param(attaching, 'settle', ('attach', 'settle', None), id='the-settling'),
        pytest.param(attaching, 'check_assumptions', ('attach', 'assume', None), id='the-assumptions'),
    ],
)
def test_a_slow_step_of_a_tidy_or_an_attach_lands_in_its_own_row(monkeypatch, owner, slowed, row):
    """Ten seconds added to one step of a tidy or an attach land in that step's row and in no other."""
    added = [0.0]
    monkeypatch.setattr(_clock, 'perf_counter', lambda: time.perf_counter() + added[0])
    real = getattr(owner, slowed)
    seen = []

    def slow(*args, **kwargs):
        seen.append(slowed)
        if len(seen) == 1:
            added[0] += 10.0
        return real(*args, **kwargs)

    monkeypatch.setattr(owner, slowed, slow)
    with sps.build(GROUPED, GROUPED_DATA) as model:
        took = model.timings().select('phase', 'step', 'entry', 'seconds').rows()
    slowest = [(phase, step, entry) for phase, step, entry, seconds in took if seconds >= 10.0]
    assert slowest == [row], 'the slowed step, the first of its kind, holds the ten seconds alone'


def test_an_update_logs_only_the_parameters_it_reads_again():
    """A parameter an update keeps from the last build is neither shaped, encoded nor checked, so it has no row."""
    with sps.build(GROUPED, GROUPED_DATA) as model:
        tidied, attached = len(_tidy_steps(model)), len(_attach_steps(model))
        model.update({'load': GROUPED_DATA['load']})
        tidy, attach = _tidy_steps(model)[tidied:], _attach_steps(model)[attached:]
    assert tidy == [
        ('read', None),
        ('shape', 'bus'),
        ('shape', 'generator'),
        ('shape', 'gen_bus'),
        ('shape', 'load'),
        ('shape', None),
    ], 'every tidy step but the kept p_max, in order'
    assert attach == [
        ('encode', 'gen_bus'),
        ('content', 'gen_bus'),
        ('encode', 'load'),
        ('content', 'load'),
        ('fill', None),
        ('settle', None),
        ('assume', None),
    ], 'every attach step but the kept p_max, in order'


def test_a_refused_source_keeps_the_laps_it_reached():
    """A parameter with a label its dimension lacks is refused at its content; the steps before it keep their rows.

    The update replaces `p_max` alone, so `load` is kept from the build and has no row.
    """
    with sps.build(GROUPED, GROUPED_DATA) as model:
        first = len(_attach_steps(model))
        stray = pl.DataFrame({'generator': ['g1', 'g3'], 'value': [10.0, 10.0]})
        with pytest.raises(sps.errors.DataError):
            model.update({'p_max': stray})
        second = _attach_steps(model)[first:]
    assert second == [
        ('encode', 'gen_bus'),
        ('content', 'gen_bus'),
        ('encode', 'p_max'),
    ], 'the rows up to the refused content check, none for it or after it, and none for the kept load'


@pytest.mark.parametrize(
    'laps',
    [
        pytest.param({0: (0.3, 0.2, 0.1)}, id='one run'),
        pytest.param({0: (0.3, 0.2, 0.1), 1: (0.1, 0.2, 0.3)}, id='two runs'),
    ],
)
def test_a_runs_clocks_never_exceed_the_whole_life(monkeypatch, laps):
    """A result's row is its run's laps summed; its clocks never pass the model's, which are every run's.

    Three laps of 0.3, 0.2 and 0.1 s summed from the last come to one unit in the
    last place more than the same three summed in the order they ran, which is
    how a run's row used to pass its model's.
    """
    now = [0.0]
    monkeypatch.setattr(_clock, 'perf_counter', lambda: now[0])
    clock = _clock.Clock()
    for seconds in laps.values():
        for took in seconds:
            now[0] = 0.0
            clock.start()
            now[0] = took
            clock.lap('build', 'constraint')
        clock.next_run()
    whole = clock.metrics(0, 0, 0).build_seconds
    assert all(clock.metrics(0, 0, 0, run).build_seconds <= whole for run in laps), (
        "each run's share is no more than the whole"
    )
