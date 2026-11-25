"""The phases a solve is clocked in add up to the call, and each step is clocked in the phase that names it.

[`Metrics`][specsolve._records.Metrics] defines the phases. The model's own
clocks are held to the whole call, on the process's CPU time, so a runner that
takes the CPU away between two laps is not read as a step left unclocked; the
archived row, written before the archive lands, to at most what the model kept.
One test slows one step at a time and finds the delay in its own phase and
nowhere else.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, NamedTuple, get_args

import pytest
import yaml

import specsolve as sps
from specsolve import _api, _archive_layout, _assumptions
from specsolve._engine import assembly, engine
from specsolve._records import PHASES, Metrics, Phase
from specsolve._result import Result
from specsolve._sinks.solver import Solver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

#: What one slowed step adds to the clock: far above what any phase takes on a
#: busy runner, and no real time at all. A step no phase clocks loses all of it.
DELAY = 10.0

#: How long a caller waits between two calls, for real.
WAIT = 0.1

#: CPU time no phase holds. The laps tile each call, so what is left is the
#: return from one call and the entry into the next: 50 to 110 µs of wall time (#2180).
SLACK = 0.005


def _with_assumption(dispatch_yaml: Path) -> dict[str, object]:
    """The dispatch spec with an assumption, so the assumptions attach the sources and the engine does not."""
    spec = yaml.safe_load(dispatch_yaml.read_text())
    return {**spec, 'assumptions': {'capacity_is_not_negative': {'holds': 'p_max >= 0'}}}


class Timed(NamedTuple):
    """One call, timed from outside and from inside."""

    #: The call's length on the clock the engine laps on.
    wall: float
    #: Seconds per phase as the model kept them, the archive's landing included.
    kept: dict[str, float]
    #: Seconds per phase as the archive's metrics row recorded them.
    recorded: dict[str, float]

    def check(self) -> None:
        """The model's phases tile the call, and the row never holds more than the model kept."""
        kept = sum(self.kept.values())
        assert kept <= self.wall, f'no step is clocked twice: {self.kept} sum past the call, {self.wall}'
        assert self.wall - kept < SLACK, f'no step is left unclocked: {self.kept} sum short of {self.wall}'
        assert all(self.recorded[phase] <= self.kept[phase] for phase in PHASES), (
            f'the row is taken before the archive lands, so it holds no more than the model: {self.recorded}'
        )


class Clock:
    """This process's CPU time plus what the slowed steps added, read by the engine and the test alike.

    A step slowed on this clock costs no real time, so a phase the runner
    slows for its own reasons cannot reach a slowed step's delay. A pause off
    the CPU adds nothing, and the work of every thread of the process counts.
    """

    def __init__(self) -> None:
        self.added = 0.0

    def __call__(self) -> float:
        return time.process_time() + self.added


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """The clock the engine laps on for this test."""
    on = Clock()
    monkeypatch.setattr(engine, 'perf_counter', on)
    return on


def _timed(call: Callable[[], sps.types.Model], out: Path, clock: Clock) -> Timed:
    """*call* timed from outside on *clock*, and the seconds per phase the model kept and the archive at *out* recorded.

    The model *call* returns is closed after the clock stops: releasing it
    comes after the archive is written, so no archive can record it.
    """
    started = clock()
    model = call()
    wall = clock() - started
    kept = model.metrics()
    model.close()
    return Timed(wall, _clocks(kept), _clocks(sps.read_archive(out).metrics))


def _clocks(metrics: Metrics) -> dict[str, float]:
    return {phase: getattr(metrics, f'{phase}_seconds') for phase in PHASES}


def _archived(spec: object, sources: Mapping[str, object], out: Path, **solve: object) -> Callable[[], sps.types.Model]:
    """A build and a solve into the archive *out*."""

    def call() -> sps.types.Model:
        model = sps.build(spec, sources)
        model.solve(archive=out, **solve)
        return model

    return call


def test_the_phases_are_the_metrics_clocks_in_the_order_they_run():
    assert PHASES == (
        'load',
        'tidy',
        'attach',
        'build',
        'handoff',
        'solve',
        'readback',
        'write',
        'archive',
    ), 'one clock per phase, from the spec as given to the archive on disk'


@pytest.mark.parametrize('suffix', [pytest.param('.zip', id='a zip'), pytest.param('', id='a directory')])
@pytest.mark.parametrize('with_assumption', [False, True], ids=['the build attaches', 'the assumptions attach'])
def test_the_phases_add_up_to_a_build_solved_into_an_archive(
    dispatch_yaml, dispatch_table_inputs, tmp_path, clock, suffix, with_assumption
):
    """From the spec as given to the archive on disk, every second is in one phase."""
    spec = _with_assumption(dispatch_yaml) if with_assumption else dispatch_yaml
    out = tmp_path / f'case{suffix}'
    timed = _timed(_archived(spec, dispatch_table_inputs, out), out, clock)

    entered = {phase for phase, spent in timed.recorded.items() if spent > 0}
    assert entered == set(PHASES) - {'write'}, 'a solve into an archive enters every phase but writing a model file'
    timed.check()


def test_the_phases_add_up_across_a_write_an_update_and_a_second_solve(
    dispatch_yaml, dispatch_table_inputs, tmp_path, clock
):
    """The model's clocks accumulate over its life, and still account for all of it.

    The archive of the second solve records that answer's share: the update
    and the solve after it, not the build, the write or the first solve.
    """
    load = dispatch_table_inputs['load']

    out = tmp_path / 'case.zip'

    def call() -> sps.types.Model:
        model = sps.build(dispatch_yaml, dispatch_table_inputs)
        model.write(tmp_path / 'model.lp')
        model.solve()
        model.update({'load': load.with_columns(value=load['value'] * 0.5)})
        model.solve(archive=out)
        return model

    timed = _timed(call, out, clock)

    assert all(spent > 0 for spent in timed.kept.values()), f'this model entered every phase: {timed.kept}'
    entered = {phase for phase, spent in timed.recorded.items() if spent > 0}
    assert entered == set(PHASES) - {'load', 'write'}, (
        f"the second answer reads its update, its solve and its archive, and the load and the write are the first's: "
        f'{timed.recorded}'
    )
    timed.check()


def test_each_answer_carries_its_share_and_the_shares_add_up_to_the_model(
    dispatch_yaml, dispatch_table_inputs, tmp_path
):
    """Two answers of one model split its clocks between them, and every phase adds back up.

    Only the archive's landing, which comes after its row is written, is the
    model's alone.
    """
    load = dispatch_table_inputs['load']
    with sps.build(dispatch_yaml, dispatch_table_inputs) as model:
        model.write(tmp_path / 'model.lp')
        first = model.solve()
        model.update({'load': load.with_columns(value=load['value'] * 0.5)})
        second = model.solve(archive=tmp_path / 'case.zip')
        kept = model.metrics()
    assert first.metrics is not None and second.metrics is not None, 'a solve hands back what it took'
    assert (first.metrics.solves, second.metrics.solves, kept.solves) == (1, 1, 2), 'a row per answer, one solve each'
    assert first.metrics.load_seconds > 0, 'the first answer carries the load'
    assert second.metrics.load_seconds == 0, 'and the second does not carry it again'
    assert first.metrics.write_seconds > 0, 'a write before a solve is in that answer'
    assert second.metrics.write_seconds == 0, 'and in no later one'
    shared = {phase: _clocks(first.metrics)[phase] + _clocks(second.metrics)[phase] for phase in PHASES}
    whole = _clocks(kept)
    assert {phase: shared[phase] for phase in PHASES if phase != 'archive'} == pytest.approx(
        {phase: whole[phase] for phase in PHASES if phase != 'archive'}
    ), 'every second outside the archive is in exactly one answer'
    assert shared['archive'] <= whole['archive'], 'the landing comes after the row, so only the model holds it'
    assert second.metrics == sps.read_archive(tmp_path / 'case.zip').metrics._replace(specsolve_run=None), (
        'the answer handed back carries the row its archive recorded'
    )


def test_an_archived_answer_reads_back_what_it_took(dispatch_yaml, dispatch_table_inputs, tmp_path):
    result = sps.solve(dispatch_yaml, dispatch_table_inputs, archive=tmp_path / 'case')
    assert sps.read_archive(tmp_path / 'case').metrics == result.metrics._replace(specsolve_run='case'), (
        'the archive writes the metrics row, with its run, and read_archive reads it back'
    )


def _slowed(real: Callable[..., object], clock: Clock) -> Callable[..., object]:
    """*real*, [`DELAY`][] slower on *clock*."""

    def slow(*args: object, **kwargs: object) -> object:
        clock.added += DELAY
        return real(*args, **kwargs)

    return slow


#: Each step a solve into an archive takes: where it lives, its name there, and
#: the phase it is clocked in. A step reached through two names is slowed
#: under both.
STEPS = [
    pytest.param([(_api, 'as_spec')], 'load', id='the spec read'),
    pytest.param([(_api, 'as_program')], 'load', id='the program compiled'),
    pytest.param([(_api, 'tidied')], 'tidy', id='the sources read'),
    pytest.param([(_api, 'validate_assumptions')], 'attach', id='the assumptions checked'),
    pytest.param([(engine, 'attach'), (_assumptions, 'attach')], 'attach', id='the sources attached'),
    pytest.param([(engine, 'short_parameters')], 'build', id='the sparse parameters counted'),
    pytest.param([(assembly.Assembly, 'run')], 'build', id='the entries assembled'),
    pytest.param([(_api.Model, '_read_start')], 'handoff', id='the start read'),
    pytest.param([(engine.Engine, '_matched_start')], 'handoff', id='the start matched'),
    pytest.param([(engine.sinks, 'loaded')], 'handoff', id='the solver loaded'),
    pytest.param([(Solver, 'start_from_values')], 'handoff', id='the start given to the solver'),
    pytest.param([(Solver, 'run')], 'solve', id='the solver run'),
    pytest.param([(engine.Engine, '_answered')], 'readback', id='the vectors laid out'),
    pytest.param([(_api.Model, '_with_record_fields')], 'readback', id='the record filled'),
    pytest.param([(_api, 'with_positions')], 'archive', id='the sources positioned for the archive'),
    pytest.param([(Result, '_save')], 'archive', id='the answer written'),
    pytest.param([(_archive_layout, '_write_catalogs')], 'archive', id='the catalog written'),
    pytest.param([(_archive_layout, '_pack')], 'archive', id='the zip packed'),
]


@pytest.mark.parametrize(('steps', 'phase'), STEPS)
def test_a_slow_step_shows_in_its_own_phase(
    dispatch_yaml, dispatch_table_inputs, tmp_path, monkeypatch, clock, steps, phase
):
    """A step that takes longer is found in the phase that names it, and the phases still add up.

    Started from an earlier answer's values, so the start is read, matched
    and given too.
    """
    earlier = sps.solve(dispatch_yaml, dispatch_table_inputs)
    start: sps.types.Start = {'primal': {'p': earlier.primal('p')}}
    for owner, name in steps:
        monkeypatch.setattr(owner, name, _slowed(getattr(owner, name), clock))
    out = tmp_path / 'case.zip'
    timed = _timed(_archived(dispatch_yaml, dispatch_table_inputs, out, start=start), out, clock)

    assert timed.recorded[phase] >= DELAY, f'the slowed step is recorded in {phase!r}: {timed.recorded}'
    others = {other: spent for other, spent in timed.kept.items() if other != phase}
    assert all(spent < DELAY for spent in others.values()), f'and in no other phase: {others}'
    timed.check()


def test_a_slow_step_after_the_archive_lands_shows_in_the_models_archive_clock(
    dispatch_yaml, dispatch_table_inputs, tmp_path, monkeypatch, clock
):
    """The answer is handed the sources its archive carries after the archive lands, and that is the archive's too.

    The archive's own row is written before the landing, so only the model's
    clock can hold it.
    """
    monkeypatch.setattr(_api, 'carried_sources', _slowed(_api.carried_sources, clock))
    out = tmp_path / 'case.zip'
    timed = _timed(_archived(dispatch_yaml, dispatch_table_inputs, out), out, clock)

    assert timed.kept['archive'] >= DELAY, f"the slowed step is in the model's archive clock: {timed.kept}"
    assert timed.recorded['archive'] < DELAY, f'and not in the row, written before it: {timed.recorded}'
    timed.check()


def _spin(seconds: float) -> Callable[[], None]:
    """*seconds* of this process's CPU time, spent."""

    def spend() -> None:
        until = time.process_time() + seconds
        while time.process_time() < until:
            pass

    return spend


@pytest.mark.parametrize(
    ('spend', 'unclocked'),
    [
        pytest.param(
            lambda: time.sleep(4 * SLACK),
            False,
            id='a pause off the CPU',
        ),
        pytest.param(_spin(4 * SLACK), True, id='work on the CPU'),
    ],
)
def test_only_work_outside_every_lap_reads_as_unclocked(
    dispatch_yaml, dispatch_table_inputs, tmp_path, monkeypatch, clock, spend, unclocked
):
    """A runner that takes the CPU away between two laps leaves no step unclocked; work the laps miss does.

    Each is put after the solve's last lap, where #2227 found the runner
    preempting the thread for 5 ms on CI.
    """
    real = engine.Engine.next_run

    def after_the_last_lap(self: engine.Engine) -> None:
        spend()
        real(self)

    monkeypatch.setattr(engine.Engine, 'next_run', after_the_last_lap)
    out = tmp_path / 'case.zip'
    timed = _timed(_archived(dispatch_yaml, dispatch_table_inputs, out), out, clock)
    left = timed.wall - sum(timed.kept.values())
    assert (left >= SLACK) is unclocked, f'{left * 1e3:.2f} ms is left outside every lap'


def test_a_lap_takes_only_a_phase_a_metrics_column_carries():
    """A phase clocked under a name no `Metrics` column carries would be dropped from every archive."""
    assert get_args(Phase) == PHASES, 'the phases a lap is typed to take are the Metrics clocks, in order'


def test_the_model_keeps_the_archive_clock_its_row_recorded(dispatch_yaml, dispatch_table_inputs, tmp_path):
    """The row adds the archive's time so far itself, so the model's own clocks are checked apart from it."""
    with sps.build(dispatch_yaml, dispatch_table_inputs) as model:
        model.solve(archive=tmp_path / 'case.zip')
        kept = model.metrics().archive_seconds
    recorded = sps.read_archive(tmp_path / 'case.zip').metrics.archive_seconds
    assert kept >= recorded > 0, f'the model keeps the {recorded} s its archive row recorded, and the rest: {kept}'


def test_the_digest_a_held_solver_reads_at_an_update_is_in_the_handoff(
    dispatch_yaml, dispatch_table_inputs, tmp_path, monkeypatch, clock
):
    """An update makes the held solver read its model's digest before the build lets the tables go.

    That read decides whether the solver keeps its model, so it is the handoff's.
    """
    load = dispatch_table_inputs['load']
    out = tmp_path / 'case.zip'
    monkeypatch.setattr(Solver, 'structure', _slowed(Solver.structure, clock))

    def call() -> sps.types.Model:
        model = sps.build(dispatch_yaml, dispatch_table_inputs)
        model.solve()
        model.update({'load': load.with_columns(value=load['value'] * 0.5)})
        model.solve(archive=out)
        return model

    timed = _timed(call, out, clock)

    assert timed.recorded['handoff'] >= DELAY, f'the slowed digest is clocked in the handoff: {timed.recorded}'
    others = {other: spent for other, spent in timed.kept.items() if other != 'handoff'}
    assert all(spent < DELAY for spent in others.values()), f'and in no other phase: {others}'
    timed.check()


@pytest.mark.parametrize(
    'call',
    [
        pytest.param(lambda model, tmp_path, load: model.solve(archive=tmp_path / 'case.zip'), id='a solve'),
        pytest.param(lambda model, tmp_path, load: model.update({'load': load}), id='an update'),
        pytest.param(lambda model, tmp_path, load: model.write(tmp_path / 'model.lp'), id='a write'),
    ],
)
def test_time_between_calls_is_in_no_phase(dispatch_yaml, dispatch_table_inputs, tmp_path, call):
    """Each call starts the clock, so a caller who waits between two calls is not billed for the wait."""
    with sps.build(dispatch_yaml, dispatch_table_inputs) as model:
        before = _clocks(model.metrics())
        time.sleep(WAIT)
        started = time.perf_counter()
        call(model, tmp_path, dispatch_table_inputs['load'])
        wall = time.perf_counter() - started
        after = _clocks(model.metrics())
    grown = {phase: after[phase] - before[phase] for phase in PHASES}
    assert sum(grown.values()) <= wall, f'the wait before the call is in a phase: {grown} sum past the call, {wall}'


def test_the_phases_table_reads_the_clocks_in_the_order_they_run(dispatch_yaml, dispatch_table_inputs, tmp_path):
    """The table a report pastes: one row per phase and its seconds."""
    sps.solve(dispatch_yaml, dispatch_table_inputs, archive=tmp_path / 'case.zip')
    taken = sps.read_archive(tmp_path / 'case.zip').metrics
    table = taken.phases

    assert table['phase'].to_list() == list(PHASES), 'one row per phase, in the order the phases run'
    assert table['seconds'].to_list() == [getattr(taken, f'{phase}_seconds') for phase in PHASES], (
        "each row's seconds are its Metrics column"
    )
