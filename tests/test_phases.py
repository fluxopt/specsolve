"""The phases a solve is clocked in add up to the call, and each step is clocked in the phase that names it.

[`Metrics`][specsolve._records.Metrics] defines the phases. The first test
holds their sum to the wall time of a whole call; the second slows one step
at a time and finds the delay in its own phase and nowhere else.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
import yaml

import specsolve as sps
from specsolve import _api, _archive_layout, _assumptions
from specsolve._engine import assembly, engine
from specsolve._records import PHASES
from specsolve._result import Result
from specsolve._sinks.solver import Solver

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

#: Wall time no phase holds: argument checks between them, and the call itself.
#: The model here builds and solves in tens of milliseconds, so a step left
#: unclocked that costs as much as one of them shows.
SLACK = 0.02

#: What one slowed step sleeps, comfortably above [`SLACK`][].
DELAY = 0.1


def _with_assumption(dispatch_yaml: Path) -> dict[str, object]:
    """The dispatch spec with an assumption, so the assumptions attach the sources and the engine does not."""
    spec = yaml.safe_load(dispatch_yaml.read_text())
    return {**spec, 'assumptions': {'capacity_is_not_negative': {'holds': 'p_max >= 0'}}}


def _timed(call: Callable[[], sps.types.Model], out: Path) -> tuple[float, dict[str, float]]:
    """The wall time of *call*, and the seconds per phase of the archive it writes to *out*.

    The model *call* returns is closed after the clock stops: releasing it
    comes after the archive is written, so no archive can record it.
    """
    started = time.perf_counter()
    model = call()
    wall = time.perf_counter() - started
    model.close()
    taken = sps.read_archive(out).metrics
    return wall, {phase: getattr(taken, f'{phase}_seconds') for phase in PHASES}


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
    dispatch_yaml, dispatch_table_inputs, tmp_path, suffix, with_assumption
):
    """From the spec as given to the archive on disk, every second is in one phase."""
    spec = _with_assumption(dispatch_yaml) if with_assumption else dispatch_yaml
    out = tmp_path / f'case{suffix}'
    wall, seconds = _timed(_archived(spec, dispatch_table_inputs, out), out)

    entered = {phase for phase, spent in seconds.items() if spent > 0}
    assert entered == set(PHASES) - {'write'}, 'a solve into an archive enters every phase but writing a model file'
    assert sum(seconds.values()) <= wall, f'no step is clocked twice: {seconds} sum past the wall time {wall}'
    assert wall - sum(seconds.values()) < SLACK, f'no step is left unclocked: {seconds} sum short of {wall}'


def test_the_phases_add_up_across_a_write_an_update_and_a_second_solve(dispatch_yaml, dispatch_table_inputs, tmp_path):
    """The clocks accumulate over a model's life, and still account for all of it."""
    load = dispatch_table_inputs['load']

    out = tmp_path / 'case.zip'

    def call() -> sps.types.Model:
        model = sps.build(dispatch_yaml, dispatch_table_inputs)
        model.write(tmp_path / 'model.lp')
        model.solve()
        model.update({'load': load.with_columns(value=load['value'] * 0.5)})
        model.solve(archive=out)
        return model

    wall, seconds = _timed(call, out)

    assert all(spent > 0 for spent in seconds.values()), f'this model entered every phase: {seconds}'
    assert sum(seconds.values()) <= wall, f'no step is clocked twice: {seconds} sum past the wall time {wall}'
    assert wall - sum(seconds.values()) < SLACK, f'no step is left unclocked: {seconds} sum short of {wall}'


def _slowed(real: Callable[..., object]) -> Callable[..., object]:
    """*real*, [`DELAY`][] slower."""

    def slow(*args: object, **kwargs: object) -> object:
        time.sleep(DELAY)
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
def test_a_slow_step_shows_in_its_own_phase(dispatch_yaml, dispatch_table_inputs, tmp_path, monkeypatch, steps, phase):
    """A step that takes longer is found in the phase that names it, and the phases still add up.

    Started from an earlier answer's values, so the start is read, matched
    and given too.
    """
    earlier = sps.solve(dispatch_yaml, dispatch_table_inputs)
    start: sps.types.Start = {'primal': {'p': earlier.primal('p')}}
    for owner, name in steps:
        monkeypatch.setattr(owner, name, _slowed(getattr(owner, name)))
    out = tmp_path / 'case.zip'
    wall, seconds = _timed(_archived(dispatch_yaml, dispatch_table_inputs, out, start=start), out)

    assert seconds[phase] >= DELAY, f'the slowed step is clocked in {phase!r}: {seconds}'
    others = {other: spent for other, spent in seconds.items() if other != phase}
    assert all(spent < DELAY for spent in others.values()), f'and in no other phase: {others}'
    assert wall - sum(seconds.values()) < SLACK, f'and nothing it took is lost: {seconds} sum short of {wall}'


def test_a_clock_with_no_column_is_refused():
    """A phase clocked under a name no `Metrics` column carries would be dropped from every archive."""
    with pytest.raises(AssertionError, match="'nap' is not a phase"), engine.Engine().clocked('nap'):
        pass


def test_the_model_keeps_the_archive_clock_its_row_recorded(dispatch_yaml, dispatch_table_inputs, tmp_path):
    """The row adds the archive's time so far itself, so the model's own clocks are checked apart from it."""
    with sps.build(dispatch_yaml, dispatch_table_inputs) as model:
        model.solve(archive=tmp_path / 'case.zip')
        kept = model.diagnostics().seconds.get('archive', 0.0)
    recorded = sps.read_archive(tmp_path / 'case.zip').metrics.archive_seconds
    assert kept >= recorded > 0, f'the model keeps the {recorded} s its archive row recorded, and the rest: {kept}'
