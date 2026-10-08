"""``outputs=``: an answer carries the primal, the duals and the declared expressions, and the rest only on request.

One rule for every verb and every shape an answer takes: a live result, a saved
or scanned one, an archived one, and a sweep held in memory, spilled, saved,
scanned or archived. Each carries what the solve was asked for, and each
refuses the rest with the same message naming the rewrite.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import get_context
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve.relational.answer_layout import OUTPUT_KINDS, OUTPUTS
from tests.test_strategy import DISPATCH, STATIC, WINDOW, WINDOW_AXIS, horizon_sources, scenario_sources

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from specsolve.types import Output, Result, Sweep

ACTIVITY: frozenset[Output] = frozenset({'activity'})
NONE: frozenset[Output] = frozenset()
SCENARIO = sps.EachCoordinate('scenario')
LOAD = [5.0, 6.0, 7.0, 8.0]
ONE = {
    **STATIC,
    'snapshot': pl.DataFrame({'snapshot': range(4)}),
    'load': pl.DataFrame({'snapshot': range(4), 'value': LOAD}),
}

type Answer = Result | Sweep


def test_every_output_carries_a_kind_and_every_kind_has_its_output() -> None:
    assert {carried.output for carried in OUTPUT_KINDS.values()} == set(OUTPUTS), (
        'an output that carries no kind would be asked for and hold nothing'
    )


# ---------------------------------------------------------------------------
# the request
# ---------------------------------------------------------------------------


VERBS = [
    pytest.param(lambda outputs: sps.solve(DISPATCH, ONE, outputs=outputs), id='solve'),
    pytest.param(lambda outputs: sps.build(DISPATCH, ONE).solve(outputs=outputs), id='model-solve'),
    pytest.param(lambda outputs: sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs), id='sweep'),
]


@pytest.mark.parametrize('verb', VERBS)
@pytest.mark.parametrize(
    ('outputs', 'refusal'),
    [
        pytest.param('activity', 'one string', id='a-bare-string'),
        pytest.param({'activty'}, "'activty'", id='a-misspelled-output'),
        pytest.param({'primal'}, 'carried always', id='a-core-kind'),
    ],
)
def test_a_request_that_names_no_output_is_refused(verb: Callable[[object], object], outputs, refusal: str) -> None:
    with pytest.raises(sps.errors.SpecsolveError, match=refusal):
        verb(outputs)


@pytest.mark.parametrize(
    ('module', 'verb'),
    [
        pytest.param(
            'specsolve.api', lambda: sps.solve(DISPATCH, ONE, outputs={'activty'}), id='solve-before-the-build'
        ),
        pytest.param(
            'specsolve.strategy',
            lambda: sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs={'activty'}),
            id='sweep-before-a-slice',
        ),
    ],
)
def test_a_misspelled_output_is_refused_before_anything_is_built(monkeypatch, module: str, verb) -> None:
    """A typo costs nothing: the refusal comes before the first model, not after the solve."""

    def built(*args: object, **kwargs: object) -> None:
        raise AssertionError('a model was built for a request that names no output')

    monkeypatch.setattr(f'{module}.build', built)
    with pytest.raises(sps.errors.SpecsolveError, match="'activty'"):
        verb()


# ---------------------------------------------------------------------------
# every shape an answer takes
# ---------------------------------------------------------------------------


def _live(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.solve(DISPATCH, ONE, outputs=outputs)


def _loaded(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.load_result(_live(outputs, tmp_path).save(tmp_path / 'saved'))


def _scanned(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.scan_result(_live(outputs, tmp_path).save(tmp_path / 'saved'))


def _archived(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    sps.solve(DISPATCH, ONE, outputs=outputs, archive=tmp_path / 'run.zip')
    archive = sps.load_archive(tmp_path / 'run.zip')
    assert isinstance(archive, sps.archive.ResultArchive)
    return archive.result


def _swept(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs)


def _threaded(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    with ThreadPoolExecutor(2) as pool:
        return sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs, executor=pool)


def _in_processes(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    with ProcessPoolExecutor(2, mp_context=get_context('spawn')) as pool:
        return sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs, executor=pool)


def _spilled(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs, spill_to=tmp_path / 'spill')


def _sweep_loaded(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.load_sweep(_swept(outputs, tmp_path).save(tmp_path / 'sweep'))


def _sweep_scanned(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    return sps.scan_sweep(_swept(outputs, tmp_path).save(tmp_path / 'sweep'))


def _sweep_archived(outputs: frozenset[Output], tmp_path: Path) -> Answer:
    sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs, archive=tmp_path / 'run.zip')
    archive = sps.load_archive(tmp_path / 'run.zip')
    assert isinstance(archive, sps.archive.SweepArchive)
    return archive.sweep


RESULTS = [
    pytest.param(_live, id='result'),
    pytest.param(_loaded, id='loaded-result'),
    pytest.param(_scanned, id='scanned-result'),
    pytest.param(_archived, id='archived-result'),
]
SWEEPS = [
    pytest.param(_swept, id='sweep'),
    pytest.param(_threaded, id='threaded-sweep'),
    pytest.param(_in_processes, id='sweep-in-processes'),
    pytest.param(_spilled, id='spilled-sweep'),
    pytest.param(_sweep_loaded, id='loaded-sweep'),
    pytest.param(_sweep_scanned, id='scanned-sweep'),
    pytest.param(_sweep_archived, id='archived-sweep'),
]


@pytest.mark.parametrize('answered', RESULTS)
def test_a_result_carries_the_activity_it_was_asked_for(answered, tmp_path: Path) -> None:
    """``balance`` is an equality, so its activity is the load it meets."""
    answer = answered(ACTIVITY, tmp_path)
    assert answer.activity('balance')['value'].to_list() == pytest.approx(LOAD), 'one row per snapshot, in label order'


@pytest.mark.parametrize('answered', RESULTS)
def test_the_bridges_read_an_output_through_its_reader(answered, tmp_path: Path) -> None:
    pytest.importorskip('pandas', reason='specsolve does not install pandas, and the floors environment has none')
    answer = answered(ACTIVITY, tmp_path)
    assert answer.to_pandas('balance', kind='activity')['value'].tolist() == pytest.approx(LOAD), (
        'the bridges read an output through the same reader'
    )


@pytest.mark.parametrize('answered', SWEEPS)
def test_a_sweep_carries_the_activity_it_was_asked_for(answered, tmp_path: Path) -> None:
    """Every slice's activity, keyed by slice, as each slice's own solve reads it."""
    answer = answered(ACTIVITY, tmp_path)
    for scenario, frame in answer.activity('balance').partition_by('scenario', as_dict=True).items():
        loads = scenario_sources()['load'].filter(pl.col('scenario') == scenario[0]).sort('snapshot')
        assert frame.sort('snapshot')['value'].to_list() == pytest.approx(loads['value'].to_list()), (
            f'the activity of slice {scenario[0]!r} is its own load'
        )
    assert answer.scan('balance', kind='activity').collect().height == 12, 'three scenarios of four snapshots'


@pytest.mark.parametrize('answered', [*RESULTS, *SWEEPS])
@pytest.mark.parametrize(
    'read',
    [
        pytest.param(lambda answer: answer.activity('balance'), id='reader'),
        pytest.param(lambda answer: answer.to_pandas('balance', kind='activity'), id='to-pandas'),
        pytest.param(lambda answer: answer.to_dataset(kind='activity'), id='to-dataset'),
    ],
)
def test_an_output_not_asked_for_is_refused_by_name(answered, read, tmp_path: Path) -> None:
    answer = answered(NONE, tmp_path)
    with pytest.raises(sps.errors.SpecsolveError, match=r"Solve again with outputs=\{'activity'\}"):
        read(answer)


@pytest.mark.parametrize('answered', [*RESULTS, *SWEEPS])
def test_an_answer_not_asked_for_an_output_writes_none(answered, tmp_path: Path) -> None:
    """The default answer stays the size it was asked to be."""
    answered(NONE, tmp_path)
    assert not list(tmp_path.rglob('activity')), 'no activity directory anywhere an answer was written'


# ---------------------------------------------------------------------------
# on disk
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('outputs', [pytest.param(NONE, id='none'), pytest.param(ACTIVITY, id='activity')])
@pytest.mark.parametrize(
    'written',
    [
        pytest.param(lambda outputs, at: sps.solve(DISPATCH, ONE, outputs=outputs).save(at), id='result'),
        pytest.param(
            lambda outputs, at: sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs, spill_to=at),
            id='spill',
        ),
        pytest.param(
            lambda outputs, at: sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, outputs=outputs).save(at),
            id='saved-sweep',
        ),
    ],
)
def test_the_stamp_names_the_outputs_the_answer_carries(written, outputs: frozenset[Output], tmp_path: Path) -> None:
    written(outputs, tmp_path / 'answer')
    stamp = json.loads((tmp_path / 'answer' / 'format.json').read_text())
    assert stamp['outputs'] == sorted(outputs), 'what a reader refuses as not asked for, rather than as absent'


def test_an_output_a_later_specsolve_wrote_is_ignored_rather_than_refused(tmp_path: Path) -> None:
    """A new output adds no layout bump, so this package can meet a name it has no reader for."""
    out = sps.solve(DISPATCH, ONE, outputs=ACTIVITY).save(tmp_path / 'saved')
    stamp = json.loads((out / 'format.json').read_text())
    (out / 'format.json').write_text(json.dumps({**stamp, 'outputs': ['activity', 'from_the_future']}))
    assert sps.load_result(out).activity('balance')['value'].to_list() == pytest.approx(LOAD), (
        'the outputs this package knows still read back'
    )


def test_a_spill_solved_with_other_outputs_is_refused_rather_than_resumed(tmp_path: Path) -> None:
    """A slice on disk carries what it was solved with, so a resume asking for more would mix two answers."""
    at = tmp_path / 'spill'
    sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, spill_to=at)
    with pytest.raises(sps.errors.SpecsolveError, match=r"outputs=\[\], and this run asks for outputs=\['activity'\]"):
        sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, spill_to=at, outputs=ACTIVITY)


def test_a_spill_solved_with_the_same_outputs_resumes(tmp_path: Path) -> None:
    at = tmp_path / 'spill'
    first = sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, spill_to=at, outputs=ACTIVITY)
    again = sps.solve_over(DISPATCH, scenario_sources(), SCENARIO, spill_to=at, outputs=ACTIVITY)
    assert again.activity('balance').equals(first.activity('balance'))


# ---------------------------------------------------------------------------
# windows
# ---------------------------------------------------------------------------


def test_a_window_sweep_stitches_the_activity_as_it_stitches_the_duals(tmp_path: Path) -> None:
    """Each snapshot carries the activity of the window that owns it, and ``per_window`` reads every window."""
    sweep = sps.solve_over(
        WINDOW,
        horizon_sources(12),
        WINDOW_AXIS,
        carry={'soc_initial': 'soc'},
        outputs=ACTIVITY,
        archive=tmp_path / 'run',
        keep_windows=True,
    )
    loads = horizon_sources(12)['load']
    stitched = sweep.activity('balance').sort('snapshot')
    assert stitched['value'].to_list() == pytest.approx(loads['value'].to_list()), (
        'an equality balance, so each snapshot carries its own load'
    )
    archive = sps.load_archive(tmp_path / 'run')
    assert isinstance(archive, sps.archive.SweepArchive)
    assert archive.sweep.activity('balance').sort('snapshot').equals(stitched)
    assert archive.sweep.activity('balance', per_window=True).equals(sweep.activity('balance', per_window=True))
