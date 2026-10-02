"""``archive=``: a spec, its data and its answer as one file, and back.

What attaches from the archive is what attached from the caller's own tables,
frame for frame, over every ported instance.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
import pytest
import yaml as pyyaml
from mathspec import to_spec

import specsolve as sps
from specsolve.api import attach_readers
from specsolve.layout import ANSWER_DIR, _staging_for
from specsolve.relational.parquet import LAYOUT, METRICS_FILE, RUN, Metrics, digest_of_file
from specsolve.sources import attachable, tidy_sources
from tests.conftest import (
    DISPATCH_COST,
    DISPATCH_GENERATORS,
    DISPATCH_P_MAX,
    DISPATCH_SNAPSHOTS,
    PORT_REFERENCES,
    _dispatch_load,
    expanded,
    override,
    port_sources,
    port_spec,
    raw_of,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from mathspec import Spec


def _question(archive: sps.SolveArchive | sps.SweepArchive) -> tuple[Spec, Mapping[str, object]]:
    """The pair every verb takes, read off an archive."""
    return archive.spec, archive.sources


def _archived(spec, sources, out: Path) -> Path:
    """The archive a solve writes, which is the only way one is made."""
    with sps.solve(spec, sources, archive=out):
        return out


@pytest.mark.parametrize('name', sorted(PORT_REFERENCES), ids=str)
def test_what_attaches_from_the_archive_is_what_attached_from_the_tables(name: str, tmp_path: Path) -> None:
    program = expanded(port_spec(name)).program
    sources = port_sources(name)
    archive = _archived(expanded(port_spec(name)), sources, tmp_path / 'model.zip')
    spec, unpacked = _question(sps.load_archive(archive, tmp_path / 'out'))

    assert set(unpacked) == set(attachable(program)), (
        'the archive holds one member per attachable key — every declared parameter, dimension and relation, '
        'and nothing a piecewise block derives'
    )
    before = tidy_sources(program, sources)
    after = tidy_sources(to_spec(spec).program, unpacked)
    differing = [key for key in before if not before[key].collect().equals(after[key].collect())]
    assert not differing, (
        f'frames that came back changed: {differing} — the archive carries the labels, values and dtypes'
    )


def test_the_round_trip_solves_to_the_same_objective(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with (
        sps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
        sps.solve(*_question(sps.load_archive(archive, tmp_path / 'out'))) as unpacked,
    ):
        assert unpacked.objective == pytest.approx(direct.objective, rel=1e-9), (
            'the archive builds the model the tables did'
        )


def test_plain_python_shapes_are_written_as_the_tables_they_stand_for(dispatch_yaml: Path, tmp_path: Path) -> None:
    """A dict, a positional sequence and a bare label range all come back as tidy frames."""
    sources = {
        'p_max': dict(zip(DISPATCH_GENERATORS, DISPATCH_P_MAX, strict=True)),
        'cost': list(DISPATCH_COST),
        'load': pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS), 'value': _dispatch_load()}),
        'snapshot': range(DISPATCH_SNAPSHOTS),
        'generator': list(DISPATCH_GENERATORS),
    }
    archive = _archived(dispatch_yaml, sources, tmp_path / 'dispatch.zip')
    _, unpacked = _question(sps.scan_archive(archive, tmp_path / 'out'))
    cost = pl.read_parquet(unpacked['cost'])
    snapshot = pl.read_parquet(unpacked['snapshot'])

    assert cost.columns == ['generator', 'value', RUN], 'a positional sequence is spread over its labels'
    assert cost['value'].to_list() == list(DISPATCH_COST), 'in the order the index declares them'
    assert snapshot.columns == ['snapshot', RUN], 'a bare label range is written as an index table'
    assert snapshot.height == DISPATCH_SNAPSHOTS, 'one row per label'


def test_the_archive_is_the_file_and_stored_parquet(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename: info.compress_type for info in zipped.infolist()}
        beside_the_answer = {name for name in members if not name.startswith('answer/')}
        assert beside_the_answer == {
            'spec.yaml',
            'sources.parquet',
            *(f'sources/{k}.parquet' for k in dispatch_frame_inputs),
        }, (
            'the layout is spec.yaml, one parquet member per source key, the table digesting them, and the '
            'answer under its own'
        )
        assert any(name.startswith('answer/') for name in members), 'every archive carries the answer that made it'
        assert set(members.values()) == {zipfile.ZIP_STORED}, 'members are stored — parquet is already compressed'
        assert to_spec(pyyaml.safe_load(zipped.read('spec.yaml'))) == to_spec(dispatch_yaml), (
            'spec.yaml is the spec the source file declares'
        )


def test_a_parquet_path_is_archived_as_its_table_and_digested_as_its_own_bytes(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The member gains the run column; the digest is of the file the caller passed."""
    load = dispatch_frame_inputs['load'].with_columns(pl.lit('a stray column').alias('note'))
    path = tmp_path / 'load.parquet'
    load.write_parquet(path)
    sps.solve(dispatch_yaml, {**dispatch_frame_inputs, 'load': str(path)}, archive=tmp_path / 'case').close()
    held = pl.read_parquet(tmp_path / 'case' / 'sources' / 'load.parquet')

    assert held.equals(load.with_columns(pl.lit('case').alias(RUN))), (
        'the table travels whole, stray column included — it is filtered where it attaches, as a path is'
    )
    digests = dict(pl.read_parquet(tmp_path / 'case' / 'sources.parquet').select('source', 'digest').iter_rows())
    assert digests['load'] == digest_of_file(path), 'and the digest is of the bytes the caller passed, not the stamp'


def test_unpack_lays_the_archive_out_in_the_directory(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    spec, sources = _question(sps.scan_archive(archive, tmp_path / 'out'))

    assert sources == {k: tmp_path / 'out' / 'sources' / f'{k}.parquet' for k in dispatch_frame_inputs}, (
        'every source comes back as the path it was extracted to, one per key'
    )
    assert all(p.is_file() for p in sources.values()), 'and each path is a file on disk'
    assert (tmp_path / 'out' / 'spec.yaml').is_file(), 'the file lands beside them, as the archive holds it'
    assert to_spec(tmp_path / 'out' / 'spec.yaml') == spec, 'and is the spec handed back'


def test_a_refused_model_writes_nothing(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    out = tmp_path / 'dispatch.zip'
    with pytest.raises(sps.DataError, match="no data provided for parameter 'cost'"):
        sps.solve(dispatch_yaml, {k: v for k, v in dispatch_frame_inputs.items() if k != 'cost'}, archive=out)
    assert not out.exists(), 'a model that cannot be built writes no archive'


@pytest.mark.parametrize(
    ('members', 'says'),
    [
        pytest.param({'sources/load.parquet': b''}, "has no 'spec.yaml'", id='no-spec'),
        pytest.param({'spec.yaml': b'', 'load.parquet': b''}, "holds ['load.parquet']", id='member-outside-sources'),
        pytest.param({'spec.yaml': b'', 'sources/load.csv': b''}, "holds ['sources/load.csv']", id='not-parquet'),
    ],
)
def test_a_zip_outside_the_layout_is_refused(members: dict[str, bytes], says: str, tmp_path: Path) -> None:
    path = tmp_path / 'other.zip'
    with zipfile.ZipFile(path, 'w') as zipped:
        for name, data in members.items():
            zipped.writestr(name, data)
    with pytest.raises(sps.LayoutError) as excinfo:
        _question(sps.load_archive(path, tmp_path / 'out'))
    assert says in str(excinfo.value), 'the message names what was found, and the layout archive= writes'
    assert not (tmp_path / 'out').exists(), 'nothing is extracted from a zip that is not an archive'


def test_a_directory_archive_holds_what_the_zip_holds_and_is_read_where_it_lies(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The suffix picks the container, and the container is all that differs."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip')
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')

    with zipfile.ZipFile(tmp_path / 'case.zip') as zipped:
        packed = sorted(zipped.namelist())
    spread = sorted(f.relative_to(tmp_path / 'case').as_posix() for f in (tmp_path / 'case').rglob('*') if f.is_file())
    assert spread == packed, 'the same members, laid out instead of packed'

    loose = sps.load_archive(tmp_path / 'case')
    unpacked = sps.load_archive(tmp_path / 'case.zip', tmp_path / 'out')
    assert loose.answer.objective == unpacked.answer.objective
    assert sps.scan_archive(tmp_path / 'case').sources['load'].parent.parent == tmp_path / 'case', (
        'a directory archive is scanned where it lies, so there is no second copy to keep alive'
    )


@pytest.mark.parametrize(
    ('read', 'suffix', 'into', 'says'),
    [
        pytest.param(sps.scan_archive, '.zip', None, 'needs somewhere to unpack', id='scan-a-zip-with-no-into'),
        pytest.param(sps.scan_archive, '', 'anywhere', 'read where it lies', id='scan-a-directory-with-an-into'),
        pytest.param(sps.load_archive, '', 'anywhere', 'read where it lies', id='load-a-directory-with-an-into'),
    ],
)
def test_into_is_asked_for_exactly_where_something_must_be_unpacked_and_kept(
    read: Callable[..., sps.SolveArchive | sps.SweepArchive],
    suffix: str,
    into: str | None,
    says: str,
    dispatch_yaml: Path,
    dispatch_frame_inputs,
    tmp_path: Path,
) -> None:
    """A scanned zip needs an `into` to unpack to; a directory has nothing to unpack."""
    out = tmp_path / f'case{suffix}'
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out)
    with pytest.raises(sps.LayoutError, match=says):
        read(out, None if into is None else tmp_path / into)


def test_loading_a_zip_needs_nowhere_to_unpack_and_leaves_nothing_behind(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """What `load_archive` reads whole it does not read again, so the members are scratch."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip')
    beside_it = sorted(path.name for path in tmp_path.iterdir())

    case = sps.load_archive(tmp_path / 'case.zip')

    assert case.answer.primal('p').height > 0, 'the answer came back with no directory named to read it off'
    assert sorted(path.name for path in tmp_path.iterdir()) == beside_it, (
        'and the unpacked members are gone, the zip beside them the only thing left'
    )


def test_a_directory_that_already_holds_something_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An archive is written whole, so it never merges into what is there."""
    out = tmp_path / 'case'
    out.mkdir()
    (out / 'mine.txt').write_text('not an archive')
    with pytest.raises(sps.LayoutError, match='already holds something'):
        sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out)
    assert sorted(f.name for f in out.iterdir()) == ['mine.txt'], 'and what was there is untouched'


@pytest.mark.parametrize(
    'verb',
    [
        pytest.param(lambda spec, sources: sps.build(spec, sources), id='build'),
        pytest.param(lambda spec, sources: sps.solve(spec, sources), id='solve'),
        pytest.param(
            lambda spec, sources: sps.solve_over(spec, sources, sps.EachCoordinate('generator')), id='solve_over'
        ),
    ],
)
def test_a_lowered_program_is_not_a_model_any_verb_takes(verb, dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """Lowering has no inverse, so a Program is refused at the door, before anything is built."""
    with pytest.raises(sps.SpecsolveError, match='lowered Program is not a spec this takes'):
        verb(sps.check(dispatch_yaml), dispatch_frame_inputs)


def test_the_archive_lands_whole(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, monkeypatch) -> None:
    """The archive is renamed into place: a missing parent is made, and a failure leaves nothing."""
    from mathspec import Spec

    out = tmp_path / 'nested' / 'dispatch.zip'
    _archived(dispatch_yaml, dispatch_frame_inputs, out)
    assert sorted(p.name for p in out.parent.iterdir()) == ['dispatch.zip'], 'the archive alone, no .part beside it'

    def fails(self):
        raise RuntimeError('the box went away')

    monkeypatch.setattr(Spec, 'to_yaml', fails)
    later = tmp_path / 'later.zip'
    with pytest.raises(RuntimeError, match='went away'):
        sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=later)
    assert not list(tmp_path.glob('later*')), 'a write that did not finish leaves nothing under either name'


def test_an_archive_carries_the_answer_beside_the_question(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The whole archive: what was asked, the data it was asked of, and what came back."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip') as solved:
        loaded = sps.load_archive(tmp_path / 'case.zip', tmp_path / 'case')

        assert loaded.answer.objective == solved.objective
        for name in to_spec(dispatch_yaml).program.variables:
            assert loaded.answer.primal(name).equals(solved.primal(name))

    with sps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(loaded.answer.objective, rel=1e-9), (
            'the question in the archive is the one its answer answered'
        )


def test_two_archives_of_one_spec_over_different_numbers_are_told_apart(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The source digests separate two runs of one document that `spec_digest` cannot."""
    halved = pl.DataFrame(
        {'snapshot': dispatch_frame_inputs['load']['snapshot'], 'value': dispatch_frame_inputs['load']['value'] * 0.5}
    )
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'base').close()
    sps.solve(dispatch_yaml, {**dispatch_frame_inputs, 'load': halved}, archive=tmp_path / 'halved').close()
    base, other = sps.load_archive(tmp_path / 'base'), sps.load_archive(tmp_path / 'halved')

    assert base.answer.spec_digest == other.answer.spec_digest, 'one document, so the spec digest cannot separate them'
    moved = (
        base.source_digests.join(other.source_digests, on='source', suffix='_other')
        .filter(pl.col('digest') != pl.col('digest_other'))['source']
        .to_list()
    )
    assert moved == ['load'], 'and the digests name the one input that moved, not merely that something did'


def test_the_digest_table_names_every_source_the_archive_holds(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A digest per member of `sources/`, so nothing is silently unattested."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case').close()
    case = sps.load_archive(tmp_path / 'case')

    assert case.source_digests.columns == [RUN, 'source', 'digest'], (
        "the archive it came from, the key, and what that key's bytes digest to"
    )
    assert case.source_digests['source'].to_list() == sorted(case.sources), (
        'one row per archived source, in source order rather than the order the caller happened to pass them'
    )
    assert case.source_digests[RUN].unique().to_list() == ['case'], (
        "every row carries the archive's own name, so a table read across a directory of them needs no paths"
    )
    with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'again') as solved:
        solved.close()
    again = sps.load_archive(tmp_path / 'again')
    assert again.source_digests.drop(RUN).equals(case.source_digests.drop(RUN)), (
        'and a digest is of the source before the run is stamped on, so one table under two names digests alike'
    )


def test_a_sweep_archive_digests_the_sources_it_was_cut_from(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A sweep archives its sources whole, so the digests are of the whole."""
    whole = tmp_path / 'load.parquet'
    _by_scenario(['low', 'high']).write_parquet(whole)
    sources = {**dispatch_frame_inputs, 'load': whole}
    sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=tmp_path / 'study')
    study = sps.load_archive(tmp_path / 'study')

    assert isinstance(study, sps.SweepArchive), 'the archive carries an axis, or this is testing the other type'
    assert study.source_digests['source'].to_list() == sorted(study.sources), 'one row per source, as for one solve'
    assert study.source_digests[RUN].unique().to_list() == ['study'], (
        'stamped with the archive name as a solve archive is, the sweep key belonging to the slices and not the data'
    )
    digests = dict(study.source_digests.select('source', 'digest').iter_rows())
    assert digests['load'] == digest_of_file(whole), 'the digest is of the whole source the sweep was cut from'


def test_an_archive_records_what_reaching_its_answer_cost(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The clocks: the one part of an archive that re-solving cannot recover."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    taken = sps.load_archive(tmp_path / 'case').metrics

    assert isinstance(taken, Metrics), 'the row comes back as the value its columns declare, not a frame of one'
    assert Metrics._fields == (
        'columns',
        'rows',
        'nonzeros',
        'solves',
        'loads',
        'attach_seconds',
        'build_seconds',
        'handoff_seconds',
        'solve_seconds',
        'write_seconds',
        RUN,
    ), 'the sizes, the counters, one clock per phase in the order the phases run, then the run that took them'
    written = pl.read_parquet(tmp_path / 'case' / ANSWER_DIR / METRICS_FILE)
    assert written.columns == list(Metrics._fields), "and the file carries the type's columns, in its order"
    assert written.height == 1, 'one solve writes one row'
    assert (taken.solves, taken.loads) == (1, 1), (
        'sps.solve builds the model it solves, so the row covers that one solve and its one load'
    )
    assert taken.build_seconds > 0.0, 'the build ran, so its clock is not the zero that says a phase did not'


def test_the_cost_row_says_how_many_solves_its_clocks_cover(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A row off a model that solved before covers every solve, and `solves` says how many."""
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.solve()
        model.solve(archive=tmp_path / 'second')

    assert sps.load_archive(tmp_path / 'second').metrics.solves == 2, (
        'two solves ran before the archive was written, and the row it carries counts both'
    )


def test_a_case_that_wrote_a_file_and_one_that_did_not_are_still_one_table(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A phase that never ran writes zero rather than no column, so the rows concatenate."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'solved')
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.write(tmp_path / 'model.lp')
        model.solve(archive=tmp_path / 'written')

    cases = ('solved', 'written')
    rows = [sps.load_archive(tmp_path / case).metrics for case in cases]
    assert [row.write_seconds == 0.0 for row in rows] == [True, False], (
        'the first case never wrote a file and the second did, or the two schemas were never in question'
    )
    files = pl.concat(pl.read_parquet(tmp_path / case / ANSWER_DIR / METRICS_FILE) for case in cases)
    assert files.height == 2, 'and the two files a warehouse globs are one table'


def test_an_archive_whose_metrics_are_short_of_a_column_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A member short of a column is refused by name; the layout stamp does not catch it."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    metrics = tmp_path / 'case' / ANSWER_DIR / METRICS_FILE
    pl.read_parquet(metrics).drop('write_seconds').write_parquet(metrics)

    with pytest.raises(sps.LayoutError, match=r"Metrics row that is short of \['write_seconds'\]") as excinfo:
        sps.load_archive(tmp_path / 'case')
    assert 'solve the model again' in str(excinfo.value), 'and the message names the way out'


def test_every_phase_a_build_clocks_has_a_column_to_travel_in(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A phase added to the engine and not to `Metrics` would be dropped in silence."""
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        model.write(tmp_path / 'model.lp')
        model.solve()
        clocked = set(model.diagnostics().seconds)

    assert clocked == {'attach', 'build', 'write', 'handoff', 'solve'}, (
        'this model entered every phase a build clocks, or the check below passes on the ones it missed'
    )
    assert not {f'{phase}_seconds' for phase in clocked} - set(Metrics._fields), (
        f'every phase the engine clocks has a Metrics column, and {clocked} does not'
    )


def test_an_updated_model_archives_the_data_it_actually_answered(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An update moves the question, and the archive holds the merged data."""
    halved = pl.DataFrame(
        {'snapshot': dispatch_frame_inputs['load']['snapshot'], 'value': dispatch_frame_inputs['load']['value'] * 0.5}
    )
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as model:
        before = model.solve().objective
        after = model.update({'load': halved}).solve(archive=tmp_path / 'updated.zip')
        assert after.objective != pytest.approx(before, rel=1e-9), 'the update moved the answer, or this proves nothing'

    loaded = sps.load_archive(tmp_path / 'updated.zip', tmp_path / 'updated')
    with sps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(after.objective, rel=1e-9), (
            'the archived question is the updated one, so it re-solves to the answer it carries'
        )


@pytest.mark.parametrize('sweep', [False, True], ids=['one solve', 'a sweep'])
def test_every_archive_holds_one_objective_file_whatever_wrote_it(
    sweep: bool, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """One glob over a warehouse of runs finds every one of them."""
    out = tmp_path / 'run'
    if sweep:
        sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high', 'mid'])}
        sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=out)
    else:
        with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out):
            pass

    answer = out / 'answer'
    assert (answer / 'record.parquet').is_file(), 'the record is one file, whichever verb wrote it'
    assert not (answer / 'objective').exists(), 'and not a directory beside it'
    assert (answer / METRICS_FILE).is_file(), 'the metrics go the same way'
    assert not (answer / 'metrics').exists(), 'and not a directory beside it either'
    assert pl.read_parquet(answer / 'record.parquet').height == (3 if sweep else 1), 'one row per slice'


def test_an_archive_stamps_its_own_name_and_when_the_solve_returned(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The two columns a warehouse of runs is keyed and ordered by."""
    before = datetime.now(UTC)
    with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'nightly-2026-09-10.zip') as solved:
        reached = solved.solved_at
    sps.load_archive(tmp_path / 'nightly-2026-09-10.zip', tmp_path / 'out')
    record = pl.read_parquet(tmp_path / 'out' / 'answer' / 'record.parquet')

    assert record[RUN].to_list() == ['nightly-2026-09-10'], "the archive's own name, suffix dropped"
    answer = sps.load_archive(tmp_path / 'nightly-2026-09-10.zip').answer
    assert answer.record.specsolve_run == 'nightly-2026-09-10', (
        'the answer read back carries the name its record was stamped with'
    )
    resaved = pl.read_parquet(answer.save(tmp_path / 'plain') / 'record.parquet')
    assert resaved[RUN].to_list() == [None], 'a plain save is not an archive, so it claims no run name'
    assert record['solved_at'].item() == reached, 'and when the solver returned, as the result reports it'
    assert before <= record['solved_at'].item() <= datetime.now(UTC), 'which is a real clock, not a placeholder'


@pytest.mark.parametrize('sweep', [False, True], ids=['one solve', 'a sweep'])
def test_every_table_an_archive_holds_says_which_run_it_came_from(
    sweep: bool, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A reader that combines files and drops their paths still gets the run, and reading back drops it again."""
    out = tmp_path / 'nightly-2026-09-10.zip'
    if sweep:
        sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
        live = sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=out).primal('p')
    else:
        with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=out) as solved:
            live = solved.primal('p')
    case = sps.load_archive(out, tmp_path / 'out')
    tables = sorted((tmp_path / 'out').rglob('*.parquet'))

    assert {
        table.relative_to(tmp_path / 'out').as_posix(): pl.read_parquet(table)[RUN].unique().to_list()
        for table in tables
    } == {table.relative_to(tmp_path / 'out').as_posix(): ['nightly-2026-09-10'] for table in tables}, (
        'every table carries the archive name, sources and answer frames alike'
    )
    assert case.answer.primal('p').equals(live), 'a frame read back is the frame the solve returned'
    assert all(RUN not in frame.columns for frame in case.sources.values()), 'and a source read back is the table given'


def test_a_saved_answer_that_was_never_archived_names_no_run(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A bare `save` leaves `specsolve_run` null, in a column of the same schema."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        record = pl.read_parquet(solved.save(tmp_path / 'answer') / 'record.parquet')
    assert record[RUN].to_list() == [None], 'nothing published it, so nothing named it'
    assert record.schema[RUN] == pl.String, 'and the column is a string either way, never an all-null one'


def test_a_loaded_archive_owes_the_members_nothing_and_a_scanned_one_owes_them_everything(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """`load_archive` holds sources and answer in memory; `scan_archive` reads them off the members."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip')
    loaded = sps.load_archive(tmp_path / 'case.zip', tmp_path / 'out')
    scanned = sps.scan_archive(tmp_path / 'case.zip', tmp_path / 'out')
    expected = loaded.answer.primal('p')

    assert isinstance(loaded.sources['load'], pl.DataFrame), 'a loaded source is the table the member holds'
    assert scanned.sources['load'] == tmp_path / 'out' / 'sources' / 'load.parquet', 'a scanned one is the path to it'
    shutil.rmtree(tmp_path / 'out')

    assert loaded.answer.primal('p').equals(expected), 'the loaded answer was read before the members went'
    with pytest.raises(FileNotFoundError):
        scanned.answer.primal('p')


def test_a_loaded_sweep_archive_answers_the_frame_readers(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """`load_archive` gives a held `Sweep`, `scan_archive` a spilled one."""
    axis = sps.EachCoordinate('scenario')
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    sps.solve_over(dispatch_yaml, sources, axis, archive=tmp_path / 'study.zip')

    loaded = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')
    scanned = sps.scan_archive(tmp_path / 'study.zip', tmp_path / 'study')

    assert loaded.answer.primal('p').equals(scanned.answer.scan('p').collect()), 'the same study, read two ways'
    with pytest.raises(sps.SpecsolveError, match=r'sweep\.scan'):
        scanned.answer.primal('p')


def test_a_scenario_sweep_is_an_archive_and_runs_again(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The axis travels with the sweep's sources, which the archive holds whole."""
    axis = sps.EachCoordinate('scenario')
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = sps.solve_over(dispatch_yaml, sources, axis, archive=tmp_path / 'study.zip')
    study = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')

    assert study.axis == axis, 'the axis comes back as the value it went in as'
    assert study.answer.record.drop(RUN).equals(runs.record.drop(RUN))
    assert study.answer.record[RUN].unique().to_list() == ['study'], (
        'and the archive stamped its own name on every slice, which the sweep in memory had none of'
    )
    assert study.sources['load'].equals(sources['load']), (
        'the sliced source is archived whole, the column the axis cuts on included'
    )
    again = sps.solve_over(study.spec, study.sources, study.axis)
    assert again.record['objective'].to_list() == pytest.approx(runs.record['objective'].to_list()), (
        'the archive re-runs to the sweep it recorded, slice for slice'
    )


def test_a_rolling_horizon_keeps_the_way_back_to_the_dimension_it_sliced(tmp_path: Path) -> None:
    """A stitched read off the archive is the stitched read off the sweep."""
    from tests.test_strategy import WINDOW, horizon_sources

    axis = sps.EachWindow('snapshot', steps=4, lookahead=2, into='t')
    sources = horizon_sources(12)
    runs = sps.solve_over(WINDOW, sources, axis, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll.zip')
    stitched = runs.primal('soc', original_index=True)
    loaded = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll')

    assert loaded.axis == axis
    assert loaded.answer.scan('soc', original_index=True).collect().equals(stitched), (
        'the lookahead rows are dropped on the way out of the archive as they were in the process'
    )


def test_an_archive_whose_answer_names_another_spec_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A hand-edited archive whose answer and model do not belong together is refused."""
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'case.zip')
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1.0})
    tampered = tmp_path / 'tampered.zip'
    with zipfile.ZipFile(archive) as held, zipfile.ZipFile(tampered, 'w') as edited:
        for name in held.namelist():
            edited.writestr(name, pyyaml.safe_dump(other) if name == 'spec.yaml' else held.read(name))

    with pytest.raises(sps.SpecsolveError, match='came back from a different spec'):
        sps.load_archive(tampered, tmp_path / 'out')
    assert sps.load_archive(archive, tmp_path / 'fine').spec == to_spec(dispatch_yaml), (
        'and the archive as written reads back as the model it holds'
    )


def test_every_slice_of_a_sweep_names_the_model_it_answered(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A sweep's digest is the sweep's, not each slice's."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'))
    alone = sps.solve(dispatch_yaml, dispatch_frame_inputs)

    assert runs.record['spec_digest'].null_count() == 0, 'no slice is left without the document it answered'
    assert runs.record['spec_digest'].unique().to_list() == [alone.spec_digest], (
        'and it is the same digest one solve of the same file carries'
    )


def test_a_sweep_archive_whose_answer_names_another_spec_is_refused(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A sweep archive with a swapped `spec.yaml` is refused."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=tmp_path / 'study')
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1.0})
    (tmp_path / 'study' / 'spec.yaml').write_text(pyyaml.safe_dump(other))

    with pytest.raises(sps.SpecsolveError, match='came back from a different spec'):
        sps.load_archive(tmp_path / 'study')


def test_saving_an_answer_twice_leaves_only_the_second(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A directory holds one answer, so the first one's frames do not survive."""
    renamed = raw_of(dispatch_yaml)
    renamed['variables']['q'] = renamed['variables'].pop('p')
    renamed['constraints']['power_balance']['expression'] = 'sum(q, over=generator) == load'
    renamed['objective']['expression'] = 'sum(q * cost)'

    sps.solve(dispatch_yaml, dispatch_frame_inputs).save(tmp_path / 'shared')
    out = sps.solve(renamed, dispatch_frame_inputs).save(tmp_path / 'shared')

    assert sorted(f.stem for f in (out / 'primal').glob('*.parquet')) == ['q'], (
        "only the second model's variable is left"
    )
    with pytest.raises(KeyError, match='unknown variable'):
        sps.load_result(out).primal('p')


def test_saving_an_answer_into_an_unpacked_archive_takes_its_metrics_row_with_it(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A metrics row belongs to the answer beside it, so a second save removes it."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case')
    answer = tmp_path / 'case' / ANSWER_DIR
    assert (answer / METRICS_FILE).is_file(), 'the archive wrote one, or this proves nothing'

    sps.solve(dispatch_yaml, dispatch_frame_inputs).save(answer)

    assert not (answer / METRICS_FILE).exists(), 'and a saved answer carries no metrics, so none is left behind'


def test_saved_cases_say_whether_they_are_comparable(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path) -> None:
    """One distinct `spec_digest` in concatenated records says they compare like with like."""
    other = override(raw_of(dispatch_yaml), **{'variables.p.bounds.upper': 1000.0})
    records = []
    for name, spec in (('base', dispatch_yaml), ('capped', other)):
        with sps.solve(spec, dispatch_frame_inputs) as solved:
            out = solved.save(tmp_path / name)
        records.append(pl.read_parquet(out / 'record.parquet').select(pl.lit(name).alias('case'), pl.all()))

    table = pl.concat(records)
    assert table['spec_digest'].n_unique() == 2, 'two models, so the table is not comparing like with like'
    assert sps.load_result(tmp_path / 'base').spec_digest == table.filter(pl.col('case') == 'base')['spec_digest'][0], (
        'and a loaded answer carries the digest its record holds'
    )


def test_a_saved_answer_is_stamped_with_its_layout_and_the_specsolve_that_wrote_it(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The version is what a later reader names when it refuses a layout it no longer reads."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        out = solved.save(tmp_path / 'solution')

    assert json.loads((out / 'format.json').read_text()) == {'layout': LAYOUT, 'specsolve': sps.__version__}, (
        'the layout this package writes, beside the version that wrote it'
    )


def test_an_answer_in_another_layout_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An answer in another layout names the specsolve that wrote it; one with no stamp is refused too."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        out = solved.save(tmp_path / 'solution')
    (out / 'format.json').write_text(json.dumps({'layout': 0, 'specsolve': '0.0.1a359'}))

    with pytest.raises(sps.LayoutError, match='solve the model again and save it') as refused:
        sps.load_result(out)
    assert f'layout 0, written by specsolve 0.0.1a359, and this package reads layout {LAYOUT}' in str(refused.value), (
        'the refusal names the layout it found and the version that wrote it'
    )

    (out / 'format.json').write_text(json.dumps({'answer': 0}))  # what 0.1.0 wrote
    with pytest.raises(sps.LayoutError, match='with no layout stamp'):
        sps.load_result(out)

    (out / 'format.json').unlink()
    with pytest.raises(sps.LayoutError, match='with no layout stamp'):
        sps.load_result(out)


def test_a_hand_built_axis_is_refused(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A list of `(key, sources)` is refused, and sent to one archive per solve."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    with pytest.raises(sps.SpecsolveError, match='archive one solve each'):
        sps.solve_over(dispatch_yaml, sources, [('a', sources)], key_name='case', archive='never.zip')


def _by_scenario(names: list[str]) -> pl.DataFrame:
    return pl.concat(
        [
            pl.DataFrame({'snapshot': range(DISPATCH_SNAPSHOTS), 'value': _dispatch_load()}).with_columns(
                pl.lit(name).alias('scenario')
            )
            for name in names
        ]
    )


def test_a_sweep_refuses_a_lowered_program_before_it_solves_a_slice(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A sweep refuses it at the same door, and before the first slice is taken."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    with pytest.raises(sps.SpecsolveError, match='was lowered from'):
        sps.solve_over(sps.check(dispatch_yaml), sources, sps.EachCoordinate('scenario'), archive='never.zip')


def test_two_writers_to_one_target_stage_in_separate_places(tmp_path: Path) -> None:
    """The staging name is not a function of the target, so two writers cannot meet in it."""
    out = tmp_path / 'case'

    assert _staging_for(out) != _staging_for(out), 'two writers to one target stage in separate places'


def test_a_written_archive_leaves_no_staging_beside_it(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path) -> None:
    """What a writer stages is its own to remove, in both containers."""
    for name in ('case', 'case.zip'):
        _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / name)
    assert sorted(path.name for path in tmp_path.iterdir()) == ['case', 'case.zip'], (
        'the staging each write allocated is gone, leaving the two archives alone'
    )


# ---------------------------------------------------------------------------
# the model digest: which data an answer came back from
# ---------------------------------------------------------------------------


#: A quantity `examples/dispatch.yaml` never names, so no saved frame carries it.
_UNDECLARED = 'sum(p * cost, over=generator)'


def test_a_solve_that_never_saves_hashes_nothing(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """The digest is computed when asked for, not at the solve."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        solved.primal('p')
        assert callable(solved._model_digest), 'reading values must not hash the model'
        assert isinstance(solved.model_digest(), str), 'and asking for it produces one'
        assert isinstance(solved._model_digest, str), 'which is then kept rather than hashed again'


def test_two_builds_of_one_model_digest_the_same(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """The digest names a model, however its sparse frames are laid out."""
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as one, sps.build(dispatch_yaml, dispatch_frame_inputs) as two:
        assert one._model_digest() == two._model_digest(), (
            'one model, one digest, however its sparse frames are laid out'
        )

    halved = dispatch_frame_inputs | {'cost': dispatch_frame_inputs['cost'].with_columns(pl.col('value') * 0.5)}
    with sps.build(dispatch_yaml, dispatch_frame_inputs) as base, sps.build(dispatch_yaml, halved) as other:
        assert base._model_digest() != other._model_digest(), 'and data that moved a cost is a different model'


def test_an_archive_whose_data_was_replaced_is_refused_at_the_rebuild(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A source member replaced since the archive was written is refused at the rebuild."""
    _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'case')
    intact = sps.load_archive(tmp_path / 'case')
    want = intact.answer.evaluate(_UNDECLARED)['value'].sum()

    moved = dispatch_frame_inputs['cost'].with_columns(pl.col('value') * 99)
    moved.write_parquet(tmp_path / 'case' / 'sources' / 'cost.parquet')

    tampered = sps.load_archive(tmp_path / 'case')
    with pytest.raises(sps.SpecsolveError, match='came back from another model') as refused:
        tampered.answer.evaluate(_UNDECLARED)
    assert 'what differs is the data' in str(refused.value), 'and the refusal says which half moved'
    assert want == pytest.approx(intact.answer.evaluate(_UNDECLARED)['value'].sum(), rel=1e-9), (
        'while the archive as written still reads'
    )


def test_an_answer_naming_no_model_is_taken_as_given(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An answer written before the column has no digest to compare, and is not refused for it."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        want = solved.evaluate(_UNDECLARED)['value'].sum()
        solved.save(tmp_path / 'answer')

    older = replace(sps.load_result(tmp_path / 'answer'), _model_digest=None)
    read = attach_readers(older, dispatch_yaml, dispatch_frame_inputs)
    assert read.evaluate(_UNDECLARED)['value'].sum() == pytest.approx(want, rel=1e-9), (
        'an answer carrying no model digest reads against the data it is handed'
    )
