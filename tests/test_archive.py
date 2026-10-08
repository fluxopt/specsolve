"""``archive=``: a spec, its data and its answer as one file, and back.

What attaches from the archive is what attached from the caller's own tables,
frame for frame, over every ported instance.
"""

from __future__ import annotations

import json
import math
import shutil
import zipfile
from dataclasses import replace
from datetime import UTC, datetime
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml
from mathspec import to_spec

import specsolve as sps
from specsolve import strategy
from specsolve.api import _provenance
from specsolve.archive import _attach_readers
from specsolve.archive_layout import ANSWER_DIR, INPUTS_LAYOUT, _staging_for
from specsolve.relational.answer_layout import (
    ANSWER_LAYOUT,
    METRICS_FILE,
    RUN,
    Metrics,
    Output,
    Provenance,
    Record,
    digest_of_data,
    digest_of_file,
    read_reasons,
    write_reasons,
)
from specsolve.relational.sinks.solvers import SOLVERS
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
from tests.test_strategy import SPENDING, WINDOW, horizon_sources

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from mathspec import Spec


def _question(archive: sps.types.ResultArchive | sps.types.SweepArchive) -> tuple[Spec, Mapping[str, object]]:
    """The pair every verb takes, read off an archive."""
    return archive.spec, archive.sources


def _unstamped_digest(member: Path, scratch: Path) -> str:
    """The digest of an archived source as the archive takes it: its table written before the run was stamped on."""
    unstamped = scratch / f'{member.stem}.unstamped.parquet'
    pl.read_parquet(member).drop(RUN).lazy().sink_parquet(unstamped, compression='zstd')
    return digest_of_file(unstamped)


def _archived(spec, sources, out: Path, outputs: frozenset[Output] = frozenset()) -> Path:
    """The archive a solve writes, which is the only way one is made."""
    with sps.solve(spec, sources, archive=out, outputs=outputs):
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
    tidied = sps.tidy(expanded(port_spec(name)), sources)
    unlike = [key for key, table in tidied.items() if not unpacked[key].equals(table)]
    assert not unlike, f'members that are not the table tidy() returns: {unlike}'


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
    assert snapshot.columns == ['snapshot', 'specsolve_position', RUN], (
        'a bare label range is written as an index table'
    )
    assert snapshot['specsolve_position'].to_list() == list(range(DISPATCH_SNAPSHOTS)), (
        'one row per label, numbered in index order'
    )


def test_the_archive_is_the_file_and_stored_parquet(dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path) -> None:
    archive = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'dispatch.zip')
    with zipfile.ZipFile(archive) as zipped:
        members = {info.filename: info.compress_type for info in zipped.infolist()}
        beside_the_answer = {name for name in members if not name.startswith('answer/')}
        assert beside_the_answer == {
            'format.json',
            'spec.yaml',
            'sources.parquet',
            'catalog.parquet',
            *(f'sources/{k}.parquet' for k in dispatch_frame_inputs),
        }, (
            'the layout is its stamp, spec.yaml, one parquet member per source key, the table digesting them, the '
            'catalog saying what each file holds, and the answer under its own'
        )
        assert json.loads(zipped.read('format.json')) == {'layout': INPUTS_LAYOUT, 'specsolve': sps.__version__}, (
            'the spec and the sources are stamped with a layout of their own, apart from the answer'
        )
        assert any(name.startswith('answer/') for name in members), 'every archive carries the answer that made it'
        assert set(members.values()) == {zipfile.ZIP_STORED}, 'members are stored — parquet is already compressed'
        assert to_spec(pyyaml.safe_load(zipped.read('spec.yaml'))) == to_spec(dispatch_yaml), (
            'spec.yaml is the spec the source file declares'
        )


def test_a_parquet_path_is_archived_as_the_table_the_solve_read(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The archive holds one form whatever arrived, so a path is neither copied nor digested as its own bytes."""
    load = dispatch_frame_inputs['load'].with_columns(pl.lit('a stray column').alias('note'))
    path = tmp_path / 'load.parquet'
    load.write_parquet(path)
    sources = {**dispatch_frame_inputs, 'load': str(path)}
    archive = _archived(dispatch_yaml, sources, tmp_path / 'dispatch')
    member = archive / 'sources' / 'load.parquet'

    held = pl.read_parquet(member)
    assert held.drop(RUN).equals(sps.tidy(dispatch_yaml, sources)['load']), (
        'the stray column is gone, as it is at attach, and the run column is the only one added'
    )
    assert held[RUN].unique().to_list() == ['dispatch'], 'stamped with the run that wrote it'
    digests = dict(pl.read_parquet(archive / 'sources.parquet').select('source', 'digest').iter_rows())
    assert digests['load'] == _unstamped_digest(member, tmp_path), 'and the digest is of that table, less the stamp'


def test_an_index_given_as_an_iterator_is_archived_as_the_labels_the_solve_read(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The archive read the sources a second time after the solve, so a one-shot iterator came back empty.

    The solve answered, then the archive raised a `DataError` naming labels
    the exhausted iterator did not hold, and the answer was lost.
    """
    sources = {**dispatch_frame_inputs, 'generator': iter(DISPATCH_GENERATORS)}
    archive = _archived(dispatch_yaml, sources, tmp_path / 'dispatch')

    held = pl.read_parquet(archive / 'sources' / 'generator.parquet')
    assert held['generator'].to_list() == list(DISPATCH_GENERATORS), 'the labels the build read, in index order'


def test_a_path_rewritten_after_the_build_is_archived_as_the_table_the_solve_read(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The archive read a path again after the solve, so a file rewritten meanwhile was archived as the new table."""
    path = tmp_path / 'load.parquet'
    dispatch_frame_inputs['load'].write_parquet(path)
    with sps.build(dispatch_yaml, {**dispatch_frame_inputs, 'load': str(path)}) as model:
        dispatch_frame_inputs['load'].with_columns(pl.col('value') * 0.5).write_parquet(path)
        with model.solve(archive=tmp_path / 'dispatch'):
            pass

    held = pl.read_parquet(tmp_path / 'dispatch' / 'sources' / 'load.parquet')
    assert held['value'].to_list() == dispatch_frame_inputs['load']['value'].to_list(), (
        'the values the build read, not the ones the file holds now'
    )


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
    with pytest.raises(sps.errors.DataError, match="no data provided for parameter 'cost'"):
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
    with pytest.raises(sps.errors.LayoutError) as excinfo:
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
    assert loose.result.objective == unpacked.result.objective
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
    read: Callable[..., sps.types.ResultArchive | sps.types.SweepArchive],
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
    with pytest.raises(sps.errors.LayoutError, match=says):
        read(out, None if into is None else tmp_path / into)


def test_loading_a_zip_needs_nowhere_to_unpack_and_leaves_nothing_behind(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """What `load_archive` reads whole it does not read again, so the members are scratch."""
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip')
    beside_it = sorted(path.name for path in tmp_path.iterdir())

    case = sps.load_archive(tmp_path / 'case.zip')

    assert case.result.primal('p').height > 0, 'the answer came back with no directory named to read it off'
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
    with pytest.raises(sps.errors.LayoutError, match='already holds something'):
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
    with pytest.raises(sps.errors.SpecsolveError, match='lowered Program is not a spec this takes'):
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


@pytest.mark.parametrize(
    'archive',
    [
        pytest.param(lambda spec, sources, out: _archived(spec, sources, out), id='solve'),
        pytest.param(
            lambda spec, sources, out: sps.solve_over(
                spec, {**sources, 'load': _by_scenario(['low', 'high'])}, sps.EachCoordinate('scenario'), archive=out
            ),
            id='solve_over',
        ),
    ],
)
def test_a_directory_of_archives_never_globs_one_still_being_written(
    archive, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, monkeypatch
) -> None:
    """A directory of archives is read as ``read_parquet('<dir>/*/<member>')``, so nothing staged may sit there.

    A sweep laid its answer out in ``<dir>/tmpXXXX/answer/``, which that glob
    read as an archive while it was written, and a reader that globbed
    during a solve failed once the scratch was removed. Looked at the moment
    the catalogs are written, when the answer and the staged archive are both
    complete on disk and nothing has landed yet.
    """
    from specsolve import archive_layout

    staged: list[Path] = []
    catalogs = archive_layout._write_catalogs

    def watched(*args, **kwargs):
        staged.extend(p.relative_to(tmp_path) for p in tmp_path.rglob('*') if p.is_file())
        return catalogs(*args, **kwargs)

    monkeypatch.setattr(archive_layout, '_write_catalogs', watched)
    out = tmp_path / 'study'
    archive(dispatch_yaml, dispatch_frame_inputs, out)
    members = {p.relative_to(out) for p in out.rglob('*') if p.is_file()}
    globbed = sorted(str(p) for p in staged if len(p.parts) > 1 and p.relative_to(p.parts[0]) in members)
    assert globbed == [], f'staged where <dir>/*/<member> reads them, before the archive landed: {globbed}'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['study'], 'and nothing staged is left beside the archive'


def test_an_archive_carries_the_answer_beside_the_question(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The whole archive: what was asked, the data it was asked of, and what came back."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'case.zip') as solved:
        loaded = sps.load_archive(tmp_path / 'case.zip', tmp_path / 'case')

        assert loaded.result.objective == solved.objective
        for name in to_spec(dispatch_yaml).program.variables:
            assert loaded.result.primal(name).equals(solved.primal(name))

    with sps.solve(*_question(loaded)) as resolved:
        assert resolved.objective == pytest.approx(loaded.result.objective, rel=1e-9), (
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

    assert base.result.spec_digest == other.result.spec_digest, 'one document, so the spec digest cannot separate them'
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


@pytest.mark.parametrize('read', [sps.load_archive, sps.scan_archive], ids=['loaded', 'scanned'])
def test_the_sources_an_archive_gives_back_archive_again_to_the_same_digests(
    read: Callable[[Path], sps.types.ResultArchive | sps.types.SweepArchive],
    dispatch_yaml: Path,
    dispatch_frame_inputs,
    tmp_path,
) -> None:
    """A scanned source is the member itself, so it carries the first archive's `specsolve_run`.

    Digested with that column, archiving what `scan_archive` gave back would
    move every digest though not one number had; the tidy table the solve
    reads leaves it out.
    """
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=tmp_path / 'first').close()
    sps.solve(*_question(read(tmp_path / 'first')), archive=tmp_path / 'second').close()
    first, second = sps.load_archive(tmp_path / 'first'), sps.load_archive(tmp_path / 'second')

    assert second.source_digests.drop(RUN).equals(first.source_digests.drop(RUN)), (
        'the same tables archived again digest alike, whichever reader gave them back'
    )
    assert pl.read_parquet(tmp_path / 'second' / 'sources' / 'load.parquet')[RUN].unique().to_list() == ['second'], (
        'and the member is stamped with the run that wrote it, not the one it was read from'
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

    assert isinstance(study, sps.types.SweepArchive), 'the archive carries an axis, or this is testing the other type'
    assert study.source_digests['source'].to_list() == sorted(study.sources), 'one row per source, as for one solve'
    assert study.source_digests[RUN].unique().to_list() == ['study'], (
        'stamped with the archive name as a solve archive is, the sweep key belonging to the slices and not the data'
    )
    held = tmp_path / 'study' / 'sources'
    assert pl.read_parquet(held / 'load.parquet')['scenario'].unique().sort().to_list() == ['high', 'low'], (
        'the member is the whole source the sweep was cut from, not one slice of it'
    )
    assert dict(study.source_digests.select('source', 'digest').iter_rows()) == {
        file.stem: _unstamped_digest(file, tmp_path) for file in held.glob('*.parquet')
    }, 'and each digest is of the whole table the member holds, less the stamp'


def test_a_sweep_over_an_index_given_as_an_iterator_reads_it_once(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """Every slice and the archive read each source again, so a one-shot iterator was empty after the first read.

    The second slice raised a `DataError` naming labels the exhausted iterator
    did not hold.
    """
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high']), 'generator': iter(DISPATCH_GENERATORS)}
    sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=tmp_path / 'study')

    held = pl.read_parquet(tmp_path / 'study' / 'sources' / 'generator.parquet')
    assert held['generator'].to_list() == list(DISPATCH_GENERATORS), 'the labels every slice read, in index order'


#: A relation with a role, a named expression over no dimension, and a
#: constraint sharing its name with a parameter, which names outside the flat
#: namespace may do.
_CATALOGED = {
    'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str', 'description': 'nodes'}},
    'relations': {'sited': {'key': 'generator', 'values': {'at': 'bus'}}},
    'parameters': {
        'p_max': {'dims': ['generator'], 'description': 'rating'},
        'cost': {'dims': ['generator']},
        'load': {'dims': ['bus']},
    },
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'load': {'dims': ['bus'], 'expression': 'sum(p, by=sited, over=generator, into=at) == load'}},
    'expressions': {'spend': {'expression': 'sum(p * cost)', 'description': 'what the dispatch costs'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def _cataloged_sources(north: float) -> dict[str, object]:
    """``_CATALOGED``'s data, *north* the load at the bus whose only generator is rated 100."""
    generators = ['wind', 'gas']
    return {
        'generator': generators,
        'bus': ['north', 'south'],
        'sited': pl.DataFrame({'generator': generators, 'at': ['north', 'south']}),
        'p_max': pl.DataFrame({'generator': generators, 'value': [100.0, 200.0]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0, 50.0]}),
        'load': pl.DataFrame({'bus': ['north', 'south'], 'value': [north, 80.0]}),
    }


def test_the_catalog_says_what_each_file_holds_and_which_column_holds_each_dimension(tmp_path: Path) -> None:
    """``name`` was the only key, and a constraint named as a parameter is a second ``load``.

    A query on ``name = 'load'`` interleaved the two. The catalog had no path to
    tell ``sources/load.parquet`` from ``answer/dual/load.parquet``, and no run
    to tell one archive's catalog from another's.
    """
    _archived(to_spec(_CATALOGED), _cataloged_sources(50.0), tmp_path / 'base', frozenset({'activity'}))

    assert pl.read_parquet(tmp_path / 'base' / 'catalog.parquet').rows() == [
        ('base', 'answer/activity/load.parquet', 'load', 'constraint', None, None, 'bus', 'bus'),
        ('base', 'answer/dual/load.parquet', 'load', 'constraint', None, None, 'bus', 'bus'),
        (
            'base',
            'answer/expression/spend.parquet',
            'spend',
            'expression',
            'what the dispatch costs',
            None,
            None,
            None,
        ),
        ('base', 'answer/primal/p.parquet', 'p', 'variable', None, None, 'generator', 'generator'),
        ('base', 'sources/bus.parquet', 'bus', 'dimension', 'nodes', 'str', 'bus', 'bus'),
        ('base', 'sources/cost.parquet', 'cost', 'parameter', None, 'float', 'generator', 'generator'),
        ('base', 'sources/generator.parquet', 'generator', 'dimension', None, 'str', 'generator', 'generator'),
        ('base', 'sources/load.parquet', 'load', 'parameter', None, 'float', 'bus', 'bus'),
        ('base', 'sources/p_max.parquet', 'p_max', 'parameter', 'rating', 'float', 'generator', 'generator'),
        ('base', 'sources/sited.parquet', 'sited', 'relation', None, None, 'at', 'bus'),
        ('base', 'sources/sited.parquet', 'sited', 'relation', None, None, 'generator', 'generator'),
    ], (
        'one row per column that holds labels of each file, in path and column order: the constraint load and the parameter '
        'load are told apart by path, a relation names each role and its dimension, a name over no dimension has '
        'one row with no column, and dtype is only what the spec declares'
    )


def test_a_parameter_passed_in_another_column_order_is_catalogued_as_its_tidy_table(tmp_path: Path) -> None:
    """A parameter passed as a path in another column order is archived as its tidy table, in the spec's order."""
    spec = to_spec(
        {
            'dimensions': {'generator': {'dtype': 'str'}, 'snapshot': {'dtype': 'int'}},
            'parameters': {'p_max': {'dims': ['generator', 'snapshot']}},
            'variables': {'p': {'dims': ['generator', 'snapshot'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
            'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
        }
    )
    path = tmp_path / 'p_max.parquet'
    pl.DataFrame(
        {
            'value': [100.0, 90.0],
            'snapshot': [0, 0],
            'note': ['a stray column'] * 2,
            'generator': ['wind', 'gas'],
        }
    ).write_parquet(path)
    _archived(spec, {'generator': ['wind', 'gas'], 'snapshot': [0], 'p_max': str(path)}, tmp_path / 'case')
    held = pl.read_parquet_schema(tmp_path / 'case' / 'sources' / 'p_max.parquet')
    catalog = pl.read_parquet(tmp_path / 'case' / 'catalog.parquet')
    places = catalog.filter(pl.col('path') == 'sources/p_max.parquet').select('column', 'dim').rows()

    assert list(held) == ['generator', 'snapshot', 'value', RUN], (
        "the archived file is the tidy table: the declared order rather than the caller's, the stray column gone"
    )
    assert places == [('generator', 'generator'), ('snapshot', 'snapshot')], (
        'the catalog lists the columns of the archived file, and not the stray one the caller wrote'
    )


def _held_files(archive: Path) -> set[str]:
    """Every source, and every frame of the answer, as its path inside *archive*: a file, or a sweep's directory."""
    held = [*(archive / 'sources').glob('*.parquet'), *(archive / ANSWER_DIR).glob('*/*')]
    return {entry.relative_to(archive).as_posix() for entry in held}


def _columns_of(held: Path) -> set[str]:
    """The columns of a frame's file, or of the first slice's where a sweep holds a directory of them."""
    return set(pl.read_parquet_schema(min(held.glob('*.parquet')) if held.is_dir() else held))


@pytest.mark.parametrize(
    ('north', 'sweep', 'answered'),
    [
        pytest.param(50.0, False, True, id='solve'),
        pytest.param(500.0, False, False, id='infeasible'),
        pytest.param(50.0, True, True, id='sweep'),
    ],
)
def test_every_archive_catalogs_the_files_it_holds_and_no_other(
    north: float, sweep: bool, answered: bool, tmp_path: Path
) -> None:
    """The catalog listed what the spec declares, so an infeasible solve's catalog named frames it has no file for."""
    out, sources = tmp_path / 'case', _cataloged_sources(north)
    if sweep:
        load = pl.concat([sources['load'].with_columns(scenario=pl.lit(name)) for name in ('low', 'high')])
        sps.solve_over(to_spec(_CATALOGED), {**sources, 'load': load}, sps.EachCoordinate('scenario'), archive=out)
    else:
        _archived(to_spec(_CATALOGED), sources, out)
    catalog = pl.read_parquet(out / 'catalog.parquet')
    held = _held_files(out)

    assert any(path.startswith(f'{ANSWER_DIR}/') for path in held) == answered, 'the case leaves the frames it says'
    assert set(catalog['name']) == {path.split('/')[-1].removesuffix('.parquet') for path in held}, (
        'the catalog names what the archive holds a file for, and nothing it does not'
    )
    assert set(catalog['path']) == held, 'every source and answer file is listed by its path, and every path is there'
    assert catalog.select('path', 'column').is_duplicated().sum() == 0, 'a path and a column name one row'
    assert catalog[RUN].unique().to_list() == ['case'], 'every row carries the archive it came from'
    for path, column in catalog.drop_nulls('column').select('path', 'column').iter_rows():
        assert column in _columns_of(out / path), f'{path} holds the column {column!r} its row names'
    assert all(RUN in _columns_of(out / path) for path in held), 'every file listed carries the run'
    assert RUN not in set(catalog['column']), 'the run on every file holds no labels, so the catalog lists it nowhere'


def _labels_of(held: Path) -> set[str]:
    """The columns of a frame's file that hold labels: neither the value nor one specsolve adds."""
    return {column for column in _columns_of(held) if column != 'value' and not column.startswith('specsolve_')}


def _unpacked(packed: Path, into: Path) -> Path:
    with zipfile.ZipFile(packed) as archive:
        archive.extractall(into)
    return into


@pytest.mark.parametrize(
    ('windowed', 'path', 'column', 'dim'),
    [
        pytest.param(True, 'answer/primal/soc.parquet', 'snapshot', 'snapshot', id='a-rolling-horizon-answer'),
        pytest.param(True, 'sources/load.parquet', 'snapshot', 'snapshot', id='a-rolling-horizon-cut-source'),
        pytest.param(False, 'answer/primal/p.parquet', 'scenario', 'scenario', id='a-coordinate-sweep-answer'),
        pytest.param(False, 'sources/load.parquet', 'scenario', 'scenario', id='a-coordinate-sweep-cut-source'),
    ],
)
def test_a_query_the_catalog_drives_reads_every_file_of_a_sweep_archive_as_written(
    windowed: bool, path: str, column: str, dim: str, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The catalog was built from the spec, so a sweep archive's files did not hold what it listed.

    A rolling horizon's answer and cut sources hold ``snapshot`` where the
    catalog said ``t``, and a coordinate sweep's files hold a ``scenario``
    column the catalog did not list.
    """
    if windowed:
        _rolling(tmp_path)
        out = _unpacked(tmp_path / 'roll.zip', tmp_path / 'roll')
    else:
        out = tmp_path / 'study'
        sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
        sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=out)
    catalog = pl.read_parquet(out / 'catalog.parquet')

    for (held,), rows in catalog.group_by('path'):
        listed = set(rows['column'].drop_nulls())
        assert listed == _labels_of(out / str(held)), f'{held} holds the label columns its rows name, and no other'
        pl.scan_parquet(out / str(held)).select(*listed).collect()
    assert catalog.filter(path=path, column=column)['dim'].to_list() == [dim], (
        f'{path} holds {column!r} as written, over the dimension the sweep cut'
    )


def test_a_rolling_horizon_lists_its_windows_in_their_own_catalog(tmp_path: Path) -> None:
    """The windows are listed apart, so the archive's catalog is the same whether they are kept or not."""
    _rolling(tmp_path / 'unkept')
    _rolling(tmp_path / 'kept', keep_windows=True)
    unkept = _unpacked(tmp_path / 'unkept' / 'roll.zip', tmp_path / 'unkept' / 'roll')
    kept = _unpacked(tmp_path / 'kept' / 'roll.zip', tmp_path / 'kept' / 'roll')
    windows = pl.read_parquet(kept / 'answer' / 'windows' / 'catalog.parquet')
    held = {
        entry.relative_to(kept).as_posix()
        for entry in [*(kept / 'answer' / 'windows').glob('*/*'), kept / 'answer' / 'windows' / 'owned.parquet']
    }

    assert pl.read_parquet(kept / 'catalog.parquet').equals(pl.read_parquet(unkept / 'catalog.parquet')), (
        "the archive's catalog does not change when the windows are kept"
    )
    assert not (unkept / 'answer' / 'windows').exists(), 'no windows and no window catalog unless they are kept'
    assert not (unkept / 'answer' / 'owned.parquet').exists(), 'what each window owns is kept with the windows'
    assert set(windows['path']) == held, 'the window catalog lists every window frame and what each window owns'
    for (path,), rows in windows.group_by('path'):
        assert set(rows['column']) == _labels_of(kept / str(path)), f'{path} holds the label columns its rows name'
    assert windows.filter(path='answer/windows/primal/soc', column='snapshot_start')['dim'].to_list() == ['snapshot'], (
        'the key a window frame carries is where the window started, a coordinate of the dimension the axis cut'
    )
    assert windows.filter(path='answer/windows/primal/soc', column='t')['dim'].to_list() == ['t'], (
        'inside a window the frame holds the local index the spec declares'
    )
    owned = windows.filter(path='answer/windows/owned.parquet').select('column', 'dim').rows()
    assert owned == [('snapshot', 'snapshot'), ('snapshot_start', 'snapshot'), ('t', 't')], (
        'what each window owns maps the window and its local index to a coordinate of the sliced dimension'
    )


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
        'slice_axis',
        'slice',
    ), (
        'the sizes, the counters, one clock per phase in the order the phases run, the run that took them, '
        'then the sweep slice, null here'
    )
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

    with pytest.raises(sps.errors.LayoutError, match=r"Metrics row that is short of \['write_seconds'\]") as excinfo:
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
    answer = sps.load_archive(tmp_path / 'nightly-2026-09-10.zip').result
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
    answer = case.sweep if isinstance(case, sps.types.SweepArchive) else case.result
    assert answer.primal('p').equals(live), 'a frame read back is the frame the solve returned'
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
    expected = loaded.result.primal('p')

    assert isinstance(loaded.sources['load'], pl.DataFrame), 'a loaded source is the table the member holds'
    assert scanned.sources['load'] == tmp_path / 'out' / 'sources' / 'load.parquet', 'a scanned one is the path to it'
    shutil.rmtree(tmp_path / 'out')

    assert loaded.result.primal('p').equals(expected), 'the loaded answer was read before the members went'
    with pytest.raises(FileNotFoundError):
        scanned.result.primal('p')


def test_a_loaded_sweep_archive_answers_the_frame_readers(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """`load_archive` gives a held `Sweep`, `scan_archive` one that reads off the unpacked files."""
    axis = sps.EachCoordinate('scenario')
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    sps.solve_over(dispatch_yaml, sources, axis, archive=tmp_path / 'study.zip')

    loaded = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')
    scanned = sps.scan_archive(tmp_path / 'study.zip', tmp_path / 'study')

    assert loaded.sweep.primal('p').equals(scanned.sweep.primal('p')), 'the same study, read two ways'
    assert loaded.sweep.primal('p').equals(scanned.sweep.scan('p').collect()), 'and lazily'


def _attached_differently(spec, sources, archived: sps.types.SweepArchive) -> list[tuple[int, str]]:
    """``(slice, source)`` where a slice cut from *archived* attaches other than one cut from *sources*."""
    before = [sps.tidy(spec, cut) for _, cut in archived.axis.slices(sources)]
    after = [sps.tidy(archived.spec, cut) for _, cut in archived.axis.slices(archived.sources)]
    return [
        (position, name)
        for position, (attached, again) in enumerate(zip(before, after, strict=True))
        for name in attached
        if not attached[name].equals(again[name])
    ]


def test_a_scenario_sweep_is_an_archive_and_runs_again(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """The axis travels with the sweep's sources, which the archive holds whole."""
    axis = sps.EachCoordinate('scenario')
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = sps.solve_over(dispatch_yaml, sources, axis, archive=tmp_path / 'study.zip')
    study = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')

    assert study.axis == axis, 'the axis comes back as the value it went in as'
    assert study.sweep.record.drop(RUN).equals(runs.record.drop(RUN))
    assert study.sweep.record[RUN].unique().to_list() == ['study'], (
        'and the archive stamped its own name on every slice, which the sweep in memory had none of'
    )
    assert study.sources['load'].columns == ['scenario', 'snapshot', 'value'], (
        'the sliced source is archived whole and tidy, the column the axis cuts on first'
    )
    differing = _attached_differently(dispatch_yaml, sources, study)
    assert not differing, f'(slice, source) pairs the archive attaches differently: {differing}'
    again = sps.solve_over(study.spec, study.sources, study.axis)
    assert again.record['objective'].to_list() == pytest.approx(runs.record['objective'].to_list()), (
        'the archive re-runs to the sweep it recorded, slice for slice'
    )


#: The overlapping rolling horizon every archive test below solves.
ROLLING = sps.EachWindow('snapshot', steps=4, lookahead=2, into='t')


def _rolling(tmp_path: Path, spec: Mapping[str, object] = WINDOW, **archive: object) -> sps.types.Sweep:
    """The rolling horizon, archived to ``roll.zip`` with *archive*'s keywords."""
    sources = horizon_sources(12)
    return sps.solve_over(
        spec, sources, ROLLING, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll.zip', **archive
    )


@pytest.mark.parametrize(
    'steps', [pytest.param(4, id='the-last-window-shorter'), pytest.param((2, 4, 4), id='the-first-window-shorter')]
)
@pytest.mark.parametrize(
    'price',
    [
        pytest.param(1.0, id='one-number'),
        pytest.param(pl.DataFrame({'t': [0, 1], 'value': [3.0, 1.0]}), id='a-table-over-the-local-index'),
        pytest.param(
            pl.DataFrame({'snapshot': range(10), 'value': [float(s % 3) for s in range(10)]}),
            id='a-table-over-the-axis',
        ),
    ],
)
def test_a_windowed_sweep_runs_again_from_its_archive_whatever_shape_a_source_over_the_window_took(
    price: object, steps: int | tuple[int, ...], tmp_path: Path
) -> None:
    """Each window attaches from the archive what it attached from the sources.

    One number over `t` was archived as the first window's table. Where a
    later window was shorter, it refused the table's extra labels as strays;
    where it was longer, it read the labels the table was short of as absent.
    `soc_initial`, one number over no dimension, is the same in every window.
    """
    from tests.test_strategy import WINDOW, horizon_sources

    spec = {**WINDOW, 'parameters': {**WINDOW['parameters'], 'price': {'dims': ['t']}}}
    spec['objective'] = {'sense': 'minimize', 'expression': 'sum(p * cost) + sum(price * charge)'}
    axis = sps.EachWindow('snapshot', steps=steps, lookahead=0, into='t')
    sources = {**horizon_sources(10), 'price': price, 'soc_initial': 0.0}
    runs = sps.solve_over(spec, sources, axis, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll.zip')
    archived = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'roll')

    differing = _attached_differently(spec, sources, archived)
    assert not differing, f'(window, source) pairs the archive attaches differently: {differing}'
    again = sps.solve_over(archived.spec, archived.sources, archived.axis, carry=archived.carry)
    assert again.record['objective'].to_list() == pytest.approx(runs.record['objective'].to_list()), (
        'the archive re-runs to the sweep it recorded, window for window'
    )


@pytest.mark.parametrize('windowed', [False, True], ids=['a-coordinate-sweep', 'a-rolling-horizon'])
def test_a_sweep_archive_holds_its_answer_where_a_single_solve_does(
    windowed: bool, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """One file per name at `answer/<kind>/<name>.parquet`, holding what the reader returns, and no windows."""
    if windowed:
        runs, name, out = _rolling(tmp_path), 'soc', tmp_path / 'roll.zip'
    else:
        sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
        out = tmp_path / 'study.zip'
        runs, name = sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=out), 'p'
    with zipfile.ZipFile(out) as packed:
        members = packed.namelist()
        archived = pl.read_parquet(packed.read(f'answer/primal/{name}.parquet'))
    assert archived[RUN].unique().to_list() == [out.stem], 'the archived file carries the run it came from'
    assert archived.drop(RUN).equals(runs.primal(name)), 'and less the stamp, it is the answer the sweep reads'
    assert not [member for member in members if member.startswith('answer/windows/')], 'no windows unless asked'
    assert not [member for member in members if member.startswith(f'answer/primal/{name}/')], 'and no file per slice'


def test_a_rolling_horizon_archive_reads_back_its_answer(tmp_path: Path) -> None:
    """The answer off the archive is the answer off the sweep, read whole or off disk."""
    runs = _rolling(tmp_path)
    loaded = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'loaded')
    scanned = sps.scan_archive(tmp_path / 'roll.zip', tmp_path / 'scanned')

    assert loaded.axis == ROLLING
    assert loaded.sweep.primal('soc').equals(runs.primal('soc')), 'the answer read whole'
    assert scanned.sweep.scan('soc').collect().equals(runs.primal('soc')), 'and read off disk'


@pytest.fixture(scope='module')
def unkept(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Where the rolling horizon is archived without its windows, solved once."""
    under = tmp_path_factory.mktemp('unkept')
    _rolling(under)
    return under / 'roll.zip'


@pytest.mark.parametrize(
    ('read', 'refused'),
    [
        pytest.param(sps.load_archive, lambda sweep, out: sweep.primal('soc', per_window=True), id='primal-per-window'),
        pytest.param(sps.scan_archive, lambda sweep, out: sweep.scan('soc', per_window=True), id='scan-per-window'),
        pytest.param(
            sps.load_archive, lambda sweep, out: sweep.evaluate('sum(p, over=generator)'), id='evaluate-undeclared'
        ),
        pytest.param(sps.load_archive, lambda sweep, out: sweep.save(out), id='save'),
    ],
)
def test_a_rolling_horizon_archive_refuses_the_windows_it_did_not_keep(
    read: Callable[..., sps.types.SweepArchive],
    refused: Callable[[sps.types.Sweep, Path], object],
    unkept: Path,
    tmp_path: Path,
) -> None:
    """Every read that needs the windows is refused, naming the way back."""
    sweep = read(unkept, tmp_path / 'opened').sweep
    with pytest.raises(sps.errors.SpecsolveError, match=r'written without keep_windows=True.*sps\.solve_over'):
        refused(sweep, tmp_path / 'resaved')


def test_a_rolling_horizon_archived_with_its_windows_reads_them_back(tmp_path: Path) -> None:
    """`keep_windows=True` writes the per-window frames too, so `per_window=True` reads off the archive."""
    runs = _rolling(tmp_path, keep_windows=True)
    loaded = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'loaded')
    scanned = sps.scan_archive(tmp_path / 'roll.zip', tmp_path / 'scanned')

    assert (tmp_path / 'scanned' / 'answer' / 'windows' / 'primal' / 'soc').is_dir()
    assert loaded.sweep.primal('soc').equals(runs.primal('soc')), 'the answer is read the same way'
    assert loaded.sweep.primal('soc', per_window=True).equals(runs.primal('soc', per_window=True))
    assert scanned.sweep.scan('soc', per_window=True).collect().equals(runs.primal('soc', per_window=True))
    assert loaded.sweep.dual('balance', per_window=True).equals(runs.dual('balance', per_window=True))


@pytest.mark.parametrize('suffix', ['.zip', ''], ids=['a-zip', 'a-directory'])
@pytest.mark.parametrize(
    ('read', 'per_window'),
    [
        pytest.param(sps.load_archive, lambda sweep: sweep.primal('soc', per_window=True), id='load'),
        pytest.param(sps.scan_archive, lambda sweep: sweep.scan('soc', per_window=True), id='scan'),
    ],
)
def test_a_rolling_horizon_whose_windows_wrote_nothing_still_kept_them(
    read: Callable[..., sps.types.SweepArchive],
    per_window: Callable[[sps.types.Sweep], object],
    suffix: str,
    tmp_path: Path,
) -> None:
    """Every window infeasible, so none writes a frame, and `per_window=True` says so as the live sweep does.

    The archive marked the windows kept by an empty `answer/windows/`, which a
    zip, holding files only, dropped: the read said the archive was written
    without keep_windows=True.
    """
    sources = horizon_sources(12)
    sources['load'] = sources['load'].with_columns(pl.col('value') + 1_000)
    out = tmp_path / f'roll{suffix}'
    runs = sps.solve_over(WINDOW, sources, ROLLING, archive=out, keep_windows=True)
    with pytest.raises(sps.errors.SpecsolveError) as live:
        runs.primal('soc', per_window=True)
    archived = read(out, tmp_path / 'opened' if suffix else None).sweep

    assert 'holds no variable frames at all' in str(live.value), 'no window solved, and the live sweep says so'
    with pytest.raises(sps.errors.SpecsolveError) as raised:
        per_window(archived)
    assert str(raised.value) == str(live.value), 'the archive says what the live sweep says'


def test_a_rolling_horizon_archive_stamps_every_table_and_reads_back_without_the_stamp(tmp_path: Path) -> None:
    """The answer, the windows and what each window owns all carry the run on disk, and no reader returns it."""
    out = tmp_path / 'nightly-2026-09-10.zip'
    runs = sps.solve_over(
        WINDOW, horizon_sources(12), ROLLING, carry={'soc_initial': 'soc'}, archive=out, keep_windows=True
    )
    loaded = sps.load_archive(out, tmp_path / 'out').sweep
    scanned = sps.scan_archive(out, tmp_path / 'out').sweep
    tables = sorted((tmp_path / 'out').rglob('*.parquet'))
    stamps = {
        table.relative_to(tmp_path / 'out').as_posix(): pl.read_parquet(table)[RUN].unique().to_list()
        for table in tables
    }
    kept = {'answer/windows/owned.parquet', 'answer/primal/soc.parquet', 'answer/windows/primal/soc/000000.parquet'}

    assert kept <= stamps.keys(), 'the answer, the windows and what each window owns are all archived'
    assert stamps == {table: ['nightly-2026-09-10'] for table in stamps}, 'and every one of them carries the run'
    assert loaded.primal('soc').equals(runs.primal('soc')), 'the answer read whole is the live one'
    assert scanned.scan('soc').collect().equals(runs.primal('soc')), 'and read off disk'
    assert loaded.primal('soc', per_window=True).equals(runs.primal('soc', per_window=True)), 'per window too'
    assert scanned.scan('soc', per_window=True).collect().equals(runs.primal('soc', per_window=True))
    assert loaded.evaluate('soc').equals(runs.primal('soc')), (
        'an expression valued per window off the archive is stitched through what each window owns, stamp left out'
    )


#: The window model with a capacity that is not over the windowed dimension,
#: so each window solves one and it has no answer over `snapshot`.
CAPPED = override(
    WINDOW,
    **{
        'variables.cap': {'dims': ['generator'], 'bounds': {'lower': 0}},
        'constraints.capped': {'dims': ['t', 'generator'], 'expression': 'p <= cap'},
        'constraints.cap_limit': {'dims': ['generator'], 'expression': 'cap <= 50'},
    },
)


@pytest.mark.parametrize(
    ('spec', 'kind', 'name', 'read'),
    [
        pytest.param(
            SPENDING,
            'expression',
            'window_spend',
            lambda sweep, per_window: sweep.evaluate('window_spend', per_window=per_window),
            id='an-expression-reduced-over-the-window',
        ),
        pytest.param(
            CAPPED,
            'primal',
            'cap',
            lambda sweep, per_window: sweep.primal('cap', per_window=per_window),
            id='a-variable-not-over-the-window',
        ),
        pytest.param(
            CAPPED,
            'dual',
            'cap_limit',
            lambda sweep, per_window: sweep.dual('cap_limit', per_window=per_window),
            id='a-constraint-not-over-the-window',
        ),
    ],
)
def test_a_name_with_no_answer_is_left_out_of_the_archive_with_its_reason(
    spec: Mapping[str, object], kind: str, name: str, read: Callable[..., pl.DataFrame], tmp_path: Path
) -> None:
    """A quantity a rolling horizon cannot stitch has no answer file; the archive says why, and keeps it per window."""
    runs = _rolling(tmp_path, spec, keep_windows=True)
    loaded = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'loaded')

    with zipfile.ZipFile(tmp_path / 'roll.zip') as packed:
        assert f'answer/{kind}/{name}.parquet' not in packed.namelist(), 'no answer, so no answer file'
        reasons = pl.read_parquet(packed.read('answer/reasons.parquet'))
    assert reasons.filter((pl.col('kind') == kind) & (pl.col('name') == name)).height == 1, 'one reason for it'
    with pytest.raises(sps.errors.SpecsolveError, match=r'not over the windowed dimension.*per_window=True'):
        read(loaded.sweep, False)
    assert read(loaded.sweep, True).equals(read(runs, True)), 'per window it is all there'


@pytest.mark.parametrize(
    ('kind', 'unstitchable'),
    [pytest.param('primal', 'cap', id='primal'), pytest.param('dual', 'cap_limit', id='dual')],
)
def test_a_rolling_horizon_reads_the_same_names_live_and_off_its_archive(
    kind: str, unstitchable: str, tmp_path: Path
) -> None:
    """`to_dataset()` with no names reads every name that has an answer, live and off the archive alike.

    Live, it listed every name the windows held, so a name not over the
    windowed dimension made it raise, where the archive had left that name
    out and read the rest.
    """
    pytest.importorskip('xarray')
    runs = _rolling(tmp_path, CAPPED)
    archived = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'loaded').sweep.to_dataset(kind=kind)
    live = runs.to_dataset(kind=kind)

    assert live.equals(archived), 'the same answer, read live or off the archive'
    assert unstitchable not in live.data_vars, 'the name with no answer over the window is left out of both'
    assert unstitchable in runs.to_dataset(kind=kind, per_window=True).data_vars, 'and per window it is read'
    with pytest.raises(sps.errors.SpecsolveError, match='not over the windowed dimension'):
        runs.to_dataset(unstitchable, kind=kind)


#: The window model whose one named expression is reduced over the window, so
#: no expression has an answer over `snapshot`.
WINDOW_TOTAL = override(WINDOW, **{'expressions.window_spend': 'sum(sum(p * cost, over=generator), over=t)'})


@pytest.mark.parametrize('archived', [False, True], ids=['live', 'archived'])
def test_a_kind_with_no_answer_over_the_window_names_what_it_left_out(archived: bool, tmp_path: Path) -> None:
    """Every slice solved, so `to_dataset()` names the expression it could not stitch.

    Off the archive it said every slice terminated optimal and the models
    did not solve.
    """
    runs = _rolling(tmp_path, WINDOW_TOTAL)
    sweep = sps.load_archive(tmp_path / 'roll.zip', tmp_path / 'loaded').sweep if archived else runs
    with pytest.raises(sps.errors.SpecsolveError, match=r'window_spend.*not over the windowed dimension'):
        sweep.to_dataset(kind='expression')


@pytest.mark.parametrize('windowed', [False, True], ids=['a-coordinate-sweep', 'a-rolling-horizon'])
def test_a_sweep_read_off_its_archive_saves_as_the_sweep_it_was(
    windowed: bool, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """`save` on an archive's sweep writes the slices the live sweep holds.

    The archive stamps `specsolve_run` on the metrics as on the record, and
    `save` passed every metrics column to `SliceMetrics`, which declared no
    `specsolve_run`, so it raised `TypeError`.
    """
    if windowed:
        runs, name, out = _rolling(tmp_path, keep_windows=True), 'soc', tmp_path / 'roll.zip'
    else:
        sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
        out = tmp_path / 'study.zip'
        runs, name = sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=out), 'p'
    resaved = sps.load_archive(out, tmp_path / 'loaded').sweep.save(tmp_path / 'resaved')
    saved = sps.load_sweep(resaved)
    stamped = [
        table.relative_to(resaved).as_posix()
        for table in sorted(resaved.rglob('*.parquet'))
        if table.parent.name not in {'record', 'metrics'} and RUN in pl.read_parquet_schema(table)
    ]

    assert saved.primal(name).equals(runs.primal(name)), 'the answer comes back'
    assert saved.metrics.drop(RUN).equals(runs.metrics.drop(RUN)), 'and the metrics are the slices own'
    assert saved.metrics[RUN].equals(saved.record[RUN]), 'naming the run they came from as the record does'
    assert stamped == [], 'no frame the save writes carries the run, what each window owns included'


def test_a_reason_for_one_dual_is_not_a_reason_for_every_dual(tmp_path: Path) -> None:
    """A dual left out by name reads back under its name, never as the reason the whole kind is absent."""
    write_reasons(tmp_path, None, {'dual': {'cap_limit': 'not over the window'}})
    assert read_reasons(tmp_path) == (None, {'dual': {'cap_limit': 'not over the window'}}), (
        'the whole kind keeps its duals, and the one name left out keeps its reason'
    )


@pytest.mark.parametrize(
    ('axis', 'archived', 'says'),
    [
        pytest.param(sps.EachCoordinate('scenario'), True, 'does not cut windows', id='a-coordinate-sweep'),
        pytest.param(
            sps.EachWindow('snapshot', steps=4, lookahead=0, into='t'), False, 'there is no archive=', id='no-archive'
        ),
    ],
)
def test_keep_windows_is_refused_where_there_are_no_windows_to_keep(
    axis: sps.EachCoordinate | sps.EachWindow, archived: bool, says: str, monkeypatch, tmp_path: Path
) -> None:
    """Refused when `solve_over` is called, before a slice is built."""

    def no_build(*args: object, **kwargs: object) -> None:
        raise AssertionError('a slice was built before the refusal')

    monkeypatch.setattr(strategy, 'build', no_build)
    out = tmp_path / 'never.zip' if archived else None
    with pytest.raises(sps.errors.SpecsolveError, match=says):
        sps.solve_over(WINDOW, horizon_sources(12), axis, archive=out, keep_windows=True)


def test_a_coordinate_sweep_archive_evaluates_an_undeclared_expression_per_slice(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A coordinate sweep's answer is its slices, so the archive alone puts each slice back against its model."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    runs = sps.solve_over(dispatch_yaml, sources, sps.EachCoordinate('scenario'), archive=tmp_path / 'study')
    for read in (sps.load_archive, sps.scan_archive):
        valued = read(tmp_path / 'study').sweep.evaluate('sum(p, over=generator)')
        by_hand = runs.primal('p').group_by('scenario', 'snapshot').agg(pl.col('value').sum())
        assert valued.sort('scenario', 'snapshot').equals(by_hand.sort('scenario', 'snapshot')), (
            f'{read.__name__}: each slice valued at its own solution, keyed by it'
        )


@pytest.mark.parametrize('table', ['record.parquet', METRICS_FILE], ids=str)
def test_a_directory_of_solves_and_sweeps_globs_into_one_table(
    table: str, dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """A warehouse tool reads a glob with the schema of one file, so every archive writes the same columns.

    A sweep once wrote its key under the axis's own name and a slice's metrics
    in other columns than a solve's, and a strict glob refused the mix.
    """
    runs = tmp_path / 'runs'
    sps.solve(dispatch_yaml, dispatch_frame_inputs, archive=runs / 'single').close()
    sweep_sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    sps.solve_over(dispatch_yaml, sweep_sources, sps.EachCoordinate('scenario'), archive=runs / 'scenarios')
    window = sps.EachWindow('snapshot', steps=4, lookahead=2, into='t')
    sps.solve_over(WINDOW, horizon_sources(12), window, carry={'soc_initial': 'soc'}, archive=runs / 'rolling')

    files = sorted(runs.glob(f'*/{ANSWER_DIR}/{table}'))
    schemas = {file.parts[-3]: pl.read_parquet_schema(file) for file in files}
    assert len({tuple(schema.items()) for schema in schemas.values()}) == 1, (
        f'one schema across a solve and two kinds of sweep: {schemas}'
    )
    globbed = pl.read_parquet(str(runs / '*' / ANSWER_DIR / table)).sort(RUN, 'slice')
    assert globbed.select(RUN, 'slice_axis', 'slice').rows() == [
        ('rolling', 'snapshot_start', '0'),
        ('rolling', 'snapshot_start', '4'),
        ('rolling', 'snapshot_start', '8'),
        ('scenarios', 'scenario', 'high'),
        ('scenarios', 'scenario', 'low'),
        ('single', None, None),
    ], 'a plain glob reads every row, the slice named as text and null for the single solve'


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

    with pytest.raises(sps.errors.SpecsolveError, match='came back from a different spec'):
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

    with pytest.raises(sps.errors.SpecsolveError, match='came back from a different spec'):
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

    assert json.loads((out / 'format.json').read_text()) == {
        'layout': ANSWER_LAYOUT,
        'specsolve': sps.__version__,
        'outputs': [],
    }, 'the layout this package writes, beside the version that wrote it and the outputs it carries, none by default'


@pytest.mark.parametrize('scale', [pytest.param(1.0, id='with-values'), pytest.param(100.0, id='infeasible')])
def test_a_record_names_the_solver_and_the_packages_that_produced_it(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, scale: float
) -> None:
    """An answer kept for months outlives the environment that solved it.

    Before, the record named no solver, no solver option and no mathspec, so an
    archive could not say what would have to be installed to solve it again.
    An answer with no values is read back on a path of its own, so it is a case.
    """
    inputs = {**dispatch_frame_inputs, 'load': dispatch_frame_inputs['load'].with_columns(pl.col('value') * scale)}
    with sps.solve(
        dispatch_yaml, inputs, solver_options={'time_limit': 60.0, 'presolve': 'on'}, archive=tmp_path / 'case'
    ) as solved:
        provenance = solved.provenance
        assert solved.has_primal == (scale == 1.0), 'the case is the one its id names'

    row = pl.read_parquet(tmp_path / 'case' / 'answer' / 'record.parquet').row(0, named=True)
    assert {name: row[name] for name in Provenance._fields} == {
        'solver': 'highs',
        'solver_version': version('highspy'),
        'solver_options': '{"presolve": "on", "time_limit": 60.0}',
        'specsolve_version': sps.__version__,
        'mathspec_version': version('mathspec'),
    }, 'the solver, its options as one JSON string with sorted keys, and the three installed versions'
    assert sps.load_archive(tmp_path / 'case').result.provenance == provenance, 'and it reads back as it was written'


def test_a_solve_with_no_options_records_an_empty_object() -> None:
    """No options is a fact about the solve, so it is written, not left null."""
    assert _provenance('highs', None).solver_options == '{}', 'null is kept for a record no solve wrote'


@pytest.mark.parametrize(
    ('value', 'decoded'),
    [
        pytest.param(math.inf, 'inf', id='inf'),
        pytest.param(-math.inf, '-inf', id='minus-inf'),
        pytest.param(math.nan, 'nan', id='nan'),
        pytest.param(np.int64(4), 4, id='numpy-int'),
        pytest.param(np.float64(0.5), 0.5, id='numpy-float'),
    ],
)
def test_the_recorded_options_are_json_every_reader_decodes(value: object, decoded: object) -> None:
    """A recorded option is read back by tools that take strict JSON only.

    Before, ``time_limit=math.inf``, which is HiGHS's own default, was written
    as ``Infinity``, which is not JSON, so polars' ``str.json_decode`` raised.
    A ``numpy.int64`` was written as the string ``"4"`` rather than the number.
    """
    written = _provenance('highs', {'time_limit': value}).solver_options
    assert pl.Series([written]).str.json_decode().to_list() == [{'time_limit': decoded}], (
        'a non-finite number is its name as a string, and a numpy scalar is the number it holds'
    )


@pytest.mark.parametrize(
    'option',
    [
        pytest.param('WLSAccessID', id='wls-access-id'),
        pytest.param('WLSSecret', id='wls-secret'),
        pytest.param('LicenseID', id='license-id'),
        pytest.param('CSAPIAccessID', id='compute-server-access-id'),
        pytest.param('CSAPISecret', id='compute-server-secret'),
        pytest.param('ServerPassword', id='server-password'),
        pytest.param('CloudAccessID', id='cloud-access-id'),
        pytest.param('CloudSecretKey', id='cloud-secret-key'),
        pytest.param('SomeParameterAFutureGurobiAdds', id='unknown-to-specsolve'),
    ],
)
def test_an_option_off_the_solvers_list_is_recorded_by_name_alone(option: str) -> None:
    """Gurobi takes its licence credentials as options, and an archive goes to shared storage.

    Only an option on the solver's list keeps its value, so a name nobody
    listed cannot leak. The name stays, so the record still says it was set.
    `TimeLimit` is on the list in another letter case, as Gurobi reads it.
    """
    pytest.importorskip('gurobipy')
    written = json.loads(_provenance('gurobi', {option: 'hunter2', 'TimeLimit': 60}).solver_options or '')
    assert written == {option: '<not recorded>', 'TimeLimit': 60}, (
        'the value is gone, the key and the listed option stay'
    )


def _known_to_highs(name: str) -> bool:
    import highspy

    status, _ = highspy.Highs().getOptionValue(name)
    return status == highspy.HighsStatus.kOk


def _known_to_gurobi(name: str) -> bool:
    import gurobipy

    return name in {parameter.casefold() for parameter in dir(gurobipy.GRB.Param)}


def _known_to_xpress(name: str) -> bool:
    import xpress

    try:
        xpress.problem().getControl(name)
    except (xpress.InterfaceError, xpress.ModelError):
        return False
    return True


@pytest.mark.parametrize(
    ('name', 'package', 'known'),
    [
        pytest.param('highs', 'highspy', _known_to_highs, id='highs'),
        pytest.param('gurobi', 'gurobipy', _known_to_gurobi, id='gurobi'),
        pytest.param('xpress', 'xpress', _known_to_xpress, id='xpress'),
    ],
)
def test_every_recorded_option_is_one_the_solver_knows(name: str, package: str, known) -> None:
    """A misspelt name on the list records nothing, and no test would see that otherwise."""
    pytest.importorskip(package)
    listed = SOLVERS[name].recorded_options
    assert [option for option in sorted(listed) if not known(option)] == [], (
        f'every option {name} records is one {name} takes'
    )


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_every_recorded_option_is_written_casefolded(name: str) -> None:
    """A caller's option is casefolded before the lookup, so a list entry with a capital never matches.

    Xpress takes a control in all upper case too, so the check against the solver passes `TIMELIMIT`.
    """
    listed = SOLVERS[name].recorded_options
    assert [option for option in sorted(listed) if option != option.casefold()] == [], (
        f'every option {name} records is written casefolded'
    )


def test_the_record_ends_with_the_provenance_columns() -> None:
    """`Record` repeats `Provenance`'s fields so that each is a column of its own."""
    assert Record._fields[-len(Provenance._fields) :] == Provenance._fields, (
        'Record spreads Provenance into its last columns, in the same order'
    )


def test_an_answer_in_another_layout_is_refused_by_name(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An answer in another layout names the specsolve that wrote it; one with no stamp is refused too."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        out = solved.save(tmp_path / 'solution')
    (out / 'format.json').write_text(json.dumps({'layout': 0, 'specsolve': '0.0.1a359'}))

    with pytest.raises(sps.errors.LayoutError, match='solve the model again and save it') as refused:
        sps.load_result(out)
    assert f'layout 0, written by specsolve 0.0.1a359, and this package reads layout {ANSWER_LAYOUT}' in str(
        refused.value
    ), 'the refusal names the layout it found and the version that wrote it'

    (out / 'format.json').write_text(json.dumps({'answer': 0}))  # what 0.1.0 wrote
    with pytest.raises(sps.errors.LayoutError, match='with no layout stamp'):
        sps.load_result(out)

    (out / 'format.json').unlink()
    with pytest.raises(sps.errors.LayoutError, match='with no layout stamp'):
        sps.load_result(out)


def _restamped(tree: Path, stamp: dict[str, object] | None, out: Path) -> Path:
    """*tree*, a directory archive, at *out* with *stamp* as the ``format.json`` its members carry, or none; packed where *out* is a zip."""
    if stamp is None:
        (tree / 'format.json').unlink()
    else:
        (tree / 'format.json').write_text(json.dumps(stamp))
    if out.suffix != '.zip':
        return tree.rename(out)
    with zipfile.ZipFile(out, 'w') as zipped:
        for member in sorted(tree.rglob('*')):
            if member.is_file():
                zipped.write(member, member.relative_to(tree).as_posix())
    return out


_IN_ANOTHER_LAYOUT = {'layout': 0, 'specsolve': '0.0.1a359'}


@pytest.mark.parametrize('suffix', [pytest.param('', id='directory'), pytest.param('.zip', id='zip')])
def test_load_inputs_reads_the_spec_and_the_data_of_an_archive_whose_answer_is_in_another_layout(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, suffix: str
) -> None:
    """The answer and the inputs are stamped apart, so an answer in an old layout does not lock the inputs away."""
    tree = _archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'case')
    (tree / ANSWER_DIR / 'format.json').write_text(json.dumps(_IN_ANOTHER_LAYOUT))
    out = _restamped(tree, json.loads((tree / 'format.json').read_text()), tmp_path / f'old{suffix}')

    with pytest.raises(sps.errors.LayoutError) as refused:
        sps.load_archive(out)
    assert f'sps.load_inputs({str(out)!r})' in str(refused.value), (
        'the refusal names the archive as the caller passed it, and the reader that still takes it'
    )

    inputs = sps.load_inputs(out)
    assert (inputs.axis, inputs.carry) == (None, {}), 'an archive of one solve holds no axis and no carry'
    with (
        sps.solve(inputs.spec, inputs.sources) as again,
        sps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
    ):
        assert again.objective == pytest.approx(direct.objective), 'the inputs read back solve the same model'


@pytest.mark.parametrize(
    'stamp', [pytest.param(_IN_ANOTHER_LAYOUT, id='another-layout'), pytest.param(None, id='no-stamp')]
)
@pytest.mark.parametrize('read', [sps.load_archive, sps.load_inputs], ids=['load_archive', 'load_inputs'])
def test_a_spec_and_sources_in_another_layout_are_refused_with_the_files_that_solve_them_again(
    read: Callable[[Path], object],
    stamp: dict[str, object] | None,
    dispatch_yaml: Path,
    dispatch_frame_inputs,
    tmp_path: Path,
) -> None:
    """The refusal names the files that hold the spec and the data, and those files solve to the same objective.

    An archive written before the spec and the sources had a stamp of their
    own carries none, and is refused the same way.
    """
    out = _restamped(_archived(dispatch_yaml, dispatch_frame_inputs, tmp_path / 'case'), stamp, tmp_path / 'old')

    with pytest.raises(sps.errors.LayoutError) as refused:
        read(out)
    says = str(refused.value)
    missing = [part for part in (repr(str(out)), 'spec.yaml', 'sources/<key>.parquet', 'to_spec') if part not in says]
    assert not missing, f'the refusal names the archive and how to read its spec and data, and is short of {missing}'

    sources = {member.stem: member for member in (out / 'sources').glob('*.parquet')}
    with (
        sps.solve(to_spec(out / 'spec.yaml'), sources) as again,
        sps.solve(dispatch_yaml, dispatch_frame_inputs) as direct,
    ):
        assert again.objective == pytest.approx(direct.objective), 'the files the refusal names solve the same model'


def test_load_inputs_gives_back_what_a_sweep_runs_again_with(tmp_path: Path) -> None:
    from tests.test_strategy import WINDOW, horizon_sources

    sps.solve_over(WINDOW, horizon_sources(12), ROLLING, carry={'soc_initial': 'soc'}, archive=tmp_path / 'roll')
    archived = sps.load_archive(tmp_path / 'roll')
    inputs = sps.load_inputs(tmp_path / 'roll')

    assert (inputs.axis, inputs.carry) == (archived.axis, archived.carry), 'the axis and carry the sweep ran with'
    unlike = [key for key in archived.sources if not inputs.sources[key].equals(archived.sources[key])]
    assert inputs.sources.keys() == archived.sources.keys(), 'one table per source the archive holds'
    assert not unlike, f'sources that differ from what load_archive gives back: {unlike}'


def test_a_hand_built_axis_is_refused(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A list of `(key, sources)` is refused, and sent to one archive per solve."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    with pytest.raises(sps.errors.SpecsolveError, match='archive one solve each'):
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
    with pytest.raises(sps.errors.SpecsolveError, match='was lowered from'):
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
    want = intact.result.evaluate(_UNDECLARED)['value'].sum()

    moved = dispatch_frame_inputs['cost'].with_columns(pl.col('value') * 99)
    moved.write_parquet(tmp_path / 'case' / 'sources' / 'cost.parquet')

    tampered = sps.load_archive(tmp_path / 'case')
    with pytest.raises(sps.errors.SpecsolveError, match='came back from another model') as refused:
        tampered.result.evaluate(_UNDECLARED)
    assert 'what differs is the data' in str(refused.value), 'and the refusal says which half moved'
    assert want == pytest.approx(intact.result.evaluate(_UNDECLARED)['value'].sum(), rel=1e-9), (
        'while the archive as written still reads'
    )


def test_an_archive_reads_back_an_undeclared_expression_however_its_rebuild_sums(tmp_path: Path) -> None:
    """``osemosys_utopia`` sums its costs over rows in no fixed order, so each build differs in the last bit.

    The check compared the built model to the last bit, so every archive of
    it refused an undeclared read as built from other data. It compares the
    data now, which a rebuild anywhere reads the same.
    """
    from tests.conftest import expanded, port_sources, port_spec

    spec = expanded(port_spec('osemosys_utopia'))
    sps.solve(spec, port_sources('osemosys_utopia'), archive=tmp_path / 'run.zip').close()
    variable = next(iter(sps.check(spec).variables))
    assert sps.load_archive(tmp_path / 'run.zip').result.evaluate(f'sum({variable})').height == 1


@pytest.mark.parametrize(
    ('change', 'same'),
    [
        pytest.param(lambda s: s | {'cost': s['cost'].reverse()}, True, id='a-table-in-another-row-order'),
        pytest.param(lambda s: s | {'generator': s['generator'].reverse()}, False, id='a-dimension-in-another-order'),
    ],
)
def test_the_data_digest_reads_a_tables_rows_in_any_order_but_a_dimensions(
    dispatch_yaml: Path, dispatch_frame_inputs, change: Any, same: bool
) -> None:
    """A dimension's row order is its coordinate order, so moving it is other data; moving a table's rows is not."""
    with (
        sps.build(dispatch_yaml, dispatch_frame_inputs) as plain,
        sps.build(dispatch_yaml, change(dispatch_frame_inputs)) as moved,
    ):
        assert (moved._model_digest() == plain._model_digest()) == same


def test_an_answer_naming_no_model_is_taken_as_given(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path
) -> None:
    """An answer written before the column has no digest to compare, and is not refused for it."""
    with sps.solve(dispatch_yaml, dispatch_frame_inputs) as solved:
        want = solved.evaluate(_UNDECLARED)['value'].sum()
        solved.save(tmp_path / 'answer')

    older = replace(sps.load_result(tmp_path / 'answer'), _model_digest=None)
    read = _attach_readers(older, dispatch_yaml, dispatch_frame_inputs)
    assert read.evaluate(_UNDECLARED)['value'].sum() == pytest.approx(want, rel=1e-9), (
        'an answer carrying no model digest reads against the data it is handed'
    )


@pytest.mark.parametrize(
    ('record_options', 'written'),
    [
        pytest.param(None, '<not recorded>', id='not-named'),
        pytest.param(['mip_max_nodes'], 1000, id='named'),
        pytest.param(['MIP_MAX_NODES'], 1000, id='named-in-another-letter-case'),
    ],
)
def test_a_caller_names_an_option_to_record_beside_the_solvers_list(
    dispatch_yaml: Path, dispatch_frame_inputs, tmp_path: Path, record_options: list[str] | None, written: object
) -> None:
    """`mip_max_nodes` is on no list, so only the caller's naming it keeps its value."""
    sps.solve(
        dispatch_yaml,
        dispatch_frame_inputs,
        solver_options={'time_limit': 60.0, 'mip_max_nodes': 1000},
        record_options=record_options,
        archive=tmp_path / 'case',
    ).close()

    row = pl.read_parquet(tmp_path / 'case' / 'answer' / 'record.parquet').row(0, named=True)
    assert json.loads(row['solver_options']) == {'mip_max_nodes': written, 'time_limit': 60.0}, (
        'a listed option keeps its value either way; the named one keeps it only when named'
    )


def test_every_slice_of_a_sweep_records_the_options_its_caller_named(
    dispatch_yaml: Path, dispatch_frame_inputs
) -> None:
    """The sweep forwards the names to each slice's solve, as it forwards the options."""
    sources = {**dispatch_frame_inputs, 'load': _by_scenario(['low', 'high'])}
    sweep = sps.solve_over(
        dispatch_yaml,
        sources,
        sps.EachCoordinate('scenario'),
        solver_options={'mip_max_nodes': 1000},
        record_options=['mip_max_nodes'],
    )

    assert sweep.record['solver_options'].to_list() == ['{"mip_max_nodes": 1000}'] * 2, (
        'both slices keep the value the caller named'
    )


def test_a_bare_string_of_options_to_record_is_refused(dispatch_yaml: Path, dispatch_frame_inputs) -> None:
    """A string is a sequence of letters, so `record_options='Seed'` would name `S`, `e` and `d`."""
    with pytest.raises(sps.errors.SpecsolveError, match=r"record_options=\['mip_max_nodes'\]"):
        sps.solve(dispatch_yaml, dispatch_frame_inputs, record_options='mip_max_nodes')


def test_the_data_digest_tells_labels_apart_where_their_characters_run_together() -> None:
    joined_alike = [pl.LazyFrame({'name': ['ab', 'c']}), pl.LazyFrame({'name': ['a', 'bc']})]
    digests = {digest_of_data('spec', {'name': table}, ordered={'name'}) for table in joined_alike}
    assert len(digests) == 2, "'ab', 'c' and 'a', 'bc' are other labels, though their characters read alike end to end"
