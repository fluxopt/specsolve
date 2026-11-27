"""The archive's layout: a spec, its data and its answer as a directory, or that directory zipped.

``spec.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``catalog.parquet`` saying what each file holds, ``answer/`` in the answer's own layout
([`specsolve._answer.layout`][]), ``axis.json`` where the
sources are cut, and ``format.json`` stamping [`INPUTS_LAYOUT`][]. A
directory archive is read where it lies; a zip is unpacked first. Reading
one back is [`specsolve._archive`][].
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs

from specsolve import _parquet as parquet
from specsolve._answer.layout import (
    FORMAT_FILE,
    HORIZON_FILE,
    MANIFEST_FILE,
    METRICS_FILE,
    other_layout,
    write_format,
    write_whole,
)
from specsolve._answer.outputs import OUTPUT_KINDS
from specsolve._collect import collected
from specsolve._inputs import as_program
from specsolve._names import RESERVED_PREFIX, RUN, VALUE
from specsolve.errors import LayoutError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence

    from mathspec.program import Program
    from mathspec.spec import Spec

    from specsolve._answer.layout import TableWriter

#: The archive's one layout. ``axis.json`` also marks a sweep archive.
SPEC_MEMBER = 'spec.yaml'
AXIS_MEMBER = 'axis.json'
CATALOG_MEMBER = 'catalog.parquet'
SOURCES_DIR = 'sources'
ANSWER_DIR = 'answer'

#: The layout of the spec and the data an archive holds: ``spec.yaml``,
#: ``sources/`` and ``axis.json``. A change to any of them raises it; any other
#: change to what an archive writes, ``catalog.parquet`` included, raises
#: [`ANSWER_LAYOUT`][specsolve._answer.layout.ANSWER_LAYOUT] instead.
#: Stamped in the archive's own ``format.json``.
INPUTS_LAYOUT = 3


def check_the_target(out: Path, *, resumable: bool = False) -> None:
    """Refuse a directory target that already holds something, before anything is solved.

    Where *resumable*, a directory holding a sweep's ``axis.json`` is taken,
    for the sweep to check its answer is its own and resume it, unless its
    spec and sources are in another layout.
    """
    if out.suffix == '.zip' or not out.is_dir() or not any(out.iterdir()):
        return
    if resumable and (out / AXIS_MEMBER).is_file():
        if (other := other_layout(out, INPUTS_LAYOUT)) is not None:
            raise LayoutError(
                f'{str(out)!r} holds a sweep archive whose spec and sources are {other}, and this package '
                f'writes layout {INPUTS_LAYOUT}, so it does not resume it. Name a directory that does not '
                f'exist, or delete this one.'
            )
        return
    what = 'something other than a sweep archive to resume' if resumable else 'something'
    raise LayoutError(
        f'{str(out)!r} already holds {what}, and a directory archive is written into an empty directory '
        f'rather than merged into what is there. Name a directory that does not exist, or delete this one. '
        f'A .zip target is replaced instead.'
    )


def _staging_for(out: Path) -> Path:
    """A staging directory of this writer's own, beside *out*.

    One per writer, so two writers to one path do not clear each other's
    members; beside *out*, so landing it is a rename. The archive is staged
    one level down in it, so no ``<parent>/*/<member>`` glob reads it early.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=out.parent, prefix=out.name + '.'))


def write_archive(
    out: Path,
    spec: Spec,
    tables: Mapping[str, pl.DataFrame | pl.LazyFrame],
    *,
    answer: Callable[[Path, TableWriter], object],
    metrics: Callable[[], pl.DataFrame],
) -> Path:
    """Write a spec, its data and one solve's answer to *out*: a directory, or one zip where the suffix is ``.zip``.

    Args:
        out: Where to write; its parent is made if it does not exist. Its
            name without ``.zip`` is the ``specsolve_run`` every table carries.
        spec: The spec as written, held as ``spec.yaml``.
        tables: The tidy table each source stands for, keyed as the file
            declares, each written as ``sources/<key>.parquet``.
        answer: Lays the answer out in the directory it is given, writing
            each table with the writer it is given, which adds the run.
        metrics: The row ``answer/metrics.parquet`` holds, called once every
            other member is written and packed, so a clock it reads covers
            them.

    Returns:
        *out*, which lands whole or not at all.
    """
    zipped = out.suffix == '.zip'
    run = out.name.removesuffix('.zip')
    write = _writer_for(run)
    staging = _staging_for(out)
    part = staging / out.name
    tree = staging / 'tree' if zipped else part
    try:
        _write_inputs(tree, spec, tables, None, write)
        answer(tree / ANSWER_DIR, write)
        _write_catalogs(as_program(spec), tree, run, None)
        if zipped:
            _pack(tree, part)
            shutil.rmtree(tree)
        _write_last(metrics(), run, tree, part if zipped else None)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not zipped and out.is_dir():
        out.rmdir()
    part.replace(out)
    shutil.rmtree(staging)
    return out


@contextmanager
def sweep_archive(
    out: Path, spec: Spec, tables: Mapping[str, pl.DataFrame | pl.LazyFrame], axis: Mapping[str, object]
) -> Generator[tuple[Path, TableWriter], None, None]:
    """A sweep's archive at *out*, open for its answer to be written into slice by slice.

    Yields the directory the answer goes in and the writer that adds the run.
    A directory archive is written where it lies: its spec, sources and axis
    first, ``axis.json`` last of them, so a resume holds the data of the run
    that finished it.
    ``catalog.parquet`` lands once the body ends without raising, so an
    interrupted sweep leaves a directory that reads back the slices it
    finished. A zip is staged beside *out* and packed at the end; an
    interrupted one leaves nothing, and is not resumed.

    *tables* and *axis* are as [`write_archive`][] takes them, *axis* as its
    ``to_dict`` writes it.
    """
    zipped = out.suffix == '.zip'
    run = out.name.removesuffix('.zip')
    write = _writer_for(run)
    staging = _staging_for(out) if zipped else None
    tree = out if staging is None else staging / 'tree'
    try:
        if staging is None and (tree / AXIS_MEMBER).is_file():
            _refuse_other_inputs(tree, spec, tables, axis)
        else:
            _write_inputs(tree, spec, tables, axis, write)
        yield tree / ANSWER_DIR, write
        _write_catalogs(as_program(spec), tree, run, axis)
        if staging is not None:
            _pack(tree, staging / out.name)
            (staging / out.name).replace(out)
    finally:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)


def _writer_for(run: str) -> TableWriter:
    """The writer that lands a table whole, as an archive named *run* holds it."""

    def write(table: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
        write_whole(_with_run(table.lazy(), run), path)

    return write


def _write_inputs(
    tree: Path,
    spec: Spec,
    tables: Mapping[str, pl.DataFrame | pl.LazyFrame],
    axis: Mapping[str, object] | None,
    write: TableWriter,
) -> None:
    """The stamp, ``spec.yaml``, ``sources/`` and, for a sweep, ``axis.json``, into *tree*."""
    (tree / SOURCES_DIR).mkdir(parents=True, exist_ok=True)
    write_format(tree, layout=INPUTS_LAYOUT)
    (tree / SPEC_MEMBER).write_bytes(spec.to_yaml().encode())
    for name, source in tables.items():
        write(source, tree / SOURCES_DIR / f'{name}.parquet')
    if axis is not None:
        (tree / AXIS_MEMBER).write_text(json.dumps(axis))


def carried_sources(
    out: Path, tables: Mapping[str, pl.DataFrame | pl.LazyFrame]
) -> dict[str, Path] | dict[str, pl.DataFrame | pl.LazyFrame]:
    """The sources an answer archived at *out* carries: the path to each member of a directory, *tables* for a zip.

    A zip has no member a path reaches, so its answer holds the tables it was
    written from, which are in memory already; a directory's holds none of them.
    """
    if out.suffix == '.zip':
        return dict(tables)
    return {name: out / SOURCES_DIR / f'{name}.parquet' for name in tables}


def _refuse_other_inputs(
    tree: Path, spec: Spec, tables: Mapping[str, pl.DataFrame | pl.LazyFrame], axis: Mapping[str, object]
) -> None:
    """Refuse a resume of the sweep archived at *tree* from another spec, axis or sources than it holds.

    The slices on disk were solved from what *tree* holds, so a resume either
    matches it and writes none of it again, or would describe answers it did
    not produce. ``sweep.json`` names the keys and not the carry or the
    lookahead, so the axis is compared here as ``axis.json`` holds it. A
    source is compared as the table it is written as, without its run, rather
    than hashed: encoding it again costs what writing it did. Called before
    anything is written, so a refused directory is left as it was found.

    Raises:
        SpecsolveError: A spec, an axis or a source that differs, each named.
    """
    differ = [SPEC_MEMBER] if (tree / SPEC_MEMBER).read_text() != spec.to_yaml() else []
    held_axis, asked_axis = json.loads((tree / AXIS_MEMBER).read_text()), json.loads(json.dumps(axis))
    if held_axis != asked_axis:
        differ.append(f'{AXIS_MEMBER} (holding {held_axis}, and this call asks for {asked_axis})')
    held = {member.stem: member for member in (tree / SOURCES_DIR).glob('*.parquet')}
    differ += [
        f'{SOURCES_DIR}/{name}.parquet'
        for name in sorted(held.keys() | tables.keys())
        if name not in held
        or name not in tables
        or not parquet.read(held[name]).drop(RUN, strict=False).equals(tables[name].lazy().pipe(collected))
    ]
    if differ:
        raise SpecsolveError(
            f'{str(tree)!r} holds this sweep solved from other inputs than this call passes: '
            f'{", ".join(differ)} differ, and the slices it holds were solved from what it holds. Archive to a '
            f'new directory, or delete this one to solve it again.'
        )


#: ``catalog.parquet``'s columns, in order.
_CATALOG_SCHEMA = {
    RUN: pl.String,
    'path': pl.String,
    'name': pl.String,
    'kind': pl.String,
    'description': pl.String,
    'dtype': pl.String,
    'column': pl.String,
    'dim': pl.String,
}


def _answered_under(under: str) -> dict[str, list[str]]:
    """The directories under *under* that hold an answer, per kind of entry, each output beside its core kind."""
    held = {'variable': [f'{under}/primal'], 'constraint': [f'{under}/dual'], 'expression': [f'{under}/expression']}
    for kind, carried in OUTPUT_KINDS.items():
        held[carried.per].append(f'{under}/{kind}')
    return held


#: The directories that hold one file per name of each kind.
_HELD_UNDER = MappingProxyType(
    {
        'dimension': [SOURCES_DIR],
        'relation': [SOURCES_DIR],
        'parameter': [SOURCES_DIR],
        **_answered_under(ANSWER_DIR),
    }
)


#: What ``answer/horizon.parquet`` holds, as its catalog row describes it.
_HORIZON = "the labels of the windowed dimension in each window's horizon, by window"


def _write_catalogs(program: Program, tree: Path, run: str, axis: Mapping[str, object] | None) -> None:
    """``catalog.parquet`` for what *tree* holds, with a row for ``answer/horizon.parquet`` where a sweep cut windows."""
    horizon = f'{ANSWER_DIR}/{HORIZON_FILE}'
    extra = [(horizon, 'horizon', 'window', _HORIZON)] if (tree / horizon).is_file() else []
    write_whole(_catalog(program, tree, run, _HELD_UNDER, _axis_dims(tree, axis), extra), tree / CATALOG_MEMBER)


def _axis_dims(tree: Path, axis: Mapping[str, object] | None) -> dict[str, str]:
    """The dimension of each column a sweep adds to its files: the axis column, and the key naming each slice.

    Empty for a single solve.
    """
    if axis is None:
        return {}
    dim = str(axis['dim'])
    key = json.loads((tree / ANSWER_DIR / MANIFEST_FILE).read_text())['key_name']
    return {dim: dim, key: dim}


def _catalog(
    program: Program,
    tree: Path,
    run: str,
    held_under: Mapping[str, Sequence[str]],
    axis_dims: Mapping[str, str],
    extra: Sequence[tuple[str, str, str, str]] = (),
) -> pl.DataFrame:
    """What each file of *tree* under *held_under* holds, one row per column of labels as the file is written.

    ``path`` and ``column`` are the key, because a constraint may share its
    name with a parameter. A column's dimension is the one the spec declares
    for it, else the one *axis_dims* gives, else the dimension of its own
    name. A name with no file has no row. *extra* is ``(path, name, kind,
    description)`` for a file the spec declares nothing for.
    """
    named = {name: name for name in program.dimensions}
    rows: list[tuple[object, ...]] = []
    for name, kind, description, dtype, columns in _declared_files(program):
        for directory in held_under.get(kind, []):
            if (path := _held(tree, directory, name)) is not None:
                head = (run, path, name, kind, description, dtype)
                rows.extend(_rows(head, tree / path, {**named, **axis_dims, **dict(columns)}))
    for path, name, kind, description in extra:
        rows.extend(_rows((run, path, name, kind, description, None), tree / path, {**named, **axis_dims}))
    return pl.DataFrame(rows, schema=_CATALOG_SCHEMA, orient='row').sort('path', 'column')


def _rows(head: tuple[object, ...], held: Path, dims: Mapping[str, str]) -> list[tuple[object, ...]]:
    """*head* once per column of labels *held* is written with, and once with no column where it has none."""
    labels = [column for column in _columns(held) if column != VALUE and not column.startswith(RESERVED_PREFIX)]
    return [(*head, column, dims.get(column)) for column in labels] or [(*head, None, None)]


def _columns(held: Path) -> list[str]:
    """The columns of *held*, or of its first slice where it is a directory of them."""
    return parquet.columns(min(held.glob('*.parquet')) if held.is_dir() else held)


def _held(tree: Path, directory: str, name: str) -> str | None:
    """*name*'s path under *directory* of *tree*: one file, a sweep's directory of slices, or ``None``."""
    for held in (f'{name}.parquet', name):
        if (tree / directory / held).exists():
            return f'{directory}/{held}'
    return None


def _declared_files(
    program: Program,
) -> Iterator[tuple[str, str, str | None, str | None, Sequence[tuple[str, str]]]]:
    """``(name, kind, description, dtype, (column, dim) pairs)`` per name, in the order the spec declares them."""
    for name, dimension in program.dimensions.items():
        yield name, 'dimension', dimension.description, dimension.dtype, [(name, name)]
    for name, relation in program.relations.items():
        yield name, 'relation', relation.description, None, relation.columns
    for name, parameter in program.parameters.items():
        yield name, 'parameter', parameter.description, parameter.dtype, [(d, d) for d in parameter.dims]
    answered = {'variable': program.variables, 'constraint': program.constraints, 'expression': program.expressions}
    for kind, declared in answered.items():
        for name, entry in declared.items():
            yield name, kind, entry.description, None, [(d, d) for d in entry.dims]


def _with_run(table: pl.LazyFrame, run: str) -> pl.LazyFrame:
    """*table* as an archive holds it, in types parquet readers agree on, with [`RUN`][] set to *run*.

    An unsigned integer up to ``UInt32`` becomes ``Int64``; ``UInt64`` stays, as ``Int64`` cannot hold it. A
    timestamp in a time zone becomes the same instant in UTC.
    """
    return table.with_columns(
        cs.by_dtype(pl.UInt8, pl.UInt16, pl.UInt32).cast(pl.Int64),
        cs.datetime(time_zone='*').dt.convert_time_zone('UTC'),
        pl.lit(run, dtype=pl.String).alias(RUN),
    )


def _write_last(row: pl.DataFrame, run: str, tree: Path, packed: Path | None) -> None:
    """*row* as ``answer/metrics.parquet``, into *tree*, or appended to the zip *packed* where there is one.

    Appended rather than packed with the rest, since what packing costs is
    one of the clocks the row records.
    """
    member = f'{ANSWER_DIR}/{METRICS_FILE}'
    written = tree / member
    written.parent.mkdir(parents=True, exist_ok=True)
    write_whole(_with_run(row.lazy(), run), written)
    if packed is not None:
        with zipfile.ZipFile(packed, 'a', compression=zipfile.ZIP_STORED) as appended:
            appended.write(written, member)


def _pack(tree: Path, into: Path) -> None:
    """*tree* as one zip at *into*, its members stored rather than compressed, parquet being compressed already."""
    with zipfile.ZipFile(into, 'w', compression=zipfile.ZIP_STORED) as packed:
        for file in sorted(tree.rglob('*')):
            if file.is_file():
                packed.write(file, file.relative_to(tree).as_posix())


def opened(path: str | Path, into: str | Path | None) -> Path:
    """Where an archive's members are on disk, unpacking it first if it is a zip.

    Args:
        path: The archive, a ``.zip`` or a directory.
        into: Where to unpack a zip, made if it does not exist. Refused for a
            directory archive, which is read where it lies.

    Returns:
        *path* for a directory archive, *into* for a zip.

    Raises:
        LayoutError: A member outside the layout, no ``spec.yaml``, a zip
            with no *into*, or an *into* given for a directory. Nothing is
            unpacked.
        zipfile.BadZipFile: A file that is not a zip archive.
    """
    parquet.require('reading an archive')
    held = Path(path)
    if held.is_dir():
        if into is not None:
            raise LayoutError(
                f'{str(held)!r} is a directory archive, so it is read where it lies and into= has nothing to '
                f'do. Drop into=; it is for unpacking a .zip.'
            )
        _refuse_stray_or_missing_members(
            held, (file.relative_to(held).as_posix() for file in held.rglob('*') if file.is_file())
        )
        return held
    if into is None:
        raise LayoutError(
            f'{str(held)!r} is one file, so reading it needs somewhere to unpack: pass into=. A directory '
            f'archive is read where it lies and needs none.'
        )
    with zipfile.ZipFile(held) as archive:
        _refuse_stray_or_missing_members(held, (name for name in archive.namelist() if not name.endswith('/')))
        archive.extractall(into)
    return Path(into)


def _refuse_stray_or_missing_members(named: Path, members: Iterable[str]) -> None:
    """Refuse anything that is not this layout, before a byte is unpacked."""
    found = sorted(members)
    strays = [
        member
        for member in found
        if member not in {SPEC_MEMBER, AXIS_MEMBER, CATALOG_MEMBER, FORMAT_FILE}
        and not member.startswith(f'{ANSWER_DIR}/')
        and not (member.startswith(f'{SOURCES_DIR}/') and member.endswith('.parquet') and member.count('/') == 1)
    ]
    if strays or SPEC_MEMBER not in found:
        what = f'holds {strays}' if strays else f'has no {SPEC_MEMBER!r}'
        raise LayoutError(
            f'{named} is not an archive: it {what}. One that archive= writes holds exactly '
            f"'spec.yaml', one 'sources/<key>.parquet' per key the file declares, "
            f"'catalog.parquet' saying what each file holds, 'answer/' holding what the solve returned, "
            f"'axis.json' where its sources are sliced, and 'format.json' stamping its layout."
        )
