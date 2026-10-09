"""The archive's layout: a spec, its data and its answer as a directory, or that directory zipped.

``spec.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``catalog.parquet`` saying what each file holds, ``answer/`` in the answer's own layout
([`specsolve.relational.answer_layout`][]), ``axis.json`` where the
sources are cut, and ``format.json`` stamping [`INPUTS_LAYOUT`][]. A
directory archive is read where it lies; a zip is unpacked first. Reading
one back is [`specsolve.archive`][].
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

from specsolve.errors import LayoutError
from specsolve.inputs import lowered
from specsolve.relational.answer_layout import FORMAT_FILE, OUTPUT_KINDS, write_format, write_whole
from specsolve.relational.names import RESERVED_PREFIX, RUN, VALUE
from specsolve.sweep import MANIFEST_FILE, OWNED_FILE, WINDOWS_DIR

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    from mathspec import Spec
    from mathspec.program import Program

    from specsolve.relational.answer_layout import FrameWriter

#: The archive's one layout. ``axis.json`` also marks a sweep archive.
SPEC_MEMBER = 'spec.yaml'
AXIS_MEMBER = 'axis.json'
CATALOG_MEMBER = 'catalog.parquet'
SOURCES_DIR = 'sources'
ANSWER_DIR = 'answer'

#: The layout of the spec and the data an archive holds: ``spec.yaml``,
#: ``sources/`` and ``axis.json``. A change to any of them raises it; any other
#: change to what an archive writes, ``catalog.parquet`` included, raises
#: [`ANSWER_LAYOUT`][specsolve.relational.answer_layout.ANSWER_LAYOUT] instead.
#: Stamped in the archive's own ``format.json``.
INPUTS_LAYOUT = 1


@contextmanager
def beside(out: Path) -> Iterator[Path]:
    """A scratch directory beside *out*, gone when the block ends, where an answer is laid out before it is packed.

    It is one level down in a directory of its own, as a staged archive is in
    ``_staging_for``'s: a directory of archives is read as
    ``<parent>/*/<member>`` while one of them is written, and a scratch
    ``answer/`` directly under the parent matches that glob.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=out.name + '.') as holder:
        scratch = Path(holder) / 'scratch'
        scratch.mkdir()
        yield scratch


def check_the_target(out: Path) -> None:
    """Refuse a directory target that already holds something, before anything is solved."""
    if out.suffix != '.zip' and out.is_dir() and any(out.iterdir()):
        raise LayoutError(
            f'{str(out)!r} already holds something, and a directory archive is written whole rather than '
            f'merged into what is there. Name a directory that does not exist, or delete this one. A .zip '
            f'target is replaced instead.'
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
    tables: Mapping[str, pl.LazyFrame],
    *,
    axis: Mapping[str, object] | None,
    answer: Callable[[Path, FrameWriter], None],
) -> Path:
    """Write a spec, its data and its answer to *out*: a directory, or one zip where the suffix is ``.zip``.

    Args:
        out: Where to write; its parent is made if it does not exist. Its
            name without ``.zip`` is the ``specsolve_run`` every table carries.
        spec: The spec as written, held as ``spec.yaml``.
        tables: The tidy table each source stands for, keyed as the file
            declares, each written as ``sources/<key>.parquet``.
        axis: The axis manifest, or ``None`` where the sources are not cut.
        answer: Lays the answer out in the directory it is given, writing
            each table with the writer it is given, which adds the run.

    Returns:
        *out*, which lands whole or not at all.
    """
    zipped = out.suffix == '.zip'
    run = out.name.removesuffix('.zip')

    def write(frame: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
        write_whole(_with_run(frame.lazy(), run), path)

    staging = _staging_for(out)
    part = staging / out.name
    tree = staging / 'tree' if zipped else part
    try:
        (tree / SOURCES_DIR).mkdir(parents=True)
        write_format(tree, layout=INPUTS_LAYOUT)
        (tree / SPEC_MEMBER).write_bytes(spec.to_yaml().encode())
        for name, table in tables.items():
            write(table, tree / SOURCES_DIR / f'{name}.parquet')
        if axis is not None:
            (tree / AXIS_MEMBER).write_text(json.dumps(axis))
        answer(tree / ANSWER_DIR, write)
        _write_catalogs(lowered(spec), tree, run, axis)
        if zipped:
            _pack(tree, part)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    if not zipped and out.is_dir():
        out.rmdir()
    part.replace(out)
    shutil.rmtree(staging)
    return out


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


#: Where an archive keeps an EachWindow sweep's per-window frames, and their own catalog.
_WINDOWS = f'{ANSWER_DIR}/{WINDOWS_DIR}'


def _answered_under(under: str) -> dict[str, list[str]]:
    """The directories under *under* that hold an answer, per kind of declaration, each output beside its core kind."""
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


#: The directories under ``answer/windows/`` that hold one directory of windows per name of each kind.
_HELD_IN_WINDOWS = MappingProxyType(_answered_under(_WINDOWS))


#: What ``answer/windows/owned.parquet`` holds, as its catalog rows describe it.
_OWNED = 'the coordinate of the sliced dimension that each window owns, by window and index inside the window'


def _write_catalogs(program: Program, tree: Path, run: str, axis: Mapping[str, object] | None) -> None:
    """``catalog.parquet`` for what *tree* holds, and ``answer/windows/catalog.parquet`` where the windows were kept."""
    dims = _axis_dims(tree, axis)
    _catalog(program, tree, run, _HELD_UNDER, dims).write_parquet(tree / CATALOG_MEMBER)
    if (tree / _WINDOWS).is_dir():
        owned = f'{_WINDOWS}/{OWNED_FILE}'
        windows = _catalog(program, tree, run, _HELD_IN_WINDOWS, dims, extra=[(owned, 'owned', 'window', _OWNED)])
        windows.write_parquet(tree / _WINDOWS / CATALOG_MEMBER)


def _axis_dims(tree: Path, axis: Mapping[str, object] | None) -> dict[str, str]:
    """The dimension of each column a sweep adds to its files: the axis column, and the key naming each slice.

    A window's answer and cut sources hold the axis column where the spec
    declares the local index. Empty for a single solve.
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
    return list(pl.read_parquet_schema(min(held.glob('*.parquet')) if held.is_dir() else held))


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
        for name, declaration in declared.items():
            yield name, kind, declaration.description, None, [(d, d) for d in declaration.dims]


def _with_run(frame: pl.LazyFrame, run: str) -> pl.LazyFrame:
    """*frame* as an archive holds it, in types parquet readers agree on, with [`RUN`][] set to *run*.

    An unsigned integer up to ``UInt32`` becomes ``Int64``; ``UInt64`` stays, as ``Int64`` cannot hold it. A
    timestamp in a time zone becomes the same instant in UTC.
    """
    return frame.with_columns(
        cs.by_dtype(pl.UInt8, pl.UInt16, pl.UInt32).cast(pl.Int64),
        cs.datetime(time_zone='*').dt.convert_time_zone('UTC'),
        pl.lit(run, dtype=pl.String).alias(RUN),
    )


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
    held = Path(path)
    if held.is_dir():
        if into is not None:
            raise LayoutError(
                f'{str(held)!r} is a directory archive, so it is read where it lies and into= has nothing to '
                f'do. Drop into=; it is for unpacking a .zip.'
            )
        _check_the_layout(held, (file.relative_to(held).as_posix() for file in held.rglob('*') if file.is_file()))
        return held
    if into is None:
        raise LayoutError(
            f'{str(held)!r} is one file, so reading it needs somewhere to unpack: pass into=. A directory '
            f'archive is read where it lies and needs none.'
        )
    with zipfile.ZipFile(held) as archive:
        _check_the_layout(held, (name for name in archive.namelist() if not name.endswith('/')))
        archive.extractall(into)
    return Path(into)


def _check_the_layout(named: Path, members: Iterable[str]) -> None:
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
