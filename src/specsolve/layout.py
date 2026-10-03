"""The archive's layout: a spec, its data and its answer as a directory, or that directory zipped.

``spec.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``sources.parquet`` digesting them, ``catalog.parquet`` saying what each
file holds, ``answer/`` in the layout both answers save, and ``axis.json``
where the sources are cut. A directory archive is read
where it lies; a zip is unpacked first.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from specsolve.errors import LayoutError
from specsolve.lanes import lowered
from specsolve.relational.parquet import METRICS_FILE, RECORD_FILE, RUN, consolidated, digest_of_file, write_whole

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from mathspec import Spec
    from mathspec.program import Program

    from specsolve.lanes import Source

#: The archive's one layout. ``axis.json`` also marks a sweep archive.
SPEC_MEMBER = 'spec.yaml'
AXIS_MEMBER = 'axis.json'
DIGESTS_MEMBER = 'sources.parquet'
CATALOG_MEMBER = 'catalog.parquet'
SOURCES_DIR = 'sources'
ANSWER_DIR = 'answer'


@contextmanager
def beside(out: Path) -> Iterator[Path]:
    """A scratch directory beside *out*, gone when the block ends, where an answer is laid out before it is packed."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        yield Path(scratch)


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
    members; beside *out*, so landing it is a rename.
    """
    out.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(dir=out.parent, prefix=out.name + '.'))


def write_archive(
    out: Path,
    spec: Spec,
    sources: Mapping[str, Source],
    *,
    tables: Mapping[str, pl.LazyFrame],
    axis: Mapping[str, object] | None,
    answer: Path,
) -> Path:
    """Write a spec, its data and its answer to *out*: a directory, or one zip where the suffix is ``.zip``.

    Args:
        out: Where to write; its parent is made if it does not exist. Its
            name without ``.zip`` is the run every table is stamped with.
        spec: The spec as written, held as ``spec.yaml``.
        sources: What was attached, keyed as the file declares. A parquet path
            is copied, less any ``specsolve_run`` it carries; anything else is
            written as *tables* has it. The digest is of those bytes, before
            the run is stamped on.
        tables: The tidy table each source stands for, for every source that
            is not a path.
        axis: The axis manifest, or ``None`` where the sources are not cut.
        answer: A directory holding the answer's own layout. Its record and
            metrics land as one file each.

    Returns:
        *out*, which lands whole or not at all. Every table in it carries
        ``specsolve_run``.
    """
    zipped = out.suffix == '.zip'
    run = out.name.removesuffix('.zip')
    staging = _staging_for(out)
    part = staging / out.name
    tree = staging / 'tree' if zipped else part
    try:
        (tree / SOURCES_DIR).mkdir(parents=True)
        (tree / SPEC_MEMBER).write_bytes(spec.to_yaml().encode())
        digests: dict[str, str] = {}
        for name, given in sources.items():
            member = tree / SOURCES_DIR / f'{name}.parquet'
            if isinstance(given, (str, Path)):
                _copied_unstamped(Path(given), member)
            else:
                tables[name].collect().write_parquet(member, compression='zstd')
            digests[name] = digest_of_file(member)
            _stamped(member, member, run)
        _digest_table(digests, run).write_parquet(tree / DIGESTS_MEMBER)
        if axis is not None:
            (tree / AXIS_MEMBER).write_text(json.dumps(axis))
        _copy_the_answer(answer, tree / ANSWER_DIR, run)
        _catalog(lowered(spec), tree, run).write_parquet(tree / CATALOG_MEMBER)
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


def _copied_unstamped(source: Path, target: Path) -> None:
    """*source* at *target* without a ``specsolve_run`` column, before the member is digested.

    A source [`scan_archive`][specsolve.scan_archive] gave back is the
    member itself, stamped with the run it came from; digested as it lies, the
    old run's name would move every digest. A file with no such column is
    copied byte for byte, so the digest stays that of the caller's own bytes.
    """
    if RUN not in pl.read_parquet_schema(source):
        shutil.copyfile(source, target)
        return
    pl.read_parquet(source).drop(RUN).write_parquet(target, compression='zstd')


def _digest_table(digests: Mapping[str, str], run: str) -> pl.DataFrame:
    """``(specsolve_run, source, digest)`` in source order, so one model's data digests to one table."""
    names = sorted(digests)
    return pl.DataFrame(
        {RUN: [run] * len(names), 'source': names, 'digest': [digests[name] for name in names]},
        schema={RUN: pl.String, 'source': pl.String, 'digest': pl.String},
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
    'dim_position': pl.Int32,
}


#: The directories that hold one file per name of each kind.
_HELD_UNDER = {
    'dimension': [SOURCES_DIR],
    'relation': [SOURCES_DIR],
    'parameter': [SOURCES_DIR],
    'variable': [f'{ANSWER_DIR}/primal'],
    'constraint': [f'{ANSWER_DIR}/dual', f'{ANSWER_DIR}/activity'],
    'expression': [f'{ANSWER_DIR}/expression'],
}


def _catalog(program: Program, tree: Path, run: str) -> pl.DataFrame:
    """What each file under ``sources/`` and ``answer/<kind>/`` of *tree* holds, one row per column of labels.

    ``path`` is the file's path inside the archive, or in a sweep archive the
    directory of an answer's slices. With ``dim_position`` it is the key,
    because a constraint may share its name with a parameter. A name the spec
    declares and *tree* holds no file for has no row. ``kind`` is
    ``dimension``, ``relation``, ``parameter``, ``variable``, ``constraint``
    or ``expression``. ``column`` is the column of the file that holds
    ``dim``'s labels, and ``dim_position`` its 0-based place among them: a
    relation's column is its role, any other name's is the dimension itself.
    A file over no dimension has one row, with all three null. ``dtype`` is
    what the spec declares: the labels' type for a dimension, the ``value``
    column's for a parameter, null for the rest. ``specsolve_run``, which
    every file carries, holds no labels and has no row. Long form, with no
    list column, so a BI tool reads it as it reads any other table.
    """
    rows: list[tuple[object, ...]] = []
    for name, kind, description, dtype, columns in _declared_files(program):
        for directory in _HELD_UNDER[kind]:
            if (path := _held(tree, directory, name)) is None:
                continue
            head = (run, path, name, kind, description, dtype)
            rows.extend((*head, column, dim, at) for at, (column, dim) in enumerate(columns))
            if not columns:
                rows.append((*head, None, None, None))
    return pl.DataFrame(rows, schema=_CATALOG_SCHEMA, orient='row').sort('path', 'dim_position')


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


def _copy_the_answer(answer: Path, into: Path, run: str) -> None:
    """*answer*'s layout under *into*, its record and metrics as one file each, every table stamped with *run*."""
    consolidating = (RECORD_FILE, METRICS_FILE)
    apart = {*consolidating, *(file.removesuffix('.parquet') for file in consolidating)}
    shutil.copytree(
        answer,
        into,
        ignore=lambda at, names: apart & set(names) if Path(at) == answer else set(),
        copy_function=lambda source, target: _stamped(Path(source), Path(target), run),
    )
    for file in consolidating:
        stamped = consolidated(answer, file).with_columns(pl.lit(run, dtype=pl.String).alias(RUN))
        stamped.write_parquet(into / file, compression='zstd')


def _stamped(source: Path, target: Path, run: str) -> None:
    """*source* at *target*, a parquet file with the ``specsolve_run`` column set to *run*.

    Streamed, so a spilled answer larger than memory is stamped too, and
    landed through a part file, so *target* may be *source*. Anything that is
    not parquet is copied as it is.
    """
    if source.suffix != '.parquet':
        shutil.copy2(source, target)
        return
    write_whole(pl.scan_parquet(source).with_columns(pl.lit(run, dtype=pl.String).alias(RUN)), target)


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
        if member not in {SPEC_MEMBER, AXIS_MEMBER, DIGESTS_MEMBER, CATALOG_MEMBER}
        and not member.startswith(f'{ANSWER_DIR}/')
        and not (member.startswith(f'{SOURCES_DIR}/') and member.endswith('.parquet') and member.count('/') == 1)
    ]
    if strays or SPEC_MEMBER not in found:
        what = f'holds {strays}' if strays else f'has no {SPEC_MEMBER!r}'
        raise LayoutError(
            f'{named} is not an archive: it {what}. One that archive= writes holds exactly '
            f"'spec.yaml', one 'sources/<key>.parquet' per key the file declares, 'sources.parquet' digesting "
            f"them, 'catalog.parquet' saying what each file holds, 'answer/' holding what the solve returned, "
            f"and 'axis.json' where its sources are sliced."
        )
