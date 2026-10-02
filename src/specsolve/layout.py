"""The archive's layout: a spec, its data and its answer as a directory, or that directory zipped.

``spec.yaml``, one ``sources/<key>.parquet`` per key the file declares,
``sources.parquet`` digesting them, ``answer/`` in the layout both answers
save, and ``axis.json`` where the sources are cut. A directory archive is read
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
from specsolve.relational.parquet import METRICS_FILE, RECORD_FILE, RUN, consolidated, digest_of_file, write_whole

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping

    from mathspec import Spec

    from specsolve.lanes import Source

#: The archive's one layout. ``axis.json`` also marks a sweep archive.
SPEC_MEMBER = 'spec.yaml'
AXIS_MEMBER = 'axis.json'
DIGESTS_MEMBER = 'sources.parquet'
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
            is copied, anything else written as *tables* has it; the digest
            is of those bytes, before the run is stamped on.
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
                shutil.copyfile(given, member)
            else:
                tables[name].collect().write_parquet(member, compression='zstd')
            digests[name] = digest_of_file(member)
            _stamped(member, member, run)
        _digest_table(digests, run).write_parquet(tree / DIGESTS_MEMBER)
        if axis is not None:
            (tree / AXIS_MEMBER).write_text(json.dumps(axis))
        _copy_the_answer(answer, tree / ANSWER_DIR, run)
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


def _digest_table(digests: Mapping[str, str], run: str) -> pl.DataFrame:
    """``(specsolve_run, source, digest)`` in source order, so one model's data digests to one table."""
    names = sorted(digests)
    return pl.DataFrame(
        {RUN: [run] * len(names), 'source': names, 'digest': [digests[name] for name in names]},
        schema={RUN: pl.String, 'source': pl.String, 'digest': pl.String},
    )


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
        if member not in {SPEC_MEMBER, AXIS_MEMBER, DIGESTS_MEMBER}
        and not member.startswith(f'{ANSWER_DIR}/')
        and not (member.startswith(f'{SOURCES_DIR}/') and member.endswith('.parquet') and member.count('/') == 1)
    ]
    if strays or SPEC_MEMBER not in found:
        what = f'holds {strays}' if strays else f'has no {SPEC_MEMBER!r}'
        raise LayoutError(
            f'{named} is not an archive: it {what}. One that archive= writes holds exactly '
            f"'spec.yaml', one 'sources/<key>.parquet' per key the file declares, 'sources.parquet' digesting "
            f"them, 'answer/' holding what the solve returned, and 'axis.json' where its sources are sliced."
        )
