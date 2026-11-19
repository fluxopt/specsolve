"""The answer's layout on disk: what a result and a sweep write, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the
[`KINDS`][specsolve._outputs.KINDS] the answer carries: one file per name from
a result, one per slice from a sweep, read back as one. Beside them, the
[`Record`][specsolve._records.Record] says how the solve terminated and the
[`Metrics`][specsolve._records.Metrics] what it took, one row of each per
result or per slice, ``spec.yaml`` is the spec the answer came from, and
``reasons.parquet`` says why a kind or a name is deliberately not there. An
archive holds this layout under its own ``answer/``
([`specsolve._archive_layout`][]).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
from mathspec import to_spec

from specsolve import _parquet as parquet
from specsolve._names import RUN
from specsolve._outputs import KINDS, is_output
from specsolve._records import installed_version
from specsolve.errors import LayoutError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from mathspec.spec import Spec

    from specsolve._outputs import Output

#: The layout a result and a sweep write to disk, and an archive under its
#: ``answer/``. A change to any of them raises it. Compared, never branched on.
ANSWER_LAYOUT = 4
FORMAT_FILE = 'format.json'


def write_format(directory: Path, outputs: Iterable[Output] | None = None, layout: int = ANSWER_LAYOUT) -> None:
    """Stamp *directory* with the layout its contents are in, the specsolve version that wrote them, and the [`OUTPUTS`][specsolve._outputs.OUTPUTS] an answer carries.

    An archive's own stamp, over its spec and its data, names no *outputs*.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp: dict[str, object] = {'layout': layout, 'specsolve': installed_version('specsolve')}
    if outputs is not None:
        stamp['outputs'] = sorted(outputs)
    (directory / FORMAT_FILE).write_text(json.dumps(stamp))


def read_outputs(directory: Path) -> frozenset[Output]:
    """The [`OUTPUTS`][specsolve._outputs.OUTPUTS] the answer under *directory* carries, as its stamp names them. Called after [`check_format`][].

    A name this package has no output for is dropped rather than refused: a
    later specsolve added it, it has no reader here, and the rest of the
    answer is in this layout.
    """
    return frozenset(name for name in json.loads((directory / FORMAT_FILE).read_text())['outputs'] if is_output(name))


def other_layout(directory: Path, layout: int = ANSWER_LAYOUT) -> str | None:
    """How *directory*'s stamp differs from *layout*, as a refusal names it, or ``None`` where it does not."""
    file = directory / FORMAT_FILE
    stamp = json.loads(file.read_text()) if file.is_file() else {}
    found = stamp.get('layout')
    if found == layout:
        return None
    writer = stamp.get('specsolve')
    which = f'in layout {found}' if found is not None else 'with no layout stamp'
    by = f', written by specsolve {writer},' if writer else ''
    return f'{which}{by}'


def check_format(directory: Path) -> None:
    """Refuse a saved answer whose layout is not the one this package reads.

    Called after whatever identifies the directory as an answer at all.

    Raises:
        LayoutError: A missing stamp, or one that is not this package's.
    """
    if (other := other_layout(directory)) is not None:
        raise LayoutError(
            f'{str(directory)!r} holds a saved answer {other} and this package reads layout '
            f'{ANSWER_LAYOUT}. The layout moves before 1.0 and nothing reads another one back: solve '
            f'the model again and save it.'
        )


def row_of[R](row_type: Callable[..., R], columns: Mapping[str, object], found: Path) -> R:
    """One row read off disk as the type that declares its columns.

    Args:
        row_type: [`Record`][specsolve._records.Record] or [`Metrics`][specsolve._records.Metrics].
        columns: The row as read, ``name: value``.
        found: What to name in the message — the file or directory it came from.

    Raises:
        LayoutError: The columns are not the ones *row_type* declares.
    """
    declared = set(row_type._fields)  # pyrefly: ignore[missing-attribute] — every caller passes a NamedTuple
    if (missing := sorted(declared - set(columns))) or (stray := sorted(set(columns) - declared)):
        short = f'is short of {missing}' if missing else f'holds {stray}'
        raise LayoutError(
            f'{str(found)!r} holds a saved {row_type.__name__} row that {short}, so it was written in a '
            f'layout this package does not read. The layout moves before 1.0 and '
            f'nothing reads an older one back: solve the model again and save it.'
        )
    return row_type(**columns)


#: The three files beside the frames: how the solve terminated, what reaching
#: it cost, and the reasons behind whatever is deliberately not there.
RECORD_FILE = 'record.parquet'
#: The spec an answer came from, as the YAML it round-trips to.
SPEC_FILE = 'spec.yaml'
METRICS_FILE = 'metrics.parquet'
REASONS_FILE = 'reasons.parquet'


def consolidated(under: Path, file: str) -> pl.DataFrame:
    """The table *file* names under *under*, whichever shape wrote it, as one frame.

    Either the file itself, or one file per slice under the directory named for
    it — ``record/`` beside ``record.parquet`` — concatenated in slice order.

    Raises:
        LayoutError: Neither shape is under *under*.
    """
    apart = file.removesuffix('.parquet')
    if (single := under / file).is_file():
        frames = [parquet.read(single)]
    elif (many := under / apart).is_dir():
        frames = [parquet.read(path) for path in sorted(many.glob('*.parquet'))]
    else:
        raise LayoutError(
            f'{str(under)!r} holds no {file!r} and no {apart!r} beside it, so it is not a sweep or a saved '
            f'answer this package wrote. Every record here is written as the fold goes — one file, or one '
            f'per slice — whether or not a slice produced values.'
        )
    return pl.concat(frames)


def clear_the_answer(directory: Path) -> None:
    """Remove what a saved answer holds, leaving anything else in *directory* alone."""
    import shutil

    for kind in KINDS:
        shutil.rmtree(directory / kind, ignore_errors=True)
    for member in (RECORD_FILE, METRICS_FILE, REASONS_FILE, FORMAT_FILE, SPEC_FILE):
        (directory / member).unlink(missing_ok=True)


def write_spec(directory: Path, spec: Spec | None) -> None:
    """*spec* as [`SPEC_FILE`][] in *directory*, or no file for an answer that came from no spec."""
    if spec is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SPEC_FILE).write_text(spec.to_yaml())


def read_spec(directory: Path) -> Spec | None:
    """The spec [`write_spec`][] left in *directory*, or ``None`` where it left none."""
    held = directory / SPEC_FILE
    return to_spec(held) if held.is_file() else None


#: How an answer's writer puts one frame at one path: [`write_whole`][], or
#: the writer an archive hands it.
type FrameWriter = Callable[[pl.DataFrame | pl.LazyFrame, Path], None]


def write_reasons(
    directory: Path, no_duals: str | None, absent: Mapping[str, Mapping[str, str]], write: FrameWriter
) -> None:
    """``(kind, name, reason)`` for what a solve could not produce, or no file at all, written with *write*.

    An empty *name* is the whole kind, which is how the duals are absent.
    *absent* is ``{kind: {name: reason}}``, one reason per name left out.
    """
    rows = [] if no_duals is None else [{'kind': 'dual', 'name': '', 'reason': no_duals}]
    rows += [
        {'kind': kind, 'name': name, 'reason': why} for kind, names in absent.items() for name, why in names.items()
    ]
    if rows:
        write(pl.DataFrame(rows), directory / REASONS_FILE)


def read_reasons(directory: Path) -> tuple[str | None, dict[str, dict[str, str]]]:
    """What [`write_reasons`][] wrote: the duals' reason, and ``{kind: {name: reason}}`` for each name left out."""
    file = directory / REASONS_FILE
    rows: list[tuple[str, str, str]] = (
        parquet.read(file, columns=['kind', 'name', 'reason']).rows() if file.is_file() else []
    )
    absent: dict[str, dict[str, str]] = {}
    for kind, name, why in rows:
        if name:
            absent.setdefault(kind, {})[name] = why
    return next((why for kind, name, why in rows if kind == 'dual' and not name), None), absent


def saved_frames(under: Path, *, eager: bool) -> dict[str, pl.LazyFrame]:
    """Every ``<name>.parquet`` under *under*, keyed by name; empty where *under* does not exist.

    An archive's [`RUN`][] column is left on disk, so a frame read out of one
    equals the frame the solve returned.

    Args:
        under: One kind's directory.
        eager: Read each frame into memory now, rather than as a
            `polars.scan_parquet` collected at the first read.
    """
    if not under.is_dir():
        return {}
    return {
        file.stem: (parquet.read(file).lazy() if eager else parquet.scan(file)).drop(RUN, strict=False)
        for file in sorted(under.glob('*.parquet'))
    }


def write_whole(frame: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    """*frame* at *path*, arriving whole: written beside it and renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    parquet.write(frame, part)
    part.replace(path)
