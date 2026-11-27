"""The answer's layout on disk: what an archive holds of a result and of a sweep, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the
[`KINDS`][specsolve._answer.outputs.KINDS] the answer carries: one file per name from
a result, one per slice from a sweep, read back as one. Beside them, the
[`Record`][specsolve._answer.records.Record] says how the solve terminated and the
[`Metrics`][specsolve._answer.records.Metrics] what it took, one row of each per
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
from specsolve._answer import WHOLE, Answer
from specsolve._answer.outputs import is_output, kinds_of
from specsolve._answer.records import METRICS_SCHEMA, RECORD_SCHEMA, Metrics, Record, installed_version
from specsolve._names import RUN
from specsolve.errors import LayoutError

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from mathspec.spec import Spec

    from specsolve._answer import Reasons
    from specsolve._answer.outputs import Output

#: The layout of what an archive holds under its ``answer/``, a result's or a
#: sweep's. A change to any of them raises it. Compared, never branched on.
ANSWER_LAYOUT = 5
FORMAT_FILE = 'format.json'


def write_format(directory: Path, outputs: Iterable[Output] | None = None, layout: int = ANSWER_LAYOUT) -> None:
    """Stamp *directory* with the layout its contents are in, the specsolve version that wrote them, and the [`OUTPUTS`][specsolve._answer.outputs.OUTPUTS] an answer carries.

    An archive's own stamp, over its spec and its data, names no *outputs*.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp: dict[str, object] = {'layout': layout, 'specsolve': installed_version('specsolve')}
    if outputs is not None:
        stamp['outputs'] = sorted(outputs)
    (directory / FORMAT_FILE).write_text(json.dumps(stamp))


def read_outputs(directory: Path) -> frozenset[Output]:
    """The [`OUTPUTS`][specsolve._answer.outputs.OUTPUTS] the answer under *directory* carries, as its stamp names them. Called once the stamp's layout is checked.

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


def row_of[R](row_type: Callable[..., R], columns: Mapping[str, object], found: Path) -> R:
    """One row read off disk as the type that declares its columns.

    Args:
        row_type: [`Record`][specsolve._answer.records.Record] or [`Metrics`][specsolve._answer.records.Metrics].
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
            f'layout this package does not read. The layout changes before 1.0 and '
            f'nothing reads an older one back: solve the model again and archive it.'
        )
    return row_type(**columns)


#: The three files beside the tables: how the solve terminated, what reaching
#: it cost, and the reasons behind whatever is deliberately not there.
RECORD_FILE = 'record.parquet'
#: The spec an answer came from, as the YAML it round-trips to.
SPEC_FILE = 'spec.yaml'
METRICS_FILE = 'metrics.parquet'
REASONS_FILE = 'reasons.parquet'
#: A sweep's manifest, its keys as their own type, and the coordinates
#: of each window's horizon.
MANIFEST_FILE = 'sweep.json'
KEYS_FILE = 'keys.parquet'
HORIZON_FILE = 'horizon.parquet'


def table_file(under: Path, kind: str, name: str, position: int | None = None) -> Path:
    """Where *name*'s table of *kind* is held under *under*: one file, or the file of the sweep slice at *position*."""
    return under / kind / f'{name}.parquet' if position is None else under / kind / name / f'{position:06d}.parquet'


def row_file(under: Path, file: str, position: int | None = None) -> Path:
    """Where the one row *file* names is held under *under*: that file, or the sweep slice at *position*'s under its stem."""
    return under / file if position is None else under / file.removesuffix('.parquet') / f'{position:06d}.parquet'


def consolidated(under: Path, file: str) -> pl.DataFrame:
    """The table a sweep writes one file per slice of, under the directory named for *file*, as one table in slice order.

    ``record/`` holds what a single solve writes as ``record.parquet``.

    Raises:
        LayoutError: No such directory under *under*, or one no slice has
            landed in.
    """
    apart = row_file(under, file, 0).parent
    files = sorted(apart.glob('*.parquet'))
    if not files:
        raise LayoutError(
            f'{str(under)!r} holds no slice under {apart.name!r}, so it is not a sweep this package wrote, or its '
            f'first slice never landed. Every record here is written as the fold goes, one file per slice, '
            f'whether or not a slice produced values.'
        )
    return pl.concat([parquet.read(path) for path in files])


def write_spec(directory: Path, spec: Spec | None) -> None:
    """*spec* as [`SPEC_FILE`][] in *directory*, or no file for an answer that came from no spec."""
    if spec is not None:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / SPEC_FILE).write_text(spec.to_yaml())


def read_spec(directory: Path) -> Spec | None:
    """The spec [`write_spec`][] left in *directory*, or ``None`` where it left none."""
    held = directory / SPEC_FILE
    return to_spec(held) if held.is_file() else None


#: How an answer's writer puts one table at one path: [`write_whole`][], or
#: the writer an archive hands it.
type TableWriter = Callable[[pl.DataFrame | pl.LazyFrame, Path], None]


def write_reasons(directory: Path, reasons: Reasons, write: TableWriter) -> None:
    """*reasons* as ``(kind, name, reason)`` rows, or no file at all where there are none, written with *write*.

    An empty *name* is the whole kind, which is how the duals are absent.
    """
    rows = [
        {'kind': kind, 'name': name, 'reason': why} for kind, names in reasons.items() for name, why in names.items()
    ]
    if rows:
        write(pl.DataFrame(rows), directory / REASONS_FILE)


def read_reasons(directory: Path) -> dict[str, dict[str, str]]:
    """What [`write_reasons`][] wrote, as ``{kind: {name: reason}}``; empty where it wrote no file."""
    file = directory / REASONS_FILE
    rows: list[tuple[str, str, str]] = (
        parquet.read(file, columns=['kind', 'name', 'reason']).rows() if file.is_file() else []
    )
    reasons: dict[str, dict[str, str]] = {}
    for kind, name, why in rows:
        reasons.setdefault(kind, {})[name or WHOLE] = why
    return reasons


def write_answer(under: Path, answer: Answer, write: TableWriter, position: int | None = None) -> None:
    """*answer*'s tables, its metrics where it holds them, and its record, under *under*, each written with *write*.

    One file per name, or, for the sweep slice at *position*, one file per
    name and slice. The record is written last, so a slice whose record
    landed is whole. The reasons are the caller's to write: a sweep keeps one
    file of them for all its slices.
    """
    for kind, tables in answer.tables.items():
        for name, table in tables.items():
            write(table, table_file(under, kind, name, position))
    if answer.metrics is not None:
        metrics = pl.DataFrame([answer.metrics._asdict()], schema=METRICS_SCHEMA)
        write(metrics, row_file(under, METRICS_FILE, position))
    write(pl.DataFrame([answer.record._asdict()], schema=RECORD_SCHEMA), row_file(under, RECORD_FILE, position))


def read_answer(under: Path, *, eager: bool) -> Answer:
    """The answer one solve left under *under*, its tables read into memory now where *eager*, else scanned.

    Scanned, *under* has to outlive the answer.

    Raises:
        LayoutError: *under* holds no record, or a row in another layout.
    """
    record_file = under / RECORD_FILE
    if not record_file.is_file():
        raise LayoutError(
            f'{str(under)!r} holds no {RECORD_FILE!r}, so it is not an answer this package wrote. Every one '
            f'carries that record whether or not the solve produced values.'
        )
    metrics_file = under / METRICS_FILE
    kinds = ('primal', 'dual', 'expression', *kinds_of(read_outputs(under)))
    return Answer(
        row_of(Record, parquet.read(record_file).row(0, named=True), record_file),
        row_of(Metrics, parquet.read(metrics_file).row(0, named=True), metrics_file),
        {kind: saved_tables(under / kind, eager=eager) for kind in kinds},
        read_reasons(under),
    )


def saved_tables(under: Path, *, eager: bool) -> dict[str, pl.LazyFrame]:
    """Every ``<name>.parquet`` under *under*, keyed by name; empty where *under* does not exist.

    An archive's [`RUN`][] column is left on disk, so a table read out of one
    equals the table the solve returned.

    Args:
        under: One kind's directory.
        eager: Read each table into memory now, rather than as a
            `polars.scan_parquet` collected at the first read.
    """
    if not under.is_dir():
        return {}
    return {
        file.stem: (parquet.read(file).lazy() if eager else parquet.scan(file)).drop(RUN, strict=False)
        for file in sorted(under.glob('*.parquet'))
    }


def write_whole(table: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    """*table* at *path*, arriving whole: written beside it and renamed into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    parquet.write(table, part)
    part.replace(path)
