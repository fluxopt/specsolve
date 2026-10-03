"""Answers on disk as parquet: the layout a result and a sweep both write, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the three kinds a solve
answers with — the primals, the duals, the named expressions. A result
writes one file under each name; a sweep one per slice, and reads them back
as one. Beside them is the [`Record`][], which says how the solve
terminated: a result writes one row, a sweep one per slice.

A saved result also holds ``activity/<name>`` for every constraint, and
``reasons.parquet`` saying why a kind or a name is deliberately not there.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, NamedTuple, get_args, get_type_hints

import polars as pl

from specsolve.errors import LayoutError, SpecsolveError
from specsolve.relational.status import SolveStatus, status_of

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

#: The three kinds of frame a solve answers with, named after the reader each
#: comes back through, and what each is a frame of.
KINDS = ('primal', 'dual', 'expression')
LABELS = {'primal': 'variable', 'dual': 'constraint', 'expression': 'named expression'}


#: The layout a result, a sweep and an archive write to disk. A change to
#: any of them raises it. Compared, never branched on.
LAYOUT = 2
FORMAT_FILE = 'format.json'


def installed(distribution: str) -> str | None:
    """The installed version of *distribution*, or ``None`` from a source tree nothing installed."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def write_format(directory: Path) -> None:
    """Stamp *directory* with the layout its contents are in, and the specsolve version that wrote them."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / FORMAT_FILE).write_text(json.dumps({'layout': LAYOUT, 'specsolve': installed('specsolve')}))


def check_format(directory: Path) -> None:
    """Refuse a saved answer whose layout is not the one this package reads.

    Called after whatever identifies the directory as an answer at all.

    Raises:
        LayoutError: A missing stamp, or one that is not this package's.
    """
    file = directory / FORMAT_FILE
    stamp = json.loads(file.read_text()) if file.is_file() else {}
    found = stamp.get('layout')
    if found != LAYOUT:
        writer = stamp.get('specsolve')
        which = f'in layout {found}' if found is not None else 'with no layout stamp'
        by = f', written by specsolve {writer},' if writer else ''
        raise LayoutError(
            f'{str(directory)!r} holds a saved answer {which}{by} and this package reads layout '
            f'{LAYOUT}. The layout moves before 1.0 and nothing reads another one back: solve '
            f'the model again and save it. An archive that archive= wrote still holds the model and '
            f'the data to do that with.'
        )


#: How many hex characters of a sha256 a digest here keeps.
_DIGEST_WIDTH = 16


def digest_of_bytes(data: bytes) -> str:
    """A short, stable name for *data* — what the digests here are made with."""
    return hashlib.sha256(data).hexdigest()[:_DIGEST_WIDTH]


def digest_of_file(path: Path) -> str:
    """The same name for a file's bytes, read a chunk at a time."""
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()[:_DIGEST_WIDTH]


def digest_of(yaml: str) -> str:
    """A short, stable name for a spec — what two answers must share to be comparable.

    Over the YAML a ``Spec`` round-trips to, which is what an archive writes
    as ``spec.yaml``. The data is not in it: two scenarios of one spec share
    this.
    """
    return digest_of_bytes(yaml.encode())


class Provenance(NamedTuple):
    """What produced an answer: the solver, the options it ran with, and the packages that built the model.

    Enough to install the same environment again and ask the same question.
    Every field is ``None`` for an answer no solve wrote, such as one built by
    hand.
    """

    #: The solver's name, as ``solver_name`` takes it.
    solver: str | None = None
    #: The installed version of the solver's Python package.
    solver_version: str | None = None
    #: The options the solver ran with, as one JSON object with sorted keys:
    #: ``{}`` where none were passed. An option that changes the answer, such
    #: as a time limit or a gap, keeps its value; any other has the value
    #: ``<not recorded>``, so a licence credential never reaches an archive.
    solver_options: str | None = None
    specsolve_version: str | None = None
    mathspec_version: str | None = None


#: The provenance of an answer no solve wrote.
NO_PROVENANCE = Provenance()


class Record(NamedTuple):
    """How a solve terminated, what it reached, and which spec it answered.

    One row per solve, and the same columns whoever wrote them: a result
    writes one, a sweep one per slice keyed by its own key. A run that left no
    values writes this and nothing else.
    """

    status: str
    termination_condition: str
    #: What the solve reached, or ``None`` where it reached nothing — null
    #: rather than ``nan``, which every aggregate reads as a number.
    #: [`Result.objective`][] is a float and reads it back as ``nan``.
    objective: float | None
    #: Whether the solve produced values, which the condition alone does not
    #: say: a run stopped at a limit before any incumbent is ``ok`` with
    #: nothing to read.
    has_primal: bool
    #: A digest of the spec this answered, or ``None`` where the solve was run
    #: off a lowered program. Null on disk, never an empty string.
    spec_digest: str | None
    #: When the solver returned, in UTC, or ``None`` for a solve that carried
    #: no clock, such as a result built by hand.
    solved_at: datetime | None = None
    #: What the archive holding this answer was called — its file name without
    #: a ``.zip``, so ``runs/nightly-2026-09-10.zip`` writes
    #: ``nightly-2026-09-10`` and a directory called ``case.v2`` keeps both
    #: halves of its name. Null until the archive is written.
    run: str | None = None
    #: A digest of the model this answered — the spec *and* its data, where
    #: [`spec_digest`][] is the document alone. ``None`` for an answer that
    #: never held one.
    model_digest: str | None = None
    #: [`Provenance`][]'s fields, one column each.
    solver: str | None = None
    solver_version: str | None = None
    solver_options: str | None = None
    specsolve_version: str | None = None
    mathspec_version: str | None = None

    @classmethod
    def of(
        cls,
        termination_condition: str,
        objective: float,
        *,
        has_primal: bool,
        spec_digest: str | None,
        solved_at: datetime | None,
        model_digest: str | None = None,
        provenance: Provenance = NO_PROVENANCE,
    ) -> Record:
        """The row a solve that terminated this way writes.

        ``status`` is derived from *termination_condition*.

        Args:
            termination_condition: What the solver said.
            objective: What the solve reached. Written only where there are
                values to read.
            has_primal: Whether there are values, which the condition alone
                does not say.
            spec_digest: A digest of the spec answered, or ``None``.
            solved_at: When the solver returned, in UTC. ``None`` where the
                solve carried no clock.
            model_digest: The built model's digest, or ``None`` where this
                answer never held one.
            provenance: What produced the answer.
        """
        return cls(
            status_of(termination_condition),
            termination_condition,
            objective if has_primal else None,
            has_primal,
            spec_digest,
            solved_at,
            model_digest=model_digest,
            **provenance._asdict(),
        )

    @property
    def provenance(self) -> Provenance:
        """What produced the answer this row records."""
        return Provenance(*(getattr(self, name) for name in Provenance._fields))

    @property
    def solve_status(self) -> SolveStatus:
        """The status this row records, without the solver's own wording."""
        return SolveStatus(self.termination_condition, has_primal=self.has_primal)


#: The column type each Python annotation of a record column is written as.
_WRITTEN_AS: Mapping[type, pl.DataType | type[pl.DataType]] = {
    str: pl.String,
    float: pl.Float64,
    bool: pl.Boolean,
    int: pl.Int64,
    datetime: pl.Datetime(time_zone='UTC'),
}


def _column_types(record: type[NamedTuple]) -> dict[str, pl.DataType | type[pl.DataType]]:
    """*record*'s columns as they are written, off its own annotations.

    ``X | None`` is written as ``X`` holding null.
    """
    written: dict[str, pl.DataType | type[pl.DataType]] = {}
    for name, hint in get_type_hints(record).items():
        declared = next((arg for arg in get_args(hint) if arg is not type(None)), hint)
        if declared not in _WRITTEN_AS:
            raise SpecsolveError(
                f'{record.__name__}.{name} is annotated {declared!r}, which nothing here writes. A record '
                f'column has to say what type it is written as: add it to _WRITTEN_AS.'
            )
        written[name] = _WRITTEN_AS[declared]
    return written


#: [`Record`][]'s columns as they are written, so an absent value keeps its
#: column's type rather than the ``Null`` one polars infers from a single row.
RECORD_SCHEMA = _column_types(Record)


class Metrics(NamedTuple):
    """What a build and its solves took, as the row an archive records beside the answer.

    The scalars of [`Diagnostics`][specsolve.relational.result.Diagnostics],
    with the same columns whoever writes them, so rows written by runs that
    never met concatenate into one table.

    **Cumulative over the model's life.** [`solves`][] says how many solves
    the clocks cover; it reads ``1`` for the archive [`specsolve.solve`][]
    writes.
    """

    #: The shape the build produced, in the solver's own vocabulary.
    columns: int
    rows: int
    nonzeros: int
    #: How many solves the row covers, and how many of those loaded the solver
    #: from scratch. The clocks are cumulative over exactly these solves.
    solves: int
    loads: int
    #: Wall-clock seconds in each phase a build clocks, in the order they run:
    #: the caller's sources onto the plan, the declarations into the model
    #: frames, the built model into a solver, the solver's own run, and the
    #: built model streamed to an LP or MPS file. A phase that never ran writes
    #: zero rather than no column. [`write_seconds`][] is
    #: [`write`][specsolve.api.Model.write]'s clock, not the archive's: what
    #: writing the archive cost is recorded nowhere.
    attach_seconds: float
    build_seconds: float
    handoff_seconds: float
    solve_seconds: float
    write_seconds: float
    #: What the archive holding this row was called, as [`Record.run`][]: its
    #: file name without a ``.zip``. Null until one is written.
    run: str | None = None


#: [`Metrics`][]'s columns as they are written, as [`RECORD_SCHEMA`][].
METRICS_SCHEMA = _column_types(Metrics)


class SliceMetrics(NamedTuple):
    """What one slice of a sweep took — [`Metrics`][] one dimension in.

    A slice's clocks are its own share rather than a cumulative total, and
    ``loaded`` says whether the solver took this slice from scratch. Written per
    slice by the spill and read back as one table.
    """

    #: The shape this slice built, as [`Metrics`][] reports a whole model's.
    columns: int
    rows: int
    nonzeros: int
    #: Whether the solver took this slice's model from scratch instead of
    #: having values pushed onto one it already held. Under a serial fold the
    #: first slice does and the rest do not, so a later ``True`` is a slice
    #: whose data moved a mask; under an executor every slice loads.
    loaded: bool
    #: This slice's own seconds per phase. A sweep writes no file per slice, so
    #: there is no ``write``.
    attach_seconds: float
    build_seconds: float
    handoff_seconds: float
    solve_seconds: float


def row_of[R](row_type: Callable[..., R], columns: Mapping[str, object], found: Path) -> R:
    """One row read off disk as the type that declares its columns.

    Args:
        row_type: [`Record`][], [`Metrics`][] or [`SliceMetrics`][].
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
        frames = [pl.read_parquet(single)]
    elif (many := under / apart).is_dir():
        frames = [pl.read_parquet(path) for path in sorted(many.glob('*.parquet'))]
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

    for kind in (*KINDS, 'activity'):
        shutil.rmtree(directory / kind, ignore_errors=True)
    for member in (RECORD_FILE, METRICS_FILE, REASONS_FILE, FORMAT_FILE):
        (directory / member).unlink(missing_ok=True)


def write_reasons(directory: Path, no_duals: str | None, no_expressions: Mapping[str, str]) -> None:
    """``(kind, name, reason)`` for what a solve could not produce, or no file at all.

    An empty *name* is the whole kind, which is how the duals are absent.
    """
    rows = [] if no_duals is None else [{'kind': 'dual', 'name': '', 'reason': no_duals}]
    rows += [{'kind': 'expression', 'name': name, 'reason': why} for name, why in no_expressions.items()]
    if rows:
        write_whole(pl.DataFrame(rows), directory / REASONS_FILE)


def read_reasons(directory: Path) -> tuple[str | None, dict[str, str]]:
    """What [`write_reasons`][] wrote: the duals' reason, and one per named expression."""
    file = directory / REASONS_FILE
    rows: list[tuple[str, str, str]] = pl.read_parquet(file).rows() if file.is_file() else []
    return (
        next((why for kind, _, why in rows if kind == 'dual'), None),
        {name: why for kind, name, why in rows if kind == 'expression'},
    )


def reader_kind(kind: str) -> str:
    """*kind*, checked to be one of [`KINDS`][].

    Raises:
        SpecsolveError: A *kind* that names no reader.
    """
    if kind not in KINDS:
        raise SpecsolveError(f'kind is one of {", ".join(KINDS)}, not {kind!r}')
    return kind


def write_whole(frame: pl.DataFrame | pl.LazyFrame, path: Path) -> None:
    """*frame* at *path*, arriving whole: written beside it and renamed into place.

    A lazy frame is sunk, so it streams to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + '.part')
    if isinstance(frame, pl.LazyFrame):
        frame.sink_parquet(part)
    else:
        frame.write_parquet(part)
    part.replace(path)
