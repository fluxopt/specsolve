"""The answer's layout on disk: what a result and a sweep write, the rows they record, and the writer that lands a file whole.

Under a directory, ``<kind>/<name>`` for each of the [`KINDS`][] the answer
carries: one file per name from a result, one per slice from a sweep, read
back as one. Beside them, the [`Record`][] says how the solve terminated and
the [`Metrics`][] what it took, one row of each per result or per slice, and
``reasons.parquet`` says why a kind or a name is deliberately not there. An
archive holds this layout under its own ``answer/``
([`specsolve.archive_layout`][]).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, TypeGuard, get_args, get_type_hints

import polars as pl

from specsolve.errors import LayoutError, SpecsolveError
from specsolve.relational.status import SolveStatus, status_of

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping
    from pathlib import Path

    from mathspec import Spec

#: What an answer carries only where the solve asked for it with ``outputs=``,
#: beside the primal, the duals and the declared expressions it always
#: carries. ``activity`` is each constraint's left-hand side at the solution,
#: ``reduced_cost`` each variable's reduced cost, ``slack`` each
#: constraint's distance to binding, and ``basis`` the basis status the solve
#: ended on, of each variable and each constraint.
Output = Literal['activity', 'reduced_cost', 'slack', 'basis']

#: Every [`Output`][], in the order a message lists them.
OUTPUTS: tuple[Output, ...] = get_args(Output)

#: What a kind of frame holds one frame per.
Per = Literal['variable', 'constraint']


class OutputKind(NamedTuple):
    """What one kind of frame an [`Output`][] carries is."""

    #: The output that asks for it.
    output: Output
    #: What it holds one frame per.
    per: Per


#: Each kind of frame the [`OUTPUTS`][] carry, named after the reader it comes
#: back through: ``basis`` carries two, ``variable_basis`` and
#: ``constraint_basis``, and every other output one, of its own name.
OUTPUT_KINDS: Mapping[str, OutputKind] = MappingProxyType(
    {
        'activity': OutputKind('activity', 'constraint'),
        'reduced_cost': OutputKind('reduced_cost', 'variable'),
        'slack': OutputKind('slack', 'constraint'),
        'variable_basis': OutputKind('basis', 'variable'),
        'constraint_basis': OutputKind('basis', 'constraint'),
    }
)

#: The kinds that exist exactly where the duals do. Where a solve left none,
#: each carries the duals' reason.
PRICED = frozenset({'dual', 'reduced_cost'})

#: The kinds ``basis`` carries, which exist only where the solve ended on a
#: basis. Where it did not, each carries [`NO_BASIS`][].
BASES = frozenset(kind for kind, carried in OUTPUT_KINDS.items() if carried.output == 'basis')

NO_BASIS = (
    'the solve ended on no basis, so there is no basis status to read. Only an LP solved by simplex, or by '
    'an interior-point method followed by crossover, ends on one: a mixed-integer model, a model with a '
    'quadratic constraint, and an interior-point run with crossover off do not. Solve with crossover on — '
    "HiGHS's 'run_crossover', Gurobi's 'Crossover', Xpress's 'crossover' — or with a simplex method."
)

#: A basis status in the one vocabulary every solver's is read into, each at
#: the index that is its code. A row's bound is its right-hand side, so a
#: binding ``<=`` row is ``at_upper``, a binding ``>=`` row ``at_lower``, and
#: a nonbasic ``==`` row, like a nonbasic variable whose bounds are equal,
#: ``fixed``. ``superbasic`` is nonbasic between its bounds.
BASIS_STATUSES = ('basic', 'at_lower', 'at_upper', 'fixed', 'superbasic')
BASIC, AT_LOWER, AT_UPPER, FIXED, SUPERBASIC = range(len(BASIS_STATUSES))

#: [`BASIS_STATUSES`][] as the dtype a basis is read back in.
BASIS = pl.Enum(BASIS_STATUSES)

#: Every kind of frame an answer can hold: the three every solve answers with,
#: then the [`OUTPUT_KINDS`][]. Each is named after the reader it comes back
#: through.
KINDS = ('primal', 'dual', 'expression', *OUTPUT_KINDS)


def is_output(name: str) -> TypeGuard[Output]:
    """Whether *name* is one of the [`OUTPUTS`][]."""
    return name in OUTPUTS


def checked_outputs(outputs: Iterable[str]) -> frozenset[Output]:
    """*outputs* as a set, once each is checked to be one of [`OUTPUTS`][].

    Raises:
        SpecsolveError: One string rather than a collection of them, or a name
            that is not an output.
    """
    if isinstance(outputs, str):
        raise SpecsolveError(
            f'outputs={outputs!r} is one string, which would name each of its letters. '
            f'Pass a set: outputs={{{outputs!r}}}.'
        )
    asked = frozenset(outputs)
    if unknown := sorted(name for name in asked if not is_output(name)):
        raise SpecsolveError(
            f'outputs names {", ".join(map(repr, unknown))}, and an answer carries only '
            f'{", ".join(map(repr, OUTPUTS))} on request. The primal, the duals and the declared '
            f'expressions are carried always and are not named here.'
        )
    return frozenset(filter(is_output, asked))


def kinds_of(outputs: Iterable[Output]) -> tuple[str, ...]:
    """Each of the [`OUTPUT_KINDS`][] *outputs* carry."""
    return tuple(kind for kind, carried in OUTPUT_KINDS.items() if carried.output in outputs)


def asked_for(kinds: Iterable[str]) -> frozenset[Output]:
    """The [`OUTPUTS`][] that carry *kinds*, each one of the [`OUTPUT_KINDS`][]."""
    return frozenset(OUTPUT_KINDS[kind].output for kind in kinds)


def not_requested_message(kind: str, name: str) -> str:
    """Why an answer refuses *kind*, one of the [`OUTPUT_KINDS`][], of *name*: the solve did not ask for it."""
    return (
        f"cannot read the {kind.replace('_', ' ')} of '{name}': the solve was not asked for it, so this "
        f'answer does not carry it. An answer carries the primal, the duals and the declared expressions, '
        f"and anything else only on request. Solve again with outputs={{'{OUTPUT_KINDS[kind].output}'}}."
    )


#: The prefix reserved, in any letter case, for the columns specsolve adds, so
#: that no name a spec declares can collide with one.
RESERVED = 'specsolve_'


def refuse_reserved(name: str, which: str) -> None:
    """Refuse *name*, which *which* describes, where it starts with [`RESERVED`][] in any letter case.

    Raises:
        SpecsolveError: A name a column specsolve adds could collide with.
    """
    if name.casefold().startswith(RESERVED):
        raise SpecsolveError(
            f'{which} starts with {RESERVED!r}, which is reserved in any letter case for the columns '
            f'specsolve adds, so it could collide with one. Rename it.'
        )


#: The column that holds the numbers beside a declaration's dimensions, in a
#: parameter's table and in every frame an answer comes back as. Besides the
#: names under [`RESERVED`][], it is the one name a dimension may not take.
VALUE = 'value'


def refuse_value(name: str, which: str) -> None:
    """Refuse *name*, which *which* describes, where it is [`VALUE`][] in any letter case.

    Raises:
        SpecsolveError: A name the ``value`` column would collide with.
    """
    if name.casefold() == VALUE:
        raise SpecsolveError(
            f'{which} has the name of the column {VALUE!r}, which holds the numbers beside the dimensions '
            f"in a parameter's table and in every frame an answer comes back as. Query engines read column "
            f'names without case, so the two columns would collide in any letter case. Rename it.'
        )


#: The column an archive adds to every table it holds, naming the run the
#: table came from. Read back, a frame comes without it.
RUN = f'{RESERVED}run'


#: The layout a result and a sweep write to disk, and an archive under its
#: ``answer/``. A change to any of them raises it. Compared, never branched on.
ANSWER_LAYOUT = 5
FORMAT_FILE = 'format.json'


def installed(distribution: str) -> str | None:
    """The installed version of *distribution*, or ``None`` from a source tree nothing installed."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def write_format(directory: Path, outputs: Iterable[Output] | None = None, layout: int = ANSWER_LAYOUT) -> None:
    """Stamp *directory* with the layout its contents are in, the specsolve version that wrote them, and the [`OUTPUTS`][] an answer carries.

    An archive's own stamp, over its spec and its data, names no *outputs*.
    """
    directory.mkdir(parents=True, exist_ok=True)
    stamp: dict[str, object] = {'layout': layout, 'specsolve': installed('specsolve')}
    if outputs is not None:
        stamp['outputs'] = sorted(outputs)
    (directory / FORMAT_FILE).write_text(json.dumps(stamp))


def read_outputs(directory: Path) -> frozenset[Output]:
    """The [`OUTPUTS`][] the answer under *directory* carries, as its stamp names them. Called after [`check_format`][].

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


#: How many hex characters of a sha256 a digest here keeps.
_DIGEST_WIDTH = 16


def digest_of_file(path: Path) -> str:
    """A short, stable name for a file's bytes, read a chunk at a time."""
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        while chunk := handle.read(1 << 20):
            sha.update(chunk)
    return sha.hexdigest()[:_DIGEST_WIDTH]


def digest_of(spec: Spec) -> str:
    """A short, stable name for a spec — what two answers must share to be comparable.

    Over the YAML *spec* round-trips to, which is what an archive writes as
    ``spec.yaml``. The data is not in it: two scenarios of one spec share
    this.
    """
    return hashlib.sha256(spec.to_yaml().encode()).hexdigest()[:_DIGEST_WIDTH]


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
    #: as a time limit or a gap, keeps its value, and an infinite or ``nan``
    #: one is the string ``"inf"``, ``"-inf"`` or ``"nan"``; any other has the
    #: value ``<not recorded>``, so a licence credential never reaches an archive.
    solver_options: str | None = None
    specsolve_version: str | None = None
    mathspec_version: str | None = None


#: The provenance of an answer no solve wrote.
NO_PROVENANCE = Provenance()


class Record(NamedTuple):
    """How a solve terminated, what it reached, and which spec it answered.

    One row per solve, and the same columns whoever wrote them: a result
    writes one, a sweep one per slice, which [`slice_axis`][] and
    [`slice`][] name. A run that left no values writes this and nothing else.
    """

    status: str
    termination_condition: str
    #: What the solve reached, or ``None`` where it reached nothing — null
    #: rather than ``nan``, which every aggregate reads as a number.
    #: [`Result.objective`][specsolve.types.Result.objective] is a float and reads it back as ``nan``.
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
    #: halves of its name. Null until the archive is written. Every other
    #: table the archive holds carries the same column, ``specsolve_run``.
    specsolve_run: str | None = None
    #: A digest of the model this answered — the spec *and* its data, where
    #: [`spec_digest`][] is the document alone. ``None`` for an answer that
    #: never held one.
    model_digest: str | None = None
    #: What the sweep that solved this called its slices — ``scenario``,
    #: ``snapshot_start``, ``draw`` — and which slice this is, as text. Both
    #: null for a single solve. Fixed names rather than a column named for the
    #: axis, so every table written here has the same columns.
    slice_axis: str | None = None
    slice: str | None = None
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
        """The row a solve that terminated this way writes; each argument fills the column of its name.

        ``status`` is derived from *termination_condition*, *objective* is
        kept only where *has_primal* says there are values, and *provenance*
        fills its own fields' columns.
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
    with the same columns whoever writes them, so rows from unrelated runs
    concatenate into one table.

    **Cumulative over the solves it counts**, which [`solves`][] says: ``1``
    for the archive [`specsolve.solve`][] writes and on each slice's row of a
    sweep, whose clocks are that slice's own share. There [`loads`][] is ``1``
    where the solver took the slice from scratch and ``0`` where values were
    pushed onto the model it held, and [`write_seconds`][] is zero, a sweep
    writing no model file.
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
    #: What the archive holding this row was called, as
    #: [`Record.specsolve_run`][]: its file name without a ``.zip``. Null
    #: until one is written.
    specsolve_run: str | None = None
    #: Which sweep slice this row is, as [`Record.slice_axis`][] and
    #: [`Record.slice`][] say it; null for a single solve.
    slice_axis: str | None = None
    slice: str | None = None

    def since(self, earlier: Metrics) -> Metrics:
        """This row less *earlier*'s counts and clocks: the share of the solves between the two."""
        return self._replace(**{name: getattr(self, name) - getattr(earlier, name) for name in _CUMULATIVE})


#: The [`Metrics`][] columns a model sums over its solves.
_CUMULATIVE = (
    'solves',
    'loads',
    'attach_seconds',
    'build_seconds',
    'handoff_seconds',
    'solve_seconds',
    'write_seconds',
)


#: [`Metrics`][]'s columns as they are written, as [`RECORD_SCHEMA`][].
METRICS_SCHEMA = _column_types(Metrics)


def row_of[R](row_type: Callable[..., R], columns: Mapping[str, object], found: Path) -> R:
    """One row read off disk as the type that declares its columns.

    Args:
        row_type: [`Record`][] or [`Metrics`][].
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

    for kind in KINDS:
        shutil.rmtree(directory / kind, ignore_errors=True)
    for member in (RECORD_FILE, METRICS_FILE, REASONS_FILE, FORMAT_FILE):
        (directory / member).unlink(missing_ok=True)


def write_reasons(directory: Path, no_duals: str | None, absent: Mapping[str, Mapping[str, str]]) -> None:
    """``(kind, name, reason)`` for what a solve could not produce, or no file at all.

    An empty *name* is the whole kind, which is how the duals are absent.
    *absent* is ``{kind: {name: reason}}``, one reason per name left out.
    """
    rows = [] if no_duals is None else [{'kind': 'dual', 'name': '', 'reason': no_duals}]
    rows += [
        {'kind': kind, 'name': name, 'reason': why} for kind, names in absent.items() for name, why in names.items()
    ]
    if rows:
        write_whole(pl.DataFrame(rows), directory / REASONS_FILE)


def read_reasons(directory: Path) -> tuple[str | None, dict[str, dict[str, str]]]:
    """What [`write_reasons`][] wrote: the duals' reason, and ``{kind: {name: reason}}`` for each name left out."""
    file = directory / REASONS_FILE
    rows: list[tuple[str, str, str]] = (
        pl.read_parquet(file, columns=['kind', 'name', 'reason']).rows() if file.is_file() else []
    )
    absent: dict[str, dict[str, str]] = {}
    for kind, name, why in rows:
        if name:
            absent.setdefault(kind, {})[name] = why
    return next((why for kind, name, why in rows if kind == 'dual' and not name), None), absent


def checked_kind(kind: str) -> str:
    """*kind*, returned once it is checked to be one of [`KINDS`][].

    Raises:
        SpecsolveError: A *kind* that names no reader.
    """
    if kind not in KINDS:
        raise SpecsolveError(f'kind is one of {", ".join(KINDS)}, not {kind!r}')
    return kind


def saved_frames(under: Path, *, whole: bool) -> dict[str, pl.LazyFrame]:
    """Every ``<name>.parquet`` under *under*, keyed by name; empty where *under* does not exist.

    An archive's [`RUN`][] column is left on disk, so a frame read out of one
    equals the frame the solve returned.

    Args:
        under: One kind's directory.
        whole: Read each frame into memory now, rather than as a
            `polars.scan_parquet` collected at the first read.
    """
    if not under.is_dir():
        return {}
    return {
        file.stem: (pl.read_parquet(file).lazy() if whole else pl.scan_parquet(file)).drop(RUN, strict=False)
        for file in sorted(under.glob('*.parquet'))
    }


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
