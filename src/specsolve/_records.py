"""The rows an answer records beside its tables: how the solve terminated, what it took, and what produced it.

One [`Record`][] and one [`Metrics`][] per solve, with the same columns whoever
writes them: a result, a sweep's slice, or an archive.
"""

from __future__ import annotations

from datetime import datetime
from functools import cache
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, NamedTuple, get_args, get_type_hints

import polars as pl

from specsolve._status import SolveStatus, status_of
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping


@cache
def installed_version(distribution: str) -> str | None:
    """The installed version of *distribution*, or ``None`` from a source tree nothing installed.

    Read once per process, as every solve records three.
    """
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


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
    #: value ``<not recorded>``, so a licence credential reaches an archive
    #: only where ``also_record`` names it.
    solver_options: str | None = None
    specsolve_version: str | None = None
    mathspec_version: str | None = None


#: The provenance of an answer no solve wrote.
NO_PROVENANCE = Provenance()


class Record(NamedTuple):
    """How a solve terminated, what it reached, and when.

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
    #: When the solver returned, in UTC, or ``None`` for a solve that carried
    #: no clock, such as a result built by hand.
    solved_at: datetime | None = None
    #: What the archive holding this answer was called — its file name without
    #: a ``.zip``, so ``runs/nightly-2026-09-10.zip`` writes
    #: ``nightly-2026-09-10`` and a directory called ``case.v2`` keeps both
    #: halves of its name. Null until the archive is written. Every other
    #: table the archive holds carries the same column, ``specsolve_run``.
    specsolve_run: str | None = None
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
        solved_at: datetime | None,
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
            solved_at,
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
    """What a build and its solves took, as the row a result records beside its answer.

    The same columns whoever writes them — a result, an archive, a sweep's
    slice, or [`Model.metrics`][specsolve._api.Model.metrics] — so rows from
    unrelated runs concatenate into one table.

    **The ``*_seconds`` columns are the phases of a call**, in the order they
    run: load, tidy, attach, build, handoff, solve, readback, write and
    archive. Each starts from what the one before it produced, so together
    they are the whole call and no step is in two of them; each column says
    what its phase holds. A phase that never ran reads zero rather than no
    column. [`phases`][] lays them out as a table.

    **Cumulative over the solves it counts**, which [`solves`][] says: every
    solve of the model on ``Model.metrics()``, and ``1`` on the row of a
    result, an archive and a sweep's slice, whose clocks are that solve's own
    share, from where the answer before it was handed back. There
    [`solver_instances`][] is ``1`` where the solve took a new solver and
    ``0`` where its values were pushed onto the one held, and
    [`load_seconds`][] is on the row of the answer that built the model. A
    slice's [`write_seconds`][] and [`archive_seconds`][] are zero: a sweep
    writes no model file, and cutting, stitching and archiving the sweep are
    not one slice's.
    """

    #: How many columns the build produced: one per variable at each coordinate
    #: it exists at.
    columns: int
    #: How many rows the build produced: one per constraint at each coordinate
    #: it is built at. A row with no term is not built; one whose terms sum
    #: to zero is.
    rows: int
    #: How many coefficients the constraint matrix holds: one per row and
    #: column a linear term reaches, the terms at one pair summed and a sum of
    #: exactly zero dropped. A quadratic term is not one.
    nonzeros: int
    #: How many solves the row covers. The clocks are cumulative over exactly
    #: these solves.
    solves: int
    #: How many solver instances those solves took: a new one wherever the
    #: solver could not keep the model and take its new values. As many as
    #: there are solves on an iterating driver means the model has a
    #: condition on a parameter that varies.
    solver_instances: int
    #: mathspec's load: ``to_spec`` on the spec as given, a path read and
    #: parsed, every rule that needs no data checked, and the program
    #: compiled. Creating a solver is [`handoff_seconds`][]'s.
    load_seconds: float
    #: The source keys checked against the spec, and each source read into
    #: its tidy table, the way [`tidy`][specsolve._api.tidy] returns it.
    tidy_seconds: float
    #: The tidy tables encoded against the program's dimensions, a
    #: parameter's missing rows filled or refused, and the spec's
    #: ``assumptions:`` checked on them.
    attach_seconds: float
    #: Every entry into the model's columns, rows and matrix.
    build_seconds: float
    #: A ``start=`` read and matched to this build, the solver checked to take
    #: the model, the model loaded into a new solver or its new numbers pushed
    #: onto the one held, and the start given. After an update it also holds
    #: the held solver's digest of its last model, which decides between the
    #: two.
    handoff_seconds: float
    #: The solver's own run, up to the vectors it returns.
    solve_seconds: float
    #: Those vectors laid out by coordinate as the result, and when and how it
    #: was solved recorded on it. Its tables are read lazily, so turning one
    #: into rows is spent where it is read: in [`archive_seconds`][] for an
    #: archive, in your code otherwise.
    readback_seconds: float
    #: The format checked to take the model, and the model streamed to an LP
    #: or MPS file by [`write`][specsolve._api.Model.write].
    write_seconds: float
    #: The spec, the sources and the answer written to disk. Every member is
    #: timed but this row itself, which is written last: writing it and
    #: landing the archive are in its run in
    #: [`Model.timings`][specsolve._api.Model.timings] and in
    #: [`Model.metrics`][specsolve._api.Model.metrics], not in this row.
    archive_seconds: float
    #: What the archive holding this row was called, as
    #: [`Record.specsolve_run`][]: its file name without a ``.zip``. Null
    #: until one is written.
    specsolve_run: str | None = None
    #: Which sweep slice this row is, as [`Record.slice_axis`][] and
    #: [`Record.slice`][] say it; null for a single solve.
    slice_axis: str | None = None
    slice: str | None = None

    @property
    def phases(self) -> pl.DataFrame:
        """The clocks as a table, one row per phase in the order the phases run: ``(phase, seconds)``.

        A phase that never ran is a row of zero. Each phase is defined beside
        its ``*_seconds`` column.
        """
        return pl.DataFrame({'phase': PHASES, 'seconds': [getattr(self, f'{phase}_seconds') for phase in PHASES]})


#: The phases a model is clocked in, in the order they run, each named as its
#: [`Metrics`][] column is less ``_seconds``.
PHASES = tuple(field.removesuffix('_seconds') for field in Metrics._fields if field.endswith('_seconds'))

#: [`Metrics`][]'s columns as they are written, as [`RECORD_SCHEMA`][].
METRICS_SCHEMA = _column_types(Metrics)

#: The columns of [`Model.timings`][specsolve._api.Model.timings]: one row per lap.
TIMINGS_SCHEMA = {
    'run': pl.UInt32,
    'phase': pl.String,
    'kind': pl.String,
    'entry': pl.String,
    'seconds': pl.Float64,
}
