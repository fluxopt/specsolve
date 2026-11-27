"""What a solve starts from: the tables ``start=`` takes, checked, read against a build, and cut for a sweep's slices.

[`read_start`][] reads one start for one build, and [`slice_starts`][] cuts
one for each slice of a sweep before any slice is built. Laying a start onto
a build's solver indices is the engine's
([`matched_basis`][specsolve._engine.readback.matched_basis],
[`matched_values`][specsolve._engine.readback.matched_values]).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal, TypedDict, cast

import polars as pl

from specsolve._answer.result import Result
from specsolve._collect import collected
from specsolve._encoding import read_values
from specsolve._inputs import as_table
from specsolve._messages import unknown_name_message
from specsolve._names import VALUE
from specsolve._sinks.solver import BASIS_STATUSES
from specsolve._sweep.sweep import Sweep
from specsolve.errors import SpecsolveError, SpecsolveWarning

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from mathspec.program import Constraint, Program, Variable

    from specsolve._attached import Dimension
    from specsolve._inputs import Label, Source
    from specsolve._sweep.axes import Axis, Slice


class Start(TypedDict, total=False):
    """What a solve starts from, given as tables: under the name of each reader, a table per entry in that reader's shape.

    Each table is ``(dims…, value)`` and is matched by coordinate, so it may
    leave out entries and coordinates. ``primal`` gives values, and
    ``variable_basis`` and ``constraint_basis`` give a basis status in
    [`Result.variable_basis`][]'s words, as a string or as that ``Enum``. An
    earlier [`Result`][] is the same, from the tables it carries. Given to
    [`solve_over`][specsolve._sweep.solve_over.solve_over], a table that carries the
    sliced dimension is cut per slice, as a source is.

    Example::

        start: Start = {
            'primal': {'on': on},
            'variable_basis': {'p': p_status},
            'constraint_basis': {'balance': balance_status},
        }
    """

    primal: Mapping[str, pl.DataFrame]
    variable_basis: Mapping[str, pl.DataFrame]
    constraint_basis: Mapping[str, pl.DataFrame]


def read_start(
    start: Start, program: Program, tidy: Mapping[str, pl.DataFrame], dimensions: Mapping[str, Dimension]
) -> dict[str, dict[str, pl.LazyFrame]]:
    """*start*'s tables, each read and checked as a parameter's source is, over its entry's dimensions.

    Every shape a parameter takes is read, from a parquet path to one number
    for every coordinate; *tidy* holds each dimension's labels as they
    arrived and *dimensions* the same as the build holds them. ``primal``
    holds numbers, and ``variable_basis`` and ``constraint_basis`` hold basis
    statuses in their words.

    Raises:
        SpecsolveError: As [`refuse_unknown_start`][] raises, or a basis
            status outside the five words.
        DataError: A table a parameter over the same dimensions would be refused
            for.
    """
    refuse_unknown_start(start, program)
    read: dict[str, dict[str, pl.LazyFrame]] = {}
    for reader, tables in _by_reader(start).items():
        per, declared = _read_by(reader, program)
        dtype = 'float' if reader == 'primal' else 'str'
        read[reader] = {}
        for name, obj in tables.items():
            kind = _kind(reader, per)
            table = read_values(kind, name, declared[name].dims, obj, tidy, dimensions, dtype).lazy()
            if reader != 'primal':
                _refuse_unknown_statuses(kind, name, table)
            read[reader][name] = table
    return read


def _kind(reader: str, per: str) -> str:
    """How a refusal names a table of *start*: the reader it is under, and what it holds a table per."""
    return f'start= under {reader!r} for {per}'


def refuse_unknown_start_word(start: object) -> None:
    """Refuse a *start* given as a word other than ``'previous'``.

    Raises:
        SpecsolveError: Any other string.
    """
    if isinstance(start, str) and start != 'previous':
        raise SpecsolveError(f"start= takes 'previous' as a word, and not {start!r}.")


def refuse_unknown_start(start: Start, program: Program) -> None:
    """Refuse a *start* keyed by no reader [`Start`][] takes, or naming no entry its reader reads.

    Asked of the names alone, so a sweep asks it before a slice is built.

    Raises:
        SpecsolveError: A key that names no reader, or a name the reader has
            no entry for.
    """
    readers = tuple(Start.__annotations__)
    by_reader = _by_reader(start)
    if unknown := sorted(set(by_reader) - set(readers)):
        raise SpecsolveError(
            f'start= takes, under the name of the reader that returns it, a table per entry: '
            f'{", ".join(map(repr, readers))}, and not {", ".join(map(repr, unknown))}.'
        )
    for reader, tables in by_reader.items():
        per, declared = _read_by(reader, program)
        if missing := next((name for name in tables if name not in declared), None):
            raise SpecsolveError(f'start= names under {reader!r} an {unknown_name_message(per, missing, declared)}')


def _by_reader(start: Start) -> Mapping[str, Mapping[str, Source]]:
    """*start* as the mapping it is, whatever keys it was given."""
    return cast('Mapping[str, Mapping[str, Source]]', start)


def _read_by(reader: str, program: Program) -> tuple[str, Mapping[str, Variable] | Mapping[str, Constraint]]:
    """What *reader* holds a table per, and those entries by name."""
    if reader == 'constraint_basis':
        return 'constraint', program.constraints
    return 'variable', program.variables


def _refuse_unknown_statuses(kind: str, name: str, table: pl.LazyFrame) -> None:
    """Refuse a basis status outside [`BASIS_STATUSES`][]."""
    held = table.select(pl.col(VALUE).cast(pl.String).unique()).pipe(collected)[VALUE]
    if unknown := sorted(set(held.drop_nulls()) - set(BASIS_STATUSES)):
        raise SpecsolveError(
            f"{kind} '{name}' gives the basis status {', '.join(map(repr, unknown))}, and a status is one of "
            f'{", ".join(map(repr, BASIS_STATUSES))}.'
        )


def _refuse_unknown_statuses_of(tables: Mapping[str, Mapping[str, Source]], program: Program) -> None:
    """Refuse a basis table of a start that holds a status outside [`BASIS_STATUSES`][], before a slice is cut.

    Each slice checks its own cut when it is solved, which is after the
    slices before it have solved and been archived.
    """
    for reader, named in tables.items():
        if reader == 'primal':
            continue
        per, _ = _read_by(reader, program)
        for name, obj in named.items():
            if (table := as_table(obj)) is not None and VALUE in table.collect_schema().names():
                _refuse_unknown_statuses(_kind(reader, per), name, table)


def slice_starts(
    start: Sweep | Result | Start | Literal['previous'] | None,
    axis: Axis,
    slices: Sequence[Slice],
    program: Program,
) -> Sequence[Start | Literal['previous'] | None]:
    """What each slice starts from: its cut of *start*, as it cut its sources, or the slice before it.

    Every cut is taken and checked here, before a slice is built, so a start
    that cannot start a slice stops the sweep before anything is solved; it
    also lets a slice solved in another process take its start as data. A
    table without the sliced column reaches every slice whole.

    Raises:
        SpecsolveError: A word other than ``'previous'``, or a name
            [`refuse_unknown_start`][] refuses.

    Warns:
        SpecsolveWarning: A slice the cut leaves nothing, which starts from
            nothing.
    """
    refuse_unknown_start_word(start)
    if start is None or isinstance(start, str):
        return [start] * len(slices)
    column = axis.dim
    tables = _start_tables(start, column)
    refuse_unknown_start(cast('Start', tables), program)
    _refuse_unknown_statuses_of(tables, program)
    starts: list[Start | None] = []
    empty: list[Label] = []
    for current in slices:
        given: dict[str, dict[str, Source]] = {}
        for reader, named in tables.items():
            pieces = {name: _cut_one(obj, column, current.cut) for name, obj in named.items()}
            if held := {name: piece for name, piece in pieces.items() if piece is not None}:
                given[reader] = held
        if not given:
            empty.append(current.key)
        starts.append(cast('Start', given) if given else None)
    if empty:
        warnings.warn(
            f'start= gives the slices {empty} no row, so they start from nothing: no table carries their '
            f'{column!r}, and none reaches every slice. The answer is the same; only the time it takes changes.',
            SpecsolveWarning,
            stacklevel=3,
        )
    return starts


def _cut_one(obj: Source, column: str, cut: Callable[[pl.LazyFrame], pl.LazyFrame]) -> Source | None:
    """*obj* as one slice takes it: a table cut, read into memory to cross a process, or ``None`` where the cut leaves no row.

    A shape that is not a table, such as one number, carries no column to cut
    on and reaches the slice whole.
    """
    table = as_table(obj)
    if table is None:
        return obj
    piece = (cut(table) if column in table.collect_schema().names() else table).pipe(collected)
    return piece if piece.height else None


def _start_tables(start: Sweep | Result | Start, column: str) -> dict[str, dict[str, Source]]:
    """*start* as tables keyed as [`Start`][] is, an earlier sweep's keyed by *column*."""
    if isinstance(start, Result):
        return {
            reader: {name: table.pipe(collected) for name, table in tables.items()}
            for reader, tables in start._start().items()
        }
    if isinstance(start, Sweep):
        return {reader: dict(named) for reader, named in start._start(column).items()}
    return {reader: dict(named) for reader, named in cast('Mapping[str, Mapping[str, Source]]', start).items()}
