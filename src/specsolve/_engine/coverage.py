"""Is the data there where a declaration needs it? The positions that ask.

Elsewhere a missing row of a ``missing: neutral`` parameter reads as a zero coefficient. A divisor and a
constant piece have no such reading, so each is asked at the last moment the
gap is still visible:

=============================  ===================================  ==========================================
position                       the gap looks like                   asked
=============================  ===================================  ==========================================
a divisor under a term         a null coefficient in the share      before the terminal aggregate reads it as 0
a divisor under a constant     a null value in the piece            before [`constant_scalar`][pieces.constant_scalar] sums it away
a constant piece the row sees  a null after the join onto the rows  on the rows pass itself
=============================  ===================================  ==========================================

A coefficient that is not finite, from a divisor that is zero or a value that
is infinite, has no reading either. It is asked at the same two moments.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._collect import collected, collected_all
from specsolve._engine.pieces import constant_scalar
from specsolve._engine.predicates import masked
from specsolve._engine.scope import join_on
from specsolve._messages import non_finite_message, sparse_divisor_message, uncovered_constant_message
from specsolve._sinks.handoff import SENSE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from specsolve._engine.pieces import Piece
    from specsolve._engine.scope import Scope


def divisors_of(*expressions: program.Expression) -> tuple[program.Expression, ...]:
    """Every divisor under *expressions*, in walk order."""
    return tuple(node.divisor for node in program.walk(*expressions) if isinstance(node, program.Divide))


def refuse_null_coefficients(stacked: pl.DataFrame, subject: str, *expressions: program.Expression) -> None:
    """A null coefficient in *stacked* means a divisor had no value where the model divided.

    One that is not finite has no number to build either.

    Asked before any cell collapses, since ``sum`` reads a null as zero.

    Raises:
        DataError: Naming the divisor parameters of *expressions* and the
            count of undefined entries.
    """
    coeff = stacked.get_column('coeff')
    divisors = program.parameters_of(*divisors_of(*expressions))
    _refuse_undefined(int(coeff.null_count()), int(coeff.is_finite().not_().sum()), divisors, subject)


def refuse_null_constants(
    pieces: Sequence[pl.LazyFrame],
    divisors: Collection[str],
    subject: str,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """A null value in a constant *piece* means a divisor had no value where the model divided.

    One that is not finite has no number to build either.

    Asked before [`constant_scalar`][pieces.constant_scalar] sums the piece,
    which reads a null as zero. *pieces* are narrowed by the caller to the
    coordinates the declaration builds. *message* words it for the position.

    Raises:
        DataError: Naming *divisors* and the count of undefined values.
    """
    if not divisors:
        return
    cval = pl.col('cval')
    counts = collected_all(
        [
            piece.select(
                cval.null_count().alias('null'), cval.cast(pl.Float64).is_finite().not_().sum().alias('non_finite')
            )
            for piece in pieces
        ]
    )
    undefined = sum(int(count.item(0, 'null')) for count in counts)
    non_finite = sum(int(count.item(0, 'non_finite')) for count in counts)
    _refuse_undefined(undefined, non_finite, divisors, subject, message)


def _refuse_undefined(
    undefined: int,
    non_finite: int,
    divisors: Collection[str],
    subject: str,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """Refuse *undefined* quotients by a divisor with no value, then *non_finite* values."""
    names = ', '.join(sorted(divisors))
    if undefined:
        raise DataError(f'{subject}: {message(names, undefined)}')
    if non_finite:
        raise DataError(f'{subject}: {non_finite_message(names, non_finite)}')


def narrowed_to_rows(rows: pl.LazyFrame, consts: Sequence[Piece]) -> list[pl.LazyFrame]:
    """Each constant piece cut to the rows built, for [`refuse_null_constants`][].

    A piece that lost the row's dims to a reduction is asked whole, but only if
    the declaration builds a row at all.
    """
    return [
        p.frame.join(rows.select(*p.dims), on=list(p.dims), how='semi')
        if p.dims
        else p.frame.join(rows.select('row').head(1), how='cross')
        for p in consts
    ]


def constant_side(
    scope: Scope,
    rows: pl.LazyFrame,
    consts: Sequence[tuple[Piece, float]],
    c: program.Constraint,
    subject: str,
) -> pl.DataFrame:
    """One constraint's rows as ``(row, sense, rhs)``, every constant piece covering the rows it is given.

    A coordinate a piece has no row for contributes zero, and the coverage
    check rides on the same pass. Every piece attached by position goes onto
    the rows before any join, which keeps no order once the build streams. A piece built under a ``cases:`` region is
    owed only inside it. A gap a reduction summed away is one summand fewer,
    as it is for a ``missing: neutral`` row anywhere inside a sum.

    Raises:
        DataError: A constant piece with no value at a row it is owed.
    """
    accumulated = pl.lit(0.0, dtype=pl.Float64)
    gaps: list[str] = []
    built_rows = rows.pipe(collected)
    whole = [set(p.dims) == set(c.dims) for p, _ in consts]
    pieces = [
        replace(p, frame=p.frame.pipe(collected).lazy()) if w else p for (p, _), w in zip(consts, whole, strict=True)
    ]
    aligned = [_aligned(built_rows, p) if w else None for p, w in zip(pieces, whole, strict=True)]
    carrier = built_rows.lazy().with_columns(
        values.alias(f'__const {i}__') for i, values in enumerate(aligned) if values is not None
    )
    for i, ((p, sign), piece, values) in enumerate(zip(consts, pieces, aligned, strict=True)):
        column = f'__const {i}__'
        if values is None:
            carrier = join_on(carrier, constant_scalar(piece).rename({'cval': column}), p.dims, 'left')
        accumulated = accumulated + sign * pl.col(column).fill_null(0.0)
        gap = pl.col(column).is_null()
        if p.region is not None:
            inside = f'__inside {i}__'
            claimed = masked(scope, c.dims, p.region).select(*c.dims).with_columns(pl.lit(True).alias(inside))
            carrier = join_on(carrier, claimed, c.dims, 'left')
            gap = gap & pl.col(inside).fill_null(False)
        carrier = carrier.with_columns(gap.alias(f'__gap {i}__'))
        gaps.append(f'__gap {i}__')

    built = carrier.select(
        'row',
        pl.lit(c.sense, dtype=SENSE).alias('sense'),
        accumulated.cast(pl.Float64).alias('rhs'),
        *gaps,
    ).pipe(collected)
    if not gaps:
        return built
    owed = [p.parameters for (p, _), column in zip(consts, gaps, strict=True) if built.get_column(column).any()]
    if owed:
        uncovered = int(built.select(pl.any_horizontal(gaps).sum()).item())
        names = ', '.join(sorted(frozenset[str]().union(*owed)))
        raise DataError(uncovered_constant_message(names, uncovered, subject))
    return built.drop(gaps)


def _aligned(rows: pl.DataFrame, piece: Piece) -> pl.Series | None:
    """*piece*'s value per row, by position, where its coordinates are the rows' own, in the rows' order.

    Asked only of a piece over every dim of the constraint. The rows are one
    per coordinate, so a match row for row leaves each row exactly one value,
    and [`constant_scalar`][]'s sum and the join onto the rows change nothing but
    the cost. A piece over fewer dims can match on those dims and still hold a
    row per label it summed away, each owed to the same row. ``None`` where the
    coordinates differ in any way, for the join to answer.
    """
    frame = piece.frame.pipe(collected)
    if frame.height != rows.height or not frame.select(piece.dims).equals(rows.select(piece.dims)):
        return None
    return frame.get_column('cval')
