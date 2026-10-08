"""Is the data there where a declaration needs it? The positions that ask.

Elsewhere an absent parameter row reads as a zero coefficient. A divisor and a
constant piece have no such reading, so each is asked at the last moment the
gap is still visible:

=============================  ===================================  ==========================================
position                       the gap looks like                   asked
=============================  ===================================  ==========================================
a divisor under a term         a null coefficient in the share      before the terminal aggregate reads it as 0
a divisor under a constant     a null value in the piece            before [`constant_scalar`][pieces.constant_scalar] sums it away
a constant piece the row sees  a null after the join onto the rows  on the rows pass itself
a constant piece summed away   a coordinate the parameter lacks     of the parameter, the piece no longer showing it
=============================  ===================================  ==========================================

A divisor that is zero has no reading either. It is asked at the same two
moments, as a NaN in place of the null.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve.errors import DataError
from specsolve.messages import (
    coordinate_text,
    sparse_divisor_message,
    uncovered_constant_message,
    zero_divisor_dims,
    zero_divisor_message,
)
from specsolve.relational.collect import collected, collected_all
from specsolve.relational.engine.pieces import constant_scalar
from specsolve.relational.engine.predicates import masked
from specsolve.relational.engine.scope import join_on
from specsolve.relational.sinks.handoff import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from specsolve.relational.engine.pieces import Piece
    from specsolve.relational.engine.scope import Scope


def divisors_of(*expressions: program.Expression) -> tuple[program.Expression, ...]:
    """Every divisor under *expressions*, in walk order."""
    return tuple(node.divisor for node in program.walk(*expressions) if isinstance(node, program.Divide))


def refuse_null_coefficients(
    stacked: pl.DataFrame,
    located: Sequence[pl.LazyFrame],
    subject: str,
    prog: program.Program,
    *expressions: program.Expression,
) -> None:
    """A null coefficient in *stacked* means a divisor had no value where the model divided, and a NaN a zero one.

    Asked before any cell collapses, since ``sum`` reads a null as zero.
    *located* are the frames *stacked* came from, still holding their dims,
    for naming a coordinate where a divisor is zero.

    Raises:
        DataError: Naming the divisor parameters of *expressions* and the
            count of undefined entries.
    """
    coeff = stacked.get_column('coeff')
    divisors = program.parameters_of(*divisors_of(*expressions))
    zeros = int(coeff.cast(pl.Float64).is_nan().sum())
    _refuse_undefined(int(coeff.null_count()), zeros, divisors, subject, located, 'coeff', prog)


def refuse_null_constants(
    pieces: Sequence[pl.LazyFrame],
    divisors: Collection[str],
    subject: str,
    prog: program.Program,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """A null value in a constant *piece* means a divisor had no value where the model divided, and a NaN a zero one.

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
            piece.select(cval.null_count().alias('null'), cval.cast(pl.Float64).is_nan().sum().alias('nan'))
            for piece in pieces
        ]
    )
    undefined = sum(int(count.item(0, 'null')) for count in counts)
    zeros = sum(int(count.item(0, 'nan')) for count in counts)
    _refuse_undefined(undefined, zeros, divisors, subject, pieces, 'cval', prog, message)


def _refuse_undefined(
    undefined: int,
    zeros: int,
    divisors: Collection[str],
    subject: str,
    located: Sequence[pl.LazyFrame],
    column: str,
    prog: program.Program,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """Refuse *undefined* quotients by a divisor with no value, then *zeros* by a zero one.

    A zero divisor reaches here as a NaN in *column*
    ([`join_mul`][specsolve.relational.engine.pieces.join_mul]).
    """
    names = ', '.join(sorted(divisors))
    if undefined:
        raise DataError(f'{subject}: {message(names, undefined)}')
    if zeros:
        at = _a_zero_at(located, column, zero_divisor_dims({name: prog.parameters[name].dims for name in divisors}))
        raise DataError(f'{subject}: {zero_divisor_message(names, zeros, at)}')


def _a_zero_at(located: Sequence[pl.LazyFrame], column: str, dims: Sequence[str]) -> str:
    """The first coordinate over *dims*, in label order, where a frame of *located* holds a NaN in *column*.

    Read only on the way to a refusal, so it pays a second pass. A frame that
    holds none of *dims* is skipped, and with no frame left ``''`` names no
    coordinate.
    """
    for frame in located:
        shown = [d for d in dims if d in frame.collect_schema().names()]
        if not shown:
            continue
        first = frame.filter(pl.col(column).cast(pl.Float64).is_nan()).select(shown).sort(shown).head(1)
        hit = first.pipe(collected)
        if hit.height:
            return coordinate_text(hit.row(0, named=True))
    return ''


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
    c: program.ConstraintDeclaration,
    subject: str,
) -> pl.DataFrame:
    """One constraint's rows as ``(row, sense, rhs)``, every constant piece covering the rows it is given.

    A coordinate a piece has no row for contributes zero, and the coverage
    check rides on the same pass. A piece built under a ``cases:`` region is
    owed only inside it. A gap an aggregation summed away is
    [`refuse_short_constants`][]' question.

    Raises:
        DataError: A constant piece with no value at a row it is owed.
    """
    accumulated = pl.lit(0.0, dtype=pl.Float64)
    uncovered: pl.Expr | None = None
    carrier = rows
    for i, (p, sign) in enumerate(consts):
        column = f'__const {i}__'
        aggregated = constant_scalar(p).rename({'cval': column})
        carrier = join_on(carrier, aggregated, p.dims, 'left')
        accumulated = accumulated + sign * pl.col(column).fill_null(0.0)
        gap = pl.col(column).is_null()
        if p.region is not None:
            inside = f'__inside {i}__'
            claimed = masked(scope, c.dims, p.region).select(*c.dims).with_columns(pl.lit(True).alias(inside))
            carrier = join_on(carrier, claimed, c.dims, 'left')
            gap = gap & pl.col(inside).fill_null(False)
        uncovered = gap if uncovered is None else uncovered | gap

    gap_column = '__uncovered__'
    built = carrier.select(
        'row',
        pl.lit(c.sense, dtype=SENSE).alias('sense'),
        accumulated.cast(pl.Float64).alias('rhs'),
        *([uncovered.alias(gap_column)] if uncovered is not None else []),
    ).pipe(collected)
    if uncovered is None:
        return built
    gaps = int(built.get_column(gap_column).sum())
    if gaps:
        names = ', '.join(sorted(program.parameters_of(c.lhs, c.rhs)))
        raise DataError(uncovered_constant_message(names, gaps, subject))
    return built.drop(gap_column)


def refuse_short_constants(
    scope: Scope,
    rows: pl.LazyFrame,
    consts: Sequence[Piece],
    c: program.ConstraintDeclaration,
    subject: str,
    sparse: Collection[str],
) -> None:
    """A parameter on a constant side must cover the coordinates the rows ask of it.

    Asked of the parameter, because once a piece is summed its gap is a row
    that was never there, not a null a join can find. Only names in *sparse*,
    the parameters short of their coordinate product, are read.

    Raises:
        DataError: Naming the first short parameter, in name order.
    """
    owed = sorted({(name, i) for i, p in enumerate(consts) for name in p.parameters if name in sparse})
    for name, i in owed:
        missing = _uncovered_coordinates(scope, rows, name, c, consts[i].region)
        if missing:
            raise DataError(uncovered_constant_message(name, missing, subject))


def _uncovered_coordinates(
    scope: Scope,
    rows: pl.LazyFrame,
    param: str,
    c: program.ConstraintDeclaration,
    region: program.Mask | None,
) -> int:
    """How many coordinates *param* owes this constraint and has no row for.

    Dims a reduction summed away are owed whole. A region narrows even where no
    dim is shared, since one claiming no built row leaves nothing owed.
    """
    dims = scope.program.parameters[param].dims
    shared = tuple(d for d in dims if d in c.dims)
    summed = tuple(d for d in dims if d not in c.dims)
    built = rows
    if region is not None:
        built = join_on(built, masked(scope, c.dims, region).select(*c.dims), c.dims, 'semi')
    keys = built.select(*shared).unique() if shared else built.select('row').head(1)
    needed = join_on(keys, masked(scope, summed, None).select(*summed), (), 'cross') if summed else keys
    holes = needed.join(scope.data.parameters[param].select(*dims), on=list(dims), how='anti')
    return int(holes.select(pl.len()).pipe(collected).item())
