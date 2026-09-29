"""Is the data there where a declaration needs it? The positions that ask.

Everything else in this lane reads an absent parameter row as a zero
coefficient (the absence rules). These are the positions that reading has no
answer for — a **divisor**, where zero is not a divisor at all, and a
**constant piece**, where zero is the bound rather than the absence of one —
and each is asked at the last moment the gap is still visible:

=============================  ===================================  ==========================================
position                       the gap looks like                   asked
=============================  ===================================  ==========================================
a divisor under a term         a null coefficient in the share      before the terminal aggregate reads it as 0
a divisor under a constant     a null value in the piece            before [`constant_scalar`][fragments.constant_scalar] sums it away
a constant piece the row sees  a null after the join onto the rows  on the rows pass itself
a constant piece summed away   a coordinate the parameter lacks     of the parameter, the piece no longer showing it
=============================  ===================================  ==========================================

Which parameters stand as constant pieces, and under which region of a
``cases:`` block, is read off the fragments the compiler built
([`parameters`][fragments.TermFragment.parameters],
[`region`][fragments.TermFragment.region]), so the rule is decided once, where
the pieces are made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve.errors import DataError, sparse_divisor_message, uncovered_constant_message
from specsolve.relational.collect import polars_engine
from specsolve.relational.engines.polars.fragments import constant_scalar, join_on
from specsolve.relational.engines.polars.predicates import masked
from specsolve.relational.sinks.handoff import SENSE

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from specsolve.relational.engines.polars.fragments import TermFragment
    from specsolve.relational.engines.polars.scope import Scope


def divisors_of(*expressions: program.Expression) -> tuple[program.Expression, ...]:
    """Every divisor under *expressions*, in walk order."""
    return tuple(node.divisor for node in program.walk(*expressions) if isinstance(node, program.Divide))


def refuse_null_coefficients(stacked: pl.DataFrame, subject: str, *expressions: program.Expression) -> None:
    """A null coefficient in *stacked* means a divisor had no value where the model divided.

    A quotient left-joins its divisor, so a missing value leaves a null — and a
    term whose row was masked out, or whose numerator variable is absent, never
    gets this far. Asked of the stack before any cell collapses, since ``sum``
    reads a null as zero.

    Raises:
        DataError: Naming the divisor parameters of *expressions* and the
            count of undefined entries.
    """
    undefined = int(stacked.get_column('coeff').null_count())
    if undefined:
        params = sorted(program.parameters_of(*divisors_of(*expressions)))
        raise DataError(f'{subject}: {sparse_divisor_message(", ".join(params), undefined)}')


def refuse_null_constants(
    pieces: Sequence[pl.LazyFrame],
    divisors: Collection[str],
    subject: str,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """A null value in a constant *piece* means a divisor had no value where the model divided.

    [`refuse_null_coefficients`][] one position over, and asked before
    [`constant_scalar`][fragments.constant_scalar] rather than after: a constant piece is
    summed per coordinate on its way to the row, and polars reads a null as
    zero, so a gap left behind for this to find is filled in by the time the
    assembled constant is joined. *pieces* are narrowed by the caller to the
    coordinates the declaration builds; *divisors* are the names the refusal
    reports, and nothing is read where there are none. *message* words it for
    the position: a reported expression constrains nothing.

    Raises:
        DataError: Naming *divisors* and the count of undefined values.
    """
    if not divisors:
        return
    counts = pl.collect_all([piece.select(pl.col('cval').null_count()) for piece in pieces])
    undefined = sum(int(count.item()) for count in counts)
    if undefined:
        raise DataError(f'{subject}: {message(", ".join(sorted(divisors)), undefined)}')


def narrowed_to_rows(rows: pl.LazyFrame, consts: Sequence[TermFragment]) -> list[pl.LazyFrame]:
    """Each constant piece cut to the rows built, for [`refuse_null_constants`][].

    A piece keeping the row's own dims is narrowed to the rows built, the
    semi-join standing in for the inner join that narrows a term; one that
    lost them to a reduction is asked whole, because the rows summed into a
    coordinate are exactly the rows a mask over the row's dims cannot speak
    about. Whole still means *if the declaration builds a row at all*, which
    is what the single carried row narrows it by — a ``where`` that emptied
    the frame has answered the question already.
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
    consts: Sequence[tuple[TermFragment, float]],
    c: program.ConstraintDeclaration,
    subject: str,
) -> pl.DataFrame:
    """One constraint's rows as ``(row, sense, rhs)``, every constant piece covering the rows it is given.

    Each constant piece is aggregated to its own coordinates and left-joined
    onto *rows*, so a coordinate it has no row for contributes zero. **The
    coverage check rides on that same pass**: the flag is a boolean column
    dropped once counted, and the refusal precedes any use of the rows. A
    piece built under a region of a ``cases:`` block is owed only inside it —
    a null outside is another region's coordinate. It answers for the piece
    the row is given, which is why a piece that arrives short of the parameter
    behind it — a translation past the edge, a group no member maps to — is
    caught here and nowhere else.

    What it cannot answer for is a gap an aggregation summed away, which is
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
    ).collect(engine=polars_engine())
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
    consts: Sequence[TermFragment],
    c: program.ConstraintDeclaration,
    subject: str,
    sparse: Collection[str],
) -> None:
    """A parameter on a constant side must cover the coordinates the rows ask of it.

    Asked of the *parameter* where [`constant_side`][] asks the assembled
    piece, because the parameter is what still has the answer once an
    aggregation has stood between the two: a summed piece carries one row per
    coordinate it does cover, so the gap it left is not a null a join can find
    but a row that was never there.

    Nothing is read for a parameter outside *sparse*, the names attaching found
    short of their coordinate product — a dense one cannot be short anywhere.

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

    The rows built carry the dims they share with the parameter; the dims a
    reduction summed away are owed whole, a ``where`` over the row's dims
    having no way to narrow them. A region narrows what is owed to the
    coordinates it claims, as it does for the assembled piece — and a region
    claiming no built row at all leaves the parameter owing nothing, which is
    why the narrowing runs even where no dim is shared.
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
    return int(holes.select(pl.len()).collect().item())
