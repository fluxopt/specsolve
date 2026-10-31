"""Is the data there where a declaration needs it? The positions that ask.

Elsewhere a missing row of a ``missing: neutral`` parameter reads as a zero coefficient. A divisor and a
constant term have no such reading, so each is asked at the last moment the
gap is still visible:

=============================  ===================================  ==========================================
position                       the gap looks like                   asked
=============================  ===================================  ==========================================
a divisor under a linear term  a null coefficient in the share      before the terminal aggregate reads it as 0
a divisor under a constant     a null value in the term             before [`constant_scalar`][terms.constant_scalar] sums it away
a constant term the row sees   a null after the join onto the rows  on the rows pass itself
=============================  ===================================  ==========================================

A coefficient that is not finite, from a divisor that is zero or a value that
is infinite, has no reading either. It is asked at the same two moments.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._collect import collected, collected_all
from specsolve._engine.predicates import masked
from specsolve._engine.scope import join_on
from specsolve._engine.terms import constant_scalar
from specsolve._messages import non_finite_message, sparse_divisor_message, uncovered_constant_message
from specsolve._sinks.handoff import SENSE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from specsolve._engine.scope import Scope
    from specsolve._engine.terms import Term


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
    frames: Sequence[pl.LazyFrame],
    divisors: Collection[str],
    subject: str,
    message: Callable[[str, int], str] = sparse_divisor_message,
) -> None:
    """A null value in a constant term means a divisor had no value where the model divided.

    One that is not finite has no number to build either.

    Asked before [`constant_scalar`][terms.constant_scalar] sums the term,
    which reads a null as zero. *frames* are narrowed by the caller to the
    coordinates the declaration builds. *message* words it for the position.

    Raises:
        DataError: Naming *divisors* and the count of undefined values.
    """
    if not divisors:
        return
    cval = pl.col('cval')
    counts = collected_all(
        [
            frame.select(
                cval.null_count().alias('null'), cval.cast(pl.Float64).is_finite().not_().sum().alias('non_finite')
            )
            for frame in frames
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


def narrowed_to_rows(rows: pl.LazyFrame, terms: Sequence[Term]) -> list[pl.LazyFrame]:
    """Each constant term cut to the rows built, for [`refuse_null_constants`][].

    A term that lost the row's dims to a reduction is asked whole, but only if
    the declaration builds a row at all.
    """
    return [
        p.frame.join(rows.select(*p.dims), on=list(p.dims), how='semi')
        if p.dims
        else p.frame.join(rows.select('row').head(1), how='cross')
        for p in terms
    ]


def constant_side(
    scope: Scope,
    rows: pl.LazyFrame,
    terms: Sequence[tuple[Term, float]],
    c: program.Constraint,
    subject: str,
) -> pl.DataFrame:
    """One constraint's rows as ``(row, sense, rhs)``, every constant term covering the rows it is given.

    A coordinate a term has no row for contributes zero, and the coverage
    check rides on the same pass. Only a term with a parameter under it is
    owed: one whose parameters all stand beside a variable is a coefficient,
    and its zero is not a bound. A term built under a ``cases:`` region is
    owed only inside it. A gap a reduction summed away is one summand fewer,
    as it is for a ``missing: neutral`` row anywhere inside a sum.

    Raises:
        DataError: A constant term with no value at a row it is owed.
    """
    accumulated = pl.lit(0.0, dtype=pl.Float64)
    gaps: dict[str, frozenset[str]] = {}
    carrier = rows
    for i, (p, sign) in enumerate(terms):
        column = f'__const {i}__'
        aggregated = constant_scalar(p).rename({'cval': column})
        carrier = join_on(carrier, aggregated, p.dims, 'left')
        accumulated = accumulated + sign * pl.col(column).fill_null(0.0)
        if not p.parameters:
            continue
        gap = pl.col(column).is_null()
        if p.region is not None:
            inside = f'__inside {i}__'
            claimed = masked(scope, c.dims, p.region).select(*c.dims).with_columns(pl.lit(True).alias(inside))
            carrier = join_on(carrier, claimed, c.dims, 'left')
            gap = gap & pl.col(inside).fill_null(False)
        carrier = carrier.with_columns(gap.alias(f'__gap {i}__'))
        gaps[f'__gap {i}__'] = p.parameters

    built = carrier.select(
        'row',
        pl.lit(c.sense, dtype=SENSE).alias('sense'),
        accumulated.cast(pl.Float64).alias('rhs'),
        *gaps,
    ).pipe(collected)
    if not gaps:
        return built
    owed = [parameters for column, parameters in gaps.items() if built.get_column(column).any()]
    if owed:
        uncovered = int(built.select(pl.any_horizontal(list(gaps)).sum()).item())
        names = ', '.join(sorted(frozenset[str]().union(*owed)))
        raise DataError(uncovered_constant_message(names, uncovered, subject))
    return built.drop(list(gaps))
