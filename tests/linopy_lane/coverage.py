"""Is the data there where a declaration needs it? The two positions that ask.

Everything else in this lane reads a missing ``neutral`` parameter row as a
zero coefficient (the absence rules). These are the two places that reading
has no answer for: **a divisor**, where zero is not a divisor at all, and **a
constant side**, where zero is the bound rather than the absence of one. Both
are decided against the rows the declaration actually builds, so a ``where``
that removed the coordinate has already answered. A ``missing: absent``
parameter is never asked: its missing row removes the row it would have
reached (:func:`~tests.linopy_lane.absence.combined`). The walk itself is
``program.children`` and ``program.parameters_of``, so "which names can reach a
divisor" is answered once for both lanes.
"""

from __future__ import annotations

from dataclasses import replace
from operator import itemgetter
from typing import TYPE_CHECKING, Any

import xarray as xr
from mathspec import program

from specsolve._messages import (
    non_finite_message,
    reported_divisor_message,
    sparse_divisor_message,
    uncovered_constant_message,
)
from specsolve.errors import DataError
from tests.linopy_lane import absence
from tests.linopy_lane.where import evaluate_where

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from tests.linopy_lane.where import EvaluationContext


def gaps_under(array: Any, mask: Any) -> int:
    """How many slots of *array* are null where *mask* still admits the row.

    The one way this lane asks "is this parameter defined where it is needed";
    ``None`` means nothing narrows the question.
    """
    missing = array.isnull()
    if mask is not None:
        missing = missing & mask
    return int(missing.sum())


def check_constant_side_covers(name: str, row: program.Constraint, ctx: EvaluationContext, mask: Any) -> None:
    """A comparison's constant side must have values wherever the row is built.

    A missing row is read as 0, and where that zero is a constant piece it *is*
    the bound — `x <= cap` becomes `x <= 0`, and `x + hi <= 100` becomes
    `x <= 100`, both binding rather than vanishing. The question is asked of
    every constant piece, a parameter in a variable-free additive position,
    whichever side it was written on and beside whatever term: a coefficient,
    which a variable stands with in a product, is a zero when it is missing (the
    absence rules) and is not one. Keyed to the rows the declaration builds, not
    to the coordinate product: a `where` that removed the coordinate has already
    answered the question.
    """
    found = [pair for side in (row.lhs, row.rhs) for pair in _constant_leaves(side, ctx, mask, coefficient=False)]
    for param, narrowed in sorted((pair for pair in found if not ctx.absent(pair[0])), key=itemgetter(0)):
        missing = gaps_under(ctx.dataset[param], narrowed)
        if missing:
            raise DataError(uncovered_constant_message(param, missing, name))


def _constant_leaves(
    node: program.Expression, ctx: EvaluationContext, mask: Any, coefficient: bool
) -> Iterator[tuple[str, Any]]:
    """Every parameter standing as a constant piece under *node*, with the rows it must cover.

    A constant piece is a parameter in a variable-free additive position — `hi`
    in `x + hi`. Under a sum a missing neutral row is one summand fewer, so a
    reduction ends the walk. A parameter a variable stands with in
    a product is a coefficient instead, and a sparse coefficient is a zero the
    absence rules allow, so ``coefficient`` records having passed through such a
    product on the way down and suppresses the parameters below it. Additive
    structure, a reduction and a division by it leave the flag as it was; a
    ``cases:`` region narrows the mask as it does for the whole walk.
    """
    if isinstance(node, (program.VariableReference, program.Sum, program.WindowSum)):
        return
    if isinstance(node, program.ParameterReference):
        if not coefficient:
            yield node.name, mask
        return
    if isinstance(node, program.Cases):
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            yield from _constant_leaves(region.value, ctx, inside if mask is None else mask & inside, coefficient)
        return
    if isinstance(node, program.Multiply):
        yield from _constant_leaves(node.left, ctx, mask, coefficient or program.carries_variable(node.right))
        yield from _constant_leaves(node.right, ctx, mask, coefficient or program.carries_variable(node.left))
        return
    if isinstance(node, program.Divide):
        yield from _constant_leaves(node.numerator, ctx, mask, coefficient)
        return
    for child in program.children(node):
        yield from _constant_leaves(child, ctx, mask, coefficient)


def _under_regions(
    node: program.Expression, ctx: EvaluationContext, mask: Any
) -> Iterator[tuple[program.Expression, Any]]:
    """Every node under *node*, each with the rows it actually has to cover.

    The mask narrows at every region of a ``cases:`` block: a region's data is
    owed only where that region applies, so asking a piece to cover the whole
    frame would refuse a model the language accepts. Sorted by the caller
    where the order decides which name an error can reach: a mask is an array,
    so the pairs are not orderable among themselves.
    """
    yield node, mask
    if isinstance(node, program.Cases):
        for region in node.regions:
            inside = evaluate_where(region.when, ctx)
            yield from _under_regions(region.value, ctx, inside if mask is None else mask & inside)
        return
    for child in program.children(node):
        yield from _under_regions(child, ctx, mask)


def check_divisors_cover(
    name: str,
    expressions: tuple[program.Expression, ...],
    ctx: EvaluationContext,
    mask: Any,
    evaluate: Callable[[program.Expression], Any],
    *,
    terms: bool = True,
) -> None:
    """A divisor must have a value, and one that is not zero, wherever this declaration divides by it.

    Not "wherever it is indexed": sparse data is the ordinary case, and a check
    keyed to the coordinate product would refuse models that never touch the
    gap. Two things can already have removed a coordinate — the row's own
    ``where``, and the mask on a variable in the numerator — and either is
    enough, so the requirement is their conjunction, narrowed at a ``cases:``
    region like the constant side is.

    Reached before :func:`~tests.linopy_lane.builder._eval`, the last moment the
    gap is visible: :func:`absence.coefficient` fills an uncovered slot with
    0.0 at the parameter leaf, and from there the division yields an infinity
    and the row is masked out silently. A zero the data holds is asked after
    the gaps, of the divisor as *evaluate* reads it, so a divisor that adds up
    to zero is found too. With *terms* false a zero under a dividend that
    carries a variable is left to the row it lands in, which a constraint asks
    once it is built.
    """
    for quotient, params, needed in _quotients(expressions, ctx, mask):
        _refuse_gaps(
            name,
            frozenset(_in_place(quotient.divisor, program.ParameterReference)),
            needed,
            ctx,
            sparse_divisor_message,
        )
        if not terms and program.carries_variable(quotient.numerator):
            continue
        zeros = _zeros_under(evaluate(quotient.divisor), needed)
        if zeros:
            raise DataError(f'{name}: {non_finite_message(", ".join(sorted(params)), zeros)}')


def check_read_divisors_cover(
    name: str, expression: program.Expression, ctx: EvaluationContext, evaluate: Callable[[EvaluationContext], Any]
) -> None:
    """A divisor must have a value at every coordinate where a quotient by it reaches the read.

    The read is taken twice by *evaluate*: once with the divisors' missing
    rows absent, and once with them read as one. A coordinate the second
    answers and the first does not is one a quotient by a missing row reaches,
    so another summand being absent there already keeps it out. A zero is not
    asked: a read takes a quotient by zero as absent (:func:`absence.divisor`).
    """
    divisors = program.parameters_of(*(n.divisor for n in program.walk(expression) if isinstance(n, program.Divide)))
    holed = sorted(p for p in divisors if not ctx.absent(p) and bool(ctx.dataset[p].isnull().any()))
    if not holed:
        return
    filled = ctx.dataset.assign({p: ctx.dataset[p].fillna(1.0) for p in holed})
    with_ones = evaluate(replace(ctx, dataset=filled))
    without = evaluate(replace(ctx, absent_parameter=lambda a: a if a.name in holed else absence.coefficient(a)))
    reached = int((xr.DataArray(with_ones).notnull() & xr.DataArray(without).isnull()).sum())
    if reached:
        raise DataError(f'{name}: {reported_divisor_message(", ".join(sorted(divisors)), reached)}')


def _quotients(
    expressions: tuple[program.Expression, ...], ctx: EvaluationContext, mask: Any
) -> Iterator[tuple[program.Divide, frozenset[str], Any]]:
    """Every quotient by a parameter under *expressions*, with its divisor's parameters and where it divides.

    Where it divides is the region's mask and the mask of every variable in
    the numerator at the quotient's own coordinate, either of which removes
    the coordinate. A quotient that stands at the row's own coordinate is
    also removed where a variable beside it there is absent, which takes the
    row; that mask narrows the quotient's own coordinates and adds none, so
    a count stays one of the quotient's coordinates. A variable declared ``missing: neutral`` is a zero where it is
    masked, and removes nothing.
    """
    at_the_row = [node for e in expressions for node in _at_the_row(e)]
    row_variables = {node.name for node in at_the_row if isinstance(node, program.VariableReference)}
    for expression in expressions:
        for quotient, region in _under_regions(expression, ctx, mask):
            if not isinstance(quotient, program.Divide):
                continue
            params = program.parameters_of(quotient.divisor)
            if not params:
                continue
            needed = region
            for variable in sorted(_in_place(quotient.numerator)):
                if ctx.program.variables[variable].missing != 'neutral':
                    present = absence.present(ctx.model, variable)
                    needed = present if needed is None else (needed & present)
            if any(quotient is node for node in at_the_row):
                own = {d for p in params for d in ctx.dataset[p].dims} | set(getattr(needed, 'dims', ()))
                for variable in sorted(row_variables):
                    if ctx.program.variables[variable].missing != 'neutral':
                        present = absence.present(ctx.model, variable)
                        present = present.any([d for d in present.dims if d not in own])
                        needed = present if needed is None else (needed & present)
            yield quotient, params, needed


def _in_place(
    expression: program.Expression,
    kind: type[program.VariableReference | program.ParameterReference] = program.VariableReference,
) -> set[str]:
    """The variables, or the parameters, under *expression* that stand at its own coordinate.

    A sum, a window, a shift or a lookup moves a variable's slots elsewhere, so
    its mask says nothing about where *expression* exists: `sum(v) / p` divides
    at every coordinate of `p`. A parameter summed there is one summand fewer
    where it has no row, never a gap in what it sums to.
    """
    if isinstance(expression, kind):
        return {expression.name}
    if isinstance(expression, program.Sum | program.WindowSum | program.Translate | program.Join):
        return set()
    return set().union(*(_in_place(child, kind) for child in program.children(expression)))


def _at_the_row(expression: program.Expression) -> Iterator[program.Expression]:
    """The nodes under *expression* that stand at its own coordinate, everywhere it has one.

    Not under a sum, a window, a shift or a lookup, which move coordinates,
    nor inside a ``cases:`` region, which holds only where it applies.
    """
    yield expression
    if isinstance(expression, program.Sum | program.WindowSum | program.Translate | program.Join | program.Cases):
        return
    for child in program.children(expression):
        yield from _at_the_row(child)


def _refuse_gaps(
    name: str, params: frozenset[str], needed: Any, ctx: EvaluationContext, message: Callable[[str, int], str]
) -> None:
    """Refuse the first of *params*, by name, that has no row where *needed* still admits the quotient."""
    for param in sorted(p for p in params if not ctx.absent(p)):
        missing = gaps_under(ctx.dataset[param], needed)
        if missing:
            raise DataError(f'{name}: {message(param, missing)}')


def _zeros_under(divisor: Any, mask: Any) -> int:
    """How many slots of *divisor* are zero where *mask* still admits the row."""
    zero = xr.DataArray(divisor) == 0
    if mask is not None:
        zero = zero & mask
    return int(zero.sum())
