"""Is the data there where a declaration needs it? The two positions that ask.

Everything else in this lane reads an absent parameter row as a zero
coefficient (the absence rules). These are the two places that reading has no
answer for: **a divisor**, where zero is not a divisor at all, and **a constant
side**, where zero is the bound rather than the absence of one. Both are
decided against the rows the declaration actually builds, so a ``where`` that
removed the coordinate has already answered. The walk itself is
``program.children`` and ``program.parameters_of``, so "which names can reach a
divisor" is answered once for both lanes.
"""

from __future__ import annotations

from operator import itemgetter
from typing import TYPE_CHECKING, Any

import xarray as xr
from mathspec import program

from specsolve.errors import DataError
from specsolve.messages import non_finite_message, sparse_divisor_message, uncovered_constant_message
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


def check_constant_side_covers(
    name: str, row: program.ConstraintDeclaration, ctx: EvaluationContext, mask: Any
) -> None:
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
    for param, narrowed in sorted(found, key=itemgetter(0)):
        missing = gaps_under(ctx.dataset[param], narrowed)
        if missing:
            raise DataError(uncovered_constant_message(param, missing, name))


def _constant_leaves(
    node: program.Expression, ctx: EvaluationContext, mask: Any, coefficient: bool
) -> Iterator[tuple[str, Any]]:
    """Every parameter standing as a constant piece under *node*, with the rows it must cover.

    A constant piece is a parameter in a variable-free additive position — `hi`
    in `x + hi`, or in `sum(x) + sum(hi)`. A parameter a variable stands with in
    a product is a coefficient instead, and a sparse coefficient is a zero the
    absence rules allow, so ``coefficient`` records having passed through such a
    product on the way down and suppresses the parameters below it. Additive
    structure, a reduction and a division by it leave the flag as it was; a
    ``cases:`` region narrows the mask as it does for the whole walk.
    """
    if isinstance(node, program.Variable):
        return
    if isinstance(node, program.Parameter):
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
    to zero is found too.
    """
    for expression in expressions:
        for quotient, region in _under_regions(expression, ctx, mask):
            if not isinstance(quotient, program.Divide):
                continue
            params = program.parameters_of(quotient.divisor)
            if not params:
                continue
            needed = region
            for variable in sorted(program.variables_of(quotient.numerator)):
                present = absence.present(ctx.model, variable)
                needed = present if needed is None else (needed & present)
            for param in sorted(params):
                missing = gaps_under(ctx.dataset[param], needed)
                if missing:
                    raise DataError(f'{name}: {sparse_divisor_message(param, missing)}')
            zeros = _zeros_under(evaluate(quotient.divisor), needed)
            if zeros:
                raise DataError(f'{name}: {non_finite_message(", ".join(sorted(params)), zeros)}')


def _zeros_under(divisor: Any, mask: Any) -> int:
    """How many slots of *divisor* are zero where *mask* still admits the row."""
    zero = xr.DataArray(divisor) == 0
    if mask is not None:
        zero = zero & mask
    return int(zero.sum())
