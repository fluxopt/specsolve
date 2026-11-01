"""The positions an absent value is spelled differently in.

Absence is **positional** in this lane: one missing row of a ``missing:
neutral`` parameter is a zero in a coefficient, an error in ``bounds:``, and
false in a ``where`` operand — so there is no single fill to apply once at
load, and each position states its own answer. A ``missing: absent`` row stays
NaN everywhere, and :func:`combined` carries it onto a linopy term as the
absence linopy v1 propagates. The convention underneath is linopy v1's, which
the lane selects on import.
"""

from __future__ import annotations

import math
import operator
from functools import reduce
from typing import TYPE_CHECKING, Any

import xarray as xr

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = [
    'coefficient',
    'combined',
    'divisor',
    'filled',
    'holes',
    'power',
    'present',
    'quotient',
    'unmapped',
    'vacated',
    'variable_term',
]


def present(model: Any, name: str) -> Any:
    """The coordinates variable *name* occupies — ``-1`` is linopy's own marker for an absent slot."""
    return model.variables[name].labels != -1


def unmapped(key: object) -> bool:
    """Whether a relation left this member in no group: ``None``, or the NaN that never equals itself."""
    return key is None or key != key


def variable_term(variable: Any, missing: str) -> Any:
    """The variable as it enters an expression, carrying its declared ``missing:``.

    The mask stays on the variable either way — it is what keeps the absent
    coordinates out of the model. Under v1 an absent slot propagates and takes
    its row, which is ``missing: absent``; ``fillna(0)`` is linopy's own
    per-expression escape back to the other reading: the slot contributes
    nothing and the row stands.
    """
    return variable.fillna(0) if missing == 'neutral' else variable


def variable_value(variable: Any, missing: str) -> Any:
    """The variable's primal as it enters a read, carrying its declared ``missing:``.

    :func:`variable_term`'s two readings over values: a masked slot is NaN,
    which propagates through the arithmetic the way the absent term takes its
    row, and ``missing: neutral`` fills it.
    """
    solution = variable.solution
    return solution.fillna(0.0) if missing == 'neutral' else solution


def coefficient(parameter: Any) -> Any:
    """A parameter in a coefficient position, its uncovered slots at zero.

    ``load_parameters`` reindexes to the master coordinates, so an uncovered
    slot arrives as NaN, which linopy refuses in a user-supplied constant.
    """
    return parameter.fillna(0.0)


def holes(side: Any) -> xr.DataArray | None:
    """Where *side*, a value rather than a linopy term, is absent; ``None`` where it holds no absence.

    A ``missing: absent`` parameter is the one value that arrives NaN at a
    build, and linopy refuses a NaN beside a term, so it is asked before the
    two meet.
    """
    if not isinstance(side, xr.DataArray) or side.dtype.kind not in 'fc':
        return None
    null = side.isnull()
    return null if bool(null.any()) else None


def combined(op: Callable[[Any, Any], Any], left: Any, right: Any, *, neutral: float = 0.0) -> Any:
    """``op(left, right)``, a NaN on a value side beside a linopy term carried as the term's absence.

    Two values meet in xarray, where NaN propagates on its own; a value and a
    term meet in linopy, which refuses NaN in a constant. So the value's NaN
    is filled with *neutral* — whatever leaves ``op`` defined there — and the
    result is masked back to absent at exactly those coordinates, which takes
    the row as any absent term does.
    """
    terms = [hasattr(side, 'to_linexpr') or hasattr(side, 'nterm') for side in (left, right)]
    gaps = [holes(side) for side in (left, right)]
    if not any(terms) or all(gap is None for gap in gaps):
        return op(left, right)
    absent = reduce(operator.or_, (gap for gap in gaps if gap is not None))
    sides = [side.fillna(neutral) if gap is not None else side for side, gap in zip((left, right), gaps, strict=True)]
    result = op(*sides)
    if hasattr(result, 'to_linexpr'):
        result = result.to_linexpr()
    return result.where(~absent)


def power(base: Any, exponent: Any) -> Any:
    """``base ** exponent``, absent wherever either side is.

    IEEE answers ``nan ** 0`` and ``1 ** nan`` with 1, and the language
    answers an absent operand with absence, so the holes are put back after
    the power is taken.
    """
    raised = base**exponent
    for side in (base, exponent):
        if (gap := holes(side)) is not None:
            raised = raised.where(~gap)
    return raised


def quotient(numerator: Any, divisor: Any) -> Any:
    """``numerator / divisor`` as a build divides, a term's zero constant staying zero over a zero divisor.

    linopy divides a term's constant with its coefficients, and ``0 / 0`` is
    the NaN that marks an absent slot, so a term with no constant there — a sum
    with no terms among them — would be absent and take its row. The
    coefficients keep the infinity or NaN a zero divisor leaves: whether one
    reaches a row the build keeps is the builder's question.
    """
    result = combined(operator.truediv, numerator, divisor, neutral=1.0)
    if not hasattr(result, 'const') or not isinstance(divisor, xr.DataArray):
        return result
    before = numerator.to_linexpr().const if hasattr(numerator, 'to_linexpr') else numerator.const
    const = result.const.where(~((before == 0) & (divisor == 0)), 0.0)
    return type(result)(result.data.assign(const=const), result.model)


def divisor(value: Any) -> Any:
    """A divisor as a read takes it: absent where it is zero, since a reported quotient by zero has no value (#1776).

    A build divides by the plain divisor (:func:`quotient`), and a zero in it
    is refused where the quotient reaches what the build keeps, as the
    relational engine refuses it.
    """
    if hasattr(value, 'where'):
        return value.where(value != 0)
    return value if value != 0 else math.nan


def vacated(shifted: Any, operand: Any, over: str, vacated: Any, fill: float) -> Any:
    """*shifted*, with the positions the shift vacated filled — and only those.

    linopy v1 counts ``.shift()`` among the operations that *create* absence,
    so the edge propagates and drops the row — the language's answer too. This
    is the opt-out, reached only under a numeric fill.

    ``fillna`` alone cannot spell it: by the time it runs, the edge the shift
    just made and a coordinate *operand* never had are both absent, and filling
    the second builds a row asserting ``x <= 0`` where the model said nothing.
    So over a term the fill lands where the shift vacated **and** the operand
    carries the coordinate, and every other slot keeps the absence it arrived
    with. Over data it lands at every vacated position, as an unpartitioned
    ``DataArray.shift`` fills it.
    """
    carried = True if isinstance(operand, xr.DataArray) else (~operand.isnull()).any(over)
    keep = carried & (~shifted.isnull() | vacated)
    return filled(shifted, fill).where(keep)


def filled(expression: Any, fill: float) -> Any:
    """*expression* with every absence in it standing as *fill*.

    ``to_linexpr()`` first when the operand is still a bare ``Variable``:
    ``Variable.fillna`` means a label fill on the released line and an
    expression fill on the v1 branch, and only the expression method is stable.
    """
    if hasattr(expression, 'to_linexpr'):
        expression = expression.to_linexpr()
    return expression.fillna(fill)
