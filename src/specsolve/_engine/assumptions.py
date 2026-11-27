"""The data-time check of a model's ``assumptions:``.

A condition is read as a ``where:`` is, so it agrees with the rows that are
built. [`attach`][specsolve._engine.attaching.attach] ends with it, so a build
and [`tidy_sources`][specsolve._sources.tidy_sources], which both lanes read
data through, pass through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathspec.program import assumption_message

from specsolve._collect import collected
from specsolve._engine.conditions import narrowed
from specsolve._engine.scope import Scope
from specsolve._engine.specialize import rules_out
from specsolve._messages import coordinate_text
from specsolve.errors import DataError

if TYPE_CHECKING:
    from mathspec.program import Assumption, Program

    from specsolve._engine.attaching import AttachedSources


def check_assumptions(program: Program, attached: AttachedSources) -> None:
    """Refuse *attached* data that does not hold what *program* assumes of it.

    An assumption whose ``where`` the data settles false has no coordinate to
    fail at, and is not read.

    Raises:
        DataError: An assumption the data does not hold, in the language's own
            words, with one coordinate it fails at.
    """
    scope = Scope(program, attached, {})
    for name, assumption in program.assumptions.items():
        if rules_out(assumption.where, attached):
            continue
        failing = _a_coordinate_it_fails_at(scope, assumption)
        if failing is not None:
            raise DataError(f'{assumption_message(name, assumption)}{failing}')


def _a_coordinate_it_fails_at(scope: Scope, assumption: Assumption) -> str | None:
    """One coordinate the assumption does not hold at, or ``None`` where it holds everywhere.

    A condition with no value at a coordinate does not hold there. The empty
    string answers a failing assumption over no dimensions. One over a dimension
    without a label has no coordinate to fail at, so it is not read.
    """
    failing = ~assumption.holds
    if assumption.where is not None:
        failing = failing & assumption.where
    dims = scope.in_dimension_order(failing.dims)
    if any(not scope.data.cardinality[d] for d in dims):
        return None
    offending = narrowed(scope, dims, failing).head(1).pipe(collected)
    if not offending.height:
        return None
    row = offending.row(0, named=True)
    at = coordinate_text({d: row[d] for d in dims})
    return f'\n  Not so at {at}' if at else ''
