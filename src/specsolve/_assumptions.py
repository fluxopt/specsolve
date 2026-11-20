"""The data-time check of a model's ``assumptions:``.

A predicate is read by the relational engine's mask walk, so it agrees with
the rows that are built. Called from
[`tidy_sources`][specsolve._sources.tidy_sources], so both lanes pass through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathspec.program import assumption_message

from specsolve._collect import collected
from specsolve._engine.attaching import attach
from specsolve._engine.predicates import folded, holds_nowhere, masked, settled
from specsolve._engine.scope import Scope
from specsolve._messages import coordinate_text
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl
    from mathspec.program import Assumption, Program

    from specsolve._engine.attaching import AttachedSources


def validate_assumptions(program: Program, sources: Mapping[str, pl.LazyFrame]) -> AttachedSources | None:
    """Refuse data that does not hold what the model assumes of it.

    Nothing is checked while a parameter is unfilled; attaching refuses that.
    An assumption whose ``where`` the data settles false has no coordinate to
    fail at, and is not read.

    Args:
        program: The lowered spec.
        sources: What [`tidy_sources`][specsolve._sources.tidy_sources] holds once every
            parameter, relation and index is read.

    Returns:
        The attached *sources* the assumptions were read against, for a build
        to attach no second time; ``None`` where nothing was attached.

    Raises:
        DataError: An assumption the data does not hold, in the language's own
            words, with one coordinate it fails at.
    """
    if not program.assumptions or any(name not in sources for name in program.parameters):
        return None
    attached = attach(program, sources)
    scope = Scope(program, attached, {})
    wheres = [a.where for a in program.assumptions.values() if a.where is not None]
    tests = settled(scope, wheres)
    for name, assumption in program.assumptions.items():
        if assumption.where is not None and holds_nowhere(folded(assumption.where, tests)):
            continue
        failing = _a_coordinate_it_fails_at(scope, assumption)
        if failing is not None:
            raise DataError(f'{assumption_message(name, assumption)}{failing}')
    return attached


def _a_coordinate_it_fails_at(scope: Scope, assumption: Assumption) -> str | None:
    """One coordinate the assumption does not hold at, or ``None`` where it holds everywhere.

    A predicate with no value at a coordinate does not hold there. The empty
    string answers a failing assumption over no dims. One over a dimension
    without a label has no coordinate to fail at, so it is not read.
    """
    failing = ~assumption.holds
    if assumption.where is not None:
        failing = failing & assumption.where
    dims = scope.in_declaration_order(failing.dims)
    if any(not scope.data.cardinality[d] for d in dims):
        return None
    offending = masked(scope, dims, failing).head(1).pipe(collected)
    if not offending.height:
        return None
    row = offending.row(0, named=True)
    at = coordinate_text({d: row[d] for d in dims})
    return f'\n  Not so at {at}' if at else ''
