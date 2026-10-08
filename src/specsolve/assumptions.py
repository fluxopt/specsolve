"""The data-time check of a model's ``assumptions:``.

A predicate is read by the relational engine's mask walk, so it agrees with
the rows that are built. Called from
[`tidy_sources`][specsolve.sources.tidy_sources], so both lanes pass through it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from mathspec.program import assumption_message

from specsolve.errors import DataError
from specsolve.messages import coordinate_text
from specsolve.relational.collect import collected
from specsolve.relational.engine.attaching import attach
from specsolve.relational.engine.predicates import masked
from specsolve.relational.engine.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl
    from mathspec.program import Assumption, Program


def validate_assumptions(program: Program, sources: Mapping[str, pl.LazyFrame]) -> None:
    """Refuse data that does not hold what the model assumes of it.

    Nothing is checked while a parameter is unfilled; attaching refuses that.

    Args:
        program: The lowered spec.
        sources: What [`tidy_sources`][specsolve.sources.tidy_sources] holds once every
            parameter, relation and index is read.

    Raises:
        DataError: An assumption the data does not hold, in the language's own
            words, with one coordinate it fails at.
    """
    if not program.assumptions or any(name not in sources for name in program.parameters):
        return
    scope = Scope(program, attach(program, sources), {})
    for name, assumption in program.assumptions.items():
        failing = _a_coordinate_it_fails_at(scope, assumption)
        if failing is not None:
            raise DataError(f'{assumption_message(name, assumption)}{failing}')


def _a_coordinate_it_fails_at(scope: Scope, assumption: Assumption) -> str | None:
    """One coordinate the assumption does not hold at, or ``None`` where it holds everywhere.

    A predicate with no value at a coordinate does not hold there. The empty
    string answers a failing assumption over no dims.
    """
    failing = ~assumption.predicate
    if assumption.where is not None:
        failing = failing & assumption.where
    dims = scope.in_declaration_order(failing.dims)
    offending = masked(scope, dims, failing).head(1).pipe(collected)
    if not offending.height:
        return None
    row = offending.row(0, named=True)
    at = coordinate_text({d: row[d] for d in dims})
    return f'\n  Not so at {at}' if at else ''
