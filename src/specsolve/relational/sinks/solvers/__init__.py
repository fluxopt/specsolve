"""The solver family: one class per solver, holding one model. See ../README.md.

One module per solver, named for it. Each defines a
[`Solver`][specsolve.relational.sinks.solvers.base.Solver] subclass of that name,
whose constructor loads the model and stops there.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.solvers.base import Solver
from specsolve.relational.sinks.solvers.gurobi import Gurobi
from specsolve.relational.sinks.solvers.highs import Highs
from specsolve.relational.sinks.solvers.xpress import Xpress

if TYPE_CHECKING:
    from collections.abc import Mapping
    from typing import Any

    from specsolve.relational.sinks.handoff import Handoff

__all__ = ['SOLVERS', 'Solver', 'loaded', 'solver']

#: Every solver a caller may name. Closed.
SOLVERS: Mapping[str, type[Solver]] = {
    'highs': Highs,
    'gurobi': Gurobi,
    'xpress': Xpress,
}


def solver(name: str) -> type[Solver]:
    """The solver called *name*, or why this build cannot give you it.

    Raises:
        SpecsolveError: A name outside the closed set.
        ModuleNotFoundError: A name inside it whose package this environment
            does not have.
    """
    try:
        found = SOLVERS[name]
    except KeyError:
        raise SpecsolveError(
            f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}.'
        ) from None
    if not found.is_available():
        raise ModuleNotFoundError(
            f'{name} is a solver this build knows, but its package is not installed here. {found.unavailable_message}'
        )
    return found


def loaded(
    held: Solver | None,
    name: str,
    handoff: Handoff,
    solver_options: Mapping[str, Any] | None = None,
) -> Solver:
    """The solver to run *handoff* on — *held* where it may keep what it holds.

    *held* is kept exactly when it is the named class holding a model that
    differs from this one in nothing but numbers — same
    [`structure`][specsolve.relational.sinks.handoff.Handoff.structure], same options — and
    then the new numbers are pushed onto it. Otherwise it is closed, after
    *name* is resolved.
    """
    wanted = solver(name)
    if held is not None:
        if type(held) is wanted and held.keeps(handoff, solver_options):
            held.push(handoff)
            return held
        held.close()
    return wanted(handoff, None, solver_options)
