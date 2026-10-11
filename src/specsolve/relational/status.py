"""What a solve returned, on two axes.

Copied spelling for spelling from ``linopy.constants``;
`tests/test_solve_status.py` asserts the tables still match.

``termination_condition`` is what the solver said, ``status`` what it means for
the caller. ``ok`` does not mean optimal: a run stopped at a time limit with an
incumbent is ``ok``.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Which termination conditions roll up to which coarse status.
STATUS_TO_TERMINATION_CONDITIONS: dict[str, frozenset[str]] = {
    'ok': frozenset({'optimal', 'time_limit', 'iteration_limit', 'terminated_by_limit', 'suboptimal', 'imprecise'}),
    'warning': frozenset({'infeasible', 'infeasible_or_unbounded', 'unbounded', 'other'}),
    'error': frozenset({'internal_solver_error', 'error'}),
    'aborted': frozenset({'user_interrupt', 'resource_interrupt', 'licensing_problems'}),
    'unknown': frozenset({'unknown'}),
}


def status_of(termination_condition: str) -> str:
    """The coarse status a termination condition rolls up to."""
    for status, conditions in STATUS_TO_TERMINATION_CONDITIONS.items():
        if termination_condition in conditions:
            return status
    return 'unknown'


@dataclass(frozen=True)
class SolveStatus:
    """The outcome of a solve, on both axes plus the solver's own wording."""

    termination_condition: str
    #: Exactly what the solver called it, for a message a user can search for.
    solver_wording: str = ''
    #: Whether the solver reports an actual primal, which the termination
    #: condition does not say.
    has_primal: bool = True

    @property
    def status(self) -> str:
        return status_of(self.termination_condition)

    @property
    def is_ok(self) -> bool:
        """The linopy rollup: no error, abort or refusal — not whether values exist ([`is_readable`][])."""
        return self.status == 'ok'

    @property
    def is_readable(self) -> bool:
        """Whether there are primal values to read.

        A MIP stopped at a time limit before finding any incumbent is ``ok``,
        and its zero-filled ``col_value`` would be read as an answer.
        """
        return self.is_ok and self.has_primal
