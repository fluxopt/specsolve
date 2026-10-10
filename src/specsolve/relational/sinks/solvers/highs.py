"""The ``highs`` solver: the whole model straight into HiGHS, in one call.

The default, and the only one whose dependency ships with the package. Every
vector crosses as a numpy buffer. ``highspy`` is imported inside the functions,
so importing this module stays free for callers that only write LP files.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from specsolve.errors import SpecsolveError
from specsolve.relational.answer_layout import AT_LOWER, AT_UPPER, BASIC, BASIS_STATUSES, FIXED, SUPERBASIC
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.handoff import SENSE_CODES
from specsolve.relational.sinks.solvers.base import (
    Basis,
    InfeasibleSubsystemIndices,
    SolveAnswer,
    Solver,
    basis_codes,
    solver_vector,
)
from specsolve.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    import polars as pl

    from specsolve.relational.sinks.handoff import ColumnVectors, Handoff, RowVectors


#: HiGHS model status -> termination condition, copied from linopy's
#: ``Highs.CONDITION_MAP``; ``tests/test_solve_status.py`` asserts it still matches.
_CONDITION_OF_HIGHS_STATUS = {
    'kNotset': 'unknown',
    'kLoadError': 'internal_solver_error',
    'kModelError': 'internal_solver_error',
    'kPresolveError': 'internal_solver_error',
    'kSolveError': 'internal_solver_error',
    'kPostsolveError': 'internal_solver_error',
    'kModelEmpty': 'unknown',
    'kMemoryLimit': 'resource_interrupt',
    'kOptimal': 'optimal',
    'kInfeasible': 'infeasible',
    'kUnboundedOrInfeasible': 'infeasible_or_unbounded',
    'kUnbounded': 'unbounded',
    'kObjectiveBound': 'terminated_by_limit',
    'kObjectiveTarget': 'terminated_by_limit',
    'kTimeLimit': 'time_limit',
    'kIterationLimit': 'iteration_limit',
    'kSolutionLimit': 'terminated_by_limit',
    'kInterrupt': 'user_interrupt',
    'kUnknown': 'unknown',
}


#: ``kIisModelStatusIrreducible``, by value: highspy does not bind the enum.
_IIS_IRREDUCIBLE = 3


def _built(handoff: Handoff, solver_options: Mapping[str, Any] | None) -> tuple[Any, ColumnVectors, RowVectors]:
    """The populated `highspy.Highs`, and the column and row vectors it was loaded with.

    ``iis_strategy`` leads the caller's options: the default checks bounds
    alone, and on a conflict between rows it returns an empty subsystem with
    no error.

    The integrality vector spans every column even where none is integer: HiGHS
    reads an empty one as whatever the memory held (1.15.1).
    """
    import highspy
    import numpy as np

    if handoff.qmatrix.height:
        raise SpecsolveError(
            'HiGHS has no quadratic-constraint concept at all — no entry point takes one — and '
            f'this model has {handoff.row_count - handoff.linear_row_count} such rows. Solving through '
            'sps.solve() '
            'refuses this earlier and names the sinks that do take it; constructing Highs '
            'directly skips that, and loading the rows without their quadratic part would be a '
            'different model that solves.'
        )

    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    strategy = highspy.IisStrategy
    h.setOptionValue('iis_strategy', int(strategy.kIisStrategyFromLp) | int(strategy.kIisStrategyIrreducible))
    for option, value in (solver_options or {}).items():
        h.setOptionValue(option, value)

    cols = handoff.dense_columns(inf)
    rows = handoff.dense_rows(inf)
    rlb, rub = _row_bounds(rows, inf)
    sense = highspy.ObjSense.kMaximize if handoff.objective_sense == 'maximize' else highspy.ObjSense.kMinimize
    empty_i = np.empty(0, dtype=np.int32)
    empty_f = np.empty(0, dtype=np.float64)
    _loaded(
        h,
        h.passModel(
            handoff.column_count,
            handoff.row_count,
            handoff.matrix.height,
            0,
            int(highspy.MatrixFormat.kRowwise),
            int(highspy.HessianFormat.kTriangular),
            int(sense),
            0.0,
            cols.cost,
            cols.lb,
            cols.ub,
            rlb,
            rub,
            handoff.row_starts.astype(np.int32),
            handoff.matrix['col'].to_numpy(),
            handoff.matrix['coeff'].to_numpy(),
            empty_i,
            empty_i,
            empty_f,
            _integrality(cols),
        ),
        'the model',
    )
    _pass_hessian(h, handoff)
    return h, cols, rows


def _integrality(cols: Any) -> Any:
    """The per-column integrality vector HiGHS reads: 0 continuous, 1 integer."""
    import numpy as np

    return cols.integral.astype(np.int32)


def _pass_hessian(h: Any, handoff: Handoff) -> None:
    r"""The objective's quadratic part, as the Hessian HiGHS reads.

    ``passHessian`` takes :math:`Q` in :math:`\frac12 x^\top Q x`, lower
    triangle only, in CSC order. A diagonal pair :math:`q\,x_i^2` needs
    :math:`Q_{ii} = 2q`; an off-diagonal pair :math:`q\,x_i x_j` is stored as
    :math:`q`, since the symmetric matrix holds it twice. It goes onto the
    loaded model, so [`Highs.push`][] replaces it without a reload.
    """
    import highspy
    import numpy as np

    if not handoff.quad.height:
        return
    lower = handoff.quad['col_r'].to_numpy().astype(np.int32, copy=False)
    upper = handoff.quad['col_l'].to_numpy().astype(np.int32, copy=False)
    diagonal = lower == upper
    values = np.where(diagonal, handoff.quad['coeff'].to_numpy() * 2.0, handoff.quad['coeff'].to_numpy())

    order = np.lexsort((lower, upper))
    starts = np.zeros(handoff.column_count + 1, dtype=np.int32)
    np.add.at(starts, upper + 1, 1)
    _loaded(
        h,
        h.passHessian(
            handoff.column_count,
            len(order),
            int(highspy.HessianFormat.kTriangular),
            np.cumsum(starts, out=starts),
            lower[order],
            values[order],
        ),
        'the quadratic objective',
    )


class Highs(Solver):
    """HiGHS, holding one model at ``.handle``, a `highspy.Highs`.

    A re-solve pushes bounds, costs and right-hand sides onto the held model
    and starts from the basis the last solve ended on.
    """

    #: The loaded model. ``close`` drops it.
    _handle: Any
    #: The column and row vectors the loaded model holds now, so [`push`][] sends only what moved.
    _column_vectors: ColumnVectors
    _row_vectors: RowVectors

    requires = ('highspy',)
    recorded_options = frozenset(
        {
            'time_limit',
            'mip_rel_gap',
            'mip_abs_gap',
            'random_seed',
            'threads',
            'solver',
            'presolve',
            'primal_feasibility_tolerance',
            'dual_feasibility_tolerance',
            'mip_feasibility_tolerance',
        }
    )
    unavailable_message = 'highspy ships with specsolve, so a build without it is broken rather than missing an extra'

    #: No SOS concept, and a Hessian beside integrality is refused; the pair is
    #: probed in ``test_sink_capability_probes.py``.
    lp_values = MappingProxyType({'complete': 'used', 'partial': 'no_gain'})
    capabilities = Capabilities(
        supports=frozenset({'integrality', 'quadratic_objective'}),
        excludes=(frozenset({'quadratic_objective', 'integrality'}),),
    )

    def _load(self, handoff: Handoff, batch_rows: int | None) -> None:
        """Load in one call — *batch_rows* is the family's parameter and this member has no batches."""
        del batch_rows
        self._handle, self._column_vectors, self._row_vectors = _built(handoff, self._options)

    @property
    def handle(self) -> Any:
        return self._handle

    def push(self, handoff: Handoff) -> None:
        """Send only the entries that differ from what the model holds: HiGHS spends its time per entry sent, moved or not."""
        import highspy
        import numpy as np

        inf = highspy.kHighsInf
        h = self._handle
        cols, was = handoff.dense_columns(inf), self._column_vectors
        rows, had = handoff.dense_rows(inf), self._row_vectors

        cost = np.flatnonzero(cols.cost != was.cost).astype(np.int32)
        _loaded(h, h.changeColsCost(len(cost), cost, cols.cost[cost]), 'new costs')
        bounds = np.flatnonzero((cols.lb != was.lb) | (cols.ub != was.ub)).astype(np.int32)
        _loaded(h, h.changeColsBounds(len(bounds), bounds, cols.lb[bounds], cols.ub[bounds]), 'new bounds')
        moved = np.flatnonzero((rows.sense != had.sense) | (rows.rhs != had.rhs)).astype(np.int32)
        lower, upper = _row_bounds(rows, inf)
        _loaded(h, h.changeRowsBounds(len(moved), moved, lower[moved], upper[moved]), 'new right-hand sides')
        self._column_vectors, self._row_vectors = cols, rows
        _pass_hessian(h, handoff)

    def _basis(self) -> tuple[Any, Any] | None:
        """``getBasis``, where it is valid; its statuses are ``kLower``, ``kBasic``, ``kUpper``, ``kZero``, ``kNonbasic``."""
        basis = self._handle.getBasis()
        if not basis.valid:
            return None
        codes = (AT_LOWER, BASIC, AT_UPPER, SUPERBASIC, SUPERBASIC)
        return (
            basis_codes(_values(basis.col_status), codes),
            basis_codes(_values(basis.row_status), codes),
        )

    def _warm(self, basis: Basis) -> None:
        """``setBasis``, a nonbasic ``fixed`` at ``kLower`` and ``superbasic`` at ``kZero``.

        highspy takes a basis only as a list of ``HighsBasisStatus`` members, so
        each status is picked from the five members by numpy rather than
        constructed one call at a time.
        """
        import highspy
        import numpy as np

        native = {BASIC: 1, AT_LOWER: 0, AT_UPPER: 2, FIXED: 0, SUPERBASIC: 3}
        members = np.array(
            [highspy.HighsBasisStatus(native[code]) for code in range(len(BASIS_STATUSES))], dtype=object
        )
        hint = highspy.HighsBasis()
        hint.col_status = members[basis.columns].tolist()
        hint.row_status = members[basis.rows].tolist()
        hint.valid = True
        _took(self._handle.setBasis(hint), 'the basis')

    def _start(self, values: Any) -> None:
        """``setSolution`` in its sparse form, which completes the columns it is not given, for an LP as for a mixed-integer model."""
        import numpy as np

        given = np.flatnonzero(~np.isnan(values)).astype(np.int32)
        _took(self._handle.setSolution(len(given), given, values[given]), 'the starting values')

    def _run(self, handoff: Handoff) -> SolveAnswer:
        """A ``kError`` from ``run()`` leaves the status unset, so on a quadratic model it is refused explicitly."""
        import highspy

        if self._handle.run() == highspy.HighsStatus.kError and handoff.quad.height:
            raise SpecsolveError(
                'the highs sink refused to run this quadratic objective, and a Hessian that is not '
                'positive semidefinite is why it refuses one: it solves convex QPs only. Convexity is a '
                "property of the coefficients' signs, which no capability table records, so it is "
                "found at the run — the sink's other quadratic refusal, a Hessian standing beside "
                'integrality, is read off the built model and caught before the load.\n'
                'Solve with a sink whose capabilities list a nonconvex quadratic objective as native, '
                'or write the model to an .lp file for a solver that '
                'takes one. A convex reformulation — the curve as a piecewise: block with '
                'method: convex — keeps the LP, and with it the duals and the warm start a quadratic '
                'objective gives up.'
            )
        status = _status_of(self._handle)
        if not status.is_readable:
            return self._unreadable(status)

        objective = self._handle.getInfo().objective_function_value + handoff.objective_constant
        solution = self._handle.getSolution()
        primal = solver_vector(solution.col_value)
        dual = solver_vector(solution.row_dual) if solution.dual_valid else None
        activity = solver_vector(solution.row_value)
        return SolveAnswer(status, objective, primal, dual, activity)

    def dual_ray(self) -> pl.Series | None:
        """``getDualRay``, signed the way the contract wants, whether or not presolve found the infeasibility."""
        _, has_ray, values = self._handle.getDualRay()
        return solver_vector(values) if has_ray else None

    def infeasible_subsystem(self) -> InfeasibleSubsystemIndices | None:
        """``getIis``, kept only where HiGHS proved it irreducible.

        A model that integrality alone makes infeasible gets one on 1.13 and
        none on 1.15.1, which searches without integrality.
        """
        import highspy
        import numpy as np

        _, found = self._handle.getIis()
        if not found.valid_ or found.status_ != _IIS_IRREDUCIBLE:
            return None
        bound = highspy.IisBoundStatus
        columns = np.asarray(found.col_index_, dtype=np.int64)
        sides = np.asarray(found.col_bound_, dtype=np.int64)
        return InfeasibleSubsystemIndices(
            np.asarray(found.row_index_, dtype=np.int64),
            columns[np.isin(sides, [int(bound.kIisBoundStatusLower), int(bound.kIisBoundStatusBoxed)])],
            columns[np.isin(sides, [int(bound.kIisBoundStatusUpper), int(bound.kIisBoundStatusBoxed)])],
        )

    def forget(self) -> None:
        """``clearSolver``: the basis and the solution go, the model stays."""
        self._handle.clearSolver()

    def close(self) -> None:
        if self._handle is not None:
            self._handle.clear()
        self._handle = None


def _row_bounds(rows: RowVectors, inf: float) -> tuple[Any, Any]:
    """HiGHS's ``(lower, upper)`` spelling of a sense code and right-hand side."""
    import numpy as np

    return (
        np.where(rows.sense == SENSE_CODES['<='], -inf, rows.rhs),
        np.where(rows.sense == SENSE_CODES['>='], inf, rows.rhs),
    )


def _loaded(h: Any, status: Any, what: str) -> None:
    """Raise unless HiGHS accepted the hand-off; it reports a refusal by return value alone."""
    import highspy

    if status == highspy.HighsStatus.kError:
        raise SpecsolveError(
            f'the solver refused {what}: {h.modelStatusToString(h.getModelStatus())!r}. '
            f'The model it holds is not the one handed over, so any answer would describe a '
            f'different one. This is an engine bug rather than a problem with the model — '
            f'please report it.'
        )


def _took(status: Any, what: str) -> None:
    """Raise unless HiGHS accepted a warm-start hint; it reports a refusal by return value alone."""
    import highspy

    if status == highspy.HighsStatus.kError:
        raise SpecsolveError(
            f'HiGHS refused {what} even though it spans the loaded model, so the solve would '
            f'silently start cold instead of warm. This is an engine bug rather than a problem '
            f'with the model — please report it.'
        )


def _status_of(h: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``has_primal`` is asked separately: a run stopped at a limit may or may not
    hold an incumbent.
    """
    model_status = h.getModelStatus()
    return SolveStatus(
        termination_condition=_CONDITION_OF_HIGHS_STATUS.get(str(model_status).rsplit('.', 1)[-1], 'unknown'),
        solver_wording=h.modelStatusToString(model_status),
        has_primal=_has_primal(h),
    )


def _has_primal(h: Any) -> bool:
    """Whether HiGHS holds a feasible primal."""
    import highspy

    return h.getInfo().primal_solution_status == int(highspy.SolutionStatus.kSolutionStatusFeasible)


def _values(statuses: list[Any]) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
    """The integer behind each ``HighsBasisStatus``: highspy hands a basis back only as a list of members."""
    import numpy as np

    return np.fromiter((status.value for status in statuses), dtype=np.int64, count=len(statuses))
