"""The ``xpress`` solver: the model in two calls, straight into the Optimizer.

The same hand-off as [`highs`][specsolve.relational.sinks.solvers.highs], reading the
same ``dense_columns``, ``dense_rows`` and ``row_blocks``. The objective's
constant is the negated objective coefficient of column ``-1``. ``xpress`` is
imported inside the functions, so importing this module stays free for a
caller who never solves with it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.solvers.base import SolveAnswer, Solver, WarmStart, solver_vector, spelled_senses
from specsolve.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from specsolve.relational.sinks.handoff import Handoff


#: Xpress solution status -> termination condition, copied from linopy's
#: ``Xpress.CONDITION_MAP``; ``tests/test_solve_status.py`` asserts the copy.
#: Keyed by value, since the enum is an optional import.
_CONDITION_OF_SOL_STATUS = {
    0: 'unknown',
    1: 'optimal',
    2: 'terminated_by_limit',
    3: 'infeasible',
    4: 'unbounded',
}

#: ``OPTIMAL`` and ``FEASIBLE``, by value: the solution statuses that carry values.
_HAS_PRIMAL = frozenset({1, 2})

#: ``SolveStatus.UNSTARTED`` and ``SolveStatus.FAILED``, by value: whether there
#: has been a run, and whether it errored, which ``solstatus`` cannot say.
_SOLVE_UNSTARTED = 0
_SOLVE_FAILED = 2


def build_xpress(
    handoff: Handoff,
    batch_rows: int | None = None,
    solver_options: Mapping[str, Any] | None = None,
) -> Xpress:
    """Load the model into an `xpress.problem` and stop there: the seam `bench/` measures.

    Returns:
        The [`Xpress`][] holding the problem, at ``.handle``. The problem
        owns its licence and releases it when it is collected.
    """
    return Xpress(handoff, batch_rows, solver_options)


class Xpress(Solver):
    """FICO Xpress, holding one model.

    A push writes bounds, costs and right-hand sides by index. Duals are
    ``None`` rather than zero-filled on a model that has none.
    """

    #: The loaded problem; ``close`` drops it, and the licence with it.
    _p: Any

    #: One package, and it carries its own solver library.
    requires = ('xpress',)
    recorded_options = frozenset(
        {
            'timelimit',
            'miprelstop',
            'mipabsstop',
            'randomseed',
            'threads',
            'defaultalg',
            'presolve',
            'feastol',
            'optimalitytol',
            'miptol',
        }
    )
    unavailable_message = 'The xpress sink requires the [xpress] extra: pip install "specsolve[xpress]"'

    #: Xpress branches on a set natively. The Optimizer takes a Hessian; this
    #: sink does not hand it one.
    capabilities = Capabilities(supports=frozenset({'integrality', 'sos'}))

    def _load(self, handoff: Handoff, batch_rows: int | None) -> None:
        self._p = _built(handoff, batch_rows, self._options)

    @property
    def handle(self) -> Any:
        return self._p

    def push(self, handoff: Handoff) -> None:
        """Whole vectors by index, in three calls; ``chgBounds`` takes both bounds, a letter per entry."""
        import numpy as np

        xpress = _xpress()
        cols = handoff.dense_columns(xpress.infinity)
        every = np.arange(handoff.column_count, dtype=np.int64)
        self._p.chgBounds(
            np.concatenate([every, every]),
            ['L'] * handoff.column_count + ['U'] * handoff.column_count,
            np.concatenate([cols.lb, cols.ub]),
        )
        self._p.chgObj(np.append(every, -1), np.append(cols.cost, -handoff.objective_constant))
        self._p.chgRHS(np.arange(handoff.row_count, dtype=np.int64), handoff.dense_rows(xpress.infinity).rhs)

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, its incumbent after a MIP, or ``None``.

        Xpress hands back a trivial all-slack basis before any solve and after a
        MIP, so both are asked of the problem directly. ``getBasis`` returns
        ``(rows, columns)``, the opposite of [`WarmStart`][]'s order.
        """
        import numpy as np

        if int(self._p.attributes.solvestatus) == _SOLVE_UNSTARTED:
            return None
        if int(self._p.attributes.mipents):
            if int(self._p.attributes.solstatus) not in _HAS_PRIMAL:
                return None
            values = np.asarray(self._p.getSolution(), dtype=np.float64)
            return WarmStart(solver='xpress', column_statuses=None, row_statuses=None, column_values=values)
        rows, columns = self._p.getBasis()
        return WarmStart(
            solver='xpress',
            column_statuses=np.asarray(columns, dtype=np.int32),
            row_statuses=np.asarray(rows, dtype=np.int32),
            column_values=None,
        )

    def _warm(self, ws: WarmStart) -> None:
        """``loadBasis`` for a basis, ``addMipSol`` for an incumbent; a basis turns ``keepbasis`` back on."""
        if (basis := ws.basis()) is not None:
            column_statuses, row_statuses = basis
            self._p.controls.keepbasis = 1
            self._p.loadBasis(row_statuses, column_statuses)
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            self._p.addMipSol(ws.column_values)

    def _run(self, handoff: Handoff) -> SolveAnswer:
        """Solve what is loaded and read it back; the objective constant is already in the model."""
        self._p.optimize()
        status = _status_of(self._p)
        if not status.is_readable:
            return self._unreadable(status)
        return SolveAnswer(
            status,
            float(self._p.attributes.objval),
            solver_vector(self._p.getSolution()),
            _duals(self._p),
            _activity(self._p),
        )

    def dual_ray(self) -> pl.Series | None:
        """``getDualRay``, signed the way the contract wants.

        ``None`` where presolve found the infeasibility, which is the default;
        ``solver_options={'presolve': 0}`` gets a ray.
        """
        values = self._p.getDualRay()
        return None if values is None else solver_vector(values)

    def forget(self) -> None:
        """``keepbasis = 0``: the next solve ignores the basis this one left.

        ``problem.reset()`` would clear the model too. The control is durable;
        [`_warm`][] turns it back on.
        """
        self._p.controls.keepbasis = 0

    def close(self) -> None:
        """Release the problem, and the licence it holds."""
        if self._p is not None:
            self._p.reset()
            self._p = None


def _built(
    handoff: Handoff,
    batch_rows: int | None,
    solver_options: Mapping[str, Any] | None,
) -> Any:
    """The loaded problem: columns with no entries, then the matrix a row block at a time.

    ``outputlog`` leads the controls so a caller's options can put the log back.
    """
    import numpy as np

    xpress = _xpress()
    p = xpress.problem()
    p.setControl({'outputlog': 0, **dict(solver_options or {})})

    cols = handoff.dense_columns(xpress.infinity)
    p.addCols(
        objcoef=cols.cost,
        start=np.zeros(handoff.column_count + 1, dtype=np.int64),
        rowind=np.empty(0, dtype=np.int64),
        rowcoef=np.empty(0, dtype=np.float64),
        lb=cols.lb,
        ub=cols.ub,
    )
    if cols.integral.any():
        integral = np.flatnonzero(cols.integral)
        p.chgColType(integral, ['I'] * integral.size)

    rows = handoff.dense_rows(xpress.infinity)
    spelling = spelled_senses(_XPRESS_SENSE)
    for chunk in handoff.row_blocks(batch_rows):
        entries = chunk.entries
        p.addRows(
            rowtype=spelling[rows.sense[chunk.lo : chunk.hi]].tolist(),
            rhs=rows.rhs[chunk.lo : chunk.hi],
            start=np.append(chunk.starts, entries.height),
            colind=entries['col'].to_numpy(),
            rowcoef=entries['coeff'].to_numpy(),
        )

    _add_sets(p, handoff, xpress)
    if handoff.objective_sense == 'maximize':
        p.chgObjSense(xpress.maximize)
    if handoff.objective_constant:
        p.chgObj([-1], [-handoff.objective_constant])
    return p


def _add_sets(p: Any, handoff: Handoff, xpress: Any) -> None:
    """Every special-ordered set, one ``addSOS`` call each: there is no bulk form."""
    if not handoff.sos.height:
        return
    for set_type, cols, weights in handoff.sets():
        p.addSOS(cols.to_list(), weights.cast(float).to_list(), type=set_type)


#: Each comparison as an Optimizer row type.
_XPRESS_SENSE = {'<=': 'L', '>=': 'G', '==': 'E'}


def _xpress() -> Any:
    """The optional dependency, or [`Xpress.unavailable_message`][]."""
    return Xpress.imported()


def _status_of(p: Any) -> SolveStatus:
    """What the solve concluded, on both axes.

    ``solvestatus`` separates a solve that errored, reported as
    ``internal_solver_error``, from one that found nothing; ``solstatus`` alone
    cannot.
    """
    solution = int(p.attributes.solstatus)
    if int(p.attributes.solvestatus) == _SOLVE_FAILED:
        return SolveStatus('internal_solver_error', _wording(solution), has_primal=False)
    return SolveStatus(
        termination_condition=_CONDITION_OF_SOL_STATUS.get(solution, 'unknown'),
        solver_wording=_wording(solution),
        has_primal=solution in _HAS_PRIMAL,
    )


def _wording(solution: int) -> str:
    """Xpress's own name for a solution status, read off the enum so an unknown one still arrives."""
    xpress = _xpress()
    names = {int(member): member.name for member in xpress.SolStatus}
    return names.get(solution, str(solution))


def _activity(p: Any) -> pl.Series:
    """Each row's left-hand side at the solution, as ``rhs - slack``: Xpress exposes only the slack."""
    import numpy as np

    slack = np.asarray(p.getSlacks(), dtype=np.float64)
    rhs = np.asarray(p.getRHS(), dtype=np.float64)
    return solver_vector(rhs - slack)


def _duals(p: Any) -> pl.Series | None:
    """Shadow prices in row order, or ``None`` on a mixed-integer model, where Xpress refuses ``getDuals``."""
    xpress = _xpress()
    try:
        return solver_vector(p.getDuals())
    except (xpress.SolverError, xpress.ModelError):
        return None
