"""The ``gurobi`` solver, loading the model straight into gurobipy.

``gurobipy`` and ``scipy`` are imported inside the functions, so importing
this module needs neither.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any

from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.solvers.base import (
    InfeasibleSubsystemIndices,
    SolveAnswer,
    Solver,
    WarmStart,
    solver_vector,
    spelled_senses,
)
from specsolve.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    import polars as pl

    from specsolve.relational.sinks.handoff import Handoff, RowVectors


#: Gurobi status -> termination condition: linopy's ``Gurobi.CONDITION_MAP``
#: except [`_LINOPY_DIVERGENCES`][].
_CONDITION_OF_GUROBI_STATUS = {
    1: 'unknown',
    2: 'optimal',
    3: 'infeasible',
    4: 'infeasible_or_unbounded',
    5: 'unbounded',
    6: 'other',
    7: 'iteration_limit',
    8: 'terminated_by_limit',
    9: 'time_limit',
    10: 'terminated_by_limit',
    11: 'user_interrupt',
    12: 'other',
    13: 'suboptimal',
    14: 'unknown',
    15: 'terminated_by_limit',
    16: 'terminated_by_limit',
    17: 'resource_interrupt',
}

#: Where the table above departs from linopy's, and why.
_LINOPY_DIVERGENCES = {
    10: 'SOLUTION_LIMIT stopped early after n incumbents; linopy calls it optimal',
    16: 'WORK_LIMIT is a limit, not a solver failure; linopy calls it internal_solver_error',
    17: 'MEM_LIMIT is the resource_interrupt linopy itself maps kMemoryLimit to on HiGHS',
}


class Gurobi(Solver):
    """Gurobi, holding one model at ``.handle``, a `gurobipy.Model`.

    ``batch_rows`` is a nonzero budget that splits the matrix across calls;
    ``None`` is one call. ``close``, or leaving a ``with``, releases the model
    and its environment.
    """

    #: The loaded model, the handles that read it back, and the environment.
    #: ``close`` drops them.
    _m: Any
    _x: Any
    _blocks: list[Any]
    #: [`_released`][] over the model and its environment, bound to this
    #: holder's lifetime.
    _release: weakref.finalize[[Any, Any], Gurobi]
    #: The quadratic constraints, in row order, after every linear one.
    _qrows: list[Any]
    _env: Any

    requires = ('gurobipy', 'scipy.sparse')
    recorded_options = frozenset(
        {
            'timelimit',
            'mipgap',
            'mipgapabs',
            'seed',
            'threads',
            'method',
            'presolve',
            'feasibilitytol',
            'optimalitytol',
            'intfeastol',
            'numericfocus',
        }
    )
    unavailable_message = (
        'The gurobi sink requires the [gurobi] extra (gurobipy, scipy): pip install "specsolve[gurobi]"'
    )

    #: The only sink with no quadratic exclusion, as
    #: ``tests/test_gurobi_capability_probes.py`` measures.
    capabilities = Capabilities(
        supports=frozenset(
            {'integrality', 'sos', 'quadratic_objective', 'nonconvex_quadratic_objective', 'quadratic_constraint'}
        )
    )

    def _load(self, handoff: Handoff, batch_rows: int | None) -> None:
        self._m, self._x, self._blocks, self._qrows, self._env = _built(handoff, batch_rows, self._options)
        self._release = weakref.finalize(self, _released, self._m, self._env)

    def dual_ray(self) -> pl.Series | None:
        """``FarkasDual``, negated to the contract's sign, or ``None``.

        Gurobi computes it only under ``solver_options={'InfUnbdInfo': 1}``.
        A model with a quadratic row has none.
        """
        import numpy as np

        gurobipy = _gurobipy()
        if self._qrows:
            return None
        try:
            slices = [block.FarkasDual for block in self._blocks]
        except (AttributeError, gurobipy.GurobiError):
            return None
        values = np.concatenate(slices) if slices else np.empty(0, dtype=np.float64)
        return solver_vector(-values)

    @property
    def handle(self) -> Any:
        return self._m

    def push(self, handoff: Handoff) -> None:
        """Whole vectors, one call per block."""
        gurobipy = _gurobipy()
        cols = handoff.dense_columns(gurobipy.GRB.INFINITY)
        self._x.LB, self._x.UB, self._x.Obj = cols.lb, cols.ub, cols.cost

        rhs = handoff.dense_rows(gurobipy.GRB.INFINITY).rhs
        for block, rows in self._per_block(rhs):
            block.RHS = rows
        self._m.ObjCon = handoff.objective_constant
        _set_quadratic(self._m, self._x, handoff, cols.cost)
        self._m.update()

    def warm_start(self) -> WarmStart | None:
        """The basis the last solve left, else its incumbent, else ``None``; Gurobi refuses ``VBasis`` without one."""
        import numpy as np

        gurobipy = _gurobipy()
        try:
            columns = np.asarray(self._x.VBasis, dtype=np.int32)
            slices = [np.asarray(block.CBasis, dtype=np.int32) for block in self._blocks]
        except (AttributeError, gurobipy.GurobiError):
            if self._m.SolCount > 0:
                values = np.asarray(self._x.X, dtype=np.float64)
                return WarmStart(solver='gurobi', column_statuses=None, row_statuses=None, column_values=values)
            return None
        rows = np.concatenate(slices) if slices else np.empty(0, dtype=np.int32)
        return WarmStart(solver='gurobi', column_statuses=columns, row_statuses=rows, column_values=None)

    def _warm(self, ws: WarmStart) -> None:
        """``VBasis``/``CBasis`` for a basis, ``Start`` for an incumbent."""
        if (basis := ws.basis()) is not None:
            column_statuses, row_statuses = basis
            self._x.VBasis = column_statuses
            for block, rows in self._per_block(row_statuses):
                block.CBasis = rows
        else:
            assert ws.column_values is not None, (
                'a warm start with no basis carries an incumbent — it holds nothing else'
            )
            self._x.Start = ws.column_values
        self._m.update()

    def _per_block(self, vector: Any) -> Iterator[tuple[Any, Any]]:
        """Each linear constraint block with its slice of a row vector. The blocks ascend by row."""
        at = 0
        for block in self._blocks:
            yield block, vector[at : at + block.shape[0]]
            at += block.shape[0]

    def _run(self, handoff: Handoff) -> SolveAnswer:
        """The one ``GurobiError`` translated is a caller's ``QCPDual`` on a nonconvex quadratic constraint."""
        gurobipy = _gurobipy()
        try:
            self._m.optimize()
        except gurobipy.GurobiError as exc:
            if 'not PSD' not in str(exc):
                raise
            raise SpecsolveError(
                f'this model has a quadratic constraint that is not convex, and the solve was asked for '
                f'quadratic duals (QCPDual), which only a convex model has. Gurobi reported: {exc}\n'
                f'Drop QCPDual from solver_options to solve it — the answer comes back without prices '
                f'for the quadratic rows, which is the default for exactly this reason.'
            ) from None
        status = _status_of(self._m)
        if not status.is_readable:
            return self._unreadable(status)
        return SolveAnswer(
            status,
            self._m.ObjVal,
            solver_vector(self._x.X),
            _duals(self._blocks, self._qrows),
            _activity(self._blocks, self._qrows),
        )

    def infeasible_subsystem(self) -> InfeasibleSubsystemIndices | None:
        """``computeIIS``, kept only where Gurobi reports it minimal.

        A search a limit stops leaves a set that is not, or nothing readable.
        Read back per block and per quadratic row, in row order.
        """
        import numpy as np

        gurobipy = _gurobipy()
        self._m.computeIIS()
        try:
            if not self._m.IISMinimal:
                return None
        except (AttributeError, gurobipy.GurobiError):
            return None
        slices = [np.asarray(block.IISConstr, dtype=bool) for block in self._blocks]
        slices += [np.asarray([row.IISQConstr], dtype=bool) for row in self._qrows]
        rows = np.concatenate(slices) if slices else np.empty(0, dtype=bool)
        return InfeasibleSubsystemIndices(
            np.flatnonzero(rows),
            np.flatnonzero(np.asarray(self._x.IISLB, dtype=bool)),
            np.flatnonzero(np.asarray(self._x.IISUB, dtype=bool)),
        )

    def forget(self) -> None:
        """``Model.reset``: the solution and the basis go; the model and its parameters stay."""
        self._m.reset()

    def close(self) -> None:
        """Release the model and the licence its environment holds."""
        self._release()
        self._m = self._x = self._env = None
        self._blocks = []
        self._qrows = []


def _released(m: Any, environment: Any) -> None:
    """Dispose the model, then its environment, which Gurobi keeps alive while a model on it lives."""
    m.dispose()
    environment.dispose()


def _built(
    handoff: Handoff,
    batch_rows: int | None,
    solver_options: Mapping[str, Any] | None,
) -> tuple[Any, Any, list[Any], list[Any], Any]:
    """The model, the handles to read it back, and the environment to release.

    Options go on the environment, since a licence parameter such as
    ``WLSAccessID`` can only be set before an environment starts. ``OutputFlag``
    leads so a caller can put the log back.
    """
    gurobipy = _gurobipy()
    environment = gurobipy.Env(params={'OutputFlag': 0, **dict(solver_options or {})})
    m = gurobipy.Model(env=environment)
    try:
        return m, *_filled(m, handoff, batch_rows, gurobipy), environment
    except BaseException:
        _released(m, environment)
        raise


def _filled(m: Any, handoff: Handoff, batch_rows: int | None, gurobipy: Any) -> tuple[Any, list[Any], list[Any]]:
    """Everything [`_built`][] loads after the environment exists."""
    import numpy as np
    import scipy.sparse

    cols = handoff.dense_columns(gurobipy.GRB.INFINITY)
    discrete: dict[str, Any] = {'vtype': np.where(cols.integral, 'I', 'C')} if cols.integral.any() else {}
    x = m.addMVar(handoff.column_count, lb=cols.lb, ub=cols.ub, obj=cols.cost, **discrete)

    rows = handoff.dense_rows(gurobipy.GRB.INFINITY)
    spelling = _spelled(gurobipy)
    blocks = []
    for chunk in handoff.row_blocks(batch_rows):
        entries = chunk.entries
        block = scipy.sparse.csr_matrix(
            (entries['coeff'].to_numpy(), entries['col'].to_numpy(), np.append(chunk.starts, entries.height)),
            shape=(chunk.height, handoff.column_count),
        )
        blocks.append(m.addMConstr(block, x, spelling[rows.sense[chunk.lo : chunk.hi]], rows.rhs[chunk.lo : chunk.hi]))

    _add_sets(m, x, handoff, gurobipy)
    quadratic = _add_quadratic_rows(m, x, handoff, rows, spelling)
    if handoff.objective_sense == 'maximize':
        m.ModelSense = gurobipy.GRB.MAXIMIZE
    m.ObjCon = handoff.objective_constant
    _set_quadratic(m, x, handoff, cols.cost)
    m.update()
    return x, blocks, quadratic


def _add_quadratic_rows(m: Any, x: Any, handoff: Handoff, rows: RowVectors, spelling: Any) -> list[Any]:
    """Every quadratic constraint, one ``addMQConstr`` call each.

    A row takes its quadratic entries unhalved, as [`_set_quadratic`][] does,
    and its linear entries from the matrix at the same row label.
    """
    import numpy as np
    import scipy.sparse

    added = []
    for row, pairs in handoff.quadratic_blocks():
        quadratic = scipy.sparse.csr_matrix(
            (pairs['coeff'].to_numpy(), (pairs['col_l'].to_numpy(), pairs['col_r'].to_numpy())),
            shape=(handoff.column_count, handoff.column_count),
        )
        entries = handoff.matrix_block(row, row + 1)
        linear = np.zeros(handoff.column_count, dtype=np.float64)
        linear[entries['col'].to_numpy()] = entries['coeff'].to_numpy()
        added.append(m.addMQConstr(quadratic, linear, spelling[rows.sense[row]], float(rows.rhs[row]), x, x, x))
    return added


def _set_quadratic(m: Any, x: Any, handoff: Handoff, cost: Any) -> None:
    r"""The objective's quadratic part, as the matrix Gurobi reads.

    ``setMObjective`` takes :math:`Q` in :math:`x^\top Q x` unhalved, so
    [`quad`][specsolve.relational.sinks.handoff.Handoff.quad] goes in as it
    stands. It sets the whole objective, so *cost* is passed again.
    """
    import scipy.sparse

    if not handoff.quad.height:
        return
    pairs = scipy.sparse.csr_matrix(
        (
            handoff.quad['coeff'].to_numpy(),
            (handoff.quad['col_l'].to_numpy(), handoff.quad['col_r'].to_numpy()),
        ),
        shape=(handoff.column_count, handoff.column_count),
    )
    m.setMObjective(pairs, cost, handoff.objective_constant, x, x, x)


def _add_sets(m: Any, x: Any, handoff: Handoff, gurobipy: Any) -> None:
    """Every special-ordered set, one ``addSOS`` call each."""
    if not handoff.sos.height:
        return
    order = {1: gurobipy.GRB.SOS_TYPE1, 2: gurobipy.GRB.SOS_TYPE2}
    columns = x.tolist()
    for set_type, cols, weights in handoff.sets():
        m.addSOS(order[set_type], [columns[at] for at in cols], weights.to_list())


#: Each comparison's ``GRB`` attribute name, since ``gurobipy`` is imported lazily.
_GUROBI_SENSE = {'<=': 'LESS_EQUAL', '>=': 'GREATER_EQUAL', '==': 'EQUAL'}


def _spelled(gurobipy: Any) -> Any:
    """The sense codes as the characters ``addMConstr`` wants, by code."""
    return spelled_senses({sense: getattr(gurobipy.GRB, name) for sense, name in _GUROBI_SENSE.items()})


def _gurobipy() -> Any:
    """The optional dependency — scipy guarded with it — or [`Gurobi.unavailable_message`][]."""
    return Gurobi.imported()


def _status_of(m: Any) -> SolveStatus:
    """What the solve concluded, with ``SolCount`` deciding whether a primal exists."""
    code = int(m.Status)
    return SolveStatus(
        termination_condition=_CONDITION_OF_GUROBI_STATUS.get(code, 'unknown'),
        solver_wording=_wording(code),
        has_primal=m.SolCount > 0,
    )


def _wording(code: int) -> str:
    """Gurobi's own name for a status code, read off ``GRB.Status`` so an unlisted one is named too."""
    gurobipy = _gurobipy()
    names = {getattr(gurobipy.GRB.Status, name): name for name in dir(gurobipy.GRB.Status) if not name.startswith('_')}
    return names.get(code, str(code))


def _activity(blocks: list[Any], qrows: list[Any]) -> pl.Series:
    r"""Each row's left-hand side at the solution, in row order.

    Gurobi exposes only ``Slack``, which is ``rhs - activity`` for every sense.
    ``QCSlack`` covers a quadratic row's whole left-hand side,
    :math:`x^\top Q x + a^\top x`.
    """
    import numpy as np

    slices = [block.RHS - block.Slack for block in blocks]
    slices += [np.asarray([row.QCRHS - row.QCSlack], dtype=np.float64) for row in qrows]
    values = np.concatenate(slices) if slices else np.empty(0, dtype=np.float64)
    return solver_vector(values)


def _duals(blocks: list[Any], qrows: list[Any]) -> pl.Series | None:
    """Shadow prices in row order, or ``None`` where Gurobi refuses them.

    Gurobi refuses ``Pi`` on a mixed-integer model, and ``QCPi`` unless
    ``solver_options={'QCPDual': 1}``.
    """
    import numpy as np

    gurobipy = _gurobipy()
    try:
        slices = [block.Pi for block in blocks]
        slices += [np.asarray([row.QCPi], dtype=np.float64) for row in qrows]
    except (AttributeError, gurobipy.GurobiError):
        return None
    values = np.concatenate(slices) if slices else np.empty(0, dtype=np.float64)
    return solver_vector(values)
