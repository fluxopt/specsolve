"""The ``gurobi`` solver, loading the model straight into gurobipy.

``gurobipy`` and ``scipy`` are imported inside the functions, so importing
this module needs neither.
"""

from __future__ import annotations

import weakref
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from specsolve.errors import SpecsolveError
from specsolve.relational.answer_layout import AT_LOWER, AT_UPPER, BASIC, FIXED, SUPERBASIC
from specsolve.relational.sinks.capabilities import Capabilities
from specsolve.relational.sinks.solvers.base import (
    Basis,
    InfeasibleSubsystemIndices,
    SolveAnswer,
    Solver,
    basis_codes,
    solver_codes,
    solver_vector,
    spelled_senses,
)
from specsolve.relational.status import SolveStatus

if TYPE_CHECKING:
    from collections.abc import Mapping

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

    The model loads in one call, so ``batch_rows`` has no effect. ``close``, or
    leaving a ``with``, releases the model and its environment.
    """

    #: The loaded model and its environment. ``close`` drops them.
    _m: Any
    _env: Any
    #: The columns as one ``MVar``, or ``None`` until [`_mvar`][] is asked.
    _x: Any
    #: [`_released`][] over the model and its environment, bound to this
    #: holder's lifetime.
    _release: weakref.finalize[[Any, Any], Gurobi]

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
    lp_values = MappingProxyType({'complete': 'no_gain', 'partial': 'no_gain'})
    capabilities = Capabilities(
        supports=frozenset(
            {'integrality', 'sos', 'quadratic_objective', 'nonconvex_quadratic_objective', 'quadratic_constraint'}
        )
    )

    def _load(self, handoff: Handoff, batch_rows: int | None) -> None:
        """Load in one call — *batch_rows* is the family's parameter and this member has no batches."""
        del batch_rows
        self._m, self._x, self._env = _built(handoff, self._options)
        self._release = weakref.finalize(self, _released, self._m, self._env)

    def _mvar(self) -> Any:
        """The columns as one ``MVar``, read off the model on first need.

        Only a quadratic objective, a quadratic row and an SOS take one.
        ``getVars`` makes one Python object per column, which costs more than
        the whole load of a large linear model, so nothing else asks.
        """
        if self._x is None:
            self._x = _gurobipy().MVar.fromlist(self._m.getVars())
        return self._x

    def dual_ray(self) -> pl.Series | None:
        """``FarkasDual``, negated to the contract's sign, or ``None``.

        Gurobi computes it only under ``solver_options={'InfUnbdInfo': 1}``.
        A model with a quadratic row has none.
        """
        import numpy as np

        gurobipy = _gurobipy()
        if self._m.NumQConstrs:
            return None
        try:
            values = np.asarray(self._m.getAttr('FarkasDual'), dtype=np.float64)
        except (AttributeError, gurobipy.GurobiError):
            return None
        return solver_vector(-values)

    @property
    def handle(self) -> Any:
        return self._m

    def push(self, handoff: Handoff) -> None:
        """Whole vectors, one call each."""
        gurobipy = _gurobipy()
        cols = handoff.dense_columns(gurobipy.GRB.INFINITY)
        for attribute, values in (('LB', cols.lb), ('UB', cols.ub), ('Obj', cols.cost)):
            self._m.setAttr(attribute, values)
        self._m.setAttr('RHS', handoff.dense_rows(gurobipy.GRB.INFINITY).rhs[: self._m.NumConstrs])
        self._m.ObjCon = handoff.objective_constant
        if handoff.quad.height:
            _set_quadratic(self._m, self._mvar(), handoff, cols.cost)
        self._m.update()

    def _basis(self) -> tuple[Any, Any] | None:
        """``VBasis`` and ``CBasis``, which Gurobi refuses where it holds no basis; each status is ``0`` or below, so negated it indexes."""
        import numpy as np

        gurobipy = _gurobipy()
        try:
            columns = np.asarray(self._m.getAttr('VBasis'), dtype=np.int64)
            rows = np.asarray(self._m.getAttr('CBasis'), dtype=np.int64)
        except (AttributeError, gurobipy.GurobiError):
            return None
        return basis_codes(-columns, (BASIC, AT_LOWER, AT_UPPER, SUPERBASIC)), basis_codes(-rows, (BASIC, AT_LOWER))

    def _warm(self, basis: Basis) -> None:
        """``VBasis`` and ``CBasis``; a row is ``0`` basic or ``-1`` not, whichever bound it is at."""
        self._m.setAttr(
            'VBasis', solver_codes(basis.columns, {BASIC: 0, AT_LOWER: -1, AT_UPPER: -2, FIXED: -1, SUPERBASIC: -3})
        )
        rows = solver_codes(basis.rows, {BASIC: 0, AT_LOWER: -1, AT_UPPER: -1, FIXED: -1, SUPERBASIC: -1})
        self._m.setAttr('CBasis', rows[: self._m.NumConstrs])
        self._m.update()

    def _start(self, values: Any) -> None:
        """``Start`` for a mixed-integer model and ``PStart`` for an LP, ``GRB.UNDEFINED`` where no value is given."""
        import numpy as np

        given = np.where(np.isnan(values), _gurobipy().GRB.UNDEFINED, values)
        self._m.setAttr('Start' if self._m.IsMIP else 'PStart', given)
        self._m.update()

    def _run(self, handoff: Handoff) -> SolveAnswer:
        """The one ``GurobiError`` translated is a caller's ``QCPDual`` on a nonconvex quadratic constraint."""
        import numpy as np

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
            solver_vector(np.asarray(self._m.getAttr('X'), dtype=np.float64)),
            _duals(self._m),
            _activity(self._m),
        )

    def infeasible_subsystem(self) -> InfeasibleSubsystemIndices | None:
        """``computeIIS``, kept only where Gurobi reports it minimal.

        A search a limit stops leaves a set that is not, or nothing readable.
        The quadratic rows follow the linear ones.
        """
        import numpy as np

        gurobipy = _gurobipy()
        self._m.computeIIS()
        try:
            if not self._m.IISMinimal:
                return None
        except (AttributeError, gurobipy.GurobiError):
            return None
        rows = np.concatenate([_flags(self._m, 'IISConstr'), _flags(self._m, 'IISQConstr')])
        return InfeasibleSubsystemIndices(
            np.flatnonzero(rows),
            np.flatnonzero(_flags(self._m, 'IISLB')),
            np.flatnonzero(_flags(self._m, 'IISUB')),
        )

    def forget(self) -> None:
        """``Model.reset``: the solution and the basis go; the model and its parameters stay."""
        self._m.reset()

    def close(self) -> None:
        """Release the model and the licence its environment holds."""
        self._release()
        self._m = self._x = self._env = None


def _released(m: Any, environment: Any) -> None:
    """Dispose the model, then its environment, which Gurobi keeps alive while a model on it lives."""
    m.dispose()
    environment.dispose()


def _built(handoff: Handoff, solver_options: Mapping[str, Any] | None) -> tuple[Any, Any, Any]:
    """The model, its columns as an ``MVar`` if the load needed one, and the environment to release.

    Options go on the environment, since a licence parameter such as
    ``WLSAccessID`` can only be set before an environment starts. ``OutputFlag``
    leads so a caller can put the log back.
    """
    gurobipy = _gurobipy()
    environment = gurobipy.Env(params={'OutputFlag': 0, **dict(solver_options or {})})
    m = None
    try:
        m = _loaded(handoff, environment, gurobipy)
        return m, _completed(m, handoff, gurobipy), environment
    except BaseException:
        if m is not None:
            m.dispose()
        environment.dispose()
        raise


def _loaded(handoff: Handoff, environment: Any, gurobipy: Any) -> Any:
    """Columns, linear rows and the objective in one ``loadModel`` call.

    ``loadModel`` reads the matrix by column, so the linear rows' CSR is
    transposed here. Its ``qobj_coo`` takes each pair's coefficient whole, as
    [`quad`][specsolve.relational.sinks.handoff.Handoff.quad] holds it.
    """
    import numpy as np
    import scipy.sparse

    cols = handoff.dense_columns(gurobipy.GRB.INFINITY)
    rows = handoff.dense_rows(gurobipy.GRB.INFINITY)
    (linear,) = handoff.row_blocks(None)
    entries = linear.entries
    matrix = scipy.sparse.csr_matrix(
        (entries['coeff'].to_numpy(), entries['col'].to_numpy(), np.append(linear.starts, entries.height)),
        shape=(linear.height, handoff.column_count),
    ).tocsc()
    quad = handoff.quad
    m = gurobipy.loadModel(
        env=environment,
        numvars=handoff.column_count,
        numconstrs=linear.height,
        modelsense=gurobipy.GRB.MAXIMIZE if handoff.objective_sense == 'maximize' else gurobipy.GRB.MINIMIZE,
        objcon=handoff.objective_constant,
        obj=cols.cost,
        lb=cols.lb,
        ub=cols.ub,
        vtype=np.where(cols.integral, 'I', 'C') if cols.integral.any() else None,
        constr_csc=(matrix.data, matrix.indices, matrix.indptr),
        sense=_spelled(gurobipy)[rows.sense[: linear.height]],
        rhs=rows.rhs[: linear.height],
        qobj_coo=(
            (quad['coeff'].to_numpy(), (quad['col_l'].to_numpy(), quad['col_r'].to_numpy())) if quad.height else None
        ),
    )
    m.update()
    return m


def _completed(m: Any, handoff: Handoff, gurobipy: Any) -> Any:
    """What ``loadModel`` cannot take: the SOS and the quadratic rows. Returns the ``MVar`` they needed, or ``None``."""
    if not handoff.sos.height and not handoff.qmatrix.height:
        return None
    x = gurobipy.MVar.fromlist(m.getVars())
    _add_sets(m, x, handoff, gurobipy)
    _add_quadratic_rows(m, x, handoff, handoff.dense_rows(gurobipy.GRB.INFINITY), _spelled(gurobipy))
    m.update()
    return x


def _add_quadratic_rows(m: Any, x: Any, handoff: Handoff, rows: RowVectors, spelling: Any) -> None:
    """Every quadratic constraint, one ``addMQConstr`` call each.

    A row takes its quadratic entries unhalved, as [`_set_quadratic`][] does,
    and its linear entries from the matrix at the same row label.
    """
    import numpy as np
    import scipy.sparse

    for row, pairs in handoff.quadratic_blocks():
        quadratic = scipy.sparse.csr_matrix(
            (pairs['coeff'].to_numpy(), (pairs['col_l'].to_numpy(), pairs['col_r'].to_numpy())),
            shape=(handoff.column_count, handoff.column_count),
        )
        entries = handoff.matrix_block(row, row + 1)
        linear = np.zeros(handoff.column_count, dtype=np.float64)
        linear[entries['col'].to_numpy()] = entries['coeff'].to_numpy()
        m.addMQConstr(quadratic, linear, spelling[rows.sense[row]], float(rows.rhs[row]), x, x, x)


def _set_quadratic(m: Any, x: Any, handoff: Handoff, cost: Any) -> None:
    r"""The objective's quadratic part, as the matrix Gurobi reads.

    ``setMObjective`` takes :math:`Q` in :math:`x^\top Q x` unhalved, so
    [`quad`][specsolve.relational.sinks.handoff.Handoff.quad] goes in as it
    stands. It sets the whole objective, so *cost* is passed again.
    """
    import scipy.sparse

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


def _flags(m: Any, attribute: str) -> Any:
    """One per-element attribute over every element of its kind, as booleans."""
    import numpy as np

    return np.asarray(m.getAttr(attribute), dtype=bool)


def _activity(m: Any) -> pl.Series:
    r"""Each row's left-hand side at the solution, in row order.

    Gurobi exposes only ``Slack``, which is ``rhs - activity`` for every sense.
    ``QCSlack`` covers a quadratic row's whole left-hand side,
    :math:`x^\top Q x + a^\top x`.
    """
    import numpy as np

    def side(rhs: str, slack: str) -> Any:
        return np.asarray(m.getAttr(rhs), dtype=np.float64) - np.asarray(m.getAttr(slack), dtype=np.float64)

    return solver_vector(np.concatenate([side('RHS', 'Slack'), side('QCRHS', 'QCSlack')]))


def _duals(m: Any) -> pl.Series | None:
    """Shadow prices in row order, or ``None`` where Gurobi refuses them.

    Gurobi refuses ``Pi`` on a mixed-integer model, and ``QCPi`` unless
    ``solver_options={'QCPDual': 1}``.
    """
    import numpy as np

    gurobipy = _gurobipy()
    try:
        values = [np.asarray(m.getAttr(attribute), dtype=np.float64) for attribute in ('Pi', 'QCPi')]
    except (AttributeError, gurobipy.GurobiError):
        return None
    return solver_vector(np.concatenate(values))
