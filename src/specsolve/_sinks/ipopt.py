"""The ``ipopt`` solver: a local, interior-point solve of a smooth model, through cyipopt.

Ipopt asks for values and derivatives rather than a matrix, so this sink
evaluates the model itself. The constraints it hands Ipopt are every row of
the handoff, then one equation per defined column, ``x[col] - op(a, b) == 0``.
Every derivative is exact: a defined column's argument is affine, so its
Jacobian row and Hessian block follow from the op's own first and second
derivatives by the chain rule ([`_derivatives`][]).

``cyipopt`` and ``scipy`` are imported inside the functions, so importing this
module needs neither.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from specsolve._sinks.capabilities import Capabilities
from specsolve._sinks.handoff import SENSE_CODES
from specsolve._sinks.solver import InfeasibleSubsystemIndices, SolveAnswer, Solver, solver_vector
from specsolve._status import SolveStatus
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Mapping

    import polars as pl

    from specsolve._sinks.handoff import Handoff
    from specsolve._sinks.solver import Basis

#: What Ipopt reads as no bound.
_INFINITY = 1e20

#: Ipopt's return status -> termination condition. A success is a local optimum.
_CONDITION_OF_IPOPT_STATUS = {
    0: 'optimal',
    1: 'suboptimal',
    2: 'infeasible',
    3: 'other',
    4: 'unbounded',
    5: 'other',
    6: 'other',
    -1: 'iteration_limit',
    -2: 'other',
    -3: 'other',
    -4: 'time_limit',
    -5: 'time_limit',
    -10: 'error',
    -11: 'error',
    -12: 'error',
    -13: 'other',
    -100: 'internal_solver_error',
    -101: 'internal_solver_error',
    -102: 'resource_interrupt',
    -199: 'internal_solver_error',
}

#: The statuses that leave a point worth reading.
_HAS_PRIMAL = frozenset({0, 1, -1, -4, -5})


class Ipopt(Solver):
    """Ipopt, holding the evaluated model at ``.handle``, a ``cyipopt.Problem``.

    A solve finds a local optimum: on a nonconvex model it is the one the
    start leads to. Duals are the local multipliers, signed as every sink's.
    """

    _problem: Any
    _model: _Model
    _x0: Any
    _release: weakref.finalize[[Any], Ipopt]

    requires = ('cyipopt', 'scipy.sparse')
    recorded_options = frozenset({'tol', 'max_iter', 'max_cpu_time', 'mu_strategy', 'hessian_approximation'})
    unavailable_message = 'The ipopt sink requires cyipopt and scipy: conda install -c conda-forge cyipopt'
    lp_values = MappingProxyType({'complete': 'used', 'partial': 'used'})
    capabilities = Capabilities(
        supports=frozenset(
            {'quadratic_objective', 'nonconvex_quadratic_objective', 'quadratic_constraint', 'nonlinear'}
        )
    )

    def _load(self, handoff: Handoff) -> None:
        import numpy as np

        self._model = _Model.of(handoff)
        self._x0 = np.full(handoff.column_count, np.nan)
        self._problem = self._model.problem(self._options)
        self._release = weakref.finalize(self, _closed, self._problem)

    @property
    def handle(self) -> Any:
        return self._problem

    def push(self, handoff: Handoff) -> None:
        """Ipopt keeps nothing between solves but the start, so new numbers are a new evaluation."""
        self._release()
        start = self._x0
        self._load(handoff)
        self._x0 = start

    def dual_ray(self) -> pl.Series | None:
        """Ipopt proves no infeasibility, so it leaves no ray."""
        return None

    def _basis(self) -> tuple[Any, Any] | None:
        """An interior-point solve ends at no vertex."""
        return None

    def _warm(self, basis: Basis) -> None:
        raise SpecsolveError('ipopt is an interior-point solver and cannot start from a basis. Start from values.')

    def _start(self, values: Any) -> None:
        self._x0 = values

    def _run(self, handoff: Handoff) -> SolveAnswer:
        import numpy as np

        x, info = self._problem.solve(self._model.start(self._x0))
        code = int(info['status'])
        status = SolveStatus(
            termination_condition=_CONDITION_OF_IPOPT_STATUS.get(code, 'unknown'),
            solver_wording=info['status_msg'].decode() if isinstance(info['status_msg'], bytes) else info['status_msg'],
            has_primal=code in _HAS_PRIMAL,
        )
        if not status.is_readable:
            return self._unreadable(status)
        rows = handoff.row_count
        sign = -1.0 if handoff.objective_sense == 'maximize' else 1.0
        return SolveAnswer(
            status,
            sign * float(info['obj_val']) + handoff.objective_constant,
            solver_vector(np.asarray(x, dtype=np.float64)),
            solver_vector(-sign * np.asarray(info['mult_g'][:rows], dtype=np.float64)),
            solver_vector(np.asarray(info['g'][:rows], dtype=np.float64)),
        )

    def infeasible_subsystem(self) -> InfeasibleSubsystemIndices | None:
        """Ipopt has no notion of one."""
        return None

    def forget(self) -> None:
        """Nothing outlives a solve but the start, which a caller sets."""

    def close(self) -> None:
        self._release()
        self._problem = None


def _closed(problem: Any) -> None:
    problem.close()


@dataclass(frozen=True)
class _Model:
    """The handoff as the callbacks Ipopt evaluates, every sparsity pattern fixed once.

    Each callback writes a fixed list of raw entries, and ``*_slot`` sums them
    onto Ipopt's pattern, a structure computed once at load
    ([`_pattern`][]).
    """

    columns: int
    sign: float
    cost: Any
    objective_pairs: Any
    matrix: Any
    row_pairs: Any
    lower: Any
    upper: Any
    row_lower: Any
    row_upper: Any
    defined: Any
    ops: Any
    first: Any
    second: Any
    first_constant: Any
    second_constant: Any
    jacobian_rows: Any
    jacobian_cols: Any
    jacobian_slot: Any
    hessian_rows: Any
    hessian_cols: Any
    hessian_slot: Any
    hessian_pairs: Any

    @classmethod
    def of(cls, handoff: Handoff) -> _Model:
        import numpy as np
        import scipy.sparse

        n, m, k = handoff.column_count, handoff.row_count, handoff.nl.height
        cols = handoff.dense_columns(_INFINITY)
        rows = handoff.dense_rows(_INFINITY)
        less, more, equal = (rows.sense == SENSE_CODES[s] for s in ('<=', '>=', '=='))
        row_lower = np.concatenate([np.where(more | equal, rows.rhs, -_INFINITY), np.zeros(k)])
        row_upper = np.concatenate([np.where(less | equal, rows.rhs, _INFINITY), np.zeros(k)])

        entries = handoff.matrix_block(0, m)
        matrix = scipy.sparse.csr_matrix(
            (entries['coeff'].to_numpy(), (entries['row'].to_numpy(), entries['col'].to_numpy())), shape=(m, n)
        )
        objective_pairs = _pairs(handoff.quad, None)
        row_pairs = _pairs(handoff.qmatrix, 'row')

        defined = handoff.nl['col'].to_numpy().astype(np.int64)
        at = {int(c): i for i, c in enumerate(defined)}
        args = handoff.nl_args
        atom = np.asarray([at[int(c)] for c in args['col']], dtype=np.int64)
        held = args['var'].is_not_null().to_numpy()
        var = args['var'].fill_null(0).to_numpy().astype(np.int64)
        coeff, arg = args['coeff'].to_numpy(), args['arg'].to_numpy()

        def argument(position: int) -> tuple[Any, Any]:
            on = (arg == position) & held
            constant = np.bincount(atom[(arg == position) & ~held], coeff[(arg == position) & ~held], minlength=k)
            return scipy.sparse.csr_matrix((coeff[on], (atom[on], var[on])), shape=(k, n)), constant

        (first, first_constant), (second, second_constant) = argument(0), argument(1)

        jacobian = (
            np.concatenate([entries['row'].to_numpy(), row_pairs[0], row_pairs[0], m + np.arange(k)]),
            np.concatenate([entries['col'].to_numpy(), row_pairs[1], row_pairs[2], defined]),
        )
        first_coo, second_coo = first.tocoo(), second.tocoo()
        jacobian = (
            np.concatenate([jacobian[0], m + first_coo.row, m + second_coo.row]),
            np.concatenate([jacobian[1], first_coo.col, second_coo.col]),
        )
        jacobian_rows, jacobian_cols, jacobian_slot = _pattern(*jacobian, n)

        hessian_pairs = _atom_pairs(first_coo, second_coo)
        hessian = (
            np.concatenate([objective_pairs[4], row_pairs[4], hessian_pairs[0]]),
            np.concatenate([objective_pairs[5], row_pairs[5], hessian_pairs[1]]),
        )
        hessian_rows, hessian_cols, hessian_slot = _pattern(*hessian, n)

        return cls(
            columns=n,
            sign=-1.0 if handoff.objective_sense == 'maximize' else 1.0,
            cost=cols.cost,
            objective_pairs=objective_pairs,
            matrix=matrix,
            row_pairs=row_pairs,
            lower=cols.lb,
            upper=cols.ub,
            row_lower=row_lower,
            row_upper=row_upper,
            defined=defined,
            ops=handoff.nl['op'].cast(str).to_numpy(),
            first=first,
            second=second,
            first_constant=first_constant,
            second_constant=second_constant,
            jacobian_rows=jacobian_rows,
            jacobian_cols=jacobian_cols,
            jacobian_slot=jacobian_slot,
            hessian_rows=hessian_rows,
            hessian_cols=hessian_cols,
            hessian_slot=hessian_slot,
            hessian_pairs=hessian_pairs,
        )

    def problem(self, options: Mapping[str, Any]) -> Any:
        """A ``cyipopt.Problem`` over this model, quiet unless *options* say otherwise."""
        import cyipopt

        problem = cyipopt.Problem(  # pyrefly: ignore[missing-attribute] — cyipopt ships no stubs
            n=self.columns,
            m=len(self.row_lower),
            problem_obj=self,
            lb=self.lower,
            ub=self.upper,
            cl=self.row_lower,
            cu=self.row_upper,
        )
        for name, value in {'print_level': 0, 'sb': 'yes', **options}.items():
            problem.add_option(name, value)
        return problem

    def start(self, given: Any) -> Any:
        """*given* where it has a value, else zero moved inside the bounds, and every defined column at its equation."""
        import numpy as np

        x = np.where(np.isnan(given), np.clip(0.0, self.lower, self.upper), given)
        free = np.isnan(given[self.defined]) if len(self.defined) else np.zeros(0, dtype=bool)
        for _ in range(len(self.defined)):
            with np.errstate(all='ignore'):
                value = _derivatives(self.ops, *self._arguments(x))[0]
            settled = np.where(free & np.isfinite(value), value, x[self.defined])
            if np.array_equal(settled, x[self.defined]):
                break
            x[self.defined] = settled
        return x

    def _arguments(self, x: Any) -> tuple[Any, Any]:
        return self.first @ x + self.first_constant, self.second @ x + self.second_constant

    # ------------------------------------------------------------------
    # the callbacks cyipopt calls by name
    # ------------------------------------------------------------------

    def objective(self, x: Any) -> float:
        _, left, right, coeff = self.objective_pairs[:4]
        return self.sign * float(self.cost @ x + (coeff * x[left] * x[right]).sum())

    def gradient(self, x: Any) -> Any:
        import numpy as np

        _, left, right, coeff = self.objective_pairs[:4]
        grad = self.cost.copy()
        np.add.at(grad, left, coeff * x[right])
        np.add.at(grad, right, coeff * x[left])
        return self.sign * grad

    def constraints(self, x: Any) -> Any:
        import numpy as np

        row, left, right, coeff = self.row_pairs[:4]
        rows = self.matrix @ x
        np.add.at(rows, row, coeff * x[left] * x[right])
        value = _derivatives(self.ops, *self._arguments(x))[0]
        return np.concatenate([rows, x[self.defined] - value])

    def jacobianstructure(self) -> tuple[Any, Any]:
        return self.jacobian_rows, self.jacobian_cols

    def jacobian(self, x: Any) -> Any:
        import numpy as np

        _, left, right, coeff = self.row_pairs[:4]
        d = _derivatives(self.ops, *self._arguments(x))
        first, second = self.first.tocoo(), self.second.tocoo()
        raw = np.concatenate(
            [
                self.matrix.tocoo().data,
                coeff * x[right],
                coeff * x[left],
                np.ones(len(self.defined)),
                -d[1][first.row] * first.data,
                -d[2][second.row] * second.data,
            ]
        )
        return np.bincount(self.jacobian_slot, raw, minlength=len(self.jacobian_rows))

    def hessianstructure(self) -> tuple[Any, Any]:
        return self.hessian_rows, self.hessian_cols

    def hessian(self, x: Any, lagrange: Any, obj_factor: float) -> Any:
        import numpy as np

        objective_factor = self.objective_pairs[6] * self.objective_pairs[3] * obj_factor * self.sign
        rows = lagrange[self.row_pairs[0]] * self.row_pairs[3] * self.row_pairs[6]
        atoms = lagrange[len(lagrange) - len(self.defined) :]
        d = _derivatives(self.ops, *self._arguments(x))
        atom, _, _, product, kind = self.hessian_pairs[2:]
        second = np.stack([d[3], d[4], d[5]])[kind, atom]
        raw = np.concatenate([objective_factor, rows, -atoms[atom] * second * product])
        return np.bincount(self.hessian_slot, raw, minlength=len(self.hessian_rows))


def _pairs(frame: pl.DataFrame, row: str | None) -> tuple[Any, ...]:
    """Quadratic entries ``coeff · x[l] · x[r]`` as ``(row, l, r, coeff, hessian row, hessian col, weight)`` arrays.

    The objective's ``row`` is all zeros. A pair off the diagonal fills one
    lower-triangle Hessian slot with ``coeff``, and a square ``2 · coeff``, so
    *weight* is 1 or 2.
    """
    import numpy as np

    left = frame['col_l'].to_numpy().astype(np.int64)
    right = frame['col_r'].to_numpy().astype(np.int64)
    rows = frame[row].to_numpy().astype(np.int64) if row is not None else np.zeros(len(left), dtype=np.int64)
    weight = np.where(left == right, 2.0, 1.0)
    return rows, left, right, frame['coeff'].to_numpy(), np.maximum(left, right), np.minimum(left, right), weight


def _atom_pairs(first: Any, second: Any) -> tuple[Any, ...]:
    """Every lower-triangle Hessian entry a defined column's equation reaches, once per pair of argument entries.

    For ``op(a, b)`` with ``a = u·x`` and ``b = v·x`` the block is
    ``φ_aa uu' + φ_ab (uv' + vu') + φ_bb vv'``. Returns ``(hessian row,
    hessian col, atom, entry, entry, u_i v_j times its multiplicity, which
    second derivative)``.
    """
    import polars as pl

    def entries(m: Any) -> pl.DataFrame:
        return pl.DataFrame({'atom': m.row, 'var': m.col, 'c': m.data})

    a, b = entries(first), entries(second)
    blocks = []
    for kind, (left, right) in enumerate(((a, a), (a, b), (b, b))):
        joined = left.join(right, on='atom', suffix='_2')
        if kind != 1:
            joined = joined.filter(pl.col('var') >= pl.col('var_2'))
        diagonal = pl.col('var') == pl.col('var_2')
        weight = pl.when(diagonal).then(2.0 if kind == 1 else 1.0).otherwise(1.0)
        blocks.append(
            joined.select(
                pl.max_horizontal('var', 'var_2').alias('r'),
                pl.min_horizontal('var', 'var_2').alias('c_'),
                'atom',
                'var',
                'var_2',
                (pl.col('c') * pl.col('c_2') * weight).alias('product'),
                pl.lit(kind).alias('kind'),
            )
        )
    stacked = pl.concat(blocks)
    return tuple(stacked[c].to_numpy() for c in ('r', 'c_', 'atom', 'var', 'var_2', 'product', 'kind'))


def _pattern(rows: Any, cols: Any, n: int) -> tuple[Any, Any, Any]:
    """The distinct ``(row, col)`` among the raw entries, and each raw entry's slot in it."""
    import numpy as np

    keys = rows.astype(np.int64) * max(n, 1) + cols.astype(np.int64)
    unique, slot = np.unique(keys, return_inverse=True)
    return unique // max(n, 1), unique % max(n, 1), slot


def _derivatives(ops: Any, a: Any, b: Any) -> tuple[Any, ...]:
    """``φ, φ_a, φ_b, φ_aa, φ_ab, φ_bb`` of each defined column's op at its arguments.

    A power's derivatives in its exponent need ``log(a)``, so they are taken
    only where the exponent varies: ``x ** 2`` at a negative ``x`` stays defined.
    """
    import numpy as np

    zero = np.zeros_like(a)
    out = [zero.copy() for _ in range(6)]
    with np.errstate(all='ignore'):
        for op, d in _ops().items():
            on = ops == op
            if on.any():
                for slot, value in enumerate(d(a[on], b[on])):
                    out[slot][on] = value
    return tuple(out)


def _power(a: Any, b: Any) -> tuple[Any, ...]:
    import numpy as np

    log = np.log(np.where(a > 0, a, np.nan))
    value = a**b
    return (
        value,
        b * a ** (b - 1),
        np.nan_to_num(value * log),
        b * (b - 1) * a ** (b - 2),
        np.nan_to_num(a ** (b - 1) * (1 + b * log)),
        np.nan_to_num(value * log**2),
    )


@cache
def _ops() -> Mapping[str, Any]:
    """Each op's value and derivatives, built on first use so importing this module imports no numpy."""
    import numpy as np

    zero = np.zeros_like
    return {
        'mul': lambda a, b: (a * b, b, a, zero(a), np.ones_like(a), zero(a)),
        'div': lambda a, b: (a / b, 1 / b, -a / b**2, zero(a), -1 / b**2, 2 * a / b**3),
        'pow': _power,
        'exp': lambda a, b: (np.exp(a), np.exp(a), zero(a), np.exp(a), zero(a), zero(a)),
        'log': lambda a, b: (np.log(a), 1 / a, zero(a), -1 / a**2, zero(a), zero(a)),
        'sqrt': lambda a, b: (np.sqrt(a), 0.5 / np.sqrt(a), zero(a), -0.25 * a**-1.5, zero(a), zero(a)),
        'sin': lambda a, b: (np.sin(a), np.cos(a), zero(a), -np.sin(a), zero(a), zero(a)),
        'cos': lambda a, b: (np.cos(a), -np.sin(a), zero(a), -np.cos(a), zero(a), zero(a)),
    }
