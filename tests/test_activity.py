"""Activity: a constraint row's left-hand side, read back after a solve.

The value is the solver's own, so a CSR recomputation agreeing with it checks
the whole chain: matrix, hand-off, solve, read-back. Unlike a dual, an
activity exists at any mixed-integer incumbent.
"""

from __future__ import annotations

import numpy as np
import pytest

import specsolve as sps
from specsolve.errors import NoSolutionError, SpecsolveError
from tests.conftest import recomputed_row_values
from tests.differential import differential
from tests.oracle import pd  # through the guard: a bare import would beat it
from tests.test_milp import COMMITMENT_YAML

ACTIVITY_RTOL = 1e-9


def _agrees_with_csr(run) -> None:
    """Each constraint's activity against its slice of the recomputation."""
    recomputed = recomputed_row_values(run.engine, run.result)
    for name, block in run.engine._model.constraints.items():
        got = run.result.activity(name)['value'].to_numpy()
        assert got == pytest.approx(recomputed[block.start : block.start + block.height], rel=ACTIVITY_RTOL), (
            f'the solver judged feasibility against a different {name!r} LHS than the CSR block holds. '
            f'Activity is the *whole* left-hand side, which is Ax on a linear row and x^T Q x + Ax '
            f'on a quadratic one — `recomputed_row_values` adds both, and this disagreement is that '
            f'invariant failing.'
        )


def test_activity_matches_the_csr_recomputation(dispatch_yaml, dispatch_inputs):
    """The solver's row values against Ax recomputed from the model's own CSR."""
    data = dispatch_inputs
    with differential(dispatch_yaml, data) as run:
        got = run.result.activity('power_balance')
        assert got.columns == ['snapshot', 'value']
        assert got.height == len(data['snapshot'])
        _agrees_with_csr(run)


def test_activity_matches_the_linopy_lane(dispatch_yaml, dispatch_inputs):
    """The linopy lane has no accessor, so its half is lhs evaluated at the solution."""
    data = dispatch_inputs
    with differential(dispatch_yaml, data) as run:
        oracle = run.model.constraints['power_balance'].lhs.solution
        expected = pd.Series(np.asarray(oracle), index=np.asarray(oracle.indexes['snapshot'])).sort_index()
        actual = run.result.activity('power_balance').sort('snapshot')['value'].to_numpy()
        assert actual == pytest.approx(expected.to_numpy(), rel=ACTIVITY_RTOL)


def test_a_milp_returns_activity_where_dual_refuses(commitment_inputs):
    """Activity is gated on `has_primal` alone: an integer incumbent has one."""
    data = commitment_inputs
    with differential(COMMITMENT_YAML, data) as run:
        assert run.result.has_primal
        _agrees_with_csr(run)

        balance = run.result.activity('balance').sort('snapshot')['value'].to_numpy()
        assert balance == pytest.approx(data['load'].sort_index().to_numpy(), rel=ACTIVITY_RTOL), (
            'the == balance row is met exactly at any feasible incumbent, so its activity is the load'
        )
        with pytest.raises(SpecsolveError, match='mixed-integer'):
            run.result.dual('balance')


def test_equality_row_activity_equals_rhs(dispatch_yaml, dispatch_frame_inputs):
    """On an `==` row activity equals the rhs up to solver tolerance, by construction."""
    data = dispatch_frame_inputs
    with sps.solve(dispatch_yaml, data) as sol:
        got = sol.activity('power_balance').sort('snapshot')['value'].to_numpy()
        assert got == pytest.approx(data['load'].sort('snapshot')['value'].to_numpy(), rel=ACTIVITY_RTOL), (
            'an == row holds at the solution, so its activity is its rhs — a residual check, not a bug'
        )

        with pytest.raises(KeyError, match='unknown constraint'):
            sol.activity('zzz')


def test_infeasible_solve_refuses_activity(dispatch_yaml, dispatch_inputs):
    """No values at all is the refusal `primal` shares — same class, same gate."""
    data = dispatch_inputs
    data = dict(data, load=pd.Series(1e6, index=data['snapshot']))  # more than every generator together

    with sps.solve(dispatch_yaml, data) as result:
        assert not result.has_primal
        with pytest.raises(NoSolutionError, match='cannot read the activity'):
            result.activity('power_balance')


def test_a_closed_result_refuses_activity(dispatch_yaml, dispatch_frame_inputs):
    """close() releases the activity frames with the primal and dual ones."""
    data = dispatch_frame_inputs
    with sps.solve(dispatch_yaml, data) as sol:
        pass
    with pytest.raises(SpecsolveError, match='closed'):
        sol.activity('power_balance')
