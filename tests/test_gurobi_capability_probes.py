"""The Gurobi column of the sink-capability table, measured rather than read.

`test_sink_capability_probes.py`'s twin. Every model is two columns wide, so it
runs under the size-limited licence gurobipy ships in its own wheel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

gurobipy = pytest.importorskip('gurobipy', reason='the gurobi sink needs the [gurobi] extra')

from specsolve.relational.sinks import SOLVERS  # noqa: E402 — after the guard, or a bare install fails at import
from specsolve.relational.sinks.capabilities import CAPABILITIES  # noqa: E402

TABLE = 'docs/about/benchmarks.md, "Sink capabilities"'

#: What these probes measure is acceptance, not precision: a spatial or mixed-integer
#: solve stops on ``MIPGap`` (1e-4 by default), so a vertex proved to that gap is a yes.
ACCEPTED = 1e-3


@pytest.fixture
def model() -> Iterator[Callable[[], Any]]:
    """A silent, disposable model — one environment per test, since each holds a licence."""
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        made: list[Any] = []

        def build() -> Any:
            m = gurobipy.Model(env=env)
            made.append(m)
            return m

        yield build
        for m in made:
            m.dispose()


def _solved(m: Any) -> float:
    m.optimize()
    assert m.Status == gurobipy.GRB.OPTIMAL, f'expected an optimal solve, got status {m.Status}'
    return float(m.ObjVal)


def test_gurobi_takes_a_convex_quadratic_objective(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] + x[1] >= 2)
    m.setObjective(x[0] * x[0] + x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(2.0), f'{TABLE} says a quadratic objective is native on Gurobi'


def test_gurobi_takes_a_nonconvex_quadratic_objective_at_default_parameters(model):
    """The automatic default reaches spatial branch-and-bound without `NonConvex=2`,
    so a refusal that names Gurobi needs no "…if you set a parameter"."""
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] + x[1] >= 2)
    m.setObjective(-x[0] * x[0] - x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(-200.0, rel=ACCEPTED), (
        f'{TABLE} says Gurobi solves a nonconvex quadratic objective, at default parameters. A '
        f'refusal here means the nonconvex row needs a parameter beside it.'
    )


def test_gurobi_takes_a_quadratic_objective_beside_integrality(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10, vtype=gurobipy.GRB.INTEGER)
    m.addConstr(x[0] + x[1] >= 3)
    m.setObjective(x[0] * x[0] + x[1] * x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(5.0, rel=ACCEPTED), f'{TABLE} says MIQP is native on Gurobi'


def test_gurobi_takes_a_quadratic_constraint(model):
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    m.addConstr(x[0] * x[1] >= 4)
    m.setObjective(x[0] + x[1], gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(4.0, rel=ACCEPTED), f'{TABLE} says a quadratic constraint is native on Gurobi'


def test_both_quadratic_parts_take_a_matrix_through_their_bulk_entry_point(model):
    """`addSOS` has no bulk form; the quadratic parts do. Measured through the
    matrix APIs the table names, since those are what this package calls."""
    m = model()
    x = m.addVars(2, lb=0, ub=10)
    columns = [x[0], x[1]]
    m.addMQConstr(np.array([[0.0, 1.0], [0.0, 0.0]]), None, gurobipy.GRB.GREATER_EQUAL, 4.0, columns, columns)
    m.setMObjective(np.eye(2), None, 0.0, columns, columns, sense=gurobipy.GRB.MINIMIZE)
    assert _solved(m) == pytest.approx(8.0, rel=ACCEPTED), (
        f'{TABLE} names setMObjective and addMQConstr as the bulk quadratic entry points. Q = I '
        f'over x0·x1 >= 4 minimises x0² + x1² at x = (2, 2), so 8.'
    )


def test_the_gurobi_descriptor_says_what_this_sink_does_with_what_it_measured():
    """The claim beside its evidence — `test_sink_capability_probes.py`'s twin.

    This sink hands gurobipy every model above, the quadratic constraint
    included.
    """
    capabilities = SOLVERS['gurobi'].capabilities
    for capability in CAPABILITIES:
        assert capability in capabilities.supports, f'{capability} solved natively above'
    assert capabilities.excludes == (), 'every combination probed above solved; nothing here is excluded'
