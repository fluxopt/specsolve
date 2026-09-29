"""The sink-capability table in `docs/about/benchmarks.md`, executed.

Every row of that table is a claim about another library, so it goes stale on
their release. A failure here is a capability that moved, not a regression.
Nothing here builds a specsolve model: these are the solver libraries at their
own API.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import highspy
import numpy as np
import pytest

from specsolve.relational.sinks import SOLVERS

if TYPE_CHECKING:
    from pathlib import Path

TABLE = 'docs/about/benchmarks.md, "Sink capabilities"'

#: Minimise ``x0² + x1²`` subject to ``x0 + x1 >= 2``: ``x = (1, 1)``, objective 2.
CONVEX_OPTIMUM = 2.0

QUADRATIC_LP = """min

obj: + [ 2 x0 ^ 2 + 2 x1 ^ 2 ] / 2

s.t.

c0: +1 x0 +1 x1 >= 2

bounds
0 <= x0 <= 10
0 <= x1 <= 10

end
"""

QUADRATIC_CONSTRAINT_LP = """min

obj: +1 x0 +1 x1

s.t.

qc0: [ +1 x0 * x1 ] >= 4

bounds
0 <= x0 <= 10
0 <= x1 <= 10

end
"""

SOS_LP = """min

obj: +1 x0 +1 x1

s.t.

c0: +1 x0 +1 x1 >= 1

bounds
0 <= x0 <= 10
0 <= x1 <= 10

sos
s0: S1 :: x0:1 x1:2

end
"""


def _highs_qp(curvature: float = 2.0, *, integral: bool = False) -> Any:
    """A two-column QP: an LP, plus ``Q = curvature · I`` through the sink's own entry point."""
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    lp = highspy.HighsLp()
    lp.num_col_ = 2
    lp.num_row_ = 1
    lp.col_cost_ = np.array([0.0, 0.0])
    lp.col_lower_ = np.array([0.0, 0.0])
    lp.col_upper_ = np.array([10.0, 10.0])
    lp.row_lower_ = np.array([2.0])
    lp.row_upper_ = np.array([h.getInfinity()])
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.start_ = np.array([0, 2])
    lp.a_matrix_.index_ = np.array([0, 1])
    lp.a_matrix_.value_ = np.array([1.0, 1.0])
    if integral:
        lp.integrality_ = [highspy.HighsVarType.kInteger, highspy.HighsVarType.kInteger]
    assert h.passModel(lp) == highspy.HighsStatus.kOk
    assert _pass_hessian(h, curvature) == highspy.HighsStatus.kOk, (
        'passHessian accepts what it is handed; a refusal comes at run(), not here'
    )
    return h


def _pass_hessian(h: Any, curvature: float) -> Any:
    return h.passHessian(
        2,
        2,
        int(highspy.HessianFormat.kTriangular),
        np.array([0, 1, 2]),
        np.array([0, 1]),
        np.array([curvature, curvature]),
    )


def test_highs_solves_a_convex_quadratic_objective():
    h = _highs_qp()
    assert h.run() == highspy.HighsStatus.kOk, f'{TABLE} says HiGHS takes a convex Hessian through passHessian'
    assert h.getModelStatus() == highspy.HighsModelStatus.kOptimal
    assert h.getObjectiveValue() == pytest.approx(CONVEX_OPTIMUM)


def test_highs_refuses_a_nonconvex_hessian():
    h = _highs_qp(curvature=-2.0)
    assert h.run() == highspy.HighsStatus.kError, (
        f'{TABLE} says HiGHS refuses a non-PSD Hessian — it printed "Cannot solve non-convex QP '
        f'problems with HiGHS" when this was measured'
    )


def test_highs_refuses_a_hessian_beside_integrality():
    h = _highs_qp(integral=True)
    assert h.run() == highspy.HighsStatus.kError, (
        f'{TABLE} says HiGHS refuses a Hessian and integrality together — the conjunction a flat '
        f'set of features cannot express'
    )


def test_highs_has_no_quadratic_constraint_entry_point():
    named = [name for name in dir(highspy.Highs) if 'quad' in name.lower() or 'qconstr' in name.lower()]
    assert named == [], f'{TABLE} says HiGHS has no quadratic-constraint concept, and {named} contradicts it'


def test_the_highs_lp_reader_takes_a_quadratic_objective_back(tmp_path: Path):
    """The differential oracle re-solves a written LP, so a section HiGHS cannot
    parse is one this package cannot write and still check."""
    path = tmp_path / 'quadratic.lp'
    path.write_text(QUADRATIC_LP)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    assert h.readModel(str(path)) == highspy.HighsStatus.kOk, (
        f'{TABLE} says the HiGHS reader takes the LP quadratic-objective section back'
    )
    assert h.run() == highspy.HighsStatus.kOk
    assert h.getObjectiveValue() == pytest.approx(CONVEX_OPTIMUM), (
        'read back it must be the model the Hessian API loads, or the LP file is a different oracle'
    )


def test_the_highs_lp_reader_refuses_the_sos_section(tmp_path: Path):
    path = tmp_path / 'sets.lp'
    path.write_text(SOS_LP)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    assert h.readModel(str(path)) == highspy.HighsStatus.kError, (
        f'{TABLE} says the HiGHS reader refuses an sos section. If it takes one now, HiGHS has the '
        f"concept and its descriptor should declare `'sos': 'native'`."
    )


def test_the_highs_lp_reader_refuses_the_quadratic_constraint_section(tmp_path: Path):
    path = tmp_path / 'quadratic_constraint.lp'
    path.write_text(QUADRATIC_CONSTRAINT_LP)
    h = highspy.Highs()
    h.setOptionValue('output_flag', False)
    assert h.readModel(str(path)) == highspy.HighsStatus.kError, (
        f'{TABLE} says the HiGHS reader refuses a quadratic constraint the way it refuses an sos '
        f'section — it printed "Quadratic constraints not supported by HiGHS" when this was '
        f'measured. If it takes one now, the `lp_file` column stops being write-only for that row.'
    )


def test_a_second_hessian_replaces_the_first_and_keeps_the_model():
    """What an update may push: the quadratic part goes over whole, but onto the
    model already loaded, so only its sparsity pattern is structure."""
    h = _highs_qp()
    h.run()
    assert h.getObjectiveValue() == pytest.approx(CONVEX_OPTIMUM)

    assert _pass_hessian(h, 8.0) == highspy.HighsStatus.kOk
    assert h.run() == highspy.HighsStatus.kOk
    assert (h.getNumCol(), h.getNumRow()) == (2, 1), 'a second passHessian must not disturb the loaded LP'
    assert h.getObjectiveValue() == pytest.approx(4.0 * CONVEX_OPTIMUM), (
        'replaced rather than accumulated — 4x the curvature at the same optimum. One that '
        'accumulated would make an update wrong rather than slow.'
    )


def test_the_highs_descriptor_says_what_these_probes_measured():
    """The HiGHS descriptor declares what the probes above measured."""
    capabilities = SOLVERS['highs'].capabilities
    assert capabilities.support('quadratic_objective') == 'native', 'a convex Hessian solved, above'
    assert capabilities.support('nonconvex_quadratic_objective') == 'absent', 'a non-PSD Hessian was refused, above'
    assert capabilities.support('quadratic_constraint') == 'absent', 'there is no entry point to call, above'
    assert capabilities.support('sos') == 'absent', (
        'HiGHS has no set concept — its own LP reader refusing the section is that same fact from the other side'
    )
    assert capabilities.excluded(['quadratic_objective', 'integrality']) is not None, (
        'the pair returned kError above, and a flat set of features cannot say so'
    )
