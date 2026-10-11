"""The capability descriptor: what it answers, and what each sink declares.

The probe modules tie each declaration to what the library does; this module
holds the descriptor's own arithmetic.
"""

from __future__ import annotations

import pytest

from specsolve.relational.sinks import SOLVERS, WRITERS
from specsolve.relational.sinks.capabilities import ALL_CAPABILITIES, Capabilities

EMPTY = Capabilities(supports=frozenset({}))

HIGHS_SHAPED = Capabilities(
    supports=frozenset({'integrality', 'quadratic_objective'}),
    excludes=(frozenset({'quadratic_objective', 'integrality'}),),
)


def test_a_capability_left_out_is_absent():
    """A descriptor lists only what a sink *can* do."""
    assert 'sos' not in EMPTY.supports
    assert 'quadratic_constraint' not in HIGHS_SHAPED.supports


def test_missing_names_only_what_is_required_and_absent():
    assert HIGHS_SHAPED.missing(['sos', 'quadratic_objective']) == ['sos']
    assert HIGHS_SHAPED.missing(['quadratic_constraint']) == ['quadratic_constraint']


def test_missing_reads_in_vocabulary_order_not_the_callers():
    """A refusal naming two capabilities reads the same way whichever order the
    program's requirements happened to be collected in.
    """
    ordered = ['nonconvex_quadratic_objective', 'quadratic_constraint']
    assert EMPTY.missing(ordered) == ordered
    assert EMPTY.missing(ordered[::-1]) == ordered, 'the caller collected them the other way round'


def test_an_exclusion_fires_only_on_the_whole_combination():
    """The case a flat feature set cannot express: both halves supported, the
    pair refused."""
    assert HIGHS_SHAPED.excluded(['quadratic_objective']) is None
    assert HIGHS_SHAPED.excluded(['integrality']) is None
    assert HIGHS_SHAPED.excluded(['quadratic_objective', 'integrality']) == frozenset(
        {'quadratic_objective', 'integrality'}
    )


def test_an_exclusion_fires_inside_a_larger_requirement():
    """A model needing a third thing as well is still the refused pair."""
    assert HIGHS_SHAPED.excluded(['quadratic_objective', 'integrality', 'sos']) is not None


def test_nothing_is_excluded_where_no_exclusion_is_declared():
    assert HIGHS_SHAPED.excluded([]) is None
    assert Capabilities(supports=frozenset({'integrality'})).excluded(ALL_CAPABILITIES) is None


@pytest.mark.parametrize(
    ('sink', 'capability', 'takes'),
    [
        pytest.param('highs', 'sos', False, id='highs-has-no-set-concept'),
        pytest.param('highs', 'quadratic_objective', True, id='highs-takes-a-convex-hessian'),
        pytest.param('highs', 'nonconvex_quadratic_objective', False, id='highs-refuses-a-nonconvex-one'),
        pytest.param('highs', 'quadratic_constraint', False, id='highs-has-no-quadratic-row-at-all'),
        pytest.param('gurobi', 'sos', True, id='gurobi-branches-on-a-set'),
        pytest.param('gurobi', 'quadratic_objective', True, id='gurobi-passes-a-hessian'),
        pytest.param('gurobi', 'nonconvex_quadratic_objective', True, id='gurobi-goes-spatial'),
        pytest.param('gurobi', 'quadratic_constraint', True, id='gurobi-takes-a-quadratic-row'),
    ],
)
def test_the_shipped_solver_table(sink, capability, takes):
    """What each sink can ingest **as shipped**, not what its library can."""
    assert (capability in SOLVERS[sink].capabilities.supports) is takes


def test_only_highs_excludes_a_combination():
    """Gurobi's column has no exclusion, which is what makes it the sink a
    refusal can name."""
    assert SOLVERS['highs'].capabilities.excludes == (frozenset({'quadratic_objective', 'integrality'}),)
    assert SOLVERS['gurobi'].capabilities.excludes == ()


def test_the_lp_writer_carries_what_it_writes():
    """A section is text, so the format has no exclusion — and what it declares
    is the *writer's*, not the reader's: HiGHS parses neither the `sos` section
    nor the quadratic-constraint one this writer emits."""
    capabilities = WRITERS['.lp'].capabilities
    assert capabilities.missing(ALL_CAPABILITIES) == [], 'every section the language can reach is emitted now'
    assert 'quadratic_constraint' in capabilities.supports, 'the section this branch taught it to write'
    assert capabilities.excludes == ()
