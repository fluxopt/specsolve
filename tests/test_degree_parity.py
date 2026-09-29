"""Degree 1, checked against the oracle lane itself.

The linopy lane refuses what the relational lane refuses, in the same words
(hard rule 3). The rule lives in ``mathspec.degree``, and both lanes ask it.
"""

from __future__ import annotations

import pytest

import specsolve as sps
from specsolve.errors import LanguageError
from tests.conftest import dispatch_spec_path
from tests.oracle import specsolve_linopy  # skips the module without the oracle


#: One entry per way the degree rule can be broken at build time on both lanes.
@pytest.mark.parametrize(
    ('patch', 'match'),
    [
        pytest.param(
            {'objective.expression': 'sum(p) * sum(p)'},
            'sums of more than one term',
            id='two-reductions-multiplied-even-in-the-objective',
        ),
        pytest.param(
            {'objective.expression': 'sum(cost / p)'},
            'divisor contains variables',
            id='variable-in-a-divisor',
        ),
        pytest.param(
            {'objective.expression': 'sum(p ** 2)'},
            'over variables',
            id='a-power-over-a-variable',
        ),
        pytest.param(
            {'objective.expression': 'sum(p / (1 - cost))'},
            'must be a single Constant/Parameter factor',
            id='a-divisor-that-adds',
        ),
    ],
)
def test_both_lanes_refuse_the_same_expression(tmp_path, dispatch_spec_inputs, patch, match):
    """Not just "both raise": both say the same thing.

    The relational lane prefixes the declaration it was lowering, so its
    message ends with the linopy one.
    """
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **patch)

    with pytest.raises(LanguageError, match=match) as linopy_lane:
        specsolve_linopy.build(path, data)

    with pytest.raises(LanguageError, match=match) as relational:
        sps.check(path)

    assert str(relational.value).endswith(str(linopy_lane.value))


def test_the_linopy_lane_still_accepts_an_affine_product(tmp_path, dispatch_spec_inputs):
    """The guard refuses degree 2, not ``variable * parameter``."""
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **{'objective.expression': 'sum(p * cost)'})
    model = specsolve_linopy.build(path, data)
    assert model.objective is not None
