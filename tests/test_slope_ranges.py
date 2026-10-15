"""The PyPSA parity gate accepts a differing dual only inside the range of duals PyPSA's model admits.

One bus, a cheap unit `a` at 10 with a cap of 45 and a dear unit `b` at 50,
and a balance row `a + b == load`. The balance row's dual is the price of one
more unit of load, worked out by hand for each load.
"""

from __future__ import annotations

import math

import pytest

from differential.pypsa.slopes import inside, slope_ranges
from tests.oracle import linopy


def _dispatch(load: float, b_cap: float) -> tuple[linopy.Model, int]:
    m = linopy.Model()
    a = m.add_variables(lower=0, upper=45, name='a')
    b = m.add_variables(lower=0, upper=b_cap, name='b')
    m.add_constraints(a + b == load, name='balance')
    m.add_objective(10 * a + 50 * b)
    return m, int(m.constraints['balance'].labels.item())


@pytest.mark.parametrize(
    ('load', 'b_cap', 'expected'),
    [
        pytest.param(30.0, 100.0, (10.0, 10.0), id='a-sets-the-price-alone'),
        pytest.param(45.0, 100.0, (10.0, 50.0), id='a-at-its-cap-admits-every-price-between'),
        pytest.param(60.0, 100.0, (50.0, 50.0), id='b-sets-the-price-alone'),
        pytest.param(45.0, 0.0, (10.0, math.inf), id='no-more-load-can-be-met'),
    ],
)
def test_the_range_is_the_slope_of_the_objective_on_each_side_of_the_load(load, b_cap, expected):
    m, row = _dispatch(load, b_cap)
    low, high = slope_ranges(m, {row})[row]
    assert low == pytest.approx(expected[0]), (
        f'below a load of {load} the next unit comes from the unit priced {expected[0]}'
    )
    assert high == pytest.approx(expected[1]), f'above a load of {load} the next unit costs {expected[1]}'


def test_measuring_the_range_leaves_the_model_as_it_was():
    m, row = _dispatch(45.0, 100.0)
    before = m.constraints['balance'].rhs.item()
    slope_ranges(m, {row})
    assert m.constraints['balance'].rhs.item() == before, 'the slopes are taken on a copy, not on the model'


@pytest.mark.parametrize(
    ('value', 'accepted'),
    [
        pytest.param(30.0, True, id='between-the-slopes'),
        pytest.param(10.0, True, id='on-an-end'),
        pytest.param(50.00001, True, id='past-an-end-by-less-than-the-tolerance'),
        pytest.param(9.999, False, id='just-below'),
        pytest.param(50.01, False, id='just-above'),
    ],
)
def test_a_dual_is_accepted_only_inside_the_range(value, accepted):
    assert inside(value, (10.0, 50.0)) is accepted
