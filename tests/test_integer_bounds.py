"""An integer variable's bounds are read as the whole numbers they admit, on both lanes and in the LP file."""

from __future__ import annotations

import pytest

from tests.differential import differential


def _one_integer(lower: float, upper: float, sense: str) -> dict:
    return {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'lo': {'dims': []}, 'hi': {'dims': []}},
        'variables': {'v': {'dims': ['g'], 'bounds': {'lower': 'lo', 'upper': 'hi'}, 'domain': 'integer'}},
        'constraints': {'c': {'dims': ['g'], 'expression': 'v <= 1.5'}},
        'objective': {'sense': sense, 'expression': 'sum(v)'},
    }, {'g': ['g3'], 'lo': lower, 'hi': upper}


@pytest.mark.parametrize(
    ('lower', 'upper', 'sense', 'expected'),
    [
        pytest.param(-0.5, 10.0, 'minimize', 0.0, id='a fractional lower bound rounds up'),
        pytest.param(-10.0, 0.5, 'maximize', 0.0, id='a fractional upper bound rounds down'),
        pytest.param(1.0000000001, 10.0, 'minimize', 1.0, id='a lower bound a hair above a whole number keeps it'),
    ],
)
def test_an_integer_variable_reads_its_bounds_as_whole_numbers(
    lower: float, upper: float, sense: str, expected: float
) -> None:
    """HiGHS answered a lower bound of -0.5 as optimal with no feasible values, and the oracle as -0.5 (#2048)."""
    spec, sources = _one_integer(lower, upper, sense)
    with differential(spec, sources, lp=True) as run:
        assert run.oracle == pytest.approx(expected), f'v is the whole number nearest its bound, {expected}'
