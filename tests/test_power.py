"""``**`` over parameters: the one exponent the language reads off the data.

The operator is degree 0 in variables wherever it appears, so it is not a
ceiling question at all — it is the arithmetic ``*`` already does, spelled the
way a discount factor is written. What it costs is one refusal per way a
variable can get underneath it, and one for an operand that adds: addition does
not distribute over ``**``, so ``(1 + rate) ** period`` is two factors wearing
one and is refused where ``growth ** period`` is not (#1175).
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest
from mathspec.program import Add, Constant, Parameter, Power, Variable

import specsolve as sps
from specsolve.errors import LanguageError
from specsolve.sources import tidy_sources
from tests.differential import differential

#: Three coordinates one period apart, so a discount factor orders them and a
#: hand-computed optimum is one line of arithmetic.
SPEC = {
    'dimensions': {'g': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['g']}, 'growth': {'dims': []}, 'period': {'dims': ['g']}},
    'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'meet': {'dims': [], 'expression': 'sum(p) >= 12'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost / growth ** period)'},
}

SOURCES = {
    'g': ['a', 'b', 'c'],
    'cost': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [5.0, 5.0, 5.0]}),
    'growth': pl.DataFrame({'value': [1.1]}),
    'period': pl.DataFrame({'g': ['a', 'b', 'c'], 'value': [0.0, 1.0, 2.0]}),
}


def spec(expression: str, **patch) -> dict:
    """SPEC with another objective — the axis every test here varies."""
    return {**SPEC, 'objective': {'sense': 'minimize', 'expression': expression}, **patch}


@pytest.mark.parametrize(
    'expression',
    [
        pytest.param('sum(p * cost / growth ** period)', id='a-discount-factor'),
        pytest.param('sum(p * cost * growth ** period)', id='a-growth-factor'),
        pytest.param('sum(p * growth ** period ** period)', id='right-associative'),
        pytest.param('sum(p * cost / growth ** 2)', id='a-literal-exponent'),
        pytest.param('sum(p * cost / 2 ** period)', id='a-literal-base'),
    ],
)
def test_both_lanes_reach_one_optimum(expression):
    """The differential oracle: a power is one number per coordinate on either lane."""
    with differential(spec(expression), SOURCES):
        pass


def test_the_discount_factor_is_the_one_a_hand_computes():
    """A published number rather than a lane agreeing with itself.

    Unit costs discount to 5, 5/1.1 and 5/1.21, so the cheapest twelve units are
    all ten of `c` and two of `b` — an ordering a *linear* cost over equal
    `cost` could not produce, which is what makes the exponent load-bearing.
    """
    result = sps.solve(SPEC, SOURCES)
    assert result.objective == pytest.approx(10 * 5 / 1.21 + 2 * 5 / 1.1), (
        'the discounted optimum is not what the exponent says it is'
    )
    filled = dict(zip(*result.primal('p').sort('g').to_dict(as_series=False).values(), strict=True))
    assert filled == pytest.approx({'a': 0.0, 'b': 2.0, 'c': 10.0}), 'the cheapest period is filled first'


# ---------------------------------------------------------------------------
# the plan boundary: two guards no file can reach
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('expression', 'match'),
    [
        pytest.param(Power(Variable('p'), Constant(2.0)), 'power over variables', id='a-variable-under-it'),
        pytest.param(
            Power(Add(Constant(1.0), Parameter('growth')), Parameter('growth')),
            'refused at load',
            id='an-operand-that-adds',
        ),
    ],
)
def test_a_power_outside_the_language_is_refused_at_the_plan_boundary(expression, match):
    """Purpose-built, because `check` refuses both before a plan exists, so the plan is built by hand.

    Addition does not distribute over `**`, so a two-fragment base silently
    folded would compile `1 ** growth` and drop the rate. Both operands are the
    *scalar* parameter: the language refuses a variable-free part of an
    objective that carries dims, so a `period`-shaped one would be turned back
    before the guard under test could speak.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'growth': {'dims': []}, 'period': {'dims': ['g']}},
        'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'meet': {'dims': [], 'expression': 'sum(p) >= 1'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p * growth)'},
    }
    sources = {
        'g': ['a'],
        'growth': pl.DataFrame({'value': [1.1]}),
        'period': pl.DataFrame({'g': ['a'], 'value': [2.0]}),
    }
    model = sps.build(spec, sources)
    program = model._program
    patched = dataclasses.replace(program, objective=dataclasses.replace(program.objective, expression=expression))
    with pytest.raises((LanguageError, AssertionError), match=match):
        model._engine.build(patched, tidy_sources(program, dict(model._sources)))
