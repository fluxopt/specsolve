"""``**`` over parameters: the one exponent the language reads off the data.

The operator is degree 0 in variables wherever it appears, so it is not a
ceiling question at all — it is the arithmetic ``*`` already does, spelled the
way a discount factor is written. What it costs is one refusal per way a
variable can get underneath it. An operand that adds is added up to one number
per coordinate before the power is taken, so ``(1 + rate) ** period`` is the
discount factor it reads as.
"""

from __future__ import annotations

import dataclasses

import polars as pl
import pytest
from mathspec.program import Constant, Power, Variable

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
        pytest.param('sum(p * cost / (1 + growth) ** period)', id='a-base-that-adds'),
        pytest.param('sum(p * cost / growth ** (period + 1))', id='an-exponent-that-adds'),
        pytest.param('sum(p * cost / (growth - 0.5) ** (2 * period - 1))', id='both-operands-add'),
    ],
)
def test_both_lanes_reach_one_optimum(expression):
    """The differential oracle: a power is one number per coordinate on either lane."""
    with differential(spec(expression), SOURCES):
        pass


@pytest.mark.parametrize(
    ('model', 'sources'),
    [
        pytest.param(SPEC, SOURCES, id='a-growth-parameter'),
        pytest.param(
            spec(
                'sum(p * cost / (1 + rate) ** period)',
                parameters={**SPEC['parameters'], 'rate': {'dims': []}},
            ),
            {**SOURCES, 'rate': pl.DataFrame({'value': [0.1]})},
            id='a-base-that-adds',
        ),
    ],
)
def test_the_discount_factor_is_the_one_a_hand_computes(model, sources):
    """A published number rather than a lane agreeing with itself.

    Unit costs discount to 5, 5/1.1 and 5/1.21, so the cheapest twelve units are
    all ten of `c` and two of `b` — an ordering a *linear* cost over equal
    `cost` could not produce, which is what makes the exponent load-bearing.
    `(1 + rate) ** period` is the same factor, added up before the power: taken
    term by term it would read `1 ** period`, and no unit would be discounted.
    """
    result = sps.solve(model, sources)
    assert result.objective == pytest.approx(10 * 5 / 1.21 + 2 * 5 / 1.1), (
        'the discounted optimum is not what the exponent says it is'
    )
    filled = dict(zip(*result.primal('p').sort('g').to_dict(as_series=False).values(), strict=True))
    assert filled == pytest.approx({'a': 0.0, 'b': 2.0, 'c': 10.0}), 'the cheapest period is filled first'


def test_a_missing_row_under_a_power_that_adds_reads_as_zero_on_both_lanes():
    """`(1 + period) ** 2` is 1 where `period` has no row, as a missing parameter row reads anywhere but a divisor.

    Per unit, `a` costs 5, `b` 20 and `c` 5, so twelve units are ten of `a` and
    two of `c`. Dropped at `c` instead, the term would cost nothing there.
    """
    sources = {**SOURCES, 'period': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 1.0]})}
    with differential(spec('sum(p * cost * (1 + period) ** 2)'), sources) as run:
        assert float(run.result.objective) == pytest.approx(60.0), 'c is priced at 5 * 1 ** 2, not left free'


# ---------------------------------------------------------------------------
# the plan boundary: a guard no file can reach
# ---------------------------------------------------------------------------


def test_a_power_over_a_variable_is_refused_at_the_plan_boundary():
    """Purpose-built, because `check` refuses it before a plan exists, so the plan is built by hand."""
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
    expression = Power(Variable('p'), Constant(2.0))
    patched = dataclasses.replace(program, objective=dataclasses.replace(program.objective, expression=expression))
    with pytest.raises((LanguageError, AssertionError), match='power over variables'):
        model._engine.build(patched, tidy_sources(program, dict(model._sources)))
