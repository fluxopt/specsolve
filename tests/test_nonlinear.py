"""PROTOTYPE: nonlinear terms, built as defined columns and solved by Gurobi and Ipopt.

No differential oracle exists for these: linopy has no nonlinear term. Three
weaker ones replace it:

1. **A closed form.** Each case minimises ``sum(x)`` against one monotone
   function of ``x``, so the optimum is the function's inverse at the data.
2. **Two solvers.** Gurobi solves globally from the expression trees the sink
   writes, Ipopt locally from the derivatives this package computes. A wrong
   derivative shows up as Ipopt alone missing the closed form.
3. **HiGHS on an LP.** Ipopt's primal, duals and activity on a model HiGHS can
   solve, which pins the dual sign convention.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError

pytest.importorskip('gurobipy', reason='the global nonlinear sink')
pytest.importorskip('cyipopt', reason='the local nonlinear sink')

SINKS = [pytest.param('gurobi', id='gurobi'), pytest.param('ipopt', id='ipopt')]

#: Ipopt's own default tolerance is 1e-8 on the scaled problem, Gurobi's
#: ``FeasibilityTol`` 1e-6 on a spatial search.
RTOL = 1e-5

DATA = [2.0, 5.0]


def _model(expression: str, *, lower: float = 0.0, upper: float = 10.0, extra: dict | None = None) -> dict:
    """``min sum(x)`` subject to ``expression >= d`` at each of two coordinates."""
    return {
        'dimensions': {'i': {}},
        'parameters': {'d': {'dims': ['i']}, 'w': {'dims': ['i']}},
        'variables': {
            'x': {'dims': ['i'], 'bounds': {'lower': lower, 'upper': upper}},
            'y': {'dims': ['i'], 'bounds': {'lower': 'w', 'upper': 'w'}},
            **(extra or {}),
        },
        'constraints': {'c': {'dims': ['i'], 'expression': f'{expression} >= d'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=i)'},
    }


def _sources(data: list[float]) -> dict:
    return {
        'i': ['a', 'b'],
        'd': pl.DataFrame({'i': ['a', 'b'], 'value': data}),
        'w': pl.DataFrame({'i': ['a', 'b'], 'value': [2.0, 3.0]}),
    }


CASES = [
    pytest.param('exp(x)', DATA, lambda d, w: math.log(d), {}, id='exp'),
    pytest.param('log(x)', [0.5, 1.0], lambda d, w: math.exp(d), {'lower': 0.1}, id='log'),
    pytest.param('sqrt(x)', DATA, lambda d, w: d * d, {'upper': 30.0}, id='sqrt'),
    pytest.param('sin(x)', [0.25, 0.5], lambda d, w: math.asin(d), {'upper': 1.5}, id='sin'),
    pytest.param('cos(x)', [0.25, 0.5], lambda d, w: -math.acos(d), {'lower': -1.5, 'upper': 0.0}, id='cos'),
    pytest.param('x ** 3', DATA, lambda d, w: d ** (1 / 3), {}, id='a-variable-base'),
    pytest.param('2 ** x', DATA, lambda d, w: math.log2(d), {}, id='a-variable-exponent'),
    pytest.param('x ** y', DATA, lambda d, w: d ** (1 / w), {'lower': 0.1}, id='a-variable-base-and-exponent'),
    pytest.param('x / y', DATA, lambda d, w: d * w, {'upper': 20.0}, id='a-variable-divisor'),
    pytest.param('x * x * x', DATA, lambda d, w: d ** (1 / 3), {}, id='degree-three'),
    pytest.param(
        'exp(2 * x + 1) / 3', DATA, lambda d, w: (math.log(3 * d) - 1) / 2, {'lower': -5.0}, id='affine-inside'
    ),
    pytest.param('exp(log(x) * 2)', DATA, lambda d, w: d**0.5, {'lower': 0.1}, id='a-function-of-a-function'),
]


@pytest.mark.parametrize('sink', SINKS)
@pytest.mark.parametrize(('expression', 'data', 'inverse', 'bounds'), CASES)
def test_the_optimum_is_the_inverse_at_the_data(sink, expression, data, inverse, bounds):
    """Each row binds at the optimum, so ``x`` is the function's inverse at ``d``."""
    with sps.solve(_model(expression, **bounds), _sources(data), solver_name=sink) as result:
        assert result.is_ok, f'{sink} on {expression}: {result.status}'
        expected = [inverse(d, w) for d, w in zip(data, [2.0, 3.0], strict=True)]
        assert result.primal('x')['value'].to_list() == pytest.approx(expected, rel=RTOL, abs=RTOL), (
            f'x is the inverse of {expression} at the data'
        )


@pytest.mark.parametrize('sink', SINKS)
def test_a_function_of_a_sum_takes_the_sum_at_each_coordinate(sink):
    """``exp(sum(x, over=j))`` is one column per ``i``, its argument summed over ``j`` before the function."""
    spec = {
        'dimensions': {'i': {}, 'j': {}},
        'parameters': {'d': {'dims': ['i']}},
        'variables': {'x': {'dims': ['i', 'j'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': {'dims': ['i'], 'expression': 'exp(sum(x, over=j)) >= d'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=[i, j])'},
    }
    sources = {'i': ['a', 'b'], 'j': ['p', 'q', 'r'], 'd': pl.DataFrame({'i': ['a', 'b'], 'value': DATA})}
    with sps.solve(spec, sources, solver_name=sink) as result:
        assert result.objective == pytest.approx(sum(map(math.log, DATA)), rel=RTOL), (
            'at each coordinate the xs sum to log(d)'
        )


@pytest.mark.parametrize('sense', ['minimize', 'maximize'])
def test_an_lp_on_ipopt_answers_as_highs_does(sense):
    """Primal, objective, duals and activity — the sign of a dual is the contract's, not Ipopt's."""
    spec = {
        'dimensions': {'i': {}},
        'parameters': {'cost': {'dims': ['i']}},
        'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {
            'floor': {'dims': [], 'expression': 'sum(x, over=i) >= 4'},
            'cap': {'dims': ['i'], 'expression': 'x <= 3'},
            'pair': {'dims': [], 'expression': 'sum(cost * x, over=i) <= 2.5'},
        },
        'objective': {'sense': sense, 'expression': 'sum(cost * x, over=i) + sum(x, over=i)'},
    }
    sources = {'i': ['a', 'b', 'c'], 'cost': pl.DataFrame({'i': ['a', 'b', 'c'], 'value': [1.0, 0.5, 0.25]})}
    outputs = {'activity'}
    with (
        sps.solve(spec, sources, solver_name='highs', outputs=outputs) as highs,
        sps.solve(spec, sources, solver_name='ipopt', outputs=outputs) as ipopt,
    ):
        assert ipopt.objective == pytest.approx(highs.objective, rel=RTOL), 'one LP, one optimum'
        assert ipopt.primal('x')['value'].to_list() == pytest.approx(
            highs.primal('x')['value'].to_list(), rel=RTOL, abs=1e-6
        ), 'the LP has one optimal vertex, which an interior point reaches to its tolerance'
        for name in ('floor', 'cap', 'pair'):
            assert ipopt.dual(name)['value'].to_list() == pytest.approx(
                highs.dual(name)['value'].to_list(), rel=RTOL, abs=1e-6
            ), f'the dual of {name!r}'
            assert ipopt.activity(name)['value'].to_list() == pytest.approx(
                highs.activity(name)['value'].to_list(), rel=RTOL, abs=1e-6
            ), f'the activity of {name!r}'


def test_a_quadratic_row_on_ipopt_reaches_gurobis_optimum():
    """``p · q >= 4`` with a cap, the model `test_quadratic_constraint` solves by hand: ``p = q = 2``."""
    spec = {
        'dimensions': {'g': {}},
        'variables': {
            'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
            'q': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {
            'cap': {'dims': [], 'expression': 'sum(p, over=g) <= 9'},
            'coupled': {'dims': ['g'], 'expression': 'p * q >= 4'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(p + q, over=g)'},
    }
    with sps.solve(spec, {'g': ['a', 'b']}, solver_name='ipopt') as result:
        assert result.objective == pytest.approx(8.0, rel=RTOL), 'p = q = 2 on both generators'


def test_a_sink_without_the_capability_names_the_ones_with_it():
    """HiGHS, Xpress and both file formats refuse, through the capability table every construct uses."""
    with pytest.raises(SpecsolveError, match=r"'highs' sink cannot take a nonlinear term.*gurobi, ipopt"):
        sps.solve(_model('exp(x)'), _sources(DATA), solver_name='highs')


def test_a_function_of_data_folds_to_a_number_and_builds_no_column():
    """``exp(d)`` holds no variable, so it is a coefficient, and HiGHS solves the model."""
    spec = _model('x - exp(log(d))', lower=-10)
    with sps.build(spec, _sources(DATA)) as model:
        assert model._engine._model.handoff.nl.height == 0, 'a function of a parameter defines no column'
        with model.solve() as result:
            assert result.primal('x')['value'].to_list() == pytest.approx([4.0, 10.0], rel=RTOL), 'x = d + w'


@pytest.mark.parametrize('sink', SINKS)
def test_evaluate_reads_a_function_at_the_solution(sink):
    """At a read every variable is a value, so the function is computed rather than defined."""
    with sps.solve(_model('exp(x)'), _sources(DATA), solver_name=sink) as result:
        assert result.evaluate('exp(x) + x / y')['value'].to_list() == pytest.approx(
            [d + math.log(d) / w for d, w in zip(DATA, [2.0, 3.0], strict=True)], rel=RTOL
        ), 'exp(x) is d where the row binds, and x / y is log(d) / w'


def test_a_defined_column_is_not_a_variable_the_result_reports():
    """The columns sit after every declared variable, so a declaration's share of the primal never reaches one."""
    with sps.build(_model('exp(x) + x / y'), _sources(DATA)) as model:
        handoff = model._engine._model.handoff
        assert handoff.nl['op'].cast(str).to_list() == ['exp', 'exp', 'div', 'div'], (
            'one column per term and coordinate'
        )
        assert handoff.nl['col'].min() == 4, 'numbered after the four declared columns'
        assert handoff.column_count == 8


def test_the_row_reader_names_a_defined_column_by_its_op():
    """A defined column has no declaration to name it, so it reads as its op and its column."""
    with sps.build(_model('exp(x) + x / y'), _sources(DATA)) as model:
        assert str(model.row('c', i='a')) == 'c[i=a]: +1 exp(…)[#4] +1 div(…)[#6] >= 2'


@pytest.mark.xfail(strict=True, reason='PROTOTYPE: the refusal reads as a KeyError naming no declared variable')
def test_a_nonlinear_model_reports_no_reduced_costs():
    """The equations' multipliers are not in the dual, so a reduced cost computed from it would be wrong."""
    with (
        sps.solve(_model('exp(x)'), _sources(DATA), solver_name='ipopt', outputs={'reduced_cost'}) as result,
        pytest.raises(SpecsolveError, match='nonlinear'),
    ):
        result.reduced_cost('x')
