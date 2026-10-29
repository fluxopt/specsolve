"""``Result.parameter_dual`` where a value sits on a ``where:`` threshold: the rate as it rises past it, from one solve.

The oracle is a forward re-solve: raise one row by a step and solve again. A
finite parameter_dual matches the step's rate; an infinite one is a jump, so the
re-solve moves the objective the same way by far more than the step. The
fleet is ``dispatch.yaml``'s, with solar at zero capacity: wind runs alone in
snapshot 0, and gas sets the price in the other three.
"""

from __future__ import annotations

import math

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveWarning

#: Small enough that no row leaves its basis, large enough to clear the solver's tolerances.
STEP = 1e-4

RTOL = 1e-5

#: A re-solve whose rate passes this is a jump rather than a slope.
JUMP = 1e3

GENERATORS = ['wind', 'solar', 'gas']
SNAPSHOTS = [0, 1, 2, 3]
SOURCES = {
    'generator': GENERATORS,
    'snapshot': SNAPSHOTS,
    'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 0.0, 200.0]}),
    'cost': pl.DataFrame({'generator': GENERATORS, 'value': [10.0, 25.0, 50.0]}),
    'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [60.0, 120.0, 180.0, 90.0]}),
    'avail': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 0.4, 1.0]}),
    'p_min': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 0.0, 0.0]}),
    'tight': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 1.0, 1.0]}),
    'retire': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 1.0, 0.0]}),
    'ramp': pl.DataFrame({'generator': GENERATORS, 'value': [0.0, 2.0, 0.0]}),
    'keep': pl.DataFrame({'generator': GENERATORS, 'value': [True, True, True]}),
}


def fleet(
    *,
    where: str = 'p_max > 0',
    upper: str | float | None = 'p_max',
    rows: dict | None = None,
    sense: str = 'minimize',
) -> dict:
    """The fleet, its mask, bound and extra rows as given, minimising cost or maximising its negative."""
    per_generator = ('p_max', 'cost', 'avail', 'p_min', 'tight', 'retire', 'ramp')
    return {
        'dimensions': {'snapshot': {'dtype': 'int', 'ordered': True}, 'generator': {'dtype': 'str'}},
        'parameters': {name: {'dims': ['generator']} for name in per_generator}
        | {'load': {'dims': ['snapshot']}, 'keep': {'dims': ['generator'], 'dtype': 'bool'}},
        'variables': {'p': {'dims': ['snapshot', 'generator'], 'where': where, 'bounds': {'lower': 0, 'upper': upper}}},
        'constraints': {
            'balance': {'dims': ['snapshot'], 'expression': 'sum(p, over=generator) == load'},
            **(rows or {}),
        },
        'objective': {
            'sense': sense,
            'expression': 'sum(p * cost)' if sense == 'minimize' else '-sum(p * cost)',
        },
    }


def risen(model: dict, name: str, generator: str) -> float:
    """The rate of a forward re-solve with *generator*'s row of *name* raised by ``STEP``, ``inf`` where it has no solution.

    Every model this is asked of where the re-solve fails minimises, so losing
    every solution is the objective rising without bound.
    """
    with sps.solve(model, SOURCES) as answer:
        base = answer.objective
    moved = SOURCES[name].with_columns(
        pl.when(pl.col('generator') == generator).then(pl.col('value') + STEP).otherwise('value').alias('value')
    )
    with sps.solve(model, {**SOURCES, name: moved}) as answer:
        return (answer.objective - base) / STEP if answer.has_primal else math.inf


def parameter_dual_of(model: dict, name: str, generator: str) -> float:
    with sps.solve(model, SOURCES) as answer:
        return answer.parameter_dual(name).filter(pl.col('generator') == generator)['value'].item()


CAP = {'cap': {'dims': ['snapshot', 'generator'], 'expression': 'p <= avail * p_max'}}
FLOOR = {'floor': {'dims': ['snapshot', 'generator'], 'where': 'p_min > 0', 'expression': 'p >= p_min'}}


@pytest.mark.parametrize(
    ('model', 'name', 'generator', 'expected'),
    [
        pytest.param(fleet(), 'p_max', 'solar', -75.0, id='a-column-whose-bound-is-the-value'),
        pytest.param(fleet(sense='maximize'), 'p_max', 'solar', 75.0, id='the-same-maximized'),
        pytest.param(fleet(upper=None, rows=CAP), 'p_max', 'solar', -30.0, id='a-zero-slack-row-caps-the-column'),
        pytest.param(
            fleet(
                upper=None,
                rows={'cap': {'dims': ['snapshot', 'generator'], 'expression': '2 * p <= 2 * avail * p_max'}},
            ),
            'p_max',
            'solar',
            -30.0,
            id='a-cap-is-read-per-unit-of-its-column',
        ),
        pytest.param(fleet(rows=FLOOR), 'p_min', 'gas', 40.0, id='a-zero-slack-row-forces-a-column-up'),
        pytest.param(fleet(rows=FLOOR), 'p_min', 'wind', 0.0, id='a-row-a-running-column-already-meets'),
    ],
)
def test_a_value_on_a_threshold_is_priced_past_it(model: dict, name: str, generator: str, expected: float) -> None:
    """Solar is cheaper than gas in the three snapshots gas sets the price in, by 25 each, and so on per case."""
    rate = parameter_dual_of(model, name, generator)
    assert rate == pytest.approx(expected, rel=RTOL), f"{generator}'s row of '{name}' is priced past its threshold"
    assert risen(model, name, generator) == pytest.approx(rate, rel=RTOL, abs=RTOL), 'a forward re-solve agrees'


@pytest.mark.parametrize(
    ('model', 'name', 'generator', 'sign'),
    [
        pytest.param(fleet(upper=100.0), 'p_max', 'solar', -1, id='a-column-with-a-fixed-bound-it-would-use'),
        pytest.param(
            fleet(rows={'floor': {'dims': ['snapshot', 'generator'], 'where': 'p_min > 0', 'expression': 'p >= 10'}}),
            'p_min',
            'gas',
            1,
            id='a-row-the-solve-violates',
        ),
        pytest.param(
            fleet(rows={'curb': {'dims': ['snapshot', 'generator'], 'where': 'tight <= 0', 'expression': 'p <= 30'}}),
            'tight',
            'wind',
            -1,
            id='a-row-with-a-dual-switched-off',
        ),
        pytest.param(fleet(where='retire <= 0'), 'retire', 'gas', 1, id='a-column-the-solve-uses-switched-off'),
        pytest.param(
            fleet(rows={'floor': {'dims': ['snapshot', 'generator'], 'expression': 'p >= ramp * p_max'}}),
            'p_max',
            'solar',
            1,
            id='a-floor-that-rises-past-its-cap',
        ),
    ],
)
def test_a_rise_that_jumps_the_objective_is_an_infinite_rate(model: dict, name: str, generator: str, sign: int) -> None:
    rate = parameter_dual_of(model, name, generator)
    assert rate == sign * math.inf, f"{generator}'s row of '{name}' jumps the objective"
    assert sign * risen(model, name, generator) > JUMP, 'a forward re-solve jumps the same way'


def test_a_zero_slack_row_across_several_columns_is_held_as_built_and_named() -> None:
    """Which of the columns moves takes a solve to decide, so the rate is that of the model as built."""
    pool = {
        'pool': {'dims': ['snapshot'], 'where': 'p_min > 0', 'expression': 'sum(p, over=generator) - load <= p_min'}
    }
    model = fleet(rows=pool)
    del model['parameters']['p_min']
    model['parameters']['p_min'] = {'dims': []}
    sources = {**SOURCES, 'p_min': pl.DataFrame({'value': [0.0]})}
    with (
        sps.solve(model, sources) as answer,
        pytest.warns(SpecsolveWarning, match='several columns, which only a solve decides'),
    ):
        rate = answer.parameter_dual('p_min')['value'].item()
    assert rate == 0.0, 'held as built: the row is redundant beside the balance'


def test_a_mask_read_through_a_sum_is_held_as_built_and_named() -> None:
    model = fleet(
        rows={
            'any': {
                'dims': ['snapshot'],
                'where': 'sum(p_min, over=generator) > 0',
                'expression': 'sum(p, over=generator) >= 0',
            }
        }
    )
    with sps.solve(model, SOURCES) as answer, pytest.warns(SpecsolveWarning, match='through a shift, a sum'):
        answer.parameter_dual('p_min')


def test_a_value_on_no_threshold_warns_of_nothing(recwarn: pytest.WarningsRecorder) -> None:
    with sps.solve(fleet(), SOURCES) as answer:
        answer.parameter_dual('cost')
    assert not [w for w in recwarn if issubclass(w.category, SpecsolveWarning)], 'no threshold, no warning'


def test_a_mask_on_the_column_that_sums_the_parameter_is_held_as_built() -> None:
    """The sum reads every generator's row, so which one a flip belongs to is not one row: nothing is charged."""
    model = fleet(where='p_max > 0 and sum(p_min, over=generator) <= 0')
    with sps.solve(model, SOURCES) as answer, pytest.warns(SpecsolveWarning, match='through a shift, a sum'):
        rates = answer.parameter_dual('p_min')['value'].to_list()
    assert rates == [0.0, 0.0, 0.0], 'held as built: p_min is read nowhere else'


def test_a_when_on_a_threshold_is_named() -> None:
    model = fleet(rows={'roomy': {'dims': ['snapshot', 'generator'], 'expression': 'p <= room'}})
    model['expressions'] = {
        'room': {
            'dims': ['generator'],
            'cases': {'tight': {'when': 'p_min > 0', 'expression': 'p_max'}},
            'otherwise': 1000,
        }
    }
    with sps.solve(model, SOURCES) as answer, pytest.warns(SpecsolveWarning, match='a when: reads'):
        answer.parameter_dual('p_min')


def test_a_quadratic_row_a_rise_switches_on_is_held_as_built() -> None:
    """Only gurobi takes a quadratic row, and leaves its duals only when asked."""
    pytest.importorskip('gurobipy', reason='a quadratic constraint has no other solver sink')
    model = fleet(rows={'q': {'dims': ['snapshot', 'generator'], 'where': 'p_min > 0', 'expression': 'p * p <= 1e6'}})
    with (
        sps.solve(model, SOURCES, solver_name='gurobi', solver_options={'QCPDual': 1}) as answer,
        pytest.warns(SpecsolveWarning, match='is quadratic'),
    ):
        answer.parameter_dual('p_min')


def test_a_bool_parameter_in_a_mask_is_read_nowhere_in_the_math(recwarn: pytest.WarningsRecorder) -> None:
    with sps.solve(fleet(where='p_max > 0 and keep'), SOURCES) as answer:
        rates = answer.parameter_dual('keep')['value'].to_list()
    assert rates == [0.0, 0.0, 0.0], 'a bool has no rise, so nothing crosses'
    assert not [w for w in recwarn if issubclass(w.category, SpecsolveWarning)], 'and nothing is warned'
