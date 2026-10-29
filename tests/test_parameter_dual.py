"""``Result.parameter_dual``: the rate at which the optimal objective rises with each value of a parameter.

The oracle is a finite difference, with no solver's convention in the loop:
move one row of the parameter by a step either way, re-solve, and the
objective moves by the parameter dual times the step. Every model here is non-degenerate at its data, so
the rate is two-sided and the step stays inside the basis. Two rates are
also checked by hand, against ``dual`` and ``primal``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from tests.test_reduced_cost import CASES as BOUND_CASES
from tests.test_reduced_cost import SOURCES as BOUND_SOURCES
from tests.test_reduced_cost import spec as bound_spec

if TYPE_CHECKING:
    from pathlib import Path

#: Small enough that no row leaves its basis, large enough to clear the solver's tolerances.
STEP = 1e-4

RTOL = 1e-5

#: Three generators of different efficiency against a load scaled by one number, ``c`` held up by its floor.
FLEET = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {'dtype': 'str'}},
    'parameters': {
        'cost': {'dims': ['g']},
        'cap': {'dims': ['g']},
        'floor': {'dims': ['g']},
        'eff': {'dims': ['g']},
        'load': {'dims': ['t']},
        'scale': {'dims': []},
    },
    'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 'floor', 'upper': 'cap'}}},
    'constraints': {'balance': {'dims': ['t'], 'expression': 'sum(p * eff, over=g) >= scale * load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(cost * p)'},
}
GENERATORS = ['a', 'b', 'c']
FLEET_SOURCES = {
    't': [0, 1, 2],
    'g': GENERATORS,
    'cost': pl.DataFrame({'g': GENERATORS, 'value': [1.0, 3.0, 6.0]}),
    'cap': pl.DataFrame({'g': GENERATORS, 'value': [4.0, 5.0, 10.0]}),
    'floor': pl.DataFrame({'g': GENERATORS, 'value': [0.0, 0.0, 0.5]}),
    'eff': pl.DataFrame({'g': GENERATORS, 'value': [0.9, 1.0, 0.8]}),
    'load': pl.DataFrame({'t': [0, 1, 2], 'value': [3.0, 6.0, 10.0]}),
    'scale': pl.DataFrame({'value': [1.0]}),
}

#: A cyclic store whose round-trip efficiency is read through a shift, beside one cheap and one dear generator.
STORAGE = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {'dtype': 'str'}},
    'parameters': {
        'cost': {'dims': ['g']},
        'cap': {'dims': ['g']},
        'load': {'dims': ['t']},
        'eta': {'dims': ['t']},
    },
    'variables': {
        'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'charge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 4}},
        'discharge': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 4}},
        'soc': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 20}},
    },
    'constraints': {
        'balance': {'dims': ['t'], 'expression': 'sum(p, over=g) + discharge - charge == load'},
        'level': {
            'dims': ['t'],
            'expression': "soc == shift(soc + charge * eta, along=t, offset=1, edge='wrap') - discharge",
        },
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(cost * p)'},
}
STORAGE_SOURCES = {
    't': [0, 1, 2, 3],
    'g': ['cheap', 'dear'],
    'cost': pl.DataFrame({'g': ['cheap', 'dear'], 'value': [1.0, 10.0]}),
    'cap': pl.DataFrame({'g': ['cheap', 'dear'], 'value': [6.0, 20.0]}),
    'load': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [3.0, 4.0, 8.0, 9.0]}),
    'eta': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [0.9, 0.85, 0.8, 0.95]}),
}

#: A quadratic cost, so a bound's reduced cost reads the primal, and an availability with rows only for ``a``.
#: ``a``'s limit binds at ``t=0`` and ``b``'s cap at ``t=1`` and ``t=2``.
QUADRATIC = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}, 'g': {'dtype': 'str'}},
    'parameters': {
        'cost': {'dims': ['g']},
        'q': {'dims': ['g']},
        'cap': {'dims': ['g']},
        'avail': {'dims': ['t', 'g'], 'missing': 'absent'},
        'load': {'dims': ['t']},
    },
    'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {
        'balance': {'dims': ['t'], 'expression': 'sum(p, over=g) == load'},
        'limit': {'dims': ['t', 'g'], 'expression': 'p <= avail * cap'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(cost * p) + sum(q * p * p)'},
}
QUADRATIC_SOURCES = {
    't': [0, 1, 2],
    'g': ['a', 'b'],
    'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0, 2.0]}),
    'q': pl.DataFrame({'g': ['a', 'b'], 'value': [0.5, 0.2]}),
    'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [6.0, 3.5]}),
    'avail': pl.DataFrame({'t': [0, 1, 2], 'g': ['a', 'a', 'a'], 'value': [0.25, 0.5, 0.8]}),
    'load': pl.DataFrame({'t': [0, 1, 2], 'value': [4.0, 6.0, 8.0]}),
}


def finite_differences(model: dict, sources: dict, name: str, solver_name: str) -> pl.DataFrame:
    """Each row of parameter *name* moved by ``STEP`` either way alone, as ``(dims…, rate)``.

    Central, because a coefficient's rate is not constant: a cost paid per
    unit of output over an efficiency is a curve in the efficiency.
    """
    table = sources[name]

    def objective(i: int, step: float) -> float:
        moved = table.with_columns(pl.when(pl.int_range(pl.len()) == i).then(pl.col('value') + step).otherwise('value'))
        with sps.solve(model, {**sources, name: moved}, solver_name=solver_name) as answer:
            return answer.objective

    rates = [(objective(i, STEP) - objective(i, -STEP)) / (2 * STEP) for i in range(table.height)]
    return table.drop('value').with_columns(pl.Series('rate', rates))


MODELS = [
    *(pytest.param(FLEET, FLEET_SOURCES, name, id=f'fleet-{name}') for name in FLEET['parameters']),
    *(pytest.param(STORAGE, STORAGE_SOURCES, name, id=f'storage-{name}') for name in STORAGE['parameters']),
    *(pytest.param(QUADRATIC, QUADRATIC_SOURCES, name, id=f'quadratic-{name}') for name in QUADRATIC['parameters']),
    *(
        pytest.param(bound_spec(*case.values), BOUND_SOURCES, name, id=f'{case.id}-{name}')
        for case in BOUND_CASES
        for name in ('price', 'floor', 'cap', 'total')
    ),
]


@pytest.mark.parametrize(('model', 'sources', 'name'), MODELS)
def test_the_marginal_is_what_the_objective_pays_per_unit_of_each_value(
    solver_name: str, model: dict, sources: dict, name: str
) -> None:
    """Every place a parameter is read: an objective coefficient, a quadratic one, a row's coefficient, a right side, a bound on either side, a scalar, a value under a shift and one with rows missing — under either sense and however the row is written."""
    if model is QUADRATIC and solver_name == 'xpress':
        pytest.skip('the xpress sink takes no quadratic objective')
    with sps.solve(model, sources, solver_name=solver_name) as answer:
        parameter_dual = answer.parameter_dual(name)
    expected = finite_differences(model, sources, name, solver_name)
    dims = [c for c in expected.columns if c != 'rate']
    assert parameter_dual.columns == [*dims, 'value'], "a parameter dual is keyed like the parameter's own table"
    joined = parameter_dual.join(expected, on=dims) if dims else parameter_dual.hstack(expected)
    assert joined.height == sources[name].height, "a parameter dual has one row per row of the parameter's table"
    assert joined['value'].to_list() == pytest.approx(joined['rate'].to_list(), rel=RTOL, abs=RTOL), (
        f"each row of '{name}' moves the objective at its parameter dual"
    )


def test_a_right_side_is_priced_at_its_dual_and_a_cost_at_its_primal() -> None:
    """By hand, on ``FLEET``: the load is read once per row on the right, and the cost once per column in the objective."""
    with sps.solve(FLEET, FLEET_SOURCES) as answer:
        assert answer.parameter_dual('load')['value'].to_list() == pytest.approx(
            answer.dual('balance')['value'].to_list()
        ), "a right side's rate is its row's dual"
        used = answer.primal('p').group_by('g', maintain_order=True).agg(pl.col('value').sum())
        assert answer.parameter_dual('cost').equals(used), "a cost's rate is the output it is paid on"


def test_a_value_a_mask_reads_is_held_where_the_mask_put_it() -> None:
    """The model is held as built: a value that switches a variable on prices the model where it is still off.

    ``dispatch.yaml`` builds no ``p`` for a generator without capacity. Its
    parameter_dual is that of the model without it, though a re-solve with any
    capacity at all is cheaper.
    """
    generators = ['wind', 'gas']
    sources = {
        'generator': generators,
        'snapshot': [0, 1],
        'p_max': pl.DataFrame({'generator': generators, 'value': [0.0, 100.0]}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0, 50.0]}),
        'load': pl.DataFrame({'snapshot': [0, 1], 'value': [40.0, 60.0]}),
    }
    with sps.solve('examples/dispatch.yaml', sources) as answer:
        held = answer.parameter_dual('p_max').filter(pl.col('generator') == 'wind')['value'].item()
        base = answer.objective
    some_wind = sources['p_max'].with_columns(
        pl.when(pl.col('generator') == 'wind').then(1.0).otherwise('value').alias('value')
    )
    with sps.solve('examples/dispatch.yaml', {**sources, 'p_max': some_wind}) as answer:
        assert answer.objective < base, 'a re-solve with wind built is cheaper'
    assert held == 0.0, 'the parameter dual is that of the model as built, where wind has no column'


@pytest.mark.parametrize(
    ('expression', 'refusal'),
    [
        pytest.param('sum(p * eff, over=g) >= load / scale', 'divides by it', id='a-divisor'),
        pytest.param('sum(p * eff, over=g) >= scale ** 2 * load', 'raises it to a power', id='a-power'),
        pytest.param('sum(p * eff, over=g) >= scale * scale * load', 'multiplies it by itself', id='a-square'),
    ],
)
def test_a_parameter_read_other_than_linearly_is_refused_with_the_rewrite(expression: str, refusal: str) -> None:
    model = {**FLEET, 'constraints': {'balance': {'dims': ['t'], 'expression': expression}}}
    with sps.solve(model, FLEET_SOURCES) as answer, pytest.raises(SpecsolveError, match=refusal) as refused:
        answer.parameter_dual('scale')
    assert 'as a parameter of its own' in str(refused.value), 'the refusal names the rewrite'


def test_an_unknown_parameter_is_a_key_error_that_names_the_near_miss() -> None:
    with sps.solve(FLEET, FLEET_SOURCES) as answer, pytest.raises(KeyError, match=r"'loads'.*'load'"):
        answer.parameter_dual('loads')


def test_an_integer_variable_leaves_no_marginal_and_says_why() -> None:
    """A parameter dual is read off the duals, so it is refused where they are, with their message."""
    integer = bound_spec('minimize', 'sum(p, over=g) >= total', domain='integer')
    with sps.solve(integer, BOUND_SOURCES) as answer, pytest.raises(SpecsolveError, match='mixed-integer'):
        answer.parameter_dual('cap')


def test_a_closed_result_refuses_a_marginal() -> None:
    answer = sps.solve(FLEET, FLEET_SOURCES)
    answer.close()
    with pytest.raises(SpecsolveError, match='closed'):
        answer.parameter_dual('load')


def test_a_result_keeps_its_marginal_after_the_model_is_updated() -> None:
    with sps.build(FLEET, FLEET_SOURCES) as model:
        answer = model.solve()
        before = answer.parameter_dual('cap')
        model.update({'load': FLEET_SOURCES['load'].with_columns(pl.col('value') * 2)}).solve()
        assert answer.parameter_dual('cap').equals(before), 'a result reads the build it answered'


def test_an_archive_reads_the_marginal_a_live_solve_gives(tmp_path: Path) -> None:
    with sps.solve(STORAGE, STORAGE_SOURCES, archive=tmp_path / 'run') as answer:
        live = answer.parameter_dual('eta')
    assert sps.scan_archive(tmp_path / 'run').result.parameter_dual('eta').equals(live)


def test_a_saved_answer_without_its_model_says_where_to_ask(tmp_path: Path) -> None:
    with sps.solve(FLEET, FLEET_SOURCES) as answer:
        answer.save(tmp_path)
    with pytest.raises(SpecsolveError, match='no model behind it'):
        sps.load_result(tmp_path).parameter_dual('load')
