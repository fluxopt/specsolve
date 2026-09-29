"""`sps.evaluate(spec, sources, expression)`: a spec's arithmetic, with no solver.

A spec with no variables is a calculation, so each expression has a value with
no solver. The engine is the one a solve reads named expressions through
(``test_expression_reader.py``), handed no solution.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import LanguageError, SpecsolveError
from specsolve.relational.engines.polars.compiler import PolarsCompiler

SPEC = {
    'dimensions': {
        'snapshot': {'dtype': 'int'},
        'generator': {'dtype': 'str'},
        'node': {'dtype': 'str'},
    },
    'parameters': {
        'dispatch': {'dims': ['snapshot', 'generator']},
        'cost': {'dims': ['generator']},
    },
    'relations': {'bus': {'key': 'generator', 'values': 'node'}},
    'expressions': {
        'cost_by_gen': 'dispatch * cost',
        'total_cost': 'sum(dispatch * cost)',
        'served': 'sum(dispatch, over=generator)',
        'by_node': 'sum(dispatch, by=bus, over=generator, into=node)',
    },
}

GENERATORS = ['wind', 'gas']


def sources() -> dict[str, object]:
    return {
        'snapshot': [0, 1, 2],
        'generator': GENERATORS,
        'node': ['n1', 'n2'],
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 50.0]}),
        'dispatch': pl.DataFrame(
            {
                'snapshot': [0, 0, 1, 1, 2, 2],
                'generator': GENERATORS * 3,
                'value': [10.0, 2.0, 20.0, 3.0, 5.0, 7.0],
            }
        ),
        'bus': pl.DataFrame({'generator': GENERATORS, 'node': ['n1', 'n2']}),
    }


def evaluate(expression: str | dict) -> pl.DataFrame:
    return sps.evaluate(SPEC, sources(), expression)


def test_a_scalar_expression_is_the_arithmetic_the_data_implies():
    frame = evaluate('total_cost')
    assert frame.columns == ['value'] and frame.height == 1, 'an expression with no dims is a single value row'
    assert frame.item() == pytest.approx(635.0), 'sum(dispatch * cost) is 10·1 + 2·50 + 20·1 + 3·50 + 5·1 + 7·50 = 635'


def test_a_sum_over_one_of_two_dims_is_the_per_coordinate_total():
    got = dict(zip(*evaluate('served').sort('snapshot'), strict=True))
    assert got == pytest.approx({0: 12.0, 1: 23.0, 2: 12.0}), (
        'sum(dispatch, over=generator) adds the two generators at each snapshot'
    )


def test_a_grouped_sum_relabels_through_a_relation():
    frame = evaluate('by_node').sort('snapshot', 'node')
    assert frame.columns == ['snapshot', 'node', 'value'], (
        'sum(by=bus, over=generator, into=node) replaces generator with the node it maps to'
    )
    got = {(s, n): v for s, n, v in frame.iter_rows()}
    assert got == pytest.approx(
        {(0, 'n1'): 10.0, (0, 'n2'): 2.0, (1, 'n1'): 20.0, (1, 'n2'): 3.0, (2, 'n1'): 5.0, (2, 'n2'): 7.0}
    ), 'each generator sits alone on its node, so the grouped value is its own dispatch'


@pytest.mark.parametrize(
    ('name', 'dims'),
    [
        pytest.param('cost_by_gen', {'snapshot', 'generator'}, id='no-reduction'),
        pytest.param('served', {'snapshot'}, id='one-dim-summed-away'),
        pytest.param('total_cost', set(), id='everything-summed-away'),
    ],
)
def test_the_frame_carries_exactly_the_dims_the_expression_survives_over(name, dims):
    frame = evaluate(name)
    assert set(frame.columns) - {'value'} == dims, (
        'the returned frame answers over the dims the expression still ranges over after its sums'
    )


def test_an_expression_the_file_never_named_reads_the_same_arithmetic():
    got = dict(zip(*evaluate('sum(dispatch, over=generator)').sort('snapshot'), strict=True))
    assert got == pytest.approx({0: 12.0, 1: 23.0, 2: 12.0}), (
        'evaluate lowers an undeclared expression against the variable-free spec and reads it as arithmetic'
    )


def test_a_mapping_is_the_other_form_the_language_writes_an_expression_in():
    assert evaluate({'expression': 'sum(dispatch, over=generator)'}).equals(evaluate('served')), (
        'the mapping form of an entry and its bare string read one value'
    )


def test_a_name_the_spec_does_not_declare_is_refused():
    with pytest.raises(LanguageError, match='nope'):
        evaluate('nope')


@pytest.mark.parametrize(
    ('decision', 'names'),
    [
        pytest.param(
            {'variables': {'x': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'cost'}}}},
            'variables (x)',
            id='a-variable',
        ),
        pytest.param(
            {
                'variables': {'x': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'cost'}}},
                'objective': {'sense': 'minimize', 'expression': 'sum(x * cost)'},
            },
            'an objective',
            id='a-variable-and-objective',
        ),
    ],
)
def test_a_spec_that_declares_a_decision_is_refused_and_names_solve(decision, names):
    spec = {**SPEC, **decision}
    with pytest.raises(SpecsolveError, match=r'sps\.solve') as caught:
        sps.evaluate(spec, sources(), 'total_cost')
    assert names in str(caught.value), 'the refusal names the decision it found, so the author sees what to drop'


def test_only_the_expression_asked_for_is_compiled(monkeypatch):
    compiled = []
    original = PolarsCompiler.expression

    def counting(self, expr, context, **kwargs):
        compiled.append(context)
        return original(self, expr, context, **kwargs)

    monkeypatch.setattr(PolarsCompiler, 'expression', counting)
    evaluate('served')
    assert compiled == ["named expression 'served'"], (
        'four expressions are declared and one is asked for, so exactly that one compiles'
    )


def test_a_quotient_is_absent_where_its_divisor_is_zero():
    """With no solve the data is the only source of a zero, and the quotient has no row there; before #1775 it read `inf`."""
    spec = {**SPEC, 'parameters': {**SPEC['parameters'], 'weight': {'dims': ['generator']}}}
    data = sources() | {'weight': pl.DataFrame({'generator': GENERATORS, 'value': [2.0, 0.0]})}
    frame = sps.evaluate(spec, data, 'cost / weight')
    assert dict(zip(frame['generator'], frame['value'], strict=True)) == pytest.approx({'wind': 0.5}), (
        'gas divides by zero, so it has no row'
    )


@pytest.mark.parametrize(
    ('expression', 'expected'),
    [
        pytest.param('sum(cost) ** 2', 51.0**2, id='a-base'),
        pytest.param('2 ** sum(cost / 50)', 2.0**1.02, id='an-exponent'),
        pytest.param('102 / sum(cost)', 2.0, id='a-divisor'),
    ],
)
def test_an_operand_that_is_a_sum_is_taken_whole(expression, expected):
    """`sum(cost)` is 51, and a power or a quotient reads that total rather than each summand.

    Before #1777 `sum(cost) ** 2` read 1² + 50².
    """
    assert evaluate(expression)['value'].to_list() == pytest.approx([expected]), 'the sum is taken whole'
