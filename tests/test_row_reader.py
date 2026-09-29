"""``model.row``: what a named row at a named coordinate says.

It reads the **built** row, not the declared one: a coefficient that data
scaled, a term whose variable was absent, a row a ``where`` removed. Every test
here is a case where the file and the built row differ.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from tests.conftest import DISPATCH_SPEC, override

if TYPE_CHECKING:
    from specsolve.relational.result import ConstraintRow

DATA = {
    'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    'p_max': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 50.0]}),
    'snapshot': pl.DataFrame({'snapshot': [0, 1, 2, 3]}),
    'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [80.0, 60.0, 100.0, 45.0]}),
}

#: Two variables in one row, and a coefficient that only data knows — the two
#: things a rendered coordinate and a built read exist for.
COMMITMENT: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'parameters': {'p_max': {'dims': ['g']}, 'load': {'dims': ['t']}},
    'variables': {
        'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'p_max'}},
        'u': {'dims': ['t', 'g'], 'domain': 'binary'},
    },
    'constraints': {
        'commit': {'dims': ['t', 'g'], 'expression': 'p <= p_max * u'},
        'balance': {'dims': ['t'], 'expression': 'sum(p, over=g) == load'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

COMMITMENT_DATA = {
    't': [0, 1],
    'g': ['wind', 'gas'],
    'p_max': pl.DataFrame({'g': ['wind', 'gas'], 'value': [40.0, 200.0]}),
    'load': pl.DataFrame({'t': [0, 1], 'value': [80.0, 60.0]}),
}


def _terms(row: ConstraintRow) -> list[tuple[str, str, float]]:
    """The row's terms as tuples, for a readable assertion."""
    return list(row.terms.iter_rows())


def test_a_row_is_its_terms_its_comparison_and_its_right_hand_side() -> None:
    """The whole shape, on a row whose right-hand side only the data knows."""
    with sps.build(DISPATCH_SPEC, DATA) as model:
        row = model.row('balance', snapshot=2)

    assert _terms(row) == [('p', '2, wind', 1.0), ('p', '2, gas', 1.0)]
    assert (row.sense, row.rhs) == ('==', 100.0), 'the right-hand side is the bound value, not the parameter name'


def test_printing_a_row_gives_the_line_linopy_gives() -> None:
    """A row prints as linopy's ``Constraint.print()`` does, with its identity on the same line."""
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        printed = str(model.row('commit', t=1, g='gas'))

    assert printed == 'commit[t=1, g=gas]: +1 p[1, gas] -200 u[1, gas] <= 0'


def test_a_row_too_wide_to_spell_out_summarises_instead_of_truncating() -> None:
    """Twelve terms of three hundred are twelve arbitrary ones.

    A wide row prints each declaration's term count and coefficient span. The
    model here has a thousand-fold spread *inside one row*.
    """
    generators = [f'g{i}' for i in range(300)]
    spec = {
        'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
        'parameters': {'cost': {'dims': ['g']}, 'load': {'dims': ['t']}},
        'variables': {
            'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 100}},
            'slack': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 9}},
        },
        'constraints': {'balance': {'dims': ['t'], 'expression': 'sum(p * cost, over=g) + slack * 1000 >= load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    data = {
        't': [0],
        'g': generators,
        'cost': pl.DataFrame({'g': generators, 'value': [0.001 * (i + 1) for i in range(300)]}),
        'load': pl.DataFrame({'t': [0], 'value': [5.0]}),
    }
    with sps.build(spec, data) as model:
        row = model.row('balance', t=0)

    assert row.terms.height == 301, 'the frame keeps every term whatever the line does'
    assert str(row) == 'balance[t=0]: 301 terms — p: 300 (|coef| 0.001…0.3), slack: 1 (|coef| 1000) >= 5'


def test_a_declaration_whose_coefficients_are_all_one_says_so_once() -> None:
    """A single magnitude prints as itself, not as a range against itself."""
    generators = [f'g{i}' for i in range(30)]
    data = {
        'generator': generators,
        'p_max': pl.DataFrame({'generator': generators, 'value': [10.0] * 30}),
        'cost': pl.DataFrame({'generator': generators, 'value': [1.0] * 30}),
        'snapshot': pl.DataFrame({'snapshot': [0]}),
        'load': pl.DataFrame({'snapshot': [0], 'value': [5.0]}),
    }
    with sps.build(DISPATCH_SPEC, data) as model:
        assert str(model.row('balance', snapshot=0)) == 'balance[snapshot=0]: 30 terms — p: 30 (|coef| 1) == 5'


def test_it_answers_on_a_model_that_was_never_solved() -> None:
    """A model too wrong to solve is the one whose rows need reading, so no solve is required."""
    with sps.build(DISPATCH_SPEC, DATA) as model:
        assert model.diagnostics().solves == 0, 'nothing has been solved, and the row still reads'
        assert model.row('balance', snapshot=0).rhs == 80.0


def test_a_row_spanning_two_declarations_names_both() -> None:
    """``p <= p_max * u`` puts a term from each of two variables in one row.

    The coefficient on ``u`` is the one the *data* supplied.
    """
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        row = model.row('commit', t=1, g='gas')

    assert _terms(row) == [('p', '1, gas', 1.0), ('u', '1, gas', -200.0)], (
        'p - p_max*u <= 0, with p_max the bound value for gas'
    )
    assert (row.sense, row.rhs) == ('<=', 0.0)


def test_the_coefficient_is_the_one_data_produced_not_the_one_declared() -> None:
    """The claim `typeset` cannot make: the file says ``p_max``, the row says 40."""
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        terms = model.row('commit', t=0, g='wind').terms
    wind = dict(zip(terms['variable'].to_list(), terms['coefficient'].to_list(), strict=True))
    assert wind['u'] == -40.0, "wind's bound, where the same declaration gives gas -200"


def test_a_term_whose_variable_is_absent_is_absent_from_the_row() -> None:
    """A masked variable leaves a *shorter* row: the file sums over both generators, the row has one term."""
    spec = override(COMMITMENT, **{'variables.p.where': 'p_max > 100'})
    with sps.build(spec, COMMITMENT_DATA) as model:
        row = model.row('balance', t=0)

    assert _terms(row) == [('p', '0, gas', 1.0)], 'wind is masked out of p, so the balance row lost its term'


def test_a_row_a_where_removed_says_so_rather_than_answering() -> None:
    """The coordinate is legal and the row does not exist — which is the answer."""
    spec = override(COMMITMENT, **{'constraints.commit.where': 'p_max > 100'})
    with sps.build(spec, COMMITMENT_DATA) as model, pytest.raises(SpecsolveError, match='built no row'):
        model.row('commit', t=0, g='wind')


def test_a_partial_coordinate_is_refused_rather_than_answered_about_one_row() -> None:
    """A partial coordinate names a block, not a row."""
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(SpecsolveError, match='declared over'):
        model.row('commit', t=0)


def test_an_unknown_constraint_lists_the_declared_ones() -> None:
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(KeyError, match='balance'):
        model.row('nope', t=0, g='wind')


def test_a_closed_model_says_it_was_closed() -> None:
    """And says which row was being asked for, which is what the reader came with."""
    model = sps.build(DISPATCH_SPEC, DATA)
    model.close()
    with pytest.raises(SpecsolveError, match="no built model to read 'balance' out of"):
        model.row('balance', snapshot=0)


def test_a_update_moves_what_the_row_says() -> None:
    """The row is read off the current build, not the one that was first bound."""
    with sps.build(DISPATCH_SPEC, DATA) as model:
        assert model.row('balance', snapshot=0).rhs == 80.0
        moved = {**DATA, 'load': pl.DataFrame({'snapshot': [0, 1, 2, 3], 'value': [7.0, 7.0, 7.0, 7.0]})}
        assert model.update(moved).row('balance', snapshot=0).rhs == 7.0


def test_the_row_read_is_the_row_the_solver_was_given() -> None:
    """Every term of every row, against the matrix handed to the sink.

    A reader that resolved the row index or the column ranges wrongly would
    still return plausible terms.
    """
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        tables = model._engine._model.handoff
        for name in ('commit', 'balance'):
            block = model._engine._model.constraints[name]
            coordinates = block.frame.collect()
            for offset in range(block.height):
                at = block.start + offset
                given = tables.matrix_block(at, at + 1)
                where = {d: coordinates.item(offset, d) for d in coordinates.columns if d != 'row'}
                read = model.row(name, **where)
                assert read.terms['coefficient'].to_list() == pytest.approx(given['coeff'].to_list()), (
                    f'{name} at {where} does not carry the coefficients the sink was handed for row {at}'
                )
                assert read.terms.height == given.height


#: A coefficient and a right-hand side that ``%g``'s six significant digits
#: cannot tell apart from their neighbours, and a scalar declaration — the
#: cases where the *rendering* is what makes a row readable or not.
PRECISE: dict[str, Any] = {
    'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['g']}, 'load': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0}}},
    'constraints': {'balance': {'dims': ['t'], 'expression': 'sum(p * cost, over=g) >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

PRECISE_DATA = {
    't': [0],
    'g': ['a', 'b'],
    'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [1.0000001, 12345678.0]}),
    'load': pl.DataFrame({'t': [0], 'value': [12345678.9]}),
}


def test_a_coefficient_prints_every_digit_the_data_gave_it() -> None:
    """A rendering that rounds agrees with the file in exactly the case worth reading.

    ``%g`` stops at six significant digits and prints ``1.0000001`` as ``1``.
    """
    with sps.build(PRECISE, PRECISE_DATA) as model:
        printed = str(model.row('balance', t=0))

    assert printed == 'balance[t=0]: +1.0000001 p[0, a] +12345678 p[0, b] >= 12345678.9', (
        'every digit the data carried survives, and a whole coefficient still reads as linopy prints it'
    )


def test_a_row_echoed_at_a_prompt_is_the_line_not_the_frame() -> None:
    """``repr`` is how a row is read in a REPL and in a notebook cell, so it is the line too."""
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        row = model.row('commit', t=1, g='gas')

    assert repr(row) == str(row) == 'commit[t=1, g=gas]: +1 p[1, gas] -200 u[1, gas] <= 0'


def test_a_row_has_one_spelling_whatever_order_its_coordinate_was_given_in() -> None:
    """One row, one identity: the declaration orders the coordinate, not the caller's keywords."""
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model:
        by_declaration = model.row('commit', t=1, g='gas')
        reversed_kwargs = model.row('commit', g='gas', t=1)

    assert str(by_declaration) == str(reversed_kwargs)
    assert list(reversed_kwargs.coordinate) == ['t', 'g'], "the declaration's dim order, not the call's"


def test_a_declaration_over_no_dims_carries_no_bracket() -> None:
    """``z``, not ``z[]`` — linopy's spelling, and an empty bracket states a
    coordinate that does not exist."""
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'variables': {
            'p': {'dims': ['g'], 'bounds': {'lower': 0}},
            'z': {'dims': [], 'bounds': {'lower': 0}},
        },
        'constraints': {'total': {'dims': [], 'expression': 'sum(p, over=g) + z <= 10'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p) + z'},
    }
    with sps.build(spec, {'g': ['wind', 'gas']}) as model:
        assert str(model.row('total')) == 'total: +1 p[wind] +1 p[gas] +1 z <= 10'


def test_a_dimension_called_name_is_still_a_coordinate() -> None:
    """``name`` is a legal dimension, and the parameter naming the constraint
    may not take it away — so the constraint is positional."""
    spec = {
        'dimensions': {'name': {'dtype': 'str'}},
        'parameters': {'p_max': {'dims': ['name']}},
        'variables': {'p': {'dims': ['name'], 'bounds': {'lower': 0}}},
        'constraints': {'cap': {'dims': ['name'], 'expression': 'p <= p_max'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }
    data = {'name': ['wind', 'gas'], 'p_max': pl.DataFrame({'name': ['wind', 'gas'], 'value': [40.0, 200.0]})}
    with sps.build(spec, data) as model:
        assert str(model.row('cap', name='wind')) == 'cap[name=wind]: +1 p[wind] <= 40'


def test_a_coefficient_the_data_made_zero_leaves_no_term() -> None:
    """The third way a built row is shorter than its file, beside a masked
    variable and a masked row: a zero coefficient is pruned, not printed as ``+0``.
    """
    zeroed = {**PRECISE_DATA, 'cost': pl.DataFrame({'g': ['a', 'b'], 'value': [0.0, 2.0]})}
    with sps.build(PRECISE, zeroed) as model:
        row = model.row('balance', t=0)

    assert _terms(row) == [('p', '0, b', 2.0)], 'a zero coefficient is not a term, so `a` is not in the row'


@pytest.mark.parametrize(
    ('coordinate', 'names'),
    [
        pytest.param({'t': '0', 'g': 'wind'}, 'Int64', id='a string against an integer dim'),
        pytest.param({'t': 0, 'g': 1}, 'Enum', id='an integer against a label dim'),
        pytest.param({'t': 0, 'g': 'nope'}, 'Enum', id='a stranger against an Enum'),
    ],
)
def test_a_label_the_dimension_cannot_hold_is_refused_in_our_own_tree(coordinate: dict[str, Any], names: str) -> None:
    """Labels arrive from JSON and CSV as the wrong type, and an ``Enum`` refuses strangers.

    All three are one failure, not a label the dimension has, and none reaches
    the caller in polars' vocabulary.
    """
    with sps.build(COMMITMENT, COMMITMENT_DATA) as model, pytest.raises(SpecsolveError, match='not one of its labels'):
        try:
            model.row('commit', **coordinate)
        except SpecsolveError as refused:
            assert names in str(refused), 'the message names the type the dimension does hold'
            raise
