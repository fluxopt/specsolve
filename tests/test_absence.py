"""Absence: what a missing value does to a coefficient, a row and a bound.

Absence is a state, not a zero (docs/reference/language/absence.md): a sparse
coefficient is still a coefficient, a constant side with a hole is refused, an
empty group is a zero, and a term whose variable is absent takes its row with
it. Each pair that differs is checked side by side.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DataError
from tests.conftest import by_coord, override
from tests.differential import RTOL, both_lanes_refuse, differential
from tests.oracle import pd

#: `p` over (node, tech) times `produces` adds `carrier`, and the sum removes `tech`.
BROADCAST_MASK_SPEC = {
    'dimensions': {
        'node': {'dtype': 'str'},
        'tech': {'dtype': 'str'},
        'carrier': {'dtype': 'str'},
    },
    'parameters': {
        'produces': {'dims': ['tech', 'carrier']},
        'demand': {'dims': ['node', 'carrier']},
        'cost': {'dims': ['tech']},
        'installed': {'dims': ['node', 'tech']},
    },
    'variables': {
        'p': {'dims': ['node', 'tech'], 'where': 'installed > 0', 'bounds': {'lower': 0, 'upper': 'installed'}},
    },
    'constraints': {
        'balance': {'dims': ['node', 'carrier'], 'expression': 'sum(p * produces, over=tech) == demand'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def _grid(dims, labels, values):
    """The full product of *labels* as a tidy frame, one row per coordinate."""
    frame = pd.MultiIndex.from_product(labels, names=dims).to_frame(index=False)
    return frame.assign(value=values)


SPARSE_COEFFICIENT_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'c': {'dims': ['t']}, 'w': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'cap': {'dims': ['t'], 'expression': 'w * x <= c'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x, over=t)'},
}


#: `w` everywhere, `c` with no row at t=0.
SPARSE_CONSTANT_DATA = {'t': [0, 1, 2], 'w': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}), 'c': pd.Series({1: 4.0, 2: 5.0})}


def test_a_sparse_coefficient_is_still_a_zero_coefficient():
    """A coordinate a coefficient does not cover contributes no term, and the row survives."""
    data = {'t': [0, 1, 2], 'w': pd.Series({1: 1.0, 2: 1.0}), 'c': pd.Series({0: 0.0, 1: 4.0, 2: 5.0})}
    with differential(SPARSE_COEFFICIENT_SPEC, data, lp=True) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 carries `<= 0` with no term: a row that exists and constrains nothing'
        )


def test_a_sparse_constant_side_is_refused_on_both_lanes():
    """The same omission on the constant side is a `DataError`, not a zero.

    There the fill is the bound: `w * x <= c` with no `c` row would read `<= 0`.
    """
    with (
        pytest.raises(DataError, match="parameter 'c' covers 1 fewer"),
        differential(SPARSE_COEFFICIENT_SPEC, SPARSE_CONSTANT_DATA),
    ):
        pass


@pytest.mark.parametrize(
    'constraint',
    [
        pytest.param({'dims': ['t'], 'expression': 'w * x <= c'}, id='at the row key'),
        pytest.param({'dims': [], 'expression': 'sum(w * x, over=t) <= sum(c, over=t)'}, id='under a sum'),
        pytest.param({'dims': ['t'], 'expression': 'w * x <= c + sum(c, over=t)'}, id='beside a sum of itself'),
    ],
)
def test_the_same_hole_is_refused_however_far_it_stands_from_the_row(constraint):
    """A reduction between the parameter and the row does not hide the hole (#1465)."""
    spec = override(SPARSE_COEFFICIENT_SPEC, **{'constraints.cap': constraint})
    both_lanes_refuse(spec, SPARSE_CONSTANT_DATA, match="parameter 'c' covers 1 fewer")


def test_a_where_is_the_escape_from_the_constant_side_check():
    """Masking the coordinate answers the question, so it is not refused.

    The check is keyed to the rows a declaration builds, not to the coordinate
    product.
    """
    masked = override(SPARSE_COEFFICIENT_SPEC, **{'constraints.cap.where': 'c'})
    with differential(masked, SPARSE_CONSTANT_DATA) as run:
        assert run.result.objective == pytest.approx(10.0 + 4.0 + 5.0, rel=RTOL), (
            't=0 has no row at all, so x runs to its bound there'
        )


def test_a_constant_piece_beside_a_term_is_refused_on_both_lanes():
    """A parameter added beside a variable term is a constant piece, and is checked (#1521).

    Filled, `w * x + c <= 100` would read `w * x <= 100` where the file says
    `w * x <= 100 - c`.
    """
    spec = override(SPARSE_COEFFICIENT_SPEC, **{'constraints.cap.expression': 'w * x + c <= 100'})
    both_lanes_refuse(spec, SPARSE_CONSTANT_DATA, match="parameter 'c' covers 1 fewer")


def test_a_constant_piece_beside_a_term_is_refused_through_a_reduction():
    """The same piece under a sum, which would sum the gap away (#1521)."""
    spec = override(
        SPARSE_COEFFICIENT_SPEC,
        **{'constraints.cap': {'dims': [], 'expression': 'sum(w * x, over=t) + sum(c, over=t) <= 100'}},
    )
    both_lanes_refuse(spec, SPARSE_CONSTANT_DATA, match="parameter 'c' covers 1 fewer")


def test_a_sparse_coefficient_beside_a_constant_piece_is_still_a_zero():
    """Beside a constant piece, a sparse coefficient is still a zero (#1521).

    `w` stands with a variable, so its missing row is a zero; only `c` is owed
    its coordinates.
    """
    data = {'t': [0, 1, 2], 'w': pd.Series({1: 1.0, 2: 1.0}), 'c': pd.Series({0: 5.0, 1: 4.0, 2: 5.0})}
    spec = override(SPARSE_COEFFICIENT_SPEC, **{'constraints.cap.expression': 'w * x + c <= 100'})
    with differential(spec, data, lp=True) as run:
        assert run.result.objective == pytest.approx(10.0 + 10.0 + 10.0, rel=RTOL), (
            't=0: w absent, so its term is zero and x runs to its bound; elsewhere the bound is slack'
        )


#: `south` is a load-only bus: both generators sit on `north`, so the group
#: behind `south`'s constant side has no members at all.
GROUPED_CONSTANT_SPEC = {
    'dimensions': {'generator': {}, 'bus': {'dtype': 'str'}},
    'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
    'parameters': {'capacity': {'dims': ['generator']}},
    'variables': {'imports': {'dims': ['bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'import_limit': {
            'dims': ['bus'],
            'expression': 'imports <= sum(capacity, by=gen_bus, over=generator, into=bus)',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(imports, over=bus)'},
}


def _grouped_constant_sources(capacity=('g1', 'g2')):
    return {
        'bus': ['north', 'south'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'north']}),
        'capacity': pl.DataFrame({'generator': list(capacity), 'value': [3.0, 4.0][: len(capacity)]}),
    }


def test_an_empty_group_on_the_constant_side_is_a_zero_and_not_a_gap():
    """A group with no members holds the empty sum, which is a value.

    `capacity` covers every generator it has, and a group with no members
    contributes nothing.
    """
    with differential(GROUPED_CONSTANT_SPEC, _grouped_constant_sources(), lp=True) as run:
        assert run.result.objective == pytest.approx(7.0, rel=RTOL), 'north imports up to 3 + 4, south up to nothing'
        built = by_coord(run.result, 'imports', 'bus')

    assert built['south'] == pytest.approx(0.0), "an empty group caps south's imports at the empty sum"


#: The same with a dim the group does not consume, so the empty label pairs with every snapshot.
SPANNED_GROUPED_CONSTANT_SPEC = {
    **GROUPED_CONSTANT_SPEC,
    'dimensions': {**GROUPED_CONSTANT_SPEC['dimensions'], 'snapshot': {'dtype': 'int'}},
    'parameters': {'capacity': {'dims': ['snapshot', 'generator']}},
    'variables': {'imports': {'dims': ['snapshot', 'bus'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'import_limit': {
            'dims': ['snapshot', 'bus'],
            'expression': 'imports <= sum(capacity, by=gen_bus, over=generator, into=bus)',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(imports, over=bus), over=snapshot)'},
}


def test_an_empty_group_spanning_another_dim_is_zero_at_every_coordinate():
    """The empty label is a row per snapshot, not one row."""
    sources = {
        'snapshot': [0, 1],
        'bus': ['north', 'south'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['north', 'north']}),
        'capacity': pl.DataFrame(
            {'snapshot': [0, 0, 1, 1], 'generator': ['g1', 'g2', 'g1', 'g2'], 'value': [3.0, 4.0, 1.0, 1.0]}
        ),
    }
    with differential(SPANNED_GROUPED_CONSTANT_SPEC, sources, lp=True) as run:
        assert run.result.objective == pytest.approx(7.0 + 2.0, rel=RTOL), 'north takes both snapshots, south neither'
        built = by_coord(run.result, 'imports', 'snapshot', 'bus')

    assert built[(0, 'south')] == pytest.approx(0.0), 'the empty group is zero at every snapshot'
    assert built[(1, 'south')] == pytest.approx(0.0), 'the empty group is zero at every snapshot'


#: A relation to two columns: `south` reaches neither technology, `north` both.
PLURAL_GROUPED_CONSTANT_SPEC = {
    **GROUPED_CONSTANT_SPEC,
    'dimensions': {**GROUPED_CONSTANT_SPEC['dimensions'], 'technology': {'dtype': 'str'}},
    'relations': {'gen_placement': {'key': 'generator', 'values': ['bus', 'technology']}},
    'variables': {'imports': {'dims': ['bus', 'technology'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'import_limit': {
            'dims': ['bus', 'technology'],
            'expression': 'imports <= sum(capacity, by=gen_placement, over=generator, into=[bus, technology])',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(imports, over=bus), over=technology)'},
}


def test_an_empty_combination_of_two_groups_is_a_zero_and_not_a_gap():
    """A combination of two groups that no member sits at holds the empty sum."""
    sources = {
        'bus': ['north', 'south'],
        'technology': ['wind', 'solar'],
        'generator': pl.DataFrame({'generator': ['g1', 'g2']}),
        'gen_placement': pl.DataFrame(
            {'generator': ['g1', 'g2'], 'bus': ['north', 'north'], 'technology': ['wind', 'solar']}
        ),
        'capacity': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [3.0, 4.0]}),
    }
    with differential(PLURAL_GROUPED_CONSTANT_SPEC, sources, lp=True) as run:
        assert run.result.objective == pytest.approx(3.0 + 4.0, rel=RTOL), 'south holds nothing at either technology'
        built = by_coord(run.result, 'imports', 'bus', 'technology')

    assert built[('south', 'wind')] == pytest.approx(0.0), 'no generator sits at (south, wind)'
    assert built[('south', 'solar')] == pytest.approx(0.0), 'no generator sits at (south, solar)'
    assert built[('north', 'solar')] == pytest.approx(4.0), 'one generator sits at (north, solar), and it is g2'


def test_a_member_with_no_value_is_still_refused_through_a_group():
    """A group short of a member's value keeps the refusal.

    Dropping `g2`'s row leaves `north`'s group with one member and a hole, not
    with no members.
    """
    with (
        pytest.raises(DataError, match="parameter 'capacity' covers 1 fewer"),
        differential(GROUPED_CONSTANT_SPEC, _grouped_constant_sources(capacity=('g1',))),
    ):
        pass


ABSENT_VARIABLE_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'dims': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'dims': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_term_whose_variable_is_absent_drops_the_row_on_both_lanes():
    """Absence propagates into the comparison; it does not zero the term.

    With ``size`` masked out, ``x - relmax * size <= 0`` drops its row rather
    than building ``x <= 0`` (linopy v1 §6, §12), so ``x`` at ``f=b`` is bounded
    only by its own declaration.
    """
    data = {
        'f': ['a', 'b'],
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_VARIABLE_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(100.0, rel=RTOL), 'unsized: the row is gone, so only the bound holds'


#: One rule per block, so the two regimes are two named constraints.
DEFINED_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'gate': {'dims': ['f'], 'dtype': 'bool'}, 'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'dims': ['f'], 'where': 'gate', 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {
        'envelope_sized': {'dims': ['f'], 'where': 'size', 'expression': 'x - relmax * size <= 0'},
        'envelope_unsized': {'dims': ['f'], 'where': 'NOT size', 'expression': 'x <= 0'},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_bare_variable_name_in_a_where_asks_whether_it_exists():
    """A bare variable name in a ``where`` asks "does this exist here".

    The two complementary clauses keep a row where the variable is absent.
    """
    data = {
        'f': ['a', 'b'],
        'gate': pd.Series({'a': True}),
        'relmax': pd.Series({'a': 0.5, 'b': 0.5}),
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(DEFINED_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: the envelope binds'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'unsized: the complementary clause pins it'


ABSENT_COEFFICIENT_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'relmax': {'dims': ['f']}, 'cost': {'dims': ['f']}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'size': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 50}},
    },
    'constraints': {'envelope': {'dims': ['f'], 'expression': 'x - relmax * size <= 0'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost, over=f)'},
}


def test_a_sparse_coefficient_on_the_bound_side_still_pins_the_variable():
    """A missing coefficient keeps the row, and ``x <= 0`` is built.

    Same expression as ``ABSENT_VARIABLE_SPEC``, but what is missing at ``f=b``
    is the parameter ``relmax``. Absence is a property of variables, so nothing
    propagates (docs/reference/language/absence.md).
    """
    data = {
        'f': ['a', 'b'],
        'relmax': pd.Series({'a': 0.5}),  # no row at 'b'
        'cost': pd.Series({'a': 1.0, 'b': 1.0}),
    }
    with differential(ABSENT_COEFFICIENT_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 'f')
        assert x['a'] == pytest.approx(25.0, rel=RTOL), 'sized: x <= 0.5 * size, size <= 50'
        assert x['b'] == pytest.approx(0.0, abs=1e-9), 'the row survived the missing coefficient and pins x'


SCALAR_MASKED_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'dims': [], 'where': 'budget > 1000', 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'cap': {'dims': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


def test_a_masked_out_scalar_variable_drops_the_row_that_uses_it():
    """Law 7 holds at no dimension either (#340), and on the bare install."""
    data = {'f': ['a', 'b'], 'cost': pl.DataFrame({'f': ['a', 'b'], 'value': [1.0, 2.0]}), 'budget': 120.0}

    with sps.solve(SCALAR_MASKED_SPEC, data) as sol:
        assert sol.dual('cap').height == 0, 'the row is gone, not slackened — a dropped row has no dual'
        assert sol.objective == pytest.approx(300.0), 'unbudgeted, both generators run flat out'


def test_a_mask_survives_a_broadcast_into_a_reduction():
    """A mask keeps its own dims through a product that widens them (#345)."""
    data = {
        'node': ['n1', 'n2'],
        'tech': ['t1', 't2'],
        'carrier': ['elec', 'heat'],
        # a tech produces exactly one carrier, which is what makes `produces` sparse
        'produces': _grid(['tech', 'carrier'], [['t1', 't2'], ['elec', 'heat']], [1.0, 0.0, 0.0, 1.0]),
        'demand': _grid(['node', 'carrier'], [['n1', 'n2'], ['elec', 'heat']], [10.0, 20.0, 10.0, 20.0]),
        'cost': pd.Series({'t1': 1.0, 't2': 2.0}),
        'installed': _grid(['node', 'tech'], [['n1', 'n2'], ['t1', 't2']], [100.0] * 4),
    }

    with differential(BROADCAST_MASK_SPEC, data) as run:
        assert run.result.objective == pytest.approx(100.0, rel=RTOL)
