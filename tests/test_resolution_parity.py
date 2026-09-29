"""The scoping divergences, checked against the oracle lane itself.

The rules are mathspec's and are swept there. This module checks that the
linopy lane refuses what the relational lane refuses, in the same place, for the
same reason.
"""

from __future__ import annotations

import datetime

import polars as pl
import pytest
import yaml as pyyaml

import specsolve as sps
from tests.conftest import DISPATCH_SPEC, dispatch_spec_path, override
from tests.differential import differential
from tests.oracle import pd, specsolve_linopy  # skips the module without the oracle


@pytest.mark.parametrize(
    ('where', 'match'),
    [
        pytest.param('typo_name > 0', "'typo_name' not found", id='a-name-nothing-declares'),
        pytest.param('generator == snapshot', 'compares against dimension', id='a-dimension-on-the-right'),
        pytest.param('nonexistent', "'nonexistent' not found", id='a-bare-name-nothing-declares'),
        pytest.param('snapshot', 'bare dimension name is true at every coordinate', id='a-bare-dimension-name'),
    ],
)
def test_both_lanes_refuse_the_same_where(tmp_path, dispatch_spec_inputs, where, match):
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **{'variables.p.where': where})

    with pytest.raises(ValueError, match=match):
        specsolve_linopy.build(path, data)

    with pytest.raises(ValueError, match=match):
        sps.check(path)


def test_both_lanes_refuse_a_comparison_that_carries_no_variable(tmp_path, dispatch_spec_inputs):
    """A constraint whose two sides are both constants decides nothing (#1171).

    It is decidable with no data attached, so both lanes refuse it at load.
    """
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **{'constraints.balance.expression': 'p_max <= 1'})

    with pytest.raises(ValueError, match='decides nothing'):
        specsolve_linopy.build(path, data)

    with pytest.raises(ValueError, match='decides nothing'):
        sps.check(path)


#: Where-strings that must build *identically* on both lanes, covering every
#: resolved predicate type (see the exhaustiveness test below). The dim
#: comparisons are always-true: a mask that removes every variable from a
#: constraint row is pinned separately below.
ACCEPTED = [
    'True',
    'p_max',
    'p_max > 0',
    'snapshot >= 0',
    'position(snapshot) >= 0',
    'NOT p_max > 150',
    'p_max > 0 AND snapshot >= 0',
    'p_max > 0 OR snapshot >= 0',
    #: Folded to `p_max > 0` at load; the claim is that a file may say it.
    'p_max > 0 AND True',
    #: The one position a literal survives to: alone, and false.
    'False',
    #: Two parameters compared, with arithmetic on a side, a count reducing a
    #: dimension away, and a predicate read at the neighbouring coordinate.
    'p_max > cost',
    'p_max >= 0.5 * p_max',
    'count(load, over=snapshot) >= 1',
    'p_max OR shift(p_max, along=generator, offset=1)',
]

#: Predicates this sweep cannot host, with the test that checks each instead. A
#: bare variable name is a self-reference in ``p``'s own where and spans a dim
#: ``balance`` does not (#469); dispatch declares no relation.
COVERED_ELSEWHERE = {
    'VariableDefined': ('tests/test_relational.py::test_a_bare_variable_name_in_a_where_asks_whether_it_exists'),
    'RelationComparison': 'tests/test_label_coords.py::test_a_where_reads_a_relation',
    'RelationPairComparison': 'tests/test_label_coords.py::test_a_relation_where_agrees_with_the_oracle',
    'RelationDefined': 'tests/test_label_coords.py::test_a_where_reads_a_relation',
    'PulledBackPredicate': 'tests/test_label_coords.py::test_a_relation_where_agrees_with_the_oracle',
}


@pytest.mark.parametrize('where', ACCEPTED)
def test_both_lanes_build_the_same_model(tmp_path, dispatch_spec_inputs, where):
    """Both lanes agree on *which* model they built, feasible or not.

    A mask that excludes snapshot 0 leaves the balance row unsatisfiable; that
    is not the claim here, and neither lane is asked to make every mask
    feasible.
    """
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **{'variables.p.where': where})

    m = specsolve_linopy.build(path, data)
    linopy_rows = int((m.variables['p'].labels != -1).sum())
    linopy_status = m.solve(solver_name='highs')[1]

    with sps.build(path, data) as model:
        relational_rows = model._engine._model.variables['p'].frame.select(pl.len()).collect().item()
        relational_status = model.solve().termination_condition

    assert linopy_rows == relational_rows, f'{where}: {linopy_rows} vs {relational_rows} variables'
    assert linopy_status == relational_status, f'{where}: {linopy_status} vs {relational_status}'


def test_every_resolved_predicate_is_parity_tested():
    """Every resolved predicate is exercised by ACCEPTED or COVERED_ELSEWHERE, so a new node cannot arrive untested."""
    import dataclasses
    from typing import get_args

    from mathspec import program, to_spec

    expected = set(get_args(program.Predicate))
    covered: set[type] = set()

    def walk(node):
        if node is None:
            return
        covered.add(type(node))
        for field in dataclasses.fields(node):
            child = getattr(node, field.name)
            if dataclasses.is_dataclass(child):
                walk(child)

    for where in ACCEPTED:
        mask = to_spec(override(DISPATCH_SPEC, **{'variables.p.where': where})).program.variables['p'].where
        walk(None if mask is None else mask.root)
    covered |= {t for t in expected if t.__name__ in COVERED_ELSEWHERE}

    missing = expected - covered
    assert not missing, (
        f'resolved predicates with no both-lanes test: {sorted(t.__name__ for t in missing)}. '
        f'Add it to ACCEPTED, or to COVERED_ELSEWHERE naming the test that does cover it.'
    )


def test_a_constraint_row_left_with_no_variables(tmp_path, dispatch_spec_inputs):
    """A masked *variable* can orphan an unmasked *constraint* row, and neither lane builds it.

    `where: "snapshot > 0"` on `p` leaves `balance` at snapshot 0 with no
    terms; kept, the row reads `0 == 80` and the model is infeasible. The
    omission is asserted too: dropping a declared row is only defensible
    because the build says it happened.
    """
    data = dispatch_spec_inputs
    path = dispatch_spec_path(tmp_path, **{'variables.p.where': 'snapshot > 0'})

    m = specsolve_linopy.build(path, data)
    linopy_status = m.solve(solver_name='highs')[1]

    with sps.build(path, data) as model:
        relational_status = model.solve().termination_condition
        assert model.diagnostics().omissions.to_dicts() == [{'constraint': 'balance', 'rows_not_built': 1}], (
            'a dropped row has to be reported, or a declared constraint goes quietly unenforced'
        )

    assert linopy_status == relational_status


#: A dimension the data leaves with **no members**, and a variable reduced over
#: it. The empty sum is a number, so the row it lands in asserts something about
#: constants alone.
EMPTY_AXIS_SPEC = {
    'dimensions': {'g': {'dtype': 'str'}, 'k': {'dtype': 'int'}},
    'parameters': {'exists': {'dims': ['g', 'k'], 'dtype': 'bool'}},
    'variables': {
        'w': {'dims': ['g', 'k'], 'where': 'exists', 'bounds': {'lower': 0, 'upper': 1}},
        'p': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'convex': {'dims': ['g'], 'expression': 'sum(w, over=k) == 1'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=g)'},
}


@pytest.mark.parametrize('sense', ['==', '<=', '>='], ids=['equality', 'upper-bound', 'lower-bound'])
def test_a_row_over_a_dimension_with_no_members_is_not_built_on_either_lane(sense: str):
    """The same rule as above, reached where the *dimension* is empty (#1108).

    A reduction over a set with no members is `0`, so `sum(w, over=k) == 1`
    is a row about constants alone and neither lane builds it, whatever the
    sense.
    """
    spec = override(EMPTY_AXIS_SPEC, **{'constraints.convex.expression': f'sum(w, over=k) {sense} 1'})
    data = {
        'g': pd.Index(['a', 'b'], name='g'),
        'k': pd.Index([], name='k', dtype='int64'),
        'exists': pd.DataFrame(
            {
                'g': pd.Series([], dtype='object'),
                'k': pd.Series([], dtype='int64'),
                'value': pd.Series([], dtype='bool'),
            }
        ),
    }

    with differential(spec, data) as run:
        assert 'convex' not in run.model.constraints, 'a row asserting something about constants only was built'
        assert run.engine.diagnostics().rows == 0, 'the relational lane built one anyway'
        assert run.engine.diagnostics().omissions.to_dicts() == [{'constraint': 'convex', 'rows_not_built': 2}], (
            'a declared row dropped for want of a term is only defensible because the build says it happened'
        )
        assert run.oracle == 20.0, 'both `p` reach their bound — an unbuilt row pins nothing'


#: The KVL shape: a block ranging over a dimension the data left with no
#: members, its expression summing over one that has some — PyPSA's
#: Kirchhoff voltage law on a network with no cycles.
EMPTY_FOREACH_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}, 'cycle': {'dtype': 'str'}, 'line': {'dtype': 'str'}},
    'parameters': {'w': {'dims': ['cycle', 'line']}, 'cost': {'dims': ['line']}},
    'variables': {'s': {'dims': ['t', 'line'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'kvl': {'dims': ['t', 'cycle'], 'expression': 'sum(w * s, over=line) == 0'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(cost * s)'},
}


def test_a_block_ranging_over_a_dimension_with_no_members_is_not_built_on_either_lane():
    """The rule of the test above, reached from the term the sum keeps.

    The sum reduces a full dimension while its term carries the empty one, so
    every row of the result is empty and neither lane builds the block.
    """
    data = {
        't': pd.Index([0, 1], name='t', dtype='int64'),
        'cycle': pd.Index([], name='cycle', dtype='object'),
        'line': pd.Index(['l1', 'l2'], name='line'),
        'w': pd.DataFrame(
            {
                'cycle': pd.Series([], dtype='object'),
                'line': pd.Series([], dtype='object'),
                'value': pd.Series([], dtype='float64'),
            }
        ),
        'cost': pd.DataFrame({'line': ['l1', 'l2'], 'value': [1.0, 2.0]}),
    }

    with differential(EMPTY_FOREACH_SPEC, data) as run:
        assert 'kvl' not in run.model.constraints, 'a block with no rows was built anyway'
        assert run.engine.diagnostics().rows == 0, 'the relational lane built a row over an empty dimension'
        assert run.oracle == 0.0, 'nothing pins `s` above its lower bound — an unbuilt block constrains nothing'


BOOL_MASK_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'active': {'dims': ['t'], 'dtype': 'bool'}, 'cap': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'floor': {'dims': ['t'], 'expression': 'x >= cap', 'where': 'active'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
}


def test_a_bool_parameter_is_a_mask_on_both_lanes():
    """A bool parameter reads as its own value: true masks in, false masks out,
    and an absent row masks out.
    """
    data = {
        't': [0, 1, 2],
        'active': pd.Series({0: True, 1: False}),
        'cap': pd.Series({0: 1.0, 1: 1.0, 2: 1.0}),
    }

    with differential(BOOL_MASK_SPEC, data) as run:
        assert run.oracle == 1.0, 'true masks the floor in at t=0 only, so exactly one x sits at its cap'


#: A budget row over no dims, because `sum` reduces the only one away — and a
#: scalar `slack` column and scalar `budget` value beside it, so one model
#: carries the empty coordinate in all three positions it can appear in.
SCALAR_ROW_SPEC = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['f']}, 'budget': {'dims': []}},
    'variables': {
        'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}},
        'slack': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'budget_row': {'dims': [], 'expression': 'sum(x, over=f) - slack <= budget'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * cost)'},
}


def test_the_empty_coordinate_builds_on_both_lanes():
    """A scalar row, a scalar column and a scalar value, in one model (#320).

    A product over no dimensions has one coordinate, not none.
    """
    data = {'f': ['a', 'b', 'c'], 'cost': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), 'budget': 120.0}

    with differential(SCALAR_ROW_SPEC, data) as run:
        assert run.oracle == 360.0
        assert run.result.dual('budget_row').height == 1, 'each claim is one — not zero, and not one per f'
        assert run.result.primal('slack').to_dicts() == [{'value': 10.0}]
        assert run.result.primal('x').sort('f')['value'].to_list() == [0.0, 30.0, 100.0]


@pytest.mark.parametrize(
    ('threshold', 'rows', 'objective'),
    [
        pytest.param(999.0, 0, 600.0, id='masked-out'),
        pytest.param(10.0, 1, 360.0, id='masked-in'),
    ],
)
def test_a_masked_scalar_variable_takes_its_row_with_it(threshold, rows, objective):
    """Absence spreads through arithmetic at no dimension either (#340).

    The mask is data, so the *same file* drops the row or keeps it depending
    only on what `budget` turns out to be.
    """
    spec = override(SCALAR_ROW_SPEC, **{'variables.slack.where': f'budget > {threshold}'})
    data = {'f': ['a', 'b', 'c'], 'cost': pd.Series({'a': 1.0, 'b': 2.0, 'c': 3.0}), 'budget': 120.0}

    with differential(spec, data) as run:
        assert run.oracle == objective
        assert run.result.dual('budget_row').height == rows, 'the row is gone, not slackened: a dropped row has no dual'


DATETIME_SPEC = {
    'dimensions': {'snapshot': {'dtype': 'datetime'}, 'generator': {'dtype': 'str'}},
    'parameters': {'cost': {'dims': ['generator']}, 'load': {'dims': ['snapshot']}},
    'variables': {'p': {'dims': ['snapshot', 'generator'], 'where': "snapshot > '2030-01-02'", 'bounds': {'lower': 0}}},
    'constraints': {
        'bal': {
            'dims': ['snapshot'],
            'where': "snapshot > '2030-01-02'",
            'expression': 'sum(p, over=generator) == load',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def test_a_datetime_boundary_is_sayable_on_both_lanes(tmp_path):
    """A quoted ISO date in a `where` names a boundary on a datetime dimension (#460)."""
    path = tmp_path / 'm.yaml'
    path.write_text(pyyaml.safe_dump(DATETIME_SPEC))
    days = [datetime.date(2030, 1, d) for d in (1, 2, 3)]
    frames = {
        'cost': pl.DataFrame({'generator': ['wind', 'gas'], 'value': [1.0, 5.0]}),
        'load': pl.DataFrame({'snapshot': days, 'value': [10.0, 20.0, 30.0]}),
        'snapshot': pl.DataFrame({'snapshot': days}),
        'generator': pl.DataFrame({'generator': ['wind', 'gas']}),
    }
    linopy_data = {
        'cost': pd.Series({'wind': 1.0, 'gas': 5.0}),
        'load': pd.Series([10.0, 20.0, 30.0], index=pd.Index(days, name='snapshot')),
    }
    linopy_data |= {
        'snapshot': pd.Index(days, name='snapshot'),
        'generator': pd.Index(['wind', 'gas'], name='generator'),
    }

    m = specsolve_linopy.build(path, linopy_data)
    m.solve(solver_name='highs')
    linopy_lane = float(m.objective.value)

    with sps.solve(path, frames) as result:
        relational = result.objective
        assert result.primal('p')['snapshot'].dtype in (pl.Date, pl.Datetime('us')), 'the coordinate keeps its dtype'
        assert result.primal('p').height == 2, 'only the third day survives the boundary'

    assert linopy_lane == relational == 30.0
