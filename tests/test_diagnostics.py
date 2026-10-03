"""What the build reports about itself: omissions, clocks, magnitudes, sparsity.

A row that lost every term is not built, and is reported; a range names the
block that holds an outlier; a parameter short of its dimensions is reported
rather than judged.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.relational.sinks import SOLVERS, solver
from tests.conftest import SOLVER_VECTOR_LOAD, SOLVER_VECTOR_SPEC
from tests.differential import RTOL, differential


@pytest.mark.parametrize('solver_name', sorted(SOLVERS))
@pytest.mark.parametrize('batch_rows', [1, 2, 7, 100_000], ids=['one', 'two', 'odd', 'whole'])
def test_a_row_with_no_terms_is_not_built_and_is_reported(solver_name, batch_rows):
    """A row that lost every term is not a constraint, and the build says so.

    `where: "t > 0"` leaves `balance` at `t = 0` with nothing to sum. Ragged
    batches, because labels are compacted when a row goes and a block loop
    has to agree about the narrower block; HiGHS ignores ``batch_rows``.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
        'parameters': {'load': {'dims': ['t']}},
        'variables': {'p': {'dims': ['t', 'g'], 'where': 't > 0', 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'balance': {'dims': ['t'], 'expression': 'sum(p, over=g) == load'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(sum(p, over=g), over=t)'},
    }
    data = {'t': [0, 1, 2], 'g': ['a', 'b'], 'load': pl.DataFrame({'t': [0, 1, 2], 'value': [5.0, 4.0, 6.0]})}
    with sps.build(spec, data) as model:
        tables = model._engine._model.handoff
        occupied = sorted(set(tables.matrix_block(0, tables.row_count)['row'].to_list()))
        assert occupied == [0, 1], 'the block closes up around the gap'
        assert model.diagnostics().omissions.to_dicts() == [{'constraint': 'balance', 'rows_not_built': 1}]
        with solver(solver_name)(tables, batch_rows, None) as sink:
            answer = sink.run(tables)
        assert answer.status.termination_condition == 'optimal'
        assert answer.objective == pytest.approx(4.0 + 6.0, rel=RTOL), 'the two built rows still bind'


def test_omissions_is_empty_when_every_declared_row_is_built():
    """The common case says nothing."""
    with sps.build(SOLVER_VECTOR_SPEC, SOLVER_VECTOR_LOAD) as model:
        assert model.diagnostics().omissions.is_empty()


def test_a_row_a_propagated_absence_deleted_is_reported_too():
    """A row deleted by absence travelling out of ``y`` is reported too (#944)."""
    spec = {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {'cap': {'dims': ['g']}, 'extra': {'dims': ['g']}},
        'variables': {
            'x': {'dims': ['g'], 'bounds': {'lower': 0, 'upper': 'cap'}},
            'y': {'dims': ['g'], 'where': 'extra', 'bounds': {'lower': 0, 'upper': 0}},
        },
        'constraints': {'both': {'dims': ['g'], 'expression': 'x + y >= 5'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=g)'},
    }
    data = {
        'g': ['a', 'b'],
        'cap': pl.DataFrame({'g': ['a', 'b'], 'value': [10.0, 10.0]}),
        'extra': pl.DataFrame({'g': ['a'], 'value': [1.0]}),
    }
    with sps.build(spec, data) as model:
        assert model.diagnostics().omissions.to_dicts() == [{'constraint': 'both', 'rows_not_built': 1}], (
            'the row absence travelled out of y and deleted is counted'
        )
        assert model.solve().objective == pytest.approx(5.0, rel=RTOL), (
            'and it is worth 5: only one of the two declared rows is enforced'
        )


def test_diagnostics_say_where_the_time_went(tmp_path):
    """A run that is slower than it should be can say which phase the time went to.

    `seconds` is advisory wall time, so nothing here asserts a magnitude.
    """
    with sps.build(SOLVER_VECTOR_SPEC, SOLVER_VECTOR_LOAD) as model:
        built = model.diagnostics().seconds
        assert set(built) == {'attach', 'build'}, (
            'a model only built has spent time attaching sources and building frames, nowhere else'
        )
        assert all(seconds >= 0 for seconds in built.values()), 'a wall clock cannot run backwards'

        model.solve()
        model.write(tmp_path / 'model.lp')
        ran = model.diagnostics().seconds
        assert set(ran) == {'attach', 'build', 'handoff', 'solve', 'write'}, (
            'a solve adds the hand-off and the solver run, a write adds the file stream'
        )
        assert all(seconds >= 0 for seconds in ran.values()), 'a wall clock cannot run backwards'

        snapshot = dict(ran)
        model.solve()
        assert model.diagnostics().seconds['solve'] >= ran['solve'], (
            'the clocks accumulate across solves, the way `solves` counts'
        )
        assert ran == snapshot, 'a diagnostics snapshot is its own dict, not a view of the running clocks'


#: Three blocks that differ only in scale. `badly_scaled` spans nine orders of
#: magnitude, `signed` carries `ordinary`'s coefficients negated, and one cost
#: is negative.
SCALING = {
    'dimensions': {'unit': {'dtype': 'str'}},
    'parameters': {'small': {'dims': ['unit']}, 'large': {'dims': ['unit']}, 'cost': {'dims': ['unit']}},
    'variables': {'p': {'dims': ['unit'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {
        'ordinary': {'dims': ['unit'], 'expression': 'p * small >= 1'},
        'badly_scaled': {'dims': ['unit'], 'expression': 'p * large <= 10000000'},
        'signed': {'dims': ['unit'], 'expression': '0 - p * small >= -100'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=unit)'},
}


SCALING_SOURCES = {
    'unit': ['a', 'b'],
    'small': pl.DataFrame({'unit': ['a', 'b'], 'value': [1.0, 4.0]}),
    'large': pl.DataFrame({'unit': ['a', 'b'], 'value': [1e6, 1e-3]}),
    'cost': pl.DataFrame({'unit': ['a', 'b'], 'value': [2.0, -0.5]}),
}


def test_the_coefficient_range_names_the_block_that_holds_the_outlier():
    """The spread of the matrix, per declaration, in magnitudes rather than signed extremes."""
    with sps.build(SCALING, SCALING_SOURCES) as model:
        spread = model.diagnostics().coefficient_range

    assert spread.to_dicts() == [
        {'constraint': 'ordinary', 'smallest': 1.0, 'largest': 4.0},
        {'constraint': 'badly_scaled', 'smallest': 1e-3, 'largest': 1e6},
        {'constraint': 'signed', 'smallest': 1.0, 'largest': 4.0},
    ], 'one row per block in build order, and `signed` is scaled exactly as `ordinary` is'

    worst = spread.select((pl.col('largest') / pl.col('smallest')).max()).item()
    assert worst == pytest.approx(1e9), 'the conditioning number to compare against what the solver reports'


def test_the_objective_range_is_read_beside_the_matrix_and_not_in_it():
    """Costs and coefficients are different faults, so they are different fields."""
    with sps.build(SCALING, SCALING_SOURCES) as model:
        seen = model.diagnostics()

    assert seen.objective_range == (0.5, 2.0), 'the objective is read off `obj`, never off the matrix'
    assert 'objective' not in seen.coefficient_range.get_column('constraint').to_list(), (
        'the objective is not a constraint block and does not appear as one'
    )


#: One variable whose bounds are the outlier and one with no finite bound.
#: `cap` carries a real cap and a 1e9 standing for uncapped.
BOUNDS = {
    'dimensions': {'unit': {'dtype': 'str'}},
    'parameters': {'cap': {'dims': ['unit']}, 'cost': {'dims': ['unit']}},
    'variables': {
        'capped': {'dims': ['unit'], 'bounds': {'lower': 0, 'upper': 'cap'}},
        'free': {'dims': ['unit'], 'bounds': {'lower': 0}},
    },
    'constraints': {
        'small_rhs': {'dims': ['unit'], 'expression': 'capped + free >= 1'},
        'large_rhs': {'dims': ['unit'], 'expression': 'capped <= 250000'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(capped * cost + free * cost, over=unit)'},
}


BOUNDS_SOURCES = {
    'unit': ['a', 'b'],
    'cap': pl.DataFrame({'unit': ['a', 'b'], 'value': [50.0, 1e9]}),
    'cost': pl.DataFrame({'unit': ['a', 'b'], 'value': [1.0, 2.0]}),
}


def test_the_bound_range_names_the_variable_whose_bounds_are_the_outlier():
    """The bound range, per declaration, on a model whose matrix is clean."""
    with sps.build(BOUNDS, BOUNDS_SOURCES) as model:
        seen = model.diagnostics()

    assert seen.bound_range.to_dicts() == [{'variable': 'capped', 'smallest': 50.0, 'largest': 1e9}], (
        'one row per variable block that declared a finite bound, and `free` declared none'
    )
    assert seen.coefficient_range.get_column('largest').max() == 1.0, (
        'the matrix is clean, which is what makes the bound range the only place the fault shows'
    )


def test_a_bound_of_zero_or_infinity_is_not_a_magnitude():
    """`lower: 0` and an unbounded side are excluded, so the pair reads as a solver's does."""
    with sps.build(BOUNDS, BOUNDS_SOURCES) as model:
        reported = model.diagnostics().bound_range.get_column('variable').to_list()

    assert reported == ['capped'], '`free` is lower: 0 with no upper, so it has no finite bound and no row'


def test_the_rhs_range_is_read_per_block_like_the_coefficients():
    """The fourth range a solver prints, and the last one answerable per declaration."""
    with sps.build(BOUNDS, BOUNDS_SOURCES) as model:
        seen = model.diagnostics().rhs_range

    assert seen.to_dicts() == [
        {'constraint': 'small_rhs', 'smallest': 1.0, 'largest': 1.0},
        {'constraint': 'large_rhs', 'smallest': 250000.0, 'largest': 250000.0},
    ], 'one row per constraint block in build order, magnitudes like every other range'


def test_the_four_ranges_are_four_fields_because_they_have_four_repairs():
    """A model can be clean on one axis and the offender on another."""
    with sps.build(BOUNDS, BOUNDS_SOURCES) as model:
        seen = model.diagnostics()

    def ratio(frame):
        return frame.select((pl.col('largest') / pl.col('smallest')).max()).item()

    assert ratio(seen.coefficient_range) == 1.0, 'every coefficient in this model is 1'
    assert ratio(seen.bound_range) == pytest.approx(2e7), 'the bounds are the fault, and only this field says so'
    assert ratio(seen.rhs_range) == 1.0, 'each block has one right-hand side, so no block spreads'
    assert seen.objective_range == (1.0, 2.0), 'costs are a separate axis with a separate repair'


def test_a_model_with_no_objective_has_no_objective_range():
    """A feasibility model has no costs to be badly scaled, and says so rather than lying with zeros."""
    feasibility = {k: v for k, v in SCALING.items() if k != 'objective'}
    with sps.build(feasibility, SCALING_SOURCES) as model:
        seen = model.diagnostics()

    assert seen.objective_range is None, 'no objective is not an objective whose coefficients span nothing'
    assert seen.coefficient_range.height == 3, 'the matrix is still reported — every constraint block is there'


#: A coefficient short of its dims, and a bound that is not.
SPARSE_SOURCE = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'p_max': {'dims': ['g']}, 'avail': {'dims': ['t', 'g']}},
    'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'capped': {'dims': ['t', 'g'], 'expression': 'p * avail <= 1'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}


SPARSE_SOURCES = {
    't': [0, 1],
    'g': ['wind', 'solar', 'gas'],
    'p_max': pl.DataFrame({'g': ['wind', 'solar', 'gas'], 'value': [1.0, 2.0, 3.0]}),
    'avail': pl.DataFrame({'t': [0, 0, 1], 'g': ['wind', 'solar', 'wind'], 'value': [1.0, 1.0, 1.0]}),
}


def test_a_parameter_short_of_its_dims_is_reported_rather_than_judged():
    """Which parameters arrived short, and by how much — not whether they should have."""
    with sps.build(SPARSE_SOURCE, SPARSE_SOURCES) as model:
        short = model.diagnostics().sparse_parameters

    assert short.to_dicts() == [{'parameter': 'avail', 'coordinates': 6, 'rows': 3, 'missing': 3}], (
        'the complete parameter is not a row here — an empty frame is the useful answer for a dense model'
    )


def test_a_model_whose_parameters_all_span_their_dims_reports_none():
    dense = {
        **SPARSE_SOURCES,
        'avail': pl.DataFrame({'t': [0, 0, 0, 1, 1, 1], 'g': ['wind', 'solar', 'gas'] * 2, 'value': [1.0] * 6}),
    }
    with sps.build(SPARSE_SOURCE, dense) as model:
        assert model.diagnostics().sparse_parameters.is_empty(), 'empty is what a complete model reports'


def test_the_sparsity_report_survives_the_model_being_released():
    """Summarised at attach, so it outlives the frames."""
    with sps.build(SPARSE_SOURCE, SPARSE_SOURCES) as model:
        held = model.diagnostics()
    released = model.diagnostics()

    assert released.sparse_parameters.equals(held.sparse_parameters)


#: A model whose one constraint divides by a parameter: with a value missing
#: the assembly refuses it, which is a raise *after* the attach has succeeded.
UNDEFINED_DIVISOR = {
    'dimensions': {'f': {'dtype': 'str'}},
    'parameters': {'d': {'dims': ['f']}},
    'variables': {'x': {'dims': ['f'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'c': {'dims': ['f'], 'expression': 'x / d <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

DENSE_DIVISOR = {'f': ['a', 'b'], 'd': pl.DataFrame({'f': ['a', 'b'], 'value': [2.0, 5.0]})}
HALF_A_DIVISOR = {'f': ['a', 'b'], 'd': pl.DataFrame({'f': ['a'], 'value': [2.0]})}


def test_a_build_that_raises_reports_the_bind_it_got_through_and_no_size():
    """A build that raised reports no size, neither its own nor the previous build's.

    What the attach measured is still true, and names the gap the build then
    refused.
    """
    with sps.build(UNDEFINED_DIVISOR, DENSE_DIVISOR) as model:
        built = model.diagnostics()
        assert (built.columns, built.rows) == (2, 2), 'the model under test builds before it is asked not to'

        with pytest.raises(sps.errors.DataError, match='used as a divisor'):
            model.update(HALF_A_DIVISOR)
        after = model.diagnostics()

    assert (after.columns, after.rows, after.nonzeros) == (0, 0, 0), (
        "a build that raised reported a size — a partial count, or the released build's"
    )
    assert after.sparse_parameters.to_dicts() == [{'parameter': 'd', 'coordinates': 2, 'rows': 1, 'missing': 1}], (
        'the attach that succeeded is still reported, and it names the gap the assembly then refused'
    )


def test_the_coefficient_range_survives_the_model_being_released():
    """Read off each share as it is built, so it outlives the frames it came from."""
    with sps.build(SCALING, SCALING_SOURCES) as model:
        held = model.diagnostics()
    released = model.diagnostics()

    assert released.coefficient_range.equals(held.coefficient_range), 'a released model still says how it was scaled'
    assert released.objective_range == held.objective_range


def test_the_largest_magnitude_agrees_with_the_oracle():
    """linopy answers the same question per constraint, and the two must not drift.

    Its `coefficientrange` is a signed min/max, so only the larger magnitude is
    comparable.
    """
    with differential(SCALING, SCALING_SOURCES) as run:
        ours = {row['constraint']: row['largest'] for row in run.engine.diagnostics().coefficient_range.to_dicts()}
        theirs = run.model.constraints.coefficientrange

    for name, largest in ours.items():
        expected = max(abs(theirs.loc[name, 'min']), abs(theirs.loc[name, 'max']))
        assert largest == pytest.approx(expected), f"the lanes disagree on the widest coefficient in '{name}'"
