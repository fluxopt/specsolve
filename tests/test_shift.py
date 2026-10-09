"""shift: time-coupled recurrences through both backends.

examples/storage.yaml is dispatch plus a cyclic battery:
soc == shift(soc, along=snapshot, offset=1, edge='wrap') + charge * 0.9 - discharge.
The linopy backend implements `edge='wrap'` with linopy's circular .roll(); the
relational backend lowers it to program.Translate — a pointwise ord-join remap.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import LanguageError
from tests.conftest import (
    DISPATCH_SPEC,
    EXAMPLES_DIR,
    by_coord,
    masked_operand_spec,
    override,
    relation,
    schema_of,
)
from tests.differential import differential
from tests.oracle import pd

STORAGE_YAML = EXAMPLES_DIR / 'storage.yaml'
STORAGE_SCHEMA = schema_of(STORAGE_YAML)


@pytest.fixture
def storage_inputs():
    """Peaky load that exceeds generation capacity at the peaks, so the
    battery is *required* (not just economic) and soc is genuinely coupled."""
    n_s = 48
    p_max = pd.Series({'wind': 80.0, 'gas': 70.0})
    cost = pd.Series({'wind': 1.0, 'gas': 40.0})
    t = np.arange(n_s)
    load = pd.Series(
        (110 + 60 * np.sin(2 * np.pi * t / 24)).round(3),  # peaks above the fleet's 150
        index=pd.RangeIndex(n_s, name='snapshot'),
    )
    return {
        'p_max': p_max,
        'cost': cost,
        'load': load,
        'snapshot': pd.RangeIndex(n_s, name='snapshot'),
        'generator': pd.Index(p_max.index, name='generator'),
    }


def _soc_trace(result):
    """(soc, prev-contribution inputs) as plain arrays, sorted by snapshot."""
    return tuple(
        result.to_pandas(name).set_index('snapshot')['value'].sort_index().to_numpy()
        for name in ('soc', 'charge', 'discharge')
    )


def _storage_variant(replacement: str) -> str:
    """``examples/storage.yaml`` with its wrap respelled as *replacement*."""
    original = STORAGE_YAML.read_text()
    wrap = "shift(soc, along=snapshot, offset=1, edge='wrap')"
    assert wrap in original, 'examples/storage.yaml no longer spells the wrap this file rewrites'
    return original.replace(wrap, replacement)


def _dimmed(storage_inputs: dict) -> dict:
    """The storage instance with its load eased, so an acyclic battery stays feasible."""
    return {**storage_inputs, 'load': (storage_inputs['load'] * 0.93).round(3)}


# ---------------------------------------------------------------------------
# the recurrence, end to end
# ---------------------------------------------------------------------------


def test_a_wrapping_edge_is_cyclic_on_both_lanes(storage_inputs):
    """`edge='wrap'` closes the recurrence, so `soc[0]` reads the last slot."""
    data = storage_inputs

    with differential(STORAGE_YAML, data, lp=True) as run:
        assert float(run.model.solution['discharge'].max()) > 1e-3, (
            'the battery must actually cycle for the model to be feasible'
        )

        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc, np.roll(soc, 1) + 0.9 * charge - discharge, atol=1e-6)


def test_shift_drops_the_row_it_has_no_predecessor_for_on_both_lanes(storage_inputs):
    """shift() = acyclic recurrence, and the first snapshot has *no* recurrence.

    ``soc[0]`` has no predecessor, so the vacated slot is absent (#289), it
    propagates through the equation, and the ``t=0`` row is not built — linopy's
    own reading of ``.shift()``. The linopy lane reaches the drop through linopy's
    absence propagation, the relational one through the vacated coordinates
    leaving the presence set.
    """
    acyclic = _storage_variant('shift(soc, along=snapshot, offset=1)')
    with differential(acyclic, _dimmed(storage_inputs)) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc[1:], soc[:-1] + 0.9 * charge[1:] - discharge[1:], atol=1e-6), (
            'the recurrence holds from the second snapshot on'
        )
        assert run.model.constraints['soc_balance'].labels.values[0] == -1, (
            't=0 is governed by its own bounds alone, so no row is built for it'
        )


def test_a_forward_shift_drops_the_row_at_the_far_end_on_both_lanes(storage_inputs):
    """`by=-1` is the mirror of the test above: the *last* snapshot has no successor (#837)."""
    forward = _storage_variant('shift(soc, along=snapshot, offset=-1)')
    with differential(forward, _dimmed(storage_inputs)) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert np.allclose(soc[:-1], soc[1:] + 0.9 * charge[:-1] - discharge[:-1], atol=1e-6), (
            'the recurrence reads forwards, and holds up to the second-to-last snapshot'
        )
        assert run.model.constraints['soc_balance'].labels.values[-1] == -1, (
            'the last snapshot has no successor, so no row is built for it'
        )


def test_a_forward_shift_with_a_zero_edge_keeps_the_far_row_on_both_lanes(storage_inputs):
    """`edge=0` fills what `by=-1` vacates, so the row at the far end survives (#830).

    The last snapshot keeps its equation, with the successor term contributing
    nothing.
    """
    filled = _storage_variant('shift(soc, along=snapshot, offset=-1, edge=0)')
    with differential(filled, _dimmed(storage_inputs)) as run:
        soc, charge, discharge = _soc_trace(run.result)
        assert run.model.constraints['soc_balance'].labels.values[-1] != -1, (
            'edge=0 asks for a value at the boundary, so the last row is built rather than dropped'
        )
        assert np.allclose(soc[-1], 0.9 * charge[-1] - discharge[-1], atol=1e-6), (
            'at the last snapshot the vacated successor contributes zero, not a wraparound'
        )


#: A mask that removes one interior coordinate, so the operand's own absence
#: sits where no edge is. `edge: 0` may fill the boundary and nothing else (#987).
MASKED_INTERIOR = masked_operand_spec('link', 'take <= shift(level, along=t, offset=1, edge=0)')


def test_a_zero_edge_fills_the_boundary_and_not_an_absence_that_was_already_there():
    """`edge:` is the opt-out for the slot the shift vacated, not for the operand.

    Three snapshots and `level` masked away at the middle one. The row at `t=0`
    is the vacated edge and is filled; the row at `t=2` reads the masked slot,
    which is absent for a reason the shift had nothing to do with, so it drops
    and `take[2]` is held by its own bound alone. Filling either alone is
    indistinguishable from filling both by row count, which is why the
    objective is asserted too.
    """
    sources = {
        't': pd.Index([0, 1, 2], name='t'),
        'usable': pd.Series([1.0, 0.0, 1.0], index=pd.Index([0, 1, 2], name='t')),
    }
    with differential(MASKED_INTERIOR, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 2, 'the boundary row and the one that reads a live slot, and no other'
        assert run.oracle == pytest.approx(10.0), (
            'take[2] reaches its bound because no row caps it — 0.0 would mean the masked slot was read as a zero'
        )


#: The same question asked of the two *gathers*: an offset that differs per
#: entity, and a shift closed inside a group. Each reaches the edge through its
#: own out-of-range mask.
BY_PARAMETER = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'usable': {'dims': ['t']}},
    'variables': {
        'level': {'dims': ['g', 't'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'link': {'dims': ['g', 't'], 'expression': 'take <= shift(level, along=t, offset=lead, edge=0)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(sum(take, over=g), over=t) - 1000 * sum(sum(level, over=g), over=t)',
    },
}

IN_GROUPS = masked_operand_spec(
    'link', 'take <= shift(level, along=t, offset=1, edge=0, by=season_of, within=season)', grouped=True
)

#: The same, with nothing masked at all: `level` carries no `where`, so the
#: operand reaches the shift with no presence frame of its own (#1061).
IN_GROUPS_UNMASKED = masked_operand_spec(
    'link', 'take <= shift(level, along=t, offset=1, edge=0, by=season_of, within=season)', grouped=True, masked=False
)

#: A fourth snapshot the relation sends nowhere, for both models above.
GROUPLESS_SOURCES = {
    't': pd.DataFrame({'t': [0, 1, 2, 3]}),
    'season_of': relation('t', 'season', [0, 1, 2, 3], ['s1', 's1', 's2', None]),
    'season': pd.Index(['s1', 's2'], name='season'),
}


def test_a_per_entity_offset_fills_its_own_edge_and_not_the_mask_under_it():
    """The gather's edge, asked the same way as the scalar shift's.

    One technology, a lead of one month, and `level` masked away at the middle
    month: `t=0` is the vacated edge and is filled, `t=2` reads the masked slot
    and drops, so two rows are built and `take` at the last month is capped by
    nothing. Which coordinates are vacated is per entity (#1049).
    """
    sources = {
        'g': pd.Index(['a'], name='g'),
        't': pd.Index([0, 1, 2], name='t'),
        'lead': pd.Series([1], index=pd.Index(['a'], name='g')),
        'usable': pd.Series([1.0, 0.0, 1.0], index=pd.Index([0, 1, 2], name='t')),
    }
    with differential(BY_PARAMETER, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 2, 'the vacated month and the one reading a live slot'
        assert run.oracle == pytest.approx(10.0), 'take at the last month is capped by nothing'


def test_a_grouped_shift_fills_each_groups_edge_and_not_the_mask_under_it():
    """And the third path: the edge is per group, the mask is not.

    Two seasons of two snapshots, `level` masked away at the first. Each
    season's opening snapshot is vacated and filled; the snapshot that reads
    the masked one drops.
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3]}),
        'season_of': relation('t', 'season', [0, 1, 2, 3], ['s1', 's1', 's2', 's2']),
        'season': pd.Index(['s1', 's2'], name='season'),
        'usable': pd.Series([0.0, 1.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
    }
    with differential(IN_GROUPS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, "both seasons' opening rows, and the one reading a live slot"
        assert run.oracle == pytest.approx(10.0), 'the snapshot whose predecessor is masked is capped by nothing'


def test_a_coordinate_the_relation_sends_nowhere_is_absent_rather_than_vacated():
    """A snapshot in no season at all: it reaches nothing for a reason the
    shift had nothing to do with, so `edge=0` does not speak for it (#1061).

    Off a group's start the shift vacated the slot and the edge fills it; a
    coordinate the relation sends nowhere never had a predecessor to lose. It
    is the null a partial relation gets everywhere else (#969, `sum(by=)`,
    `at()`).
    """
    with differential(IN_GROUPS_UNMASKED, GROUPLESS_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, (
            "each season's opening row and the one reading a live slot — not the group-less snapshot"
        )
        assert run.oracle == pytest.approx(10.0), 'take at the group-less snapshot is capped by nothing'


def test_a_group_less_coordinate_stays_absent_under_a_mask_that_removes_nothing():
    """The same question with a `where` on the operand that masks no row.

    With a presence frame the edge is rebuilt from the vacated set, without one
    from the grouped labels; a mask removing nothing decides no row.
    """
    sources = {**GROUPLESS_SOURCES, 'usable': pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t'))}
    with differential(IN_GROUPS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'a mask that removes nothing changes no row'
        assert run.oracle == pytest.approx(10.0), 'take at the group-less snapshot is capped by nothing'


#: A nonzero edge over a variable-free operand, reached with a per-entity
#: offset: the other reader of the edge frame, and the other half of #1049.
BY_PARAMETER_CONSTANT = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}, 'eff': {'dims': ['g', 't']}},
    'variables': {'x': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}}},
    'constraints': {'link': {'dims': ['g', 't'], 'expression': 'x * shift(eff, along=t, offset=lead, edge=1) <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(x, over=g), over=t)'},
}


def test_a_per_entity_offset_writes_a_nonzero_edge_where_that_entity_vacates():
    """The fill is a value, so the vacated coordinate is written rather than
    left out — and with a lead of one month only the first month is vacated.

    Efficiencies of 1, 2 and 4 give caps of 10, 10 and 5: the first from the
    edge's own 1, the second and third from the month before.
    """
    sources = {
        'g': pd.Index(['a'], name='g'),
        't': pd.Index([0, 1, 2], name='t'),
        'lead': pd.Series([1], index=pd.Index(['a'], name='g')),
        'eff': pd.DataFrame({'g': ['a'] * 3, 't': [0, 1, 2], 'value': [1.0, 2.0, 4.0]}),
    }
    with differential(BY_PARAMETER_CONSTANT, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'every month is capped, the first by the edge it was given'
        assert run.oracle == pytest.approx(25.0), '10 + 10 + 5'


#: The two gathers *together*: a lag that is per group rather than per entity,
#: which is a parameter declared over the dimension the partition groups into.
#: A period's own construction lead time, on a flat snapshot axis (#1161).
PER_GROUP_OFFSET = {
    'dimensions': {'t': {'dtype': 'int'}, 'period': {'dtype': 'int'}},
    'relations': {'period_of': {'key': 't', 'values': 'period'}},
    'parameters': {'lead': {'dims': ['period'], 'dtype': 'int'}, 'v': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': -100, 'upper': 100}}},
    'constraints': {
        'reads': {
            'dims': ['t'],
            'expression': 'p == shift(v, along=t, offset=lead, by=period_of, within=period, edge=0)',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

#: The same lag over a *variable*, where the edge is a term that is not there
#: rather than a number written into a frame — and one snapshot the relation
#: sends nowhere, which no lag reaches and no edge speaks for.
PER_GROUP_OFFSET_TERMS = {
    'dimensions': {'t': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'relations': {'season_of': {'key': 't', 'values': 'season'}},
    'parameters': {'lead': {'dims': ['season'], 'dtype': 'int'}, 'cap': {'dims': ['t']}},
    'variables': {
        'level': {'dims': ['t'], 'bounds': {'lower': 'cap', 'upper': 'cap'}},
        'take': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}},
    },
    'constraints': {
        'link': {
            'dims': ['t'],
            'expression': 'take <= shift(level, along=t, offset=lead, by=season_of, within=season, edge=0)',
        }
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(take, over=t)'},
}

#: Per entity *and* per group at once: one key the frame carries and one it
#: reaches through the relation, in the same join.
PER_ENTITY_AND_PER_GROUP = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'relations': {'season_of': {'key': 't', 'values': 'season'}},
    'parameters': {'lead': {'dims': ['g', 'season'], 'dtype': 'int'}, 'v': {'dims': ['g', 't']}},
    'variables': {'p': {'dims': ['g', 't'], 'bounds': {'lower': -100, 'upper': 100}}},
    'constraints': {
        'reads': {
            'dims': ['g', 't'],
            'expression': 'p == shift(v, along=t, offset=lead, by=season_of, within=season, edge=0)',
        }
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}


def _by_t(result, name: str) -> list[float]:
    """One variable's primals in coordinate order, which is what a lag is about."""
    return result.primal(name).sort('t')['value'].to_list()


def test_a_per_group_offset_translates_each_group_by_its_own_lag():
    """The lag is the group's, so two periods with different lead times each
    reach back their own distance and vacate their own opening rows.

    The first period leads by one and the second by two, which no single
    number reproduces: shifting the whole axis by either lands three of the
    six coordinates somewhere else.
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3, 4, 5]}),
        'period_of': relation('t', 'period', [0, 1, 2, 3, 4, 5], [2030] * 3 + [2050] * 3),
        'period': pd.Index([2030, 2050], name='period'),
        'lead': pd.Series([1, 2], index=pd.Index([2030, 2050], name='period')),
        'v': pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], index=pd.Index(range(6), name='t')),
    }
    with differential(PER_GROUP_OFFSET, sources, lp=True) as run:
        assert list(run.model.constraints['reads'].coords) == ['t'], (
            'the row is the snapshot, and the period its lag was read at is not a coordinate of it'
        )
        assert run.model.constraints['reads'].size == run.engine.diagnostics().rows == 6, (
            'one row per snapshot, and the same number of them on both lanes'
        )
        assert _by_t(run.result, 'p') == [0.0, 10.0, 20.0, 0.0, 0.0, 40.0], (
            "each period reaches back its own lead inside its own group, and its opening rows take the edge's zero"
        )
        assert run.oracle == pytest.approx(70.0), '0 + 10 + 20 + 0 + 0 + 40'


def test_a_per_group_offset_over_a_variable_vacates_each_groups_opening_rows():
    """The same lag where the operand carries terms rather than values.

    ``level`` is pinned to ``cap`` by its bounds, so what ``take`` may reach is
    the lag read plainly: the second season leads by two, so both of its
    opening rows are capped by the edge's zero rather than by a term. The
    snapshot in no season is capped by nothing at all — it belongs to no group,
    so it reaches nothing, and ``edge=0`` does not speak for it (#1061).
    """
    sources = {
        't': pd.DataFrame({'t': [0, 1, 2, 3, 4, 5, 6]}),
        'season_of': relation('t', 'season', range(7), ['s1'] * 3 + ['s2'] * 3 + [None]),
        'season': pd.Index(['s1', 's2'], name='season'),
        'lead': pd.Series([1, 2], index=pd.Index(['s1', 's2'], name='season')),
        'cap': pd.Series([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], index=pd.Index(range(7), name='t')),
    }
    with differential(PER_GROUP_OFFSET_TERMS, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 6, 'one row per snapshot in a season, and none for the one in no season'
        assert _by_t(run.result, 'take') == [0.0, 10.0, 20.0, 0.0, 0.0, 40.0, 100.0], (
            'each season reaches back its own lead, and the group-less snapshot is capped by nothing'
        )
        assert run.oracle == pytest.approx(170.0)


def test_an_offset_may_differ_per_entity_and_per_group_at_once():
    """Two keys, one join: the entity's own column and the group's, which the
    frame carries only as the relation's value.

    The second unit leads by two in the second season and by one everywhere
    else, so it alone vacates both of that season's rows.
    """
    units, periods = ['a', 'b'], [0, 1, 2, 3]
    sources = {
        'g': pd.Index(units, name='g'),
        't': pd.DataFrame({'t': periods}),
        'season_of': relation('t', 'season', periods, ['s1', 's1', 's2', 's2']),
        'season': pd.Index(['s1', 's2'], name='season'),
        'lead': pd.DataFrame({'g': ['a', 'a', 'b', 'b'], 'season': ['s1', 's2'] * 2, 'value': [1, 1, 1, 2]}),
        'v': pd.DataFrame(
            {
                'g': [u for u in units for _ in periods],
                't': periods * 2,
                'value': [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0],
            }
        ),
    }
    with differential(PER_ENTITY_AND_PER_GROUP, sources, lp=True) as run:
        read = run.result.primal('p').sort('g', 't')
        assert read['value'].to_list() == [0.0, 10.0, 0.0, 30.0, 0.0, 1.0, 0.0, 0.0], (
            "the lead is read at the pair, not at either key alone — 'b' vacates both rows of the season it leads by two"
        )
        assert run.oracle == pytest.approx(41.0), '(0 + 10 + 0 + 30) + (0 + 1 + 0 + 0)'


def test_shift_semantics_are_positional_not_lexicographic(storage_inputs):
    """Coords whose sorted order differs from declared order (string labels:
    lexicographic t0,t1,t10,... vs positional t0..t47). Both backends must
    couple the same neighbours."""
    labels = pd.Index([f't{i}' for i in range(len(storage_inputs['snapshot']))], name='snapshot')
    assert list(labels.sort_values()) != list(labels), 'the fixture is only a fixture if sorted != positional'
    data = {**storage_inputs, 'load': storage_inputs['load'].set_axis(labels), 'snapshot': labels}

    original = STORAGE_YAML.read_text()
    assert 'dtype: int' in original
    with differential(original.replace('dtype: int', 'dtype: str'), data):
        pass  # agreement on the objective is the whole assertion


RAMP_SPEC = override(
    DISPATCH_SPEC,
    **{
        'parameters.ramp_max': {'dims': ['generator']},
        'constraints.ramp_up': {
            'dims': ['snapshot', 'generator'],
            'where': 'snapshot > 0',
            'expression': 'p - shift(p, along=snapshot, offset=1) <= ramp_max',
        },
    },
)


def test_a_where_on_dimension_coordinates_means_the_same_on_both_lanes():
    """`where: "snapshot > 0"` means the same on both lanes: a ramp that skips the first snapshot."""
    n_s = 12
    rng = np.random.default_rng(11)
    data = {
        'p_max': pd.Series({'wind': 80.0, 'gas': 200.0}),
        'cost': pd.Series({'wind': 1.0, 'gas': 40.0}),
        'ramp_max': pd.Series({'wind': 100.0, 'gas': 25.0}),  # binding on gas
        'load': pd.Series(
            (rng.uniform(0.3, 0.9, n_s) * 200.0).round(3),
            index=pd.RangeIndex(n_s, name='snapshot'),
        ),
    }
    data |= {'snapshot': pd.RangeIndex(n_s, name='snapshot'), 'generator': ['wind', 'gas']}

    with differential(RAMP_SPEC, data) as run:
        active = int((run.model.constraints['ramp_up'].labels != -1).sum())
        assert active == (n_s - 1) * 2, (
            'the mask must bite: the first snapshot is dropped per generator, and a masked row on '
            'the linopy lane carries label -1'
        )


# ---------------------------------------------------------------------------
# lowering
# ---------------------------------------------------------------------------


FILL_IDENTITY_SPEC = """
dimensions: {t: {dtype: int}}
parameters:
  eff: {dims: [t]}
variables:
  x: {dims: [t], bounds: {lower: 0, upper: 100}}
constraints:
  c:
    dims: [t]
    expression: "x * shift(eff, along=t, offset=1, edge=1) <= 10"
objective: {sense: maximize, expression: "sum(x, over=t)"}
"""


def test_the_fill_a_product_wants_is_one_not_zero():
    """``fill=`` takes the identity of the *position*, which is why it takes a number.

    0 is the identity of a sum, 1 of a product. ``x * shift(eff, along=t,
    offset=1, edge=0)`` would force ``x`` to zero at the first coordinate, where
    ``fill=1`` leaves it governed by its own bound.
    """
    data = {'t': [0, 1, 2], 'eff': pd.Series({0: 2.0, 1: 4.0, 2: 5.0})}
    with differential(FILL_IDENTITY_SPEC, data, lp=True) as run:
        x = by_coord(run.result, 'x', 't')
        assert x[0] == pytest.approx(10.0), 't=0: the fill is 1, so the bound is 10/1'
        assert x[1] == pytest.approx(5.0), 't=1: eff[0] = 2, so 10/2'
        assert x[2] == pytest.approx(2.5), 't=2: eff[1] = 4, so 10/4'


EDGE_SPEC = {
    'dimensions': {'t': {'dtype': 'int'}, 'wrap': {'dtype': 'str'}},
    'parameters': {'c': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t', 'wrap'], 'bounds': {'lower': 0, 'upper': 5}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x * c)'},
}


def _with(expr):
    return {**EDGE_SPEC, 'constraints': {'r': {'dims': ['t', 'wrap'], 'expression': expr}}}


@pytest.mark.parametrize(
    'edge',
    ["edge='wrap'", 'edge="wrap"', 'edge=0'],
    ids=['single', 'double', 'zero fill'],
)
def test_an_edge_policy_is_quoted_or_a_number(edge):
    """The keyword is quoted; the fill is bare.

    A bare word in a kwarg value is a *name to resolve* — `over=wrap` names a
    dimension — so the one closed keyword `edge=` takes has to say it is a
    literal. Numbers need no quotes because a number is never a name.
    """
    sps.check(_with(f'x - shift(x, along=t, offset=1, {edge}) <= 1'))


def test_a_bare_wrap_names_a_dimension_and_is_refused():
    """A bare `wrap` in `edge=` is refused, so one token cannot mean two things.

    The model declares a dimension called `wrap`, which `over=wrap` would name.
    """
    with pytest.raises(ValueError) as exc:
        sps.check(_with('x - shift(x, along=t, offset=1, edge=wrap) <= 1'))

    assert 'bare name where a keyword belongs' in str(exc.value)
    assert "edge='wrap'" in str(exc.value), 'the refusal has to name the rewrite'


def test_a_quoted_keyword_outside_a_kwarg_does_not_parse():
    """Quotes are for closed keywords in kwarg values, not for arithmetic: the grammar refuses this."""
    with pytest.raises(ValueError) as exc:
        sps.check(_with("x - 'wrap' <= 1"))

    assert 'Failed to parse expression' in str(exc.value)


def _shift_over_data(where: str | None = None, edge: str | None = None) -> dict[str, object]:
    shift = f'shift(dt, along=t, offset=1, edge={edge})' if edge else 'shift(dt, along=t, offset=1)'
    constraint: dict[str, object] = {'dims': ['t'], 'expression': f'x <= {shift}'}
    if where is not None:
        constraint['where'] = where
    return {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'c': constraint},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }


def test_the_bare_shift_refusal_names_the_pair_that_actually_omits_the_row():
    """`edge=` and `where:` are a companion pair here, not a choice.

    Each is wrong alone: a `where` does not lift the refusal, because it is
    decided on the expression before any mask is read; and `edge=0` alone
    leaves a row at the vacated coordinate whose bound is that zero. The
    wording is held too, since `edge=0` alone builds, solves and is wrong.
    """
    with pytest.raises(LanguageError, match='vacated positions') as bare:
        sps.check(_shift_over_data())
    with pytest.raises(LanguageError, match='vacated positions') as masked:
        sps.check(_shift_over_data(where='t > 0'))
    assert str(bare.value) == str(masked.value), 'a mask lifts the refusal, so it is an alternative after all'

    message = str(bare.value)
    assert 'where' in message, 'the way to omit the row has to be reachable from the error'
    assert "edge='wrap'" in message
    assert 'edge=0 alone' in message, 'the trap has to be named, not just the remedy'


def test_edge_zero_alone_binds_the_vacated_row_and_a_where_frees_it():
    """`edge=0` alone solves, and the answer is wrong in the direction that looks like a tight model."""
    sources = {'t': [0, 1, 2], 'dt': pl.DataFrame({'t': [0, 1, 2], 'value': [1.0, 1.0, 1.0]})}
    pinned = sps.solve(_shift_over_data(edge='0'), sources)
    omitted = sps.solve(_shift_over_data(edge='0', where='t > 0'), sources)

    assert pinned.primal('x')['value'].to_list()[0] == 0.0, 'edge=0 alone should pin the vacated row'
    assert omitted.primal('x')['value'].to_list()[0] == 5.0, 'the where should omit it entirely'
    assert omitted.objective > pinned.objective


NESTED_SHIFTS = {
    'same-dim': 'shift(shift(p, along=t, offset=1), along=t, offset=1)',
    'cross-dim': 'shift(shift(p, along=t, offset=1), along=g, offset=1)',
    'cross-dim-reversed': 'shift(shift(p, along=g, offset=1), along=t, offset=1)',
    'triple-mixed': 'shift(shift(shift(p, along=t, offset=1), along=g, offset=1), along=t, offset=1)',
    'inner-fill': 'shift(shift(p, along=t, offset=1, edge=0), along=t, offset=1)',
    'outer-wrap': "shift(shift(p, along=t, offset=1), along=t, offset=1, edge='wrap')",
}


@pytest.mark.parametrize('rhs', NESTED_SHIFTS.values(), ids=list(NESTED_SHIFTS))
def test_a_nested_shift_agrees_with_the_oracle(rhs: str):
    """A shift over a shift, in every arrangement of edge and dimension.

    The coefficient and the `+ 1` make the row bind: without them every
    variable sits at its upper bound.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int'}, 'g': {'dtype': 'str'}},
        'parameters': {'c': {'dims': ['g']}},
        'variables': {'p': {'dims': ['t', 'g'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'k': {'dims': ['t', 'g'], 'expression': f'p <= 0.5 * {rhs} + 1'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p * c)'},
    }
    data = {'t': [0, 1, 2, 3, 4], 'g': ['a', 'b'], 'c': pd.Series([1.0, 2.0], index=pd.Index(['a', 'b'], name='g'))}
    with differential(spec, data) as run:
        primal = run.result.primal('p')['value'].to_numpy()
        assert not np.allclose(primal, 5.0), 'nothing binds, so the lanes would agree on an unconstrained model'


@pytest.mark.parametrize('edge', ["'wrap'", '0'], ids=['wrap', 'fill'])
def test_an_offset_may_differ_per_entity(edge: str):
    """`by=` names a parameter: each entity is translated by its own amount.

    An order placed at *t* arrives at *t + lead*, demand falls only in the last
    period, and the two units have different leads, so an offset read once for
    both would order in the wrong period.
    """
    lead = {'slow': 1, 'fast': 2}
    units, periods = list(lead), [0, 1, 2, 3]
    spec = {
        'dimensions': {
            'g': {'dtype': 'str'},
            't': {'dtype': 'int'},
        },
        'parameters': {
            'lead': {'dims': ['g'], 'dtype': 'int'},
            'c': {'dims': ['g']},
            'demand': {'dims': ['g', 't']},
        },
        'variables': {'order': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 9}}},
        'constraints': {
            'arrive': {
                'dims': ['g', 't'],
                'expression': f'shift(order, along=t, offset=lead, edge={edge}) >= demand',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(order * c)'},
    }
    data = {
        'g': units,
        't': periods,
        'lead': pd.Series([lead[u] for u in units], index=pd.Index(units, name='g')),
        'c': pd.Series([1.0, 1.0], index=pd.Index(units, name='g')),
        'demand': pd.DataFrame(
            {
                'g': [u for u in units for _ in periods],
                't': periods * len(units),
                'value': [0.0, 0.0, 0.0, 5.0] * len(units),
            }
        ),
    }
    with differential(spec, data) as run:
        assert run.result.objective == pytest.approx(10.0)
        placed = run.result.primal('order').filter(pl.col('value') > 1e-9)
        assert dict(zip(placed['g'].to_list(), placed['t'].to_list(), strict=True)) == {'slow': 2, 'fast': 1}, (
            'each unit orders one of its own lead times before the demand, not a shared one'
        )


def test_a_named_offset_must_say_what_the_vacated_positions_contribute():
    """The absent edge is refused for a named offset.

    Absence is carried by a presence frame keyed by the translated dimension
    alone, and a per-entity offset vacates a *different* slot for each entity.
    The two edges that write their own answer are allowed.
    """
    spec = {
        'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
        'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}},
        'variables': {'x': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'k': {'dims': ['g', 't'], 'expression': 'x >= shift(x, along=t, offset=lead)'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x * 1.0)'},
    }
    with pytest.raises(LanguageError, match='vacated positions absent'):
        sps.check(spec)


def _reindexed_parameter_spec(op: str) -> dict:
    return {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'dt': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {'r': {'dims': ['t'], 'expression': f'x <= {op}'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x, over=t)'},
    }


@pytest.mark.parametrize(
    ('op', 'expected'),
    [
        pytest.param(
            "shift(dt, along=t, offset=1, edge='wrap')",
            {0: 7.0, 1: 5.0, 2: 6.0},
            id='cyclic-vacates-nothing-so-t0-reads-the-last-value',
        ),
        pytest.param(
            'shift(dt, along=t, offset=1, edge=0)',
            {0: 0.0, 1: 5.0, 2: 6.0},
            id='the-vacated-position-contributes-zero-which-pins',
        ),
    ],
)
def test_roll_and_filled_shift_re_index_a_parameter_not_only_a_variable(op, expected):
    """``array`` is any node, so these operators read a parameter.

    ``fill=0`` pins ``t=0``; spelled out, it is a legitimate thing to ask for.
    """
    data = {'t': [0, 1, 2], 'dt': pd.Series({0: 5.0, 1: 6.0, 2: 7.0})}
    with differential(_reindexed_parameter_spec(op), data, lp=True) as run:
        x = by_coord(run.result, 'x', 't')
        for t, want in expected.items():
            assert x[t] == pytest.approx(want, abs=1e-9), f'{op} at t={t}'


def test_a_bare_shift_over_data_is_refused_rather_than_filled():
    """The pin, removed at its source (#289).

    ``x <= shift(dt, along=t, offset=1)`` would bound ``x`` at the first
    coordinate from a slot that has no value, and a parameter has no absence to
    propagate. ``sps.check()`` refuses it, since the operand is variable-free by
    declaration.
    """
    spec = _reindexed_parameter_spec('shift(dt, along=t, offset=1)')
    with pytest.raises(LanguageError) as exc:
        sps.check(spec)
    assert 'edge=0' in str(exc.value), 'the refusal must name the escape hatch'
    assert "edge='wrap'" in str(exc.value), 'and the policy for a genuinely cyclic horizon'


@pytest.mark.parametrize(
    ('edge', 'expected'),
    [pytest.param("edge='wrap'", 7.0, id='wrap'), pytest.param('edge=0', 3.0, id='fill')],
)
def test_an_offset_with_no_dimensions_is_one_offset_for_every_position(edge: str, expected: float):
    """A dimensionless int offset moves every position as the literal ``offset=1`` does (#1891).

    The offset was joined to the rows on its keys, which are none, and polars
    refused the join.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'k': {'dims': [], 'dtype': 'int'}, 'p': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0}}},
        'constraints': {'c': {'dims': ['t'], 'expression': f'x >= shift(p, along=t, offset=k, {edge})'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
    }
    with differential(spec, {'t': [1, 2, 3], 'k': 1, 'p': [1.0, 2.0, 4.0]}) as run:
        assert run.oracle == pytest.approx(expected), 'each x holds the value one position before it, or the edge'
