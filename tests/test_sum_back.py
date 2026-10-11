"""sum_back: a trailing window whose width is data, through both backends.

A minimum up time, a rolling emissions budget, a delivery horizon: the width
is a column in the source data.
"""

from __future__ import annotations

from copy import deepcopy

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DimensionError, LanguageError
from tests.conftest import masked_operand_spec, relation
from tests.differential import differential
from tests.oracle import pd

MIN_UP = {'slow': 3, 'fast': 1}
UNITS, PERIODS = list(MIN_UP), [0, 1, 2, 3, 4]


def up_time_spec(edge: str | None) -> dict:
    """A unit that starts must stay on for its own minimum up time.

    The start is forced at the *last* period, which is what makes the edge
    visible: acyclically the window reaches nothing beyond the axis, so only
    that period is held on, while under ``wrap`` it reaches around into the
    first two.
    """
    window = 'sum_back(started, along=t, window=min_up' + (f", edge='{edge}'" if edge else '') + ')'
    return {
        'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
        'parameters': {
            'min_up': {'dims': ['g'], 'dtype': 'int'},
            'must_start': {'dims': ['g', 't']},
            'cost': {'dims': ['g']},
        },
        'variables': {
            'started': {'dims': ['g', 't'], 'domain': 'binary'},
            'on': {'dims': ['g', 't'], 'domain': 'binary'},
        },
        'constraints': {
            'starts_when_told': {'dims': ['g', 't'], 'expression': 'started >= must_start'},
            'stays_up_its_own_time': {'dims': ['g', 't'], 'expression': f'{window} <= on'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(on * cost)'},
    }


UP_TIME_DATA = {
    'g': UNITS,
    't': PERIODS,
    'min_up': pd.Series([MIN_UP[u] for u in UNITS], index=pd.Index(UNITS, name='g')),
    'cost': pd.Series([1.0, 1.0], index=pd.Index(UNITS, name='g')),
    'must_start': pd.DataFrame(
        {
            'g': [u for u in UNITS for _ in PERIODS],
            't': PERIODS * len(UNITS),
            'value': [0.0, 0.0, 0.0, 0.0, 1.0] * len(UNITS),
        }
    ),
}


@pytest.mark.parametrize(
    ('edge', 'expected'),
    [
        pytest.param(None, {('slow', 4), ('fast', 4)}, id='acyclic'),
        pytest.param('wrap', {('slow', 4), ('slow', 0), ('slow', 1), ('fast', 4)}, id='wrap'),
    ],
)
def test_a_window_width_may_differ_per_entity(edge: str | None, expected: set):
    """`window=` names a parameter: each entity gets a window of its own length.

    The instance discriminates twice over. The two units carry different
    widths, so a width read once for both would hold the wrong one on; and the
    start sits at the last period, so an edge read wrongly would either lose
    the wrapped periods or invent them.
    """
    with differential(up_time_spec(edge), UP_TIME_DATA) as run:
        assert run.result.objective == pytest.approx(float(len(expected)))
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == expected, (
            'each unit is held on for its own window, reaching the edge the way the model asked'
        )


def test_a_literal_width_is_the_last_n_positions():
    """A number where the width is the same everywhere, and 1 is the operand."""
    spec = up_time_spec(None)
    spec['constraints']['stays_up_its_own_time']['expression'] = 'sum_back(started, along=t, window=2) <= on'
    with differential(spec, UP_TIME_DATA) as run:
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == {('slow', 4), ('fast', 4)}, (
            'the window reaches back two positions, and only the last one is on the axis'
        )


def test_an_operand_carrying_a_constant_owes_the_window_of_constants():
    """`x - p` puts a constant beside the variable in every lag, and both add up.

    Each lag's constant is summed into the row. The right-hand sides are
    pinned as numbers because the differential agreement alone would stay
    green if both lanes dropped every constant at once.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int'}},
        'parameters': {'p': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'w': {'dims': ['t'], 'expression': 'sum_back(x - p, along=t, window=2) >= 0'}},
        'objective': {'sense': 'minimize', 'expression': 'sum(x, over=t)'},
    }
    data = {'t': [0, 1, 2, 3], 'p': pd.DataFrame({'t': [0, 1, 2, 3], 'value': [1.0, 2.0, 3.0, 4.0]})}
    with differential(spec, data) as run:
        assert run.oracle == pytest.approx(10.0), 'the optimum pays exactly the constants the windows owe'
        rows = run.model.constraints['w'].to_polars()
        rhs = sorted(rows.group_by('labels').agg(pl.col('rhs').first())['rhs'].to_list())
        assert rhs == [1.0, 3.0, 5.0, 7.0], 'each row owes its own window of constants, moved to the right-hand side'


#: A window over an operand that is masked away at one interior position. The
#: width decides what the mask means: a wider window reaches it *and* live
#: positions, a width of 1 reaches nothing else at all (#1059, #1060).
MASKED_WINDOW = masked_operand_spec('held', 'take <= sum_back(level, along=t, window=1)')

MASKED_WINDOW_SOURCES = {
    't': pd.Index([0, 1, 2, 3], name='t'),
    'usable': pd.Series([1.0, 0.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
}


def masked_window_spec(window: str) -> dict:
    """`MASKED_WINDOW` with the window respelled, the rest of it fixed."""
    spec = deepcopy(MASKED_WINDOW)
    spec['constraints']['held']['expression'] = f'take <= sum_back(level, along=t, {window})'
    return spec


def test_a_window_that_reaches_nothing_builds_no_row():
    """A window is short, not empty — until every position it spans is masked.

    Width 1 at the masked snapshot: the window is exactly the slot that is not
    there, so nothing live is left to sum and the row is a statement about
    constants alone. It is not built, and `take[1]` — capped by no row —
    reaches its own bound, which is what tells the two apart from the objective
    rather than only from the row count (#1059).
    """
    with differential(MASKED_WINDOW, MASKED_WINDOW_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'the snapshot whose whole window is masked away holds no row'
        assert run.oracle == pytest.approx(10.0), (
            'take[1] is capped by nothing — 0.0 would mean a row asserting take <= 0 was built there'
        )


ZERO_WIDTH = {
    'dimensions': {'t': {'dtype': 'int'}, 'u': {'dtype': 'str'}},
    'parameters': {'w': {'dims': ['u'], 'dtype': 'int'}, 'need': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t', 'u'], 'bounds': {'lower': 0}}},
    'constraints': {
        'meet': {'dims': ['t'], 'expression': 'sum(x, over=u) >= need'},
        'window': {'dims': ['t', 'u'], 'expression': 'sum_back(x, along=t, window=w) >= 0'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
}

ZERO_WIDTH_SOURCES = {
    't': pd.Index([0, 1, 2], name='t'),
    'u': pd.Index(['a', 'b'], name='u'),
    'w': pd.Series([0, 0], index=pd.Index(['a', 'b'], name='u')),
    'need': pd.Series([1.0, 2.0, 3.0], index=pd.Index([0, 1, 2], name='t')),
}


def test_a_window_whose_every_width_is_zero_builds_no_row():
    """A per-entity width can be zero everywhere, and then no window row is built.

    A min-up-time model on a fleet with no committable unit (#1306).
    """
    with differential(ZERO_WIDTH, ZERO_WIDTH_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'only the three meet rows are built; every window reached nothing'
        assert run.oracle == pytest.approx(6.0), 'the objective is the demand alone, no window row binding'


#: A width declared over the group's own dimension, and one coordinate the
#: relation sends nowhere: the width is read through the relation, so the unmapped
#: snapshot carries no width at all.
UNMAPPED_WIDTH = {
    'dimensions': {'t': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
    'relations': {'season_of': {'key': 't', 'values': 'season'}},
    'parameters': {'w': {'dims': ['season'], 'dtype': 'int'}, 'price': {'dims': ['t']}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 5}}},
    'constraints': {
        'rolling': {'dims': ['t'], 'expression': 'sum_back(x, along=t, window=w, by=season_of, within=season) <= 4'}
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(x * price, over=t)'},
}

UNMAPPED_WIDTH_SOURCES = {
    't': pd.Index([0, 1, 2, 3], name='t'),
    'season': pd.Index(['summer'], name='season'),
    'season_of': pd.DataFrame({'t': [0, 1, 2], 'season': ['summer'] * 3}),
    'w': pd.Series([2], index=pd.Index(['summer'], name='season')),
    'price': pd.Series([1.0, 1.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
}


def test_a_width_read_through_a_relation_that_maps_nothing_there_builds_no_row():
    """A coordinate in no group has no width, which is a window of nothing (#1535)."""
    with differential(UNMAPPED_WIDTH, UNMAPPED_WIDTH_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 3, 'the snapshot in no season carries no width, so it holds no row'
        assert run.oracle == pytest.approx(13.0), (
            'the three seasoned snapshots are capped at 8 between them; the unmapped one reaches its own bound of 5'
        )


@pytest.mark.parametrize('window', ['window=2', "window=2, edge='wrap'"], ids=['acyclic', 'wrap'])
def test_a_masked_slot_the_window_reaches_is_a_zero_not_an_absence(window: str):
    """The masked slot contributes nothing and takes nothing with it.

    Two positions wide, so every snapshot's window reaches at least one live
    slot even where it also reaches the masked one. Absence stops at a
    reduction, so all four rows are asserted and every `take` is capped —
    losing a row would show up as an uncapped `take` worth 10 (#1059, #1060).
    """
    with differential(masked_window_spec(window), MASKED_WINDOW_SOURCES, lp=True) as run:
        assert run.engine.diagnostics().rows == 4, 'a window reaching one live slot is a row, masked neighbour or not'
        assert run.oracle == pytest.approx(0.0), 'every take is capped, so raising one costs more than it earns'


#: The same question asked of a width that differs per entity: one unit's
#: window is the masked slot alone, the other's reaches past it.
PER_ENTITY_WINDOW = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int'}},
    'parameters': {'width': {'dims': ['g'], 'dtype': 'int'}, 'usable': {'dims': ['t']}},
    'variables': {
        'level': {'dims': ['g', 't'], 'where': 'usable > 0', 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'held': {'dims': ['g', 't'], 'expression': 'take <= sum_back(level, along=t, window=width)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(sum(take, over=g), over=t) - 1000 * sum(sum(level, over=g), over=t)',
    },
}


def test_a_per_entity_window_reaching_nothing_is_that_entitys_row_alone():
    """Whose window reached something is asked per entity, not per position.

    `narrow` spans one snapshot and `wide` two, and the masked snapshot is the
    same one for both. The row `narrow` holds there reaches nothing and is not
    built; the one `wide` holds reaches the snapshot before it and is. A width
    read only as a coefficient — zeroing the terms outside it but still
    counting them as reached — would keep `narrow`'s row asserting `take <= 0`
    (#1059).
    """
    sources = {
        'g': pd.Index(['narrow', 'wide'], name='g'),
        't': pd.Index([0, 1, 2, 3], name='t'),
        'width': pd.Series([1, 2], index=pd.Index(['narrow', 'wide'], name='g')),
        'usable': pd.Series([1.0, 0.0, 1.0, 1.0], index=pd.Index([0, 1, 2, 3], name='t')),
    }
    with differential(PER_ENTITY_WINDOW, sources, lp=True) as run:
        assert run.engine.diagnostics().rows == 7, "every row but narrow's at the masked snapshot"
        assert run.oracle == pytest.approx(10.0), 'take[narrow, 1] is capped by nothing, and no other take is free'


#: Representative days are separate samples rather than consecutive hours, so a
#: unit started at the last hour of one must not be held on into the next. The
#: start is forced *inside* a day, which is what the seam is read at.
DAY_WINDOW = {
    'description': 'A minimum up time that stops at each representative day.',
    'dimensions': {'t': {'dtype': 'int'}, 'day': {'dtype': 'str'}},
    'relations': {'day_of': {'key': 't', 'values': 'day'}},
    'parameters': {'must_start': {'dims': ['t']}},
    'variables': {
        'started': {'dims': ['t'], 'domain': 'binary'},
        'on': {'dims': ['t'], 'domain': 'binary'},
    },
    'constraints': {
        'starts_when_told': {'dims': ['t'], 'expression': 'started >= must_start'},
        'stays_up_inside_its_day': {'dims': ['t'], 'expression': 'WINDOW <= on'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(on)'},
}

DAYS = ['early'] * 3 + ['late'] * 3


def day_window(window: str, *, starts: list[float], days: list[str | None] = DAYS) -> tuple[dict, dict]:
    """`DAY_WINDOW` carrying *window*, and the sources that force *starts*."""
    spec = deepcopy(DAY_WINDOW)
    spec['constraints']['stays_up_inside_its_day']['expression'] = f'{window} <= on'
    sources = {
        't': pd.Index(range(6), name='t'),
        'day': pd.Index(['early', 'late'], name='day'),
        'day_of': relation('t', 'day', range(6), days),
        'must_start': pd.Series(starts, index=pd.Index(range(6), name='t')),
    }
    return spec, sources


def _on(result) -> list[float]:
    """`on` in coordinate order, which is what a window's reach is about."""
    return result.primal('on').sort('t')['value'].to_list()


def test_a_window_stops_at_the_edge_of_the_group_it_is_partitioned_by():
    """The seam is where the axis stops being one run of positions.

    The start sits at the last hour of the first day, so an unpartitioned
    window of three would hold the unit on there and at the first two hours of
    the *next* day — hours that follow it in the axis and not in time. Inside
    its own group the window reaches back over positions that are its
    neighbours, and only that hour is held.
    """
    spec, sources = day_window('sum_back(started, along=t, window=3, by=day_of, within=day)', starts=[0, 0, 1, 0, 0, 0])
    with differential(spec, sources) as run:
        assert _on(run.result) == [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], (
            'the start holds its own hour on, and no hour of the day after it'
        )
        assert run.oracle == pytest.approx(1.0), 'one hour held on, where the axis-wide window holds three'


def test_a_partitioned_window_wraps_inside_its_own_group():
    """``edge='wrap'`` closes on the group's size, not the axis's.

    A representative day stands for a cycle of its own, so the window at its
    first hour comes round to its last — and stops there. Three hours in the
    group and a width of three is every hour of it, which is what makes the
    contrast with the acyclic run above visible in one number.
    """
    spec, sources = day_window(
        "sum_back(started, along=t, window=3, by=day_of, within=day, edge='wrap')", starts=[0, 0, 1, 0, 0, 0]
    )
    with differential(spec, sources) as run:
        assert _on(run.result) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], (
            "the start reaches every hour of its own day and none of the other day's"
        )
        assert run.oracle == pytest.approx(3.0), 'every hour of the started day, and no hour of the other'


def test_a_window_width_may_be_read_per_group():
    """``window=`` over the dimension the partition groups into.

    No frame carries a column of `day`: what travels with an hour is the
    relation's own value, so the width is read under the group's column and each
    group is reached by its own. The two days differ, which is what a single
    width cannot reproduce.
    """
    spec, sources = day_window(
        'sum_back(started, along=t, window=up, by=day_of, within=day)', starts=[0, 1, 0, 1, 0, 0]
    )
    spec['parameters']['up'] = {'dims': ['day'], 'dtype': 'int'}
    sources['up'] = pd.Series([1, 3], index=pd.Index(['early', 'late'], name='day'))
    with differential(spec, sources) as run:
        assert _on(run.result) == [0.0, 1.0, 0.0, 1.0, 1.0, 1.0], (
            'the early day holds for one hour and the late day for three, each its own width'
        )
        assert run.oracle == pytest.approx(4.0), 'one hour held for the early day and three for the late one'


def test_an_hour_the_relation_places_in_no_day_reaches_nothing():
    """A coordinate in no group is the one way a window loses a row.

    Unpartitioned a window always contains the position it sits at, so every
    row is built. A partitioned one reaches inside a group, and an hour that
    belongs to none reaches nothing at all — not even itself — so its row has
    no terms and is not built, the reading a partial relation gets everywhere
    else.
    """
    spec, sources = day_window(
        'sum_back(started, along=t, window=3, by=day_of, within=day)',
        starts=[0, 0, 1, 0, 0, 0],
        days=[*DAYS[:5], None],
    )
    with differential(spec, sources) as run:
        assert run.engine.diagnostics().rows == 11, (
            'five window rows — one per hour in a day, none for the hour in none — and six starts_when_told rows'
        )


@pytest.mark.parametrize('width', ['0', '1.5', '-2'], ids=['zero', 'fractional', 'negative'])
def test_a_literal_width_is_a_whole_number_of_positions(width: str):
    """Below one position there is no window, and no reading that says so.

    A negative width is the case the sign has to survive to reach: stripped
    before the ``at least 1`` test, ``-2`` reads as ``2``, passes, and arrives
    at lowering as the one node kind it has no case for (mathspec#222).
    """
    spec = up_time_spec(None)
    spec['constraints']['stays_up_its_own_time']['expression'] = f'sum_back(started, along=t, window={width}) <= on'
    with pytest.raises(LanguageError, match='whole number of positions of at least 1'):
        sps.check(spec)


def test_a_window_refuses_a_numeric_edge():
    """A window has no vacated slot to fill — a short window is simply short."""
    spec = up_time_spec(None)
    spec['constraints']['stays_up_its_own_time']['expression'] = (
        'sum_back(started, along=t, window=min_up, edge=0) <= on'
    )
    with pytest.raises(LanguageError, match="takes 'wrap' or nothing"):
        sps.check(spec)


def test_a_window_needs_the_dimension_it_sums_over():
    """Summing back along an axis the operand does not carry says nothing."""
    spec = up_time_spec(None)
    spec['constraints']['stays_up_its_own_time']['expression'] = (
        'sum_back(sum(started, over=t), along=t, window=min_up) <= on'
    )
    with pytest.raises(DimensionError, match='sum_back\\(along=t\\)'):
        sps.check(spec)


def test_a_window_at_the_first_position_is_short_not_empty():
    """The row at the start of the axis survives, holding what it can see.

    The lags that reach past the start contribute a zero rather than an
    absence. Absence propagates in the linopy lane, so an unreachable lag added
    to a reachable one would annihilate the whole row and leave the first
    positions unconstrained — silently, and only near the edge.
    """
    spec = up_time_spec(None)
    data = dict(UP_TIME_DATA)
    data['must_start'] = pd.DataFrame(
        {
            'g': [u for u in UNITS for _ in PERIODS],
            't': PERIODS * len(UNITS),
            'value': [1.0, 0.0, 0.0, 0.0, 0.0] * len(UNITS),
        }
    )
    with differential(spec, data) as run:
        held = run.result.primal('on').filter(pl.col('value') > 0.5)
        assert set(zip(held['g'].to_list(), held['t'].to_list(), strict=True)) == {
            ('slow', 0),
            ('slow', 1),
            ('slow', 2),
            ('fast', 0),
        }, 'a start at the first position is held for its window, and the row that says so exists'


def test_a_window_wider_than_the_axis_counts_each_position_once():
    """A cyclic window is capped at the dimension, not wrapped around twice.

    Nothing else pins the cap: acyclically the lags past the end contribute
    zeros and the answer survives them, so only the cyclic case can tell a
    capped window from one that reads some positions twice.
    """
    spec = up_time_spec('wrap')
    data = dict(UP_TIME_DATA)
    data['min_up'] = pd.Series([len(PERIODS) + 2] * len(UNITS), index=pd.Index(UNITS, name='g'))
    with differential(spec, data) as run:
        assert run.result.objective == pytest.approx(float(len(UNITS) * len(PERIODS))), (
            'a window at least as wide as the axis holds every position on, once'
        )


#: A width with no dimensions: one count the whole axis shares, as OSeMOSYS
#: storage reads its daily time brackets.
SCALAR_WIDTH = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'w': {'dims': [], 'dtype': 'int'}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0}}},
    'constraints': {'c': {'dims': ['t'], 'expression': 'sum_back(x, along=t, window=w) >= 1'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(x)'},
}


@pytest.mark.parametrize(
    'width',
    [pytest.param(2, id='one-number'), pytest.param(pl.DataFrame({'value': [2]}), id='one-row-table')],
)
def test_a_width_with_no_dimensions_is_one_window_for_every_position(width):
    """A dimensionless int width reads as the literal ``window=2`` does (#1891).

    One number was widened to float on the way in, so an ``int`` declaration
    refused it. A one-row table attached, and then the width was joined to the
    rows on its keys, which are none, and polars refused the join.
    """
    with differential(SCALAR_WIDTH, {'t': [1, 2, 3], 'w': width}) as run:
        assert run.oracle == pytest.approx(2.0), (
            'x[1] and x[3] cover every two-wide window, as a literal width of 2 does'
        )
