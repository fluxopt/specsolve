"""What the linopy lane builds of the language's sets and properties, and what it refuses.

The linopy lane builds a property, of a dimension or of a set, and refuses the
other shapes naming the relational lane.

A property of a set — a generator's zone that changes by period — is read
under a condition the row already carries: `sum(by=)` groups per condition,
`at(by=)` reads back per condition, and a `where:` tests the value at both
member dimensions. A property of a set leaves out a pair rather than a label.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DataError
from tests.conftest import by_coord, override, raw_of
from tests.differential import RTOL, differential
from tests.linopy_lane.loader import OracleCannotBuildError
from tests.oracle import pd, specsolve_linopy

SPEC = """
description: a zonal limit where a generator's zone changes by period

dimensions:
  generator: {dtype: str, description: a generating unit}
  period: {dtype: int, description: a planning period, ordered: true}
  zone: {dtype: str, description: a bidding zone}

sets:
  generator_period: {dims: [generator, period], description: each generator in each period}

properties:
  generator_period:
    zone: {into: zone, missing: absent, description: the zone a generator bids into in one period}

parameters:
  cost: {dims: [generator], description: marginal cost of a unit of output}
  zone_cap: {dims: [period, zone], description: how much one zone may run in one period}
  demand: {dims: [period], description: output the system must reach in one period}

variables:
  p:
    dims: [generator, period]
    bounds: {lower: 0}
    description: output of a generator in a period

constraints:
  zonal:
    dims: [period, zone]
    expression: sum(p, over=generator, by=generator_period[zone]) <= zone_cap
    description: a zone's output stays under its cap, period by period
  meet_demand:
    dims: [period]
    expression: sum(p, over=generator) >= demand
    description: each period meets its demand

objective:
  sense: minimize
  expression: sum(p * cost)
  description: total marginal cost
"""

GENERATORS = ['g1', 'g2']
PERIODS = [1, 2]

#: `g1` bids into the capped zone in period 1 and joins `g2` in period 2.
ZONE = pl.DataFrame(
    {
        'generator': ['g1', 'g1', 'g2', 'g2'],
        'period': [1, 2, 1, 2],
        'zone': ['n', 's', 's', 's'],
    }
)


def _inputs() -> dict[str, object]:
    return {
        'generator': pd.Index(GENERATORS, name='generator'),
        'period': pd.Index(PERIODS, name='period'),
        'zone': pd.Index(['n', 's'], name='zone'),
        'generator_period': ZONE.select('generator', 'period'),
        'generator_period.zone': ZONE,
        'cost': pd.Series([1.0, 2.0], index=pd.Index(GENERATORS, name='generator')),
        'zone_cap': pl.DataFrame(
            {'period': [1, 1, 2, 2], 'zone': ['n', 's', 'n', 's'], 'value': [4.0, 10.0, 10.0, 6.0]}
        ),
        'demand': pd.Series([10.0, 5.0], index=pd.Index(PERIODS, name='period')),
    }


# ---------------------------------------------------------------------------
# both lanes
# ---------------------------------------------------------------------------


def test_a_property_of_a_set_groups_per_condition():
    """The optimum, hand-derived, and the same on both lanes and the LP file.

    Period 1 caps `g1` at 4 in zone n, so the 10 it owes comes as 4 at cost 1
    and 6 at cost 2. Period 2 puts both generators in zone s under a cap of 6,
    which the 5 it owes fits inside, so the cheaper `g1` takes all of it.
    """
    with differential(SPEC, _inputs(), lp=True) as run:
        assert run.oracle == pytest.approx(4 * 1.0 + 6 * 2.0 + 5 * 1.0, rel=RTOL), (
            'the hand-derived optimum, reached through the same property on both lanes and in the LP file'
        )
        built = by_coord(run.result, 'p', 'generator', 'period')

    assert built[('g1', 1)] == pytest.approx(4.0), "zone n's cap binds in period 1"
    assert built[('g2', 1)] == pytest.approx(6.0), 'the rest of period 1 comes from the dearer generator'
    assert built[('g1', 2)] == pytest.approx(5.0), 'period 2 puts both in one zone and the cheaper one serves it'
    assert built[('g2', 2)] == pytest.approx(0.0), 'priced out where it shares the zone'


def test_a_pair_the_property_leaves_out_is_in_no_group():
    """A *pair* without a zone is not a label without one: only (g1, 1) goes missing.

    `g1` is in no zone in period 1, so no cap reaches it there and it serves
    the whole 10 at cost 1 — while period 2, where it has a zone, is untouched.
    """
    sources = _inputs()
    sources['generator_period.zone'] = ZONE.filter(~((pl.col('generator') == 'g1') & (pl.col('period') == 1)))

    with differential(SPEC, sources) as run:
        assert run.oracle == pytest.approx(10 * 1.0 + 5 * 1.0, rel=RTOL), (
            'the pair without a zone costs the cheap generator nothing, and its other period is unchanged'
        )
        built = by_coord(run.result, 'p', 'generator', 'period')

    assert built[('g1', 1)] == pytest.approx(10.0), 'in no zone in period 1, so no cap binds it'
    assert built[('g1', 2)] == pytest.approx(5.0), 'still in a zone in period 2, where that zone caps the pair'


def test_a_pullback_reads_a_property_of_a_set_at_the_row_it_stands_on():
    """`at(by=)` reads the cap of this generator's zone in this period: (g2, 1) reads zone s."""
    spec = override(
        raw_of(SPEC),
        **{
            'constraints.zonal': {
                'dims': ['generator', 'period'],
                'expression': 'p <= at(zone_cap, by=generator_period[zone])',
            }
        },
    )
    with differential(spec, _inputs()) as run:
        assert run.oracle == pytest.approx(4 * 1.0 + 6 * 2.0 + 5 * 1.0, rel=RTOL), (
            'each generator bounded by the cap of the zone its own pair reads'
        )
        built = by_coord(run.result, 'p', 'generator', 'period')

    assert built[('g1', 1)] == pytest.approx(4.0), "read through its own pair, g1 takes zone n's cap"
    assert built[('g2', 1)] == pytest.approx(6.0), 'its pair reads zone s, whose cap is the larger one'


def test_a_where_reads_a_property_of_a_set_at_both_member_dimensions():
    """`generator_period[zone] == 'n'` selects the pairs, not the generators.

    Only (g1, 1) sits in zone n.
    """
    spec = override(raw_of(SPEC), **{'variables.p.where': "generator_period[zone] == 'n'"})
    sources = _inputs()
    sources['demand'] = pd.Series([4.0, 0.0], index=pd.Index(PERIODS, name='period'))

    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(4.0, rel=RTOL), 'the one surviving pair serves its own period'
        built = by_coord(run.result, 'p', 'generator', 'period')

    assert sorted(built) == [('g1', 1)], 'the one pair the property puts in zone n'


# ---------------------------------------------------------------------------
# what the linopy lane refuses, and the relational lane builds
# ---------------------------------------------------------------------------


def _shaped(declared: dict, expression: str, dims: list[str]) -> dict:
    """A model whose one constraint walks the *declared* sets and properties, both ends named."""
    dimensions = {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'period': {'dtype': 'int', 'ordered': True}}
    members = [d for entry in declared.get('sets', {}).values() for d in entry['dims']]
    into = [
        p if isinstance(p, str) else p['into'] for ps in declared.get('properties', {}).values() for p in ps.values()
    ]
    used = {'generator', 'period', *dims, *members, *into}
    return {
        'dimensions': {name: dtype for name, dtype in dimensions.items() if name in used},
        **declared,
        'parameters': {'load': {'dims': dims}},
        'variables': {'p': {'dims': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'bal': {'dims': dims, 'expression': expression}},
        'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
    }


@pytest.mark.parametrize(
    ('declared', 'expression', 'dims', 'match'),
    [
        pytest.param(
            {
                'sets': {'generator_period': {'dims': ['generator', 'period']}},
                'properties': {'generator_period': {'bus': 'bus'}},
            },
            'shift(p, along=period, offset=1, edge=0, within=generator_period[bus]) >= load',
            ['generator', 'period'],
            r"a partition by 'generator_period\[bus\]' reads a subject whose members are \['generator', 'period'\]",
            id='a-partition-by-a-property-of-a-set',
        ),
    ],
)
def test_the_linopy_lane_refuses_a_shape_it_does_not_build_and_names_the_lane_that_does(
    declared: dict, expression: str, dims: list[str], match: str
) -> None:
    """An `OracleCannotBuildError` at the lane's door, before any data is read, naming the relational lane."""
    spec = _shaped(declared, expression, dims)
    sps.check(spec)
    with pytest.raises(OracleCannotBuildError, match=match) as caught:
        specsolve_linopy.build(spec, {})
    assert 'sps.build()/sps.solve()' in str(caught.value), 'the refusal names the route around it'


# ---------------------------------------------------------------------------
# the door
# ---------------------------------------------------------------------------


def test_a_pair_supplied_twice_is_refused():
    """The row is the pair, so a second zone for one pair is what "single-valued" refuses."""
    sources = _inputs()
    sources['generator_period.zone'] = pl.concat([ZONE, ZONE.head(1)])
    with pytest.raises(
        DataError,
        match=r"property 'zone' of 'generator_period' has 1 row\(s\) of 'generator_period' more than once: "
        r'generator=g1, period=1',
    ):
        sps.solve(SPEC, sources)


def test_a_property_of_a_set_short_of_a_member_column_is_refused():
    """Every declared column arrives, and the message names the ones it wants."""
    sources = _inputs()
    sources['generator_period.zone'] = ZONE.drop('period')
    with pytest.raises(DataError, match=r"must carry the columns \['generator', 'period', 'zone'\]"):
        sps.solve(SPEC, sources)


def test_a_member_column_naming_no_label_is_refused():
    """Both members of the set are checked against their own dimension's labels."""
    sources = _inputs()
    strayed = (pl.col('generator') == 'g1') & (pl.col('period') == 1)
    sources['generator_period'] = ZONE.select('generator', 'period').with_columns(
        pl.when(strayed).then(9).otherwise(pl.col('period')).alias('period')
    )
    with pytest.raises(
        DataError, match=r"set 'generator_period' has value\(s\) in 'period' that are not 'period' labels: 9"
    ):
        sps.solve(SPEC, sources)
