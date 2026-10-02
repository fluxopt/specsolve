"""Every shape of relation the language admits, built by the relational lane.

mathspec's relation is a table over any number of dimensions, keyed by any
number of its columns, walked in whichever direction a call names. The
single-valued map — two columns, one of them the key — is one shape of it:

- **a key of several columns** reads the map *under a condition* the row
  carries, so the join carries the rest of the key through;
- **several value columns** land one walk on a product of dimensions;
- **a bare relation** fans a member out to every target it is related to;
- **a partition by a conditioned map** groups a shift, a window or a position
  by the value at the rest of the key, so a neighbour is a neighbour inside
  *this generator's* season;
- **a self-map** produces the dimension it consumes, so the walk's landing
  column needs a name of its own until the consumed one is dropped;
- **two columns over one dimension** name a line's two ends in one table.

Every optimum here is hand-derived, and the written LP file re-solves to it;
the linopy lane refuses these shapes (`test_conditioned_relations.py` holds
what it builds).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import DataError, DimensionError
from tests.conftest import by_coord, solve_written_file

if TYPE_CHECKING:
    from pathlib import Path

#: Solver precision, as the differential harness holds it.
RTOL = 1e-9


def _solved(spec: dict[str, Any], sources: dict[str, Any], tmp_path: Path) -> tuple[float, dict[str, Any]]:
    """The objective the relational lane reaches, checked against the LP file it writes, and the primals asked for."""
    with sps.build(spec, sources) as model:
        model.write(tmp_path / 'model.lp')
        result = model.solve()
        assert result.is_ok, f'the relational lane reached no solution: {result.status}'
        objective = result.objective
        primals = {name: by_coord(result, name, *spec['variables'][name]['dims']) for name in spec['variables']}
    assert solve_written_file(tmp_path / 'model.lp') == pytest.approx(objective, rel=RTOL), (
        'the written LP file re-solves to a different objective than the lane reached'
    )
    return objective, primals


# ---------------------------------------------------------------------------
# several value columns — one walk onto a product of dimensions
# ---------------------------------------------------------------------------

GENERATORS = ['g1', 'g2', 'g3', 'g4']

#: `g3` and `g4` share (b, wind), so a limit binds across two generators, and
#: nothing sits at (b, sun), which is the empty combination.
GEN_BT = pl.DataFrame(
    {'generator': GENERATORS, 'bus': ['a', 'a', 'b', 'b'], 'technology': ['wind', 'sun', 'wind', 'wind']}
)


def _two_value_spec(constraint: str, demand: float, headroom: bool = False) -> dict[str, Any]:
    spec: dict[str, Any] = {
        'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}, 'technology': {'dtype': 'str'}},
        'relations': {'gen_bt': {'key': 'generator', 'values': ['bus', 'technology']}},
        'parameters': {
            'cost': {'dims': ['generator']},
            'limit': {'dims': ['bus', 'technology']},
            'bonus': {'dims': ['generator']},
            'demand': {'dims': []},
        },
        'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 100}}},
        'constraints': {
            'capped': {'dims': ['bus', 'technology'], 'expression': constraint},
            'meet_demand': {'dims': [], 'expression': 'sum(p, over=generator) >= demand'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(p * cost, over=generator)'},
    }
    if headroom:
        spec['variables']['headroom'] = {'dims': ['bus', 'technology'], 'bounds': {'lower': 0, 'upper': 100}}
        spec['objective']['expression'] += ' - sum(headroom)'
    return spec


def _two_value_sources(demand: float) -> dict[str, Any]:
    return {
        'generator': GENERATORS,
        'bus': ['a', 'b'],
        'technology': ['wind', 'sun'],
        'gen_bt': GEN_BT,
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 3.0, 4.0]}),
        'limit': pl.DataFrame(
            {'bus': ['a', 'a', 'b', 'b'], 'technology': ['wind', 'sun', 'wind', 'sun'], 'value': [10.0, 5.0, 7.0, 1.0]}
        ),
        'bonus': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 1.0, 1.0, 1.0]}),
        'demand': demand,
    }


def test_a_sum_through_one_relation_lands_on_a_product_of_dimensions(tmp_path):
    """`sum(p, by=gen_bt, over=generator, into=[bus, technology])` is one walk onto both value columns.

    (a, wind) caps `g1` at 10 and (a, sun) caps `g2` at 5, which is 15 of the
    20 demanded. The rest comes from (b, wind), shared by `g3` and `g4`, so
    the cheaper `g3` takes it: 10 + 10 + 15.
    """
    spec = _two_value_spec('sum(p, by=gen_bt, over=generator, into=[bus, technology]) <= limit', demand=20)
    objective, primals = _solved(spec, _two_value_sources(20), tmp_path)
    assert objective == pytest.approx(35.0, rel=RTOL), '10 at cost 1, 5 at cost 2, and 5 of the shared pair at cost 3'
    assert primals['p'] == pytest.approx({'g1': 10.0, 'g2': 5.0, 'g3': 5.0, 'g4': 0.0}), (
        'each generator capped by the limit at the pair its row names, and the shared pair by one limit'
    )


def test_a_pullback_reads_a_two_column_slot_at_each_generator(tmp_path):
    """`at(limit, by=gen_bt, over=[bus, technology], into=generator)` is the adjoint: each generator's own limit.

    Bounded one by one rather than as a group, `g3` and `g4` each take (b,
    wind)'s 7, so 24 is reachable: 10 + 5 + 7 + 2.
    """
    spec = _two_value_spec('p <= at(limit, by=gen_bt, over=[bus, technology], into=generator)', demand=24)
    spec['constraints']['capped']['dims'] = ['generator']
    objective, primals = _solved(spec, _two_value_sources(24), tmp_path)
    assert objective == pytest.approx(10 * 1.0 + 5 * 2.0 + 7 * 3.0 + 2 * 4.0, rel=RTOL), (
        'each generator fills its own limit, cheapest first'
    )
    assert primals['p']['g4'] == pytest.approx(2.0), 'the dearest generator serves what the others cannot'


def test_a_combination_no_member_lands_on_is_an_empty_sum_on_a_constant_side(tmp_path):
    """(b, sun) has no generator, so a constant grouped there is a zero rather than a hole.

    `headroom` is paid for at every pair, and at (b, sun) it is bounded by the
    limit less an empty sum of bonuses — 1. A hole there would refuse the
    model for a constant side with no value where the row is built.
    """
    spec = _two_value_spec(
        'headroom <= limit - sum(bonus, by=gen_bt, over=generator, into=[bus, technology])', demand=0, headroom=True
    )
    objective, primals = _solved(spec, _two_value_sources(0), tmp_path)
    assert primals['headroom'][('b', 'sun')] == pytest.approx(1.0), 'the empty combination is worth its whole limit'
    assert objective == pytest.approx(-(9.0 + 4.0 + 5.0 + 1.0), rel=RTOL), (
        'every other pair loses one bonus per generator sitting on it'
    )


def _masked_sum_spec(expression: str) -> dict:
    return {
        'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
        'relations': {'gen_bus': {'key': 'generator', 'values': 'bus'}},
        'parameters': {'load': {'dims': ['bus']}, 'cap': {'dims': ['bus']}},
        'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {'capped': {'dims': ['bus'], 'expression': expression}},
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }


MASKED_SUM_SOURCES = {
    'generator': ['g1', 'g2'],
    'bus': ['a', 'b'],
    'gen_bus': pl.DataFrame({'generator': ['g1', 'g2'], 'bus': ['a', 'b']}),
    'load': pl.DataFrame({'bus': ['a', 'b'], 'value': [1.0, 2.0]}),
    'cap': pl.DataFrame({'bus': ['a', 'b'], 'value': [4.0, 4.0]}),
}


def test_a_walk_onto_a_dimension_the_operand_carries_is_refused_and_names_the_rewrite():
    """A walk brings the dimension it lands on, so the operand may not already carry it.

    Inside the operator, `load[bus]` and a walk onto `bus` read as a product
    over every (generator, bus) pair masked back down to the pairs the map
    agrees with — which is the same answer as multiplying outside, reached by a
    spelling that hides what the call adds.
    """
    spec = _masked_sum_spec('sum(load * p, by=gen_bus, over=generator, into=bus) <= cap')
    with pytest.raises(DimensionError, match=r"sum\(by=gen_bus\) lands on \['bus'\], which the expression already"):
        sps.check(spec)


def test_the_factor_over_the_landed_dimension_multiplies_outside(tmp_path):
    """The rewrite the refusal names, and the answer it reaches.

    Bus a holds `load_a * p1` and bus b holds `load_b * p2`, under caps of 4 —
    so 4 and 2.
    """
    spec = _masked_sum_spec('load * sum(p, by=gen_bus, over=generator, into=bus) <= cap')
    objective, primals = _solved(spec, MASKED_SUM_SOURCES, tmp_path)
    assert primals['p'] == pytest.approx({'g1': 4.0, 'g2': 2.0}), "each generator weighed by its own bus's load"
    assert objective == pytest.approx(6.0, rel=RTOL), '4 under a load of 1, and 2 under a load of 2'


# ---------------------------------------------------------------------------
# a bare relation — many-to-many
# ---------------------------------------------------------------------------

#: `g1` reaches both buses, `g2` one, and nothing reaches `c`.
CONNECTION = pl.DataFrame({'generator': ['g1', 'g1', 'g2'], 'bus': ['a', 'b', 'b']})


def _bare_spec() -> dict[str, Any]:
    return {
        'dimensions': {'generator': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
        'relations': {'connection': {'key': ['generator', 'bus']}},
        'parameters': {'limit': {'dims': ['bus']}, 'w': {'dims': ['generator']}},
        'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 10}}},
        'constraints': {
            'capped': {'dims': ['bus'], 'expression': 'sum(p, by=connection, over=generator, into=bus) <= limit'}
        },
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }


def _bare_sources() -> dict[str, Any]:
    return {
        'generator': ['g1', 'g2'],
        'bus': ['a', 'b', 'c'],
        'connection': CONNECTION,
        'limit': pl.DataFrame({'bus': ['a', 'b', 'c'], 'value': [5.0, 5.0, 5.0]}),
        'w': pl.DataFrame({'generator': ['g1', 'g2'], 'value': [1.0, 2.0]}),
    }


def test_a_sum_through_a_bare_relation_counts_a_member_in_every_group_it_is_related_to(tmp_path):
    """`g1` is connected to both buses, so its term belongs in two sums.

    Bus `a` holds `p1` under 5 and bus `b` holds `p1 + p2` under 5, so the
    two together reach 5. A lane landing `g1` once would let `p2` run to 5 on
    its own and reach 10.
    """
    objective, primals = _solved(_bare_spec(), _bare_sources(), tmp_path)
    assert objective == pytest.approx(5.0, rel=RTOL), 'bus b holds both generators under one limit'
    assert primals['p']['g1'] + primals['p']['g2'] == pytest.approx(5.0), 'bus b holds both'


def test_a_constant_summed_through_a_bare_relation_reads_zero_where_nothing_reaches(tmp_path):
    """The weights land on every bus a member reaches, and an unreached bus gets the empty sum."""
    spec = _bare_spec()
    spec['variables']['x'] = {'dims': ['bus'], 'bounds': {'lower': 0, 'upper': 10}}
    spec['constraints']['under'] = {
        'dims': ['bus'],
        'expression': 'x <= sum(w, by=connection, over=generator, into=bus)',
    }
    spec['objective'] = {'sense': 'maximize', 'expression': 'sum(x)'}
    objective, primals = _solved(spec, _bare_sources(), tmp_path)
    assert primals['x'] == pytest.approx({'a': 1.0, 'b': 3.0, 'c': 0.0}), (
        'a reads g1 alone, b reads both, and c — reached by nothing — reads the empty sum'
    )
    assert objective == pytest.approx(4.0, rel=RTOL), '1 at a, 3 at b, and the empty sum at c'


def test_a_bare_where_tests_that_a_row_exists(tmp_path):
    """`where: connection` over `[generator, bus]` keeps the pairs the table relates and no other."""
    spec = _bare_spec()
    spec['variables'] = {'x': {'dims': ['generator', 'bus'], 'where': 'connection', 'bounds': {'lower': 0, 'upper': 1}}}
    spec['constraints'] = {}
    spec['objective'] = {'sense': 'maximize', 'expression': 'sum(x)'}
    objective, primals = _solved(spec, _bare_sources(), tmp_path)
    assert sorted(primals['x']) == [('g1', 'a'), ('g1', 'b'), ('g2', 'b')], 'one column per related pair'
    assert objective == pytest.approx(3.0, rel=RTOL), 'one unit per related pair'


#: A start covers its own snapshot and the next, and the last start covers itself alone.
COVER = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)]


def test_both_lanes_carry_a_member_through_a_bare_relation_whose_two_roles_range_over_one_dimension():
    """`start` and `covered` are both snapshots, so the relation is keyed by one dimension twice.

    One start is made, and the window it opens is priced at the snapshots it
    covers: from 1 it covers 1 and 2 at a cost of 1 each, the cheapest window.
    """
    from tests.differential import differential
    from tests.oracle import pd

    spec = {
        'dimensions': {'snapshot': {'dtype': 'int'}},
        'relations': {'cover': {'key': {'start': 'snapshot', 'covered': 'snapshot'}}},
        'parameters': {'cost': {'dims': ['snapshot']}},
        'variables': {
            's': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 1}},
            'm': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 1}},
        },
        'constraints': {
            'window': {'dims': ['snapshot'], 'expression': 'm == sum(s, by=cover, over=start, into=covered)'},
            'once': {'dims': [], 'expression': 'sum(s) == 1'},
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(m * cost)'},
    }
    sources = {
        'snapshot': pd.RangeIndex(4, name='snapshot'),
        'cover': pd.DataFrame(COVER, columns=['start', 'covered']),
        'cost': pd.Series([5.0, 1.0, 1.0, 5.0], index=pd.RangeIndex(4, name='snapshot')),
    }
    with differential(spec, sources) as run:
        assert float(run.result.objective) == pytest.approx(2.0, rel=RTOL), 'the window from 1 costs 1 + 1'


def test_a_bare_relation_holding_a_pair_twice_is_refused():
    """A table has each coordinate at most once, and a bare relation's coordinate is the whole row."""
    sources = _bare_sources() | {'connection': pl.concat([CONNECTION, CONNECTION.head(1)])}
    with pytest.raises(DataError, match=r"relates 1 tuple\(s\) more than once: generator='g1', bus='a'"):
        sps.solve(_bare_spec(), sources)


# ---------------------------------------------------------------------------
# a partition by a map keyed on more than the dimension it walks
# ---------------------------------------------------------------------------

PERIODS = [1, 2, 3, 4]

#: Each generator's own seasons: `g1` splits the year in half, `g2` after one period.
SEASON_OF = pl.DataFrame(
    {
        'generator': ['g1'] * 4 + ['g2'] * 4,
        'period': PERIODS * 2,
        'season': ['summer', 'summer', 'winter', 'winter', 'summer', 'winter', 'winter', 'winter'],
    }
)


def _conditioned_partition_spec(constraint: dict[str, Any]) -> dict[str, Any]:
    return {
        'dimensions': {'generator': {'dtype': 'str'}, 'period': {'dtype': 'int'}, 'season': {'dtype': 'str'}},
        'relations': {'season_of': {'key': ['generator', 'period'], 'values': 'season'}},
        'parameters': {},
        'variables': {'p': {'dims': ['generator', 'period'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': {'rule': constraint},
        'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
    }


def _conditioned_partition_sources() -> dict[str, Any]:
    return {'generator': ['g1', 'g2'], 'period': PERIODS, 'season': ['summer', 'winter'], 'season_of': SEASON_OF}


def test_a_shift_grouped_by_a_conditioned_map_stays_inside_each_generators_own_season(tmp_path):
    """The neighbour is the period before *in this generator's* season, and each season starts from the edge.

    `p <= shift(p) + 1` with `edge=0` lets a season ramp 1, 2, 3… from its
    first period. `g1` has two seasons of two periods (1 + 2, twice) and `g2`
    one of one and one of three (1, then 1 + 2 + 3): 6 + 7. Walking the whole
    axis would reach 10 per generator.
    """
    spec = _conditioned_partition_spec(
        {
            'dims': ['generator', 'period'],
            'expression': 'p <= shift(p, along=period, offset=1, edge=0, by=season_of, within=season) + 1',
        }
    )
    objective, primals = _solved(spec, _conditioned_partition_sources(), tmp_path)
    assert objective == pytest.approx(13.0, rel=RTOL), "g1's 1 + 2 twice, and g2's 1 beside 1 + 2 + 3"
    assert primals['p'][('g2', 4)] == pytest.approx(3.0), "the third period of g2's winter ramps to 3"
    assert primals['p'][('g1', 3)] == pytest.approx(1.0), "g1's winter restarts from its edge"


def test_a_window_grouped_by_a_conditioned_map_stops_at_each_generators_own_season_edge(tmp_path):
    """A window of two inside each (generator, season) group.

    `g1` fits 3 per season, 6 in all; `g2` fits 3 in its one-period summer and
    3 + 0 + 3 in its three-period winter: 9. A window across the whole axis
    reaches 6 per generator.
    """
    spec = _conditioned_partition_spec(
        {
            'dims': ['generator', 'period'],
            'expression': 'sum_back(p, along=period, window=2, by=season_of, within=season) <= 3',
        }
    )
    objective, _primals = _solved(spec, _conditioned_partition_sources(), tmp_path)
    assert objective == pytest.approx(15.0, rel=RTOL), "g1's 3 per season, and g2's 3 beside 3 + 0 + 3"


def test_a_position_grouped_by_a_conditioned_map_marks_each_generators_own_first_period(tmp_path):
    """`position(period, by=season_of, within=season) == 0` is the first period of each (generator, season) group.

    Four of the eight coordinates are a group's first — `g1` at 1 and 3, `g2`
    at 1 and 2 — and the rule pins those to zero.
    """
    spec = _conditioned_partition_spec(
        {
            'dims': ['generator', 'period'],
            'where': 'position(period, by=season_of, within=season) == 0',
            'expression': 'p <= 0',
        }
    )
    objective, primals = _solved(spec, _conditioned_partition_sources(), tmp_path)
    pinned = sorted(coordinate for coordinate, value in primals['p'].items() if value == 0.0)
    assert pinned == [('g1', 1), ('g1', 3), ('g2', 1), ('g2', 2)], 'the first period of each generator-season'
    assert objective == pytest.approx(20.0, rel=RTOL), 'the four unpinned coordinates at their bound of 5'


def test_a_pair_the_conditioned_partition_leaves_out_reaches_nothing(tmp_path):
    """`g2` in period 4 is in no season, so it has no neighbour and its row is not built.

    Unrowed, it runs to its bound; its group's third period is gone, so
    `g2`'s winter is two periods, 1 + 2, beside its summer's 1. With `g1`'s 6
    and the free 5, 15.
    """
    sources = _conditioned_partition_sources()
    sources['season_of'] = SEASON_OF.filter(~((pl.col('generator') == 'g2') & (pl.col('period') == 4)))
    spec = _conditioned_partition_spec(
        {
            'dims': ['generator', 'period'],
            'expression': 'p <= shift(p, along=period, offset=1, edge=0, by=season_of, within=season) + 1',
        }
    )
    objective, primals = _solved(spec, sources, tmp_path)
    assert primals['p'][('g2', 4)] == pytest.approx(5.0), 'in no group, so no rule reaches it'
    assert objective == pytest.approx(6.0 + (1.0 + 3.0) + 5.0, rel=RTOL), (
        "g1 unchanged, g2's shortened winter, and the ungrouped pair at its bound"
    )


# ---------------------------------------------------------------------------
# a self-map — a dimension related to itself; the walks are test_self_map.py's, this is the partition
# ---------------------------------------------------------------------------

SNAPSHOTS = [0, 1, 2]

#: Snapshot 0 stands for itself and for 1; snapshot 2 stands for itself.
REP_OF = pl.DataFrame({'snapshot': SNAPSHOTS, 'rep': [0, 0, 2]})


def _self_map_spec(constraints: dict[str, Any]) -> dict[str, Any]:
    return {
        'dimensions': {'snapshot': {'dtype': 'int'}},
        'relations': {'rep_of': {'key': 'snapshot', 'values': {'rep': 'snapshot'}}},
        'parameters': {'price': {'dims': ['snapshot']}},
        'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0, 'upper': 5}}},
        'constraints': constraints,
        'objective': {'sense': 'maximize', 'expression': 'sum(p * price)'},
    }


def _self_map_sources() -> dict[str, Any]:
    return {
        'snapshot': SNAPSHOTS,
        'rep_of': REP_OF,
        'price': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0]}),
    }


def test_a_shift_partitioned_by_a_self_map_walks_inside_each_representatives_group(tmp_path):
    """The snapshots one representative stands for are one group, and the shift stays inside it.

    `p <= shift(p) + 1` from each group's edge: 1 then 2 for the pair, 1 for
    the singleton: 1 + 4 + 3.
    """
    spec = _self_map_spec(
        {
            'ramp': {
                'dims': ['snapshot'],
                'expression': 'p <= shift(p, along=snapshot, offset=1, edge=0, by=rep_of, within=rep) + 1',
            }
        }
    )
    objective, primals = _solved(spec, _self_map_sources(), tmp_path)
    assert objective == pytest.approx(8.0, rel=RTOL), '1 + 4 + 3 under the ramp inside each group'
    assert primals['p'] == pytest.approx({0: 1.0, 1: 2.0, 2: 1.0}), 'the pair ramps 1 then 2, the singleton stays at 1'


# ---------------------------------------------------------------------------
# two columns over one dimension — a line's two ends in one table
# ---------------------------------------------------------------------------

#: A chain a → b → c, and a self-loop at c that the where excludes.
ENDS = pl.DataFrame({'line': ['l1', 'l2', 'l3'], 'bus0': ['a', 'b', 'c'], 'bus1': ['b', 'c', 'c']})


def _ends_spec() -> dict[str, Any]:
    return {
        'dimensions': {'line': {'dtype': 'str'}, 'bus': {'dtype': 'str'}},
        'relations': {'ends': {'key': 'line', 'values': {'bus0': 'bus', 'bus1': 'bus'}}},
        'parameters': {'load': {'dims': ['bus']}, 'cost': {'dims': ['bus']}},
        'variables': {
            'f': {'dims': ['line'], 'where': 'ends.bus0 != ends.bus1', 'bounds': {'lower': 0, 'upper': 10}},
            'gen': {'dims': ['bus'], 'bounds': {'lower': 0, 'upper': 10}},
        },
        'constraints': {
            'balance': {
                'dims': ['bus'],
                'expression': (
                    'gen + sum(f, by=ends, over=line, into=bus1) - sum(f, by=ends, over=line, into=bus0) == load'
                ),
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(gen * cost)'},
    }


def _ends_sources() -> dict[str, Any]:
    return {
        'line': ['l1', 'l2', 'l3'],
        'bus': ['a', 'b', 'c'],
        'ends': ENDS,
        'load': pl.DataFrame({'bus': ['a', 'b', 'c'], 'value': [0.0, 0.0, 5.0]}),
        'cost': pl.DataFrame({'bus': ['a', 'b', 'c'], 'value': [1.0, 2.0, 3.0]}),
    }


def test_one_table_walked_to_each_of_its_two_ends_is_the_nodal_balance(tmp_path):
    """Flow enters at `bus1` and leaves at `bus0`, so the cheap generator at `a` serves the load at `c`.

    Walked the wrong way round the flow could only run c → a, and the load
    would be served at `c` for 15 rather than at `a` for 5.
    """
    objective, primals = _solved(_ends_spec(), _ends_sources(), tmp_path)
    assert objective == pytest.approx(5.0, rel=RTOL), 'the load at c served from a at cost 1'
    assert primals['gen'] == pytest.approx({'a': 5.0, 'b': 0.0, 'c': 0.0}), 'the cheapest bus generates it all'
    assert primals['f'] == pytest.approx({'l1': 5.0, 'l2': 5.0}), 'along the chain, and the self-loop has no column'


def test_a_where_compares_two_columns_of_one_relation_row_by_row():
    """`ends.bus0 != ends.bus1` excludes the self-loop and nothing else."""
    with sps.solve(_ends_spec(), _ends_sources()) as result:
        assert sorted(result.primal('f')['line'].to_list()) == ['l1', 'l2'], 'the self-loop `l3` has no column'
