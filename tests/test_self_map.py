"""A relation whose two columns sit over one dimension — a representative snapshot.

`rep_of: {key: snapshot, values: {rep: snapshot}}` maps each snapshot to the
one that stands for it. The walk lands on the dimension it started from, so the
column it produces and the column it was joined on are over one dimension.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import by_coord, override, raw_of
from tests.differential import RTOL, differential
from tests.oracle import pd

SPEC = """
description: each snapshot priced at the snapshot that represents it

dimensions:
  snapshot: {dtype: int, description: a step of the horizon}

relations:
  rep_of:
    key: snapshot
    values: {rep: snapshot}
    description: the snapshot that stands for this one

parameters:
  price: {dims: [snapshot], description: what output earns in a snapshot}

variables:
  p:
    dims: [snapshot]
    bounds: {lower: 0, upper: 5}
    description: output in a snapshot

constraints:
  capped:
    dims: [snapshot]
    expression: p <= at(price, by=rep_of, over=rep, into=snapshot)
    description: output stays under the price of the snapshot that represents this one

objective:
  sense: maximize
  expression: sum(p * price)
  description: total earnings
"""

SNAPSHOTS = [0, 1, 2]

#: 0 and 1 are both represented by 0; 2 represents itself.
REP_OF = pl.DataFrame({'snapshot': [0, 1, 2], 'rep': [0, 0, 2]})


def _inputs() -> dict[str, object]:
    return {
        'snapshot': pd.Index(SNAPSHOTS, name='snapshot'),
        'rep_of': REP_OF,
        'price': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0]}),
    }


def test_a_pullback_through_a_self_map_reads_the_representative():
    """`at(price, by=rep_of, over=rep, into=snapshot)` is the price at the snapshot that stands for this one (#1652)."""
    with differential(SPEC, _inputs()) as run:
        assert run.oracle == pytest.approx(1 * 1.0 + 1 * 2.0 + 3 * 3.0, rel=RTOL), (
            'each snapshot capped at its representative price, both lanes agreeing'
        )
        built = by_coord(run.result, 'p', 'snapshot')

    assert built[0] == pytest.approx(1.0), 'represented by itself, at price 1'
    assert built[1] == pytest.approx(1.0), "represented by snapshot 0, so capped at 0's price"
    assert built[2] == pytest.approx(3.0), 'represents itself, at price 3'


def test_a_group_through_a_self_map_sums_the_snapshots_it_represents():
    """`sum(p, by=rep_of, over=snapshot, into=rep)` adds each snapshot's output into its representative's row.

    The adjoint of the pullback above (#1652).
    """
    spec = override(
        raw_of(SPEC),
        **{
            'parameters.cap': {'dims': ['snapshot']},
            'constraints.capped': {
                'dims': ['snapshot'],
                'expression': 'sum(p, by=rep_of, over=snapshot, into=rep) <= cap',
            },
        },
    )
    sources = _inputs() | {'cap': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [4.0, 9.0, 6.0]})}

    with differential(spec, sources) as run:
        assert run.oracle == pytest.approx(4 * 2.0 + 5 * 3.0, rel=RTOL), (
            'the pair sharing a representative fills the dearer of the two, and snapshot 2 fills its bound'
        )
        built = by_coord(run.result, 'p', 'snapshot')

    assert built[1] == pytest.approx(4.0), "the whole of representative 0's cap, at the dearer price of the pair"
    assert built[2] == pytest.approx(5.0), 'its own group, held by its bound rather than by its cap'


def test_a_where_reads_a_self_map_at_its_key():
    """A self-map read as a mask compares at its key."""
    spec = override(
        raw_of(SPEC),
        **{
            'variables.p.where': 'rep_of == 0',
            'constraints.capped': {'dims': ['snapshot'], 'expression': 'p <= price'},
        },
    )
    with differential(spec, _inputs()) as run:
        built = by_coord(run.result, 'p', 'snapshot')

    assert sorted(built) == [0, 1], 'the two snapshots the map sends to 0'


def test_a_self_map_supplied_twice_for_one_key_is_refused():
    """Both columns are over one dimension; the key is still the one the file named."""
    sources = _inputs()
    sources['rep_of'] = pl.concat([REP_OF, REP_OF.head(1)])
    with pytest.raises(sps.DataError, match=r'maps 1 key\(s\) more than once: snapshot=0'):
        sps.solve(SPEC, sources)
