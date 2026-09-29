"""``reserves``: every many-to-many shape at once, each proven load-bearing.

``examples/reserves.yaml`` holds both many-to-many idioms: a pair set reified as
a dimension with leg relations (lines bus-to-bus, offers as (generator, market,
tranche) triples), and weighted membership as an incidence parameter
(overlapping reserve zones). Each shape gets the one data mutation that must
move the optimum.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from tests.conftest import EXAMPLES_DIR, port_sources
from tests.differential import RTOL, differential

RESERVES_YAML = EXAMPLES_DIR / 'reserves.yaml'

#: Hand-derived, and what both lanes and the reference script reach. Energy:
#: b2's surplus exports over l1 (pinned at 15 by ``bus_cap``, not its own 20)
#: and l2 (its own 8), so g3 runs 40 local + 23 export = 63 and g1 covers the
#: rest of b1, 47 — cost 785. Reserves: m1's 55 takes both parallel g1 offers
#: at their 25 caps (o2 first at cost 1, o1 at 2) plus 5 of o3, whose seat on
#: g2 is also what closes zone z2 at exactly 25; m2's 20 is o4 at its cap —
#: cost 130.
OPTIMUM = 915.0


def test_both_lanes_and_the_lp_file_reach_the_hand_derived_optimum():
    with differential(RESERVES_YAML, port_sources('reserves'), lp=True) as run:
        assert run.oracle == pytest.approx(OPTIMUM, rel=RTOL), 'the linopy lane disagrees with the hand derivation'


def _drop_line(sources: dict, line: str) -> dict:
    for key in ('line', 'cap', 'neg_cap', 'line_from', 'line_to'):
        sources[key] = sources[key].filter(pl.col('line') != line)
    return sources


def _drop_offer(sources: dict, offer: str) -> dict:
    for key in ('offer', 'offer_cost', 'gen_of', 'market_of', 'tranche_of'):
        sources[key] = sources[key].filter(pl.col('offer') != offer)
    return sources


def _repoint_dangling(sources: dict) -> dict:
    sources['line_to'] = pl.concat([sources['line_to'], pl.DataFrame({'line': ['l4'], 'bus': ['b1']})])
    return sources


def _uncap_exporter(sources: dict) -> dict:
    sources['bus_cap'] = sources['bus_cap'].with_columns(
        pl.when(pl.col('bus') == 'b2').then(pl.lit(100.0)).otherwise(pl.col('value')).alias('value')
    )
    return sources


def _zero_zone_share(sources: dict) -> dict:
    sources['zone_share'] = sources['zone_share'].filter((pl.col('generator') != 'g2') | (pl.col('zone') != 'z2'))
    return sources


@pytest.mark.parametrize(
    ('mutate', 'direction'),
    [
        pytest.param(lambda s: _drop_line(s, 'l2'), 'dearer', id='a-parallel-edge-carries-real-flow'),
        pytest.param(_uncap_exporter, 'cheaper', id='the-pullback-caps-the-exporting-bus'),
        pytest.param(_repoint_dangling, 'cheaper', id='a-dangling-leg-carries-nothing-until-pointed'),
        pytest.param(lambda s: _drop_offer(s, 'o2'), 'dearer', id='a-duplicate-pair-is-real-capacity'),
        pytest.param(_zero_zone_share, 'dearer', id='an-incidence-weight-binds-the-zone'),
    ],
)
def test_each_many_to_many_shape_moves_the_optimum(mutate, direction):
    sources = mutate(dict(port_sources('reserves')))
    with sps.solve(RESERVES_YAML, sources) as run:
        assert run.is_ok, 'the mutation must re-price the model, not break it'
        moved = run.objective > OPTIMUM if direction == 'dearer' else run.objective < OPTIMUM
        assert moved, (
            f'objective stayed at {run.objective} — the shape this mutation removes was '
            f'decoration, and the model proves nothing about it'
        )


def test_the_instance_actually_holds_every_shape():
    """The mutations above prove effect; this pins presence, so neither can rot alone."""
    program = sps.check(RESERVES_YAML)
    maps = {name for name, r in program.relations.items() if 'offer' in {r.dim(k) for k in r.key}}
    assert maps == {'gen_of', 'market_of', 'tranche_of'}, 'the offer set is three-legged — the k-ary case'
    sources = port_sources('reserves')
    endpoints = (
        sources['line_from']
        .join(sources['line_to'], on='line', how='left', suffix='_to')
        .select(pl.col('bus').alias('from'), pl.col('bus_to').alias('to'))
        .to_dicts()
    )
    assert endpoints.count({'from': 'b2', 'to': 'b1'}) == 2, 'l1 and l2 are parallel edges between one bus pair'
    assert any(row['to'] is None for row in endpoints), 'l4 dangles: line_to has no row for it, so its leg is open'
    zones_of_g2 = port_sources('reserves')['zone_share'].filter(pl.col('generator') == 'g2')
    assert zones_of_g2.height == 2, 'g2 backs two zones — membership is many-to-many'
    assert set(zones_of_g2['value'].to_list()) == {0.5, 1.0}, 'and at different weights, so the value is a weight'


def test_the_offer_cap_is_two_pullbacks_through_two_legs():
    """A per-offer number assembled from two other dimensions' parameters —
    ``at()`` through ``tranche_of`` times ``at()`` through ``gen_of`` — priced
    into the linopy lane's own solution: o4 sits exactly at 0.25 * 80."""
    with differential(RESERVES_YAML, port_sources('reserves')) as run:
        r = run.result.primal('r')
        assert r.filter(pl.col('offer') == 'o4')['value'][0] == pytest.approx(20.0, rel=RTOL), (
            'o4 must sit at its tranche_frac * p_max cap for the cap to be binding'
        )
