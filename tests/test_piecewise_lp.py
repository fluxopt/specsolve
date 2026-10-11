"""``method: lp`` — the curve as its own segment lines, and no weights at all.

Where the curve is convex and the linked expression is bounded from below by
it, the epigraph *is* the intersection of the segments' half-planes, so the
rows state it directly. The tests hold that it is the same model as
``method: convex``, that it trades columns for rows, and that the curvature is
checked: lines that envelope a convex curve cut a concave one, and the solve
comes back optimal either way.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml

import specsolve as sps
from specsolve.errors import DataError
from tests.conftest import expanded, override, schema_of
from tests.differential import RTOL, differential
from tests.oracle import pd, specsolve_linopy
from tests.piecewise_models import LP_SPEC as SPEC

PER_UNIT_SPEC = """
description: the same curve, one per unit — the shape a corpus of cost curves arrives in

dimensions:
  snapshot: {dtype: int, description: dispatch periods}
  unit: {dtype: str, description: dispatchable units}
  bp: {dtype: int, description: breakpoints of the cost curve}

parameters:
  load: {dims: [snapshot], description: demand to be met}
  bp_x: {dims: [unit, bp], description: 'breakpoint output levels, one curve per unit'}
  bp_y: {dims: [unit, bp], description: 'cost at each breakpoint, one curve per unit'}

variables:
  p:
    dims: [snapshot, unit]
    bounds: {lower: 0, upper: 100}
    description: dispatched power
  op_cost:
    dims: [snapshot, unit]
    bounds: {lower: 0}
    description: operating cost, read off the unit's own curve

piecewise:
  cost_curve:
    description: each unit's cost bounded below by its own segment lines
    over: bp
    links:
      - [p, bp_x]
      - [op_cost, bp_y, '>=']
    method: lp

constraints:
  balance:
    dims: [snapshot]
    expression: sum(p, over=unit) == load
    description: the fleet meets demand

objective:
  sense: minimize
  expression: sum(op_cost)
  description: total operating cost
"""

#: Slopes 1, 2 then 3 — convex, and strictly increasing in x.
BREAKPOINTS_X = [0.0, 10.0, 20.0, 30.0]
BREAKPOINTS_Y = [0.0, 10.0, 30.0, 60.0]
SNAPSHOTS = [0, 1, 2]
LOAD = [5.0, 15.0, 25.0]


def _inputs():
    bp = pd.Index(range(len(BREAKPOINTS_X)), name='bp')
    snapshot = pd.Index(SNAPSHOTS, name='snapshot')
    return {
        'snapshot': snapshot,
        'bp': bp,
        'load': pd.Series(LOAD, index=snapshot),
        'bp_x': pd.Series(BREAKPOINTS_X, index=bp),
        'bp_y': pd.Series(BREAKPOINTS_Y, index=bp),
    }


def _on_the_curve(x: float) -> float:
    return float(np.interp(x, BREAKPOINTS_X, BREAKPOINTS_Y))


# ---------------------------------------------------------------------------
# it is the same model
# ---------------------------------------------------------------------------


def test_the_cost_lands_on_the_curve_and_both_lanes_agree():
    """The expansion is schema-level, so both lanes get identical affine rows.

    Under minimisation the epigraph binds, so each snapshot's cost is the
    curve read at its load — checked against a numpy interpolation, which is
    an oracle that involves no formulation at all.
    """
    with differential(SPEC, _inputs(), lp=True) as run:
        assert run.oracle == pytest.approx(sum(_on_the_curve(x) for x in LOAD), rel=RTOL)
        costs = dict(run.result.primal('op_cost').select('snapshot', 'value').iter_rows())

    for t, load in zip(SNAPSHOTS, LOAD, strict=True):
        assert costs[t] == pytest.approx(_on_the_curve(load), rel=1e-9), (
            f'the cost at load {load} is the curve read there, not a point above it'
        )


def test_a_curve_that_does_not_start_at_the_origin():
    """The case that decides whether the first breakpoint's row is excluded.

    The slope is a difference one position apart, so the first breakpoint has
    no predecessor and its row must not exist. `edge=0` makes the shift read a
    zero there, which is the *origin* — so the row that would be written is the
    line from the origin through the first breakpoint, extended.

    On a curve starting at (0, 0) that line is the first segment and the row is
    harmless, which is why this needs a curve that starts somewhere else. Here
    the ray has slope 10 against segment slopes of 1 and 2, so left in it would
    force a cost of 30 where the curve says 13 — and nothing would say so.
    """
    xs, ys, loads = [1.0, 2.0, 3.0], [10.0, 11.0, 13.0], [1.0, 2.0, 3.0]
    with sps.solve(expanded(pyyaml.safe_load(SPEC)), _relational(load=loads, xs=xs, ys=ys)) as result:
        assert result.objective == pytest.approx(sum(ys)), (
            'each load sits on a breakpoint, so the total is the curve read at each'
        )


def test_lp_and_convex_reach_one_optimum():
    """The two formulations of the same convex curve are the same model.

    `convex` pins the cost to a convex combination of the breakpoints; `lp`
    bounds it below by the segment lines. Under minimisation both are exact,
    and an objective that differed would mean one of them is not.
    """
    hull = override(
        pyyaml.safe_load(SPEC),
        **{'piecewise.cost_curve.method': 'convex', 'piecewise.cost_curve.links': [['p', 'bp_x'], ['op_cost', 'bp_y']]},
    )

    with (
        sps.solve(expanded(pyyaml.safe_load(SPEC)), _relational()) as lines,
        sps.solve(expanded(hull), _relational()) as weights,
    ):
        assert lines.objective == pytest.approx(weights.objective, rel=RTOL)


def test_the_domain_rows_hold_the_output_inside_the_curve():
    """A line does not stop where its segment does, which is why they are emitted.

    Asked for more than the last breakpoint, the model is **infeasible** rather
    than extrapolating along the last segment's slope — which is what `convex`
    does, and what `linopy`'s own `_add_lp` emits its domain rows for.
    """
    beyond = pyyaml.safe_load(SPEC)
    sources = _relational(load=[5.0, 15.0, 45.0])
    with sps.solve(expanded(beyond), sources) as result:
        assert not result.is_ok, 'a load past the last breakpoint is outside the curve, not on its last slope'


def test_the_bounded_link_may_be_written_first():
    """Which link is the curve's x is the sign, not the position in `links:`.

    `PiecewiseBlock.curve` reads the pinned link as x wherever it sits, so the
    two spellings of one block are one model — and the chord rows have to come
    out against `bp_x` either way round.
    """
    swapped = override(
        pyyaml.safe_load(SPEC), **{'piecewise.cost_curve.links': [['op_cost', 'bp_y', '>='], ['p', 'bp_x']]}
    )

    with sps.solve(expanded(swapped), _relational()) as result:
        assert result.objective == pytest.approx(sum(_on_the_curve(x) for x in LOAD), rel=RTOL), (
            'the bounded link written first is the same curve, not a curve with the axes swapped'
        )


def test_a_convex_curve_that_falls_is_still_convex():
    """Convexity is the slopes rising, which says nothing about their sign.

    A curve of falling cost — the returns-to-scale shape — has slopes -3, -2,
    -1, so it is convex and `>=` is exact on it. Nothing in the chord row's
    derivation assumes a rise, and the run it is multiplied through by is
    positive here as everywhere.
    """
    falling = [60.0, 30.0, 10.0, 0.0]
    loads = [0.0, 10.0, 30.0]
    with sps.solve(expanded(pyyaml.safe_load(SPEC)), _relational(load=loads, ys=falling)) as result:
        assert result.objective == pytest.approx(60.0 + 30.0 + 0.0, rel=RTOL), (
            'each load sits on a breakpoint of a falling convex curve, so the total is read off it'
        )


@pytest.mark.parametrize('method', ['adjacency', 'sos2', 'convex'])
def test_a_one_breakpoint_curve_is_that_point_under_the_weight_methods(method):
    """What the answer on a degenerate curve is, taken from the three that agree.

    One weight, forced to 1 by the convexity row, puts both links on the only
    breakpoint there is. This is the number the case below asks `lp` for.
    """
    point = override(
        pyyaml.safe_load(SPEC),
        **{'piecewise.cost_curve.method': method, 'piecewise.cost_curve.links': [['p', 'bp_x'], ['op_cost', 'bp_y']]},
    )

    with sps.solve(expanded(point), _relational(load=[10.0, 10.0, 10.0], xs=[10.0], ys=[25.0])) as result:
        assert result.objective == pytest.approx(3 * 25.0, rel=RTOL), (
            f'method: {method} pins the cost to the one breakpoint the curve has'
        )


@pytest.mark.parametrize('spelling', ['values-parameter', 'boolean-mask'])
def test_a_ragged_curve_down_to_one_point_is_refused(spelling):
    """The count that decides is the curve's, and `points:` makes them differ.

    `bp` carries three breakpoints and unit `b` runs over one of them, so a
    check asking the *dimension* clears a curve that has no segment: `b`'s
    chord row is excluded as its own first point, its two domain rows pin only
    `p`, and `op_cost` settles on its lower bound for an objective of 0 where
    `b`'s single point says 25. Its neighbour with two points is the same model
    built one breakpoint longer, and solves.

    Both spellings of `points:`, because they put the length in different
    places: naming a values parameter leaves the unrun breakpoints out of the
    table, and a boolean mask marks them in a table that may be dense. A count
    of the rows that carry a value gets the first right and the second wrong.
    """
    mask = spelling == 'boolean-mask'
    ragged = override(pyyaml.safe_load(PER_UNIT_SPEC), **{'piecewise.cost_curve.points': 'runs_to' if mask else 'bp_x'})
    if mask:
        ragged['parameters']['runs_to'] = {'dims': ['unit', 'bp'], 'dtype': 'bool', 'description': 'curve length'}

    with sps.solve(expanded(ragged), _per_unit_points(short=False, mask=mask)) as result:
        assert result.objective == pytest.approx(25.0, rel=RTOL), 'two points is one segment, and that is enough'

    with pytest.raises(DataError, match='needs at least two breakpoints') as refusal:
        sps.build(expanded(ragged), _per_unit_points(short=True, mask=mask))
    assert 'unit=b' in str(refusal.value), 'the refusal names the curve that is short, not the block'


def test_values_past_the_mask_are_not_part_of_the_curve():
    """`points:` says which breakpoints the curve runs over, and the guard reads it.

    A table may be dense where the mask is not — the row exists, it is simply
    not on this curve. Judged by which rows carry a value instead, `b`'s
    unmarked third point would count, and it runs backwards *and* bends the
    wrong way.
    """
    sources = _per_unit_points(short=False, mask=True)
    past = (pl.col('unit') == 'b') & (pl.col('bp') == 2)
    for name, unusable in (('bp_x', 1.0), ('bp_y', 500.0)):
        sources[name] = sources[name].with_columns(
            pl.when(past).then(unusable).otherwise(pl.col('value')).alias('value')
        )

    ragged = override(
        pyyaml.safe_load(PER_UNIT_SPEC),
        **{
            'parameters.runs_to': {'dims': ['unit', 'bp'], 'dtype': 'bool', 'description': 'curve length'},
            'piecewise.cost_curve.points': 'runs_to',
        },
    )

    with sps.solve(expanded(ragged), sources) as result:
        assert result.objective == pytest.approx(25.0, rel=RTOL), (
            'the answer is the marked curves, and the values past them are not read at all'
        )


def test_a_one_breakpoint_curve_is_refused_rather_than_dropped():
    """A curve of one point has no segment, which is the whole of what lp states.

    The chord row is written at the later of the two breakpoints it joins, so
    a single breakpoint writes none, and the two domain rows pin only the
    *pinned* link: the bounded one sits on its own bound, 0 under minimisation
    against a curve that says 25 (#1121). The three weight methods reach 25 on
    this data, so the refusal names them.
    """
    point = _relational(load=[10.0, 10.0, 10.0], xs=[10.0], ys=[25.0])

    with pytest.raises(DataError, match='needs at least two breakpoints') as refusal:
        sps.build(expanded(pyyaml.safe_load(SPEC)), point)
    assert 'adjacency' in str(refusal.value), 'and names the methods a one-point curve does mean something under'


# ---------------------------------------------------------------------------
# what it costs
# ---------------------------------------------------------------------------


def test_the_saving_is_columns_paid_for_in_rows():
    """The trade, measured on one model rather than argued.

    `convex` declares a weight per breakpoint per frame row; `lp` declares
    none, and instead writes a row per segment plus two holding the domain.
    So this is columns traded for rows, and the numbers say how much.
    """
    sizes = {}
    for method, links in (
        ('convex', [['p', 'bp_x'], ['op_cost', 'bp_y']]),
        ('lp', [['p', 'bp_x'], ['op_cost', 'bp_y', '>=']]),
    ):
        spec = override(
            pyyaml.safe_load(SPEC),
            **{'piecewise.cost_curve.method': method, 'piecewise.cost_curve.links': links},
        )
        built = sps.build(expanded(spec), _relational())
        sizes[method] = built.diagnostics()
        built.close()

    weights = len(SNAPSHOTS) * len(BREAKPOINTS_X)
    assert sizes['convex'].columns - sizes['lp'].columns == weights, (
        f'lp declares no weights, so it is exactly the {weights} lambda columns lighter'
    )
    assert sizes['lp'].rows > sizes['convex'].rows, 'and it pays for them in rows — this is a trade, not a free win'


# ---------------------------------------------------------------------------
# the curvature that makes it exact
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('sign', 'sense', 'ys', 'wanted'),
    [
        pytest.param('>=', 'minimize', [0.0, 30.0, 50.0, 60.0], 'convex', id='a-cost-bounded-below-by-a-concave-curve'),
        pytest.param('<=', 'maximize', BREAKPOINTS_Y, 'concave', id='a-yield-bounded-above-by-a-convex-curve'),
    ],
)
def test_the_curvature_the_sign_states_is_required(sign, sense, ys, wanted):
    """The silent case, and the reason the guard is directional.

    Lines through a *concave* curve's breakpoints lie above it, so bounding a
    cost below by them cuts the feasible region — and the solve comes back
    optimal with a wrong answer. `method: convex`'s own guard would pass such a
    curve, since it is not mixed; only `lp` cares which way it bends.
    """
    spec = override(
        pyyaml.safe_load(SPEC),
        **{'piecewise.cost_curve.links': [['p', 'bp_x'], ['op_cost', 'bp_y', sign]], 'objective.sense': sense},
    )
    with pytest.raises(DataError, match=f'exact only for a {wanted} curve'):
        sps.solve(expanded(spec), _relational(ys=ys))
    assert schema_of(SPEC) is not None, 'and the schema alone is fine — this needs the values'


def test_breakpoints_that_do_not_increase_are_refused():
    """The run is what the row is multiplied through by, so it must be positive."""
    spec = pyyaml.safe_load(SPEC)
    with pytest.raises(DataError, match='requires strictly increasing breakpoints'):
        sps.solve(expanded(spec), _relational(xs=[0.0, 10.0, 10.0, 30.0]))


def test_each_curve_of_a_frame_is_checked_on_its_own():
    """A block carries one curve per frame row, and one bad one is enough.

    The guard broadcasts over the dims the values carry and walks the
    breakpoints of each row, so a concave curve hidden among convex ones is
    refused on its own account — which is the shape a per-unit corpus arrives
    in.
    """
    convex, concave = [0.0, 10.0, 30.0, 60.0], [0.0, 30.0, 50.0, 60.0]

    sps.build(
        expanded(pyyaml.safe_load(PER_UNIT_SPEC)), _per_unit(convex, convex)
    ).close()  # every curve convex, nothing to say

    with pytest.raises(DataError, match='exact only for a convex curve') as refusal:
        sps.build(expanded(pyyaml.safe_load(PER_UNIT_SPEC)), _per_unit(convex, concave))
    assert 'unit=b' in str(refusal.value), 'the refusal names the curve that bends the wrong way'


def test_a_curve_bound_to_a_path_is_checked_like_one_in_memory(tmp_path):
    """The verdict is a property of the numbers, not of how they were handed over.

    Unchecked, this concave curve handed over as parquet comes back optimal at
    155 where the curve says 110 (#1123). Both lanes scan a path for the two
    columns `validate_curve_extent` already reads.
    """
    concave = [0.0, 30.0, 50.0, 60.0]
    sources = _relational(ys=concave)
    for name in ('bp_x', 'bp_y'):
        sources[name].write_parquet(tmp_path / f'{name}.parquet')
        sources[name] = tmp_path / f'{name}.parquet'

    for lane in (sps.build, specsolve_linopy.build):
        with pytest.raises(DataError, match='exact only for a convex curve'):
            lane(expanded(pyyaml.safe_load(SPEC)), sources)


def test_a_concave_curve_is_refused_whatever_the_breakpoints_are_measured_in():
    """The guard's tolerance is in the units of what it compares, so x cancels.

    `diff(diff(ys) / dx)` is a difference of slopes, y per x. Judged against
    `1e-9 * max(|y|)`, which carries no x, this curve stretched along x would
    pass: it is concave by 3000 cost units, and `lp` would return 4502000 where
    the curve says 4497500 (#1124).
    """
    xs = [0.0, 1e6, 2e6, 3e6]
    concave = [0.0, 1e6, 2e6 - 1000.0, 3e6 - 3000.0]

    stretched = override(pyyaml.safe_load(SPEC), **{'variables.p.bounds.upper': 3e6})

    with pytest.raises(DataError, match='exact only for a convex curve'):
        sps.build(expanded(stretched), _relational(load=[5e5, 1.5e6, 2.5e6], xs=xs, ys=concave))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _relational(load=None, xs=None, ys=None):
    xs = BREAKPOINTS_X if xs is None else xs
    ys = BREAKPOINTS_Y if ys is None else ys
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'bp': pl.DataFrame({'bp': list(range(len(xs)))}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': LOAD if load is None else load}),
        'bp_x': pl.DataFrame({'bp': list(range(len(xs))), 'value': xs}),
        'bp_y': pl.DataFrame({'bp': list(range(len(ys))), 'value': ys}),
    }


def _per_unit(first, second):
    units = ['a', 'b']
    return {
        'snapshot': pl.DataFrame({'snapshot': SNAPSHOTS}),
        'unit': pl.DataFrame({'unit': units}),
        'bp': pl.DataFrame({'bp': list(range(len(BREAKPOINTS_X)))}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': LOAD}),
        'bp_x': pl.DataFrame(
            {
                'unit': [u for u in units for _ in BREAKPOINTS_X],
                'bp': list(range(len(BREAKPOINTS_X))) * 2,
                'value': BREAKPOINTS_X * 2,
            }
        ),
        'bp_y': pl.DataFrame(
            {
                'unit': [u for u in units for _ in BREAKPOINTS_X],
                'bp': list(range(len(BREAKPOINTS_X))) * 2,
                'value': [*first, *second],
            }
        ),
    }


def _per_unit_points(short, mask=False):
    """`b` runs over one breakpoint or two; `mask` says so in a table of its own.

    Under the boolean spelling the curve tables stay **dense** — every unit
    carries all three breakpoints — which is the shape that separates reading
    the mask from counting the rows that have a value.
    """
    run = [('a', 0), ('a', 1), ('a', 2), *([('b', 0)] if short else [('b', 0), ('b', 1)])]
    rows = [('a', 0), ('a', 1), ('a', 2), ('b', 0), ('b', 1), ('b', 2)] if mask else run
    curve = {('a', 0): (0.0, 0.0), ('a', 1): (10.0, 10.0), ('a', 2): (20.0, 30.0)}
    curve |= {('b', 0): (5.0, 25.0), ('b', 1): (15.0, 60.0), ('b', 2): (40.0, 200.0)}
    sources = {
        'snapshot': pl.DataFrame({'snapshot': [0]}),
        'unit': pl.DataFrame({'unit': ['a', 'b']}),
        'bp': pl.DataFrame({'bp': [0, 1, 2]}),
        'load': pl.DataFrame({'snapshot': [0], 'value': [5.0]}),
        'bp_x': pl.DataFrame(
            {'unit': [u for u, _ in rows], 'bp': [k for _, k in rows], 'value': [curve[r][0] for r in rows]}
        ),
        'bp_y': pl.DataFrame(
            {'unit': [u for u, _ in rows], 'bp': [k for _, k in rows], 'value': [curve[r][1] for r in rows]}
        ),
    }
    if mask:
        sources['runs_to'] = pl.DataFrame(
            {'unit': [u for u, _ in run], 'bp': [k for _, k in run], 'value': [True] * len(run)}
        )
    return sources
