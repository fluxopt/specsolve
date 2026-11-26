"""piecewise costs: the λ-formulation block, and the epigraph that needs none.

The ``piecewise:`` expansion runs before either backend, so both lanes receive
identical affine entries. Nonconvex correctness is checked by the linked
primals lying on the curve, against a numpy interpolation; the ``convex:`` flag
produces the hull instead. The last section writes convex piecewise as
epigraph constraints: ordinary affine YAML with no ``piecewise:`` block.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
import yaml as pyyaml
from mathspec.errors import DimensionError

import specsolve as sps
from specsolve._sources import attachable, tidy_sources
from specsolve.errors import DataError, SpecsolveError
from tests.conftest import EXAMPLES_DIR, expanded, override, primal_by_coordinate, raw_of, schema_of
from tests.differential import differential
from tests.oracle import pd, specsolve_linopy
from tests.piecewise_models import CHP_YAML, GATED_YAML, NONCONVEX_YAML, SOS2_SPEC, TWO_DIM_YAML, curve_table

#: The same model with the hull instead of the curve — `method: convex` and
#: nothing else changed.
CONVEX_SPEC = override(raw_of(NONCONVEX_YAML), **{'piecewise.cost_curve.method': 'convex'})


#: Breakpoints whose x goes backwards — the shape the curvature guard refuses.
BACKWARDS_BP_X = pd.Series([0.0, 50.0, 40.0], index=pd.RangeIndex(3, name='bp'))


def curve(p, bp_x, bp_y) -> float:
    return float(np.interp(p, np.asarray(bp_x), np.asarray(bp_y)))


@pytest.fixture
def nonconvex_inputs():
    """A concave curve — economies of scale — and a load that reaches into it.

    The convex hull's lower envelope is the chord, which undercuts a concave
    curve, so the adjacency binaries are load-bearing on this fixture.
    """
    rng = np.random.default_rng(13)
    n_s = 12
    bp_x = pd.Series([0.0, 40.0, 100.0], index=pd.RangeIndex(3, name='bp'))
    bp_y = pd.Series([0.0, 30.0, 55.0], index=pd.RangeIndex(3, name='bp'))
    load = pd.Series(rng.uniform(5, 95, n_s).round(2), index=pd.RangeIndex(n_s, name='snapshot'))
    return {'load': load, 'bp_x': bp_x, 'bp_y': bp_y, 'snapshot': load.index, 'bp': bp_x.index}


# ---------------------------------------------------------------------------
# the λ formulation, end to end
# ---------------------------------------------------------------------------


def test_the_solution_sits_on_the_curve_not_on_its_hull(nonconvex_inputs):
    """The λ formulation reaches the curve itself, not the chord under it."""
    data = nonconvex_inputs
    expected = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])

    with differential(NONCONVEX_YAML, data) as run:
        assert run.oracle == pytest.approx(expected, rel=1e-6), 'ON the curve, not on the hull'

        cost = primal_by_coordinate(run.result, 'op_cost', 'snapshot')
        for s, load_v in data['load'].items():
            assert cost[s] == pytest.approx(curve(load_v, data['bp_x'], data['bp_y']), abs=1e-6)


def test_the_convex_flag_gives_the_hull_and_stays_a_pure_lp(nonconvex_inputs):
    """`method: convex` drops the binaries, and says so in the answer.

    The same concave curve relaxes to its hull, whose lower envelope is the
    chord, so the objective must land below the curve.
    """
    data = nonconvex_inputs

    program = expanded(CONVEX_SPEC, 'piecewise').program
    assert all(v.domain == 'continuous' for v in program.variables.values()), 'method: convex is a pure LP'

    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])
    chord = sum(0.55 * v for v in data['load'])  # the (100, 55) chord from the origin
    with differential(CONVEX_SPEC, data) as run:
        assert run.oracle == pytest.approx(chord, rel=1e-6)
        assert run.oracle < on_curve, 'the hull undercuts a concave curve'


def test_three_links_all_track_the_same_curve_position():
    n_s = 8
    rng = np.random.default_rng(21)
    power_bp = pd.Series([0.0, 50.0, 100.0], index=pd.RangeIndex(3, name='bp'))
    fuel_bp = pd.Series([10.0, 60.0, 140.0], index=pd.RangeIndex(3, name='bp'))
    heat_bp = pd.Series([0.0, 20.0, 60.0], index=pd.RangeIndex(3, name='bp'))
    load = pd.Series(rng.uniform(10, 90, n_s).round(2), index=pd.RangeIndex(n_s, name='snapshot'))
    data = {
        'load': load,
        'power_bp': power_bp,
        'fuel_bp': fuel_bp,
        'heat_bp': heat_bp,
        'snapshot': load.index,
        'bp': power_bp.index,
    }

    with differential(CHP_YAML, data) as run:
        fuel = primal_by_coordinate(run.result, 'fuel', 'snapshot')
        heat = primal_by_coordinate(run.result, 'heat', 'snapshot')
        for s, load_v in load.items():
            assert fuel[s] == pytest.approx(curve(load_v, power_bp, fuel_bp), abs=1e-6)
            assert heat[s] == pytest.approx(curve(load_v, power_bp, heat_bp), abs=1e-6)


def test_activity_gates_the_curve_off(nonconvex_inputs):
    """`activity:` decides whether the curve applies at a coordinate at all.

    Gated on, the cost sits on the curve at the pinned load; gated off, it is
    pinned to zero.
    """
    data = nonconvex_inputs
    on_flag = pd.Series([1.0, 0.0] * 6, index=pd.RangeIndex(12, name='snapshot'))
    data = {**data, 'on_flag': on_flag}

    with differential(GATED_YAML, data) as run:
        cost = primal_by_coordinate(run.result, 'op_cost', 'snapshot')
        for s in on_flag.index:
            expected = curve(data['load'][s], data['bp_x'], data['bp_y']) if on_flag[s] else 0.0
            assert cost[s] == pytest.approx(expected, abs=1e-6)


def _of(table, generator):
    """One generator's breakpoint values, in order — the tidy-table `xs`."""
    return table.loc[table['generator'] == generator, 'value'].to_numpy()


def test_breakpoints_may_vary_along_another_dim():
    """examples/piecewise.yaml: convex per-generator curves (breakpoints vary
    along the generator dimension — the thing flat breakpoint lists can't do).

    Each generator gets an increasing marginal cost of a different shape, and
    each one's cost has to sit on its own curve: the hull is exact here,
    because the curves are convex and the objective minimises.
    """
    example = EXAMPLES_DIR / 'piecewise.yaml'
    rng = np.random.default_rng(31)
    n_s = 24
    gens = pd.Index(['cheap', 'mid'], name='generator')
    bps = pd.RangeIndex(3, name='bp')
    p_max = pd.Series({'cheap': 100.0, 'mid': 120.0})
    per_generator = pd.MultiIndex.from_product([gens, bps], names=['generator', 'bp']).to_frame(index=False)
    bp_x = per_generator.assign(value=[0.0, 40.0, 100.0, 0.0, 60.0, 120.0])
    bp_y = per_generator.assign(value=[0.0, 200.0, 800.0, 0.0, 900.0, 2700.0])
    load = pd.Series(
        (rng.uniform(0.3, 0.9, n_s) * p_max.sum()).round(1),
        index=pd.RangeIndex(n_s, name='snapshot'),
    )
    data = {
        'p_max': p_max,
        'load': load,
        'bp_x': bp_x,
        'bp_y': bp_y,
        'snapshot': load.index,
        'generator': gens,
        'bp': bps,
    }

    expanded(example, 'piecewise')

    with differential(example, data) as run:
        p = primal_by_coordinate(run.result, 'p', 'snapshot', 'generator')
        cost = primal_by_coordinate(run.result, 'op_cost', 'snapshot', 'generator')
        for (s, g), pv in p.items():
            expected = curve(pv, _of(bp_x, g), _of(bp_y, g))
            assert cost[(s, g)] == pytest.approx(expected, abs=1e-5)


# ---------------------------------------------------------------------------
# what the expansion emits, and what it refuses
# ---------------------------------------------------------------------------


def test_the_sos2_method_states_the_restriction_instead_of_building_it():
    """The same weights, the same convexity row, and no binaries at all.

    What changes is only *how λ is restricted*: the segment variable and the
    two rows that pick and neighbour it are gone, replaced by a set over the
    weights the block already emits.
    """
    program = expanded(SOS2_SPEC, 'piecewise').program

    assert list(program.variables) == ['cost_curve_lam', 'op_cost', 'p'], (
        'the weights are emitted and the segment binaries a method with none would need are not'
    )
    assert set(program.constraints) == {
        'cost_curve_convexity',
        'cost_curve_p',
        'cost_curve_op_cost',
        'balance',
    }, 'the two rows that pick and neighbour a segment are gone with the variable they restricted'
    assert all(v.domain == 'continuous' for v in program.variables.values()), 'sos2 emits no binary of its own'
    assert [(s.variable, s.sos_type, s.along) for s in program.sos.values()] == [('cost_curve_lam', 2, 'bp')], (
        'one set, over the weights, of the declared type'
    )


def test_the_sos2_method_reaches_the_curve_the_binaries_reach(nonconvex_inputs):
    """Two spellings of one restriction, so they must agree on the answer.

    The concave fixture is what makes this a claim: the hull undercuts the
    curve there, so a set that failed to restrict anything would show up as
    the chord rather than as a near miss.
    """
    data = nonconvex_inputs
    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])

    with differential(SOS2_SPEC, data) as run:
        assert run.result.objective == pytest.approx(on_curve, rel=1e-6), 'ON the curve, not on the hull'
        cost = primal_by_coordinate(run.result, 'op_cost', 'snapshot')
        for s, load_v in data['load'].items():
            assert cost[s] == pytest.approx(curve(load_v, data['bp_x'], data['bp_y']), abs=1e-6)


def test_the_sos2_method_solves_natively_where_the_sink_has_the_concept(nonconvex_inputs):
    """The whole point of saying it rather than building it."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    data = nonconvex_inputs
    on_curve = sum(curve(v, data['bp_x'], data['bp_y']) for v in data['load'])
    assert sps.solve(expanded(SOS2_SPEC, 'piecewise'), data, 'gurobi').objective == pytest.approx(on_curve, rel=1e-6)


def test_the_sos2_method_gates_off_like_the_binaries_do(nonconvex_inputs):
    """``activity`` is a property of the weights, so every method keeps it.

    A gated-off block pins the convexity row to zero, which sets every weight
    to zero — a state the set admits, since at most two nonzero is satisfied
    by none. ``method: convex`` is the one that refuses ``activity``.
    """
    data = nonconvex_inputs
    gated = override(raw_of(GATED_YAML), **{'piecewise.cost_curve.method': 'sos2'})
    on_flag = pd.Series([1.0, 0.0] * 6, index=pd.RangeIndex(12, name='snapshot'))

    with differential(gated, {**data, 'on_flag': on_flag}) as run:
        cost = primal_by_coordinate(run.result, 'op_cost', 'snapshot')
        for s in on_flag.index:
            expected = curve(data['load'][s], data['bp_x'], data['bp_y']) if on_flag[s] else 0.0
            assert cost[s] == pytest.approx(expected, abs=1e-6)


def test_the_adjacency_row_survives_at_the_first_breakpoint(nonconvex_inputs):
    """The adjacency row exists at the first breakpoint, where the shifted term has no predecessor.

    Adjacency is ``lam <= seg + shift(seg, along=bp, offset=1, edge=0)``.
    Filled, the missing predecessor contributes zero and the row reads
    ``lam <= seg``. Absent, it would drop the row (#289) and leave the first
    lambda bounded only by ``[0, 1]``, a wrong MILP that still solves.
    """
    data = nonconvex_inputs
    with differential(NONCONVEX_YAML, data) as run:
        first = run.model.constraints['cost_curve_adjacency'].labels.isel({'bp': 0}).values
        assert (first != -1).all(), 'the first breakpoint lost its adjacency row'


def test_both_lanes_check_the_entries_a_formulation_emits(tmp_path):
    """A stray dimension is named on the link that carries it, on both lanes.

    A values parameter carrying a dimension the links do not is a stray dimension in
    generated math — one row per zone where the file reads as one per
    snapshot. The refusal names the link, not a generated constraint the
    author never wrote.
    """
    raw = override(
        raw_of(NONCONVEX_YAML),
        **{'dimensions.zone': {'dtype': 'str'}, 'parameters.bp_y': {'dims': ['zone', 'bp']}},
    )
    stray = r"link 'op_cost' values parameter 'bp_y' is over \['zone'\], and its row \['snapshot'\] is not"

    with pytest.raises(DimensionError, match=stray):
        expanded(raw)

    path = tmp_path / 'stray_dim.yaml'
    path.write_text(pyyaml.safe_dump(raw))
    with pytest.raises(DimensionError, match=stray):
        specsolve_linopy.build(expanded(path), {})


def test_a_curve_left_as_written_is_refused_by_both_lanes(tmp_path):
    """A ``piecewise:`` block is refused rather than expanded, and the refusal names the expansion.

    Both lanes compile a model through `as_program`, so both refuse the same file in
    the same words (hard rule 3), and neither expands it on the caller's
    behalf: nothing expands a formulation unasked, here or in the language.
    """
    path = tmp_path / 'as_written.yaml'
    path.write_text(NONCONVEX_YAML)

    with pytest.raises(SpecsolveError, match="piecewise: 'cost_curve' is still a curve") as relational:
        sps.check(path)
    with pytest.raises(SpecsolveError, match="piecewise: 'cost_curve' is still a curve") as linopy_lane:
        specsolve_linopy.build(path, {})
    assert "expand('piecewise')" in str(relational.value) and str(relational.value) == str(linopy_lane.value), (
        'one refusal, naming the expansion, on both lanes'
    )


# ---------------------------------------------------------------------------
# the data guard: `method: convex` is a promise about the breakpoints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ('breakpoints', 'match'),
    [
        pytest.param(
            {
                'bp': [0, 1, 2, 3],
                'bp_x': pd.Series([0.0, 30.0, 60.0, 100.0], index=pd.RangeIndex(4, name='bp')),
                'bp_y': pd.Series([0.0, 10.0, 40.0, 50.0], index=pd.RangeIndex(4, name='bp')),
            },
            'exact only for a single bend',
            id='convex-then-concave-the-hull-would-cut-corners',
        ),
        pytest.param(
            {'bp_x': BACKWARDS_BP_X},
            'strictly increasing',
            id='breakpoints-that-go-backwards',
        ),
    ],
)
def test_convex_breakpoints_that_are_not_convex_are_refused(nonconvex_inputs, breakpoints, match):
    data = nonconvex_inputs
    schema = schema_of(CONVEX_SPEC)

    with pytest.raises(DataError, match=match):
        tidy_sources(schema.expand('piecewise').program, {**data, **breakpoints})


def test_the_curvature_guard_also_fires_through_the_relational_adapter(nonconvex_inputs):
    """`tidy_sources` is the one function the relational lane reads data through, so the guard
    has to live behind it too — not only in the linopy loader."""
    data = nonconvex_inputs
    schema = schema_of(CONVEX_SPEC)

    tidy_sources(schema.expand('piecewise').program, data)  # consistent (concave) curvature passes

    bad = {**data, 'bp_x': BACKWARDS_BP_X}
    with pytest.raises(DataError, match='strictly increasing'):
        tidy_sources(schema.expand('piecewise').program, bad)


# ---------------------------------------------------------------------------
# the data guard reads the curve in the order the model builds it
# ---------------------------------------------------------------------------


#: `nonconvex_inputs`' three breakpoints, labelled the same and written in
#: another order. A tidy table carries its coordinates in its labels, so this
#: is the same curve — the engine joins it by label and reaches one model.
OUT_OF_ORDER_BP = pd.Index([2, 0, 1], name='bp')


def test_a_curve_written_out_of_order_is_the_same_curve(nonconvex_inputs):
    """A row order is not a breakpoint order, and only the second is the model's.

    Rows reach a lane in whatever order the join or the group-by that made
    them left behind; what orders the breakpoints is the order of `bp`'s labels, which is
    ascending here (#1122).
    """
    shuffled = {
        **nonconvex_inputs,
        'bp_x': pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP),
        'bp_y': pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP),
    }

    schema = schema_of(CONVEX_SPEC)

    tidy_sources(schema.expand('piecewise').program, shuffled)


def test_a_breakpoint_dimension_with_no_labels_keeps_its_own_message(nonconvex_inputs):
    """With no labels there is no order, so the guard has no question to answer.

    A curve written out of order reaches the message that names the missing
    labels, not "requires strictly increasing breakpoints".
    """
    orphaned = {k: v for k, v in nonconvex_inputs.items() if k != 'bp'}
    orphaned['bp_x'] = pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP)
    orphaned['bp_y'] = pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP)

    schema = schema_of(CONVEX_SPEC)

    with pytest.raises(DataError, match='has no labels'):
        tidy_sources(schema.expand('piecewise').program, orphaned)


def test_the_linopy_lane_reads_the_curve_in_label_order(nonconvex_inputs, tmp_path):
    """The linopy loader lays the values out first, so the guard walks the dimension's order.

    The shuffled curve is attached and the backwards labels are refused; the relational
    lane owes the same two answers.
    """
    path = tmp_path / 'convex.yaml'
    path.write_text(pyyaml.safe_dump(CONVEX_SPEC))
    shuffled = {
        **nonconvex_inputs,
        'bp_x': pd.Series([100.0, 0.0, 40.0], index=OUT_OF_ORDER_BP),
        'bp_y': pd.Series([55.0, 0.0, 30.0], index=OUT_OF_ORDER_BP),
    }

    specsolve_linopy.build(expanded(path), shuffled)  # a row order is not a breakpoint order

    with pytest.raises(DataError, match='strictly increasing'):
        specsolve_linopy.build(expanded(path), {**nonconvex_inputs, 'bp': pd.Index([2, 1, 0], name='bp')})


def test_breakpoint_labels_that_run_backwards_are_refused(nonconvex_inputs):
    """Breakpoint labels that run backwards are refused (#1122).

    A dimension's labels are its order — `shift` walks them and `index(bp, 0)`
    names the first — so labels written `[2, 1, 0]` put the fixture's
    breakpoints at x = 100, 40, 0. `adjacency` would then pair segments that
    are not neighbours, and `lp` would write its chords against a negative run.
    """
    backwards = {**nonconvex_inputs, 'bp': pd.Index([2, 1, 0], name='bp')}
    schema = schema_of(CONVEX_SPEC)

    with pytest.raises(DataError, match='strictly increasing'):
        tidy_sources(schema.expand('piecewise').program, backwards)


# ---------------------------------------------------------------------------
# the data guard: a curve carries a value at every breakpoint it is built over
# ---------------------------------------------------------------------------


@pytest.fixture
def ragged_inputs():
    """Generator B supplies two of the three breakpoints the dimension declares.

    B's curve starts at (10, 100), so the row it never wrote is not a harmless
    repeat of its first point: read as a zero coefficient it is a vertex at
    (0, 0), and the weights mix onto it to run B below the minimum output its
    own curve states.
    """
    return {
        'snapshot': [0],
        'generator': ['A', 'B'],
        'bp': [0, 1, 2],
        'load': pd.Series([25.0], index=pd.RangeIndex(1, name='snapshot')),
        'bp_x': curve_table({('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0}),
        'bp_y': curve_table({('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0}),
    }


def test_a_curve_short_of_a_breakpoint_is_refused(ragged_inputs):
    """A missing breakpoint row read as a zero coefficient is a vertex at the origin, so it is refused.

    On this fixture B would interpolate between its real (20, 130) and the
    (0, 0) it never declared, for an optimum of 147.5 where its own two points
    put it at 195. A curve with no `where:` reads its tables under the default
    `missing: refused`, so the refusal names the breakpoint B has no row at.
    """
    schema = schema_of(raw_of(TWO_DIM_YAML))

    with pytest.raises(DataError, match=r"parameter 'bp_x' has no row at bp=2, generator=B"):
        tidy_sources(schema.expand('piecewise').program, dict(ragged_inputs))


def test_the_curve_guard_fires_on_the_linopy_lane_too(ragged_inputs, tmp_path):
    """Both lanes take the same sources, so both refuse the same table (hard rule 3)."""
    path = tmp_path / 'two_dim.yaml'
    path.write_text(TWO_DIM_YAML)

    with pytest.raises(DataError, match=r"parameter 'bp_x' has no row at bp=2, generator=B"):
        specsolve_linopy.build(expanded(path), dict(ragged_inputs))


def test_a_curve_supplied_at_every_breakpoint_passes(ragged_inputs):
    """The guard is about holes, not about how the table is written."""
    whole = dict(ragged_inputs)
    whole['bp_x'] = curve_table(
        {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0, ('B', 2): 30.0}
    )
    whole['bp_y'] = curve_table(
        {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0, ('B', 2): 200.0}
    )

    schema = schema_of(raw_of(TWO_DIM_YAML))

    tidy_sources(schema.expand('piecewise').program, whole)


def test_a_dict_shaped_curve_is_read_for_holes_too(ragged_inputs, tmp_path):
    """The linopy lane takes the caller's mapping unspread, so the guard reads that spelling.

    A ``{label: value}`` curve is the one plain-Python shape that can be short:
    a sequence and a single number are dense against the labels they spread
    over, a dict carries only the keys it was written with.
    """
    path = tmp_path / 'one_dim.yaml'
    path.write_text(NONCONVEX_YAML)
    data = {
        'snapshot': [0],
        'bp': [0, 1, 2],
        'load': pd.Series([25.0], index=pd.RangeIndex(1, name='snapshot')),
        'bp_x': {0: 0.0, 1: 10.0},
        'bp_y': {0: 0.0, 1: 50.0},
    }

    with pytest.raises(DataError, match=r"parameter 'bp_x' has no row at bp=2"):
        specsolve_linopy.build(expanded(path), data)


def test_a_dimension_with_no_labels_keeps_its_own_message(ragged_inputs):
    """The guard runs before the labels are attached, and must not answer for its absence.

    Where nothing declares the breakpoints, the curve's own labels are all
    there is — it cannot be short of a breakpoint no one declared — so a
    complete curve has to reach the message that names the missing labels.
    """
    whole = {k: v for k, v in ragged_inputs.items() if k != 'bp'}
    whole['bp_x'] = curve_table({('A', 0): 0.0, ('A', 1): 20.0, ('B', 0): 10.0, ('B', 1): 20.0})
    whole['bp_y'] = curve_table({('A', 0): 0.0, ('A', 1): 140.0, ('B', 0): 100.0, ('B', 1): 130.0})

    schema = schema_of(raw_of(TWO_DIM_YAML))

    with pytest.raises(DataError, match='has no labels'):
        tidy_sources(schema.expand('piecewise').program, whole)


# ---------------------------------------------------------------------------
# where: a curve shorter than its breakpoint dimension
# ---------------------------------------------------------------------------

SHORT_CURVE = """
dimensions:
  generator: {dtype: str}
  bp: {dtype: int, ordered: true}

parameters:
  p_max: {dims: [generator]}
  load: {dims: []}
  bp_x: {dims: [generator, bp], missing: absent}
  bp_y: {dims: [generator, bp], missing: absent}
  bp_present: {dims: [generator, bp], dtype: bool, missing: false}

variables:
  p:
    dims: [generator]
    bounds: {lower: 0, upper: p_max}
  op_cost:
    dims: [generator]
    bounds: {lower: 0}

piecewise:
  cost_curve:
    along: bp
    dims: [generator]
    where: bp_present
    links:
      p: [p, bp_x]
      op_cost: [op_cost, bp_y]

constraints:
  balance:
    dims: []
    expression: sum(p, over=generator) == load

objective:
  sense: minimize
  expression: sum(op_cost, over=generator)
"""

#: A three-breakpoint dimension where B is a two-point curve starting at (10, 100):
#: at a load of 25 the cheap answer runs B at 20 and A at 5, for 155.
A_AND_SHORT_B = {
    'x': {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0},
    'y': {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0},
}


@pytest.fixture
def short_curve_inputs():
    """B's rows stop at its second breakpoint, and the condition says so."""
    present = {(g, k): ((g, k) in A_AND_SHORT_B['x']) for g in ('A', 'B') for k in range(3)}
    return {
        'generator': ['A', 'B'],
        'bp': [0, 1, 2],
        'load': pl.DataFrame({'value': [25.0]}),
        'p_max': pl.DataFrame({'generator': ['A', 'B'], 'value': [20.0, 20.0]}),
        'bp_x': curve_table(A_AND_SHORT_B['x']),
        'bp_y': curve_table(A_AND_SHORT_B['y']),
        'bp_present': curve_table(present),
    }


@pytest.mark.parametrize('method', ['adjacency', 'convex', 'lp'])
def test_both_lanes_agree_on_a_narrowed_curve(short_curve_inputs, method, tmp_path):
    """Whatever the condition reaches has to reach it on both lanes (hard rule 3).

    `lp` is the one whose rows the condition reaches directly, and the one whose
    domain rows sit on each curve's own first and last breakpoint rather than
    the axis'.
    """
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.method': method})
    if method == 'lp':
        raw['piecewise']['cost_curve']['links']['op_cost'] = ['op_cost', 'bp_y', '>=']
    path = tmp_path / 'masked.yaml'
    path.write_text(pyyaml.safe_dump(raw))

    built = specsolve_linopy.build(expanded(path), short_curve_inputs)
    built.solve('highs', output_flag=False)

    assert float(built.objective.value) == pytest.approx(155.0)
    assert sps.solve(expanded(raw), short_curve_inputs).objective == pytest.approx(155.0), (
        'and the same on the other lane'
    )


@pytest.mark.parametrize('method', ['adjacency', 'sos2', 'convex', 'lp'])
def test_a_narrowed_curve_reaches_the_optimum_its_own_points_put_it_at(short_curve_inputs, method):
    """Every method reads the condition, and two of them have no other way to take a short curve.

    `convex` and `lp` require strictly increasing breakpoints, so they refuse
    the padding that serves `adjacency` and `sos2`.
    """
    raw = override(raw_of(SHORT_CURVE), **{'piecewise.cost_curve.method': method})
    if method == 'lp':
        raw['piecewise']['cost_curve']['links']['op_cost'] = ['op_cost', 'bp_y', '>=']

    result = sps.solve(expanded(raw), short_curve_inputs)

    assert result.objective == pytest.approx(155.0), 'B runs at 20 on its own two points, A at 5'


def test_the_condition_is_smaller_than_padding_the_curve_out(short_curve_inputs):
    """What the condition buys: the padded breakpoint costs a weight and a binary."""
    padded = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}
    padded['bp_x'] = curve_table({**A_AND_SHORT_B['x'], ('B', 2): 20.0})
    padded['bp_y'] = curve_table({**A_AND_SHORT_B['y'], ('B', 2): 130.0})
    unmasked = raw_of(SHORT_CURVE)
    del unmasked['piecewise']['cost_curve']['where']
    del unmasked['parameters']['bp_present']

    with sps.build(expanded(raw_of(SHORT_CURVE)), short_curve_inputs) as masked_model:
        masked = masked_model.metrics()
        assert masked_model.solve('highs').objective == pytest.approx(155.0)
    with sps.build(expanded(unmasked), padded) as padded_model:
        grown = padded_model.metrics()
        assert padded_model.solve('highs').objective == pytest.approx(155.0), 'the same answer, larger'

    assert masked.columns < grown.columns, 'an absent breakpoint declares no weight and no segment binary'


def test_an_absent_breakpoint_declares_no_segment_binary(short_curve_inputs):
    """The condition is on the entries, and the binaries are half of what it saves.

    The answer alone cannot see this: a binary the condition keeps at a breakpoint no
    weight reaches is slack the solver never uses, so the objective is right
    either way and the MILP is bigger for nothing.
    """
    result = sps.solve(expanded(raw_of(SHORT_CURVE)), short_curve_inputs)

    built = {(row['generator'], row['bp']) for row in result.primal('cost_curve_seg').to_dicts()}

    assert ('B', 2) not in built, "B's curve stops at bp 1, so bp 2 has no segment to pick"
    assert ('A', 2) in built, 'A runs the whole axis'


@pytest.mark.parametrize(
    ('present', 'match'),
    [
        pytest.param(
            {('A', 0): True, ('A', 1): True, ('A', 2): True, ('B', 0): True, ('B', 1): False, ('B', 2): True},
            'must mark a consecutive run',
            id='a-gap-in-the-condition',
        ),
    ],
)
def test_a_condition_with_a_gap_in_it_is_refused(short_curve_inputs, present, match):
    """The emitted rows read the condition as a length, so a gap builds a different curve.

    The chord joins a breakpoint to the one before it and the upper domain row
    is written where the condition stops; across a gap both are wrong, and neither
    is wrong in a way the answer shows.

    Both curves are supplied whole, so the shape of the condition is the only thing
    left to refuse: a curve short of a row the condition admits is the other
    refusal, and it would fire first.
    """
    whole = {
        'bp_x': {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0, ('B', 2): 30.0},
        'bp_y': {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0, ('B', 2): 200.0},
    }
    data = {**short_curve_inputs, **{n: curve_table(v) for n, v in whole.items()}, 'bp_present': curve_table(present)}
    schema = schema_of(raw_of(SHORT_CURVE))

    with pytest.raises(DataError, match=match):
        tidy_sources(schema.expand('piecewise').program, data)


def test_a_condition_that_marks_no_breakpoint_leaves_no_curve(short_curve_inputs):
    """A curve exists where its `where:` admits a breakpoint, so a condition of none is no curve rather than a gap.

    B's condition is false at every breakpoint, so B has no weights and no row ties
    its cost to a curve: B runs at its bound of 20 for nothing, and A covers
    the remaining 5 on its own curve, for 25.
    """
    nothing = {(g, k): g == 'A' for g in ('A', 'B') for k in range(3)}
    whole = {
        'bp_x': {('A', 0): 0.0, ('A', 1): 10.0, ('A', 2): 20.0, ('B', 0): 10.0, ('B', 1): 20.0, ('B', 2): 30.0},
        'bp_y': {('A', 0): 0.0, ('A', 1): 50.0, ('A', 2): 140.0, ('B', 0): 100.0, ('B', 1): 130.0, ('B', 2): 200.0},
    }
    data = {**short_curve_inputs, **{n: curve_table(v) for n, v in whole.items()}, 'bp_present': curve_table(nothing)}

    result = sps.solve(expanded(raw_of(SHORT_CURVE)), data)

    assert {row['generator'] for row in result.primal('cost_curve_lam').to_dicts()} == {'A'}, (
        'B declares no weight at all'
    )
    assert result.objective == pytest.approx(25.0), 'B is free of any curve, and A covers the rest on its own'


def test_values_missing_where_the_condition_says_present_are_still_refused(short_curve_inputs):
    """#1105's guard follows the condition rather than the whole product."""
    thin = {k: v for k, v in A_AND_SHORT_B['x'].items() if k != ('A', 2)}
    data = {**short_curve_inputs, 'bp_x': curve_table(thin)}
    schema = schema_of(raw_of(SHORT_CURVE))

    with pytest.raises(DataError, match='every breakpoint the curve runs through needs a row'):
        tidy_sources(schema.expand('piecewise').program, data)


def test_the_hole_message_offers_the_condition_to_a_block_that_has_none(short_curve_inputs):
    """A ragged curve meets this message first, so it is where a `where:` over `bp` is discovered.

    With a condition already declared the same advice would be wrong — the reader
    said how far the curve runs and the values disagree — so the way out is
    named against what the block has.
    """
    unmasked = raw_of(SHORT_CURVE)
    del unmasked['piecewise']['cost_curve']['where'], unmasked['parameters']['bp_present']
    ragged = {k: v for k, v in short_curve_inputs.items() if k != 'bp_present'}
    without = schema_of(unmasked)

    with pytest.raises(DataError, match='declare a where: that sets how far the curve runs'):
        tidy_sources(without.expand('piecewise').program, ragged)

    thin = {k: v for k, v in A_AND_SHORT_B['x'].items() if k != ('A', 2)}
    masked = schema_of(raw_of(SHORT_CURVE))
    with pytest.raises(DataError, match=r"narrow where: 'bp_present' to where the curve runs"):
        tidy_sources(masked.expand('piecewise').program, {**short_curve_inputs, 'bp_x': curve_table(thin)})


#: One curve over ``bp``, its breakpoints handed over as bare sequences —
#: dense against the labels they spread over, and carrying no coordinates of
#: their own for a length to be read from.
_ONE_DIM_CURVE = {
    'snapshot': [0],
    'bp': [0, 1, 2],
    'load': pl.DataFrame({'snapshot': [0], 'value': [25.0]}),
    'bp_x': [0.0, 10.0, 40.0],
    'bp_y': [0.0, 50.0, 140.0],
}


def _nominated_condition_spec():
    """`NONCONVEX_YAML` with its length named as one of its own breakpoints.

    `where: bp_x` narrows the weights to where `bp_x` has a row, which is what
    a bare parameter name means in any `where`.
    """
    return override(
        raw_of(NONCONVEX_YAML),
        **{
            'piecewise.cost_curve.where': 'bp_x',
            'parameters.bp_x.missing': 'absent',
            'parameters.bp_y.missing': 'absent',
        },
    )


def test_a_curve_narrowed_by_its_own_breakpoints_asks_for_nothing_extra():
    """A `where:` naming a values parameter declares no second name to attach.

    The block narrows the weights by a parameter the file already wrote, so what
    the caller attaches is exactly what the file declares and the curve still
    solves.
    """
    program = expanded(_nominated_condition_spec(), 'piecewise').program

    assert sorted(attachable(program)) == ['bp', 'bp_x', 'bp_y', 'load', 'snapshot'], (
        "the file's own parameters and dimensions, and nothing a block invented"
    )
    assert sorted(tidy_sources(program, _ONE_DIM_CURVE)) == ['bp', 'bp_x', 'bp_y', 'load', 'snapshot'], (
        'and `tidy_sources` gives back one table per name it takes'
    )
    assert sps.solve(expanded(_nominated_condition_spec()), _ONE_DIM_CURVE).objective == pytest.approx(95.0), (
        'and the data still attaches and the curve solves, narrowed to the rows of its own breakpoints'
    )


def test_values_the_condition_leaves_out_are_left_alone(short_curve_inputs):
    """A table wider than the block uses is ordinary, not an error."""
    spare = {**A_AND_SHORT_B['x'], ('B', 2): 999.0}
    data = {**short_curve_inputs, 'bp_x': curve_table(spare)}

    assert sps.solve(expanded(raw_of(SHORT_CURVE)), data).objective == pytest.approx(155.0), (
        'the row the condition leaves out is not read'
    )


@pytest.mark.parametrize('method', ['adjacency', 'sos2'])
def test_a_gate_that_does_not_exist_leaves_the_curve_ungated(nonconvex_inputs, method):
    """Where the gate does not exist the block is ungated, so the weights sum
    to 1 — what a block with no `activity:` at all gets (#1158).

    The convexity row is ``sum(lam, over=bp) == (activity)`` and absence does
    not spread out of a reduction, so an absent gate that took the *row* with it
    would relax the curve silently: 900.00 where the answer is 1050.00, on a
    curve with a no-load intercept.
    """
    raw = raw_of(GATED_YAML)
    raw['piecewise']['cost_curve']['method'] = method
    raw['parameters']['gate_rows'] = {'dims': ['snapshot'], 'dtype': 'bool', 'missing': False}
    raw['variables']['u'] = {'dims': ['snapshot'], 'domain': 'binary', 'where': 'gate_rows'}

    gated = [True, False] * 6
    data = {
        **nonconvex_inputs,
        'on_flag': pd.Series([0.0 if g else 1.0 for g in gated], index=pd.RangeIndex(12, name='snapshot')),
        'gate_rows': pd.Series(gated, index=pd.RangeIndex(12, name='snapshot')),
    }
    model = sps.build(expanded(raw), data)
    omitted = {c for c in model.diagnostics().omissions['constraint'] if c.startswith('cost_curve')}
    assert not omitted, 'every coordinate gets a convexity row — gated by the variable, or ungated at 1'

    result = model.solve()
    cost = primal_by_coordinate(result, 'op_cost', 'snapshot')
    for s, is_gated in enumerate(gated):
        expected = 0.0 if is_gated else curve(data['load'][s], data['bp_x'], data['bp_y'])
        assert cost[s] == pytest.approx(expected, abs=1e-6), (
            'the gate decides where it exists; where it does not, the curve holds unconditionally'
        )


def test_a_narrowed_gate_declaring_its_absence_pins_the_curve_off(nonconvex_inputs):
    """`missing: 0` is the spelling the refusal above asks for, and it holds
    the formulation together: the row is built everywhere, reading
    ``sum(lam) == 0`` where the gate does not exist."""
    raw = raw_of(GATED_YAML)
    raw['parameters']['gate_rows'] = {'dims': ['snapshot'], 'dtype': 'bool', 'missing': False}
    raw['variables']['u'] = {
        'dims': ['snapshot'],
        'domain': 'binary',
        'where': 'gate_rows',
        'missing': 0,
    }

    rows = dict(expanded(raw, 'piecewise').program.constraints)
    assert 'cost_curve_convexity_ungated' not in rows, (
        'a gate that says its absence is zero wants one row, not the ungated half of a pair'
    )
    assert rows['cost_curve_convexity'].where is None, (
        'and that row has no condition — the gate reads 0 where it does not exist, so the row still builds'
    )

    gated = [True, False] * 6
    data = {
        **nonconvex_inputs,
        'on_flag': pd.Series([float(g) for g in gated], index=pd.RangeIndex(12, name='snapshot')),
        'gate_rows': pd.Series(gated, index=pd.RangeIndex(12, name='snapshot')),
    }
    model = sps.build(expanded(raw), data)
    omitted = set(model.diagnostics().omissions['constraint'])
    assert not [c for c in omitted if c.startswith('cost_curve')], (
        'a gate that says what its absence means leaves every row the block emits standing'
    )

    result = model.solve()
    cost = primal_by_coordinate(result, 'op_cost', 'snapshot')
    for s, on in enumerate(gated):
        expected = curve(data['load'][s], data['bp_x'], data['bp_y']) if on else 0.0
        assert cost[s] == pytest.approx(expected, abs=1e-6), 'the curve is off where the gate does not exist'
