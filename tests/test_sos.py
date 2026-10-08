"""``sos:`` — one construct, three sinks, and one of them without the concept.

`differential` checks that the language means the same thing on both lanes.
That a set **written out** (``Spec.expand()``) is the same feasible set as one a
sink branches on is checked against an optimum enumerated here (:func:`best`),
since both lanes read the expansion. Two sites choose among four sizes, and
SOS1, SOS2 and the unrestricted LP all have *different* optima.
"""

from __future__ import annotations

import itertools
from typing import Any

import polars as pl
import pytest
from mathspec import to_spec

import specsolve as sps
from specsolve.errors import LanguageError, SpecsolveError
from tests.conftest import EXAMPLES_DIR, expanded, solve_written_file

SOS_YAML = EXAMPLES_DIR / 'sos.yaml'

SITES = ['north', 'south']
SIZES = [0, 1, 2, 3]

#: Value per unit and how much of it there is, by ``(site, size)``. Chosen so
#: the three regimes separate: unrestricted takes everything (39), SOS2 takes
#: an adjacent pair (26), SOS1 takes one member (14).
VALUE = {
    ('north', 0): 1.0, ('north', 1): 3.0, ('north', 2): 2.0, ('north', 3): 2.5,
    ('south', 0): 5.0, ('south', 1): 1.0, ('south', 2): 4.0, ('south', 3): 0.5,
}  # fmt: skip
CAP = {
    ('north', 0): 4.0, ('north', 1): 2.0, ('north', 2): 3.0, ('north', 3): 1.0,
    ('south', 0): 1.0, ('south', 1): 6.0, ('south', 2): 2.0, ('south', 3): 3.0,
}  # fmt: skip


def _table(values: dict[tuple[str, int], float]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            'site': [site for site, _ in values],
            'size': [size for _, size in values],
            'value': list(values.values()),
        }
    )


DATA = {'site': SITES, 'size': SIZES, 'value': _table(VALUE), 'cap': _table(CAP)}

BASE: dict[str, Any] = {
    'dimensions': {'site': {'dtype': 'str'}, 'size': {'dtype': 'int'}},
    'parameters': {'value': {'dims': ['site', 'size']}, 'cap': {'dims': ['site', 'size']}},
    'variables': {'take': {'dims': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(sum(take * value, over=site), over=size)'},
}


def spec(sos_type: int) -> dict[str, Any]:
    """The base model with one set over ``take``'s ``size`` dim."""
    return BASE | {'sos': {'pick': {'variable': 'take', 'along': 'size', 'type': sos_type}}}


def best(sos_type: int | None, sizes: list[int] = SIZES) -> float:
    """The optimum by enumeration — the oracle neither lane can be.

    Every set is independent (nothing couples the sites), and a set admits a
    known family of nonzero patterns: any single member for SOS1, plus any
    consecutive pair for SOS2. Taking each admitted member at its cap is
    optimal because every value is positive.
    """
    if sos_type is None:
        patterns = [tuple(sizes)]
    else:
        patterns = [(size,) for size in sizes]
        if sos_type == 2:
            patterns += list(itertools.pairwise(sizes))
    return sum(
        max(sum(VALUE[site, size] * CAP[site, size] for size in pattern) for pattern in patterns) for site in SITES
    )


def test_the_three_regimes_have_different_optima():
    """The premise every assertion below rests on, stated as a test.

    A set that does not bind is satisfied by ignoring it, so a model where the
    LP optimum already respects the set would pass every test here with the
    reformulation deleted.
    """
    assert (best(None), best(2), best(1)) == (39.0, 26.0, 14.0), 'the enumeration no longer separates the regimes'


# ---------------------------------------------------------------------------
# the language
# ---------------------------------------------------------------------------


#: A dim the model declares and ``take`` does not carry, so a set over it has
#: no members — the one case needing a declaration the base model lacks.
UNCARRIED = {'dimensions': BASE['dimensions'] | {'other': {'dtype': 'str'}}}

#: A member the set cannot hold at zero from one side, because the model
#: declares no coefficient there. Each replaces ``take``'s bounds block.
OPEN_ABOVE = {'variables': {'take': {'dims': ['site', 'size'], 'bounds': {'lower': 0}}}}
OPEN_BELOW = {'variables': {'take': {'dims': ['site', 'size'], 'bounds': {'upper': 'cap'}}}}


@pytest.mark.parametrize(
    ('blocks', 'expected', 'also'),
    [
        pytest.param(
            {'pick': {'variable': 'nope', 'along': 'size'}}, "'nope' is not a declared variable", {}, id='no-var'
        ),
        pytest.param({'pick': {'variable': 'take', 'along': 'site2'}}, "undeclared dimension 'site2'", {}, id='no-dim'),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'other'}},
            "along 'other' is not a dim of variable 'take'",
            UNCARRIED,
            id='a-dim-the-variable-does-not-carry',
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'size'}, 'again': {'variable': 'take', 'along': 'size', 'type': 2}},
            "already carries the set declared by 'pick'",
            {},
            id='two-sets-over-one-variable',
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'size', 'type': 3}}, 'sos type must be 1 or 2', {}, id='third-order'
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'size'}}, 'has no upper bound', OPEN_ABOVE, id='open-above'
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'size'}}, 'has no lower bound', OPEN_BELOW, id='open-below'
        ),
        pytest.param(
            {'pick': {'variable': 'take', 'along': 'size', 'kind': 1}}, "unknown key 'kind'", {}, id='unknown-key'
        ),
    ],
)
def test_a_malformed_set_is_a_load_error_naming_it(blocks, expected, also):
    """Everything decidable without data, in one table.

    ``type: 1`` is filled in where the case is not about it, so each row shows
    only what it is testing.
    """
    filled = {name: {'type': 1} | block for name, block in blocks.items()}
    with pytest.raises(LanguageError, match=expected):
        sps.check(BASE | also | {'sos': filled})


# ---------------------------------------------------------------------------
# what the sinks do with it
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_both_lanes_and_the_enumeration_agree(sos_type):
    """The differential claim, plus the one oracle that is nobody's code.

    The harness is imported **inside the test**: importing it is the
    oracle's guard, and the rest of this module runs on the bare install.
    """
    from tests.differential import differential

    linopy_lane = {'site': SITES, 'size': SIZES} | {
        name: _table(v).to_pandas() for name, v in (('value', VALUE), ('cap', CAP))
    }
    with differential(spec(sos_type), linopy_lane, lp=True) as run:
        assert run.result.objective == pytest.approx(best(sos_type)), 'the set does not restrict what it claims to'


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_written_out_solution_is_a_member_of_the_set(sos_type):
    """An optimum can be right while the formulation admits shapes it should
    not, so the pattern itself is checked."""
    result = sps.solve(expanded(spec(sos_type)), DATA)
    taken = result.primal('take')
    for site in SITES:
        nonzero = taken.filter((pl.col('site') == site) & (pl.col('value') > 1e-9))['size'].to_list()
        assert len(nonzero) <= sos_type, f'{site} has {len(nonzero)} nonzero members in an SOS{sos_type} set'
        if sos_type == 2 and len(nonzero) == 2:
            assert nonzero[1] - nonzero[0] == 1, f'{site} took two members that are not consecutive'


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_native_sink_reaches_the_same_optimum(sos_type):
    """Gurobi branches on the set; HiGHS is handed the set written out. One answer."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    assert sps.solve(spec(sos_type), DATA, 'gurobi').objective == pytest.approx(best(sos_type))


@pytest.mark.parametrize('sos_type', [1, 2], ids=['sos1', 'sos2'])
def test_the_lp_file_carries_the_set_and_a_reader_agrees(sos_type, tmp_path):
    """The section, in label order, and read back by a solver that takes it."""
    path = sps.write(spec(sos_type), DATA, tmp_path / 'model.lp')
    text = path.read_text()

    assert '\nsos\n' in text, 'the LP file dropped the set entirely'
    section = text.split('\nsos\n')[1].split('\nend')[0].strip().splitlines()
    assert section == [
        f's0: S{sos_type} :: x0:1 x1:2 x2:3 x3:4',
        f's1: S{sos_type} :: x4:1 x5:2 x6:3 x7:4',
    ], 'the section is not the sets in label order, weighted by declared position'

    gurobipy = pytest.importorskip('gurobipy', reason='no shipped solver but gurobi reads an SOS section')
    with gurobipy.Env(params={'OutputFlag': 0}) as env:
        read = gurobipy.read(str(path), env=env)
        read.optimize()
        assert read.ObjVal == pytest.approx(best(sos_type)), 'the written file is not the model that was built'
        read.dispose()


def test_highs_refuses_the_written_section_which_is_why_a_set_is_written_out_for_it(tmp_path):
    """The capability finding itself, pinned rather than described.

    If HiGHS ever grows an SOS concept this fails, and its capability
    descriptor should then list ``'sos'``.
    """
    path = sps.write(spec(1), DATA, tmp_path / 'model.lp')
    with pytest.raises(AssertionError):
        solve_written_file(path)


# ---------------------------------------------------------------------------
# a sink with no concept of a set
# ---------------------------------------------------------------------------


def test_a_set_on_a_sink_with_no_concept_is_refused_naming_the_expansion():
    """Nothing is rewritten at the hand-off: the language writes a set out, and the refusal says so."""
    with pytest.raises(SpecsolveError, match='special-ordered sets') as refusal:
        sps.solve(spec(1), DATA)
    assert 'expand()' in str(refusal.value) and 'gurobi' in str(refusal.value), (
        'the way past the refusal is the expansion, or a sink that has the concept'
    )
    assert sps.solve(expanded(spec(1)), DATA).objective == pytest.approx(best(1)), (
        'and the set written out restricts exactly what the set restricts'
    )


def test_a_member_that_may_go_negative_is_held_at_zero_from_below():
    """The expansion links each member to its binary from both sides.

    With every member allowed down to -1, an unpicked one still sits at zero
    rather than dropping below it, so the optimum is the one the set alone
    admits.
    """
    raw = spec(1)
    raw['variables'] = {'take': {'dims': ['site', 'size'], 'bounds': {'lower': -1, 'upper': 'cap'}}}
    assert sps.solve(expanded(raw), DATA).objective == pytest.approx(best(1))


# ---------------------------------------------------------------------------
# what a set does to the rest of the model
# ---------------------------------------------------------------------------


def test_a_masked_member_leaves_the_set_and_its_neighbours_adjacent():
    """Membership is the variable's own, so a mask closes the gap it leaves.

    With size 1 masked out at every site the members that exist are 0, 2 and
    3. A weight orders the members and nothing more, so a sink branching on
    the set reads ``{0, 2}`` as consecutive in the list it was handed, which
    the unmasked model refuses.
    """
    raw = spec(2)
    raw['variables'] = {
        'take': {'dims': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'where': 'size != 1'}
    }
    with sps.build(raw, DATA) as model:
        sets = model._engine._model.handoff.sos
    assert sets['weight'].to_list() == [1, 3, 4] * 2, 'the masked member is still in the list, or the order moved'


def test_a_set_written_out_is_integrality_and_the_dual_message_names_it():
    """Written out, a set is binaries the file never declared, and the ordinary message names them."""
    result = sps.solve(expanded(spec(1)), DATA)
    with pytest.raises(SpecsolveError, match="mixed-integer model: 'pick_seg'"):
        result.dual('pick_pick')


def test_a_sos2_set_of_one_member_restricts_nothing():
    """A set with one member has no segment, so it restricts nothing.

    The member is left alone rather than linked to a neighbour that does not
    exist — which would pin the one member of the set to zero and quietly
    delete it from the model.

    Masked down per site, so one set keeps three members and the other has
    one: a set that is dropped whole must also not shift the rows the sets
    after it own.
    """
    raw = spec(2)
    raw['parameters'] = raw['parameters'] | {'live': {'dims': ['site', 'size'], 'dtype': 'bool'}}
    raw['variables'] = {'take': {'dims': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'where': 'live'}}
    live = DATA | {
        'live': _table({(site, size): (site == 'north' or size == 0) for site in SITES for size in SIZES}).with_columns(
            pl.col('value').cast(pl.Boolean)
        )
    }
    north = max(
        VALUE['north', a] * CAP['north', a] + VALUE['north', b] * CAP['north', b]
        for a, b in itertools.pairwise([0, 1, 2, 3])
    )
    south = VALUE['south', 0] * CAP['south', 0]
    assert sps.solve(expanded(raw), live).objective == pytest.approx(north + south), (
        'the lone member was pinned to zero'
    )


def test_regrouping_the_members_is_a_different_model_to_a_loaded_solver():
    """A set is structure, and it is the one structure nothing else reports.

    Two builds differ in *nothing a solver was handed* — two columns, the same
    bounds, no rows, no matrix — except which sets those columns fall into:
    two members of one set against one member each of two. A digest that did
    not read the sets would call the second the model it already holds.
    """
    raw = {
        'dimensions': {'site': {'dtype': 'str'}, 'size': {'dtype': 'int'}},
        'parameters': {'worth': {'dims': ['site', 'size']}, 'live': {'dims': ['site', 'size'], 'dtype': 'bool'}},
        'variables': {'take': {'dims': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 1}, 'where': 'live'}},
        'sos': {'pick': {'variable': 'take', 'along': 'size', 'type': 1}},
        'objective': {'sense': 'maximize', 'expression': 'sum(sum(take * worth, over=site), over=size)'},
    }
    worth = _table({('north', 0): 3.0, ('north', 1): 5.0, ('south', 0): 5.0, ('south', 1): 0.0})
    together = {'north': [True, True], 'south': [False, False]}
    apart = {'north': [True, False], 'south': [True, False]}

    def live(mask: dict[str, list[bool]]) -> dict[str, Any]:
        return {
            'site': SITES,
            'size': [0, 1],
            'worth': worth,
            'live': _table({(s, k): mask[s][k] for s in SITES for k in (0, 1)}),
        }

    with sps.build(raw, live(together)) as model:
        one_set = model._engine._model.handoff
        model.update(live(apart))
        two_sets = model._engine._model.handoff
        assert (one_set.cols.equals(two_sets.cols), one_set.column_count, one_set.row_count) == (True, 2, 0), (
            'the two builds differ in something other than their sets, so this proves nothing'
        )
        assert two_sets.structure != one_set.structure, 'the digest calls a regrouped set the same model'


def test_a_set_that_runs_along_a_leading_dim_still_arrives_grouped():
    """The one shape where the stream is not already in ``(set, weight)`` order.

    With ``over`` first in the ``dims``, a set's members are ``|site|``
    apart in label order, so every set interleaves with every other. The sinks
    read a set's edges off neighbouring rows, so an ungrouped stream links
    members to the wrong binaries — and still reaches a plausible optimum.
    """
    raw = spec(1)
    raw['variables'] = {'take': {'dims': ['size', 'site'], 'bounds': {'lower': 0, 'upper': 'cap'}}}
    with sps.build(raw, DATA) as model:
        sets = model._engine._model.handoff.sos
        assert sets['set'].to_list() == [0, 0, 0, 0, 1, 1, 1, 1], 'the members of a set did not end up together'
        assert sets['weight'].to_list() == [1, 2, 3, 4] * 2, 'a set is not in weight order'
        assert sets['col'].to_list() == [0, 2, 4, 6, 1, 3, 5, 7], 'a member is not the column its coordinate got'


@pytest.mark.parametrize('dims', [['site', 'size'], ['size', 'site']], ids=['over-last', 'over-first'])
def test_a_mask_that_drops_nothing_places_the_sets_where_the_arithmetic_does(dims: list[str]):
    """A label is a position where nothing was dropped, and both paths say so.

    An unmasked variable's sets and weights are arithmetic on the label, which
    *is* the coordinate's position in the declared product. A masked one has
    to read the coordinates, a dropped row being a label that no longer
    decomposes. A mask that drops nothing sends the same model down both
    paths, so either the two frames are one frame or one of them is wrong —
    and a member in the wrong set is linked to another set's binaries.

    Both orders of the ``dims``, because the split is where ``over`` sits
    in it: last leaves the sets contiguous, first interleaves them.
    """
    take = {'dims': dims, 'bounds': {'lower': 0, 'upper': 'cap'}}
    raw = spec(2) | {'variables': {'take': take}}
    masked = raw | {
        'parameters': raw['parameters'] | {'live': {'dims': ['site', 'size'], 'dtype': 'bool'}},
        'variables': {'take': take | {'where': 'live'}},
    }
    live = _table(dict.fromkeys(((site, size) for site in SITES for size in SIZES), 1.0)).with_columns(
        pl.col('value').cast(pl.Boolean)
    )
    with sps.build(raw, DATA) as placed, sps.build(masked, DATA | {'live': live}) as counted:
        assert placed._engine._model.handoff.sos.equals(counted._engine._model.handoff.sos), (
            'the two placements disagree about which coordinate is in which set, or at which weight'
        )


def test_a_mask_that_empties_a_set_leaves_the_numbering_dense():
    """A set number is a dense index, the way a column's and a row's are.

    A coordinate's *position* is not: it counts the product the mask never
    materialised, which is what may pass 2^31 while every survivor fits
    (``engine._DTYPES``). So a set is renumbered wherever a row was dropped,
    and what reaches a sink counts the sets that exist rather than the ones
    the declaration would have had.

    The mask empties the *first* set here, which is the one arrangement where
    a hole and no hole are different numbers.
    """
    take = {'dims': ['site', 'size'], 'bounds': {'lower': 0, 'upper': 'cap'}, 'where': 'live'}
    raw = spec(1) | {'variables': {'take': take}}
    raw['parameters'] = raw['parameters'] | {'live': {'dims': ['site', 'size'], 'dtype': 'bool'}}
    live = _table({(site, size): float(site == 'south') for site in SITES for size in SIZES}).with_columns(
        pl.col('value').cast(pl.Boolean)
    )
    with sps.build(raw, DATA | {'live': live}) as model:
        assert model._engine._model.handoff.sos['set'].to_list() == [0, 0, 0, 0], (
            'the emptied set left a hole, so a set number is a position rather than an index'
        )


# examples/sos.yaml — `method: sos2`, the curve said as a set
# ---------------------------------------------------------------------------

#: A *concave* curve per generator — economies of scale. The hull's lower
#: envelope is the chord, which undercuts a concave curve, so the set is
#: load-bearing on this instance rather than decoration.
CURVE_X = {'cheap': [0.0, 40.0, 100.0], 'mid': [0.0, 60.0, 120.0]}
CURVE_Y = {'cheap': [0.0, 30.0, 55.0], 'mid': [0.0, 50.0, 85.0]}
LOADS = [30.0, 95.0, 140.0, 190.0]


def _curve_sources() -> dict[str, pl.DataFrame]:
    breakpoints = [{'generator': g, 'bp': k, 'value': v} for g, values in CURVE_X.items() for k, v in enumerate(values)]
    costs = [{'generator': g, 'bp': k, 'value': v} for g, values in CURVE_Y.items() for k, v in enumerate(values)]
    return {
        'generator': pl.DataFrame({'generator': list(CURVE_X)}),
        'snapshot': pl.DataFrame({'snapshot': range(len(LOADS))}),
        'bp': pl.DataFrame({'bp': range(len(next(iter(CURVE_X.values()))))}),
        'p_max': pl.DataFrame({'generator': list(CURVE_X), 'value': [xs[-1] for xs in CURVE_X.values()]}),
        'load': pl.DataFrame({'snapshot': range(len(LOADS)), 'value': LOADS}),
        'bp_x': pl.DataFrame(breakpoints),
        'bp_y': pl.DataFrame(costs),
    }


def _interpolated(generator: str, dispatched: float) -> float:
    """The curve's own cost at *dispatched* — one linear piece, read by hand."""
    xs, ys = CURVE_X[generator], CURVE_Y[generator]
    for left in range(len(xs) - 1):
        if xs[left] - 1e-9 <= dispatched <= xs[left + 1] + 1e-9:
            share = (dispatched - xs[left]) / (xs[left + 1] - xs[left])
            return ys[left] + share * (ys[left + 1] - ys[left])
    raise AssertionError(f'{generator} dispatched {dispatched}, which is off its curve')


def test_the_example_prices_on_the_curve_and_not_on_its_hull():
    """``examples/sos.yaml``: the claim the whole model exists to make.

    Every generator's cost has to sit on its *own* curve at the level it was
    dispatched to — which is what "at most two, and neighbours" buys. Without
    the set the same LP prices the chord, so the objective is asserted to be
    strictly worse than that relaxation as well: a formulation that quietly
    stopped restricting anything would match the curve nowhere.
    """
    sources = _curve_sources()
    result = sps.solve(expanded(SOS_YAML), sources)
    assert result.is_ok

    dispatched = result.primal('p')
    cost = result.primal('op_cost')
    on_curve = 0.0
    for row, priced in zip(dispatched.iter_rows(named=True), cost.iter_rows(named=True), strict=True):
        expected = _interpolated(row['generator'], row['value'])
        assert priced['value'] == pytest.approx(expected, abs=1e-6), (
            f'{row["generator"]} at t={row["snapshot"]} is priced off its curve'
        )
        on_curve += expected

    assert result.objective == pytest.approx(on_curve, abs=1e-6)
    relaxed = sps.solve(_without_the_set(), sources)
    assert relaxed.objective < result.objective - 1e-6, 'the hull costs the same, so the set restricts nothing here'


def _without_the_set() -> dict[str, Any]:
    """``examples/sos.yaml`` with the restriction dropped — the λ hull.

    Written out as the weights, convexity and link rows the ``piecewise:``
    block expands into, minus the set. ``method: convex`` cannot spell this
    relaxation: the curvature guard refuses it on a concave curve.
    """
    raw = to_spec(SOS_YAML).to_dict()
    raw.pop('piecewise')
    raw['variables']['lam'] = {'dims': ['snapshot', 'generator', 'bp'], 'bounds': {'lower': 0, 'upper': 1}}
    raw['constraints'] |= {
        'convexity': {'dims': ['snapshot', 'generator'], 'expression': 'sum(lam, over=bp) == 1'},
        'dispatch': {'dims': ['snapshot', 'generator'], 'expression': 'p == sum(lam * bp_x, over=bp)'},
        'cost': {'dims': ['snapshot', 'generator'], 'expression': 'op_cost == sum(lam * bp_y, over=bp)'},
    }
    return raw


def test_a_one_shot_solve_lets_go_of_the_sets_before_the_solver_runs(monkeypatch):
    """The solver holds the sets once loaded, so ``sps.solve`` drops the build's copy before the run."""
    pytest.importorskip('gurobipy', reason='the native SOS path needs the [gurobi] extra')
    from specsolve.relational.sinks.solvers.base import Solver

    members: list[int] = []
    run = Solver.run

    def recorded(self: Solver, handoff: Any, **options: Any) -> Any:
        members.append(handoff.sos.height)
        return run(self, handoff, **options)

    monkeypatch.setattr(Solver, 'run', recorded)
    assert sps.solve(spec(1), DATA, 'gurobi').objective == pytest.approx(best(1))
    assert members == [0], 'the run is handed no set members: the solver already holds them'
