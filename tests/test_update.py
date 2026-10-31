"""Re-solving one built model with new numbers.

**An update answers what a fresh build answers**: `build(spec, sources |
change)` is the reference for every rung below. The fast path is *only* a fast
path: an update that moves a mask renumbers labels, so the engine rebuilds and
solves cold, and `diagnostics().loads` says which happened.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import replace
from typing import Any, NamedTuple

import polars as pl
import pytest
from mathspec import to_spec
from polars.testing import assert_frame_equal

import specsolve as sps
from specsolve._sources import attachable
from tests.conftest import KNAPSACK, expanded, knapsack_sources, override, port_sources

GENERATORS = ['wind', 'solar', 'gas']
SNAPSHOTS = [0, 1, 2, 3]
COORDS = {'snapshot': SNAPSHOTS, 'generator': GENERATORS}


def sources() -> dict[str, pl.DataFrame]:
    """`examples/dispatch.yaml`'s data, small enough to read in a failure."""
    return {
        'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 200.0]}),
        'cost': pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 50.0]}),
        'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0, 80.0, 55.0, 95.0]}),
    }


#: Which plants may serve which zone, and how well. Every matrix entry of
#: `examples/dispatch.yaml` is a 1 and its objective has no constant, so this
#: model moves what that one cannot: a coefficient, an entry's column, an
#: entry's row, and the objective constant.
ZONES = ['north', 'south']
PLANTS = ['a', 'b', 'c', 'd']
REACH = {
    'dimensions': {'zone': {'dtype': 'str'}, 'plant': {'dtype': 'str'}},
    'parameters': {
        'reach': {'dims': ['zone', 'plant'], 'missing': 'neutral'},
        'cost': {'dims': ['plant']},
        'demand': {'dims': ['zone']},
        'levy': {'dims': []},
    },
    'variables': {'p': {'dims': ['plant'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {'meet': {'dims': ['zone'], 'expression': 'sum(reach * p, over=plant) >= demand'}},
    #: `levy` is the objective's **constant**: the one term with no column.
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost) + levy'},
}


def reaching(*served: tuple[str, str, float]) -> pl.DataFrame:
    """A `reach` frame. An absent row is a zero coefficient (the data-attachment rules), so it drops the entry."""
    return pl.DataFrame(
        {'zone': [z for z, _, _ in served], 'plant': [p for _, p, _ in served], 'value': [v for _, _, v in served]},
        schema={'zone': pl.String, 'plant': pl.String, 'value': pl.Float64},
    )


def reach_sources() -> dict[str, pl.DataFrame]:
    """North reached by the two cheap plants, south by the two dear ones."""
    return {
        'zone': pl.DataFrame({'zone': ZONES}),
        'plant': pl.DataFrame({'plant': PLANTS}),
        'reach': reaching(('north', 'a', 1.0), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0)),
        'cost': pl.DataFrame({'plant': PLANTS, 'value': [1.0, 2.0, 3.0, 4.0]}),
        'demand': pl.DataFrame({'zone': ZONES, 'value': [60.0, 30.0]}),
        'levy': pl.DataFrame({'value': [5.0]}),
    }


class Rung(NamedTuple):
    """One row of the update table: which model, what changes, what may be kept."""

    model: str
    change: dict[str, pl.DataFrame]
    keeps_the_solver: bool


def _case(rung: Rung, dispatch_yaml: Any) -> tuple[Any, dict[str, Any]]:
    """The rung's model and the sources that build it, index included."""
    return {
        'dispatch': lambda: (dispatch_yaml, sources() | COORDS),
        'reach': lambda: (REACH, reach_sources()),
        'knapsack': lambda: (KNAPSACK, knapsack_sources()),
    }[rung.model]()


#: Each rung of the update contract (``Model.update``): the model it moves, what
#: changes, and whether the loaded solver may be kept. `p_max` appears twice: it
#: gates ``where: p_max > 0`` *and* bounds the variable, so whether it is
#: structural depends on its values.
#:
#: The `reach` rungs move the objective constant, a coefficient, an entry's
#: column and an entry's row, and each changes the answer. A coefficient at
#: 0.8 leaves `a` alone cheapest in the north: at 0.5 it would tie `b`, and a
#: warm solve and a cold one may stop at either optimum.
RUNGS = [
    pytest.param(
        Rung('dispatch', {'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})}, True),
        id='rhs',
    ),
    pytest.param(
        Rung('dispatch', {'cost': pl.DataFrame({'generator': GENERATORS, 'value': [9.0, 2.0, 1.0]})}, True),
        id='objective',
    ),
    pytest.param(
        Rung('dispatch', {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [80.0, 70.0, 90.0]})}, True),
        id='bounds',
    ),
    pytest.param(
        Rung('dispatch', {'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}, False),
        id='mask',
    ),
    pytest.param(Rung('reach', {'levy': pl.DataFrame({'value': [500.0]})}, True), id='objective constant'),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'a', 0.8), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
            True,
        ),
        id='a coefficient moved',
    ),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'c', 1.0), ('north', 'd', 1.0), ('south', 'a', 1.0), ('south', 'b', 1.0))},
            False,
        ),
        id='an entry changed column',
    ),
    pytest.param(
        Rung(
            'reach',
            {'reach': reaching(('north', 'a', 1.0), ('south', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
            False,
        ),
        id='an entry changed row',
    ),
    pytest.param(Rung('knapsack', {'capacity': pl.DataFrame({'value': [9.0]})}, True), id='integer'),
]


@pytest.fixture
def model(dispatch_yaml):
    """The example dispatch on its own data, built and open for the test's duration."""
    with sps.build(dispatch_yaml, sources() | COORDS) as model:
        yield model


def _priced(program: Any) -> list[str]:
    """The constraints an answer carries prices for — none, where a variable is discrete."""
    if any(v.domain != 'continuous' for v in program.variables.values()):
        return []
    return list(program.constraints)


@pytest.mark.parametrize('rung', RUNGS)
def test_a_update_answers_what_a_fresh_build_answers(dispatch_yaml, rung, solver_name):
    """The oracle. Every rung, one assertion: the reference build is the truth.

    Read-back is keyed by coordinate, so this holds even where the rung moved
    every label. Over **every** declaration and **every** sink that can stay
    loaded, since each sink writes its own push.
    """
    spec, given = _case(rung, dispatch_yaml)
    program = to_spec(spec).program
    with (
        sps.solve(spec, {**given, **rung.change}, solver_name=solver_name) as reference,
        sps.build(spec, given) as model,
    ):
        model.solve(solver_name=solver_name)
        updated = model.update(rung.change).solve(solver_name=solver_name)

        assert updated.objective == pytest.approx(reference.objective), 'the update reached a different optimum'
        for name in program.variables:
            assert updated.primal(name).equals(reference.primal(name)), f"'{name}' came back laid out differently"
        for name in _priced(program):
            assert updated.dual(name).equals(reference.dual(name)), f"'{name}' came back priced differently"


@pytest.mark.parametrize('rung', [rung for rung in RUNGS if rung.values[0].keeps_the_solver])
def test_an_update_and_its_undoing_answer_on_one_loaded_solver(dispatch_yaml, rung, solver_name):
    """There and back: the second update restores the numbers the model was loaded with.

    A sink that sends only what moved compares against what it holds, so one
    that forgot the first push would find nothing to send on the way back.
    """
    spec, given = _case(rung, dispatch_yaml)
    variables = list(to_spec(spec).program.variables)

    def answer(result: Any) -> tuple[float, list[list[tuple[Any, ...]]]]:
        return result.objective, [result.primal(name).rows() for name in variables]

    with sps.build(spec, given) as model:
        first = answer(model.solve(solver_name=solver_name))
        moved = answer(model.update(rung.change).solve(solver_name=solver_name))
        back = answer(model.update({name: given[name] for name in rung.change}).solve(solver_name=solver_name))

        assert moved != first, 'the rung has to move the answer, or the way back proves nothing'
        assert back == first, 'the solver kept a number the second update took back'
        assert model.diagnostics().loads == 1, 'each update moved only numbers, so the solver stayed loaded'


# ---------------------------------------------------------------------------
# the same oracle, over models nobody here wrote
# ---------------------------------------------------------------------------

#: One walk over a port's own data. **1.0 first** pins determinism: two builds
#: of one model have to hash alike or no driver takes the fast path. Then two
#: scalings, each updated off the state the last one left.
WALK = [1.0, 1.25, 0.8]

#: `tsp_mtz` walks nowhere: gr17's branch-and-bound is seconds a solve and a
#: walk takes three of them per model, which is a third of the suite's runtime
#: for one port. The other three discrete ports reach the same paths in a tenth
#: of the time.
TOO_SLOW_TO_WALK = {'tsp_mtz'}

#: `transport_modes` prices two of its eleven connections at 12 — `d1_c1_road`
#: and `d2_c2_rail` — so once the walk scales the stocks the optimum is reached
#: at more than one vertex, and which one a solve lands on is a simplex route
#: rather than an answer, so its primal values are not compared. Book data, so
#: the tie is not ours to perturb away.
ALTERNATE_OPTIMA = {'transport_modes'}

#: Constraints whose prices the walk compares for layout but not for numbers.
#: An investment optimum sits on a kink of the piecewise-linear value of
#: capacity — the marginal MW is worth more than capex on one side of a load
#: level and less on the other — so how the capacity rent splits across the
#: snapshots binding there is a free dual ray, and a warm-started re-solve
#: legitimately lands on a different split than a cold one. The objective, the
#: layouts and every other price (`balance`, whose degeneracy would move the
#: objective) stay exact.
NONUNIQUE_PRICES: dict[str, set[str]] = {'multi_period': {'within_cap'}}


def _declared(given: dict[str, Any], program: Any) -> dict[str, Any]:
    """*given* less the names the model never declares, which `update` refuses and `build` ignores."""
    return {name: value for name, value in given.items() if name in attachable(program)}


def _scaled(given: dict[str, Any], by: float) -> dict[str, Any]:
    """*given* with every dimensioned real-valued table scaled, and nothing else.

    Dimensioned, because a **scalar** in this corpus is as often a formulation
    constant as a quantity — `tsp_mtz`'s ``n`` is the Miller-Tucker-Zemlin
    bound — and scaling one of those writes a different *model* rather than
    different numbers for the same one. Real-valued for the same reason: an
    integer index or a label is structure.
    """
    scalable = pl.col('value') * by
    return {
        name: value.with_columns(scalable)
        if isinstance(value, pl.DataFrame) and value.schema.get('value') == pl.Float64 and len(value.columns) > 1
        else value
        for name, value in given.items()
    }


def _prices(result: Any, program: Any) -> dict[str, pl.DataFrame] | None:
    """Every constraint's prices, or ``None`` where this answer carries none."""
    try:
        return {name: result.dual(name) for name in program.constraints}
    except sps.errors.SpecsolveError:
        return None


def _laid_out_alike(got: pl.DataFrame, want: pl.DataFrame, *, values: bool, where: str) -> None:
    """*got* is *want*'s frame: same coordinates in the same order, same numbers."""
    assert got.drop('value').equals(want.drop('value')), f'{where}: came back keyed or ordered differently'
    if values:
        assert got['value'].to_list() == pytest.approx(want['value'].to_list()), f'{where}: different numbers'


def test_a_update_walk_answers_what_a_fresh_build_answers(port):
    """The oracle again, over ported models, three updates deep.

    - **The objective and the layout, always.** A read-back that sliced the
      solver's vector wrongly puts the right numbers on the wrong coordinates.
    - **The numbers, where the answer carries prices**, which is where the
      model is continuous; a discrete model's optimum is not unique. The
      continuous ports agree to 1e-14 on both sinks, so `approx` rather than
      `equals`.

    On `highs` alone, the default.
    """
    if port['name'] in TOO_SLOW_TO_WALK:
        pytest.skip(f'{port["name"]} is too slow to walk — see TOO_SLOW_TO_WALK')

    program = expanded(port['spec']).program
    given = _declared(port_sources(port['name']), program)

    with sps.build(expanded(port['spec']), given) as model:
        model.solve()
        for step, factor in enumerate(WALK):
            change = _scaled(given, factor)
            where = f'{port["name"]} x{factor}'
            with sps.solve(expanded(port['spec']), change) as reference:
                got = model.update(change).solve(start=None)

                assert got.termination_condition == reference.termination_condition, f'{where}: terminated differently'
                assert got.has_primal == reference.has_primal, f'{where}: one left values and the other did not'
                if reference.has_primal:
                    assert got.objective == pytest.approx(reference.objective), f'{where}: a different optimum'
                    wanted = _prices(reference, program)
                    assert (_prices(got, program) is None) == (wanted is None), f'{where}: one is priced and one is not'
                    unique = port['name'] not in ALTERNATE_OPTIMA
                    for name in program.variables:
                        _laid_out_alike(
                            got.primal(name),
                            reference.primal(name),
                            values=wanted is not None and unique,
                            where=f'{where} {name}',
                        )
                    for name in wanted or {}:
                        exact = name not in NONUNIQUE_PRICES.get(port['name'], set())
                        _laid_out_alike(got.dual(name), wanted[name], values=exact, where=f'{where} {name} price')

            if not step:
                assert model.diagnostics().loads == 1, (
                    'the same numbers updated have to hash alike, or no driver ever takes the fast path'
                )


def _same_handoff(got: Any, want: Any, where: str) -> None:
    """Two hand-offs hold the same model: every frame alike, a sum in another order alike to its last bits."""
    for frame in ('cols', 'rows', 'matrix', 'qmatrix', 'quad', 'sos'):
        assert_frame_equal(getattr(got, frame), getattr(want, frame), rel_tol=1e-12, abs_tol=0.0)
    assert_frame_equal(got.obj.sort('col'), want.obj.sort('col'), rel_tol=1e-12, abs_tol=0.0)
    assert (got.row_starts == want.row_starts).all(), f'{where}: the matrix rows start elsewhere'
    assert (got.column_count, got.row_count) == (want.column_count, want.row_count), f'{where}: another shape'
    assert got.objective_constant == pytest.approx(want.objective_constant, rel=1e-12), f'{where}: another constant'


def _same_diagnostics(got: Any, want: Any, where: str) -> None:
    """Two models report the same shape and the same ranges, a block at a time in build order."""
    assert (got.columns, got.rows, got.nonzeros) == (want.columns, want.rows, want.nonzeros), f'{where}: another shape'
    for frame in ('omissions', 'sparse_parameters', 'coefficient_range', 'bound_range', 'rhs_range'):
        assert_frame_equal(getattr(got, frame), getattr(want, frame), rel_tol=1e-12, abs_tol=0.0)
    assert got.objective_range == pytest.approx(want.objective_range, rel=1e-12), f'{where}: another objective range'


def test_a_change_of_numbers_alone_hands_over_what_a_fresh_build_does(port, monkeypatch):
    """Each real-valued parameter of a port, its numbers moved alone, gives the hand-off and diagnostics a fresh build gives.

    Whichever way the update went: a parameter that feeds only bounds,
    constant sides, linear coefficients and the objective is read again on the old labels
    ([`refilled`][specsolve._engine.assembly.refilled]), and any other is
    built in full. A change the fresh build refuses is left out; the model
    goes back to the port's own numbers after each.
    """
    if port['name'] in TOO_SLOW_TO_WALK:
        pytest.skip(f'{port["name"]} is too slow to walk — see TOO_SLOW_TO_WALK')
    from specsolve._engine import engine as engine_module

    spec = expanded(port['spec'])
    given = _declared(port_sources(port['name']), spec.program)
    refilled = engine_module.refilled
    took: list[str] = []

    def spy(built: Any, attached: Any, changed: Any, measured: Any) -> Any:
        took.extend(changed)
        return refilled(built, attached, changed, measured)

    monkeypatch.setattr(engine_module, 'refilled', spy)
    with sps.build(spec, given) as model:
        for name, value in given.items():
            if not (isinstance(value, pl.DataFrame) and value.schema.get('value') == pl.Float64):
                continue
            change = {name: value.with_columns(pl.col('value') * 1.25 + 0.5)}
            try:
                reference = sps.build(spec, given | change)
            except sps.errors.SpecsolveError:
                continue
            with reference:
                model.update(change)
                where = f'{port["name"]} {name}'
                _same_handoff(model._engine._model.handoff, reference._engine._model.handoff, where)
                _same_diagnostics(model.diagnostics(), reference.diagnostics(), where)
            model.update({name: value})


#: One parameter in each role: `cap` a bound, `demand` a constant side, `price`
#: a cost, and `on` and `eff` unused until a case writes them in.
ROLES = {
    'dimensions': {'generator': {'dtype': 'str'}},
    'parameters': {
        'cap': {'dims': ['generator']},
        'on': {'dims': ['generator']},
        'eff': {'dims': ['generator']},
        'price': {'dims': ['generator']},
        'demand': {'dims': []},
    },
    'variables': {'p': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'constraints': {'meet': {'dims': [], 'expression': 'sum(p, over=generator) >= demand'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * price)'},
}
ROLE_GENERATORS = ['a', 'b']
ROLE_DATA = {
    'generator': ROLE_GENERATORS,
    'cap': pl.DataFrame({'generator': ROLE_GENERATORS, 'value': [10.0, 10.0]}),
    'on': pl.DataFrame({'generator': ROLE_GENERATORS, 'value': [1.0, 1.0]}),
    'eff': pl.DataFrame({'generator': ROLE_GENERATORS, 'value': [1.0, 1.0]}),
    'price': pl.DataFrame({'generator': ROLE_GENERATORS, 'value': [1.0, 2.0]}),
    'demand': pl.DataFrame({'value': [8.0]}),
}


COEFFICIENT = 'sum(eff * p, over=generator) >= demand'


@pytest.mark.parametrize(
    ('declarations', 'change', 'refilled', 'loads'),
    [
        pytest.param({}, {'cap': [8.0, 9.0]}, True, 1, id='a bound'),
        pytest.param({}, {'demand': [5.0]}, True, 1, id='a constant side'),
        pytest.param({}, {'price': [3.0, 1.0]}, True, 1, id='a cost'),
        pytest.param({'constraints.meet.expression': COEFFICIENT}, {'eff': [0.5, 2.0]}, True, 1, id='a coefficient'),
        pytest.param(
            {
                'constraints.meet.expression': COEFFICIENT,
                'constraints.each': {'dims': ['generator'], 'expression': 'p <= cap'},
            },
            {'eff': [0.0, 2.0]},
            True,
            2,
            id='a coefficient to zero before another constraint',
        ),
        pytest.param(
            {'constraints.meet.expression': 'sum(p / eff, over=generator) >= demand'},
            {'eff': [0.5, 2.0]},
            True,
            1,
            id='a divisor',
        ),
        pytest.param({'variables.p.where': 'on > 0'}, {'on': [0.0, 1.0]}, False, 2, id='a variable where'),
        pytest.param({'constraints.meet.where': 'demand > 10'}, {'demand': [15.0]}, False, 2, id='a constraint where'),
        pytest.param(
            {'parameters.cap.missing': 'absent'},
            {'cap': pl.DataFrame({'generator': ['a'], 'value': [8.0]})},
            False,
            1,
            id='fewer coordinates',
        ),
        pytest.param({}, {'generator': ROLE_GENERATORS}, False, 1, id='an index'),
    ],
)
def test_new_numbers_are_read_again_on_the_old_labels_only_where_nothing_else_can_move(
    monkeypatch, declarations, change, refilled, loads
):
    """Numbers that feed bounds, constant sides, linear coefficients and costs, at the coordinates they had, skip the build.

    A parameter a `where` reads moves which columns or rows exist; so do new
    coordinates or a new index. Each of those builds in full. Either way the
    hand-off is a fresh build's. A coefficient falling to zero takes its
    matrix entry with it, on the old labels, and the entries after it move up.
    """
    from specsolve._engine import engine as engine_module

    spec = override(ROLES, **declarations)
    new = {
        name: ROLE_DATA[name].with_columns(pl.Series('value', value))
        if name in ROLES['parameters'] and isinstance(value, list)
        else value
        for name, value in change.items()
    }
    real = engine_module.refilled
    took: list[bool] = []
    monkeypatch.setattr(engine_module, 'refilled', lambda *args: took.append(True) or real(*args))
    with sps.build(spec, ROLE_DATA) as model, sps.build(spec, ROLE_DATA | new) as reference:
        model.solve()
        model.update(new)
        _same_handoff(model._engine._model.handoff, reference._engine._model.handoff, str(change))
        assert model.solve().objective == pytest.approx(reference.solve().objective), 'a fresh build answers alike'
        assert bool(took) is refilled, f'the update {"skipped" if took else "ran"} the build'
        assert model.diagnostics().loads == loads, f'the solver was loaded {model.diagnostics().loads} times'


def test_a_coefficient_of_a_quadratic_constraint_builds_in_full(monkeypatch):
    """A quadratic row goes to a sink whole, coefficients and all, so its coefficients are not read again."""
    from specsolve._engine import engine as engine_module

    spec = override(ROLES, **{'constraints.meet.expression': 'sum(eff * p * p, over=generator) <= demand'})
    change = {'eff': ROLE_DATA['eff'].with_columns(pl.Series('value', [0.5, 2.0]))}
    real = engine_module.refilled
    took: list[bool] = []
    monkeypatch.setattr(engine_module, 'refilled', lambda *args: took.append(True) or real(*args))
    with sps.build(spec, ROLE_DATA) as model, sps.build(spec, ROLE_DATA | change) as reference:
        model.update(change)
        _same_handoff(model._engine._model.handoff, reference._engine._model.handoff, 'eff on a quadratic row')
    assert not took, 'a quadratic row was read again on the old labels'


#: `eff` scales `p` on each generator's own row, after `meet`'s, so a moved
#: coefficient sits in a row the matrix does not start with.
PER_ROW = override(
    ROLES,
    **{
        'constraints.meet.expression': 'sum(p, over=generator) >= demand',
        'constraints.limit': {'dims': ['generator'], 'expression': 'eff * p <= cap'},
    },
)


def test_a_moved_coefficient_is_taken_in_place_and_taken_back(solver_name):
    """A changed coefficient reaches the loaded solver in place, and so does the change undone.

    `eff` of `a` doubles, so `a` reaches half its capacity and `b`, dearer,
    covers the rest. Each answer is a fresh build's on the same solver.
    """
    doubled = {'eff': ROLE_DATA['eff'].with_columns(pl.Series('value', [2.0, 1.0]))}
    with sps.build(PER_ROW, ROLE_DATA) as model:
        model.solve(solver_name)
        for data in (ROLE_DATA | doubled, ROLE_DATA):
            with sps.build(PER_ROW, data) as reference:
                got = model.update({'eff': data['eff']}).solve(solver_name).objective
                assert got == pytest.approx(reference.solve(solver_name).objective), 'a fresh build answers alike'
        assert model.diagnostics().loads == 1, 'the coefficients moved in place, so the solver stayed loaded'


def test_a_quadratic_constraint_whose_linear_part_moved_loads_again():
    """A quadratic row goes to the solver whole, so a coefficient of its linear part loads the model again."""
    pytest.importorskip('gurobipy', reason='only the gurobi sink takes a quadratic constraint')
    spec = override(ROLES, **{'constraints.meet.expression': 'sum(eff * p + p * p, over=generator) >= demand'})
    changed = {'eff': ROLE_DATA['eff'].with_columns(pl.Series('value', [0.5, 4.0]))}
    with sps.build(spec, ROLE_DATA) as model, sps.build(spec, ROLE_DATA | changed) as reference:
        model.solve('gurobi')
        got = model.update(changed).solve('gurobi').objective
        assert got == pytest.approx(reference.solve('gurobi').objective, rel=1e-6), 'a fresh build answers alike'
        assert model.diagnostics().loads == 2, 'a quadratic row whose linear part moved was kept'


def test_a_divisor_falling_to_zero_is_refused_as_a_fresh_build_refuses_it():
    """`p / eff` read again on the old labels still refuses a divisor of zero, in a fresh build's words."""
    spec = override(ROLES, **{'constraints.meet.expression': 'sum(p / eff, over=generator) >= demand'})
    change = {'eff': ROLE_DATA['eff'].with_columns(pl.Series('value', [0.0, 2.0]))}
    with pytest.raises(sps.errors.DataError, match='eff') as fresh:
        sps.build(spec, ROLE_DATA | change)
    with sps.build(spec, ROLE_DATA) as model, pytest.raises(sps.errors.DataError) as updated:
        model.update(change)
    assert str(updated.value) == str(fresh.value), 'the update refuses what a fresh build refuses, in its words'


#: A variable whose coefficient a `cases:` condition picks: 2 where `flag`
#: holds, 1 elsewhere.
CASES_ON_A_VARIABLE = {
    'dimensions': {'t': {'dtype': 'int', 'ordered': True}},
    'parameters': {'flag': {'dims': ['t'], 'dtype': 'bool'}},
    'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'expressions': {
        'scaled': {'dims': ['t'], 'cases': {'flagged': {'when': 'flag', 'expression': '2 * x'}}, 'otherwise': 'x'}
    },
    'constraints': {'under': {'dims': ['t'], 'expression': 'scaled <= 10'}},
    'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
}

#: A shift whose offset is a parameter, so `lead` picks which column each row reads.
SHIFT_BY_A_PARAMETER = {
    'dimensions': {'g': {'dtype': 'str'}, 't': {'dtype': 'int', 'ordered': True}},
    'parameters': {'lead': {'dims': ['g'], 'dtype': 'int'}},
    'variables': {
        'level': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
        'take': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 10}},
    },
    'constraints': {'link': {'dims': ['g', 't'], 'expression': 'take <= shift(level, along=t, offset=lead, edge=0)'}},
    'objective': {
        'sense': 'maximize',
        'expression': 'sum(sum(take, over=g), over=t) - sum(sum(level, over=g), over=t)',
    },
}


@pytest.mark.parametrize(
    ('spec', 'data', 'change'),
    [
        pytest.param(
            CASES_ON_A_VARIABLE,
            {'t': [0, 1, 2, 3], 'flag': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [True, False, True, False]})},
            {'flag': pl.DataFrame({'t': [0, 1, 2, 3], 'value': [False, False, True, True]})},
            id='a cases condition',
        ),
        pytest.param(
            SHIFT_BY_A_PARAMETER,
            {
                'g': ['a', 'b'],
                't': [0, 1, 2],
                'lead': pl.DataFrame({'g': ['a', 'b'], 'value': [1, 1]}, schema={'g': pl.String, 'value': pl.Int64}),
            },
            {'lead': pl.DataFrame({'g': ['a', 'b'], 'value': [0, 1]}, schema={'g': pl.String, 'value': pl.Int64})},
            id='a shift offset',
        ),
    ],
)
def test_a_parameter_that_picks_which_entries_exist_is_built_again(monkeypatch, spec, data, change):
    """A `cases:` condition over variables, or a shift's named offset, decides which matrix entries exist.

    Neither is a scaling of a variable the walk of products can see, so each
    has its own branch; either one missed reads new numbers onto old labels.
    """
    from specsolve._engine import engine as engine_module

    real = engine_module.refilled
    took: list[bool] = []
    monkeypatch.setattr(engine_module, 'refilled', lambda *args: took.append(True) or real(*args))
    with sps.build(spec, data) as model, sps.build(spec, data | change) as reference:
        model.update(change)
        _same_handoff(model._engine._model.handoff, reference._engine._model.handoff, str(change))
    assert not took, 'a parameter that decides which entries exist was read again on the old labels'


def test_as_many_coordinates_but_other_ones_build_in_full(monkeypatch):
    """A bound moved from one generator to another keeps its height, and still moves which side is open.

    `cap` declares `missing: absent`, so a generator it has no row for has no
    upper bound; the new table names `b` where the old named `a`.
    """
    from specsolve._engine import engine as engine_module

    spec = override(ROLES, **{'parameters.cap.missing': 'absent'})
    before = ROLE_DATA | {'cap': pl.DataFrame({'generator': ['a'], 'value': [8.0]})}
    change = {'cap': pl.DataFrame({'generator': ['b'], 'value': [8.0]})}
    real = engine_module.refilled
    took: list[bool] = []
    monkeypatch.setattr(engine_module, 'refilled', lambda *args: took.append(True) or real(*args))
    with sps.build(spec, before) as model, sps.build(spec, before | change) as reference:
        model.update(change)
        _same_handoff(model._engine._model.handoff, reference._engine._model.handoff, 'cap moved to b')
    assert not took, 'other coordinates were read again on the old labels'


def test_new_numbers_on_a_row_the_build_dropped_are_refused_as_a_fresh_build_refuses_them():
    """A constraint that dropped a row with no terms builds in full, because its checks read the rows it labelled.

    `a` has no row at t=2, so `a * x` has no term there: the row is labelled,
    keeps no term and is dropped. A divisor of zero at t=2 is refused by a
    fresh build; read again on the rows that survived, it would go unseen.
    """
    spec = {
        'dimensions': {'t': {'dtype': 'int', 'ordered': True}},
        'parameters': {'a': {'dims': ['t'], 'missing': 'neutral'}, 'cap': {'dims': ['t']}, 'eff': {'dims': ['t']}},
        'variables': {'x': {'dims': ['t'], 'bounds': {'lower': 0}}},
        'constraints': {'limit': {'dims': ['t'], 'expression': 'a * x <= cap / eff'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    t = [0, 1, 2]
    data = {
        't': t,
        'a': pl.DataFrame({'t': [0, 1], 'value': [1.0, 1.0]}),
        'cap': pl.DataFrame({'t': t, 'value': [5.0, 5.0, 5.0]}),
        'eff': pl.DataFrame({'t': t, 'value': [1.0, 1.0, 1.0]}),
    }
    change = {'eff': pl.DataFrame({'t': t, 'value': [1.0, 1.0, 0.0]})}
    with pytest.raises(sps.errors.DataError, match='eff') as fresh:
        sps.build(spec, data | change)
    with sps.build(spec, data) as model, pytest.raises(sps.errors.DataError) as updated:
        model.update(change)
    assert str(updated.value) == str(fresh.value), 'the update refuses what a fresh build refuses, in its words'


@pytest.mark.parametrize('rung', RUNGS)
def test_only_a_update_that_moves_a_label_loads_the_solver_again(dispatch_yaml, rung, solver_name):
    """The fast path is taken exactly when the structure held.

    The first solve always loads — there was nothing to keep — so a driver on
    the fast path leaves `diagnostics().loads` at one however many times round.
    The rule is the digest's, so it is the same rule for every sink that can
    stay loaded.
    """
    spec, given = _case(rung, dispatch_yaml)
    with sps.build(spec, given) as model:
        model.solve(solver_name=solver_name)
        assert model.diagnostics().loads == 1, 'the first solve has nothing loaded to keep'

        model.update(rung.change).solve(solver_name=solver_name)
        seen = model.diagnostics()
        expected = 1 if rung.keeps_the_solver else 2
        assert seen.solves == 2, 'both solves are counted whichever path each took'
        assert seen.loads == expected, (
            'an update that keeps every label pushes values onto the loaded solver; '
            'one that moves a label has to load the model again'
        )


def _tables(spec: Any) -> Any:
    """*model*'s solver tables, read off it built on the reach data."""
    with sps.build(spec, reach_sources()) as built:
        return built._engine._model.handoff


#: The three fields of the digest **no rung above can reach**: a variable's
#: type, a row's comparison and the objective's direction all come from the
#: YAML, so they are pinned one edit of the declaration apart.
DECLARED = [
    pytest.param({'variables.p.domain': 'integer'}, id='a variable type'),
    pytest.param({'constraints.meet.expression': 'sum(reach * p, over=plant) == demand'}, id="a row's comparison"),
    pytest.param({'objective.sense': 'maximize'}, id='the sense'),
]


@pytest.mark.parametrize('edited', DECLARED)
def test_the_digest_moves_where_a_declaration_moved(edited):
    """What a re-solve may not change, checked directly for want of a rung."""
    assert _tables(REACH).structure != _tables(override(REACH, **edited)).structure, (
        'a model a re-solve may not be pushed onto has to hash differently'
    )


#: The counts, which no edit of a declaration reaches either.
COUNTS = [pytest.param('column_count', id='the column count'), pytest.param('row_count', id='the row count')]


@pytest.mark.parametrize('count', COUNTS)
def test_the_digest_reads_the_counts_that_frame_its_vectors(count):
    """The counts say where one hashed vector ends and the next begins.

    Every vector goes in as raw bytes with nothing between them, so a model
    with one column more and one row fewer offers the digest the same bytes in
    the same order.
    """
    tables = _tables(REACH)
    moved = replace(tables, **{count: getattr(tables, count) + 1})
    assert moved.structure != tables.structure, f'{count} is framing, not decoration: the same bytes split elsewhere'


def _hashes(monkeypatch) -> list[int]:
    """A counter of every ask for a digest, from however many objects ask.

    A plain property in place of the `cached_property`, so one object asked
    twice counts twice. Each count below is one object asked once.
    """
    from specsolve._sinks.handoff import Handoff

    taken: list[int] = []
    real = Handoff.structure.func
    monkeypatch.setattr(
        Handoff,
        'structure',
        property(lambda self: (taken.append(1), real(self))[1]),
    )
    return taken


def test_a_solve_that_is_never_rebuilt_never_hashes_the_model(model, monkeypatch):
    """One solve, no digest — the comparison it would feed does not exist (#1608)."""
    taken = _hashes(monkeypatch)
    model.solve()
    assert taken == [], f'a first solve has nothing to compare against, so it hashed {len(taken)} time(s) for nothing'


def test_a_rebuild_takes_the_evidence_and_the_fast_path_still_holds(model, monkeypatch):
    """One digest per solve, and the fast path still holds (#1608).

    A push leaves the digest still describing what the solver holds, so the
    second rebuild finds one taken and reads nothing.
    """
    taken = _hashes(monkeypatch)
    model.solve()
    model.update({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})}).solve()
    assert model.diagnostics().loads == 1, 'a pushable update still takes the fast path'
    assert len(taken) == 2, (
        f'two solves take two digests — the outgoing model at the rebuild and the incoming one at the '
        f'comparison — and not {len(taken)}'
    )

    model.update({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [11.0, 21.0, 31.0, 41.0]})}).solve()
    assert model.diagnostics().loads == 1, 'and again'
    assert len(taken) == 3, f'each further solve adds one, not two: {len(taken)} after three solves'


def test_solving_the_same_model_twice_keeps_it_without_a_rebuild_between(model, monkeypatch):
    """A second solve of an *unchanged* model still takes the fast path (#1608).

    `keeps` asking is itself the first ask for a digest.
    """
    taken = _hashes(monkeypatch)
    model.solve()
    model.solve()
    assert model.diagnostics().loads == 1, 'and keeping it means not loading it twice'
    assert len(taken) == 2, f'the outgoing model and the incoming one, as ever, not {len(taken)}'


def test_a_rebuild_leaves_the_held_solver_pinning_none_of_the_old_model(model):
    """What outlives a build is the digest, never the frames it was read from (#1608).

    A solver holding the tables it loaded has to drop them before the next
    build allocates, or a re-solving loop stands at two models' peak. **Asked
    between the rebuild and the next solve**, the only window where it shows.
    The rows are asked for, being what a new `load` replaces; the matrix is
    the new model's too, since only numbers changed.
    """
    model.solve()
    released = weakref.ref(model._engine._model.handoff.rows)
    model.update({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [12.0, 22.0, 32.0, 42.0]})})
    gc.collect()
    assert released() is None, "the rebuilt-over model's rows are still reachable, so the solver kept a whole model"


#: The option name each sink gives a time limit; `solver_options` is forwarded
#: verbatim.
LIMITS = {'highs': 'time_limit', 'gurobi': 'TimeLimit', 'xpress': 'timelimit'}


def test_a_solve_asking_for_other_options_loads_the_model_again(model, solver_name):
    """Options are recorded at the load, so they are part of what may be kept.

    A solver holds what it was told when it took the model. Keeping it for a
    solve that asked for others would run that solve under the *first* one's
    limits and report the answer as the one asked for — a gap left loose, or a
    time limit that was never the caller's.
    """
    option = LIMITS[solver_name]
    model.solve(solver_name=solver_name, solver_options={option: 60.0})
    model.solve(solver_name=solver_name, solver_options={option: 120.0})
    assert model.diagnostics().loads == 2, 'a solver told the first options cannot be told others'

    model.solve(solver_name=solver_name, solver_options={option: 120.0})
    assert model.diagnostics().loads == 2, 'the same options ask for the model the solver already holds'


def test_a_update_takes_a_change_at_a_time_and_keeps_the_rest(dispatch_yaml, model):
    """Partial by construction: what is not named keeps what `build` bound."""
    every = {**sources(), 'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [1.0, 2.0, 3.0, 4.0]})}
    with sps.solve(dispatch_yaml, every | COORDS) as reference:
        updated = model.update({'load': every['load']}).solve()
        assert updated.objective == pytest.approx(reference.objective)


def test_a_update_may_be_repeated_and_each_answer_is_its_own(model):
    """The loop `update` exists for: attach, solve, read, attach again."""
    served = []
    for scale in (0.5, 1.0, 1.5):
        load = pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [40.0 * scale] * len(SNAPSHOTS)})
        served.append(model.update({'load': load}).solve().primal('p')['value'].sum())

    assert served == sorted(served), f'{served} — more load must dispatch more power'
    assert model.diagnostics().loads == 1, 'a scaled right-hand side moves no label'


def test_a_result_from_before_a_update_keeps_reading(model):
    """A result owns its read-back, so nothing done to the model expires it.

    The label frames are immutable and shared: an update builds new ones
    without touching what earlier results hold, so an old answer stays an
    answer over its own coordinates — a driver keeps any result it still
    wants, at the price of keeping that build's label frames alive.
    """
    before = model.solve()
    kept = before.primal('p')
    prices = before.dual('power_balance')

    after = model.update({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [3.0, 1.0, 2.0]})}).solve()

    assert after.objective != before.objective, 'reordered costs move the optimum, so the two answers differ'
    assert before.primal('p').equals(kept), 'the old result still reads, and reads its own build'
    assert before.dual('power_balance').equals(prices), 'its duals too'


def test_closing_a_result_never_touches_the_model(model):
    """`close` releases what the result holds — its values and its hold on the
    label frames — never the model or the solver, which are the handle's to
    close. So a result closed on the way out of a `with` block cannot take
    down the model a loop is still solving, and a sibling result, holding its
    own read-back, keeps reading.
    """
    first = model.solve()
    sibling = model.solve()
    first.close()

    assert sibling.primal('p').height > 0, 'a sibling holds its own read-back'
    assert model.solve().primal('p').height > 0, 'the model is still there to solve'


@pytest.mark.parametrize(
    ('call', 'unknown'),
    [
        pytest.param(lambda model: model.update({'p_maxx': 1}), 'p_maxx', id='sources'),
        pytest.param(lambda model: model.update({'snapshots': [0]}), 'snapshots', id='an index'),
    ],
)
def test_a_update_refuses_a_name_the_model_does_not_declare(model, call, unknown):
    """An update names what changed, so a name nothing reads is the one failure
    a driver cannot see: it re-solves the numbers already attached and reports the
    answer. `build` needs no such check — it attaches every declared name or
    fails."""
    with pytest.raises(sps.errors.DataError, match=unknown):
        call(model)


def test_a_dimension_index_updates_as_a_source(model):
    """A dimension index is a source (the data-attachment rules), so `update` takes it where it
    takes any other — the refusal above is for names the model never declared,
    not for names that happen not to be parameters."""
    change = {'snapshot': [0, 1], 'load': pl.DataFrame({'snapshot': [0, 1], 'value': [5.0, 6.0]})}
    assert model.update(change).solve().primal('p').height > 0, 'a dimension index is a source, and updates as one'


@pytest.mark.parametrize(
    ('change', 'checked'),
    [
        pytest.param(
            {'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [41.0, 81.0, 56.0, 96.0]})},
            {'load'},
            id='a parameter alone',
        ),
        pytest.param({'snapshot': SNAPSHOTS}, {'load'}, id='an index and the parameter over it'),
        pytest.param({'generator': GENERATORS}, {'p_max', 'cost'}, id='an index and the parameters over it'),
    ],
)
def test_an_update_checks_again_only_the_parameters_it_can_change(model, monkeypatch, change, checked):
    """A parameter whose own source and whose dims' indices an update leaves alone is not read or checked again."""
    from specsolve import _sources

    real = _sources._checked_parameter
    seen: list[str] = []

    def spy(name, *args, **kwargs):
        seen.append(name)
        return real(name, *args, **kwargs)

    monkeypatch.setattr(_sources, '_checked_parameter', spy)
    model.update(change)
    assert set(seen) == checked, f'the update checked {sorted(seen)} again'


def test_an_update_that_drops_a_label_refuses_the_unchanged_parameter_still_over_it(model):
    """A parameter kept from the last build is still checked against an index the update changed.

    `load` is not in the update, but its rows at snapshot 3 now name a label
    the index no longer has.
    """
    with pytest.raises(sps.errors.DataError, match="parameter 'load' has label"):
        model.update({'snapshot': [0, 1, 2]})


def test_a_update_can_grow_a_dimension():
    """Appending rows is an update — the Benders master, in three lines.

    A cut family is declared once and its members come from data (the data-attachment rules), so
    an iteration hands over a longer table and the coordinates to match. The
    labels of the rows that were already there do not move, but the model has
    more rows than the solver holds, so it is loaded again.
    """
    master = {
        'dimensions': {'generator': {'dtype': 'str'}, 'cut': {'dtype': 'int'}},
        'parameters': {
            'invest': {'dims': ['generator']},
            'cut_const': {'dims': ['cut']},
            'cut_slope': {'dims': ['cut', 'generator']},
        },
        'variables': {
            'cap': {'dims': ['generator'], 'bounds': {'lower': 0, 'upper': 100}},
            'theta': {'dims': [], 'bounds': {'lower': 0}},
        },
        'constraints': {
            'optimality_cut': {
                'dims': ['cut'],
                'expression': 'theta >= cut_const + sum(cut_slope * cap, over=generator)',
            }
        },
        'objective': {'sense': 'minimize', 'expression': 'sum(cap * invest) + theta'},
    }
    invest = pl.DataFrame({'generator': ['wind', 'gas'], 'value': [90.0, 30.0]})

    def cuts(n: int) -> dict[str, pl.DataFrame]:
        return {
            'cut_const': pl.DataFrame({'cut': list(range(n)), 'value': [500.0 * (i + 1) for i in range(n)]}),
            'cut_slope': pl.DataFrame(
                {
                    'cut': [i for i in range(n) for _ in range(2)],
                    'generator': ['wind', 'gas'] * n,
                    'value': [-5.0 * (i + 1) for i in range(n) for _ in range(2)],
                }
            ),
        }

    with (
        sps.solve(
            master, {'invest': invest, **cuts(3)} | {'cut': [0, 1, 2], 'generator': ['wind', 'gas']}
        ) as reference,
        sps.build(master, {'invest': invest, **cuts(1)} | {'cut': [0], 'generator': ['wind', 'gas']}) as model,
    ):
        model.solve()
        grown = model.update(cuts(3) | {'cut': [0, 1, 2]}).solve()

        assert grown.objective == pytest.approx(reference.objective)
        assert grown.primal('cap').equals(reference.primal('cap'))
        assert model.diagnostics().loads == 2, 'more rows than the solver holds is a load, not a push'


def test_a_written_file_follows_the_update(model, tmp_path):
    """`write` reads the built model, so it reads the updated one."""
    model.write(tmp_path / 'before.lp')
    model.update({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [7.0, 7.0, 7.0, 7.0]})})
    model.write(tmp_path / 'after.lp')

    after = (tmp_path / 'after.lp').read_text()
    assert (tmp_path / 'before.lp').read_text() != after
    assert '7' in after


def test_a_update_that_cannot_build_leaves_nothing_half_built(model):
    """The build's own rule, one call later: a failure releases the model.

    A handle holding half a model would answer the next `solve` with a mixture
    of two, which is worse than having nothing to answer with.
    """
    model.solve()
    with pytest.raises(sps.errors.DataError):
        model.update({'load': pl.DataFrame({'snapshot': [0, 0, 1], 'value': [1.0, 2.0, 3.0]})})

    with pytest.raises(sps.errors.SpecsolveError, match='no built model to hand over'):
        model.solve()


def test_an_update_after_a_refused_one_reads_the_refused_source_again(model):
    """`model.update(x)` answers what a fresh build of every source given so far answers.

    The refused `load` is still the model's source, so an update of `cost`
    alone is refused for it, as a fresh build of the same sources is. The
    door refused `load` before the last good tables were replaced, and the
    next update kept that good `load` and built, until a failure forgot them.
    """
    with pytest.raises(sps.errors.DataError):
        model.update({'load': pl.DataFrame({'snapshot': [0, 0, 1], 'value': [1.0, 2.0, 3.0]})})
    with pytest.raises(sps.errors.DataError, match='load'):
        model.update({'cost': pl.DataFrame({'generator': GENERATORS, 'value': [9.0, 2.0, 1.0]})})


def test_an_update_interrupted_while_choosing_what_to_keep_reads_every_source_again(model, dispatch_yaml, monkeypatch):
    """`model.update(x)` answers what a fresh build of every source given so far answers.

    The first update of `load` stops while it chooses what to keep, after the
    new `load` is attached. The next update, of `cost` alone, must still read
    that `load` rather than keep the tables of the build before it. The choice
    ran before the cleanup that forgets the last build's tables, so the next
    update kept the `load` built before the attached one.
    """
    import specsolve._api as api

    load = pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [10.0, 20.0, 30.0, 40.0]})
    cost = pl.DataFrame({'generator': GENERATORS, 'value': [1.0, 2.0, 60.0]})
    choose = api.kept_parameters

    def interrupted(*args):
        monkeypatch.setattr(api, 'kept_parameters', choose)
        raise KeyboardInterrupt

    monkeypatch.setattr(api, 'kept_parameters', interrupted)
    with pytest.raises(KeyboardInterrupt):
        model.update({'load': load})
    model.update({'cost': cost})
    with sps.build(dispatch_yaml, sources() | COORDS | {'load': load, 'cost': cost}) as fresh:
        expected = fresh.solve().objective
    assert model.solve().objective == pytest.approx(expected), 'the attached load, not the one built before it'


def test_diagnostics_report_the_shape_the_solver_was_handed(dispatch_yaml):
    """The size question `check` cannot answer, needing no data where this needs all of it.

    `examples/dispatch.yaml` masks on `p_max > 0`, so the shape is what
    *survived* the mask rather than what the declarations multiply out to.
    """
    with sps.build(dispatch_yaml, sources() | COORDS) as model:
        model.solve()
        seen = model.diagnostics()

        assert (seen.columns, seen.rows) == (len(SNAPSHOTS) * len(GENERATORS), len(SNAPSHOTS))
        assert seen.nonzeros == seen.columns, 'one balance row per snapshot, one entry per generator in it'
        assert seen.omissions.is_empty(), 'every declared row reached the solver'

    assert model.diagnostics().nonzeros == seen.nonzeros, 'a released model still says how big it was'


def test_a_mask_that_removes_a_column_removes_it_from_the_shape(dispatch_yaml):
    """Read off the built model, so a mask that moved moves the counts with it."""
    zeroed = {**sources(), 'p_max': pl.DataFrame({'generator': GENERATORS, 'value': [100.0, 60.0, 0.0]})}
    with sps.build(dispatch_yaml, zeroed | COORDS) as model:
        assert model.diagnostics().columns == len(SNAPSHOTS) * (len(GENERATORS) - 1)


def test_a_cost_falling_to_zero_shrinks_the_objective_and_keeps_the_solver():
    """The objective frame may change height across an update. The solver may not.

    A zero cost is pruned, so `obj` holds one row fewer, while `structure`
    does not read `obj`. A push that read `obj` positionally would hand the
    solver one plant's cost under another's name.
    """
    given = reach_sources()
    with sps.build(REACH, given) as model:
        model.solve()
        before = model._engine._model.handoff.obj.height
        assert model.diagnostics().loads == 1, 'the first solve has nothing loaded to keep'

        zeroed = pl.DataFrame({'plant': PLANTS, 'value': [0.0, 2.0, 3.0, 4.0]})
        updated = model.update({'cost': zeroed}).solve()

        assert model._engine._model.handoff.obj.height == before - 1, (
            'the zero cost should have left the objective frame'
        )
        assert model.diagnostics().loads == 1, 'a cost is pushed, so a cost falling to zero may not reload'

    with sps.build(REACH, {**given, 'cost': zeroed}) as fresh:
        assert updated.objective == fresh.solve().objective, (
            'the pushed cost vector disagrees with the one a cold build hands over'
        )
