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

import specsolve as sps
from specsolve.sources import attachable
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
        'reach': {'dims': ['zone', 'plant']},
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
#: The four `reach` rungs move **one field of the digest each**, and each
#: changes the answer.
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
            {'reach': reaching(('north', 'a', 0.5), ('north', 'b', 1.0), ('south', 'c', 1.0), ('south', 'd', 1.0))},
            False,
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
                got = model.update(change).solve()

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
    from specsolve.relational.sinks.handoff import Handoff

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
    assert model.solve().kept == 'solver', 'an unchanged model is the easiest thing there is to keep'
    assert model.diagnostics().loads == 1, 'and keeping it means not loading it twice'
    assert len(taken) == 2, f'the outgoing model and the incoming one, as ever, not {len(taken)}'


def test_a_rebuild_leaves_the_held_solver_pinning_none_of_the_old_model(model):
    """What outlives a build is the digest, never the frames it was read from (#1608).

    A solver holding the tables it loaded has to drop them before the next
    build allocates, or a re-solving loop stands at two models' peak. **Asked
    between the rebuild and the next solve**, the only window where it shows.
    """
    model.solve()
    released = weakref.ref(model._engine._model.handoff.matrix)
    model.update({'load': pl.DataFrame({'snapshot': SNAPSHOTS, 'value': [12.0, 22.0, 32.0, 42.0]})})
    gc.collect()
    assert released() is None, "the rebuilt-over model's matrix is still reachable, so the solver kept a whole model"


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
