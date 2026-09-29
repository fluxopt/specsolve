"""How much of a session a solve keeps, and the machinery under it.

**The public half** is `solve(keep=...)` and `result.kept`. A session holds the
solver with the model on it and the work that solver did, dropped in that
order: `'nothing'` keeps neither, `'solver'` keeps the first, `'progress'`
keeps both. `loads` says whether the model was handed over again, and the
iteration count says whether the work survived. `'nothing'` discards the held
solver, so the fresh one has nothing to begin from.

**The sink half** is `Solver.warm_start()` / `warm()`, with no caller above the
family yet (#382). A carried basis starts the simplex at the optimum, and every
refusal it can check is checked.

Iteration counts are deterministic, so none of this needs an idle box.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.relational.result import KEEPS
from specsolve.relational.sinks import SOLVERS
from tests.conftest import ITEMS, KNAPSACK, knapsack_sources

# ---------------------------------------------------------------------------
# models: an LP big enough to make the simplex work, and a MIP
# ---------------------------------------------------------------------------

SNAPSHOTS = list(range(40))
GENERATORS = [f'g{i}' for i in range(6)]
DISPATCH = {
    'dimensions': {'snapshot': {'dtype': 'int'}, 'generator': {'dtype': 'str'}},
    'parameters': {
        'p_max': {'dims': ['generator']},
        'cost': {'dims': ['snapshot', 'generator']},
        'load': {'dims': ['snapshot']},
    },
    'variables': {'p': {'dims': ['snapshot', 'generator'], 'bounds': {'lower': 0, 'upper': 'p_max'}}},
    'constraints': {'balance': {'dims': ['snapshot'], 'expression': 'sum(p, over=generator) == load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p * cost)'},
}


def dispatch_sources(snapshots: list[int] = SNAPSHOTS, generators: list[str] = GENERATORS) -> dict[str, pl.DataFrame]:
    """Deterministic data whose per-snapshot costs keep the simplex busy."""
    rng = np.random.default_rng(7)
    return {
        'snapshot': snapshots,
        'generator': generators,
        'p_max': pl.DataFrame({'generator': generators, 'value': rng.uniform(50.0, 150.0, len(generators))}),
        'cost': pl.DataFrame(
            {
                'snapshot': [s for s in snapshots for _ in generators],
                'generator': generators * len(snapshots),
                'value': rng.uniform(1.0, 50.0, len(snapshots) * len(generators)),
            }
        ),
        'load': pl.DataFrame({'snapshot': snapshots, 'value': rng.uniform(60.0, 250.0, len(snapshots))}),
    }


#: The same columns under more rows — what makes the row-span refusal its own
#: case rather than the column refusal arriving first.
DISPATCH_CAPPED = {
    **DISPATCH,
    'parameters': {**DISPATCH['parameters'], 'cap': {'dims': ['generator']}},
    'constraints': {
        **DISPATCH['constraints'],
        'capped': {'dims': ['generator'], 'expression': 'sum(p, over=snapshot) <= cap'},
    },
}


def capped_sources() -> dict[str, pl.DataFrame]:
    return {
        **dispatch_sources(),
        'cap': pl.DataFrame({'generator': GENERATORS, 'value': [4000.0] * len(GENERATORS)}),
    }


def _tables(spec: dict[str, Any], given: dict[str, Any]) -> Any:
    """*model*'s solver tables, read off it built on *given*."""
    with sps.build(spec, given) as built:
        return built._engine._model.handoff


#: Each member's own iteration counter — the noise-free observable of warmth.
SIMPLEX_ITERATIONS = {
    'highs': lambda solver: int(solver._handle.getInfo().simplex_iteration_count),
    'gurobi': lambda solver: int(solver._m.IterCount),
    'xpress': lambda solver: int(solver._p.attributes.simplexiter),
}


# ---------------------------------------------------------------------------
# the key claim: a carried basis is warm where a fresh session is cold
# ---------------------------------------------------------------------------


def test_a_carried_basis_starts_at_the_optimum_not_from_scratch(solver_name):
    """The same tables in a fresh session: cold works, warm starts done.

    Sink-level on purpose — the iteration counters are the members' own, so
    this is where warmth is observable rather than inferred from timing.
    """
    tables = _tables(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS})
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        first.run(tables)
        cold = SIMPLEX_ITERATIONS[solver_name](first)
        ws = first.warm_start()
    finally:
        first.close()

    assert cold > 0, 'the model must make the simplex work, or warmth would be unobservable'
    assert ws is not None and ws.column_statuses is not None, 'an LP solve leaves a basis to carry'

    fresh = member(tables)
    try:
        fresh.warm(ws)
        fresh.run(tables)
        assert SIMPLEX_ITERATIONS[solver_name](fresh) == 0, 'a carried basis starts at the optimum, not from scratch'
    finally:
        fresh.close()


def test_a_carried_basis_answers_what_the_cold_session_answered(solver_name):
    """A carry moves the route, never the answer — the oracle for the primitive."""
    tables = _tables(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS})
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        cold = first.run(tables)
        ws = first.warm_start()
    finally:
        first.close()
    assert ws is not None

    fresh = member(tables)
    try:
        fresh.warm(ws)
        warm = fresh.run(tables)
        assert warm.objective == pytest.approx(cold.objective), 'the carried basis reached a different optimum'
        assert warm.primal.to_list() == pytest.approx(cold.primal.to_list()), 'and a different vertex'
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# the three keeps, told apart by what each holds on to
# ---------------------------------------------------------------------------


def test_the_three_keeps_hold_the_two_things_independently(solver_name):
    """Each word keeps one more than the last, and both halves are observed.

    The solver kept shows up as `loads`, the work kept as the iteration count.
    `solver` skips the hand-off *and* begins from nothing.
    """
    with sps.build(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}) as model:
        first = model.solve(solver_name=solver_name)
        scratch = SIMPLEX_ITERATIONS[solver_name](model._engine._solver)
        assert first.kept == 'nothing', 'a first solve has nothing to keep'
        assert scratch > 0, 'the model must make the simplex work, or none of this is observable'

        carried = model.solve(solver_name=solver_name, keep='progress')
        assert carried.kept == 'progress', 'a kept solver asked to carry on reports that it did'
        assert model.diagnostics().loads == 1, 'progress keeps the solver too'
        assert SIMPLEX_ITERATIONS[solver_name](model._engine._solver) < scratch, (
            'carrying the last solve on must cost less work than starting over, or it buys nothing'
        )

        reused = model.solve(solver_name=solver_name, keep='solver')
        assert reused.kept == 'solver', 'the default keeps the solver and drops the work it did'
        assert model.diagnostics().loads == 1, 'solver keeps the loaded model — that is the half it shares'
        assert SIMPLEX_ITERATIONS[solver_name](model._engine._solver) == scratch, (
            'keeping only the solver begins from nothing, so it repeats the first solve iteration for iteration'
        )

        cold = model.solve(solver_name=solver_name, keep='nothing')
        assert cold.kept == 'nothing', 'nothing is kept however much the session held'
        assert model.diagnostics().loads == 2, 'keeping nothing discards the held solver, so the model loads again'

        assert reused.objective == pytest.approx(first.objective), 'a keep moves the route, never the answer'
        assert carried.objective == pytest.approx(first.objective)
        assert cold.objective == pytest.approx(first.objective)


def test_the_solver_is_kept_by_default_and_its_progress_is_not(solver_name):
    """Not `progress`: carrying the solver's work on is opt-in."""
    with sps.build(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}) as model:
        model.solve(solver_name=solver_name)
        assert model.solve(solver_name=solver_name).kept == 'solver'


def test_an_unknown_keep_names_the_three(solver_name):
    with (
        sps.build(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}) as model,
        pytest.raises(sps.SpecsolveError, match='unknown keep') as raised,
    ):
        model.solve(solver_name=solver_name, keep='warm')
    assert all(word in str(raised.value) for word in KEEPS), 'the refusal has to say what the three are'


def test_keeping_nothing_after_a_mip_solve_is_cold_too(solver_name):
    """The structural guarantee covers the MIP state no basis carries.

    An incumbent, a MIP start, cut pools — whatever the member squirrels away
    dies with the discarded solver, with no per-solver scrubbing to forget.
    """
    with sps.build(KNAPSACK, knapsack_sources()) as model:
        first = model.solve(solver_name=solver_name)
        cold = model.solve(solver_name=solver_name, keep='nothing')

        assert cold.kept == 'nothing'
        assert cold.objective == pytest.approx(first.objective)
        assert model.diagnostics().loads == 2, 'the discarded solver is the guarantee, and it shows up here'


# ---------------------------------------------------------------------------
# the MIP arm: no valid basis survives one, the incumbent crosses instead
# ---------------------------------------------------------------------------


def test_a_mixed_integer_solve_carries_an_incumbent_not_a_basis(solver_name):
    tables = _tables(KNAPSACK, knapsack_sources())
    member = SOLVERS[solver_name]
    first = member(tables)
    try:
        before = first.run(tables).objective
        ws = first.warm_start()
    finally:
        first.close()

    assert ws is not None, 'a solved MIP still leaves something to carry'
    assert ws.column_statuses is None and ws.row_statuses is None, 'a solved MIP leaves no valid basis on any solver'
    assert ws.column_values is not None, 'what crosses a MIP rebuild is the incumbent'

    fresh = member(tables)
    try:
        fresh.warm(ws)
        assert fresh.run(tables).objective == pytest.approx(before), 'an incumbent bounds the search, never the answer'
    finally:
        fresh.close()


# ---------------------------------------------------------------------------
# refusals: everything checkable is checked, everything else is the caller's
# ---------------------------------------------------------------------------

SHAPES = [
    pytest.param(
        DISPATCH,
        dispatch_sources() | {'snapshot': SNAPSHOTS},
        (DISPATCH, dispatch_sources(SNAPSHOTS, GENERATORS[:5])),
        id='a column short',
    ),
    pytest.param(
        DISPATCH,
        dispatch_sources() | {'snapshot': SNAPSHOTS},
        (DISPATCH_CAPPED, capped_sources() | {'snapshot': SNAPSHOTS}),
        id='rows extra under the same columns',
    ),
    pytest.param(
        KNAPSACK,
        knapsack_sources(),
        (KNAPSACK, knapsack_sources(ITEMS[:8])),
        id='an incumbent for other items',
    ),
]


def _read_from(solver_name: str, spec: dict[str, Any], given: dict[str, Any]) -> Any:
    """The warm start one solve of *spec* leaves, the session closed behind it."""
    tables = _tables(spec, given)
    session = SOLVERS[solver_name](tables)
    try:
        session.run(tables)
        return session.warm_start()
    finally:
        session.close()


@pytest.mark.parametrize(('spec', 'given', 'other'), SHAPES)
def test_a_warm_start_for_a_differently_shaped_model_is_refused(solver_name, spec, given, other):
    """A basis is positional, so a wrong span is a start about a different model.

    The refusal a cutting-plane master meets on its own update: a master that
    gained a row is exactly the second column of this table (#382).
    """
    ws = _read_from(solver_name, spec, given)
    other_spec, other_given = other
    tables = _tables(other_spec, other_given)
    session = SOLVERS[solver_name](tables)
    try:
        with pytest.raises(sps.SpecsolveError, match='warm start carries'):
            session.warm(ws)
    finally:
        session.close()


def test_a_warm_start_from_another_solver_is_refused(solver_name):
    """Statuses are the reading solver's own encoding; nothing else takes them."""
    ws = _read_from(solver_name, DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS})
    assert ws is not None
    tables = _tables(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS})
    session = SOLVERS[solver_name](tables)
    try:
        with pytest.raises(sps.SpecsolveError, match='read from'):
            session.warm(replace(ws, solver='someone_else'))
    finally:
        session.close()


def test_a_solver_that_has_not_run_has_nothing_to_carry(solver_name):
    """Loaded is not solved: there is no basis and no incumbent to read yet."""
    tables = _tables(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS})
    session = SOLVERS[solver_name](tables)
    try:
        assert session.warm_start() is None, 'a model that has not been solved has left nothing behind'
    finally:
        session.close()


class _Refusing:
    """A HiGHS handle refusing the named hint call — the probe for the `_took` guard.

    Nothing reachable from the tables makes the real handle refuse a
    span-checked hint.
    """

    def __init__(self, handle: Any, call: str) -> None:
        self._handle = handle
        self._call = call

    def __getattr__(self, name: str) -> Any:
        if name == self._call:
            import highspy

            return lambda hint: highspy.HighsStatus.kError
        return getattr(self._handle, name)


HINTS = [
    pytest.param(DISPATCH, dispatch_sources() | {'snapshot': SNAPSHOTS}, 'setBasis', id='a basis'),
    pytest.param(KNAPSACK, knapsack_sources(), 'setSolution', id='an incumbent'),
]


@pytest.mark.parametrize(('spec', 'given', 'call'), HINTS)
def test_a_hint_the_solver_refuses_is_loud_not_a_silent_cold_start(spec, given, call):
    """The `_took` guard: a refused hint raises instead of solving cold."""
    tables = _tables(spec, given)
    member = SOLVERS['highs']
    session = member(tables)
    try:
        session.run(tables)
        ws = session.warm_start()
        assert ws is not None
        session._handle = _Refusing(session._handle, call)
        with pytest.raises(sps.SpecsolveError, match='refused'):
            session.warm(ws)
    finally:
        session.close()
