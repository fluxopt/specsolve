"""Reduced costs: computed from the duals, so one sign convention on every sink.

Three oracles. A finite difference holds the reader to its docstring with no
solver's convention in the loop: raise the binding bound, re-solve, and the
objective moves by the reduced cost. Each sink's own reduced costs, read off
the loaded solver, agree with the computed ones, which catches a sink whose
duals are signed or scaled differently. And one optimum is done by hand.

Every answer here asks for ``outputs={'reduced_cost'}``; ``test_outputs`` holds
the rule for one that does not.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import NoSolutionError, SpecsolveError
from specsolve.relational.sinks import SOLVERS

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output, Sweep

RTOL = 1e-6

RC: frozenset[Output] = frozenset({'reduced_cost'})

#: A bound moved by this much stays binding on every generator in *SOURCES*,
#: and the price it is moved against does not change.
STEP = 1e-3

#: Three generators against one balance. Whichever way the balance is
#: written and whichever way the objective points, ``a`` and ``c`` each sit
#: on a bound and ``b`` is the marginal one strictly between its bounds.
G = ['a', 'b', 'c']
SOURCES = {
    'g': G,
    'price': pl.DataFrame({'g': G, 'value': [1.0, 2.0, 3.0]}),
    'floor': pl.DataFrame({'g': G, 'value': [0.0, 0.0, 1.0]}),
    'cap': pl.DataFrame({'g': G, 'value': [3.0, 4.0, 4.0]}),
    'total': pl.DataFrame({'value': [5.0]}),
}


def spec(sense: str, balance: str, objective: str = 'sum(price * p, over=g)', domain: str = 'continuous') -> dict:
    """The three generators, with the balance and the objective written as given."""
    return {
        'dimensions': {'g': {'dtype': 'str'}},
        'parameters': {
            'price': {'dims': ['g']},
            'floor': {'dims': ['g']},
            'cap': {'dims': ['g']},
            'total': {'dims': []},
        },
        'variables': {'p': {'dims': ['g'], 'bounds': {'lower': 'floor', 'upper': 'cap'}, 'domain': domain}},
        'constraints': {'balance': {'dims': [], 'expression': balance}},
        'objective': {'sense': sense, 'expression': objective},
    }


CASES = [
    pytest.param('minimize', 'sum(p, over=g) >= total', id='min-ge'),
    pytest.param('minimize', 'total <= sum(p, over=g)', id='min-le-reversed'),
    pytest.param('minimize', 'sum(p, over=g) == total', id='min-eq'),
    pytest.param('maximize', 'sum(p, over=g) <= total', id='max-le'),
    pytest.param('maximize', 'total >= sum(p, over=g)', id='max-ge-reversed'),
    pytest.param('maximize', 'total == sum(p, over=g)', id='max-eq'),
]


def native(solver_name: str, handle: Any) -> list[float]:
    """The loaded solver's own reduced costs, in column order."""
    if solver_name == 'highs':
        return list(handle.getSolution().col_dual)
    if solver_name == 'gurobi':
        return handle.getAttr('RC', handle.getVars())
    return list(handle.getRedCosts())


def objective(model_spec: dict, sources: dict, solver_name: str) -> float:
    with sps.solve(model_spec, sources, solver_name=solver_name, outputs=RC) as answer:
        return answer.objective


@pytest.mark.parametrize(('sense', 'balance'), CASES)
def test_the_reduced_cost_is_what_the_objective_pays_per_unit_of_binding_bound(
    solver_name: str, sense: str, balance: str
) -> None:
    """Raise each generator's binding bound by a step, and the objective rises by the reduced cost times it."""
    model_spec = spec(sense, balance)
    with sps.solve(model_spec, SOURCES, solver_name=solver_name, outputs=RC) as answer:
        base = answer.objective
        primal = answer.primal('p')['value'].to_list()
        reduced = answer.reduced_cost('p')['value'].to_list()

    floor, cap = SOURCES['floor']['value'].to_list(), SOURCES['cap']['value'].to_list()
    for i, (x, rc) in enumerate(zip(primal, reduced, strict=True)):
        binding = 'cap' if x == pytest.approx(cap[i]) else 'floor' if x == pytest.approx(floor[i]) else None
        if binding is None:
            assert rc == pytest.approx(0.0, abs=RTOL), f'{G[i]} is between its bounds, so no bound prices it'
            continue
        moved = SOURCES[binding].with_columns(
            pl.when(pl.col('g') == G[i]).then(pl.col('value') + STEP).otherwise('value')
        )
        rate = (objective(model_spec, {**SOURCES, binding: moved}, solver_name) - base) / STEP
        assert rc == pytest.approx(rate, rel=RTOL, abs=RTOL), (
            f"{G[i]}'s {binding} moves the objective at its reduced cost"
        )


@pytest.mark.parametrize(('sense', 'balance'), CASES)
def test_the_reduced_cost_agrees_with_the_solvers_own(solver_name: str, sense: str, balance: str) -> None:
    """Every sink's native reduced costs carry the same convention, so a sink whose duals drifted shows here."""
    with sps.build(spec(sense, balance), SOURCES) as model, model.solve(solver_name, outputs=RC) as answer:
        computed = answer.reduced_cost('p')['value'].to_list()
        own = native(solver_name, model._engine._solver.handle)
    assert computed == pytest.approx(own, abs=RTOL), f"{solver_name}'s own reduced costs agree with the computed ones"


@pytest.mark.parametrize(('sense', 'balance'), CASES)
def test_the_reduced_cost_does_not_depend_on_how_the_balance_is_written(sense: str, balance: str) -> None:
    """By hand: ``b`` is marginal at a price of 2, so ``a`` at its bound is worth 1 less and ``c`` 1 more.

    Minimizing, ``a`` is at its cap and ``c`` at its floor; maximizing, the
    reverse. The reduced costs come out the same either way, because each is
    the rate in the bound that binds.
    """
    with sps.solve(spec(sense, balance), SOURCES, outputs=RC) as answer:
        assert answer.reduced_cost('p')['value'].to_list() == pytest.approx([-1.0, 0.0, 1.0], abs=RTOL), (
            'one reduced cost per generator, in label order'
        )


@pytest.mark.parametrize(
    ('sense', 'objective_written'),
    [
        pytest.param('minimize', 'sum(price * p * p, over=g)', id='min'),
        pytest.param('maximize', 'sum(-price * p * p, over=g)', id='max'),
    ],
)
def test_a_quadratic_objective_contributes_its_gradient_at_the_solution(
    solver_name: str, sense: str, objective_written: str
) -> None:
    """The cost is the objective's gradient at the primal, which a squared term makes depend on it."""
    if 'quadratic_objective' not in SOLVERS[solver_name].capabilities.supports:
        pytest.skip(f'{solver_name} takes no quadratic objective')
    model_spec = spec(sense, 'sum(p, over=g) == total', objective_written)
    with sps.build(model_spec, SOURCES) as model, model.solve(solver_name, outputs=RC) as answer:
        computed = answer.reduced_cost('p')['value'].to_list()
        own = native(solver_name, model._engine._solver.handle)
    assert computed == pytest.approx(own, abs=RTOL), f"{solver_name}'s own reduced costs agree with the computed ones"
    assert computed[2] != pytest.approx(0.0, abs=RTOL), 'c is held at its floor, so its bound has a price'


def test_a_quadratic_row_contributes_its_gradient_weighted_by_its_dual() -> None:
    """``p·q >= 4`` holds both columns strictly inside their bounds, so each reduced cost is zero.

    Leaving out the row's quadratic gradient would read 1, the objective's cost.
    """
    pytest.importorskip('gurobipy', reason='a quadratic constraint has no other solver sink')
    from tests.test_quadratic_constraint import SOURCES as QUADRATIC_SOURCES
    from tests.test_quadratic_constraint import SPEC as QUADRATIC

    with sps.solve(
        QUADRATIC, QUADRATIC_SOURCES, solver_name='gurobi', solver_options={'QCPDual': 1}, outputs=RC
    ) as answer:
        for name in ('p', 'q'):
            assert answer.reduced_cost(name)['value'].to_list() == pytest.approx([0.0, 0.0], abs=1e-3), (
                f'{name} is interior on both generators; 1e-3 is the barrier the QCP stops at'
            )


def test_an_integer_variable_leaves_no_reduced_cost_and_says_why() -> None:
    """A reduced cost is computed from the duals, so it is refused where they are, with their message."""
    integer = spec('minimize', 'sum(p, over=g) >= total', domain='integer')
    with sps.solve(integer, SOURCES, outputs=RC) as answer, pytest.raises(SpecsolveError, match='mixed-integer'):
        answer.reduced_cost('p')


@pytest.mark.parametrize('saved', [pytest.param(False, id='live'), pytest.param(True, id='saved')])
def test_an_integer_answer_bridged_to_xarray_gives_the_duals_reason(tmp_path: Path, saved: bool) -> None:
    """A bridge reads every name the output holds, and a mixed-integer answer holds none.

    It listed nothing and failed unpacking that, rather than giving the reason.
    """
    pytest.importorskip('xarray', reason='specsolve does not install xarray, and the floors environment has none')
    integer = spec('minimize', 'sum(p, over=g) >= total', domain='integer')
    with sps.solve(integer, SOURCES, outputs=RC) as answer:
        if saved:
            answer.save(tmp_path)
        read = sps.load_result(tmp_path) if saved else answer
        with pytest.raises(SpecsolveError, match='mixed-integer'):
            read.to_dataset(kind='reduced_cost')


def test_a_saved_answer_reads_back_the_reduced_costs_it_was_asked_for(tmp_path: Path) -> None:
    with sps.solve(spec('minimize', 'sum(p, over=g) >= total'), SOURCES, outputs=RC) as answer:
        live = answer.reduced_cost('p')
        answer.save(tmp_path)
    assert sps.load_result(tmp_path).reduced_cost('p').equals(live)


def test_a_saved_integer_answer_gives_the_duals_reason(tmp_path: Path) -> None:
    """The reason a mixed-integer solve has no duals is saved once, and the reduced costs read it back."""
    with sps.solve(spec('minimize', 'sum(p, over=g) >= total', domain='integer'), SOURCES, outputs=RC) as answer:
        answer.save(tmp_path)
    assert not (tmp_path / 'reduced_cost').exists(), 'nothing to write where there are no duals'
    with pytest.raises(SpecsolveError, match='mixed-integer'):
        sps.load_result(tmp_path).reduced_cost('p')


def test_a_solve_not_asked_for_reduced_costs_names_the_output() -> None:
    with (
        sps.solve(spec('minimize', 'sum(p, over=g) >= total'), SOURCES) as answer,
        pytest.raises(SpecsolveError, match=r"outputs=\{'reduced_cost'\}"),
    ):
        answer.reduced_cost('p')


def test_an_infeasible_solve_has_no_values_to_read() -> None:
    """The refusal ``primal`` gives, ahead of the one about the duals."""
    short = {**SOURCES, 'total': pl.DataFrame({'value': [100.0]})}
    with (
        sps.solve(spec('minimize', 'sum(p, over=g) >= total'), short, outputs=RC) as answer,
        pytest.raises(NoSolutionError),
    ):
        answer.reduced_cost('p')


def test_a_closed_result_says_so() -> None:
    answer = sps.solve(spec('minimize', 'sum(p, over=g) >= total'), SOURCES, outputs=RC)
    answer.close()
    with pytest.raises(SpecsolveError, match='closed'):
        answer.reduced_cost('p')


def test_a_variable_nothing_declares_is_a_key_error() -> None:
    with sps.solve(spec('minimize', 'sum(p, over=g) >= total'), SOURCES, outputs=RC) as answer, pytest.raises(KeyError):
        answer.reduced_cost('balance')


def test_the_frame_is_tidy_over_the_variable_dims() -> None:
    with sps.solve(spec('minimize', 'sum(p, over=g) >= total'), SOURCES, outputs=RC) as answer:
        frame = answer.reduced_cost('p')
    assert frame.columns == ['g', 'value'], 'the primal frame shape: the dims, then the value'
    assert frame['g'].to_list() == G, 'rows come back in label order'
    assert np.isfinite(frame['value'].to_numpy()).all()


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

#: Two scenarios. At a total of 5 ``b`` is marginal at a price of 2; at 9 ``b``
#: is capped and ``c`` is marginal at 3, so every generator's reduced cost moves.
SCENARIOS = {**SOURCES, 'total': pl.DataFrame({'scenario': ['low', 'high'], 'value': [5.0, 9.0]})}
BY_HAND = {'low': [-1.0, 0.0, 1.0], 'high': [-2.0, -1.0, 0.0]}


def _swept(tmp_path: Path, model_spec: dict, how: str) -> Sweep:
    axis = sps.EachCoordinate('scenario')
    if how == 'archived':
        sps.solve_over(model_spec, SCENARIOS, axis, outputs=RC, archive=tmp_path / 'run.zip')
        archive = sps.load_archive(tmp_path / 'run.zip')
        assert isinstance(archive, sps.archive.SweepArchive)
        return archive.sweep
    spill = {'spill_to': tmp_path / 'spill'} if how == 'spilled' else {}
    return sps.solve_over(model_spec, SCENARIOS, axis, outputs=RC, **spill)


SHAPES = [pytest.param(how, id=how) for how in ('held', 'spilled', 'archived')]


@pytest.mark.parametrize('how', SHAPES)
def test_a_sweep_reads_each_slices_reduced_costs(tmp_path: Path, how: str) -> None:
    """Each slice's reduced costs, keyed by slice, as that slice solved alone reads them."""
    sweep = _swept(tmp_path, spec('minimize', 'sum(p, over=g) >= total'), how)
    for (scenario,), frame in sweep.reduced_cost('p').partition_by('scenario', as_dict=True).items():
        assert frame['value'].to_list() == pytest.approx(BY_HAND[scenario], abs=RTOL), (
            f'slice {scenario!r} is priced at its own marginal generator'
        )


@pytest.mark.parametrize('how', SHAPES)
def test_an_integer_sweep_gives_the_duals_reason(tmp_path: Path, how: str) -> None:
    sweep = _swept(tmp_path, spec('minimize', 'sum(p, over=g) >= total', domain='integer'), how)
    with pytest.raises(SpecsolveError, match='mixed-integer'):
        sweep.reduced_cost('p')
