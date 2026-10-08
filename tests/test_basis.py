"""Basis status: where each column and row sat when the solve ended, in one vocabulary on every sink.

The oracle is a two-variable LP whose vertex is worked out by hand, built so
that every status but ``superbasic`` appears: a column strictly inside its
bounds, one at each bound, one whose bounds are equal, and a row binding from
each side next to one that is loose.
"""

from __future__ import annotations

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.answer_layout import NO_BASIS

if TYPE_CHECKING:
    from pathlib import Path

    from specsolve.types import Output, Sweep

BASIS: frozenset[Output] = frozenset({'basis'})

SPEC = {
    'variables': {
        'x': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}},
        'y': {'dims': [], 'bounds': {'lower': 0, 'upper': 10}},
        'z': {'dims': [], 'bounds': {'lower': 3, 'upper': 3}},
        'w': {'dims': [], 'bounds': {'lower': -10, 'upper': 10}},
        'u': {'dims': [], 'bounds': {'lower': 0, 'upper': 2}},
    },
    'constraints': {
        'cap': {'dims': [], 'expression': 'x + y <= 4'},
        'floor': {'dims': [], 'expression': 'x - y >= 1'},
        'fixed': {'dims': [], 'expression': 'w + z == 5'},
        'loose': {'dims': [], 'expression': 'x <= 100'},
    },
    'objective': {'sense': 'maximize', 'expression': 'x + 2*y + w + u'},
}

#: ``x = 2.5`` and ``y = 1.5`` sit where ``cap`` and ``floor`` cross, ``w = 2``
#: is what ``fixed`` leaves it, ``u`` is pushed onto its cap, and ``z`` cannot
#: move.
BY_HAND_COLUMNS = {'x': 'basic', 'y': 'basic', 'z': 'fixed', 'w': 'basic', 'u': 'at_upper'}
BY_HAND_ROWS = {'cap': 'at_upper', 'floor': 'at_lower', 'fixed': 'fixed', 'loose': 'basic'}


def test_every_sink_reads_the_same_basis_in_the_same_words(solver_name: str) -> None:
    with sps.solve(SPEC, {}, solver_name=solver_name, outputs=BASIS) as answer:
        columns = {name: answer.variable_basis(name)['value'].item() for name in BY_HAND_COLUMNS}
        rows = {name: answer.constraint_basis(name)['value'].item() for name in BY_HAND_ROWS}
    assert columns == BY_HAND_COLUMNS, 'the columns of the vertex worked out by hand'
    assert rows == BY_HAND_ROWS, 'a binding row reads the side its right-hand side bounds, a loose one basic'


def test_a_basis_reads_back_as_an_enum_of_the_five_statuses() -> None:
    with sps.solve(SPEC, {}, outputs=BASIS) as answer:
        dtype = answer.variable_basis('x')['value'].dtype
    assert dtype == pl.Enum(['basic', 'at_lower', 'at_upper', 'fixed', 'superbasic']), 'the five words, in order'


def test_a_saved_answer_reads_back_its_basis(tmp_path: Path) -> None:
    with sps.solve(SPEC, {}, outputs=BASIS) as answer:
        live = answer.variable_basis('u'), answer.constraint_basis('cap')
        answer.save(tmp_path)
    loaded = sps.load_result(tmp_path)
    assert loaded.variable_basis('u').equals(live[0])
    assert loaded.constraint_basis('cap').equals(live[1])


def test_a_constraint_named_as_a_variable_keeps_its_own_basis(tmp_path: Path) -> None:
    """A constraint may share a variable's name, so the two halves of a basis lie in directories of their own.

    ``loose`` is renamed ``x`` here: one directory for both halves would hold
    one ``x.parquet``, and one of the two statuses would be lost.
    """
    constraints = {('x' if name == 'loose' else name): row for name, row in SPEC['constraints'].items()}
    spec = {**SPEC, 'constraints': constraints}
    sps.solve(spec, {}, outputs=BASIS, archive=tmp_path / 'case').save(tmp_path / 'saved')
    loaded = sps.load_result(tmp_path / 'saved')
    archived = sps.load_archive(tmp_path / 'case')
    assert isinstance(archived, sps.archive.ResultArchive)
    catalog = pl.read_parquet(tmp_path / 'case' / 'catalog.parquet').filter(pl.col('name') == 'x')

    for answer in (loaded, archived.result):
        assert answer.variable_basis('x')['value'].item() == 'basic', 'the variable x is inside its bounds'
        assert answer.constraint_basis('x')['value'].item() == 'basic', 'the constraint x is not binding'
    assert {'answer/variable_basis/x.parquet', 'answer/constraint_basis/x.parquet'} <= set(catalog['path']), (
        'the catalog names each half of the basis at its own path'
    )


@pytest.mark.parametrize(('reader', 'name'), [('variable_basis', 'x'), ('constraint_basis', 'cap')])
def test_a_solve_not_asked_for_its_basis_names_the_output(reader: str, name: str) -> None:
    with (
        sps.solve(SPEC, {}) as answer,
        pytest.raises(SpecsolveError, match=r"outputs=\{'basis'\}"),
    ):
        getattr(answer, reader)(name)


INTEGER = {
    **SPEC,
    'variables': {**SPEC['variables'], 'u': {'dims': [], 'bounds': {'lower': 0, 'upper': 2}, 'domain': 'integer'}},
}

#: Each sink's own way to an LP answer that is not a vertex. HiGHS's presolve
#: solves this model outright and then reports its status as unknown.
NO_CROSSOVER = {
    'highs': {'solver': 'ipm', 'run_crossover': 'off', 'presolve': 'off'},
    'gurobi': {'Method': 2, 'Crossover': 0},
    'xpress': {'lpflags': 4, 'crossover': 0},
}


@pytest.mark.parametrize(
    ('spec', 'how'),
    [
        pytest.param(INTEGER, None, id='mixed-integer'),
        pytest.param(SPEC, 'no-crossover', id='interior-point-without-crossover'),
    ],
)
def test_a_solve_that_ended_on_no_basis_says_why(solver_name: str, spec: dict, how: str | None) -> None:
    options = NO_CROSSOVER[solver_name] if how else None
    with (
        sps.solve(spec, {}, solver_name=solver_name, solver_options=options, outputs=BASIS) as answer,
        pytest.raises(SpecsolveError, match=NO_BASIS[:40]),
    ):
        answer.variable_basis('x')


def test_a_model_with_a_quadratic_constraint_ends_on_no_basis() -> None:
    """Gurobi solves one by barrier, which has no crossover to a vertex for it to take."""
    pytest.importorskip('gurobipy')
    spec = {**SPEC, 'constraints': {**SPEC['constraints'], 'disc': {'dims': [], 'expression': 'x*x + y*y <= 9'}}}
    with (
        sps.solve(spec, {}, solver_name='gurobi', outputs=BASIS) as answer,
        pytest.raises(SpecsolveError, match=NO_BASIS[:40]),
    ):
        answer.constraint_basis('cap')


# ---------------------------------------------------------------------------
# sweeps
# ---------------------------------------------------------------------------

#: The dispatch model of ``test_strategy``: ``balance`` is an equality, so it
#: is ``fixed`` or ``basic`` in every scenario, never at one side.
SHAPES = [pytest.param(how, id=how) for how in ('held', 'process-pool', 'spilled', 'archived')]


def _swept(tmp_path: Path, how: str, outputs: frozenset[Output] = BASIS) -> Sweep:
    from tests.test_strategy import DISPATCH, scenario_sources

    axis = sps.EachCoordinate('scenario')
    if how == 'archived':
        sps.solve_over(DISPATCH, scenario_sources(), axis, outputs=outputs, archive=tmp_path / 'run.zip')
        archive = sps.load_archive(tmp_path / 'run.zip')
        assert isinstance(archive, sps.archive.SweepArchive)
        return archive.sweep
    if how == 'process-pool':
        with ProcessPoolExecutor(2, mp_context=multiprocessing.get_context('spawn')) as pool:
            return sps.solve_over(DISPATCH, scenario_sources(), axis, outputs=outputs, executor=pool)
    spill = {'spill_to': tmp_path / 'spill'} if how == 'spilled' else {}
    return sps.solve_over(DISPATCH, scenario_sources(), axis, outputs=outputs, **spill)


@pytest.mark.parametrize('how', SHAPES)
def test_a_sweep_reads_each_slices_basis(tmp_path: Path, how: str) -> None:
    sweep = _swept(tmp_path, how)
    rows = sweep.constraint_basis('balance')
    columns = sweep.variable_basis('p')
    assert rows.height == 12, 'three scenarios of four snapshots'
    assert set(rows['value']) <= {'fixed', 'basic'}, 'an equality is never at one side of its right-hand side'
    assert columns['value'].dtype == pl.Enum(['basic', 'at_lower', 'at_upper', 'fixed', 'superbasic'])


def test_a_sweep_of_slices_that_ended_on_no_basis_says_why(tmp_path: Path) -> None:
    from tests.test_strategy import DISPATCH, scenario_sources

    sweep = sps.solve_over(
        DISPATCH,
        scenario_sources(),
        sps.EachCoordinate('scenario'),
        solver_options=NO_CROSSOVER['highs'],
        outputs=BASIS,
    )
    with pytest.raises(SpecsolveError, match=NO_BASIS[:40]):
        sweep.variable_basis('p')


@pytest.mark.parametrize('side', ['columns', 'rows'])
def test_a_basis_that_does_not_span_the_model_is_refused(monkeypatch: pytest.MonkeyPatch, side: str) -> None:
    """A basis is positional, as a primal is, so one a member reads short is a different model's.

    The double overrides `_basis`, the half a sink writes; the guard is the
    base's `run` around it, so every sink is checked.
    """
    from specsolve.relational.sinks import SOLVERS
    from specsolve.relational.sinks.solvers.highs import Highs

    class Crooked(Highs):
        def _basis(self):
            columns, rows = super()._basis()
            return (columns[:-1], rows) if side == 'columns' else (columns, rows[:-1])

    monkeypatch.setitem(SOLVERS, 'highs', Crooked)
    which = {'columns': 'column basis values for a model with 5', 'rows': 'row basis values for a model with 4'}
    with pytest.raises(SpecsolveError, match=which[side]):
        sps.solve(SPEC, {}, outputs=BASIS)
