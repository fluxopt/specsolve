"""The solve outcome, and linopy as its oracle.

`relational/status.py` copies linopy's status vocabulary spelling for
spelling, and each solver sink copies linopy's own mapping for its solver.
These tests import linopy and compare, so a divergence on either side fails
here.
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, NamedTuple

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import NoSolutionError
from specsolve.relational.parquet import Metrics, Record, _column_types
from specsolve.relational.sinks.solvers.gurobi import _CONDITION_OF_GUROBI_STATUS, _LINOPY_DIVERGENCES
from specsolve.relational.sinks.solvers.highs import _CONDITION_OF_HIGHS_STATUS
from specsolve.relational.sinks.solvers.xpress import _CONDITION_OF_SOL_STATUS
from specsolve.relational.status import STATUS_TO_TERMINATION_CONDITIONS, SolveStatus
from tests.conftest import CASES

# ---------------------------------------------------------------------------
# linopy as the oracle for the vocabulary
# ---------------------------------------------------------------------------


def test_the_status_rollup_matches_linopy():
    constants = pytest.importorskip('linopy.constants')
    theirs = {
        status.value: {condition.value for condition in conditions}
        for status, conditions in constants.STATUS_TO_TERMINATION_CONDITION_MAP.items()
    }
    assert {k: set(v) for k, v in STATUS_TO_TERMINATION_CONDITIONS.items()} == theirs


def test_the_highs_mapping_matches_linopy():
    assert _linopy_condition_map('Highs', ast.Attribute, 'attr') == _CONDITION_OF_HIGHS_STATUS


def test_the_gurobi_mapping_matches_linopy_where_it_claims_to():
    """The copy, minus three declared exceptions.

    linopy's Gurobi map contradicts Gurobi's own documented status codes in
    three places, each listed in ``_LINOPY_DIVERGENCES`` with its reason.
    Everything else still matches, and every declared divergence still
    diverges.
    """
    theirs = _linopy_condition_map('Gurobi', ast.Constant, 'value')
    assert set(theirs) == set(_CONDITION_OF_GUROBI_STATUS), (
        'linopy and this package no longer cover the same Gurobi status codes'
    )
    for code, condition in theirs.items():
        if code in _LINOPY_DIVERGENCES:
            assert _CONDITION_OF_GUROBI_STATUS[code] != condition, (
                f'linopy now agrees with us on status {code} — drop the entry from _LINOPY_DIVERGENCES'
            )
        else:
            assert _CONDITION_OF_GUROBI_STATUS[code] == condition


def test_the_xpress_mapping_matches_linopy():
    """Copied entry for entry, with nothing claimed as an exception.

    The map is keyed by ``SolStatus`` *value* here and by the enum member
    there, so a member renamed upstream fails here.
    """
    xpress = pytest.importorskip('xpress', reason='the xpress sink needs the [xpress] extra')
    theirs = _linopy_condition_map('Xpress', ast.Attribute, 'attr', ast.Constant, 'value')
    assert theirs, 'linopy no longer spells its Xpress map as SolStatus attributes against strings'
    assert {int(xpress.SolStatus[name]): condition for name, condition in theirs.items()} == _CONDITION_OF_SOL_STATUS


def test_the_xpress_sink_adds_to_linopys_answer_rather_than_contradicting_it():
    """The xpress sink reads a second axis linopy never looks at, so there is
    nothing to disagree with — and the word it reports still has to be one
    linopy defines.
    """
    every = set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())
    assert set(_CONDITION_OF_SOL_STATUS.values()) <= every
    assert 'internal_solver_error' in every, 'the condition the second axis reports is linopy vocabulary'


def test_every_gurobi_divergence_stays_inside_linopys_vocabulary():
    """Every condition this package reports is one linopy also defines."""
    assert set(_CONDITION_OF_GUROBI_STATUS.values()) <= set().union(*STATUS_TO_TERMINATION_CONDITIONS.values())


def _linopy_condition_map(
    solver: str,
    node: type[ast.expr],
    attribute: str,
    value_node: type[ast.expr] | None = None,
    value_attribute: str | None = None,
) -> dict[Any, Any]:
    """linopy's ``CONDITION_MAP`` for *solver*, read out of its source.

    Each solver spells the map differently, so the node type and the attribute
    holding the value are arguments, and the two sides may differ. The map is
    a local inside a method, so there is nothing to import.
    """
    solvers = pytest.importorskip('linopy.solvers')
    tree = ast.parse(inspect.getsource(solvers))
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == solver), None)
    assert cls is not None, f'linopy no longer has a {solver} solver class — re-verify the copy by hand'
    literals = [
        n.value
        for n in ast.walk(cls)
        if isinstance(n, (ast.AnnAssign, ast.Assign)) and 'CONDITION_MAP' in ast.dump(n)
        if isinstance(n.value, ast.Dict)
    ]
    assert literals, f'linopy no longer defines {solver}.CONDITION_MAP as a dict literal — re-verify by hand'
    values, value_of = value_node or node, value_attribute or attribute
    return {
        getattr(key, attribute): getattr(value, value_of)
        for key, value in zip(literals[0].keys, literals[0].values, strict=True)
        if isinstance(key, node) and isinstance(value, values)
    }


# ---------------------------------------------------------------------------
# what the two axes mean here
# ---------------------------------------------------------------------------


def test_ok_means_values_worth_reading_not_optimality():
    """A run stopped at a time limit still has an incumbent."""
    assert SolveStatus('optimal').is_ok
    assert SolveStatus('time_limit').is_ok
    assert SolveStatus('suboptimal').is_ok
    assert not SolveStatus('infeasible').is_ok
    assert not SolveStatus('unbounded').is_ok


def test_an_infeasible_solve_reports_both_axes_and_a_nan_objective():
    with sps.solve(*CASES['INFEASIBLE']) as solution:
        assert solution.status == 'warning'
        assert solution.termination_condition == 'infeasible'
        assert not solution.is_ok
        assert solution.objective != solution.objective, 'nan, not 0.0'


def test_reading_results_without_a_solution_raises():
    """HiGHS returns a full-length vector of zeros whatever the status, so
    handing it back would be indistinguishable from an answer."""
    with sps.solve(*CASES['INFEASIBLE']) as solution:
        with pytest.raises(NoSolutionError, match='infeasible'):
            solution.primal('p')
        with pytest.raises(NoSolutionError, match='infeasible'):
            solution.dual('meet')


def test_a_solve_that_left_no_values_writes_the_record_and_no_frames(tmp_path):
    """An export of a run that did not solve is the record alone."""
    with sps.solve(*CASES['INFEASIBLE']) as solution:
        out = solution.save(tmp_path / 'infeasible')
    assert sorted(entry.name for entry in out.iterdir()) == ['format.json', 'record.parquet'], (
        'the record and the layout it is in; no values, so no primal/, dual/ or expression/'
    )
    record = pl.read_parquet(out / 'record.parquet')
    assert record.row(0, named=True)['termination_condition'] == 'infeasible'
    assert record['objective'].to_list() == [None], 'no objective was reached, so the column holds none'


@pytest.mark.parametrize('case', ['LP', 'MIP', 'INFEASIBLE'])
def test_a_result_hands_back_the_row_its_save_writes(case, tmp_path):
    """`result.record` is the record as a value, so reading it needs no save and no parquet.

    The same row whichever side it is read from: the live result, the file
    `save` wrote, and the result `load_result` reads back off that file.
    """
    with sps.solve(*CASES[case]) as solution:
        record = solution.record
        out = solution.save(tmp_path / case)
    written = Record(**pl.read_parquet(out / 'record.parquet').row(0, named=True))
    assert record == written, 'the property and the file are one row, not two readings of the solve'
    assert sps.load_result(out).record == written, 'and a result read back hands back the row it was read from'
    assert (record.objective is None) == (not record.has_primal), 'null where nothing was reached, never nan'


def test_a_case_that_reached_no_objective_does_not_poison_the_others(tmp_path):
    """A directory per case is a table, and in a table an absent number is null.

    nan is a *number* to every aggregate that meets it: one infeasible case
    among a hundred turns the mean of the hundred into nan.
    """
    for name, case in (('solved', 'LP'), ('unsolved', 'INFEASIBLE')):
        with sps.solve(*CASES[case]) as solution:
            solution.save(tmp_path / name)

    table = pl.read_parquet(tmp_path / '*' / 'record.parquet')
    assert table['objective'].null_count() == 1, 'one of the two cases reached no objective'
    assert table['objective'].is_nan().sum() == 0, 'and it is written as no value rather than as nan'
    assert table['objective'].mean() == table.filter('has_primal')['objective'].item(), (
        'so the mean over the cases is the mean over the ones that solved'
    )


def test_a_record_column_that_names_no_written_type_is_refused_at_import():
    """The schema is derived from `Record`, so a column added to it cannot skip declaring one."""

    class Unwritable(NamedTuple):
        when: bytes

    with pytest.raises(sps.SpecsolveError, match='_WRITTEN_AS'):
        _column_types(Unwritable)

    for row_type in (Record, Metrics):
        assert tuple(_column_types(row_type)) == row_type._fields, (
            f'and {row_type.__name__} derives all of its own, so a field nothing writes fails at import'
        )


# ---------------------------------------------------------------------------
# solver options, and the incumbent question they make reachable
# ---------------------------------------------------------------------------


@pytest.fixture(scope='module')
def knapsack():
    """A MIP big enough that HiGHS does not finish it instantly."""
    import random

    random.seed(0)
    n = 60
    weights = [random.randint(10**6, 2 * 10**6) for _ in range(n)]
    spec = {
        'dimensions': {'i': {'dtype': 'int'}, 'one': {'dtype': 'int'}},
        'parameters': {'w': {'dims': ['i']}, 'cap': {'dims': ['one']}},
        'variables': {'x': {'dims': ['i'], 'domain': 'binary'}},
        'constraints': {'budget': {'dims': ['one'], 'expression': 'sum(x * w, over=i) <= cap'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x * w, over=i)'},
    }
    sources = {
        'i': list(range(n)),
        'one': [0],
        'w': pl.DataFrame({'i': list(range(n)), 'value': [float(v) for v in weights]}),
        'cap': pl.DataFrame({'one': [0], 'value': [float(sum(weights) // 2)]}),
    }
    return spec, sources


def test_solver_options_reach_the_solver(knapsack):
    """Forwarded verbatim, the way linopy's are. `time_limit=0` is the cheapest
    proof: without it this model solves to optimality."""
    spec, sources = knapsack
    with sps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.termination_condition == 'time_limit'
    with sps.solve(spec, sources) as result:
        assert result.termination_condition == 'optimal'


def test_a_time_limit_with_no_incumbent_is_ok_but_unreadable(knapsack):
    """The gap `is_ok` alone cannot see.

    A MIP stopped before it found any feasible point rolls up to `ok`;
    `has_primal` carries the solver's own verdict.
    """
    spec, sources = knapsack
    with sps.solve(spec, sources, solver_options={'time_limit': 0.0}) as result:
        assert result.is_ok, "linopy's rollup says the run was not an error"
        assert not result.has_primal, 'but nothing was found'
        assert result.objective != result.objective, 'nan, not 0.0'
        with pytest.raises(NoSolutionError, match='time_limit'):
            result.primal('x')


def test_an_optimal_solve_is_both_ok_and_readable(knapsack):
    spec, sources = knapsack
    with sps.solve(spec, sources) as result:
        assert result.is_ok
        assert result.has_primal
        assert result.primal('x')['value'].sum() > 0
