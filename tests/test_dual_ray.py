"""The certificate an infeasible solve leaves, and the one convention the sinks sign it with.

``dual_ray`` is the only reader that answers on an infeasible solve. HiGHS and
Xpress sign a row the way the row is written; Gurobi signs it the other way and
the sink negates, so the three come back indistinguishable.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import SpecsolveError
from specsolve.relational.sinks import SOLVERS

#: Two rows that cannot both hold, and no bound in the argument: ``p`` is held
#: only by a lower bound of zero, which delivers nothing into the combination,
#: so the certificate is the simple ``Σ weight * rhs > 0``.
SHORT = {
    'dimensions': {'snapshot': {'dtype': 'int'}},
    'parameters': {'need': {'dims': ['snapshot']}, 'cap': {'dims': ['snapshot']}},
    'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0}}},
    'constraints': {
        'demand': {'dims': ['snapshot'], 'expression': 'p >= need'},
        'limit': {'dims': ['snapshot'], 'expression': 'p <= cap'},
    },
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

NEED = [3.0, 0.0]
CAP = [1.0, 5.0]
SOURCES = {
    'snapshot': pl.DataFrame({'snapshot': [0, 1]}),
    'need': pl.DataFrame({'snapshot': [0, 1], 'value': NEED}),
    'cap': pl.DataFrame({'snapshot': [0, 1], 'value': CAP}),
}

#: What each sink has to be asked before it will certify anything, as the
#: engine's refusal names it. HiGHS needs nothing.
ASKED: dict[str, dict[str, Any]] = {'highs': {}, 'gurobi': {'InfUnbdInfo': 1}, 'xpress': {'presolve': 0}}


def certified(solver_name: str, options: dict[str, Any] | None) -> sps.Result:
    """*SHORT* solved, which is infeasible on every sink."""
    with sps.build(SHORT, SOURCES) as model:
        answer = model.solve(solver_name, solver_options=options)
        assert answer.termination_condition == 'infeasible'
        return answer


def test_the_rows_that_cannot_hold_are_weighted_into_a_contradiction(solver_name: str) -> None:
    """Farkas' lemma, as the reader hands it over: the combined row demands what no column can deliver."""
    answer = certified(solver_name, ASKED[solver_name])
    demand = answer.dual_ray('demand')['value'].to_list()
    limit = answer.dual_ray('limit')['value'].to_list()
    weighted = sum(w * rhs for w, rhs in zip(demand, NEED, strict=True))
    weighted += sum(w * rhs for w, rhs in zip(limit, CAP, strict=True))
    assert weighted > 0, (
        f'the certificate has to come out positive to be one: demand {demand}, limit {limit}, weighted {weighted}'
    )
    answer.close()


def test_every_sink_signs_a_row_the_same_way(solver_name: str) -> None:
    """The point of one convention: a driver reads a ray without asking who solved.

    The values as well as the signs, since these two rows are the same
    combination whichever simplex found it.
    """
    answer = certified(solver_name, ASKED[solver_name])
    assert answer.dual_ray('demand')['value'].to_list() == pytest.approx([1.0, 0.0]), (
        'a >= row that cannot be met is weighted positively, and the slack snapshot not at all'
    )
    assert answer.dual_ray('limit')['value'].to_list() == pytest.approx([-1.0, 0.0]), (
        'the <= row it contradicts carries the opposite sign'
    )
    answer.close()


def test_a_ray_spans_the_constraint_it_names(solver_name: str) -> None:
    """One weight per row, laid out over the constraint's own coordinates — ``dual``'s shape."""
    answer = certified(solver_name, ASKED[solver_name])
    ray = answer.dual_ray('demand')
    assert ray.columns == ['snapshot', 'value'], 'a ray is tidy over the constraint dims, like every other reader'
    assert ray['snapshot'].to_list() == [0, 1], 'rows come back in label order'
    answer.close()


@pytest.mark.parametrize(
    ('needs', 'names'),
    [
        pytest.param('gurobi', 'InfUnbdInfo', id='gurobi-infunbdinfo'),
        pytest.param('xpress', 'presolve', id='xpress-presolve'),
    ],
)
def test_a_sink_that_was_not_asked_names_what_to_ask(needs: str, names: str) -> None:
    """The refusal is the rewrite. Both solvers compute a certificate only on request."""
    if not SOLVERS[needs].is_available():
        pytest.skip(f'{needs} is not installed here')
    answer = certified(needs, None)
    with pytest.raises(SpecsolveError, match=names):
        answer.dual_ray('demand')
    answer.close()


def test_a_solve_that_found_an_answer_has_nothing_to_certify() -> None:
    """A ray is about the absence of a solution, so a solve with one says to read dual() instead."""
    feasible = {**SOURCES, 'need': pl.DataFrame({'snapshot': [0, 1], 'value': [0.5, 0.0]})}
    with sps.solve(SHORT, feasible) as answer:
        assert answer.termination_condition == 'optimal'
        with pytest.raises(SpecsolveError, match='terminated'):
            answer.dual_ray('demand')


def test_a_closed_result_says_so_before_it_says_anything_else() -> None:
    """A ray is released with the primals, and reports the same way they do."""
    answer = certified('highs', {})
    answer.close()
    with pytest.raises(SpecsolveError, match='closed'):
        answer.dual_ray('demand')


def test_a_constraint_nothing_declares_is_a_key_error() -> None:
    """Named like every other reader, and wrong in the same way."""
    answer = certified('highs', {})
    with pytest.raises(KeyError):
        answer.dual_ray('nope')
    answer.close()
