"""Duals: the shadow price read-back, and the two models that have none.

A nodal balance's dual is the price at that node, which is why this is a
headline output rather than a diagnostic — and why the differential test below
compares *values*, not just presence: a sign convention that disagreed with
linopy would be a silently wrong answer of exactly the kind the two-lane claim
exists to catch.

The other half of the feature is the refusals. A MILP has no dual solution and
an infeasible solve has no valid one, and in both cases returning zeros would
look like an answer.
"""

from __future__ import annotations

import re

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.errors import NoSolutionError, SpecsolveError
from tests.differential import differential
from tests.oracle import pd  # through the guard: a bare import would beat it
from tests.test_milp import COMMITMENT_YAML

DUAL_RTOL = 1e-9


def test_dual_matches_the_linopy_lane(dispatch_yaml, dispatch_inputs):
    """The price at each snapshot, both lanes, same sign and same magnitude."""
    data = dispatch_inputs

    with differential(dispatch_yaml, data) as run:
        got = run.result.dual('power_balance')

        assert got.columns == ['snapshot', 'value']
        assert got.height == len(data['snapshot'])

        oracle = run.model.constraints['power_balance'].dual
        expected = pd.Series(np.asarray(oracle), index=np.asarray(oracle.indexes['snapshot']))
        actual = got.sort('snapshot')['value'].to_numpy()

        assert actual == pytest.approx(expected.sort_index().to_numpy(), rel=DUAL_RTOL)

        assert set(np.round(actual, 6)) <= set(np.round(data['cost'].to_numpy(), 6)), (
            'a price is what the marginal unit costs: with distinct costs and a binding balance, '
            'every dual sits on one of the generator costs'
        )


def test_dual_respects_the_where_mask(dispatch_yaml, dispatch_inputs):
    """Duals are a label join, so a masked row is absent — never a zero."""
    data = dispatch_inputs
    trimmed = dict(data, snapshot=data['snapshot'][:12], load=data['load'].iloc[:12])

    with differential(dispatch_yaml, trimmed) as run:
        got = run.result.dual('power_balance')
        assert got.height == 12
        assert sorted(got['snapshot'].to_list()) == list(range(12))


def test_milp_refuses_duals_and_names_the_variable(commitment_inputs):
    """Integrality is decidable from the program, so the message says which.

    **A deliberate divergence from the oracle, not an oversight.** Asked for
    the duals of this same model, linopy hands back an array of zeros — the
    ``dual`` entry exists and every value is ``0.0`` — which is a plausible
    number with no signal attached, the ``bug:silent`` class. (On an
    *unsolved* model the two lanes agree that it must raise: linopy through
    its ``has_optimized_model`` gate, we through ``_require_solution`` — see
    the infeasible test below.)

    Hard rule 3 governs the *language*, and both lanes still accept this
    model and agree on its objective; what differs is a post-solve read-back
    that has no defined answer. #78 fixed the direction: a model with any
    binary or integer variable must raise, naming the reason, not return
    zeros. Parity here would be parity with the bug.
    """
    data = commitment_inputs

    with differential(COMMITMENT_YAML, data) as run:
        with pytest.raises(SpecsolveError) as excinfo:
            run.result.dual('balance')

        message = str(excinfo.value)
        assert 'mixed-integer' in message
        assert "'u'" in message, 'the refusal should name the non-continuous variable'
        assert len(run.result.primal('u')) > 0, 'the primal is still perfectly readable — only duals are undefined'


def test_infeasible_solve_refuses_duals(dispatch_yaml, dispatch_inputs):
    """No values at all is the *other* refusal — the one `primal` shares.

    `dual` goes through `_require_solution` before it looks at duals, so an
    infeasible solve raises `NoSolutionError` exactly as `primal` does rather
    than reporting the narrower "this model has no duals".
    """
    data = dispatch_inputs
    data = dict(data, load=pd.Series(1e6, index=data['snapshot']))  # more than every generator together

    with sps.solve(dispatch_yaml, data) as result:
        assert not result.has_primal
        assert result.termination_condition == 'infeasible'

        with pytest.raises(NoSolutionError, match='cannot read the dual'):
            result.dual('power_balance')
        with pytest.raises(NoSolutionError):  # the same refusal, same class, for the primal
            result.primal('p')


RAMP_BLOCK = {
    'dimensions': {'t': {'dtype': 'int'}},
    'parameters': {'lim': {'dims': ['t']}},
    'variables': {'p': {'dims': ['t'], 'bounds': {'lower': 0, 'upper': 100}}},
    'constraints': {
        'ramp_up': {'dims': ['t'], 'expression': "p - shift(p, along=t, offset=1, edge='wrap') <= lim"},
        'ramp_down': {'dims': ['t'], 'expression': "shift(p, along=t, offset=1, edge='wrap') - p <= lim"},
    },
    'objective': {'sense': 'maximize', 'expression': 'sum(p, over=t)'},
}


@pytest.mark.parametrize(
    ('asked', 'expected'),
    [
        pytest.param(
            'ramp',
            'but 2 begin with it — ramp_down, ramp_up',
            id='a-family-name-where-nearest-match-would-imply-the-sibling-is-absent',
        ),
        pytest.param('ramp_dwn', "unknown constraint 'ramp_dwn'. Did you mean 'ramp_down'?", id='a-typo'),
        pytest.param('zzz', 'Declared: ramp_down, ramp_up.', id='nothing-like-it'),
    ],
)
def test_reading_back_an_unknown_name_says_what_was_built(asked, expected):
    """The error rules ask a message to name the fix, and this is where it matters most.

    One name can expand into several — a `piecewise:` block becomes a handful of
    constraints, and a rule split by regime is conventionally ``x`` and
    ``x_initial`` — so a caller can reasonably ask for a name that was never
    built. A bare ``KeyError`` left them to find out from the source which one
    was.

    Single-line on purpose: these raise ``KeyError``, whose ``str`` is the repr
    of its argument, so a newline would reach the reader as a literal ``\\n``.
    """
    import polars as pl

    sources = {'t': [0, 1, 2], 'lim': pl.DataFrame({'t': [0, 1, 2], 'value': [10.0, 10.0, 10.0]})}
    with sps.solve(RAMP_BLOCK, sources) as sol, pytest.raises(KeyError, match=re.escape(expected)):
        sol.dual(asked)


#: One row, "p is at least 2 + b", written each way round, with how raising `b`
#: moves the row's right side against its left: `+1` where it raises
#: `rhs - lhs`, `-1` where it lowers it.
WRITTEN = [
    pytest.param('p >= 2 + b', 1, id='b-on-the-right'),
    pytest.param('2 + b <= p', -1, id='b-on-the-left'),
    pytest.param('-p <= -2 - b', -1, id='negated'),
    pytest.param('p - b >= 2', 1, id='b-moved-to-the-left'),
    pytest.param('p == 2 + b', 1, id='an-equality'),
    pytest.param('2 + b == p', -1, id='an-equality-the-other-way-round'),
]


def _floor_model(row: str, sense: str) -> dict[str, object]:
    return {
        'dimensions': {'snapshot': {'dtype': 'int'}},
        'parameters': {'b': {'dims': ['snapshot']}},
        'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0}}},
        'constraints': {'floor': {'dims': ['snapshot'], 'expression': row}},
        'objective': {'sense': sense, 'expression': 'sum(3 * p)' if sense == 'minimize' else 'sum(-3 * p)'},
    }


@pytest.mark.parametrize('sense', ['minimize', 'maximize'])
@pytest.mark.parametrize(('row', 'moves'), WRITTEN)
def test_a_dual_is_the_rate_of_the_optimum_in_its_rows_right_side(
    solver_name: str, row: str, moves: int, sense: str
) -> None:
    """The sign mathspec defines `dual(c)` with: read `lhs <= rhs` as `lhs <= rhs + d`, and the dual is the rate in `d`.

    The same for every comparator and under either sense, so which side a
    term is written on decides the sign. A finite difference in `b` is that
    rate times how `b` moves the right side, and the row binds, so the
    difference is exact.
    """
    step = 1e-3

    def solved(b: float) -> tuple[float, float]:
        sources = {'snapshot': pl.DataFrame({'snapshot': [0]}), 'b': pl.DataFrame({'snapshot': [0], 'value': [b]})}
        with sps.solve(_floor_model(row, sense), sources, solver_name) as answer:
            return answer.objective, answer.dual('floor')['value'].item()

    objective, dual = solved(1.0)
    rate = (solved(1.0 + step)[0] - objective) / step
    assert dual == pytest.approx(moves * rate, abs=1e-6), (
        f'{row} under {sense}: the dual is {dual}, and the optimum moves {rate} per unit of b'
    )
