"""Every expression read at one chosen point, on both lanes, with no solver.

``test_expression_sweep.py`` puts each expression in a constraint and compares
the objectives. That sees a term only where its row binds: ``sum(y + w, over=f)
<= 10`` builds a wrong right-hand side on the relational lane, and the sweep
passes it, because ``y`` is not in the objective. Here every variable is held at
a seeded value and each expression is *read*, as ``result.evaluate`` reads it, on
both lanes (``tests.differential.at_a_point``). Every coordinate of every term
is compared, and three limits of the solve sweep fall away:

- **no solve**, so a case costs one read on each lane rather than a build and
  two solves;
- **no feasible data to keep**, so nothing is skipped for having no answer;
- **every degree**, so a quadratic expression and a variable-free one are cases
  rather than operands only.

Three tables, one fixture:

- ``node`` — every expression :mod:`tests.expression_space` builds to
  ``--sweep-depth``, sharded by ``--sweep-shard`` like the solve sweep;
- ``rewrite`` — the rewrites that must not move a value, compared frame to frame;
- ``OPERATORS`` — one line per construct the generator does not build. **Add a
  case here** as a ``pytest.param`` whose ``id`` names what it exercises.

Agreement is the claim, not the value: a rule both lanes break the same way
passes here (#311). The values the language promises are pinned in
``test_evaluate.py`` and ``test_expression_reader.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from specsolve.errors import SpecsolveError
from tests.conftest import law_data, law_spec, relation
from tests.differential import at_a_point, by_coordinate
from tests.expression_space import rewrites, space, stride

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    import polars as pl

    from tests.expression_space import Node, Rewrite

    Read = Callable[[str | Mapping[str, Any]], pl.DataFrame]


def _spec() -> dict[str, Any]:
    """The law model, with a season over ``t`` and a per-``f`` lead, so every operator has something to act on."""
    spec = law_spec('x <= 100', dims=['f', 't'])
    spec['dimensions']['s'] = {'dtype': 'str'}
    spec['parameters']['lead'] = {'dims': ['f'], 'dtype': 'int'}
    spec['relations'] = {'season_of': {'key': 't', 'values': 's'}}
    return spec


def _data() -> dict[str, Any]:
    """``law_data`` over three snapshots, two in one season, so a window and a group each have an inside and an edge."""
    from tests.oracle import pd

    snapshots = [0, 1, 2]
    return law_data() | {
        't': snapshots,
        's': ['warm', 'cold'],
        'season_of': relation('t', 's', snapshots, ['warm', 'warm', 'cold']),
        'lead': pd.Series({'a': 1, 'b': 2}),
    }


#: What the lanes read differently today. Strict: a fix turns its case red until
#: its entry here goes.
DIVERGENCES = {
    'sum((y) + (w), over=f)': (
        'the relational lane keeps w[b] where y is absent, in a read and in a built row alike; '
        'the summand is absent there, so the sum skips it (test_arithmetic_laws: #311)'
    ),
    'sum((w) + (y), over=f)': 'the same, operands swapped',
}

#: Every twentieth rewrite joins the depth-two space in the census below.
CENSUS_STEP = 20
#: Of the 73 cases that samples, how many answered when the floor was set: all
#: 47 expressions, and 13 of the 26 rewrites. The other 13 are #1137's refusal.
ANSWERS = 60


@pytest.fixture(scope='module')
def agreed() -> Iterator[Read]:
    """The fixture held at one point, built once per module and worker: ``agreed(expression)`` is the frame both lanes read, or a failure naming both."""
    with at_a_point(_spec(), _data()) as read:
        yield read


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise the generated tables at whatever ``--sweep-depth`` and ``--sweep-shard`` ask for."""
    shard = metafunc.config.getoption('--sweep-shard')
    if 'node' in metafunc.fixturenames:
        nodes = stride(space(metafunc.config.getoption('--sweep-depth')), shard)
        metafunc.parametrize('node', [_known(node) for node in nodes])
    if 'rewrite' in metafunc.fixturenames:
        metafunc.parametrize('rewrite', stride(rewrites(), shard), ids=lambda r: f'{r.rule}: {r.before}')


def _known(node: Node) -> Any:
    """*node* as a case, marked where ``DIVERGENCES`` names it."""
    reason = DIVERGENCES.get(str(node))
    marks = [pytest.mark.xfail(reason=reason, strict=True)] if reason else []
    return pytest.param(node, id=str(node), marks=marks)


def _answer(agreed: Read, expression: str | Mapping[str, Any]) -> pl.DataFrame | None:
    """The agreed frame, or None where the relational lane refuses a construct it cannot build.

    That refusal is #1137's: a ``sum`` or ``shift`` along a dimension a
    constant part of the expression does not carry. The linopy lane reads it,
    so it is a gap already settled, not a disagreement.
    """
    try:
        return agreed(expression)
    except SpecsolveError as exc:
        if 'specsolve cannot build' not in str(exc):
            raise
        return None


def test_every_expression_reads_the_same_on_both_lanes(agreed: Read, node: Node) -> None:
    if _answer(agreed, str(node)) is None:
        pytest.skip('specsolve cannot build it (#1137)')


def test_a_rewrite_that_must_not_change_the_meaning_does_not_change_the_value(agreed: Read, rewrite: Rewrite) -> None:
    before, after = _answer(agreed, str(rewrite.before)), _answer(agreed, str(rewrite.after))
    if before is None or after is None:
        pytest.skip('specsolve cannot build it (#1137)')
    assert by_coordinate(before) == pytest.approx(by_coordinate(after)), (
        f'`{rewrite.before}` and `{rewrite.after}` are one value under {rewrite.rule}, and read '
        f'{by_coordinate(before)} and {by_coordinate(after)}'
    )


OPERATORS = [
    pytest.param('sum(x)', id='sum-over-every-dim'),
    pytest.param('sum(y)', id='sum-over-every-dim-of-a-masked-variable'),
    pytest.param('sum(x * y, over=t)', id='sum-of-a-product-of-two-variables'),
    pytest.param('x * w - y', id='subtract-a-masked-variable'),
    pytest.param('x / w', id='divide-by-a-parameter'),
    pytest.param('w / x', id='divide-by-a-variable'),
    pytest.param('x / y', id='divide-by-a-masked-variable'),
    pytest.param(
        'x / (x - x)',
        id='divide-by-zero',
        marks=pytest.mark.xfail(
            reason='the linopy lane reads inf where #1776 made a reported quotient by zero absent', strict=True
        ),
    ),
    pytest.param('x ** 2', id='square-a-variable'),
    pytest.param('(x + y) ** 2', id='square-a-sum-with-a-masked-term'),
    pytest.param('2 ** x', id='a-variable-exponent'),
    pytest.param('x ** w', id='a-parameter-exponent'),
    pytest.param("shift(x, along=t, offset=1, edge='wrap')", id='shift-wrap'),
    pytest.param('shift(x, along=t, offset=lead, edge=0)', id='shift-by-a-parameter-offset'),
    pytest.param('shift(y, along=t, offset=2)', id='shift-a-masked-variable-past-most-of-the-axis'),
    pytest.param('shift(x, along=t, offset=1, by=season_of, within=s)', id='shift-within-a-group'),
    pytest.param('sum_back(x, along=t, window=2)', id='sum-back'),
    pytest.param("sum_back(y, along=t, window=2, edge='wrap')", id='sum-back-wrap-over-a-masked-variable'),
    pytest.param('sum_back(x, along=t, window=lead)', id='sum-back-by-a-parameter-window'),
    pytest.param('sum_back(x, along=t, window=2, by=season_of, within=s)', id='sum-back-within-a-group'),
    pytest.param('sum(x, by=season_of, over=t, into=s)', id='sum-by-a-relation'),
    pytest.param('sum(y, by=season_of, over=t, into=s)', id='sum-a-masked-variable-by-a-relation'),
    pytest.param(
        {
            'dims': ['f', 't'],
            'cases': {'first': {'when': 'position(t) == 0', 'expression': 'w'}},
            'otherwise': 'shift(x, along=t, offset=1)',
        },
        id='cases-by-position',
    ),
    pytest.param(
        {'dims': ['f', 't'], 'cases': {'gated': {'when': 'gate', 'expression': 'y'}}, 'otherwise': 'x'},
        id='cases-by-a-mask',
    ),
]


@pytest.mark.parametrize('expression', OPERATORS)
def test_every_operator_reads_the_same_on_both_lanes(agreed: Read, expression: str | Mapping[str, Any]) -> None:
    assert _answer(agreed, expression) is not None, 'a construct in this table is one both lanes build'


def test_enough_of_the_sweep_reaches_an_answer(agreed: Read) -> None:
    """The floor under the skips, because a skip reads exactly like a pass.

    Both tables above skip what the relational lane refuses (#1137), which is
    right, and a change that made every case skip would leave them green. One
    test, so xdist cannot split the count, and always the depth-two slice, so
    the number does not move with ``--sweep-depth``.
    """
    cases = [str(node) for node in space(2) if str(node) not in DIVERGENCES]
    cases += [str(pair.before) for pair in rewrites()[::CENSUS_STEP]]
    answered = sum(_answer(agreed, case) is not None for case in cases)
    assert answered >= ANSWERS, (
        f'{answered} of {len(cases)} sampled cases reached an answer, below the {ANSWERS} measured '
        f'when this floor was set — a lane has lost ground, and a skip would have hidden it'
    )
