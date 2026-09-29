"""Every expression to a bounded depth on both lanes, and the rewrites that must not move it.

The AST is closed, so the set of spellings at a given depth is finite and
"every one of them agrees" can be checked rather than sampled. The model is the
one ``conftest.law_spec`` builds for ``test_arithmetic_laws.py``.

**Agreement**: each expression builds to one model on the linopy lane and the
relational one.

**Invariance**: a rewrite that must not change the meaning does not change the
answer. Agreement alone misses a mistake both lanes share, such as
``sum(a + b)`` and ``sum(a) + sum(b)`` parting where an operand is absent
(#311).

Depth three is 2,576 expressions and minutes of CPU, against depth two's thirty
and about a second (#1203), so ``--sweep-depth 3`` is its own job, split across
runners by ``--sweep-shard i/n``. The rewrites are built over an operand pool
and give 480 cases at either depth.

A model this fixture makes infeasible or unbounded is skipped, and
``test_enough_of_the_sweep_reaches_an_answer`` stops that from becoming every
case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from specsolve.errors import SpecsolveError
from tests.conftest import law_data
from tests.differential import RTOL, NoFiniteAnswerError, differential
from tests.expression_space import expressions, rewrites, row_spec, stride
from tests.linopy_lane.loader import OracleCannotBuildError

if TYPE_CHECKING:
    from tests.expression_space import Node, Rewrite

#: The shared fixture, taken once: `x` total, `y` masked at `f=b`, `w` dense.
DATA = law_data()

#: Every twentieth rewrite joins the depth-two space in the census below — a
#: fixed stride over a fixed order, so the sample is the same on every machine.
CENSUS_STEP = 20
#: Of the 56 cases that samples, what answered and what a lane refused when the
#: floor and the ceiling were measured (#1213). Every refusal is the asymmetry
#: #1137 settled: a `sum` acting along a dimension a constant part of the
#: expression does not carry, which the linopy lane builds and the relational
#: one refuses by name.
ANSWERS = 38
LANE_REFUSALS = 10


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrise both sweeps at whatever ``--sweep-depth`` and ``--sweep-shard`` ask for."""
    shard = metafunc.config.getoption('--sweep-shard')
    if 'node' in metafunc.fixturenames:
        depth = metafunc.config.getoption('--sweep-depth')
        metafunc.parametrize('node', stride(expressions(depth), shard), ids=str)
    if 'rewrite' in metafunc.fixturenames:
        metafunc.parametrize('rewrite', stride(rewrites(), shard), ids=lambda r: f'{r.rule}: {r.before}')


@dataclass(frozen=True)
class Answer:
    """What both lanes said about one expression, or why neither was asked.

    A model this fixture makes infeasible or unbounded, and a lane that refuses
    the expression by name, are skips rather than failures.

    Attributes:
        value: The objective both lanes reached, or None if neither was asked.
        skipped: Why there is no value, in the words the skip reports.
        refused: Whether it was a lane that refused, rather than the fixture.
    """

    value: float | None = None
    skipped: str = ''
    refused: bool = False


def _answer(node: Node) -> Answer:
    try:
        with differential(row_spec(node), DATA) as run:
            return Answer(value=float(run.result.objective))
    except OracleCannotBuildError as exc:
        return Answer(skipped=f'the oracle cannot build it: {str(exc).splitlines()[0]}', refused=True)
    except SpecsolveError as exc:
        if 'specsolve cannot build' not in str(exc):
            raise
        return Answer(skipped=f'specsolve cannot build it: {str(exc).splitlines()[0]}', refused=True)
    except NoFiniteAnswerError:
        return Answer(skipped='this fixture admits no finite answer')


def test_every_expression_means_the_same_on_both_lanes(node: Node) -> None:
    if reason := _answer(node).skipped:
        pytest.skip(reason)


def test_a_rewrite_that_must_not_change_the_meaning_does_not_change_the_answer(rewrite: Rewrite) -> None:
    before, after = _answer(rewrite.before), _answer(rewrite.after)
    if reason := before.skipped or after.skipped:
        pytest.skip(reason)
    assert before.value == pytest.approx(after.value, rel=RTOL), (
        f'`{rewrite.before}` and `{rewrite.after}` are one model under {rewrite.rule}, '
        f'and reached {before.value} and {after.value}'
    )


@pytest.mark.parametrize('spec', ['2/2', '-1/2', '0/0', '1', 'a/2', '0/2/2'], ids=str)
def test_a_shard_the_division_does_not_contain_is_refused(spec: str) -> None:
    """A shard nobody runs is coverage lost in a green job, so the spec is
    checked rather than clamped: `2/2` would silently be empty."""
    with pytest.raises(pytest.UsageError):
        stride(expressions(1), spec)


def test_every_case_is_in_exactly_one_shard() -> None:
    cases = expressions(2)
    for n in (2, 3, 4):
        legs = [stride(cases, f'{i}/{n}') for i in range(n)]
        seen = [case for leg in legs for case in leg]
        assert sorted(map(str, seen)) == sorted(map(str, cases)), (
            f'{n} shards of the depth-two space do not partition it — a case in none of them is '
            f'one the sharded CI job never runs'
        )


def test_enough_of_the_sweep_reaches_an_answer() -> None:
    """The floor under the skips, because a skip reads exactly like a pass.

    A slice is counted two ways: a floor under the answers, and a ceiling over
    the lane refusals. One test, so xdist cannot split the count across
    workers, and always the depth-two slice, so the numbers do not move with
    ``--sweep-depth``.
    """
    answers = [_answer(node) for node in expressions(2)]
    answers += [_answer(pair.before) for pair in rewrites()[::CENSUS_STEP]]
    answered = sum(answer.value is not None for answer in answers)
    refused = sum(answer.refused for answer in answers)

    assert answered >= ANSWERS, (
        f'{answered} of {len(answers)} sampled cases reached an answer, below the {ANSWERS} measured '
        f'when this floor was set — the fixture has gone degenerate, not the language'
    )
    assert refused <= LANE_REFUSALS, (
        f'{refused} of {len(answers)} sampled cases were refused by a lane, above the {LANE_REFUSALS} '
        f'measured when this ceiling was set — a lane has lost ground, and a skip would have hidden it'
    )
