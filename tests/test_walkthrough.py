"""The architecture walkthrough keeps running, and keeps printing what it claims.

``examples/walkthrough.py`` calls the real pipeline stage by stage. Its whole
output is committed as ``examples/walkthrough.out`` and compared line for line;
regenerate it with ``pixi run pytest tests/test_walkthrough.py --update-golden``.
"""

from __future__ import annotations

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

WALKTHROUGH = EXAMPLES_DIR / 'walkthrough.py'
GOLDEN = EXAMPLES_DIR / 'walkthrough.out'


@pytest.fixture(scope='module')
def output() -> str:
    """One run of the whole pipeline, shared by both tests."""
    return run_example(WALKTHROUGH, 'walkthrough')


def test_walkthrough_matches_golden(output: str, pytestconfig: pytest.Config) -> None:
    assert_golden(
        output,
        GOLDEN,
        pytestconfig,
        drifted='the walkthrough narrates something the pipeline no longer does.\n'
        'If this run is the correct story, regenerate the golden file:\n'
        '    pixi run pytest tests/test_walkthrough.py --update-golden\n',
    )


def test_walkthrough_claims_hold(output: str) -> None:
    """The golden file catches *any* change; this names the ones that matter.

    A failure here says which architectural property lapsed, not which line.
    """
    for stage in range(1, 8):
        assert f'[{stage}]' in output, f'stage {stage} did not run'

    (core,) = [line for line in output.splitlines() if line.strip().startswith('core AST ')]
    assert 'weighted_sum' in output and 'weighted_sum' not in core, (
        'the macro is what the file wrote and not what the program carries'
    )
    assert 'row absence' in output and 'not 24' in output, 'a mask removes rows, not values'
    assert 'ok (optimal)' in output
    assert 'degree 3' in output, 'the ceiling still bites — the objective takes 2 and no more'
    assert 'caught by check()' in output, 'and with no data attached, so CI can run it'
