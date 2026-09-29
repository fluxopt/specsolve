"""The decomposition example must keep reaching the monolith, or it is not evidence.

`examples/benders/run.py` expresses Benders with cuts as data; the decomposed
answer equals the monolith's on the same sources. Regenerate the committed
output with ``--update-golden``.
"""

from __future__ import annotations

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

EXAMPLE = EXAMPLES_DIR / 'benders' / 'run.py'
GOLDEN = EXAMPLE.with_name('run.out')


@pytest.fixture(scope='module')
def output() -> str:
    return run_example(EXAMPLE, 'benders_example')


def test_the_decomposition_reaches_the_monolith(output: str) -> None:
    """The oracle, asserted apart from the golden, which would pass a stable drift."""
    assert 'difference: 0.0e+00' in output, output


def test_the_example_matches_its_committed_output(output: str, pytestconfig: pytest.Config) -> None:
    assert_golden(output, GOLDEN, pytestconfig, drifted='the example no longer prints what the docs show:')
