"""The two `solve_over` examples keep making their claims.

`examples/rolling/` and `examples/myopic/` each have one *substantive*
assertion here, apart from their golden:

- rolling: lookahead really does close the myopia gap, and enough of it reaches
  full foresight exactly
- myopic: the fleet each period starts from is the one the last period ended
  with, which is the promise `carry` makes

Regenerate the goldens with ``--update-golden``.
"""

from __future__ import annotations

import pytest

from tests.conftest import EXAMPLES_DIR, assert_golden, run_example

STRATEGIES = ['rolling', 'myopic']


@pytest.fixture(scope='module')
def outputs() -> dict[str, str]:
    return {name: run_example(EXAMPLES_DIR / name / 'run.py', f'{name}_example') for name in STRATEGIES}


def test_lookahead_closes_the_myopia_gap(outputs: dict[str, str]) -> None:
    """A golden alone keeps passing if every number drifts together, or if the gap closes because storage goes unused."""
    lines = [line for line in outputs['rolling'].splitlines() if line.startswith('rolling')]
    gaps = [float(line.rsplit('+', 1)[1].rstrip('% ')) for line in lines]
    peaks = [float(line.split('peak soc')[1].split()[0]) for line in lines]

    assert gaps[0] > 0, 'a window with no lookahead must pay for its myopia'
    assert gaps == sorted(gaps, reverse=True), 'more lookahead must never cost more'
    assert gaps[-1] == 0.0, 'enough lookahead must reach full foresight'
    assert all(peak > 0 for peak in peaks), (
        'the store must cycle in every schedule — a gap that came from storage '
        'going unused would be arithmetic, not myopia'
    )


def test_the_pathway_inherits_each_fleet(outputs: dict[str, str]) -> None:
    """The example asserts this internally; this is the check that it ran at all."""
    assert 'each period starts from the fleet the last one left' in outputs['myopic']


@pytest.mark.parametrize('name', STRATEGIES)
def test_the_example_matches_its_committed_output(
    name: str, outputs: dict[str, str], pytestconfig: pytest.Config
) -> None:
    assert_golden(
        outputs[name],
        EXAMPLES_DIR / name / 'run.out',
        pytestconfig,
        drifted=f'the {name} example no longer prints what the docs show:',
    )
