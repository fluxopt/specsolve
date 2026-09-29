"""The benchmark corpus still loads, and still builds.

The benchmark workflow runs only on request, so this is the gate that opens
`bench/models/` (#343). `check()` needs no data and runs on the bare install;
the build needs `bench.cases`, which imports pandas, so it skips there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import specsolve as sps

MODELS = Path(__file__).resolve().parent.parent / 'bench' / 'models'


def _specs() -> list[Path]:
    return sorted(MODELS.glob('*/spec.yaml'))


#: Case names are the directory each model sits in, read off the tree so the load gate runs on a bare install.
CASE_NAMES = [p.parent.name for p in _specs()]


@pytest.mark.parametrize('spec', _specs(), ids=lambda p: p.parent.name)
def test_a_bench_spec_loads(spec: Path):
    """Every language change has to migrate this corpus too, or fail here."""
    sps.check(spec)


def test_the_corpus_is_not_empty():
    """The parametrised tests pass vacuously if the glob stops matching."""
    assert len(_specs()) >= 6, f'expected the bench corpus to be found; got {CASE_NAMES}'


@pytest.fixture(scope='module')
def bench_cases():
    return pytest.importorskip('bench.cases', reason='needs pandas; the bare install has none')


@pytest.mark.parametrize('case', CASE_NAMES)
def test_a_bench_case_builds_on_the_smallest_rung(case: str, tmp_path: Path, bench_cases):
    """Loading is not building (#345)."""
    bench_case = bench_cases.CASES[case]
    sources = bench_case.write(bench_case.shape('xs'), tmp_path)
    with sps.build(bench_case.spec, sources) as model:
        assert model is not None


def test_every_model_backs_a_case(bench_cases):
    """No model nothing runs, and no case whose model was renamed away.

    A case that generates its model per rung is gated by `bench/test_harness.py`.
    """
    static = sorted(name for name, case in bench_cases.CASES.items() if case.spec is not None)
    assert static == CASE_NAMES
    for name, case in bench_cases.CASES.items():
        assert (case.spec is None) != (case.generate_spec is None), (
            f'{name}: a case carries a committed spec or a generator — never both, never neither'
        )
