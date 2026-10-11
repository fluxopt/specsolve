"""The ``specsolve`` command, driven through Typer's test runner on the example manifest."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from specsolve import cli

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture
def study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """examples/manifest copied beside the spec it names, and made the working directory."""
    pytest.importorskip('typer')
    pytest.importorskip('fastexcel')
    shutil.copy(REPO / 'examples' / 'dispatch.yaml', tmp_path / 'dispatch.yaml')
    shutil.copytree(REPO / 'examples' / 'manifest', tmp_path / 'manifest')
    monkeypatch.chdir(tmp_path / 'manifest')
    return tmp_path / 'manifest'


def _invoked(*arguments: str) -> tuple[int, str]:
    from typer.testing import CliRunner

    outcome = CliRunner().invoke(cli.app(), list(arguments))
    return outcome.exit_code, outcome.output


def test_solve_runs_the_named_runs_and_skips_one_already_archived(study: Path) -> None:
    code, output = _invoked('solve', 'base', 'high_gas')
    assert code == 0, output
    assert output.splitlines()[0].startswith('base: optimal, objective 7500,'), output
    assert output.splitlines()[1].startswith('high_gas: optimal, objective 9500,'), output
    assert (study / 'runs' / 'base').is_dir() and (study / 'runs' / 'high_gas').is_dir()
    assert not (study / 'runs' / 'scenarios').exists(), 'a run not named is not solved'

    code, output = _invoked('solve', 'specsolve.yaml')
    assert code == 0, output
    assert output.splitlines()[:2] == [
        'base: skipped, runs/base exists',
        'high_gas: skipped, runs/high_gas exists',
    ], 'an archive that exists is never overwritten'
    assert output.splitlines()[2].startswith('scenarios: 2 slices: 2 optimal,'), output


def test_list_says_which_runs_sweep_and_which_are_archived(study: Path) -> None:
    _invoked('solve', 'base')
    code, output = _invoked('list')
    assert code == 0, output
    assert [line.split(maxsplit=1) for line in output.splitlines()] == [
        ['base', 'solve  archived'],
        ['high_gas', 'solve  not archived'],
        ['scenarios', 'sweep  not archived'],
    ], 'one line per run, in manifest order'


def test_check_reports_a_missing_source_by_run_and_exits_1(study: Path) -> None:
    (study / 'data' / 'timeseries' / 'load.csv').unlink()
    code, output = _invoked('check')
    assert code == 1, output
    lines = output.splitlines()
    assert lines[0] == "base: DataError: no data provided for parameter 'load'", output
    assert lines[1].startswith('high_gas: DataError'), output
    assert lines[2] == 'scenarios: ok', 'the sweep brings its own load, so only the two solves miss it'


def test_a_run_the_manifest_does_not_have_exits_2_naming_the_runs(study: Path) -> None:
    code, output = _invoked('solve', 'bse')
    assert code == 2, output
    assert 'has no run bse. Runs: base, high_gas, scenarios.' in output


def test_a_failed_run_is_reported_and_the_next_still_solves(study: Path) -> None:
    (study / 'data' / 'high_gas.xlsx').unlink()
    code, output = _invoked('solve', 'high_gas', 'base')
    assert code == 1, output
    assert output.splitlines()[0].startswith('high_gas: failed:'), output
    assert output.splitlines()[1].startswith('base: optimal'), output


def test_a_solve_that_ends_other_than_optimal_exits_1(study: Path) -> None:
    short = study / 'data' / 'short'
    short.mkdir()
    (short / 'load.csv').write_text('snapshot,value\n0,60.0\n1,110.0\n2,1000.0\n3,90.0\n')
    manifest = study / 'specsolve.yaml'
    manifest.write_text(
        manifest.read_text() + '  short:\n    sources: [data/base.xlsx, data/timeseries/snapshot.csv, data/short/]\n'
    )
    code, output = _invoked('solve', 'short')
    assert code == 1, output
    assert output.startswith('short: infeasible'), output


def test_main_without_typer_exits_with_the_install_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    real = cli.importlib.util.find_spec
    monkeypatch.setattr(cli.importlib.util, 'find_spec', lambda name: None if name == 'typer' else real(name))
    with pytest.raises(SystemExit, match=r"pip install 'specsolve\[cli\]'"):
        cli.main()
