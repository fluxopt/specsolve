"""The mutation instrument, which is only worth having if it cannot lie.

Three of these matter more than the rest: a mutation whose deletion would not
parse still has to be applied, a run has to put the file back even though it
restores from git rather than from memory, and an uncommitted change to a
tracked file has to stop the run before anything is written — because
`git checkout --` would destroy that work rather than restore it. An untracked
file is not at risk and must not block the run.
"""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest

from tools import mutate as tool

if TYPE_CHECKING:
    from pathlib import Path

GUARDED = """\
def check(value: int) -> int:
    if value < 0:
        raise ValueError('negative')
    return value
"""


@pytest.fixture
def guarded(tmp_path: Path) -> Path:
    (tmp_path / 'g.py').write_text(GUARDED)
    return tmp_path


@pytest.mark.parametrize(
    ('spec', 'start', 'end'),
    [
        pytest.param('g.py:2', 2, 2, id='one line'),
        pytest.param('g.py:2-3', 2, 3, id='a range'),
        pytest.param('g.py:2-3#the negativity guard', 2, 3, id='a range with a label'),
    ],
)
def test_a_guard_argument_names_lines(guarded: Path, spec: str, start: int, end: int) -> None:
    guard = tool.parse_guard(spec, root=guarded)
    assert (guard.start, guard.end) == (start, end), 'the span comes off the argument as written'


def test_a_label_given_is_the_label_used(guarded: Path) -> None:
    guard = tool.parse_guard('g.py:2-3#the negativity guard', root=guarded)
    assert guard.label == 'the negativity guard', 'a label after # is taken verbatim'


def test_a_label_left_out_is_read_off_the_source(guarded: Path) -> None:
    guard = tool.parse_guard('g.py:2', root=guarded)
    assert guard.label == 'drop `if value < 0:`', 'the first mutated line names the row when nothing else does'


def test_a_long_line_is_cut_rather_than_run_into_the_table(guarded: Path) -> None:
    (guarded / 'g.py').write_text(f'x = {"a" * 200}\n')
    label = tool.parse_guard('g.py:1', root=guarded).label
    assert '…' in label and len(label) < 80, f'a table cell cannot take 200 characters, got {len(label)}'


@pytest.mark.parametrize(
    'spec',
    [
        pytest.param('g.py', id='no line at all'),
        pytest.param('g.py:x', id='not a number'),
        pytest.param('nowhere.py:1', id='no such file'),
        pytest.param('g.py:99', id='past the end'),
        pytest.param('g.py:3-2', id='backwards'),
        pytest.param('g.py:0', id='before the first line'),
    ],
)
def test_an_unusable_argument_says_what_is_wrong(guarded: Path, spec: str) -> None:
    with pytest.raises(tool.SpecError):
        tool.parse_guard(spec, root=guarded)


def test_a_whole_statement_is_deleted(guarded: Path) -> None:
    guard = tool.parse_guard('g.py:2-3', root=guarded)
    mutated, form = tool.mutate(GUARDED, guard)
    assert form == 'deleted', 'removing a whole if leaves the file parsing, so nothing has to stand in for it'
    assert 'ValueError' not in mutated, 'the guard is gone'


def test_a_body_becomes_pass_because_deleting_it_would_not_parse(guarded: Path) -> None:
    guard = tool.parse_guard('g.py:3', root=guarded)
    mutated, form = tool.mutate(GUARDED, guard)
    assert form == 'pass', 'an if with nothing under it is a syntax error, so the body is replaced instead'
    assert '        pass' in mutated, 'the stand-in keeps the body indentation'
    assert 'ValueError' not in mutated, 'the guard is still gone'


def test_a_range_no_mutation_of_can_parse_is_refused(guarded: Path) -> None:
    (guarded / 'g.py').write_text('def f(\n    a: int,\n) -> int:\n    return a\n')
    guard = tool.parse_guard('g.py:1', root=guarded)
    with pytest.raises(tool.SpecError, match='narrow the range'):
        tool.mutate((guarded / 'g.py').read_text(), guard)


def test_a_dirty_tree_stops_the_run_before_anything_is_written(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tool, 'dirty_paths', lambda: ['src/specsolve/api.py'])
    assert tool.main(['src/specsolve/api.py:1']) == 2, (
        'an uncommitted change to a tracked file is refused, because checkout would destroy it'
    )


def test_the_bytecode_is_dropped_so_a_restored_file_is_really_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The precaution the whole tool rests on, and the one hardest to see fail.

    Python's staleness check is mtime plus size, so a same-length restore inside
    the same second keeps the *mutant* running and the next row measures a tree
    nobody is looking at. Nothing else here can catch that: the mutation still
    gets reported, just against the wrong code.

    Checked directly rather than through a run, because the failure is silent by
    construction.
    """
    monkeypatch.setattr(tool, 'REPO', tmp_path)
    swept = tmp_path / 'src' / 'specsolve' / '__pycache__'
    swept.mkdir(parents=True)
    (swept / 'engine.cpython-311.pyc').write_bytes(b'stale')
    spared = tmp_path / '.venv' / 'lib' / 'site-packages' / '__pycache__'
    spared.mkdir(parents=True)

    tool._drop_bytecode()

    assert not swept.exists(), "the mutant's bytecode outlived its source, so the next row is measured wrong"
    assert spared.exists(), 'the virtualenv is not ours to sweep, and sweeping it costs a reinstall'


def test_an_untracked_file_does_not_block_the_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The narrowing: `git checkout --` cannot touch a file git does not track.

    Counting untracked files would refuse to start in any working tree holding a
    scratch note or an editor leftover, and guard nothing while doing it.
    """
    repo = _repo(tmp_path)
    (repo / 'a-scratch-note.md').write_text('not committed, not at risk\n')
    monkeypatch.setattr(tool, 'REPO', repo)

    code = tool.main(['src/g.py:2-3#the negativity guard', '--tests', 'test_g.py'])

    out = capsys.readouterr().out
    assert code == 0, f'an untracked file is not uncommitted work, got {code} — {out}'
    assert '**caught**' in out, f'and the run went ahead: {out}'
    assert (repo / 'a-scratch-note.md').exists(), 'and the file it ignored is still there'


def _repo(tmp_path: Path) -> Path:
    """A committed one-file repo whose test catches the guard being deleted."""
    subprocess.run(['git', 'init', '-q', str(tmp_path)], check=True)
    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'g.py').write_text(GUARDED)
    (tmp_path / 'test_g.py').write_text(
        'import sys; sys.path.insert(0, "src")\n'
        'import pytest\n'
        'from g import check\n'
        'def test_it_refuses_a_negative():\n'
        '    with pytest.raises(ValueError):\n'
        '        check(-1)\n'
    )
    for args in (['add', '-A'], ['-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'x']):
        subprocess.run(['git', '-C', str(tmp_path), *args], check=True)
    return tmp_path


def test_a_caught_guard_is_reported_and_the_file_is_put_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The end-to-end claim: mutate, notice, restore — with the restore checked.

    The restore goes through ``git checkout --`` rather than a variable, so a
    killed run leaves nothing behind, and the run says that the tree came back
    clean.
    """
    repo = _repo(tmp_path)
    monkeypatch.setattr(tool, 'REPO', repo)
    monkeypatch.setattr(sys, 'argv', ['mutate'])

    code = tool.main(['src/g.py:2-3#the negativity guard', '--tests', 'test_g.py'])

    out = capsys.readouterr().out
    assert code == 0, f'a clean run exits 0, got {code} — {out}'
    assert '| the negativity guard (`g.py:2-3`) | **caught** |' in out, f'the row is the #658 table row: {out}'
    assert 'Tree clean after the run.' in out, 'the run asserts what it left behind, rather than assuming it'
    assert (repo / 'src' / 'g.py').read_text() == GUARDED, 'the guard is back, byte for byte'
    assert not list(repo.rglob('__pycache__')), 'and no bytecode is left for the next mutation to read'


@pytest.mark.parametrize(
    ('returncode', 'summary', 'expected'),
    [
        pytest.param(0, '3352 passed in 42.61s', '3352 passed in 42.61s', id='green — the guard survived'),
        pytest.param(1, '1 failed, 12 passed in 3.10s', '**caught**', id='failed, plain'),
        pytest.param(2, '1 failed in 0.38s', '**caught**', id='failed under -x, which xdist exits 2 for'),
        pytest.param(5, 'no tests ran in 0.01s', '**nothing ran**', id='--tests matched nothing'),
        pytest.param(2, '1 error in 0.37s', '**errored, not failed**', id='the mutation broke an import'),
        pytest.param(3, 'INTERNALERROR', '**pytest exited 3**', id='pytest itself fell over'),
    ],
)
def test_a_verdict_needs_a_failure_not_merely_a_non_zero_exit(returncode: int, summary: str, expected: str) -> None:
    """Three different outcomes share exit code 2, and only one of them is a catch.

    A collection error is the dangerous one: the mutation broke an import, no
    test ran, and a code-only reading would certify the guard as reached.
    """
    assert tool.verdict(returncode, summary).startswith(expected), (
        f'exit {returncode} with {summary!r} must read as {expected!r}, not as something stronger'
    )


@pytest.mark.parametrize(
    ('returncode', 'summary'),
    [
        pytest.param(5, 'no tests ran in 0.01s', id='nothing collected'),
        pytest.param(2, '1 error in 0.37s', id='a collection error'),
        pytest.param(3, 'INTERNALERROR', id='pytest fell over'),
    ],
)
def test_a_row_that_proves_nothing_says_so_in_the_cell(returncode: int, summary: str) -> None:
    """The cell is the whole record, so a worthless row cannot look like a result."""
    cell = tool.verdict(returncode, summary)
    assert 'proves nothing' in cell, f'a reader of the table alone has to see it: {cell!r}'
    assert '**caught**' not in cell, f'and it must not read as a catch: {cell!r}'


def test_the_command_only_asks_for_xdist_where_xdist_is_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The bare-install job has no xdist, and pytest answers `-n` with exit 4.

    A row built on that reads "pytest exited 4" and proves nothing.
    """
    monkeypatch.setattr(tool.importlib.util, 'find_spec', lambda name: None)
    assert '-n' not in tool.command(['tests']), 'without xdist the flag is left off rather than sent and rejected'

    monkeypatch.setattr(tool.importlib.util, 'find_spec', lambda name: object())
    assert '-n' in tool.command(['tests']), 'and it is used where it is there, because the suite is minutes without it'


def test_a_run_that_printed_only_to_stderr_still_says_why() -> None:
    """A usage error writes nothing to stdout, and "no output" hides the fix."""
    assert tool._summary('', 'error: unrecognized arguments: -n') == 'error: unrecognized arguments: -n', (
        'the reason a run never started is the only useful thing in that row'
    )


def test_a_tree_left_dirty_is_reported_rather_than_passed_over(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The last line of defence: the run says what it left behind.

    Everything else here checks that the restore works; this checks what the
    run says when it does not. Tracked only, for the same reason the start-up
    check is.
    """
    repo = _repo(tmp_path)
    monkeypatch.setattr(tool, 'REPO', repo)
    calls = iter([[], ['src/g.py']])
    monkeypatch.setattr(tool, 'dirty_paths', lambda: next(calls))

    code = tool.main(['src/g.py:2-3#the negativity guard', '--tests', 'test_g.py'])

    out = capsys.readouterr().out
    assert code == 1, f'a run that left the tree dirty does not exit 0, got {code}'
    assert 'a tracked file was left modified' in out, f'and it says which file: {out}'


def test_a_run_that_collected_nothing_is_not_a_catch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The lie a code-only reading would tell.

    Pointed at a path with no tests, pytest exits non-zero — the same way it
    does under ``-x`` when a failure interrupts the session. A row that read the
    exit code alone would call that "caught" and certify a guard nothing
    touched.
    """
    repo = _repo(tmp_path)
    monkeypatch.setattr(tool, 'REPO', repo)

    tool.main(['src/g.py:2-3#the negativity guard', '--tests', 'src'])

    out = capsys.readouterr().out
    assert '**caught**' not in out, f'a run that collected nothing caught nothing: {out}'
    assert 'proves nothing' in out, f'and the row has to say so rather than look like a result: {out}'


def test_a_guard_no_test_reaches_is_reported_as_survived(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The row that is the whole point: green here means the guard is unreachable."""
    repo = _repo(tmp_path)
    (repo / 'test_g.py').write_text('def test_nothing_about_the_guard():\n    assert True\n')
    subprocess.run(['git', '-C', str(repo), 'add', '-A'], check=True)
    subprocess.run(
        ['git', '-C', str(repo), '-c', 'user.email=t@t', '-c', 'user.name=t', 'commit', '-qm', 'y'], check=True
    )
    monkeypatch.setattr(tool, 'REPO', repo)

    code = tool.main(['src/g.py:2-3#the negativity guard', '--tests', 'test_g.py'])

    out = capsys.readouterr().out
    assert code == 0, f'the run itself succeeded, got {code} — {out}'
    assert 'passed' in out and '**caught**' not in out, f'a guard nothing reaches survives, and says so: {out}'
