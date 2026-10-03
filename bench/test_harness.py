"""The harness measures what it says it measures.

`test_ladder.py` is the measurement; this is the part of it that has to be true
for a number to mean anything. It is fast — nothing here solves above a tiny
rung or builds above `xs` — so it runs on a bare `pytest bench` before anything
is timed.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import pickle
import subprocess
import sys
import tomllib
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import polars as pl
import pytest

from bench import conftest as harness
from bench import floor, plot, profile_build, profile_phases, report, results, tidy, warm_payoff
from bench import results as bench_results
from bench.arms import ARMS, solved, unmeasurable
from bench.arms.specsolve import _handoff, checked_sources
from bench.cases import CASES, Shape, _declaration_sweep, _declarations_spec, shortened
from bench.conftest import (
    MIN_ROUNDS,
    _holder_if_alive,
    flag_passed,
    pytest_benchmark_update_machine_info,
    refuse_unless_idle,
    take_lock,
)
from bench.test_ladder import RELOADS
from specsolve.relational.engine.labels import Labelled
from specsolve.relational.sinks.solvers.base import WarmStart

# ---------------------------------------------------------------------------
# the machine interlock (#705)
# ---------------------------------------------------------------------------


def _dead_pid() -> int:
    """A pid that was alive a moment ago and is not any more."""
    child = subprocess.Popen(['true'])
    child.wait()
    return child.pid


def test_a_second_session_is_refused_naming_the_holder(tmp_path) -> None:
    """A concurrent run's file looks complete and its numbers are junk (#419),
    so the second session is refused before anything is timed."""
    lock = tmp_path / 'bench.lock'
    take_lock(lock)
    with pytest.raises(pytest.UsageError, match='another benchmark is running') as caught:
        take_lock(lock)
    assert f'pid {os.getpid()}' in str(caught.value), 'the refusal has to say who holds the machine'
    assert '--i-know-another-is-running' in str(caught.value), 'and how to get past it deliberately'


def test_a_dead_holders_lock_is_evicted(tmp_path) -> None:
    lock = tmp_path / 'bench.lock'
    lock.write_text(f'pid {_dead_pid()}, started 03:14')
    take_lock(lock)
    assert f'pid {os.getpid()}' in lock.read_text(), 'a crashed session must not wedge every one after it'


def test_an_unreadable_lock_counts_as_held(tmp_path) -> None:
    lock = tmp_path / 'bench.lock'
    lock.write_text('garbage')
    assert _holder_if_alive(lock) == 'garbage', 'a lock we cannot interpret is safer read as held than free'


def test_a_busy_machine_is_refused_and_an_idle_one_is_not() -> None:
    with pytest.raises(pytest.UsageError, match='already working'):
        refuse_unless_idle(9.5, 8)
    refuse_unless_idle(7.5, 8)


def test_the_interlock_is_wired_into_session_start(tmp_path) -> None:
    """A real second session, refused.

    The lock path follows `TMPDIR`, so the child sees a lock owned by this live
    process. `CI` is stripped because CI bypasses the interlock.
    """
    (tmp_path / 'specsolve-bench.lock').write_text(f'pid {os.getpid()}, started 03:14')
    env = {k: v for k, v in os.environ.items() if k != 'CI'}
    env['TMPDIR'] = str(tmp_path)
    child = subprocess.run(
        [sys.executable, '-m', 'pytest', 'bench/test_harness.py', '--collect-only', '-q'],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path(__file__).resolve().parent.parent,
        check=False,
    )
    assert child.returncode != 0, 'the second session has to refuse to start, not collect quietly'
    assert 'another benchmark is running' in child.stderr + child.stdout


def test_the_fingerprint_carries_the_load_triple(request: pytest.FixtureRequest) -> None:
    machine_info: dict[str, object] = {}
    pytest_benchmark_update_machine_info(request.config, machine_info)
    load = machine_info['load_avg']
    assert isinstance(load, tuple) and len(load) == 3, (
        'the 1/5/15-minute triple is what lets a contaminated file be recognised after the fact'
    )


# ---------------------------------------------------------------------------
# a contaminated minimum is marked rather than published as fact (#797)
# ---------------------------------------------------------------------------


def _timing(arm: str, **over: Any) -> dict[str, Any]:
    """One `timing` record in the shape `bench.results` emits, tight by default."""
    return {
        'record': 'timing',
        'case': 'dispatch',
        'size': 'm',
        'sink': 'lp',
        'arm': arm,
        'wall_seconds': 1.0,
        'median': 2.0,
        'iqr': 0.0,
        'rounds': 9,
        'peak_rss_bytes': 1e9,
        'live_fraction': 1.0,
        'counts': {'columns': 1000, 'rows': 100, 'nonzeros': 1000},
    } | over


def _plotted() -> dict[str, Any]:
    """One rung of a panel line, in the shape `plot.series` emits."""
    return {'wall': 1.0, 'lo': 0.9, 'hi': 1.1, 'peak': 0.5, 'vars': 1000}


def _ceiling_record(ladder: str, size: str, **over: Any) -> dict[str, Any]:
    return {
        'record': 'ceiling',
        'arm': 'linopy',
        'case': 'transport',
        'ladder': ladder,
        'sink': 'highs',
        'size': size,
        'budget': 30.0,
        'memory_budget': 6.0,
        'stopped_by': 'time',
    } | over


def _rendered(**over: Any) -> str:
    return report.table('dispatch', report.best([_timing('specsolve', **over), _timing('linopy')]), 'lp')


def _loop(case: str, arm: str, width: int) -> dict[str, Any]:
    """One `loop` record — the first-vs-steady pair the marginal table reads."""
    return {
        'record': 'loop',
        'case': case,
        'size': 'm',
        'arm': arm,
        'nominal_variables': width,
        'first_build_seconds': 0.1,
        'steady_build_seconds': 0.05,
        'counts': {'columns': width, 'rows': 10, 'nonzeros': width},
    }


# ---------------------------------------------------------------------------
# a short run cannot replace the published provenance
# ---------------------------------------------------------------------------


def _config(sizes: list[str], destination: str) -> Any:
    return SimpleNamespace(
        invocation_params=SimpleNamespace(args=(f'--benchmark-json={destination}',)),
        getoption=lambda name: sizes if name == '--sizes' else None,
    )


def test_a_short_run_may_not_write_the_committed_results(tmp_path: Path) -> None:
    """`pixi run ladder xs` pointed at a committed file would replace every
    published table's provenance with four measurements."""
    for published in sorted(harness.published_results()):
        with pytest.raises(harness.pytest.UsageError, match='cannot write'):
            harness.refuse_to_overwrite_the_provenance(_config(['xs'], str(published)))


def test_a_short_run_pointed_anywhere_else_is_nobody_business(tmp_path: Path) -> None:
    harness.refuse_to_overwrite_the_provenance(_config(['xs'], str(tmp_path / 'scratch.json')))


def test_the_guard_names_the_files_the_published_run_actually_writes() -> None:
    """One per sink and case, as `ladder-ci` writes them."""
    published = harness.published_results()

    assert len(published) == 8, 'four cases through two sinks'
    assert {p.name for p in published} == {
        f'latest-{sink}-{case}.json'
        for sink in ('highs', 'gurobi')
        for case in ('dispatch', 'transport', 'storage', 'fleet')
    }, 'the names the published run writes, sink by case'
    assert not any(p.name == 'latest.json' for p in published), 'no run has written that name since ladder-ci'


def test_a_smoke_run_may_still_write_its_own_file(tmp_path: Path) -> None:
    """`ladder-smoke` lands in `bench/results` too, and is meant to be narrow."""
    smoke = Path.cwd() / 'bench/results/latest-smoke.json'
    harness.refuse_to_overwrite_the_provenance(_config(['xs'], str(smoke)))


def test_narrower_sinks_still_write_the_provenance() -> None:
    """The scheduled run takes one sink per job, and each half is still the
    published run; leaving out rungs is what makes a smoke test."""
    whole = sorted(harness.published_rungs())
    for published in sorted(harness.published_results()):
        harness.refuse_to_overwrite_the_provenance(_config(whole, str(published)))


# ---------------------------------------------------------------------------
# the published selection has one home
# ---------------------------------------------------------------------------


def test_no_workflow_retypes_the_published_selection() -> None:
    """A run that retypes the published selection is a number whose fingerprint
    no longer describes it. `bench.yml` and `codspeed.yml` take narrower
    selections of their own; the published one lives in the `ladder` task.
    """

    root = Path(__file__).resolve().parents[1]
    task = tomllib.loads((root / 'pyproject.toml').read_text())
    cases = ' '.join(task['tool']['pixi']['feature']['bench']['tasks']['ladder']['cmd'].split())
    marker = cases[cases.index('--cases') : cases.index('--sizes')].strip()

    guilty = [w.name for w in sorted((root / '.github' / 'workflows').glob('*.y*ml')) if marker in w.read_text()]
    assert not guilty, f'{guilty} spell out `{marker}`; call `pixi run ladder` so the selection has one home'


def test_the_ci_ladder_defaults_to_the_published_memory_budget() -> None:
    """`ladder-ci` falls back to the published memory budget, which therefore
    exists twice and must not drift."""

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    tasks = tasks['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'memory')
    fallback = tasks['ladder-ci']['cmd'].split('BENCH_MEMORY_BUDGET:-', 1)[1].split('}', 1)[0]
    assert fallback == published, f'ladder-ci falls back to {fallback} GB, the published ladder is {published} GB'


def _committed(path: str) -> list[str]:
    """Repository-relative paths git has under *path*; a published run clears the working tree's copy."""
    root = Path(__file__).resolve().parents[1]
    listed = subprocess.run(['git', 'ls-files', path], cwd=root, capture_output=True, text=True, check=True)
    return listed.stdout.split()


def test_the_run_clears_every_committed_result_before_it_measures() -> None:
    """A killed case does not overwrite its own file, so whatever the run does
    not clear is published as though this run had measured it."""
    import fnmatch

    root = Path(__file__).resolve().parents[1]
    workflow = (root / '.github/workflows/published-benchmark.yml').read_text()
    globs = [
        line.split('rm -f', 1)[1].strip()
        for line in workflow.splitlines()
        if 'rm -f' in line and 'bench/results' in line
    ]
    assert globs, 'the run must clear the committed results before it measures'

    missed = [f for f in _committed('bench/results') if not any(fnmatch.fnmatch(f, g) for g in globs)]
    assert not missed, f"{missed} survive {globs} and would be published as this run's numbers"


def test_the_cell_being_measured_is_on_disk_before_it_is_measured() -> None:
    """The process that would record a cell that exhausts the machine is the one
    that dies, so the note is written before the measurement."""
    harness.pytest_runtest_logstart('bench/test_ladder.py::test_emit[transport-w100-linopy-highs]', ())
    assert harness.INFLIGHT.read_text() == 'bench/test_ladder.py::test_emit[transport-w100-linopy-highs]'


def test_a_casualty_list_is_not_read_as_measurements(tmp_path: Path) -> None:
    """It is a list of records, not a run document."""
    (tmp_path / 'latest.json').write_text('{"benchmarks": [], "machine_info": {}, "commit_info": {}, "datetime": ""}')
    (tmp_path / 'casualties.json').write_text('[{"record": "casualty", "cell": "x"}]')
    found = [p.name for p in bench_results.files(tmp_path)]
    assert found == ['latest.json'], f'the casualty list is not a results file, got {found}'


def test_the_ci_ladder_covers_every_published_case() -> None:
    """`ladder-ci` runs one pytest per case, so the case list exists twice."""

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    tasks = tasks['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'cases').split()
    asked = next(a['default'] for a in tasks['ladder-ci']['args'] if a['arg'] == 'cases').split()
    assert asked == published, f'ladder-ci runs {asked} and the published ladder is {published}'


def test_a_case_the_box_cannot_hold_leaves_the_others_their_turn() -> None:
    """A dead case must not end the loop; the ladder still fails once it has
    taken what it can."""

    tasks = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    cmd = tasks['tool']['pixi']['feature']['bench']['tasks']['ladder-ci']['cmd']
    body, tail = cmd.split(' do ', 1)[1].split('; done', 1)
    assert 'exit' not in body, f'a dead case must not end the loop, and this body exits: {body.strip()}'
    assert 'exit 1' in tail, 'and the ladder still fails, once the cases it could take are taken'


def test_every_published_case_is_measured_and_uploaded_on_its_own() -> None:
    """A runner that dies skips every step it has left, `if: always()` included,
    so each case uploads before the next can take the box."""

    root = Path(__file__).resolve().parents[1]
    tasks = tomllib.loads((root / 'pyproject.toml').read_text())['tool']['pixi']['feature']['bench']['tasks']
    published = next(a['default'] for a in tasks['ladder']['args'] if a['arg'] == 'cases').split()

    workflow = (root / '.github/workflows/published-benchmark.yml').read_text()
    for sink in ('highs', 'gurobi'):
        for case in published:
            step = f'          sink: {sink}\n          case: {case}\n'
            assert step in workflow, f'{sink}/{case} has no step of its own, so it cannot be uploaded on its own'
    assert workflow.count('uses: ./.github/actions/ladder-case') == 2 * len(published), (
        'one step per case per sink, and no more'
    )

    action = (root / '.github/actions/ladder-case/action.yml').read_text()
    assert 'bash bench/memory-watchdog.sh &' not in action, 'the step does not start its own watchdog'
    ladder_ci = tomllib.loads((root / 'pyproject.toml').read_text())
    ladder_ci = ladder_ci['tool']['pixi']['feature']['bench']['tasks']['ladder-ci']['cmd']
    assert 'bash bench/memory-watchdog.sh &' in ladder_ci, 'each case runs under the watchdog'
    assert 'uses: actions/upload-artifact@v4' in action, 'and hands back what it measured before the next one starts'


def test_the_watchdog_says_something_even_when_nothing_moves() -> None:
    """The high-water line prints only when it moves, so a quiet watchdog and a
    dead one would read alike."""
    script = (Path(__file__).resolve().parents[1] / 'bench/memory-watchdog.sh').read_text()
    assert 'BENCH_MEMORY_HEARTBEAT_SECONDS' in script, 'silence has to be distinguishable from death'


def test_the_watchdog_reaches_the_process_that_holds_the_model() -> None:
    """Killing the case by its pytest flags leaves the memory behind.

    `benchmem(isolate=True)` measures in a `multiprocessing` spawn child, which
    carries none of pytest's arguments. Read off the script, because a watchdog
    exercised here would kill the ladder running this file.
    """
    script = (Path(__file__).resolve().parents[1] / 'bench/memory-watchdog.sh').read_text()
    assert 'multiprocessing.spawn import spawn_main' in script, (
        "the spawned child is what holds the model, and pytest's own flags do not name it"
    )
    assert '-P "$pid"' in script, (
        'children go first — once the pytest is gone the child is reparented and only its own argv is left'
    )
    assert 'available again' in script, (
        'after a kill it watches for the memory to come back rather than sleeping through the seconds '
        'in which the next case starts'
    )


def test_the_reproduction_script_runs_what_the_task_runs() -> None:
    """A reproduction running a different selection would be worth less than none."""
    from bench import reproduce

    selection = ' '.join(reproduce.published())
    for expected in ('--cases dispatch transport storage fleet', '--budget 30', 'bench/results/latest.json'):
        assert expected in selection, f'the reproduction lost `{expected}` from the task definition'
    assert 'PUBLISHED' not in Path(reproduce.__file__).read_text(), 'the selection is read, not repeated'


# ---------------------------------------------------------------------------
# the reproduction environment carries every library the harness measures
# ---------------------------------------------------------------------------


def test_the_lock_pins_every_library_an_arm_needs() -> None:
    """An arm the harness measures must be installable from `bench/reproduce.py.lock`."""
    locked = (Path(__file__).resolve().parent / 'reproduce.py.lock').read_text()
    for name, module in sorted(ARMS.items()):
        for required in getattr(module, 'REQUIRES', ()):
            assert f'name = "{required}"' in locked, (
                f'the {name} arm needs {required}, which bench/reproduce.py.lock does not pin — '
                f're-run `uv lock --script bench/reproduce.py`'
            )


def test_the_lock_freezes_the_branch_linopy_moves_on() -> None:
    """linopy is installed from `master`, which only a commit can pin."""
    locked = (Path(__file__).resolve().parent / 'reproduce.py.lock').read_text()
    assert 'git = "https://github.com/PyPSA/linopy?rev=master#' in locked, (
        'the lock has to name the linopy commit, not the branch'
    )


def _locked_commit(package: dict[str, Any]) -> str:
    """The commit a locked git install resolves to, empty for one from an index."""
    return str(package.get('source', {}).get('git', '')).partition('#')[2]


def _same_install(measured: str, locked: str, commit: str = '') -> bool:
    """Whether the lock installs the build a published number was measured on.

    A git install is identified by the commit in its local segment, since its
    release number follows whatever tag is nearby. A commit that is a release
    tag has no local segment in the lock, so *commit* is the locked source's.
    """
    if '+' not in measured:
        return measured == locked
    short = measured.partition('+')[2].partition('.')[0].removeprefix('g')
    return commit.startswith(short) or locked.partition('+')[2].startswith(f'g{short}')


def test_the_lock_installs_what_the_published_numbers_were_taken_on() -> None:
    """Every version a published file was measured on is the one the lock installs (#1490).

    Read from git at `HEAD`, because this runs inside a measuring run, which
    clears and rewrites `bench/results/`.
    """
    root = Path(__file__).resolve().parents[1]
    lock = tomllib.loads((root / 'bench/reproduce.py.lock').read_text())
    locked = {package['name']: (package.get('version', ''), _locked_commit(package)) for package in lock['package']}

    published = [
        path for path in _committed('bench/results') if path.endswith('.json') and not path.endswith('.ceilings.json')
    ]
    assert published, 'nothing is published, so this guard would pass without reading a version'

    for path in published:
        blob = subprocess.run(
            ['git', 'show', f'HEAD:{path}'], cwd=root, capture_output=True, text=True, check=True
        ).stdout
        measured = json.loads(blob)['machine_info']['versions']
        for name, version in measured.items():
            assert name in locked, f'{path} was measured on {name}, which the lock does not install at all'
            pinned, commit = locked[name]
            installs = f'{pinned} at {commit[:9]}' if commit else pinned
            assert _same_install(version, pinned, commit), (
                f'{path} was measured on {name} {version} and the lock installs {installs}, '
                f'so the documented reproduction does not re-take these numbers — '
                f're-run `uv lock --script bench/reproduce.py`'
            )


# ---------------------------------------------------------------------------
# the width ladder reaches the size ladder's rungs by a different route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_name', ['transport', 'storage'])
def test_a_width_rung_matches_its_size_twin_variable_for_variable(case_name: str) -> None:
    """`w10` is `s`, `w1000` is `l` — same variables, same rows, different shape."""
    ladder = {shape.label: shape for shape in CASES[case_name].ladder}
    for width, size in (('w1', 'xs'), ('w10', 's'), ('w100', 'm'), ('w1000', 'l')):
        assert ladder[width].nominal_variables == ladder[size].nominal_variables, (
            f'{case_name}/{width} is {ladder[width].nominal_variables:,} variables '
            f"against {size}'s {ladder[size].nominal_variables:,} — the ladders have drifted apart"
        )


@pytest.mark.parametrize('case_name', ['transport', 'storage'])
def test_a_width_rung_grows_entities_and_holds_the_snapshots(case_name: str) -> None:
    """Every other ladder grows `snapshot` and freezes the entity counts."""
    width = [s for s in CASES[case_name].ladder if s.label.startswith('w')]
    snapshots = {s.sizes['snapshot'] for s in width}
    assert len(snapshots) == 1, f'a width rung must hold the snapshot count fixed, got {sorted(snapshots)}'
    entities = [sum(v for k, v in s.sizes.items() if k != 'snapshot') for s in width]
    assert entities == sorted(entities) and entities[0] * 1000 == entities[-1], (
        f'entities have to grow by the stated factors across {[s.label for s in width]}, got {entities}'
    )


# ---------------------------------------------------------------------------
# an arm stops climbing a ladder it cannot afford (bench/conftest.py)
# ---------------------------------------------------------------------------


def _ceiling(budget: float, selected: tuple[str, ...] = ('xs', 's', 'm', 'l'), memory: float = 0.0) -> Any:
    return harness.Ceiling(budget, {}, selected, memory)


# ---------------------------------------------------------------------------
# an arm stops climbing a ladder whose next rung will not fit the machine
# ---------------------------------------------------------------------------


def test_a_rung_that_projects_over_the_memory_budget_stops_the_ladder() -> None:
    """Over-memory leaves the run with no runner at all (#1416).

    The rungs grow tenfold and a measurement holds the model twice, so 3 GB at
    `xs` projects to 60 GB at `s`.
    """
    ceiling = _ceiling(0.0, memory=16.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 1.0, 3e9)
    reason = ceiling.reached('linopy', 'dispatch', 'xs', 'lp')
    assert reason is not None, 'a projection over the memory budget stops the ladder'
    assert '60 GB' in reason and '3 GB' in reason, 'the reason carries the measurement and the projection'


def test_a_measurement_over_the_memory_budget_stops_without_projecting() -> None:
    ceiling = _ceiling(0.0, memory=16.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 1.0, 31e9)
    reason = ceiling.reached('linopy', 'dispatch', 'xs', 'lp')
    assert reason is not None and 'projects' not in reason, 'the rung itself was over, so there is nothing to project'


def test_a_rung_inside_both_budgets_lets_the_ladder_continue() -> None:
    ceiling = _ceiling(120.0, memory=16.0)
    ceiling.record('specsolve', 'dispatch', 'xs', 'lp', 0.5, 0.2e9)
    assert ceiling.reached('specsolve', 'dispatch', 'xs', 'lp') is None


def test_no_memory_budget_measures_everything() -> None:
    """The memory budget is off by default."""
    ceiling = _ceiling(120.0)
    ceiling.record('linopy', 'dispatch', 'xs', 'lp', 0.5, 900e9)
    assert ceiling.reached('linopy', 'dispatch', 'xs', 'lp') is None


def test_a_rung_that_projects_over_budget_stops_the_ladder() -> None:
    """The rungs grow tenfold, so a 20 s build at `dispatch/xs` projects to 200 s."""
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 20.0)
    reason = ceiling.reached('pyomo', 'dispatch', 'xs', 'lp')
    assert reason is not None, 'a projection over budget stops the ladder'
    assert '200 s' in reason and '20 s' in reason, 'the reason carries the measurement and the projection'


def test_a_rung_inside_budget_lets_the_ladder_continue() -> None:
    ceiling = _ceiling(120.0)
    ceiling.record('specsolve', 'dispatch', 'xs', 'lp', 0.5)
    assert ceiling.reached('specsolve', 'dispatch', 'xs', 'lp') is None, '0.5 s projects to 5 s, well inside 120 s'


def test_a_measurement_over_budget_stops_the_ladder_without_projecting() -> None:
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'm', 'lp', 300.0)
    reason = ceiling.reached('pyomo', 'dispatch', 'xs', 'lp')
    assert reason is not None and 'projects' not in reason, (
        'a rung that already blew the budget needs no arithmetic about the next one'
    )


def test_the_top_of_the_run_never_projects() -> None:
    """The last rung this run asked for has nothing after it, whatever the case defines."""
    ceiling = _ceiling(120.0, selected=('xs', 's', 'm', 'l'))
    ceiling.record('specsolve', 'dispatch', 'l', 'lp', 100.0)
    assert ceiling.reached('specsolve', 'dispatch', 'l', 'lp') is None, (
        '`l` is the top of this run; `xl` is not being measured and cannot stop it'
    )


def test_a_ceiling_on_one_ladder_leaves_the_other_alone() -> None:
    """A case can carry two ladders: `l` is the longest model, `w1000` the widest."""
    ceiling = _ceiling(120.0, selected=('xs', 's', 'm', 'l', 'w1', 'w10', 'w100', 'w1000'))
    ceiling.record('pyomo', 'transport', 'm', 'lp', 20.0)
    assert ceiling.reached('pyomo', 'transport', 'l', 'lp') is not None, 'the size ladder stops'
    assert ceiling.reached('pyomo', 'transport', 'w10', 'lp') is None, 'the width ladder is a separate climb'


def test_the_sidecar_says_which_budget_stopped_each_climb() -> None:
    """The record carries both budgets, so the one that fired has to be on it too."""
    ceiling = _ceiling(30.0, memory=6.0)
    ceiling.record('linopy', 'dispatch', 'm', 'highs', 1.0, 3e9)
    ceiling.record('pyomo', 'dispatch', 'm', 'highs', 20.0, 0.1e9)

    stopped = {row['arm']: row['stopped_by'] for row in ceiling.rows()}
    assert stopped == {'linopy': 'memory', 'pyomo': 'time'}, (
        '3 GB projects past the 6 GB budget and 20 s past the 30 s one, each on its own axis'
    )


@pytest.mark.parametrize(
    ('stopped_by', 'expected'),
    [
        pytest.param('memory', '>6 GB', id='a-memory-stop-prints-the-memory-budget'),
        pytest.param('time', '>30 s', id='a-time-stop-prints-the-time-budget'),
        pytest.param(None, '>30 s', id='a-sidecar-from-before-the-field-prints-seconds'),
    ],
)
def test_a_bound_names_the_budget_that_actually_stopped_the_climb(stopped_by: str | None, expected: str) -> None:
    """Both budgets are on the record and only one of them fired.

    `None` is a sidecar from runs that carried no memory budget.
    """
    ceiling = _ceiling_record('size', 'm')
    ceiling.pop('stopped_by')
    if stopped_by is not None:
        ceiling['stopped_by'] = stopped_by
    assert results.bound_label(ceiling) == expected


def test_both_renderers_read_the_same_bound() -> None:
    """The table and the chart print the same cell through one function."""
    ceiling = _ceiling_record('size', 'm', stopped_by='memory')
    report.CEILINGS[:] = [ceiling]
    taken = {
        ('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's')},
        ('transport', 'highs', 'specsolve'): {r: _plotted() for r in ('xs', 's', 'm', 'l')},
    }

    charted = plot.panels(taken, [ceiling])['transport — highs — length']['series']['linopy']['bound']
    tabled = report.over_budget('transport', 'l', 'highs', 'linopy')
    assert tabled == '>6 GB', 'the table names the budget that fired'
    assert charted == [None, None, None, tabled], 'and the chart says the same thing at the same rungs'


def test_a_ceiling_is_per_sink() -> None:
    """Writing an LP file and filling a solver are different costs, and an arm
    that cannot afford one may still afford the other."""
    ceiling = _ceiling(120.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 20.0)
    assert ceiling.reached('pyomo', 'dispatch', 'xs', 'highs') is None, 'the other sink was never measured'


def test_no_budget_measures_everything() -> None:
    ceiling = _ceiling(0.0)
    ceiling.record('pyomo', 'dispatch', 'xs', 'lp', 9999.0)
    assert ceiling.reached('pyomo', 'dispatch', 'xs', 'lp') is None, (
        '--budget 0 is the way to take the slow number anyway'
    )


# ---------------------------------------------------------------------------
# a hand-written arm is the same model, or it is not an arm
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('case_name', ['dispatch', 'transport', 'storage', 'fleet', 'nodal'])
@pytest.mark.parametrize('dialect', [a for a in sorted(ARMS) if a != 'specsolve'])
def test_a_hand_written_arm_builds_the_same_model(case_name: str, dialect: str) -> None:
    """Every arm but `specsolve` is a model somebody typed twice.

    A transposed index builds a different model that benchmarks perfectly, so
    the smallest rung of each case is solved both ways and the objectives
    compared. `unmeasurable` decides which cells there are.
    """
    reason = unmeasurable(dialect, case_name, ARMS[dialect].SINKS[0])
    if reason:
        pytest.skip(reason)
    case = CASES[case_name]
    smallest = case.ladder[0].label
    paths = case.data(case.shape(smallest))
    ours = solved('specsolve', case_name, smallest, paths, {})
    theirs = solved(dialect, case_name, smallest, paths, {})
    assert theirs == pytest.approx(ours, rel=1e-9), (
        f'{dialect} solves {case_name}/{smallest} to {theirs}, specsolve to {ours} — not the same model'
    )


# ---------------------------------------------------------------------------
# the entry points still run
# ---------------------------------------------------------------------------


def test_the_report_renders_from_the_committed_results() -> None:
    """The readers take the `bench/results` directory, and find something in it."""
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert report.main([]) == 0
    assert '| variables |' in out.getvalue(), 'the default target is bench/results, and it renders a table'


def test_the_long_table_renders_from_the_committed_results() -> None:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        assert tidy.main([]) == 0
    lines = out.getvalue().splitlines()
    assert lines[0] == 'run,case,size,sink,arm,phase,variables,metric,value', 'the header is the schema'
    assert len(lines) > 1, 'the committed provenance produces rows, not just a header'


def test_the_profilers_wrap_the_class_that_actually_builds() -> None:
    """Both profilers monkeypatch private engine names, so a refactor in `src/`
    can retire them without touching `bench/` (#1245)."""
    import importlib

    from specsolve.relational.engine.assembly import Assembly

    for module_path, class_name, method in profile_build.STEPS:
        module = importlib.import_module(module_path)
        owner = module if class_name is None else getattr(module, class_name, None)
        assert owner is not None, f'profile_build patches {class_name} in {module_path}, which moved'
        assert hasattr(owner, method), f'profile_build patches {class_name or module_path}.{method}, which moved'
    for method in profile_phases.PHASES:
        assert hasattr(Assembly, method), f'profile_phases patches Assembly.{method}, which moved'


# ---------------------------------------------------------------------------
# the long table: a row per number, and no holes (bench/tidy.py)
# ---------------------------------------------------------------------------


def _long(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(tidy.measurements(records, run='r'))


def test_a_timing_record_fans_into_one_row_per_metric() -> None:
    rows = _long([_timing('specsolve')])
    assert [r['metric'] for r in rows] == [
        'wall_seconds',
        'peak_rss_bytes',
        'iqr_seconds',
        'median_seconds',
        'rounds',
        'live_fraction',
        'columns',
        'rows',
        'nonzeros',
    ], 'every number the record carries becomes a row, counts last, memray absent because it was not measured'
    assert {r['case'] for r in rows} == {'dispatch'}, 'the dims repeat down the rows — that is what long form is'
    assert {r['arm'] for r in rows} == {'specsolve'}


def test_a_number_the_run_did_not_produce_is_an_absent_row() -> None:
    """A null would have to mean something, and nothing it could mean is true of
    a measurement never taken."""
    rows = _long([_timing('specsolve', peak_rss_bytes=None, counts={'columns': 10, 'rows': 1, 'nonzeros': None})])
    assert [r['metric'] for r in rows].count('peak_rss_bytes') == 0, 'no isolate=True, so no rss row at all'
    assert [r['metric'] for r in rows].count('nonzeros') == 0, 'an arm that cannot count nonzeros writes none'
    assert all(r['value'] is not None for r in rows), 'every value column is complete'


def test_the_rebuild_loop_becomes_two_phases() -> None:
    rows = [r for r in _long([_loop('dispatch', 'specsolve', 1200)]) if r['metric'] == 'wall_seconds']
    assert [(r['phase'], r['value']) for r in rows] == [('first', 0.1), ('steady', 0.05)], (
        'first and steady answer different questions, so they are two rows and never one'
    )
    assert {r['sink'] for r in rows} == {''}, 'the rebuild loop has no sink — it stops before one'


def test_the_long_table_and_the_published_table_agree_on_a_cell() -> None:
    """The two renderings read one extraction."""
    record = _timing('specsolve', wall_seconds=0.5)
    wall = next(r['value'] for r in _long([record]) if r['metric'] == 'wall_seconds')
    assert f'{wall:.2f}' in report.table('dispatch', report.best([record]), 'lp')


def test_the_fingerprint_is_long_too() -> None:
    run = {'record': 'run', 'python': '3.12.0', 'versions': {'polars': '1.0', 'gurobipy': None}, 'commits': {}}
    rows = list(tidy.fingerprint([run], run='r'))
    assert [(r['key'], r['value']) for r in rows] == [('python', '3.12.0'), ('version:polars', '1.0')], (
        'a package the environment does not have is absent, not blank — blank would read as a version'
    )


def test_the_run_record_carries_the_cpu_pytest_benchmark_collected(tmp_path: Path) -> None:
    """`platform.processor()` answers `x86_64` on every Linux runner."""
    doc = {
        'machine_info': {
            'system': 'Linux',
            'release': '6.8.0',
            'machine': 'x86_64',
            'processor': 'x86_64',
            'python_version': '3.12.0',
            'cpu': {'brand_raw': 'AMD EPYC 7763', 'count': 4},
        },
        'benchmarks': [],
    }
    path = tmp_path / 'latest.json'
    path.write_text(json.dumps(doc))
    run = next(r for r in results.records(path) if r['record'] == 'run')
    assert (run['cpu'], run['cores']) == ('AMD EPYC 7763', 4), 'the brand and the core count, not the arch'


def _run(cpu: str, cores: int = 4) -> dict[str, object]:
    return {
        'record': 'run',
        'platform': 'Linux 6.8.0',
        'cpu': cpu,
        'cores': cores,
        'python': '3.12.0',
        'versions': {'polars': '1.0'},
    }


def test_one_machine_prints_as_one_line() -> None:
    assert report.provenance([_run('AMD EPYC 7763')]) == (
        'AMD EPYC 7763, 4 cores (Linux 6.8.0), python 3.12.0 — polars 1.0.'
    )


def test_a_page_merged_from_two_machines_says_so() -> None:
    """The ladder takes one sink per job (#1315), so two runners can draw different CPUs."""
    line = report.provenance([_run('AMD EPYC 7763'), _run('Intel Xeon Platinum 8370C')])
    assert 'Taken on 2 machines' in line, 'a merged page must not print one machine for rows from two'
    assert 'AMD EPYC 7763' in line and 'Intel Xeon Platinum 8370C' in line, 'both boxes are named'


def test_two_files_from_one_machine_are_not_marked() -> None:
    assert 'machines' not in report.provenance([_run('AMD EPYC 7763'), _run('AMD EPYC 7763')])


def test_a_record_from_before_the_harness_carried_a_machine_still_renders() -> None:
    """A `.jsonl` result is taken verbatim, so an older run record must render."""
    assert report.provenance([{'record': 'run', 'platform': None}]) == '? (?), python ? — .'
    assert report.provenance([{'record': 'run'}]) == '? (?), python ? — .'


#: Renders the marginal table for three cases of identical width. Run twice
#: under different hash seeds, it is the whole of the determinism check below.
_RENDER_TIED = """
import sys
sys.path.insert(0, %r)
from bench import report
rows = [
    dict(record='loop', case=c, size='m', arm=a, nominal_variables=1200,
         first_build_seconds=0.1, steady_build_seconds=0.05,
         counts={'columns': 1200, 'rows': 10, 'nonzeros': 1200})
    for c in ('fleet', 'nodal', 'profiled') for a in ('specsolve', 'linopy')
]
print(report.marginal(rows))
"""


def test_the_marginal_table_does_not_reshuffle_between_processes() -> None:
    """A published table a re-render reshuffles has a diff that means nothing.

    `fleet`, `nodal` and `profiled` tie on width, and the rows come out of a
    set. Two processes under different `PYTHONHASHSEED`, because one process
    has a fixed seed.
    """
    root = str(Path(__file__).resolve().parent.parent)
    out = []
    for seed in ('0', '1'):
        child = subprocess.run(
            [sys.executable, '-c', _RENDER_TIED % root],
            capture_output=True,
            text=True,
            env=os.environ | {'PYTHONHASHSEED': seed},
            check=True,
        )
        out.append(child.stdout)
    assert out[0] == out[1], 'two hash seeds rendered two row orders — a refresh diff would be noise'
    names = [line.split('|')[1].strip() for line in out[0].splitlines() if line.startswith('| ')]
    assert names[1:] == ['fleet', 'nodal', 'profiled'], (
        'tied widths break by name, so the order is stated rather than inherited from a hash'
    )


def test_the_marginal_table_survives_a_file_that_never_measured_specsolve() -> None:
    """`--arms linopy` is a legitimate run."""
    assert report.marginal([_loop('dispatch', 'linopy', 1200)]) is not None


@pytest.mark.parametrize(
    ('iqr', 'marked'),
    [
        pytest.param(2.0 * (report.SPREAD_BUDGET + 0.01), True, id='spread-past-the-budget-is-marked'),
        pytest.param(2.0 * (report.SPREAD_BUDGET - 0.01), False, id='spread-inside-the-budget-is-not'),
        pytest.param(None, False, id='a-file-written-before-the-spread-was-carried-is-not'),
    ],
)
def test_a_cell_is_marked_by_its_spread_over_its_own_median(iqr: float | None, marked: bool) -> None:
    """The signal is iqr/median: a whole distribution spread is the contamination
    `min` cannot filter out (#797)."""
    assert (report.MARK in _rendered(iqr=iqr)) is marked, (
        f'iqr/median of {iqr} against a budget of {report.SPREAD_BUDGET} must {"mark" if marked else "leave"} the cell'
    )


def test_the_note_appears_exactly_where_a_cell_is_marked() -> None:
    assert report._SPREAD_NOTE not in _rendered(), 'a table with nothing to doubt must not carry the warning'
    assert _rendered(iqr=1.9).count(report._SPREAD_NOTE) == 1, (
        'a marked table says once what the mark means, or the mark is decoration'
    )


def test_marking_leaves_the_published_number_alone() -> None:
    """The mark is a doubt about the minimum, not a different statistic: the
    number in a marked cell is the same one an unmarked run would print."""
    clean, dirty = _rendered(), _rendered(iqr=1.9)
    assert '| 1.00 s |' in clean and '| 1.00 s~ |' in dirty, 'the marked cell still prints its minimum'
    assert dirty.removesuffix('\n\n' + report._SPREAD_NOTE).replace(report.MARK, '') == clean, (
        'marking must annotate the table, not restate it'
    )


def test_the_ratio_beside_a_marked_cell_is_marked_too() -> None:
    """A ratio is only as quotable as the two numbers it divides (#797)."""
    marked = report.density(report.best([_timing('specsolve', size='d100', iqr=1.9), _timing('linopy', size='d100')]))
    assert '| 1.00x~ |' in marked, 'a ratio drawn from a marked minimum carries the doubt'


def test_a_cell_with_no_number_in_it_is_never_marked() -> None:
    """A mark on an em dash claims doubt about a measurement nobody took."""
    rows = report.best([_timing('specsolve', iqr=1.9), _timing('linopy', size='l')])
    table = report.table('dispatch', rows, 'lp')
    assert '| \u2014 |' in table, 'the arm that did not run this rung still renders as absent'
    assert f'\u2014{report.MARK}' not in table, 'an absent measurement cannot be noisy'


_FENCED = """# A page

Prose the numbers are read with.

<!-- bench:results -->

| old | table |

<!-- bench:/results -->

More prose.
"""


def test_a_fenced_block_is_replaced_and_the_prose_around_it_is_not() -> None:
    """Only what sits inside a fence is mechanical."""
    written, skipped = report.splice(_FENCED, {'results': '| new | table |'})
    assert skipped == [], 'the page has the fence, so nothing was skipped'
    assert '| new | table |' in written and '| old | table |' not in written
    assert 'Prose the numbers are read with.' in written, 'everything outside the fence survives'
    assert 'More prose.' in written


def test_writing_twice_changes_nothing_the_second_time() -> None:
    once, _ = report.splice(_FENCED, {'results': '| new | table |'})
    assert report.splice(once, {'results': '| new | table |'})[0] == once, 'a re-render has an empty diff'


@pytest.mark.parametrize(
    ('page', 'complaint'),
    [
        pytest.param('<!-- bench:results -->\nhalf a fence\n', 'half a `results` fence', id='unclosed'),
        pytest.param('<!-- bench:/results -->\n<!-- bench:results -->\n', 'closes the results fence', id='inverted'),
    ],
)
def test_a_page_that_cannot_take_the_block_is_refused(page: str, complaint: str) -> None:
    """Refused rather than appended to."""
    with pytest.raises(SystemExit, match=complaint):
        report.splice(page, {'results': '| new | table |'})


def test_a_page_without_a_fence_is_told_so_rather_than_failed() -> None:
    """Named in the return rather than raised, so the caller can print what it
    had nowhere to put."""
    written, skipped = report.splice('# A page\n\nno fence here\n', {'results': '| new | table |'})
    assert skipped == ['results'], 'the fragment is reported, not written'
    assert written == '# A page\n\nno fence here\n', 'and the page is untouched'


def test_an_empty_fragment_never_blanks_the_page() -> None:
    """A results file that rendered nothing would otherwise publish nothing, silently."""
    with pytest.raises(SystemExit, match='refusing to blank the page'):
        report.splice(_FENCED, {'results': '   '})


def test_the_marginal_table_carries_no_ratio_between_libraries() -> None:
    """A build-only number is not comparable across libraries: one that defers
    its coefficients to its writer pays at the seam instead."""
    table = report.marginal([_loop('dispatch', 'specsolve', 1200), _loop('dispatch', 'linopy', 1200)])
    assert 'specsolve: steady' in table and 'linopy: steady' in table, 'both libraries still get their columns'
    assert '\u00f7' not in table, 'no ratio column here — the build is not the same work in each'
    assert 'not across the row' in table, 'and the table says so where it is read'


def test_a_model_table_shows_the_numbers_and_leaves_the_dividing_to_the_reader() -> None:
    """The per-model table leaves the ratio to the reader; the sweeps keep theirs."""
    rows = report.best([_timing('specsolve'), _timing('linopy')])
    table = report.table('dispatch', rows, 'lp')
    assert 'wall: specsolve' in table and 'wall: linopy' in table, 'every library measured is still a column'
    assert '\u00f7' not in table, 'the per-model table carries no ratio'


def test_a_run_of_one_arm_has_no_ratio_column() -> None:
    """A number divided by itself is not a comparison, and a column of 1.00x
    reads like one."""
    table = report.table('dispatch', report.best([_timing('specsolve')]), 'lp')
    assert 'wall: specsolve' in table, 'the arm that ran is still a column'
    assert '\u00f7' not in table, 'nothing to divide against, so no ratio column at all'


def test_a_measurement_without_a_peak_is_skipped_rather_than_divided(tmp_path: Path) -> None:
    """`peak_rss_bytes` is `None` for a run taken without `benchmem(isolate=True)`."""
    path = tmp_path / 'results.jsonl'
    records = [_timing('specsolve'), _timing('linopy', size='l', peak_rss_bytes=None)]
    path.write_text('\n'.join(json.dumps(r) for r in records))

    taken = plot.series(path)
    assert 'l' not in taken.get(('dispatch', 'lp', 'linopy'), {}), (
        'a record with no peak cannot be plotted, so it is dropped'
    )
    assert 'm' in taken[('dispatch', 'lp', 'specsolve')], 'and the records around it still are'


def test_a_ceiling_from_the_width_ladder_does_not_bound_the_size_panel() -> None:
    """A case carries two ladders and a panel plots one of them.

    The width record is second, so a key without the ladder in it would also
    lose the size ceiling.
    """
    taken = {
        ('transport', 'highs', 'specsolve'): {r: _plotted() for r in ('xs', 's', 'm', 'l')},
        ('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's', 'm')},
    }
    ceilings = [_ceiling_record('size', 'm'), _ceiling_record('width', 'w100')]

    panel = plot.panels(taken, ceilings)['transport — highs — length']
    assert panel['series']['linopy']['bound'] == [None, None, None, '>30 s'], (
        'the size ceiling still bounds the rung above it, and the width one says nothing here'
    )


def test_a_width_panel_is_bounded_by_its_own_ladders_ceiling() -> None:
    """The other half of the pair above: a width ceiling bounds the width panel and nothing else."""
    taken = {
        ('transport', 'highs', 'specsolve'): {r: _plotted() for r in ('w1', 'w10', 'w100', 'w1000')},
        ('transport', 'highs', 'linopy'): {r: _plotted() for r in ('w1', 'w10')},
    }
    panels = plot.panels(taken, [_ceiling_record('width', 'w10')])

    assert list(panels) == ['transport — highs — width'], 'no size panel, nothing having been measured on that ladder'
    bound = panels['transport — highs — width']['series']['linopy']['bound']
    assert bound == [None, None, '>30 s', '>30 s'], 'the rungs past the ceiling say what stopped the climb'


def test_a_ceiling_on_a_rung_no_line_could_plot_bounds_nothing() -> None:
    """`series` drops a measurement taken without a peak, so a ceiling can name
    a rung the axis does not hold."""
    taken = {('transport', 'highs', 'linopy'): {r: _plotted() for r in ('xs', 's')}}
    ceilings = [_ceiling_record('size', 'm')]

    panel = plot.panels(taken, ceilings)['transport — highs — length']
    assert panel['series']['linopy']['bound'] == [None, None], 'no rung is above one the axis does not carry'


@pytest.mark.parametrize(
    ('args', 'given', 'expected'),
    [
        pytest.param((), 5, MIN_ROUNDS, id='the-plugin-default-becomes-the-documented-nine'),
        pytest.param(('--benchmark-min-rounds=3',), 3, 3, id='an-explicit-flag-wins'),
        pytest.param(('--benchmark-min-rounds', '3'), 3, 3, id='an-explicit-flag-wins-when-spaced'),
    ],
)
def test_the_rounds_default_is_the_documented_one_and_an_explicit_flag_wins(
    args: tuple[str, ...], given: int, expected: int
) -> None:
    config = SimpleNamespace(
        invocation_params=SimpleNamespace(args=args),
        option=SimpleNamespace(benchmark_min_rounds=given),
    )
    harness.pytest_configure(config)  # pyrefly: ignore[bad-argument-type]
    assert config.option.benchmark_min_rounds == expected, (
        'docs/about/benchmarks.md publishes nine rounds per measurement; a run that asks for another count keeps it'
    )


def test_the_rounds_default_is_silent_where_the_plugin_is_absent() -> None:
    """The CodSpeed job runs this same suite under a plugin with no such option."""
    config = SimpleNamespace(invocation_params=SimpleNamespace(args=()), option=SimpleNamespace())
    harness.pytest_configure(config)  # pyrefly: ignore[bad-argument-type]
    assert not hasattr(config.option, 'benchmark_min_rounds'), 'nothing to set, so nothing is set'


def test_the_rounds_default_is_wired_into_the_session(request: pytest.FixtureRequest) -> None:
    """The session applies the default."""
    if not hasattr(request.config.option, 'benchmark_min_rounds'):
        pytest.skip('pytest-benchmark is not installed — bench/ runs through `pixi run -e bench`')
    if flag_passed(request.config, '--benchmark-min-rounds'):
        pytest.skip('this session asked for a round count of its own')
    assert request.config.option.benchmark_min_rounds == MIN_ROUNDS, (
        'the documented command has to reproduce the documented method'
    )


@pytest.mark.parametrize('label', [pytest.param(s.label, id=s.label) for s in CASES['declarations'].ladder])
def test_the_generated_declaration_model_is_the_language(label: str, tmp_path: Path) -> None:
    """A generated model file still has to pass the front door."""
    from mathspec import to_spec

    case = CASES['declarations']
    shape = case.shape(label)
    schema = to_spec(str(case.spec_path(shape, cache=tmp_path)))
    n = shape.sizes['declaration']
    assert len(schema.variables) == n, 'one variable declaration per unit of the swept count'
    assert len(schema.constraints) == n + 1, 'a capacity constraint per declaration, plus one balance'


def test_the_generated_declaration_model_builds(tmp_path: Path) -> None:
    """Loading is not building (#345). The sweep's smallest rung is a million
    variables, so this builds a tiny shape of the same generated model."""
    import specsolve as sps

    case = CASES['declarations']
    shape = Shape('tiny', {'declaration': 2, 'unit': 8, 'snapshot': 20}, 20 * 16)
    paths = case.write(shape, tmp_path)
    sources = {k: v for k, v in paths.items() if k in ('p_max', 'cost', 'demand', 'unit', 'snapshot')}
    with sps.build(case.spec_path(shape, cache=tmp_path), sources) as model:
        assert model is not None


def test_only_the_masked_declaration_rungs_carry_a_where() -> None:
    """The paired rungs differ by the mask and by nothing else."""
    case = CASES['declarations']
    for shape in case.ladder:
        text = _declarations_spec(shape)
        assert ('where:' in text) == shape.masked, (
            f'{shape.label}: a where: belongs on the masked rungs and only on those'
        )


def test_the_masked_declaration_rungs_pair_with_a_dense_twin() -> None:
    ladder = {s.label: s for s in CASES['declarations'].ladder}
    masked = [s for s in ladder.values() if s.masked]
    assert [s.label for s in masked] == ['n008m', 'n128m'], 'the masked rungs are the low and high end of the axis'
    for shape in masked:
        twin = ladder[shape.label.removesuffix('m')]
        assert shape.sizes == twin.sizes, 'a masked rung and its twin build the same model, one keyword apart'
        assert shape.nominal_variables == twin.nominal_variables, 'the pair must agree on what live_fraction is of'


@pytest.mark.parametrize(
    ('counts', 'masked', 'complaint'),
    [
        pytest.param((2, 8), (4,), 'no dense twin', id='masked-without-a-twin'),
        pytest.param((2, 8), (3,), 'does not split', id='masked-count-that-does-not-divide-the-pool'),
    ],
)
def test_a_masked_rung_without_a_dense_twin_is_refused(counts, masked, complaint) -> None:
    """The pair is the measurement, so half of one is a broken ladder, not a rung."""
    with pytest.raises(ValueError, match=complaint):
        _declaration_sweep(pool=8, snapshots=10, counts=counts, masked=masked)


@pytest.mark.parametrize(
    ('label', 'sweep'),
    [
        pytest.param('n008', 'declaration', id='dense-rung'),
        pytest.param('n008m', 'declaration', id='masked-rung'),
        pytest.param('d50', 'density', id='density-rung'),
        pytest.param('w10', 'width', id='width-rung'),
        pytest.param('m', 'size ladder', id='size-rung'),
    ],
)
def test_every_rung_label_lands_in_its_own_sweep(label: str, sweep: str) -> None:
    """A masked rung is the declaration sweep's, not the size ladder's."""
    found = report._sweep_of(label)
    named = {
        report._DECLARATION_RUNG: 'declaration',
        report._DENSITY_RUNG: 'density',
        report._WIDTH_RUNG: 'width',
        None: 'size ladder',
    }
    assert named[found] == sweep, f'{label} belongs to the {sweep} axis'


@pytest.mark.parametrize(
    ('label', 'printed'),
    [
        pytest.param('n008', '8', id='count'),
        pytest.param('n128', '128', id='larger-count'),
        pytest.param('n008m', '8 masked', id='masked-twin-prints-beside-its-count'),
    ],
)
def test_a_declaration_rung_prints_its_count_and_whether_it_is_masked(label: str, printed: str) -> None:
    assert report._rung_value(label) == printed, 'the sweep column is read straight off the rung label'


def test_a_masked_rung_sorts_after_the_twin_it_ties_with() -> None:
    """A vacuous mask leaves the column count identical, so the label breaks the tie."""
    rows = report.best(
        [
            _timing(arm, case='declarations', size=size, counts={'columns': 1000, 'rows': 100, 'nonzeros': 1000})
            for size in ('n128m', 'n008', 'n128', 'n008m')
            for arm in ('specsolve', 'linopy')
        ]
    )
    order = report.sizes_of('declarations', rows, 'lp', sweep=report._DECLARATION_RUNG)
    assert order == ['n008', 'n008m', 'n128', 'n128m'], 'each rung is followed by its masked twin, counts tied'


def test_the_declaration_rungs_do_not_share_a_cache_key() -> None:
    keys = [s.key for s in CASES['declarations'].ladder]
    assert len(set(keys)) == len(keys), "rungs sharing a cache key would read each other's data and generated model"


def test_the_declaration_sweep_holds_the_model_size_flat() -> None:
    ladder = CASES['declarations'].ladder
    totals = {s.sizes['declaration'] * s.sizes['unit'] * s.sizes['snapshot'] for s in ladder}
    assert len(totals) == 1, 'a rung that moves total variables confounds the declaration axis with model size'
    assert {s.nominal_variables for s in ladder} == totals, (
        'nominal_variables must count every declaration, or live_fraction misreports the sweep'
    )


@pytest.mark.parametrize('name', [pytest.param(n, id=n) for n in sorted(CASES) if CASES[n].generate_spec is None])
def test_a_static_case_still_reads_its_committed_model(name: str) -> None:
    case = CASES[name]
    assert case.spec is not None and case.spec.exists(), 'a static case names a committed YAML file'
    assert case.spec_path(case.ladder[0]) == case.spec, (
        'spec_path must stay the committed file for every case that does not generate one'
    )


def test_the_milp_case_lowers_with_both_domains() -> None:
    """`commitment` only measures the vtype stream if the plan actually carries it."""
    from mathspec import to_spec

    program = to_spec(str(CASES['commitment'].spec)).program
    domains = {n: v.domain for n, v in program.variables.items()}
    assert domains == {'u': 'binary', 'p': 'continuous'}, (
        'the MILP case must declare one binary and one continuous variable, or vtype streaming goes unmeasured'
    )


def test_the_floor_builds_the_model_specsolve_builds() -> None:
    """The floor's counts match specsolve's on `transport/xs`, so its headroom claim is about one model.

    Columns, rows and nonzeros are the cheap fingerprint; the objectives are
    compared by the test below.
    """
    import specsolve as sps

    case = CASES[floor.CASE]
    paths = case.data(case.ladder[0])
    floor_model = floor.arrays(floor.read(paths))

    sources = checked_sources(case, case.ladder[0].label, paths)
    with sps.build(case.spec, sources) as model:
        tables = _handoff(model)
        assert floor_model.column_count == tables.column_count, 'the floor holds a different number of variables'
        assert floor_model.row_count == tables.row_count, 'the floor holds a different number of constraints'
        assert floor_model.nonzeros == tables.matrix.height, 'the floor holds a different coefficient matrix'


def test_a_spliced_basis_reproduces_the_cold_answer() -> None:
    """A carried basis may move the route and never the optimum."""
    run = warm_payoff.sweep(warm_payoff.SIZES['xs'], n_snap=4, steps=8)
    assert len(run.steps) > 1, 'a single rebuild carries nothing, so the splice would go unexercised'
    for i, step in enumerate(run.steps):
        assert step.warm_objective == pytest.approx(step.cold_objective, rel=1e-9), (
            f'step {i}: a carried basis moved the answer'
        )


def test_the_splice_shifts_a_later_declarations_rows() -> None:
    """`feasibility_cut` follows `optimality_cut`, so a row gained by the first
    moves every row of the second."""
    was = {'optimality_cut': Labelled(pl.LazyFrame(), 0, 2), 'feasibility_cut': Labelled(pl.LazyFrame(), 2, 2)}
    now = {'optimality_cut': Labelled(pl.LazyFrame(), 0, 3), 'feasibility_cut': Labelled(pl.LazyFrame(), 3, 2)}
    previous = WarmStart(
        solver='highs',
        column_statuses=np.zeros(4, dtype=np.int8),
        row_statuses=np.array([10, 11, 20, 21], dtype=np.int8),
        column_values=None,
    )
    order = ['optimality_cut', 'feasibility_cut']

    carried = warm_payoff.spliced(previous, was, now, order, 5).row_statuses
    assert list(carried) == [10, 11, warm_payoff.BASIC, 20, 21], (
        'the second declaration keeps its own statuses at its new start, and the gained row starts basic'
    )
    assert list(warm_payoff.prefixed(previous, 5).row_statuses) == [10, 11, 20, 21, warm_payoff.BASIC], (
        'the prefix carry is the mistake this splice exists to avoid; it must stay measurably different'
    )


def test_the_floor_and_specsolve_agree_on_the_answer() -> None:
    """`check()` runs, and the two models solve to one objective; the counts match a permuted floor too."""
    ours, specsolve = floor.check()

    assert ours == pytest.approx(specsolve, rel=1e-9), (
        f'the floor solves a different model than specsolve: {ours} against {specsolve}'
    )


# ---------------------------------------------------------------------------
# every verb reaches the isolated child it is measured in
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('named_arm', sorted(ARMS))
def test_every_verb_an_isolated_pass_measures_can_be_pickled(named_arm: str) -> None:
    """`benchmem(isolate=True)` ships the action to a spawned child, so a verb
    that is not picklable raises before anything is timed (#1617)."""
    module = ARMS[named_arm]
    for verb in (
        'prepare',
        'build_and_emit',
        'build_only',
        'objective',
        'window_setup',
        'sweep',
        'window',
        'read_setup',
        'read',
    ):
        target = getattr(module, verb, None)
        if target is None:
            continue
        try:
            pickle.loads(pickle.dumps(target))
        except Exception as exc:
            pytest.fail(f"the {named_arm} arm's {verb} does not pickle, so an isolated pass cannot measure it: {exc}")


def test_the_window_verb_is_split_so_the_build_stays_out_of_the_clock() -> None:
    """The build is outside the measured call and inside the spawned child, so
    `window_setup` and `window` come as a pair."""
    for name, module in sorted(ARMS.items()):
        assert hasattr(module, 'window') == hasattr(module, 'window_setup'), (
            f'the {name} arm offers one half of the window pair — `window_setup` builds and loads '
            f'outside the clock, `window` is what is timed, and neither means anything alone'
        )
        assert hasattr(module, 'read') == hasattr(module, 'read_setup'), (
            f'the {name} arm offers one half of the read pair — `read_setup` builds and answers '
            f'outside the clock, `read` is what is timed, and neither means anything alone'
        )


def test_the_window_payload_an_isolated_pass_ships_can_be_pickled() -> None:
    """The `(action, setup)` pair `benchmem(isolate=True)` pickles to its child (#1617).

    Each half can pickle on its own and the payload still carry a closure. A
    payload over 1 MiB ships pre-built state.
    """
    plugin = pytest.importorskip(
        'pytest_benchmem.pytest_plugin',
        reason='pytest-benchmem is not installed — bench/ runs through `pixi run -e bench`',
    )

    from bench.test_ladder import _CollectedSetup

    prepared = ('spec.yaml', {'p': 'p.parquet'})
    for name, module in sorted(ARMS.items()):
        if not hasattr(module, 'window'):
            continue
        setup = _CollectedSetup(partial(module.window_setup, 'highs', prepared, prepared, 'values'))
        mem_setup, tracked = plugin._pedantic_action(module.window, (), {}, setup)
        try:
            blob = pickle.dumps((tracked, mem_setup))
        except Exception as exc:
            pytest.fail(f"the {name} arm's window payload does not reach an isolated child: {exc}")
        assert len(blob) < 1024 * 1024, (
            f'the {name} arm ships {len(blob)} bytes to the child, so it carries pre-built state — '
            f'the isolated rss would measure deserializing that, not the window'
        )


def test_a_timing_record_says_which_rung_it_came_off(tmp_path: Path) -> None:
    """`test_emit` and both `test_window` changes measure the same cell, so without a phase they are one key."""
    cell = {'case_name': 'dispatch', 'size': 'xs', 'arm': 'specsolve', 'sink': 'highs'}
    doc = {
        'benchmarks': [
            {
                'name': f'{rung}[dispatch-xs-specsolve-highs]',
                'params': {**cell, **extra},
                'stats': {'median': 1.0, 'min': 1.0},
                'extra_info': {},
            }
            for rung, extra in (
                ('test_emit', {}),
                ('test_window', {'change': 'values'}),
                ('test_window', {'change': 'shape'}),
                ('test_window', {'change': 'one'}),
                ('test_window', {'change': 'cold'}),
                ('test_window', {}),
                ('test_sweep', {}),
                ('test_read', {'into': 'frames'}),
            )
        ]
    }
    path = tmp_path / 'latest.json'
    path.write_text(json.dumps(doc))
    phases = [r.get('phase') for r in bench_results.records(path) if r.get('record') == 'timing']
    assert phases == [
        'emit',
        'window',
        'window-reshaped',
        'window-one',
        'window-cold',
        'window',
        'sweep',
        'read-frames',
    ], (
        'each rung names its own phase, in the order the file writes them, and a window from before '
        'the change was a parameter is the values window it measured'
    )


@pytest.mark.parametrize('change', sorted(RELOADS))
@pytest.mark.parametrize(
    'case_name', [pytest.param(n, id=n) for n in sorted(CASES) if any(s.label == 'xs' for s in CASES[n].ladder)]
)
def test_a_window_takes_the_path_its_change_names(case_name: str, change: str) -> None:
    """`test_window`'s changes are there to measure the paths an update can take.

    The ladder checks which one ran, but only when it measures; this holds every
    case's smallest rung to it on every pull request, and with it that the
    shorter rung generates and builds at all.
    """
    module = ARMS['specsolve']
    case = CASES[case_name]
    shape = case.shape('xs')
    prepared = module.prepare(case_name, 'xs', case.data(shape), {})
    following = prepared
    if change == 'shape':
        following = module.prepare(case_name, 'xs', case.data(shortened(shape)), {})
    args, kwargs = module.window_setup('highs', prepared, following, change)
    counts = module.window(*args, **kwargs)
    assert counts['reloaded'] == RELOADS[change], (
        f'{case_name}: a {change} window should '
        f'{"load the solver from scratch" if RELOADS[change] else "push onto the loaded solver"}'
    )


def test_a_window_measurement_is_not_published_as_a_build(tmp_path: Path) -> None:
    """A later window is faster than the build, so in the same key it would replace it.

    Both published readers take the `emit` phase; `bench.tidy` carries both.
    """
    build = _timing('specsolve', phase='emit', wall_seconds=1.0)
    window = _timing('specsolve', phase='window', wall_seconds=0.1)

    path = tmp_path / 'latest.jsonl'
    path.write_text('\n'.join(json.dumps(r) for r in (build, window)))
    _run, _gates, timings, _loop = report.load(path)
    published = report.best(timings)
    assert published[('dispatch', 'm', 'lp', 'specsolve')]['wall_seconds'] == 1.0, (
        'the table publishes the build, not the faster window measured against it'
    )
    plotted = plot.series(path)[('dispatch', 'lp', 'specsolve')]['m']
    assert plotted['wall'] == 1.0, 'and the chart page plots the build, not the window that shares its key'

    rows = list(tidy.measurements([build, window], 'run'))
    assert sorted({row['phase'] for row in rows}) == ['emit', 'window'], (
        'the long CSV carries both, under phases that tell them apart'
    )


def test_the_specsolve_arm_splits_each_verb_by_the_engine_clock() -> None:
    """Every verb reports the engine's phases, and a window only its own share of them.

    The engine's clock sums over the model's life, so a window that reported it
    whole would carry the setup's build too.
    """
    module = ARMS['specsolve']
    case = CASES['dispatch']
    prepared = module.prepare('dispatch', 'xs', case.data(case.shape('xs')), {})

    emitted = module.build_and_emit('lp', prepared)['phases']
    assert {'attach', 'build', 'write'} <= emitted.keys(), 'an LP emit is attached, built and written'
    assert {'attach', 'build'} <= module.build_only(prepared)['phases'].keys(), 'a build is attached and built'

    args, kwargs = module.window_setup('highs', prepared, prepared, 'values')
    model = args[0]
    window = module.window(*args, **kwargs)['phases']
    assert {'attach', 'build', 'handoff'} <= window.keys(), 'a window rebuilds and hands off'
    whole = model._engine._seconds
    assert all(0 <= window[phase] < whole[phase] for phase in ('attach', 'build')), (
        "a window's phases are its own, not the setup's build summed in"
    )


def test_the_long_table_carries_each_phase_as_its_own_metric() -> None:
    record = _timing('specsolve', phase='emit', phase_seconds={'attach': 0.1, 'build': 0.4})
    rows = {row['metric']: row['value'] for row in tidy.measurements([record], 'run')}
    assert (rows['attach_seconds'], rows['build_seconds']) == (0.1, 0.4), 'each phase is a row of its own'
    assert rows['wall_seconds'] == 1.0, 'beside the wall time, not instead of it'


@pytest.mark.parametrize('into', ['frames', 'parquet'])
@pytest.mark.parametrize(
    'case_name', [pytest.param(n, id=n) for n in sorted(CASES) if any(s.label == 'xs' for s in CASES[n].ladder)]
)
def test_an_answer_reads_back_without_a_solve(case_name: str, into: str) -> None:
    """`test_read`'s synthetic answer is one the engine lays out, every case's smallest rung, both ways.

    A mixed-integer case is answered without duals, and a reader that asked for
    them would raise.
    """
    module = ARMS['specsolve']
    case = CASES[case_name]
    prepared = module.prepare(case_name, 'xs', case.data(case.shape('xs')), {})
    args, kwargs = module.read_setup(prepared, into)
    model, answer, _ = args
    assert answer.primal.len() == _handoff(model).column_count, 'the answer spans every built column'
    assert (answer.dual is None) == bool(model._engine._discrete()), (
        'duals exactly where a real solve leaves them: none for a model with an integer variable'
    )
    assert module.read(*args, **kwargs)['columns'] == answer.primal.len(), 'and reads back the build it answered'


@pytest.mark.parametrize(
    'case_name', [pytest.param(n, id=n) for n in sorted(CASES) if any(s.label == 'xs' for s in CASES[n].ladder)]
)
def test_a_sweep_loads_once_and_says_what_the_solver_took(case_name: str) -> None:
    """`test_sweep` asserts one load and attributes the solve, on every case's smallest rung."""
    module = ARMS['specsolve']
    case = CASES[case_name]
    counts = module.sweep('highs', module.prepare(case_name, 'xs', case.data(case.shape('xs')), {}))
    assert counts['loads'] == 1, 'the first slice loads and every later one, its values unchanged, pushes'
    assert counts['phases']['solve'] > 0, "the solver's share is attributed, not left inside the wall time"


def test_a_sink_whose_solver_is_not_installed_is_skipped_by_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every arm reaching a solver sink skips, naming the package, rather than failing inside the clock."""
    import importlib.util

    from bench import arms

    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, 'find_spec', lambda name, *a: None if name == 'xpress' else real(name, *a))
    reason = arms.unmeasurable('specsolve', 'dispatch', 'xpress')
    assert reason is not None and 'xpress' in reason, 'a missing solver package names itself as the reason'


@pytest.mark.parametrize('named_sink', harness.SINKS)
def test_every_sink_has_a_caption_the_report_prints(named_sink: str) -> None:
    """A table for a sink the report has no sentence for would raise when the page is written."""
    assert report._SEAM.get(named_sink), (
        f'{named_sink} reaches the report with nothing saying what each arm ended up holding'
    )
