"""The harness: selection, and the data every arm reads.

Every moving part is pytest's own: the case x size x sink x arm product is a
`parametrize`, isolation is `benchmem(isolate=True)`, repetition is
pytest-benchmark's rounds, and the output is `--benchmark-json`.

Cases have different ladders, so the (case, size) axis is built here: a rung a
case does not have is skipped. `case_name` and `size` stay two parameters, so
pytest-benchmem can group by each.
"""

from __future__ import annotations

import contextlib
import math
import os
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from bench.arms import ARMS
from bench.cases import CASES
from bench.published import files, ladder_defaults, ladder_of

if TYPE_CHECKING:
    from collections.abc import Sequence

    from bench.cases import Shape


#: Every destination `--sinks` can name.
SINKS = ('lp', 'mps', 'highs', 'gurobi', 'xpress')


def pytest_addoption(parser: pytest.Parser) -> None:
    g = parser.getgroup('ladder', 'the specsolve benchmark ladder')
    g.addoption('--cases', nargs='+', default=sorted(CASES), choices=sorted(CASES))
    g.addoption('--sizes', nargs='+', default=['xs', 's', 'm'], help="rung labels, or 'all' for every rung a case has")
    g.addoption('--arms', nargs='+', default=sorted(ARMS), choices=sorted(ARMS))
    g.addoption(
        '--sinks',
        nargs='+',
        default=['lp', 'highs'],
        choices=SINKS,
        help='where each built model goes. `lp` and `highs` by default: the LP file is the '
        "artifact fewest callers want, and it is not the same comparison — HiGHS's own model "
        'is resident in both arms and narrows the gap. `gurobi` is opt-in because it needs the '
        '[gurobi] extra, and it is measured against linopy the same way, through `to_gurobipy()`. '
        '`mps` is the other file format, and `xpress` the third solver, which only specsolve '
        'reaches; its bundled Community licence limits the solve, and nothing here solves.',
    )
    g.addoption('--builds', type=int, default=5, help='rebuilds per process in the first-vs-steady pass; 0 skips it')
    g.addoption(
        '--budget',
        type=float,
        default=120.0,
        help='seconds a measurement may take before its arm stops climbing that ladder; 0 measures everything',
    )
    g.addoption(
        '--memory-budget',
        type=float,
        default=0.0,
        help='GB a measurement may take before its arm stops climbing that ladder; 0 measures everything',
    )
    g.addoption(
        '--i-know-another-is-running',
        action='store_true',
        help='start despite the machine lock or a high load average (#705); the numbers are on you',
    )


#: Rounds a measurement gets, by how long one round is expected to take: nine
#: below a second, five below five, three above. A slow cell's median moves least
#: when rounds are dropped, and it is where the run spends its hours.
ROUNDS = ((1.0, 9), (5.0, 5), (math.inf, 3))


def flag_passed(config: pytest.Config, flag: str) -> bool:
    """Whether *flag* was given on the command line, in either `--x=v` or `--x v` form."""
    return any(arg == flag or arg.startswith(f'{flag}=') for arg in config.invocation_params.args)


def rounds_for(seconds: float | None) -> int:
    """How many rounds a cell gets whose round is expected to take *seconds*; the most where nothing is known."""
    return next(rounds for below, rounds in ROUNDS if seconds is None or seconds < below)


_PACE = pytest.StashKey[dict]()


def _pace_key(node: Any) -> tuple:
    """One cell's identity up to its rung: the test and every parameter but the size, and which ladder."""
    params = node.callspec.params
    rest = tuple(sorted((k, str(v)) for k, v in params.items() if k != 'size'))
    return (node.originalname, rest, ladder_of(params['size']))


def expected_seconds(node: Any) -> float | None:
    """The last rung this same cell measured on its ladder, scaled by how much wider this rung is.

    None until the cell has measured a rung of that ladder.
    """
    last = node.config.stash.get(_PACE, {}).get(_pace_key(node))
    if last is None:
        return None
    size, seconds = last
    case_name, here = node.callspec.params['case_name'], node.callspec.params['size']
    return seconds * shape_of(case_name, here).nominal_variables / shape_of(case_name, size).nominal_variables


def remember(node: Any, seconds: float | None) -> None:
    """Keep this cell's median, for `expected_seconds` at the rung above."""
    if seconds is not None:
        node.config.stash.setdefault(_PACE, {})[_pace_key(node)] = (node.callspec.params['size'], seconds)


#: Machine-global, so a run from another worktree is refused too (#705).
BENCH_LOCK = Path(tempfile.gettempdir()) / 'specsolve-bench.lock'

_TOOK_LOCK = pytest.StashKey[bool]()


def _holder_if_alive(path: Path) -> str | None:
    """The lock's own description of its holder, or None when that process is gone.

    A lock that cannot be read as `pid N, ...`, or whose pid this user may not
    signal, counts as held.
    """
    try:
        content = path.read_text().strip()
    except FileNotFoundError:
        return None
    try:
        os.kill(int(content.removeprefix('pid ').split(',')[0]), 0)
    except ProcessLookupError:
        return None
    except (ValueError, PermissionError):
        pass  # unreadable, or another user's live process: treat the lock as held
    return content


def take_lock(path: Path) -> None:
    """Claim the machine for one benchmark session, or refuse to start.

    The claim is atomic. A lock whose holder is no longer alive is evicted.

    Raises:
        pytest.UsageError: If another live session holds the lock.
    """
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = _holder_if_alive(path)
        if holder is not None:
            raise pytest.UsageError(
                f'another benchmark is running ({holder}). Benchmarks do not share a machine: '
                f'the numbers would be wrong and would look fine. Wait, or pass '
                f'--i-know-another-is-running. Lock: {path}'
            ) from None
        path.unlink(missing_ok=True)
        take_lock(path)
        return
    os.write(fd, f'pid {os.getpid()}, started {time.strftime("%H:%M")}'.encode())
    os.close(fd)


def refuse_unless_idle(load1: float, cores: int) -> None:
    """Refuse to benchmark a machine that is already working.

    Raises:
        pytest.UsageError: If the 1-minute load average exceeds the core count.
    """
    if load1 > cores:
        raise pytest.UsageError(
            f'1-minute load average is {load1:.2f} on {cores} cores — this machine is already '
            f'working, and a benchmark taken now would be wrong and look fine (#419 measured '
            f'noise floors of 176% and 344% at load 25.78 on 8 cores). Wait for idle, or pass '
            f'--i-know-another-is-running.'
        )


#: How many copies of the model a measurement holds at once — the timed rounds
#: in this process, and `benchmem(isolate=True)` again in a child. The
#: projection to the next rung is weighed at all of them; the rung just
#: measured, at one.
COPIES_HELD = 2


def published_results() -> dict[Path, set[str]]:
    """The result files the published run writes, to the rungs each holds — derived from the task."""
    defaults = ladder_defaults()
    return {
        (Path.cwd() / path).resolve(): set(rungs)
        for sink in defaults['sinks'].split()
        for case in defaults['cases'].split()
        for path, rungs in files(sink, case, defaults['sizes'].split()).items()
    }


def refuse_to_overwrite_the_provenance(config: pytest.Config) -> None:
    """A short run may not write the file the published tables are drawn from.

    Narrower sinks or libraries are allowed; a run missing a published rung is
    not.

    Raises:
        pytest.UsageError: If the run is missing a published rung and would
            still write the committed file.
    """
    destination = next(
        (arg.split('=', 1)[1] for arg in config.invocation_params.args if arg.startswith('--benchmark-json=')), None
    )
    rungs = published_results().get(Path(destination).resolve()) if destination else None
    if rungs is None:
        return
    missing = rungs - set(config.getoption('--sizes'))
    if missing:
        raise pytest.UsageError(
            f'this run leaves out {sorted(missing)}, so it cannot write {destination} — the published '
            f'tables are drawn from that file and a shorter run replaces them with fewer rows, '
            f'silently. Point --benchmark-json somewhere else, or take the whole ladder '
            f'(`pixi run ladder`).'
        )


def pytest_sessionstart(session: pytest.Session) -> None:
    """One benchmark per machine, refused up front rather than found in the numbers (#705).

    CI is exempt: its runners are single-purpose.
    """
    refuse_to_overwrite_the_provenance(session.config)
    if session.config.getoption('--i-know-another-is-running') or os.environ.get('CI'):
        return
    take_lock(BENCH_LOCK)
    session.config.stash[_TOOK_LOCK] = True
    refuse_unless_idle(os.getloadavg()[0], os.cpu_count() or 1)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Release the machine — only a lock this session took, never another holder's."""
    if session.config.stash.get(_TOOK_LOCK, False):
        BENCH_LOCK.unlink(missing_ok=True)


#: Fingerprinted into every result file: a number measured against a different
#: version of any of these is a different number.
TRACKED = (
    'specsolve',
    'highspy',
    'gurobipy',
    'scipy',
    'polars',
    'pandas',
    'numpy',
    'pyarrow',
    'pytest-benchmem',
)


@pytest.hookimpl(optionalhook=True)
def pytest_benchmark_update_machine_info(config: pytest.Config, machine_info: dict[str, Any]) -> None:
    """Stamp the result file with the dependency versions and the load average (#705).

    ``optionalhook``, because the CodSpeed job runs without pytest-benchmark.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for pkg in TRACKED:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = None
    machine_info['versions'] = versions
    machine_info['load_avg'] = os.getloadavg()


def _rungs(config: pytest.Config) -> list[tuple[str, str]]:
    """Every (case, rung) the selection asks for and the case actually has."""
    wanted = config.getoption('--sizes')
    out = []
    for name in config.getoption('--cases'):
        labels = [s.label for s in CASES[name].ladder]
        out += [(name, s) for s in (labels if wanted == ['all'] else wanted) if s in labels]
    return out


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    names = set(metafunc.fixturenames)
    if {'case_name', 'size'} <= names:
        rungs = _rungs(metafunc.config)
        metafunc.parametrize(('case_name', 'size'), rungs, ids=[f'{c}-{s}' for c, s in rungs])
    if 'arm' in names:
        metafunc.parametrize('arm', metafunc.config.getoption('--arms'))
    if 'sink' in names:
        metafunc.parametrize('sink', metafunc.config.getoption('--sinks'))


#: What CodSpeed does not run. `test_rebuild`, because its instruments ignore
#: rounds in pedantic mode; `test_sweep`, because it solves, and the memory
#: instrument tracks the solver's own allocations — `commitment` swept at `s` ran
#: that 20-minute job out on its own; `test_fresh`, because its work happens in a
#: child process that no instrument follows.
NOT_UNDER_CODSPEED = ('test_rebuild', 'test_sweep', 'test_fresh')


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Deselect `NOT_UNDER_CODSPEED` under CodSpeed."""
    if not getattr(config.option, 'codspeed', False):
        return
    dropped = [i for i in items if i.name.startswith(NOT_UNDER_CODSPEED)]
    if dropped:
        config.hook.pytest_deselected(items=dropped)
        items[:] = [i for i in items if not i.name.startswith(NOT_UNDER_CODSPEED)]


@pytest.fixture(scope='session')
def paths() -> Any:
    """``(case, rung) -> parquet paths``, generated once per session outside every measured region."""

    def resolve(case_name: str, size: str) -> dict[str, str]:
        case = CASES[case_name]
        return case.data(case.shape(size))

    return resolve


@pytest.fixture(scope='session')
def builds(request: pytest.FixtureRequest) -> int:
    return int(request.config.getoption('--builds'))


#: Where the cell being measured is noted, for a watchdog that outlives the
#: process measuring it.
INFLIGHT = Path(__file__).resolve().parent / 'results' / '.inflight'

#: Where a ladder stopped, and why.
CEILINGS = pytest.StashKey[dict]()


class Ceiling:
    """Which arms have stopped climbing which ladders, and the sentence why.

    The projection to the next rung is stated in the reason, never recorded as
    a measurement.
    """

    def __init__(self, budget: float, stash: dict, selected: Sequence[str], memory: float = 0.0) -> None:
        self.budget = budget
        self.memory = memory
        self.reasons = stash
        self.selected = selected

    def reached(self, arm: str, case_name: str, size: str, sink: str) -> str | None:
        found = self.reasons.get((arm, case_name, ladder_of(size), sink))
        return found[1] if found else None

    def rows(self) -> list[dict[str, Any]]:
        """Every ceiling as a record, for the file beside the measurements; ``stopped_by`` names the budget."""
        return [
            {
                'record': 'ceiling',
                'arm': arm,
                'case': case_name,
                'ladder': ladder,
                'sink': sink,
                'size': size,
                'budget': self.budget,
                'memory_budget': self.memory,
                'stopped_by': stopped_by,
                'reason': reason,
            }
            for (arm, case_name, ladder, sink), (size, reason, stopped_by) in self.reasons.items()
        ]

    def record(
        self,
        arm: str,
        case_name: str,
        size: str,
        sink: str,
        seconds: float | None,
        peak_bytes: float | None = None,
        rungs: Sequence[str] | None = None,
    ) -> None:
        """Take one measurement, and decide whether the next rung is worth taking.

        *rungs* is the part of the selection this measurement climbs, where it
        leaves some out; the next rung is the next of those. Memory is checked before time: `transport/w100` on `linopy` takes 51 s
        and 31 GB, and the memory took the machine down (#1416).
        """
        key = (arm, case_name, ladder_of(size), sink)
        growth = _growth(case_name, size, self.selected if rungs is None else rungs)
        if self.memory and peak_bytes:
            self._memory(key, arm, case_name, size, peak_bytes, growth)
            if key in self.reasons:
                return
        if not self.budget or seconds is None:
            return
        if seconds > self.budget:
            self.reasons[key] = (
                size,
                f'{arm} took {_seconds(seconds)} on {case_name}/{size}, over the {_seconds(self.budget)} budget',
                'time',
            )
            return
        projected = seconds * growth
        if projected > self.budget:
            self.reasons[key] = (
                size,
                f'{arm} took {_seconds(seconds)} on {case_name}/{size}, so the next rung projects to '
                f'{_seconds(projected)} — over the {_seconds(self.budget)} budget',
                'time',
            )

    def _memory(self, key: tuple, arm: str, case_name: str, size: str, peak_bytes: float, growth: float) -> None:
        """Stop the ladder when this rung, or the next, does not fit the machine."""
        peak = peak_bytes / 1e9
        if peak > self.memory:
            self.reasons[key] = (
                size,
                f'{arm} took {peak:.3g} GB on {case_name}/{size}, over the {self.memory:.3g} GB budget',
                'memory',
            )
            return
        projected = peak * growth * COPIES_HELD
        if projected > self.memory:
            self.reasons[key] = (
                size,
                f'{arm} took {peak:.3g} GB on {case_name}/{size}, so the next rung projects to '
                f'{projected:.3g} GB — over the {self.memory:.3g} GB budget',
                'memory',
            )


def _seconds(value: float) -> str:
    """Seconds at a precision that survives both ends: a 0.013 s rung and a 4000 s projection."""
    return f'{value:.3g} s'


def _growth(case_name: str, size: str, selected: Sequence[str]) -> float:
    """How much wider the next rung of this run's selection is. 1.0 when there is none."""
    ladder = ladder_of(size)
    rungs = [s for s in CASES[case_name].ladder if s.label in set(selected) and ladder_of(s.label) == ladder]
    labels = [s.label for s in rungs]
    if size not in labels or labels.index(size) + 1 >= len(rungs):
        return 1.0
    here, following = rungs[labels.index(size)], rungs[labels.index(size) + 1]
    return following.nominal_variables / here.nominal_variables if here.nominal_variables else 1.0


@pytest.fixture(scope='session')
def ceiling(request: pytest.FixtureRequest) -> Ceiling:
    stash = request.config.stash.setdefault(CEILINGS, {})
    return Ceiling(
        float(request.config.getoption('--budget')),
        stash,
        request.config.getoption('--sizes'),
        float(request.config.getoption('--memory-budget')),
    )


def pytest_terminal_summary(terminalreporter: Any, exitstatus: int, config: pytest.Config) -> None:
    """Every ladder an arm stopped climbing, printed where the run ends."""
    del exitstatus
    reasons = config.stash.get(CEILINGS, {})
    if not reasons:
        return
    terminalreporter.write_sep('-', 'over budget')
    for key in sorted(reasons):
        where = f' [{key[-1]} sink]' if key[-1] else ''
        terminalreporter.write_line(f'{reasons[key][1]}{where}')
    _write_ceilings(config)


def pytest_runtest_logstart(nodeid: str, location: tuple) -> None:
    """Leave the cell about to be measured where `bench/memory-watchdog.sh` can read it."""
    del location
    with contextlib.suppress(OSError):
        INFLIGHT.parent.mkdir(parents=True, exist_ok=True)
        INFLIGHT.write_text(nodeid)


def _write_ceilings(config: pytest.Config) -> None:
    """The ceilings, beside the file `--benchmark-json` names, read off the command line."""
    import json

    reasons = config.stash.get(CEILINGS, {})
    destination = next(
        (arg.split('=', 1)[1] for arg in config.invocation_params.args if arg.startswith('--benchmark-json=')), None
    )
    if not destination or not reasons:
        return
    stash = Ceiling(
        float(config.getoption('--budget')),
        reasons,
        config.getoption('--sizes'),
        float(config.getoption('--memory-budget')),
    )
    path = Path(str(destination)).with_suffix('.ceilings.json')
    path.write_text(json.dumps(stash.rows(), indent=1))


def shape_of(case_name: str, size: str) -> Shape:
    return CASES[case_name].shape(size)
