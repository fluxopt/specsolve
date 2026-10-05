"""The ``specsolve`` command: check, solve and list the runs of a manifest.

Typer is the ``cli`` extra's, so it is imported when the command runs, and
``main`` without it says how to install it.
"""

from __future__ import annotations

import importlib.util
import sys
import time
import warnings
from typing import TYPE_CHECKING

from specsolve.api import tidy
from specsolve.manifest import INSTALL_HINT, Manifest, Run, load_manifest, read_sources
from specsolve.relational.result import Result

if TYPE_CHECKING:
    import typer

    from specsolve.sweep import Sweep

#: The manifest a command reads when none is named.
DEFAULT_MANIFEST = 'specsolve.yaml'

#: The suffixes that mark a command's first argument as the manifest rather than a run.
MANIFEST_SUFFIXES = ('.yaml', '.yml')

#: What a command's arguments are, as its help shows them.
ARGUMENTS = '[MANIFEST] [RUN]...'


def main() -> None:
    """Run the ``specsolve`` command; without typer installed, exit naming the extra that installs it."""
    if importlib.util.find_spec('typer') is None:
        sys.exit(f'the specsolve command needs typer, which the cli extra installs: {INSTALL_HINT}')
    app()()


def app() -> typer.Typer:
    """The command, as a Typer application: ``check``, ``solve`` and ``list``."""
    import typer

    command = typer.Typer(no_args_is_help=True, add_completion=False)
    arguments = typer.Argument(None, metavar=ARGUMENTS, show_default=False)

    @command.command('check')
    def check_runs(names: list[str] = arguments) -> None:
        """Read every run's spec and data, and report each problem by run. Exit 1 on any."""
        _, selected = _selected(names or [])
        failed = False
        for run in selected:
            problem = _problem(run)
            failed = failed or problem is not None
            typer.echo(f'{run.name}: {problem or "ok"}')
        if failed:
            raise typer.Exit(1)

    @command.command('solve')
    def solve_runs(names: list[str] = arguments) -> None:
        """Solve the named runs, or every run, in manifest order; skip a run whose archive exists.

        Exit 1 where a run failed or ended other than optimal.
        """
        _, selected = _selected(names or [])
        failed = False
        for run in selected:
            if run.archive is not None and run.archive.exists():
                typer.echo(f'{run.name}: skipped, {run.archive} exists')
                continue
            started = time.perf_counter()
            try:
                answer = run.solve()
            except Exception as error:  # one run failing is reported, and the next still runs
                typer.echo(f'{run.name}: failed: {error}')
                failed = True
                continue
            summary, optimal = _summary(answer)
            failed = failed or not optimal
            typer.echo(f'{run.name}: {summary}, {time.perf_counter() - started:.1f}s')
        if failed:
            raise typer.Exit(1)

    @command.command('list')
    def list_runs(names: list[str] = arguments) -> None:
        """List the runs: whether each solves or sweeps, and whether its archive exists."""
        _, selected = _selected(names or [])
        width = max((len(run.name) for run in selected), default=0)
        for run in selected:
            verb = 'sweep' if run.is_sweep else 'solve'
            typer.echo(f'{run.name:<{width}}  {verb}  {_archived(run)}')

    return command


def _selected(arguments: list[str]) -> tuple[Manifest, list[Run]]:
    """The manifest the arguments name, or the default, and the runs they name, or every run.

    The first argument is the manifest where it ends in ``.yaml`` or ``.yml``,
    so ``specsolve solve base`` reads ``base`` as a run. An unknown run exits 2,
    naming the runs there are.
    """
    import typer

    named = arguments[0] if arguments and arguments[0].endswith(MANIFEST_SUFFIXES) else None
    names = arguments[1:] if named else arguments
    try:
        manifest = load_manifest(named or DEFAULT_MANIFEST)
    except (OSError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(1) from None
    unknown = [name for name in names if name not in manifest.runs]
    if unknown:
        typer.echo(f'{manifest.path} has no run {", ".join(unknown)}. Runs: {", ".join(manifest.runs)}.', err=True)
        raise typer.Exit(2)
    return manifest, [manifest.runs[name] for name in names] if names else list(manifest.runs.values())


def _problem(run: Run) -> str | None:
    """What stops *run* attaching its data, or ``None``; a sweep attaches every slice.

    Advice the spec draws is not a problem, and is not reported. What a sweep
    refuses of the model as a whole, such as a window its rows cannot be cut
    into, is the solve's to report.
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            sources = read_sources(run.sources)
            parts = [sources] if run.axis is None else [part for _, part in run.axis.slices(sources)]
            for part in parts:
                tidy(run.spec, part)
    except Exception as error:  # every problem of every run is reported, not the first raised
        return f'{type(error).__name__}: {error}'
    return None


def _summary(answer: Result | Sweep) -> tuple[str, bool]:
    """How a run ended, as one line, and whether every solve in it reached an optimum."""
    if isinstance(answer, Result):
        return (
            f'{answer.termination_condition}, objective {answer.objective:.6g}',
            answer.termination_condition == 'optimal',
        )
    ended = answer.record['termination_condition'].value_counts(sort=True)
    counts = ', '.join(f'{count} {condition}' for condition, count in ended.iter_rows())
    return f'{answer.record.height} slices: {counts}', set(ended['termination_condition']) == {'optimal'}


def _archived(run: Run) -> str:
    if run.archive is None:
        return 'no archive'
    return 'archived' if run.archive.exists() else 'not archived'
