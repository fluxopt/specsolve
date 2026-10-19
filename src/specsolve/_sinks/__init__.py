"""Sinks: how a built model leaves the engine. See README.md.

A *solver* runs the handoff and is chosen by name (``SOLVERS``); a *writer*
renders it to a file and is chosen by suffix (``WRITERS``). One module per
sink, named for it. Each reads ``handoff.py`` and declares what it takes in
``capabilities.py``; a solver shares ``solver.py`` and a writer ``text.py``,
and neither family imports the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from specsolve._messages import unknown_name_message
from specsolve._sinks import capabilities as caps
from specsolve._sinks.capabilities import spelled
from specsolve._sinks.gurobi import Gurobi
from specsolve._sinks.handoff import Handoff
from specsolve._sinks.highs import Highs
from specsolve._sinks.lp_file import LP_FILE_CAPABILITIES, write_lp_file
from specsolve._sinks.mps_file import MPS_FILE_CAPABILITIES, write_mps_file
from specsolve._sinks.solver import Solver
from specsolve._sinks.xpress import Xpress
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence
    from pathlib import Path
    from typing import Any

    Write = Callable[[Handoff, Path], None]

__all__ = [
    'SOLVERS',
    'WRITERS',
    'Handoff',
    'Solver',
    'Writer',
    'loaded',
    'refusal',
    'sink_capabilities',
    'solver',
    'writer',
]

#: Every solver a caller may name. Closed.
SOLVERS: Mapping[str, type[Solver]] = {
    'highs': Highs,
    'gurobi': Gurobi,
    'xpress': Xpress,
}


@dataclass(frozen=True)
class Writer:
    """One format: how to render it, and what it can carry."""

    write: Write
    capabilities: caps.Capabilities


#: What can be written today, by suffix. Closed.
WRITERS: Mapping[str, Writer] = {
    '.lp': Writer(write_lp_file, LP_FILE_CAPABILITIES),
    '.mps': Writer(write_mps_file, MPS_FILE_CAPABILITIES),
}


def solver(name: str) -> type[Solver]:
    """The solver called *name*, or why this build cannot give you it.

    Raises:
        SpecsolveError: A name outside the closed set.
        ModuleNotFoundError: A name inside it whose package this environment
            does not have.
    """
    try:
        found = SOLVERS[name]
    except KeyError:
        raise SpecsolveError(
            f'unknown solver {name!r} — this build solves with {", ".join(sorted(SOLVERS))}.'
        ) from None
    if not found.is_available():
        raise ModuleNotFoundError(
            f'{name} is a solver this build knows, but its package is not installed here. {found.unavailable_message}'
        )
    return found


def loaded(
    held: Solver | None,
    name: str,
    handoff: Handoff,
    solver_options: Mapping[str, Any] | None = None,
) -> Solver:
    """The solver to run *handoff* on — *held* where it may keep what it holds.

    *held* is kept exactly when it is the named class holding a model that
    differs from this one in nothing but numbers — same
    [`structure`][specsolve._sinks.handoff.Handoff.structure], same options — and
    then the new numbers are pushed onto it. Otherwise it is closed, after
    *name* is resolved.
    """
    wanted = solver(name)
    if held is not None:
        if type(held) is wanted and held.keeps(handoff, solver_options):
            held.push(handoff)
            return held
        held.close()
    return wanted(handoff, solver_options)


def writer(suffix: str) -> Writer:
    """The writer for *suffix*.

    Raises:
        SpecsolveError: A format nothing writes; the message lists what can be.
    """
    if suffix in WRITERS:
        return WRITERS[suffix]
    raise SpecsolveError(f'unknown output format {suffix!r} — this build writes {", ".join(sorted(WRITERS))}.')


def sink_capabilities(name: str) -> caps.Capabilities:
    """What the sink called *name* can ingest — a solver name, or a suffix.

    Answered without importing the solver.

    Raises:
        SpecsolveError: A name belonging to neither family.
    """
    if name in SOLVERS:
        return SOLVERS[name].capabilities
    if (suffix := name.lower()) in WRITERS:
        return WRITERS[suffix].capabilities
    raise SpecsolveError(unknown_name_message('sink', name, (*SOLVERS, *WRITERS)))


def _blocker(name: str, needed: Collection[caps.Capability]) -> Callable[[Sequence[str]], str] | None:
    """The sink called *name*'s refusal of *needed* as a function of the takers, or ``None``.

    The one home for what "takes" means, so the refusal and the takers it names
    cannot disagree.
    """
    table = sink_capabilities(name)
    if missing := table.missing(needed):
        return lambda takers: _sink_refuses_message(name, missing, takers)
    if combination := table.excluded(needed):
        return lambda takers: _sink_refuses_combination_message(name, sorted(combination), takers)
    return None


def refusal(handoff: Handoff, name: str) -> str | None:
    """The sink called *name*'s refusal of the built model in *handoff*, or ``None`` where it takes it.

    The refusal names the construct, the sink, and the sinks that do take it.
    Answered without importing the sink.
    """
    needed = caps.required(handoff)
    if (refuses := _blocker(name, needed)) is None:
        return None
    return refuses([other for other in (*SOLVERS, *WRITERS) if other != name and _blocker(other, needed) is None])


def _instead(takers: Sequence[str]) -> str:
    """The third clause of the refusal contract: who *does* take it."""
    if not takers:
        return 'No sink this build has takes it.'
    return f'Sinks that do take it: {", ".join(sorted(takers))}.'


def _sink_refuses_combination_message(sink: str, combination: Sequence[str], takers: Sequence[str]) -> str:
    """A sink that has both halves of a pair and refuses them together."""
    return (
        f'the {sink!r} sink takes {spelled(combination)} separately and refuses them together, '
        f'which is a limit of that solver rather than of the model. {_instead(takers)}'
    )


def _sink_refuses_message(sink: str, missing: Sequence[str], takers: Sequence[str]) -> str:
    """A sink asked for a capability it does not have at all."""
    way_out = (
        ' Or write the sets out: mathspec.to_spec(...).expand() states each as binaries and linking '
        'rows, which every sink takes.'
        if 'sos' in missing
        else ''
    )
    return (
        f'the {sink!r} sink cannot take {spelled(missing)}: it has no such concept, so there is '
        f'nothing to hand the model to. {_instead(takers)}{way_out}'
    )
