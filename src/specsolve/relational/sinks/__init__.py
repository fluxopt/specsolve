"""Sinks: how a built model leaves the engine. See README.md.

A *solver* runs the handoff (``solvers/``, chosen by name); a *writer* renders
it to a file (``writers/``, chosen by suffix). Both read ``handoff.py`` and
declare what they take in ``capabilities.py``; neither imports the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from specsolve.errors import SpecsolveError, unknown_name_message
from specsolve.relational.sinks import capabilities as caps
from specsolve.relational.sinks.capabilities import spelled
from specsolve.relational.sinks.handoff import Handoff
from specsolve.relational.sinks.solvers import SOLVERS, Solver, loaded, solver
from specsolve.relational.sinks.writers import WRITERS, writer

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Sequence

    from mathspec import program

__all__ = [
    'SOLVERS',
    'WRITERS',
    'Handoff',
    'Solver',
    'loaded',
    'refusal',
    'sink_capabilities',
    'solver',
    'writer',
]


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


def refusal(program: program.Program, name: str) -> str | None:
    """The sink called *name*'s refusal of *program*, or ``None`` where it takes it.

    The refusal names the construct, the sink, and the sinks that do take it.
    """
    needed = caps.required(program)
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
