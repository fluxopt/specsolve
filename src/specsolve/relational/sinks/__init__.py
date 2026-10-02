"""Sinks: how a built model leaves the engine. See README.md.

A *solver* runs the handoff (``solvers/``, chosen by name); a *writer* renders
it to a file (``writers/``, chosen by suffix); an *export* hands it to a
modelling library (``pyomo.py``, chosen by name). Each reads ``handoff.py`` and
declares what it takes in ``capabilities.py``; none imports another.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from specsolve.errors import SpecsolveError, unknown_name_message
from specsolve.relational.sinks import capabilities as caps
from specsolve.relational.sinks.capabilities import spelled
from specsolve.relational.sinks.handoff import Handoff
from specsolve.relational.sinks.pyomo import PYOMO_CAPABILITIES
from specsolve.relational.sinks.solvers import SOLVERS, Solver, loaded, solver
from specsolve.relational.sinks.writers import WRITERS, writer

if TYPE_CHECKING:
    from collections.abc import Callable, Collection, Mapping, Sequence

__all__ = [
    'EXPORTS',
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


#: Every export a caller may name, with what it can ingest. Closed.
EXPORTS: Mapping[str, caps.Capabilities] = {'pyomo': PYOMO_CAPABILITIES}

#: Every sink's name: solvers, suffixes, then exports.
_SINKS = (*SOLVERS, *WRITERS, *EXPORTS)


def sink_capabilities(name: str) -> caps.Capabilities:
    """What the sink called *name* can ingest — a solver name, a suffix, or an export.

    Answered without importing the sink.

    Raises:
        SpecsolveError: A name belonging to no family.
    """
    if name in SOLVERS:
        return SOLVERS[name].capabilities
    if (suffix := name.lower()) in WRITERS:
        return WRITERS[suffix].capabilities
    if name in EXPORTS:
        return EXPORTS[name]
    raise SpecsolveError(unknown_name_message('sink', name, _SINKS))


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
    return refuses([other for other in _SINKS if other != name and _blocker(other, needed) is None])


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
