"""The writer family: the handoff in, a file out. See ../README.md.

One module per format, chosen by the output's suffix. Each answers
``(handoff, path, names) -> None``, and streams.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from specsolve.errors import SpecsolveError
from specsolve.relational.sinks.writers.lp_file import LP_FILE_CAPABILITIES, write_lp_file
from specsolve.relational.sinks.writers.mps_file import MPS_FILE_CAPABILITIES, write_mps_file

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from specsolve.relational.sinks.capabilities import Capabilities
    from specsolve.relational.sinks.handoff import Handoff
    from specsolve.relational.sinks.writers.base import Names

    Write = Callable[[Handoff, Path, Names], None]

__all__ = ['WRITERS', 'Writer', 'writer']


@dataclass(frozen=True)
class Writer:
    """One format: how to render it, and what it can carry."""

    write: Write
    capabilities: Capabilities


#: What can be written today, by suffix. Closed.
WRITERS: Mapping[str, Writer] = {
    '.lp': Writer(write_lp_file, LP_FILE_CAPABILITIES),
    '.mps': Writer(write_mps_file, MPS_FILE_CAPABILITIES),
}


def writer(suffix: str) -> Writer:
    """The writer for *suffix*.

    Raises:
        SpecsolveError: A format nothing writes; the message lists what can be.
    """
    if suffix in WRITERS:
        return WRITERS[suffix]
    raise SpecsolveError(f'unknown output format {suffix!r} — this build writes {", ".join(sorted(WRITERS))}.')
