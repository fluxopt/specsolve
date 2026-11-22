"""What the data decides about a program before an entry is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once. What it
settles there, before any query of an entry's own, is answered here, and
[`Assembly`][specsolve._engine.assembly.Assembly] builds nothing the data has
already ruled out. Every such decision lives in this module, so a build's
shortcuts are one list rather than a check in each place that takes one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathspec import program

    from specsolve._engine.attaching import AttachedSources


def rowless(entry: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *entry* holds no row whatever its terms: it spans a dimension without a label."""
    return any(not attached.cardinality[d] for d in entry.dims)
