"""What the data decides about a program before a declaration is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once. What it
settles there, before any query of a declaration's own, is answered here, and
[`Assembly`][specsolve._engine.assembly.Assembly] builds nothing the data has
already ruled out. Every such decision lives in this module, so a build's
shortcuts are one list rather than a check in each place that takes one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mathspec import program

    from specsolve._engine.attaching import AttachedSources


def rowless(declaration: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *declaration* holds no row whatever its terms: it spans a dimension without a label."""
    return any(not attached.cardinality[d] for d in declaration.dims)
