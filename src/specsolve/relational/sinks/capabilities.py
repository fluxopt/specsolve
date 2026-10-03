"""What a sink can ingest, separate from the solver-independent ceiling.

One descriptor per sink, so a construct the language says and a sink cannot
take is a refusal naming both rather than a ``kError`` from inside a library.

- A sink takes a construct or it does not; nothing is rewritten at the
  hand-off.
- An exclusion is a pair a sink takes separately and refuses together.
- What a model needs is read off the hand-off, so a square the data prices
  at zero, an integer variable no column is built for or a set with no
  members asks for nothing. ``nonconvex_quadratic_objective`` is decided by
  the coefficients' signs as well, so [`required`][] never asks for it; a
  descriptor still records it, the capability probes check it, and the sink
  that meets one at solve time sends the caller to the sinks that list it.
- A descriptor describes the sink as shipped, not the library it wraps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, get_args

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

    from specsolve.relational.sinks.handoff import Handoff

#: What a model may need a sink to have.
Capability = Literal[
    'integrality',
    'sos',
    'quadratic_objective',
    'nonconvex_quadratic_objective',
    'quadratic_constraint',
]

ALL_CAPABILITIES: tuple[Capability, ...] = get_args(Capability)


@dataclass(frozen=True)
class Capabilities:
    """One sink's answer for every capability, and the pairs it refuses.

    Attributes:
        supports: The capabilities the sink takes; one left out it cannot, so
            a descriptor lists only what it *can* do.
        excludes: Sets of capabilities it has individually and refuses together.
    """

    supports: frozenset[Capability]
    excludes: tuple[frozenset[Capability], ...] = ()

    def missing(self, required: Collection[Capability]) -> list[Capability]:
        """Those of *required* this sink cannot take at all, in [`ALL_CAPABILITIES`][] order."""
        return [c for c in ALL_CAPABILITIES if c in required and c not in self.supports]

    def excluded(self, required: Collection[Capability]) -> frozenset[Capability] | None:
        """The first conjunction *required* contains that this sink refuses.

        Returns:
            The excluded set, or ``None``.
        """
        for combination in self.excludes:
            if combination <= set(required):
                return combination
        return None


def required(handoff: Handoff, /) -> frozenset[Capability]:
    """What the built model in *handoff* needs a sink to have.

    Read off what was built rather than what the file declares: the build
    drops a zero coefficient, so a square the data never prices is not a
    quadratic term, and a declared integer variable with no column built asks
    for no integrality. Convexity never appears.
    """
    needed: set[Capability] = set()
    if (handoff.cols['vtype'] != 'continuous').any():
        needed.add('integrality')
    if handoff.quad.height:
        needed.add('quadratic_objective')
    if handoff.qmatrix.height:
        needed.add('quadratic_constraint')
    if handoff.sos.height:
        needed.add('sos')
    return frozenset(needed)


#: How a capability reads in a sentence.
_SPELLED: Mapping[str, str] = {
    'integrality': 'binary or integer variables',
    'sos': 'special-ordered sets (`sos:`)',
    'quadratic_objective': 'a quadratic objective',
    'nonconvex_quadratic_objective': 'a nonconvex quadratic objective',
    'quadratic_constraint': 'a quadratic constraint',
}


def spelled(capabilities: Collection[str]) -> str:
    """Capabilities as a refusal names them."""
    return ', '.join(_SPELLED[c] for c in capabilities)
