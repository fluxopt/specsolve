"""What the data decides about a program before an entry is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once and, in the
same pass, every atom of a ``where`` ([`settle`][]). What that read settles
is answered here: the program a build reads is [`specialized`][] to it, and
[`Assembly`][specsolve._engine.assembly.Assembly] builds nothing the data has
already ruled out ([`rowless`][]). Every such decision lives in this module, so
a build's shortcuts are one list rather than a check in each place that takes
one.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING

from mathspec import program

from specsolve._collect import collected_all, in_memory
from specsolve._engine.conditions import counted_over_nothing, passes_without_a_row, passing
from specsolve._engine.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Mapping

    from specsolve._engine.attaching import AttachedSources


def settle(compiled: program.Program, attached: AttachedSources) -> dict[program.Condition, bool]:
    """Each atom of the ``where`` of a variable, constraint or assumption that *attached* settles.

    An atom is read anywhere in a ``where``, a named condition read through:
    ``True`` where it holds everywhere, ``False`` where nowhere. An atom the
    data leaves partial is not in the answer. *attached* need not hold the
    answer yet; this is what computes it.

    Each atom is read once over its parameter's own table, all in one batch
    and in memory: a model gates hundreds of entries on the same few
    flags, and reading a flag per entry over the product of its dimensions
    was most of the cost of the entries that build nothing. A coordinate
    without a row of the parameter reads its missing value, which passes
    the atom or fails it everywhere at once. A parameter with no row, and a
    count along a dimension without a label, settle without a query.
    """
    entries = [*compiled.variables.values(), *compiled.constraints.values(), *compiled.assumptions.values()]
    masks = [e.where for e in entries if e.where is not None]
    scope = Scope(compiled, attached, {})
    atoms = {atom for mask in masks for atom in mask.atoms}
    out: dict[program.Condition, bool] = {}
    queried: list[program.ParameterDefined | program.ParameterComparison] = []
    for atom in atoms:
        if isinstance(atom, program.CountComparison) and not scope.data.cardinality[atom.over]:
            out[atom] = counted_over_nothing(atom)
        elif isinstance(atom, program.ParameterDefined | program.ParameterComparison):
            if scope.data.parameter_rows[atom.name]:
                queried.append(atom)
            else:
                out[atom] = passes_without_a_row(scope, atom)
    with in_memory():
        counts = collected_all(passing(scope, atom) for atom in queried)
    for atom, passed in zip(queried, counts, strict=True):
        everywhere = math.prod(scope.data.cardinality[d] for d in atom.dims)
        count = passed.item()
        if passes_without_a_row(scope, atom):
            count += everywhere - scope.data.parameter_rows[atom.name]
        if not count:
            out[atom] = False
        elif count == everywhere:
            out[atom] = True
    return out


def specialized(compiled: program.Program, attached: AttachedSources) -> program.Program:
    """*compiled* with each variable's and constraint's ``where`` specialized to the atoms *attached* settled.

    A ``where`` the data settles true is dropped, so the entry takes the
    path of one without a ``where``: dense labels, and terms placed on their
    rows without a join. One it settles false is kept as that literal, for
    [`rowless`][] to skip. The program a build keeps is this one, so readback
    reads the entries it built. Assumptions keep their ``where`` as
    written, since a refusal quotes it; [`rules_out`][] answers for them.
    """
    settled = attached.settled

    def specialized_entries[D: (program.Variable, program.Constraint)](entries: Mapping[str, D]) -> dict[str, D]:
        out = {}
        for name, entry in entries.items():
            where = entry.where
            if where is None:
                out[name] = entry
                continue
            kept = _specialized_mask(where, settled)
            out[name] = replace(entry, where=None if _holds(kept, value=True) else kept)
        return out

    if not settled:
        return compiled
    return replace(
        compiled,
        variables=specialized_entries(compiled.variables),
        constraints=specialized_entries(compiled.constraints),
    )


def rules_out(where: program.Mask | None, attached: AttachedSources) -> bool:
    """Whether the data settled *where* false at every coordinate, so nothing it gates is read."""
    return where is not None and _holds(_specialized_mask(where, attached.settled), value=False)


def rowless(entry: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *entry* holds no row whatever its terms: it spans a dimension without a label, or its ``where`` is ruled out."""
    return any(not attached.cardinality[d] for d in entry.dims) or rules_out(entry.where, attached)


def _specialized_mask(mask: program.Mask, settled: Mapping[program.Condition, bool]) -> program.Mask:
    """*mask* with each settled atom written as the literal it is; constructing the mask simplifies what is left."""
    return program.Mask(_substituted(mask.root, settled))


def _holds(mask: program.Mask, *, value: bool) -> bool:
    """Whether *mask* is the literal *value*."""
    return isinstance(mask.root, program.BooleanLiteral) and mask.root.value == value


def _substituted(node: program.Condition, settled: Mapping[program.Condition, bool]) -> program.Condition:
    """*node* rebuilt with each settled atom as a literal, a named condition read as its body."""
    if node in settled:
        return program.BooleanLiteral(settled[node])
    if isinstance(node, program.And):
        return program.And(_substituted(node.left, settled), _substituted(node.right, settled))
    if isinstance(node, program.Or):
        return program.Or(_substituted(node.left, settled), _substituted(node.right, settled))
    if isinstance(node, program.Not):
        return program.Not(_substituted(node.operand, settled))
    if isinstance(node, program.ConditionReference):
        return _substituted(node.body, settled)
    return node
