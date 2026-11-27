"""What the data decides about a program before an entry is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once and, in the
same pass, every atom of a ``where`` or a ``cases:`` region ([`settle`][]).
What that read settles is answered here: the program a build reads is
[`specialized`][] to it, and
[`Assembly`][specsolve._engine.assembly.Assembly] builds nothing the data has
already ruled out ([`is_empty`][]). Every such decision lives in this module, so
a build's shortcuts are one list rather than a check in each place that takes
one.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, assert_never

from mathspec import program
from mathspec.program import walk

from specsolve._collect import collected_all, in_memory
from specsolve._engine.conditions import counted_over_nothing, passes_without_a_row, passing
from specsolve._engine.scope import Scope

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from specsolve._attached import AttachedSources


def settle(compiled: program.Program, attached: AttachedSources) -> dict[program.Condition, bool]:
    """Each atom of *compiled*'s conditions that *attached* settles.

    The conditions are the ``where`` of every variable, constraint and
    assumption, and of every ``cases:`` region the rows and bounds read.

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
    bounds = [b for v in compiled.variables.values() for b in (v.lower, v.upper) if b is not None]
    regions = [
        r.when for node in walk(*compiled.roots, *bounds) if isinstance(node, program.Cases) for r in node.regions
    ]
    masks = [*(e.where for e in entries if e.where is not None), *regions]
    scope = Scope(compiled, attached, {})
    atoms = {atom for mask in masks for atom in mask.atoms}
    out: dict[program.Condition, bool] = {}
    queried: list[program.ParameterDefined | program.ParameterComparison] = []
    for atom in atoms:
        if isinstance(atom, program.CountComparison) and not scope.data.dimensions[atom.over].size:
            out[atom] = counted_over_nothing(atom)
        elif isinstance(atom, program.ParameterDefined | program.ParameterComparison):
            if scope.data.parameter_rows[atom.name]:
                queried.append(atom)
            else:
                out[atom] = passes_without_a_row(scope, atom)
    with in_memory():
        counts = collected_all(passing(scope, atom) for atom in queried)
    for atom, passed in zip(queried, counts, strict=True):
        everywhere = math.prod(scope.data.dimensions[d].size for d in atom.dims)
        count = passed.item()
        if passes_without_a_row(scope, atom):
            count += everywhere - scope.data.parameter_rows[atom.name]
        if not count:
            out[atom] = False
        elif count == everywhere:
            out[atom] = True
    return out


def specialized(compiled: program.Program, attached: AttachedSources) -> program.Program:
    """*compiled* with each ``where`` and ``cases:`` region it builds rows from specialized to the atoms *attached* settled.

    A variable's or constraint's ``where`` the data settles true is dropped, so
    the entry takes the path of one without a ``where``: dense labels,
    and terms placed on their rows without a join. One it settles false is
    kept as that literal, for [`is_empty`][] to skip. A region it settles
    false is dropped from its ``cases:``, and one left alone holding true
    stands for its ``cases:``: a build places a term on its rows by the
    term's own dimensions, and a sum spreads a term along what it sums, so
    the region's dimensions are not needed to place it.

    The program a build keeps is this one, so readback reads the
    entries it built. Assumptions keep their ``where`` as written, since
    a refusal quotes it; [`rules_out`][] answers for them. The ``expressions``
    stay as written too: a read takes its dimensions from its terms, and a
    region may be the only one over a dimension.
    """
    settled = attached.settled
    if not settled:
        return compiled

    def where(entry: program.Variable | program.Constraint) -> program.Mask | None:
        if entry.where is None:
            return None
        kept = _specialized_mask(entry.where, settled)
        return None if _holds(kept, value=True) else kept

    def bound(b: program.Expression | None) -> program.Expression | None:
        return None if b is None else _regions_specialized(b, settled)

    variables = {
        n: replace(v, where=where(v), lower=bound(v.lower), upper=bound(v.upper)) for n, v in compiled.variables.items()
    }
    constraints = {
        n: replace(
            c, where=where(c), lhs=_regions_specialized(c.lhs, settled), rhs=_regions_specialized(c.rhs, settled)
        )
        for n, c in compiled.constraints.items()
    }
    o = compiled.objective
    objective = None if o is None else replace(o, expression=_regions_specialized(o.expression, settled))
    return replace(compiled, variables=variables, constraints=constraints, objective=objective)


def rules_out(where: program.Mask | None, attached: AttachedSources) -> bool:
    """Whether the data settled *where* false at every coordinate, so nothing it gates is read."""
    return where is not None and _holds(_specialized_mask(where, attached.settled), value=False)


def is_empty(entry: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *entry* holds nothing whatever its terms, no column or no row: it spans a dimension without a label, or its ``where`` is ruled out."""
    return any(not attached.dimensions[d].size for d in entry.dims) or rules_out(entry.where, attached)


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


def _regions_specialized(e: program.Expression, settled: Mapping[program.Condition, bool]) -> program.Expression:
    """*e* with each ``cases:`` region specialized as [`specialized`][] says, a named expression rewritten where it is read."""
    if isinstance(e, program.Cases):
        regions = tuple(
            program.Region(when, _regions_specialized(r.value, settled))
            for r in e.regions
            if not _holds(when := _specialized_mask(r.when, settled), value=False)
        )
        if len(regions) == 1 and _holds(regions[0].when, value=True):
            return regions[0].value
        return replace(e, regions=regions)
    return _rebuilt(e, lambda child: _regions_specialized(child, settled))


def _rebuilt(e: program.Expression, f: Callable[[program.Expression], program.Expression]) -> program.Expression:
    """*e* with *f* applied to each sub-expression ``mathspec.program.children`` names, the rest kept."""
    if isinstance(e, program.Constant | program.ParameterReference | program.VariableReference | program.Dual):
        return e
    if isinstance(e, program.Add | program.Multiply):
        return replace(e, left=f(e.left), right=f(e.right))
    if isinstance(e, program.Divide):
        return replace(e, numerator=f(e.numerator), divisor=f(e.divisor))
    if isinstance(e, program.Power):
        return replace(e, base=f(e.base), exponent=f(e.exponent))
    if isinstance(e, program.Negate | program.Sum | program.Join | program.Translate | program.WindowSum):
        return replace(e, operand=f(e.operand))
    if isinstance(e, program.ExpressionReference):
        return replace(e, body=f(e.body))
    if isinstance(e, program.Cases):
        return replace(e, regions=tuple(replace(r, value=f(r.value)) for r in e.regions))
    assert_never(e)
