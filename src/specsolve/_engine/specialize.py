"""What the data decides about a program before a declaration is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once and, in the
same pass, every test a ``where`` or a ``cases:`` region makes ([`settle`][])
and every price in the objective that is zero ([`zero_parameters`][]). What
that read settles is answered here: the program a build compiles is
[`specialized`][] to it, and
[`Assembly`][specsolve._engine.assembly.Assembly] builds nothing the data has
already ruled out ([`rowless`][]). Every such decision lives in this module, so
a build's shortcuts are one list rather than a check in each place that takes
one.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, assert_never

import polars as pl
from mathspec import program
from mathspec.program import walk

from specsolve._collect import collected_all, in_memory
from specsolve._engine.predicates import counted_over_nothing, passing
from specsolve._engine.scope import Scope
from specsolve._names import VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from specsolve._engine.attaching import AttachedSources


def settle(lowered: program.Program, attached: AttachedSources) -> dict[program.Condition, bool]:
    """Each test *lowered*'s conditions make that *attached* decides at every coordinate.

    The conditions are the ``where`` of every variable, constraint and
    assumption, and of every ``cases:`` region the rows and bounds read.

    A test is a parameter's bare name or comparison, anywhere in a mask, a
    named condition read through: ``True`` where it holds everywhere,
    ``False`` where nowhere. A test the data leaves partial is not in the
    answer. *attached* need not hold the answer yet; this is what computes it.

    Each test is read once over its parameter's own table, all in one batch
    and in memory: a model gates hundreds of declarations on the same few
    flags, and reading a flag per declaration over the product of its dims
    was most of the cost of the declarations that build nothing. A parameter
    with no row, and a count along a dimension without a label, settle
    without a query.
    """
    declared = [*lowered.variables.values(), *lowered.constraints.values(), *lowered.assumptions.values()]
    bounds = [b for v in lowered.variables.values() for b in (v.lower, v.upper) if b is not None]
    regions = [
        r.when for node in walk(*lowered.roots, *bounds) if isinstance(node, program.Cases) for r in node.regions
    ]
    masks = [*(d.where for d in declared if d.where is not None), *regions]
    scope = Scope(lowered, attached, {})
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
                out[atom] = False
    with in_memory():
        counts = collected_all(passing(scope, atom) for atom in queried)
    for atom, passed in zip(queried, counts, strict=True):
        count = passed.item()
        if not count:
            out[atom] = False
        elif count == math.prod(scope.data.cardinality[d] for d in atom.dims):
            out[atom] = True
    return out


def zero_parameters(lowered: program.Program, attached: AttachedSources) -> frozenset[str]:
    """The parameters the objective multiplies a variable by that are zero at every row they have.

    Asked of the objective alone, named expressions read through, since there
    a zero term adds nothing, where in a constraint it would still keep its
    row. Any value ``missing:`` fills counts. One batch, in memory.
    """
    if lowered.objective is None:
        return frozenset()
    names = sorted(
        {n for e in walk(lowered.objective.expression) if isinstance(e, program.Multiply) for n in _prices(e)}
    )
    with in_memory():
        found = collected_all(attached.parameters[n].select((pl.col(VALUE) != 0).any()) for n in names)
    return frozenset(n for n, nonzero in zip(names, found, strict=True) if not nonzero.item())


def specialized(lowered: program.Program, attached: AttachedSources) -> program.Program:
    """*lowered* with each ``where`` and ``cases:`` region it builds rows from folded over the tests *attached* settled.

    A variable's or constraint's ``where`` that folds to true is dropped, so
    the declaration takes the path of one without a ``where``: dense labels,
    and terms placed on their rows without a join. One that folds to false is
    kept as that literal, for [`rowless`][] to skip. A region that folds to
    false is dropped from its ``cases:``, and one left alone holding true
    stands for its ``cases:``: a build places a term on its rows by the
    term's own dims, and a sum spreads a term along what it sums, so the
    region's dims are not needed to place it. The objective, which a build
    only sums, also loses each product of a variable and a parameter zero at
    every row it has; a constraint keeps it, and with it its row.

    The program a build keeps is this one, so readback reads the
    declarations it built. Assumptions keep their ``where`` as written, since
    a refusal quotes it; [`rules_out`][] answers for them. The ``expressions``
    stay as written too: a read takes its dims from its terms, and a region
    may be the only one over a dim.
    """
    tests, zero = attached.settled, attached.zero
    if not tests and not zero:
        return lowered

    def where(declaration: program.Variable | program.Constraint) -> program.Mask | None:
        if declaration.where is None:
            return None
        kept = _folded(declaration.where, tests)
        return None if _holds(kept, value=True) else kept

    def bound(b: program.Expression | None) -> program.Expression | None:
        return None if b is None else _regions_folded(b, tests)

    variables = {
        n: replace(v, where=where(v), lower=bound(v.lower), upper=bound(v.upper)) for n, v in lowered.variables.items()
    }
    constraints = {
        n: replace(c, where=where(c), lhs=_regions_folded(c.lhs, tests), rhs=_regions_folded(c.rhs, tests))
        for n, c in lowered.constraints.items()
    }
    o = lowered.objective
    objective = None if o is None else replace(o, expression=_priced(_regions_folded(o.expression, tests), zero))
    return replace(lowered, variables=variables, constraints=constraints, objective=objective)


def rules_out(where: program.Mask | None, attached: AttachedSources) -> bool:
    """Whether the data settled *where* false at every coordinate, so nothing it gates is read."""
    return where is not None and _holds(_folded(where, attached.settled), value=False)


def rowless(declaration: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *declaration* holds no row whatever its terms: it spans a dimension without a label, or its ``where`` is ruled out."""
    return any(not attached.cardinality[d] for d in declaration.dims) or rules_out(declaration.where, attached)


def _folded(mask: program.Mask, tests: Mapping[program.Condition, bool]) -> program.Mask:
    """*mask* with each settled test written as the literal it is; constructing the mask folds what is left."""
    return program.Mask(_substituted(mask.root, tests))


def _holds(mask: program.Mask, *, value: bool) -> bool:
    """Whether *mask* folded to the literal *value*."""
    return isinstance(mask.root, program.BooleanLiteral) and mask.root.value == value


def _substituted(node: program.Condition, tests: Mapping[program.Condition, bool]) -> program.Condition:
    """*node* rebuilt with each settled test as a literal, a named condition read as its body."""
    if node in tests:
        return program.BooleanLiteral(tests[node])
    if isinstance(node, program.And):
        return program.And(_substituted(node.left, tests), _substituted(node.right, tests))
    if isinstance(node, program.Or):
        return program.Or(_substituted(node.left, tests), _substituted(node.right, tests))
    if isinstance(node, program.Not):
        return program.Not(_substituted(node.operand, tests))
    if isinstance(node, program.ConditionReference):
        return _substituted(node.body, tests)
    return node


def _regions_folded(e: program.Expression, tests: Mapping[program.Condition, bool]) -> program.Expression:
    """*e* with each ``cases:`` region folded as [`specialized`][] says, a named expression rewritten where it is read."""
    if isinstance(e, program.Cases):
        regions = tuple(
            program.Region(when, _regions_folded(r.value, tests))
            for r in e.regions
            if not _holds(when := _folded(r.when, tests), value=False)
        )
        if len(regions) == 1 and _holds(regions[0].when, value=True):
            return regions[0].value
        return replace(e, regions=regions)
    return _rebuilt(e, lambda child: _regions_folded(child, tests))


def _rebuilt(e: program.Expression, f: Callable[[program.Expression], program.Expression]) -> program.Expression:
    """*e* with *f* applied to each sub-expression [`children`][mathspec.program.children] names, the rest kept."""
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


def _prices(e: program.Multiply) -> Iterator[str]:
    """The parameters *e* multiplies straight onto a factor that carries a variable."""
    for f, g in ((e.left, e.right), (e.right, e.left)):
        if isinstance(f, program.ParameterReference) and program.carries_variable(g):
            yield f.name


def _priced(e: program.Expression, zero: frozenset[str]) -> program.Expression:
    """An objective *e* less each product priced by a parameter in *zero*, the literal zero where nothing is left."""
    kept = _less_zero_prices(e, zero)
    return program.Constant(0.0, dims=()) if kept is None else kept


def _less_zero_prices(e: program.Expression, zero: frozenset[str]) -> program.Expression | None:
    """*e* less each product priced by a parameter in *zero*; ``None`` where nothing is left.

    Nothing times anything is nothing, and so is every node over it, but two:
    a sum of two keeps the other side, and a ``cases:`` its other regions.
    """
    if isinstance(e, program.Multiply) and not zero.isdisjoint(_prices(e)):
        return None
    if isinstance(e, program.Add):
        left, right = _less_zero_prices(e.left, zero), _less_zero_prices(e.right, zero)
        if left is None or right is None:
            return left if right is None else right
        return replace(e, left=left, right=right)
    if isinstance(e, program.Cases):
        regions = tuple(replace(r, value=v) for r in e.regions if (v := _less_zero_prices(r.value, zero)) is not None)
        return replace(e, regions=regions) if regions else None
    emptied = []

    def kept(child: program.Expression) -> program.Expression:
        out = _less_zero_prices(child, zero)
        emptied.append(out is None)
        return child if out is None else out

    rebuilt = _rebuilt(e, kept)
    return None if any(emptied) else rebuilt
