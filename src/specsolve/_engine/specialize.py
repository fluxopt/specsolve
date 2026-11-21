"""What the data decides about a program before a declaration is built.

[`attach`][specsolve._engine.attaching.attach] reads the data once and, in the
same pass, every test a ``where`` or a ``cases:`` region makes ([`settle`][]).
What that read settles is answered here: the program a build compiles is
[`specialized`][] to it, and [`Assembly`][specsolve._engine.assembly.Assembly]
builds nothing the data has already ruled out ([`rowless`][]). Two decisions
are taken in the compiler, over facts read here: a region the data settles,
folded in a build alone ([`Scope.settled`][specsolve._engine.scope.Scope.settled]),
and a term of the objective whose coefficient is zero everywhere
([`zero_parameters`][]).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program
from mathspec.program import walk

from specsolve._collect import collected_all, in_memory
from specsolve._engine.predicates import folded, holds_everywhere, holds_nowhere, settled
from specsolve._engine.scope import Scope
from specsolve._names import VALUE

if TYPE_CHECKING:
    from collections.abc import Mapping

    from specsolve._engine.attaching import AttachedSources


def settle(lowered: program.Program, attached: AttachedSources) -> dict[program.Condition, bool]:
    """The tests *lowered*'s conditions make, settled over *attached*.

    The conditions are the ``where`` of every variable, constraint and
    assumption, and of every ``cases:`` region the rows and bounds read. One
    batch for the whole program: a model gates hundreds of declarations on
    the same few flags. In memory, since each test reads one parameter's
    table whole. *attached* need not hold the answer yet; this is what
    computes it.
    """
    declared = [*lowered.variables.values(), *lowered.constraints.values(), *lowered.assumptions.values()]
    bounds = [b for v in lowered.variables.values() for b in (v.lower, v.upper) if b is not None]
    regions = [
        r.when for node in walk(*lowered.roots, *bounds) if isinstance(node, program.Cases) for r in node.regions
    ]
    with in_memory():
        return settled(Scope(lowered, attached, {}), [*(d.where for d in declared if d.where is not None), *regions])


def zero_parameters(lowered: program.Program, attached: AttachedSources) -> frozenset[str]:
    """The parameters the objective multiplies a variable by that are zero at every row they have.

    Asked of the objective alone, named expressions read through, since there
    a zero term adds nothing, where in a constraint it would still keep its
    row. Any value ``missing:`` fills counts. One batch, in memory.
    """
    if lowered.objective is None:
        return frozenset()
    names = sorted(
        {
            f.name
            for e in walk(lowered.objective.expression)
            if isinstance(e, program.Multiply)
            for f, g in ((e.left, e.right), (e.right, e.left))
            if isinstance(f, program.ParameterReference) and program.carries_variable(g)
        }
    )
    with in_memory():
        found = collected_all(attached.parameters[n].select((pl.col(VALUE) != 0).any()) for n in names)
    return frozenset(n for n, nonzero in zip(names, found, strict=True) if not nonzero.item())


def specialized(lowered: program.Program, attached: AttachedSources) -> program.Program:
    """*lowered* with each variable's and constraint's ``where`` folded over the tests *attached* settled.

    A ``where`` that folds to true is dropped, so the declaration takes the
    path of one without a ``where``: dense labels, and terms placed on their
    rows without a join. One that folds to false is kept as that literal, for
    [`rowless`][] to skip. The program a build keeps is this one, so readback
    reads the declarations it built. Assumptions keep their ``where`` as
    written, since a refusal quotes it; [`rules_out`][] answers for them.
    """
    tests = attached.settled

    def narrowed[D: (program.Variable, program.Constraint)](declarations: Mapping[str, D]) -> dict[str, D]:
        out = {}
        for name, declaration in declarations.items():
            where = declaration.where
            if where is None:
                out[name] = declaration
                continue
            kept = folded(where, tests)
            out[name] = replace(declaration, where=None if holds_everywhere(kept) else kept)
        return out

    if not tests:
        return lowered
    return replace(lowered, variables=narrowed(lowered.variables), constraints=narrowed(lowered.constraints))


def rules_out(where: program.Mask | None, attached: AttachedSources) -> bool:
    """Whether the data settled *where* false at every coordinate, so nothing it gates is read."""
    return where is not None and holds_nowhere(folded(where, attached.settled))


def rowless(declaration: program.Variable | program.Constraint, attached: AttachedSources) -> bool:
    """Whether *declaration* holds no row whatever its terms: it spans a dimension without a label, or its ``where`` is ruled out."""
    return any(not attached.cardinality[d] for d in declaration.dims) or rules_out(declaration.where, attached)
