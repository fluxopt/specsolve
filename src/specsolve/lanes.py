"""What a spec and its sources may arrive as, and the one door every verb lowers a spec through."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from mathspec import to_spec
from mathspec.program import Program

from specsolve.errors import LanguageError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    import polars as pl
    from mathspec import Spec

#: Anything a verb takes as the spec: a YAML path, a mapping, or a ``Spec``
#: with its ``piecewise:`` blocks written out. Not a ``Program``: lowering has
#: no inverse.
type Buildable = str | Path | Mapping[str, object] | Spec


def declared(spec: Buildable) -> Spec:
    """*spec* as the document it is, whatever shape it arrived in.

    Raises:
        SpecsolveError: A lowered ``Program``.
        LanguageError: Anything the language does not accept.
    """
    if isinstance(spec, Program):
        raise SpecsolveError(
            'a lowered Program is not a spec this takes. Lowering has no inverse, so an answer from one '
            'could not say which document it came back from, and nothing built from one could be archived. '
            'Pass what it was lowered from — a path, a mapping, or mathspec.to_spec() of either, which is '
            'the form worth keeping, since it carries its program. '
            'sps.check() still hands back the Program, for reading the plan.'
        )
    return to_spec(spec)


@runtime_checkable
class ArrowTable(Protocol):
    """Any table that exposes the Arrow PyCapsule stream — pyarrow, DuckDB, ibis — read without importing it."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


#: A pandas Series of any dtype a source may carry. Spelled out because pandas
#: types ``Series`` invariantly.
type PandasSeries = pd.Series[float] | pd.Series[int] | pd.Series[bool] | pd.Series[str] | pd.Series[datetime]

#: A label along a dimension: the Python type of each dtype an index may
#: declare (`mathspec.program.DimensionDtype`).
type Label = int | float | str | datetime

#: Anything a verb takes under one name of ``sources``. A parameter: a parquet
#: path, a table — polars, pandas, or any [`ArrowTable`][] — or one of the
#: plain-Python shapes a hand-written model reaches for, a ``{label: value}``
#: map, a sequence in the dimension's own label order, and one number for
#: every coordinate. A dimension's index: a table carrying a column named
#: after it, or a bare sequence of its labels. A relation: the table of the
#: rows it holds, one column per column it declares.
type Source = (
    str
    | Path
    | pl.DataFrame
    | pl.LazyFrame
    | pd.DataFrame
    | PandasSeries
    | ArrowTable
    | Mapping[Label, float]
    | Collection[Label]
    | float
)


def _case_collision(program: Program) -> str | None:
    """Two declarations of one namespace whose names differ only by case, as the sentence refusing them."""
    flat = (
        *(('dimension', name) for name in program.dimensions),
        *(('relation', name) for name in program.relations),
        *(('parameter', name) for name in program.parameters),
        *(('variable', name) for name in program.variables),
        *(('named expression', name) for name in program.expressions),
    )
    for namespace in (flat, tuple(('constraint', name) for name in program.constraints)):
        seen: dict[str, tuple[str, str]] = {}
        for kind, name in namespace:
            if (earlier := seen.get(name.casefold())) is not None:
                return (
                    f"{kind} '{name}' and {earlier[0]} '{earlier[1]}' differ only by case, and one answer "
                    f'on disk cannot hold both: a declaration is written as a file named after it, and a '
                    f'case-insensitive filesystem — a stock macOS volume, a stock Windows one — folds the '
                    f'two into one, so the second overwrites the first and one name comes back carrying '
                    f"the other's values. Tell them apart by a suffix rather than a capital: 'p_rated' "
                    f"beside 'p'."
                )
            seen[name.casefold()] = (kind, name)
    return None


#: The prefix reserved, in any letter case, for the columns specsolve adds, so
#: that no name a spec declares can collide with one.
RESERVED = 'specsolve_'


def _reserved_name(program: Program) -> str | None:
    """The first declaration whose name starts with ``specsolve_`` in any letter case, as the sentence refusing it."""
    named = (
        *((f"dimension '{name}'", name) for name in program.dimensions),
        *((f"relation '{name}'", name) for name in program.relations),
        *(
            (f"column '{role}' of relation '{relation}'", role)
            for relation, held in program.relations.items()
            for role, _ in held.columns
        ),
        *((f"parameter '{name}'", name) for name in program.parameters),
        *((f"variable '{name}'", name) for name in program.variables),
        *((f"constraint '{name}'", name) for name in program.constraints),
        *((f"named expression '{name}'", name) for name in program.expressions),
        *((f"sos set '{name}'", name) for name in program.sos),
        *((f"assumption '{name}'", name) for name in program.assumptions),
    )
    for which, name in named:
        if name.casefold().startswith(RESERVED):
            return (
                f'{which} starts with {RESERVED!r}, which is reserved in any letter case for the columns '
                f'specsolve adds, so a declared name cannot collide with one. Rename it.'
            )
    return None


def lowered(spec: Buildable) -> Program:
    """*spec* as a program, refusing what this package cannot build or keep apart.

    Every verb lowers through here. A ``piecewise:`` block is refused, not
    expanded.

    Raises:
        LanguageError: A construct outside the streaming language, a
            ``piecewise:`` block still to be written out, or a fragment that
            reads a name under ``given:``.
        SpecsolveError: Two declarations of one namespace whose names differ only
            by case, or a name that starts with ``specsolve_`` in any letter case.
    """
    program = declared(spec).program
    if program.given:
        given = program.given
        read = sorted({*given.parameters, *given.variables, *given.constraints, *given.expressions})
        raise LanguageError(
            f'this spec reads {", ".join(repr(name) for name in read)} under given:, which another file declares, '
            f'so it is a fragment of a model rather than the whole of one. Compose it with the files that declare '
            f'them first: mathspec.merge([...]) folds each given name into the file that introduces it.'
        )
    if program.piecewise:
        named = ', '.join(f"'{name}'" for name in program.piecewise)
        raise LanguageError(
            f'piecewise: {named} is still a curve, and specsolve builds only the rows a curve is expanded into. '
            f"Expand it first: to_spec(spec).expand('piecewise') keeps every sos: block for a sink that "
            f'takes a set, and to_spec(spec).expand() expands the sets into binaries too.'
        )
    if (refused := _case_collision(program)) is not None:
        raise SpecsolveError(refused)
    if (reserved := _reserved_name(program)) is not None:
        raise SpecsolveError(reserved)
    return program
