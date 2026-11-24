"""What a spec and its sources may arrive as, how a source is read as a table, and [`as_program`][], the one function every verb compiles a spec through.

A caller's table is read through the Arrow PyCapsule protocol, so neither
pyarrow nor pandas is a dependency. A ``pandas.Series`` is unwrapped first, and
only when pandas is already imported. An ``xarray.DataArray`` is not a table
and is not read.

An expression the file never named passes through [`as_program`][] too: [`expression_node`][] splices
it into the spec as a named one and compiles the whole spec again. This sits
above both lanes because nothing
in the engine or a sink may see the spec as written
(docs/development/architecture.md, hard rule 2).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import polars as pl
from mathspec import to_spec
from mathspec.program import Program

from specsolve import _parquet as parquet
from specsolve._names import VALUE, refuse_reserved, refuse_value
from specsolve.errors import LanguageError, SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping, Sequence
    from datetime import datetime

    import pandas as pd
    from mathspec.program import Expression
    from mathspec.spec import Spec

#: Anything a verb takes as the spec: a YAML path, a mapping, or a ``Spec``
#: with its ``piecewise:`` blocks written out. Not a ``Program``: compiling has
#: no inverse.
type SpecLike = str | Path | Mapping[str, object] | Spec


def as_spec(spec: SpecLike) -> Spec:
    """*spec* as the spec it is, whatever shape it arrived in.

    Raises:
        SpecsolveError: A compiled ``Program``.
        LanguageError: Anything the language does not accept.
    """
    if isinstance(spec, Program):
        raise SpecsolveError(
            'a compiled Program is not a spec this takes. Compiling has no inverse, so an answer from one '
            'could not say which document it came back from, and nothing built from one could be archived. '
            'Pass what it was compiled from — a path, a mapping, or mathspec.to_spec() of either, which is '
            'the form worth keeping, since it carries its program. '
            'sps.check() still hands back the Program, for reading.'
        )
    return to_spec(spec)


@runtime_checkable
class ArrowTable(Protocol):
    """Any table that exposes the Arrow PyCapsule stream — pyarrow, DuckDB, ibis — read without importing it."""

    def __arrow_c_stream__(self, requested_schema: object = None) -> object: ...


#: A pandas Series of any dtype a source may carry. Spelled out because pandas
#: types ``Series`` invariantly.
type PandasSeries = pd.Series[float] | pd.Series[int] | pd.Series[bool] | pd.Series[str] | pd.Series[datetime]

#: A label along a dimension: the Python type of each dtype a dimension may
#: declare (`mathspec.program.DimensionDtype`).
type Label = int | float | str | datetime

#: Anything a verb takes under one name of ``sources``. A parameter: a parquet
#: path, a table — polars, pandas, or any [`ArrowTable`][] — or one of the
#: plain-Python shapes a hand-written model reaches for, a ``{label: value}``
#: map, a sequence in the dimension's own label order, and one number for
#: every coordinate. A dimension's labels: a table carrying a column named
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


def to_pandas(table: pl.DataFrame) -> pd.DataFrame:
    """A polars table as pandas, column by column, without pyarrow."""
    import pandas as pd

    encoded = [name for name, kind in table.schema.items() if kind in (pl.Categorical, pl.Enum)]
    if encoded:
        table = table.with_columns(pl.col(name).cast(pl.String) for name in encoded)
    return pd.DataFrame({name: table[name].to_numpy() for name in table.columns})


def as_table(obj: Source, dims: Sequence[str] = ()) -> pl.LazyFrame | None:
    """One source as a lazy table: a parquet path scanned, or an in-memory table normalised.

    The one place a string is read as a parquet path. *dims* names the columns
    a pandas index becomes.

    Returns:
        The table, or ``None`` where *obj* is not table-shaped.
    """
    import sys

    if isinstance(obj, (str, Path)):
        parquet.require(f'a source given as the path {str(obj)!r}')
        return parquet.scan(obj)
    if isinstance(obj, pl.LazyFrame):
        return obj
    if isinstance(obj, pl.DataFrame):
        return obj.lazy()

    if 'pandas' in sys.modules:
        import pandas as pd

        if isinstance(obj, pd.Series):
            table = _series_to_table(obj, dims)
            return _from_pandas(table) if table is not None else None
        if isinstance(obj, pd.DataFrame):
            return _from_pandas(obj)

    if isinstance(obj, ArrowTable):
        try:
            return pl.DataFrame(obj).lazy()
        except (TypeError, ValueError, pl.exceptions.PolarsError):
            return None
    return None


def is_dense_array(obj: object) -> bool:
    """Whether *obj* is an ``xarray.DataArray``, the one shape recognised and not read."""
    import sys

    xr = sys.modules.get('xarray')
    return xr is not None and isinstance(obj, xr.DataArray)


def is_multi_indexed(obj: Source) -> bool:
    """Whether *obj* is a pandas Series carrying more than one index level."""
    import sys

    if 'pandas' not in sys.modules:
        return False
    import pandas as pd

    return isinstance(obj, pd.Series) and obj.index.nlevels > 1


def _series_to_table(series: pd.Series, dims: Sequence[str]) -> pd.DataFrame | None:
    """A pandas Series with its one index level promoted to a column.

    A named level keeps its name: renaming it to *dims* would transpose the
    data when two dimensions share a label space.

    Returns:
        The tidy table, or ``None`` where the entry is not over one dimension.
    """
    if len(dims) != 1:
        return None
    if series.index.name is None:
        series = series.rename_axis(dims[0])
    return series.rename(VALUE).reset_index()


#: The numpy datetime units polars reads; a coarser one is cast to microseconds without loss.
_POLARS_DATETIMES = frozenset({'datetime64[ms]', 'datetime64[us]', 'datetime64[ns]'})


def _from_pandas(table: pd.DataFrame) -> pl.LazyFrame:
    """A pandas table, column by column, without pyarrow.

    Object arrays go through a list so numpy's ``nan`` becomes a null rather
    than a string. A datetime in a unit polars does not read, such as the
    seconds ``pd.to_datetime`` gives a date, is cast to microseconds first.
    """
    columns: dict[str, object] = {}
    for name in table.columns:
        values = table[name].to_numpy()
        if values.dtype == object:
            columns[name] = pl.Series(name, [None if _is_missing(v) else v for v in values], strict=False)
        elif values.dtype.kind == 'M' and values.dtype.name not in _POLARS_DATETIMES:
            columns[name] = values.astype('datetime64[us]')
        else:
            columns[name] = values
    return pl.DataFrame(columns).lazy()


def _is_missing(value: object) -> bool:
    """Whether an object-array element is pandas' rendering of "no value"."""
    return value is None or (isinstance(value, float) and value != value)


def _case_collision(program: Program) -> str | None:
    """Two entries of one namespace whose names differ only by case, as the sentence refusing them."""
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
                    f'on disk cannot hold both: an entry is written as a file named after it, and a '
                    f'case-insensitive filesystem — a stock macOS volume, a stock Windows one — folds the '
                    f'two into one, so the second overwrites the first and one name comes back carrying '
                    f"the other's values. Tell them apart by a suffix rather than a capital: 'p_rated' "
                    f"beside 'p'."
                )
            seen[name.casefold()] = (kind, name)
    return None


def _refuse_reserved_names(program: Program) -> None:
    """Refuse an entry whose name starts with ``specsolve_``, and a dimension called ``value``, in any letter case."""
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
        refuse_reserved(name, which)
    for name in program.dimensions:
        refuse_value(name, f"dimension '{name}'")


def as_program(spec: SpecLike) -> Program:
    """*spec* as a program, refusing what this package cannot build or keep apart.

    Every verb compiles through here.

    Raises:
        LanguageError: A construct outside the language, a
            ``piecewise:`` block still to be written out, or a fragment that
            reads a name under ``given:``.
        SpecsolveError: Two entries of one namespace whose names differ only
            by case, a name that starts with ``specsolve_``, or a dimension
            called ``value``, in any letter case.
    """
    program = as_spec(spec).program
    if program.given:
        given = program.given
        read = sorted({*given.parameters, *given.variables, *given.constraints, *given.expressions, *given.conditions})
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
    _refuse_reserved_names(program)
    return program


#: The name an unnamed expression is spliced under, stepped over where the
#: spec declares it.
_EVALUATED = '_evaluated'


def expression_node(spec: Spec, expression: str | Mapping[str, object]) -> Expression:
    """One unnamed expression as a program node, read in *spec*'s namespace.

    The expression is spliced under ``expressions:`` with a name no section
    of *spec* declares, so a section added later is covered too.

    Args:
        spec: The spec whose names the expression may use.
        expression: What one ``expressions:`` entry takes — a string, or the
            mapping carrying ``cases:`` with ``dims:`` and ``otherwise:``.

    Returns:
        The node a declared named expression of *spec* compiles to.

    Raises:
        LanguageError: A construct outside the language, or a name *spec* does
            not declare.
        SchemaError: Something that is not an expression.
    """
    written = spec.to_dict()
    taken = {name for section in written.values() if isinstance(section, dict) for name in section}
    name = _EVALUATED
    while name in taken:
        name += '_'
    section = written.get('expressions')
    written['expressions'] = {**section, name: expression} if isinstance(section, dict) else {name: expression}
    return as_program(written).expressions[name].expression
