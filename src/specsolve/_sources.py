"""Attach runtime data to a compiled program: [`tidy_sources`][], the one function both lanes read data through.

Parquet paths, any table exposing the Arrow PyCapsule protocol, or a
plain-Python shape become the tidy tables both lanes read by name. Every check
on whether that data is usable is made here, once.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import polars.selectors as cs
from mathspec.errors import did_you_mean
from mathspec.program import Parameter

from specsolve._assumptions import validate_assumptions
from specsolve._collect import collected, collected_all, in_memory
from specsolve._messages import coordinate_text, coordinates_text
from specsolve._names import POSITION, VALUE
from specsolve._tables import as_table, is_dense_array, is_multi_indexed
from specsolve.errors import DataError

if TYPE_CHECKING:
    from mathspec.program import Dimension, Program, Relation

    from specsolve._engine.attaching import AttachedSources
    from specsolve._inputs import Label, Source


def attachable(program: Program) -> dict[str, Parameter | Dimension | Relation]:
    """Every name data may be attached to — declared parameters, dimensions and relations, one flat namespace."""
    return {**program.parameters, **program.dimensions, **program.relations}


def refuse_unknown_sources(program: Program, data: Mapping[str, object]) -> None:
    """Refuse a key of *data* that names no parameter, dimension or relation *program* declares.

    Raises:
        DataError: A key naming nothing the spec declares.
    """
    known = attachable(program)
    if unknown := set(data) - set(known):
        raise DataError(_unknown_source_keys_message(unknown, known))


def tidy_sources(program: Program, data: Mapping[str, Source]) -> dict[str, pl.LazyFrame]:
    """Read the caller's ``sources`` into the tables both lanes build against.

    Every source comes back as an in-memory `polars.LazyFrame`: a parameter
    as tidy ``(dims…, value)``, a dimension's labels as one column
    under its own name in the order they arrived, a relation as one column
    per declared column. A datetime label is held in microseconds, and a date
    under a ``datetime`` dimension as the microsecond its day starts on.

    Raises:
        DataError: A key naming nothing the spec declares; a declared
            dimension, relation or parameter with no data; a source no reader
            accepts or short of the columns its entry needs; a dimension's
            labels holding one twice; a parameter with two rows for one
            coordinate, a label its dimension lacks, a null or NaN value, or a
            column of another type than it declares; a relation with a null, a
            row twice, or a label its column's dimension lacks; a datetime
            label finer than a microsecond, or in another time zone than its
            dimension's labels; or an
            ``assumptions:`` entry the data does not hold, a ``piecewise:``
            method's conditions on its breakpoints among them.
    """
    return tidied_and_attached(program, data)[0]


def tidied_and_attached(
    program: Program,
    data: Mapping[str, Source],
    previous: Mapping[str, pl.LazyFrame] | None = None,
    changed: Collection[str] = (),
) -> tuple[dict[str, pl.LazyFrame], AttachedSources | None]:
    """[`tidy_sources`][], and the attach its assumptions were read against, so a build attaches once.

    The attach is ``None`` where the assumptions attached nothing, and a build
    attaches itself. *previous* is what this returned before *data* replaced
    the sources named *changed*: a parameter that replacement leaves alone is
    taken from it, neither read nor checked again ([`_kept`][]). Dimensions,
    relations and the assumptions are read and checked every time.
    """
    refuse_unknown_sources(program, data)
    sources = _read(program, data, _kept(program, previous or {}, changed))
    return sources, validate_assumptions(program, sources)


def _kept(program: Program, previous: Mapping[str, pl.LazyFrame], changed: Collection[str]) -> dict[str, pl.LazyFrame]:
    """The parameters of *previous* that a change of the sources named *changed* leaves as the door checked them.

    A parameter's checks read its own source and the index of each of its
    dimensions, a positional source included, so it is kept only where none of
    them is in *changed*.
    """
    return {
        name: previous[name]
        for name, p in program.parameters.items()
        if name in previous and name not in changed and not set(p.dims) & set(changed)
    }


@in_memory()
def _read(program: Program, data: Mapping[str, Source], kept: Mapping[str, pl.LazyFrame]) -> dict[str, pl.LazyFrame]:
    """[`tidy_sources`][] short of the assumptions, on polars' in-memory engine: every read here holds one table whole.

    [`validate_assumptions`][] runs
    after it. It attaches in memory too, and checks each assumption over the
    dimensions on the default engine.
    """
    _check_relation_sources(program, data)
    data = _read_paths_at_once(program, {name: source for name, source in data.items() if name not in kept})
    sources: dict[str, pl.LazyFrame] = {}
    for dname, declared in program.dimensions.items():
        if dname in data:
            sources[dname] = _labels_table(data[dname], dname, declared.dtype)
    relations = {
        name: _read_relation(data[name], name, relation, sources) for name, relation in program.relations.items()
    }
    for dname in program.dimensions:
        if dname not in sources and (authors := [f'sources[{name!r}]' for name in _relations_over(program, dname)]):
            raise DataError(_relation_needs_labels_message(dname, authors))
    _check_relations_hold_labels(program, relations, sources)
    sources |= relations

    read = {name: p for name, p in program.parameters.items() if name not in kept}
    for pname, pdef in read.items():
        if pname not in data:
            raise DataError(_no_parameter_source_message(pname))
        sources[pname] = parameter_table(pname, pdef, data[pname], sources)
    for pname, pdef in read.items():
        sources[pname] = _checked_parameter(pname, pdef, sources[pname], sources)
    sources |= kept

    for dname in program.dimensions:
        if dname not in sources:
            raise DataError(_no_labels_source_message(dname))

    return sources


def _read_paths_at_once(program: Program, data: Mapping[str, Source]) -> dict[str, Source]:
    """*data* with every parquet path read into memory, the files in one concurrent pass.

    Reading a file costs a fixed time however few rows it holds, and files read
    one after the other add those times up. Each file is read for the columns
    its entry reads. A file short of one stays a path, so the reader that
    names the missing column refuses it.
    """
    wanted = (
        {name: [name] for name in program.dimensions}
        | {name: list(relation.roles) for name, relation in program.relations.items()}
        | {name: [*p.dims, VALUE] for name, p in program.parameters.items()}
    )
    scans: dict[str, pl.LazyFrame] = {}
    for name, source in data.items():
        if (
            isinstance(source, (str, Path))
            and (table := as_table(source)) is not None
            and set(wanted[name]) <= set(table.collect_schema().names())
        ):
            scans[name] = table.select(wanted[name])
    return {**data, **dict(zip(scans, collected_all(scans.values()), strict=True))}


def with_positions(program: Program, tidied: Mapping[str, pl.LazyFrame]) -> dict[str, pl.LazyFrame]:
    """*tidied*, as [`tidy_sources`][] gave it, with each dimension's labels numbered.

    A dimension comes back as ``(dim, specsolve_position)``, the position an
    ``Int64`` from 0 in label order; a parameter and a relation unchanged.
    Nothing is read again, so a model archives what its build read.
    """
    tables = dict(tidied)
    for dim in program.dimensions:
        tables[dim] = tables[dim].with_row_index(POSITION).select(dim, pl.col(POSITION).cast(pl.Int64))
    return tables


def _unknown_source_keys_message(keys: Iterable[str], known: Iterable[str]) -> str:
    """A source key naming nothing the file declares."""
    unknown = sorted(keys)
    lead = f'source key {unknown[0]!r} names' if len(unknown) == 1 else f'source keys {unknown} name'
    return (
        f'{lead} neither a parameter, a dimension nor a relation this spec declares. '
        f'{did_you_mean(unknown[0], known, label="Declared")} Pass only what the '
        f'spec takes — a table carrying more than that is filtered here, not attached.'
    )


def _no_parameter_source_message(name: str) -> str:
    """A declared parameter with no source."""
    return f"no data provided for parameter '{name}'"


def _no_labels_source_message(dim: str) -> str:
    """A dimension with no labels."""
    return (
        f"dimension '{dim}' has no labels: pass them under key '{dim}' — a table "
        f'carrying that column, a parquet path, or a bare sequence of them. Its labels are what '
        f'say which coordinates exist, and without them a mistyped label is indistinguishable from '
        f'a new one.'
    )


def _relation_needs_labels_message(dim: str, authors: Iterable[str]) -> str:
    """A dimension whose relations have an author and whose labels have none."""
    declared = ', '.join(sorted(authors))
    return (
        f"dimension '{dim}' has its maps ({declared}) but no labels of its own, so nothing says "
        f'which of its labels exist. A relation is a table over the dimension, not the dimension '
        f"itself — it may omit labels, and its row order is arbitrary. Pass the labels of '{dim}' "
        f"under key '{dim}': every relation column over it is checked against them, and a label no "
        f'row mentions is unmapped.'
    )


# ---------------------------------------------------------------------------
# dimensions and relations
# ---------------------------------------------------------------------------


def _labels_table(source: Source, dim: str, dtype: str) -> pl.LazyFrame:
    """One dimension's labels, read once and held in memory as one column in the order they arrived.

    Raises:
        DataError: A table with no column named after the dimension, labels
            no table can be made of, a null label, or a label held twice.
    """
    given = as_table(source, (dim,))
    table = given if given is not None else _labels_from_sequence(dim, source, dtype)
    available = table.collect_schema().names()
    if dim not in available:
        raise DataError(
            f"the labels of dimension '{dim}' are a table without a '{dim}' column (has "
            f'{list(available)}). The label column is named after the dimension.'
        )
    labels = in_microseconds(
        table.select(dim).pipe(collected),
        f"the labels table of dimension '{dim}'",
        (dim,) if dtype == 'datetime' else (),
    )
    _check_labels_are_present(dim, labels[dim])
    _check_labels_are_unique(dim, labels[dim], given_as_table=given is not None)
    return labels.lazy()


def dimension_labels(program: Program, data: Mapping[str, Source], dim: str) -> pl.Series:
    """*dim*'s labels as [`tidy_sources`][] reads them out of *data*, in label order.

    Raises:
        DataError: No labels for *dim*, or labels [`tidy_sources`][] refuses.
    """
    if dim not in data:
        raise DataError(_no_labels_source_message(dim))
    _refuse_a_relation_on_the_labels(program, dim, data[dim])
    return _labels_of(dim, _labels_table(data[dim], dim, program.dimensions[dim].dtype))


def _check_labels_are_present(dim: str, labels: pl.Series) -> None:
    """Refuse labels holding a null: a coordinate is a label, and a null names none."""
    if not (nulls := labels.null_count()):
        return
    raise DataError(
        f"the labels of dimension '{dim}' hold a null ({nulls} row(s)). A coordinate is a label, and a "
        'null names none. Drop it: polars .drop_nulls(), pandas .dropna().'
    )


def _check_labels_are_unique(dim: str, labels: pl.Series, *, given_as_table: bool) -> None:
    """Refuse labels that hold one twice: a label's position is the row it is on."""
    twice = labels.filter(labels.is_duplicated()).unique(maintain_order=True).to_list()
    if not twice:
        return
    shown = ', '.join(repr(label) for label in twice[:5]) + (' …' if len(twice) > 5 else '')
    rewrite = (
        f'Pass the label column alone, each label once: table.select({dim!r}).unique(maintain_order=True)'
        if given_as_table
        else 'Pass each label once: list(dict.fromkeys(labels))'
    )
    raise DataError(
        f"the labels of dimension '{dim}' hold {len(twice)} label(s) more than once: {shown}. A dimension's "
        f"labels list each label once, and a label's row is the position `shift` counts. {rewrite} keeps the first "
        f'occurrence of each.'
    )


#: The declared dimension dtypes as the column an empty labels table becomes.
_DECLARED: dict[str, pl.DataType] = {
    'int': pl.Int64(),
    'float': pl.Float64(),
    'str': pl.String(),
    'datetime': pl.Datetime('us'),
}


def _labels_from_sequence(dim: str, values: Source, dtype: str) -> pl.LazyFrame:
    """A one-column labels table from a plain sequence of labels.

    An empty table takes the dimension's declared dtype: polars infers
    ``Null`` from no labels, and a ``Null`` key joins against nothing.
    """
    if not isinstance(values, Iterable):
        raise DataError(_not_labels(dim, values))
    try:
        labels = list(values)
        if not labels:
            return pl.LazyFrame(schema={dim: _DECLARED[dtype]})
        return pl.LazyFrame({dim: labels})
    except (TypeError, pl.exceptions.PolarsError) as exc:
        raise DataError(_not_labels(dim, values)) from exc


def _not_labels(dim: str, values: object) -> str:
    """One wording for a labels key nothing can read labels out of."""
    return (
        f"dimension '{dim}': cannot read labels out of "
        f'{type(values).__name__} — pass a sequence of labels, a table '
        f'polars can read with a {dim!r} column, or a parquet path'
    )


def _check_relation_sources(program: Program, data: Mapping[str, Source]) -> None:
    """Refuse a relation nothing supplies, and a relation column carried beside a dimension's labels."""
    for name, relation in program.relations.items():
        if name not in data:
            raise DataError(_unsupplied_relation_message(name, relation))

    for dim in program.dimensions:
        if dim in data:
            _refuse_a_relation_on_the_labels(program, dim, data[dim])


def _refuse_a_relation_on_the_labels(program: Program, dim: str, source: Source) -> None:
    """Refuse the labels *source* of *dim* where it carries a column named for a relation over *dim*."""
    carried = _column_names(source, dim)
    for name in _relations_over(program, dim):
        if name in carried:
            raise DataError(
                f"the labels table of dimension '{dim}' carries a '{name}' column, and '{name}' "
                f"is a relation with a column over '{dim}'. A relation is supplied under its own key, not "
                f'beside the labels of a dimension it runs over: pass it as sources[{name!r}], a table of '
                f'the rows it holds.'
            )


def in_microseconds(table: pl.DataFrame, owner: str, midnight: Collection[str] = ()) -> pl.DataFrame:
    """*table* with its datetime columns in microseconds, so a join or a membership test never meets two units.

    Args:
        table: The table whose datetime columns are cast.
        owner: What carried *table*, as a refusal names it.
        midnight: The columns whose dates are labels of a datetime dimension.
            A date there becomes the microsecond its day starts on, the same
            instant.

    Raises:
        DataError: A label with a part below one microsecond, which the cast would drop.
    """
    for column, dtype in table.schema.items():
        if isinstance(dtype, pl.Datetime) and dtype.time_unit == 'ns':
            finer = table[column].filter(table[column].to_physical() % 1000 != 0)
            if finer.len():
                raise DataError(
                    f"{owner} holds '{column}' label(s) finer than a microsecond, such as {finer.cast(pl.String)[0]}. "
                    f"Round them before attaching: polars .dt.round('1us'), pandas .dt.round('us')."
                )
    return table.with_columns(
        cs.datetime().dt.cast_time_unit('us'), (cs.by_name(midnight) & cs.date()).cast(pl.Datetime('us'))
    )


def _over_datetimes(columns: Iterable[tuple[str, str]], labels: Mapping[str, pl.LazyFrame]) -> list[str]:
    """The columns of *columns*, ``(column, dimension)`` pairs, whose dimension's labels are datetimes.

    The labels are read first and follow their entry, so a column follows
    the labels rather than the entry: a date beside labels that are dates
    stays a date.
    """
    return [c for c, dim in columns if dim in labels and isinstance(labels[dim].collect_schema()[dim], pl.Datetime)]


def _check_same_clock(owner: str, column: str, dim: str, given: pl.DataType, listed: pl.DataType) -> None:
    """Refuse a datetime column in another time zone than its dimension's labels: the two compare on one clock only."""
    if isinstance(given, pl.Datetime) and isinstance(listed, pl.Datetime) and given.time_zone != listed.time_zone:
        raise DataError(
            f"{owner} holds '{column}' in {given.time_zone or 'no time zone'}, and the labels of dimension "
            f"'{dim}' are in {listed.time_zone or 'no time zone'}. Put it on theirs first: polars "
            f'.dt.convert_time_zone, or .dt.replace_time_zone for a column without one.'
        )


def _relations_over(program: Program, dim: str) -> tuple[str, ...]:
    """The relations with a column over *dim*, in entry order."""
    return tuple(name for name, lk in program.relations.items() if any(over == dim for _, over in lk.columns))


def _unsupplied_relation_message(name: str, relation: Relation) -> str:
    """A relation nothing gives a table for."""
    rows = f'one row per {list(relation.key)} it maps' if relation.values else 'one row per tuple it relates'
    return (
        f"no data provided for relation '{name}'. Pass it under key '{name}' as a table with "
        f'columns {list(relation.roles)} — {rows}, and no row for one it does not.'
    )


def _column_names(source: Source, dim: str) -> frozenset[str]:
    """What a supplied labels table carries, or nothing where it is a bare label sequence."""
    table = as_table(source, (dim,))
    return frozenset(table.collect_schema().names()) if table is not None else frozenset()


def _check_relations_hold_labels(
    program: Program, relations: Mapping[str, pl.LazyFrame], labels: Mapping[str, pl.LazyFrame]
) -> None:
    """Every column of every relation holds labels of its dimension.

    Raises:
        DataError: A column holding a label its dimension lacks.
    """
    for name, relation in program.relations.items():
        for role, dim in relation.columns:
            _check_column_holds_labels(relations[name], name, role, dim, _labels_of(dim, labels[dim]))


def _check_column_holds_labels(rows: pl.LazyFrame, name: str, role: str, dim: str, labels: pl.Series) -> None:
    """Refuse a relation column holding a value that is not a label of its dimension.

    Offenders keep their own type — a python native off polars, never a numpy
    scalar — because the message reprs them.
    """
    _check_same_clock(f"relation '{name}'", role, dim, rows.collect_schema()[role], labels.dtype)
    known = set(labels.to_list())
    strays: dict[object, None] = {v: None for v in rows.select(role).pipe(collected)[role].to_list() if v not in known}
    if not strays:
        return
    shown = ', '.join(repr(v) for v in list(strays)[:5]) + (' …' if len(strays) > 5 else '')
    spelled = [str(x) for x in labels.to_list()]
    raise DataError(
        f"relation '{name}' has value(s) in '{role}' that are not '{dim}' labels: {shown}. '{dim}' takes "
        f'its labels from the data here, and they are {spelled[:8]}{" …" if len(spelled) > 8 else ""}. A '
        f'relation relates the labels that exist — a value matching none of them would place its terms '
        f'nowhere, so it is a typo on one side or a label missing from the other.'
    )


def _labels_of(dim: str, table: pl.LazyFrame) -> pl.Series:
    """One dimension's labels."""
    return table.select(dim).pipe(collected)[dim]


def _read_relation(source: Source, name: str, relation: Relation, labels: Mapping[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One supplied relation as the table both lanes read: one column per declared column, under its own name.

    A column is read in its dimension's label type, from *labels*.

    Raises:
        DataError: A source no reader accepts, a table short of a column,
            carrying a null, or holding a row twice.
    """
    roles = list(relation.roles)
    table = as_table(source, tuple(roles))
    if table is None:
        raise DataError(
            f"relation '{name}': cannot adapt {type(source).__name__} to a table — pass any "
            f'table polars can read with columns {roles} (polars, pyarrow, pandas), or a parquet '
            f'path.'
        )
    available = table.collect_schema().names()
    if any(c not in available for c in roles):
        keyed = (
            f'{list(relation.key)} is the key it is single-valued per'
            if relation.values
            else 'it is a bare relation, every column in its key'
        )
        raise DataError(
            f"relation '{name}' must carry a column per column it declares, {roles} (has "
            f'{list(available)}). {keyed}, and every column is over a dimension of its own.'
        )
    rows = in_microseconds(
        table.select(*roles).pipe(collected), f"relation '{name}'", _over_datetimes(relation.columns, labels)
    )

    holes = rows.filter(pl.any_horizontal(pl.col(c).is_null() for c in roles))
    if holes.height:
        holed = [c for c in roles if holes[c].null_count()]
        where = repr(holed[0]) if len(holed) == 1 else str(holed)
        shown = coordinates_text(roles, holes.head(5).rows())
        raise DataError(
            f"relation '{name}' carries {holes.height} row(s) with a null in {where}: {shown}. A relation "
            f'is partial by leaving a row out, not by relating a label to nothing — drop the row and '
            f'the label is unmapped, which is what every operator reading the relation already '
            f'means by it.'
        )

    key = list(relation.key)
    twice = rows.group_by(key).len().filter(pl.col('len') > 1).sort(key)
    if twice.height:
        shown = coordinates_text(key, twice.select(key).head(5).rows())
        if relation.values:
            raise DataError(
                f"relation '{name}' maps {twice.height} key(s) more than once: {shown}. "
                f'{key} is the declared key, so each key it maps takes exactly one row.'
            )
        raise DataError(
            f"relation '{name}' relates {twice.height} tuple(s) more than once: {shown}. "
            f'A relation holds each row at most once, so drop the repeats.'
        )

    return rows.lazy()


# ---------------------------------------------------------------------------
# parameters
# ---------------------------------------------------------------------------


def parameter_table(
    name: str, p: Parameter, obj: Source, sources: Mapping[str, pl.LazyFrame], kind: str = 'parameter'
) -> pl.LazyFrame:
    """The caller's object for one parameter as a lazy table, whatever shape it took.

    Raises:
        DataError: A shape neither a table reader nor [`_spread`][] accepts.
    """
    if is_dense_array(obj):
        raise DataError(
            f"{kind} '{name}': an xarray.DataArray is not a source. specsolve reads tables — "
            f'rows under named columns — and hands arrays back rather than taking them. Pass '
            f'array.to_series().reset_index() for a tidy table, whose columns attach by name. '
            f'Result.to_dataarray() is the way back out.'
        )
    if is_multi_indexed(obj):
        raise DataError(
            f"{kind} '{name}': a pandas Series with a MultiIndex is not a source. An index is "
            f'a pandas idea with no counterpart in the tables a build reads, and its depth is a '
            f"second claim about what '{name}' is over. Pass a tidy table carrying {[*p.dims, VALUE]} — "
            f'series.reset_index() is the whole change.'
        )
    table = as_table(obj, p.dims)
    return table if table is not None else _spread(name, obj, p.dims, sources, kind)


def read_values(
    kind: str, name: str, dims: Sequence[str], obj: Source, labels: Mapping[str, pl.LazyFrame], dtype: str
) -> pl.LazyFrame:
    """*obj*, ``(dims…, value)`` of *dtype* over *dims*, read and checked as a parameter's source is.

    For values that are no parameter's, such as a start's. *kind* and *name*
    say whose they are in a refusal, and *labels* holds each dimension's
    labels under its own name.

    Raises:
        DataError: A table a parameter over the same dimensions would be refused for.
    """
    declared = Parameter(dims=tuple(dims), dtype=dtype)  # pyrefly: ignore[bad-argument-type] — the two dtypes a start reads
    return _checked_parameter(name, declared, parameter_table(name, declared, obj, labels, kind), labels, kind)


def least_value(program: Program, sources: Mapping[str, Source], name: str) -> int:
    """The least value parameter *name*'s source holds, read without any labels; ``0`` where it holds no rows.

    Raises:
        DataError: No source for *name*, or a shape no reader accepts.
    """
    if name not in sources:
        raise DataError(_no_parameter_source_message(name))
    obj = sources[name]
    if isinstance(obj, (bool, int, float)):
        least = float(obj)
    elif isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        least = min(map(float, obj), default=None)  # pyrefly: ignore[bad-argument-type]  — a parameter's sequence holds numbers; a label sequence is a dimension's
    else:
        least = (
            parameter_table(name, program.parameters[name], obj, {}).select(pl.col(VALUE).min()).pipe(collected).item()
        )
    return 0 if least is None else int(least)


def _spread(
    name: str, obj: Source, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame], kind: str = 'parameter'
) -> pl.LazyFrame:
    """A parameter written as plain Python, spread over the dimensions it declares.

    A ``{label: value}`` map, a sequence in the dimension's own label order,
    or one number standing for every coordinate. One number keeps its Python
    type, so an int attaches to an ``int`` entry and a bool stays
    boolean: a condition's truthiness is read off the column type.

    Raises:
        DataError: A shape that does not fit the declared dimensions, a sequence of
            the wrong length, or a dimension whose labels nothing supplies.
    """
    if isinstance(obj, Mapping):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a dict maps one label to one value', dims, kind))
        return pl.LazyFrame({dims[0]: list(obj.keys()), VALUE: list(obj.values())})

    if isinstance(obj, bool):
        return _broadcast(name, pl.lit(obj, dtype=pl.Boolean), dims, sources, kind)
    if isinstance(obj, int):
        return _broadcast(name, pl.lit(obj, dtype=pl.Int64), dims, sources, kind)
    if isinstance(obj, float):
        return _broadcast(name, pl.lit(obj, dtype=pl.Float64), dims, sources, kind)

    if isinstance(obj, Collection) and not isinstance(obj, (str, bytes)):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a sequence runs along one dimension', dims, kind))
        labels = _labels(name, dims[0], sources, kind)
        values = list(obj)
        if len(values) != len(labels):
            raise DataError(
                f"{kind} '{name}': {len(values)} values against {len(labels)} "
                f"'{dims[0]}' labels. A sequence is positional, so it must have "
                f'one value per label, in label order.'
            )
        return pl.LazyFrame({dims[0]: labels, VALUE: values})

    raise DataError(
        f"{kind} '{name}': cannot adapt {type(obj).__name__} to a tidy "
        f'table — pass any table polars can read with columns '
        f'{[*dims, VALUE]} (polars, pyarrow, pandas), a parquet path, or the '
        f'plain-Python shapes: a dict, a sequence, or one number.'
    )


def _wrong_rank(name: str, said: str, dims: Sequence[str], kind: str) -> str:
    """One wording for a plain-Python shape against the dimensions it cannot cover."""
    return f"{kind} '{name}': {said}, and '{name}' is over {dims}. Pass a table with columns {[*dims, VALUE]} instead."


def _broadcast(
    name: str, value: pl.Expr, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame], kind: str
) -> pl.LazyFrame:
    """One number over every coordinate of *dims*."""
    table = pl.LazyFrame({'__one__': [0]})
    for dim in dims:
        table = table.join(pl.LazyFrame({dim: _labels(name, dim, sources, kind)}), how='cross')
    return table.drop('__one__').with_columns(value.alias(VALUE))


def _labels(name: str, dim: str, sources: Mapping[str, pl.LazyFrame], kind: str) -> list[Label]:
    """*dim*'s labels, in label order, for a shape that has none of its own.

    Raises:
        DataError: Nothing supplies the labels.
    """
    source = sources.get(dim)
    if source is None:
        raise DataError(
            f"{kind} '{name}' is written positionally over '{dim}', so it says what the "
            f'values are but not what they are labelled — and nothing else supplies the labels '
            f"of '{dim}'. Pass '{dim}': [...] in sources, or pass '{name}' as a table "
            f"carrying its own '{dim}' column."
        )
    return source.select(dim).pipe(collected)[dim].to_list()


def _checked_parameter(
    name: str,
    p: Parameter,
    table: pl.LazyFrame,
    sources: Mapping[str, pl.LazyFrame],
    kind: str = 'parameter',
) -> pl.LazyFrame:
    """*table* held to what its entry claims, collected once.

    The collect is the one model-sized materialisation on the way in.

    Raises:
        DataError: The table lacks a declared dimension or ``value``, holds two rows
            for one coordinate or a label its dimension lacks, carries a null
            or NaN value, or types the column differently from the entry.
    """
    wanted = [*p.dims, VALUE]
    available = table.collect_schema().names()
    if missing := set(wanted) - set(available):
        raise DataError(
            f"source for {kind} '{name}' is missing columns {sorted(missing)} "
            f'(need dimensions {list(p.dims)} plus {VALUE!r}; has {available}). Rename them to '
            f'the declared dimensions, or drop the index names to attach positionally.'
        )
    checked = in_microseconds(
        table.select(wanted).pipe(collected), f"{kind} '{name}'", _over_datetimes(((d, d) for d in p.dims), sources)
    )
    _check_one_row_per_coordinate(name, p, checked, sources, kind)
    _check_values_are_present(name, p, checked, kind)
    _check_value_dtype(name, p, checked, kind)
    return checked.lazy()


def _check_one_row_per_coordinate(
    name: str, p: Parameter, table: pl.DataFrame, sources: Mapping[str, pl.LazyFrame], kind: str
) -> None:
    """A table of values is a function of its dimensions: one row per coordinate, every label a real one.

    Labels are checked against the dimensions whose labels have been read; one
    still missing is refused once every source is in.
    """
    if not p.dims:
        if table.height > 1 or (table.height == 0 and p.missing == 'refused'):
            raise DataError(
                f"{kind} '{name}' is declared with no dimensions, which means one value "
                f'broadcast everywhere — but its source has {table.height} rows. '
                f'Declare the dimensions it is indexed by, reduce the source to a single row, or declare '
                f'`missing:` to say what no row means.'
            )
        return

    known = {d: _labels_of(d, sources[d]) for d in p.dims if d in sources}
    for d, labels in known.items():
        _check_same_clock(f"{kind} '{name}'", d, d, table.schema[d], labels.dtype)
    answers = table.select(
        pl.struct(p.dims).is_duplicated().any().alias('#duplicated'),
        *(_is_label(d, labels).all().alias(f'#known {d}') for d, labels in known.items()),
    ).row(0, named=True)

    for d, labels in known.items():
        if not answers[f'#known {d}']:
            strangers = table.filter(~_is_label(d, labels)).select(pl.col(d).unique())[d].to_list()
            shown = ', '.join(repr(s) for s in strangers[:5])
            more = f' (and {len(strangers) - 5} more)' if len(strangers) > 5 else ''
            raise DataError(
                f"{kind} '{name}' has label(s) in dimension '{d}' that are not labels "
                f'of it: {shown}{more}.\n'
                f'  {d} has: {sorted(str(k) for k in labels.to_list())[:10]}\n'
                f'A missing row means no value there, but a label the dimension does not have is a '
                f'typo: its row joins nothing, so the coordinate it was meant for silently reads '
                f'as absent. Fix the label, or add it to the labels of {d}.'
            )

    if not answers['#duplicated']:
        return
    duplicated = table.group_by(p.dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3)
    shown = '; '.join(
        f'{coordinate_text({d: row[d] for d in p.dims})} ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(
        f"{kind} '{name}' has more than one row for a coordinate: {shown}. "
        f'It holds one value per coordinate, so which value applies is undefined — '
        f'aggregate the source to one row per {list(p.dims)} before attaching it.'
    )


def _is_label(dim: str, labels: pl.Series) -> pl.Expr:
    """Whether each row's *dim* is one of *labels*: false for a null, which ``is_in`` leaves null and ``all`` skips."""
    return pl.col(dim).is_in(labels.implode()).fill_null(False)


def _check_values_are_present(name: str, p: Parameter, table: pl.DataFrame, kind: str) -> None:
    """Every row carries a value: a null or a NaN is refused rather than read."""
    value = pl.col(VALUE)
    holed = value.is_null() | value.is_nan() if table.schema[VALUE].is_float() else value.is_null()
    holes = int(table.select(holed.sum()).item())
    if not holes:
        return
    shown = coordinates_text(p.dims, table.filter(holed).select(p.dims).head(3).rows()) if p.dims else ''
    at = f': {shown}' if shown else ''
    raise DataError(
        f"{kind} '{name}' carries {holes} row(s) with no value — null or NaN{at}. "
        f'In long form the absence of a value is the absence of the row, and such a row '
        f'says the coordinate exists and denies it in the same breath.\n'
        f'  Drop them     polars .drop_nulls({VALUE!r}).drop_nans({VALUE!r}), pandas .dropna(subset=[{VALUE!r}])\n'
        f'  Supply them   if a number was what was meant'
    )


#: The column each declared dtype *is*, in polars types.
_COLUMNS: Mapping[str, tuple[type[pl.DataType], ...]] = {
    'float': (pl.Float32, pl.Float64),
    'int': (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64),
    'bool': (pl.Boolean,),
    'str': (pl.String, pl.Categorical, pl.Enum),
}

#: What each declared dtype accepts. ``int`` serving ``float`` is the one
#: widening.
ACCEPTED_VALUE_TYPES: Mapping[str, tuple[type[pl.DataType], ...]] = {
    **_COLUMNS,
    'float': _COLUMNS['float'] + _COLUMNS['int'],
}


def declared_dtype_of(column: pl.DataType) -> str | None:
    """The declared dtype a polars column is, or ``None`` for a column no declaration names."""
    return next((name for name, types in _COLUMNS.items() if column in types), None)


def _check_value_dtype(name: str, p: Parameter, table: pl.DataFrame, kind: str) -> None:
    """The attached column is the type the entry claims.

    Asked after the holes, so a column of nothing but nulls — which polars
    types ``Null`` — is told it has no values rather than the wrong kind.
    """
    column = table.schema[VALUE]
    if column in ACCEPTED_VALUE_TYPES[p.dtype]:
        return
    arrived = declared_dtype_of(column) or str(column)
    raise DataError(
        f"{kind} '{name}' is declared '{p.dtype}' and its values arrived as '{arrived}'. "
        f'A declared dtype is a claim about the values, and it is checked here — the file '
        f'says what the column is, or the column is not attached.\n'
        f'  Cast the column to {p.dtype}, if the declared dtype is what you meant\n'
        f'  Or declare what the data has: {{dtype: {arrived}}}'
    )
