"""Attach runtime data to a lowered program — the one door both lanes enter.

What the caller passed (parquet paths, any table exposing the Arrow PyCapsule
protocol, or a plain-Python shape) becomes the tidy frames both lanes read by
name, and every check on whether that data is usable is made here, once.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

import polars as pl

from specsolve.assumptions import validate_assumptions
from specsolve.errors import DataError, did_you_mean
from specsolve.frames import as_frame, is_dense_array, is_multi_indexed
from specsolve.relational.collect import polars_engine
from specsolve.relational.parquet import RESERVED

if TYPE_CHECKING:
    from mathspec.program import DimensionDeclaration, ParameterDeclaration, Program, RelationDeclaration

    from specsolve.lanes import Label, Source


def attachable(program: Program) -> dict[str, ParameterDeclaration | DimensionDeclaration | RelationDeclaration]:
    """Every name data may be attached to — declared parameters, dimensions and relations, one flat namespace."""
    return {**program.parameters, **program.dimensions, **program.relations}


def tidy_sources(program: Program, data: Mapping[str, Source]) -> dict[str, pl.LazyFrame]:
    """Read the caller's ``sources`` into the frames both lanes build against.

    Every source comes back as an in-memory `polars.LazyFrame`: a
    parameter as tidy ``(dims…, value)``, a dimension's index as its one
    column of labels under the dimension's own name, in the order they
    arrived, a relation as the table it declares, one column per column under
    the column's own name.

    Args:
        program: The lowered spec.
        data: Parameter, dimension and relation names to the caller's tables.

    Raises:
        DataError: A key naming nothing the spec declares; a declared
            dimension, relation or parameter with no data; a source no reader
            accepts or short of the columns its declaration needs; an index
            holding a label twice; a parameter
            with two rows for one coordinate, a label its dimension lacks, a
            null or NaN value, or a column of another type than it declares;
            a relation with a null, a row twice, or a label its column's
            dimension lacks; or an ``assumptions:`` entry the data does not
            hold, a ``piecewise:`` method's conditions on its breakpoints
            among them.
    """
    known = attachable(program)
    if unknown := set(data) - set(known):
        raise DataError(unknown_source_keys_message(unknown, known))

    _check_relation_sources(program, data)
    sources: dict[str, pl.LazyFrame] = {}
    for dname, declared in program.dimensions.items():
        if dname in data:
            sources[dname] = _index(data[dname], dname, declared.dtype)
    relations = {name: _read_relation(data[name], name, relation) for name, relation in program.relations.items()}
    for dname in program.dimensions:
        if dname not in sources and (authors := [f'sources[{name!r}]' for name in _relations_over(program, dname)]):
            raise DataError(_relation_needs_labels_message(dname, authors))
    _check_relations_hold_labels(program, relations, sources)
    sources |= relations

    for pname, pdef in program.parameters.items():
        if pname not in data:
            raise DataError(f"no data provided for parameter '{pname}'")
        sources[pname] = _parameter_frame(pname, pdef, data[pname], sources)
    for pname, pdef in program.parameters.items():
        sources[pname] = _checked_parameter(pname, pdef, sources[pname], sources)

    for dname in program.dimensions:
        if dname not in sources:
            raise DataError(no_index_source_message(dname))

    validate_assumptions(program, sources)
    return sources


#: The column a dimension's table numbers its labels in, from 0 in index order.
POSITION = f'{RESERVED}position'


def tidy_tables(program: Program, data: Mapping[str, Source]) -> dict[str, pl.LazyFrame]:
    """The tables a solve attaches: [`tidy_sources`][] with each dimension's labels numbered.

    A dimension comes back as ``(dim, specsolve_position)``, the position an
    ``Int64`` from 0 in index order; a parameter and a relation as
    [`tidy_sources`][] gives them.

    Raises:
        DataError: As [`tidy_sources`][] raises.
    """
    tables = tidy_sources(program, data)
    for dim in program.dimensions:
        tables[dim] = tables[dim].with_row_index(POSITION).select(dim, pl.col(POSITION).cast(pl.Int64))
    return tables


def unknown_source_keys_message(keys: Iterable[str], known: Iterable[str]) -> str:
    """A source key naming nothing the file declares."""
    unknown = sorted(keys)
    lead = f'source key {unknown[0]!r} names' if len(unknown) == 1 else f'source keys {unknown} name'
    return (
        f'{lead} neither a parameter, a dimension nor a relation this spec declares. '
        f'{did_you_mean(unknown[0], known, label="Declared")} Pass only what the '
        f'spec takes — a table carrying more than that is filtered here, not attached.'
    )


def no_index_source_message(dim: str) -> str:
    """A dimension with no index."""
    return (
        f"dimension '{dim}' has no index: pass its labels under key '{dim}' — a table "
        f'carrying that column, a parquet path, or a bare sequence of them. The index is what '
        f'says which labels exist, and without one a mistyped label is indistinguishable from '
        f'a new one.'
    )


def _relation_needs_labels_message(dim: str, authors: Iterable[str]) -> str:
    """A dimension whose relations have an author and whose labels have none."""
    declared = ', '.join(sorted(authors))
    return (
        f"dimension '{dim}' has its maps ({declared}) but no index of its own, so nothing says "
        f'which of its labels exist. A relation is a table over the dimension, not the dimension '
        f"itself — it may omit members, and its row order is arbitrary. Pass an index for '{dim}' "
        f'under that key: every relation column over it is checked against the index, and a label no '
        f'row mentions is unmapped.'
    )


# ---------------------------------------------------------------------------
# dimensions and relations
# ---------------------------------------------------------------------------


def _index(source: Source, dim: str, dtype: str) -> pl.LazyFrame:
    """One dimension's labels, read once and held in memory as one column in the order they arrived.

    Raises:
        DataError: A table with no column named after the dimension, labels
            no frame can be made of, or a label held twice.
    """
    given = as_frame(source, (dim,))
    table = given if given is not None else _labels_frame(dim, source, dtype)
    available = table.collect_schema().names()
    if dim not in available:
        raise DataError(
            f"index for dimension '{dim}' is a table without a '{dim}' column (has "
            f'{list(available)}). The label column is named after the dimension.'
        )
    labels = table.select(dim).collect()
    _check_labels_are_unique(dim, labels[dim], given_as_table=given is not None)
    return labels.lazy()


def _check_labels_are_unique(dim: str, labels: pl.Series, *, given_as_table: bool) -> None:
    """Refuse an index that holds a label twice: a label's position is the row it is on.

    The rewrite the message names is for the shape the index came in.
    """
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
        f"index for dimension '{dim}' holds {len(twice)} label(s) more than once: {shown}. An index "
        f'lists each label once, and its row is the position `shift` counts. {rewrite} keeps the first '
        f'occurrence of each.'
    )


#: The declared dimension dtypes as the column an empty index becomes.
_DECLARED: dict[str, pl.DataType] = {
    'int': pl.Int64(),
    'float': pl.Float64(),
    'str': pl.String(),
    'datetime': pl.Datetime('us'),
}


def _labels_frame(dim: str, values: Source, dtype: str) -> pl.LazyFrame:
    """A one-column index frame from a plain sequence of labels.

    An empty index takes the dimension's declared dtype: polars infers
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
    """One wording for an index nothing can read labels out of."""
    return (
        f"index for dimension '{dim}': cannot read labels out of "
        f'{type(values).__name__} — pass a sequence of labels, a table '
        f'polars can read with a {dim!r} column, or a parquet path'
    )


def _check_relation_sources(program: Program, data: Mapping[str, Source]) -> None:
    """Refuse a relation nothing supplies, and a relation column carried on an index."""
    for name, relation in program.relations.items():
        if name not in data:
            raise DataError(_unsupplied_relation_message(name, relation))

    for dim in program.dimensions:
        if dim not in data:
            continue
        carried = _column_names(data[dim], dim)
        for name in _relations_over(program, dim):
            if name in carried:
                raise DataError(
                    f"index for dimension '{dim}' carries a '{name}' column, and '{name}' "
                    f"is a relation with a column over '{dim}'. A relation is supplied under its own key, not "
                    f'as a column of an index it runs over: pass it as sources[{name!r}], a table of '
                    f'the rows it holds.'
                )


def _relations_over(program: Program, dim: str) -> tuple[str, ...]:
    """The relations with a column over *dim*, in declaration order."""
    return tuple(name for name, lk in program.relations.items() if any(over == dim for _, over in lk.columns))


def _unsupplied_relation_message(name: str, relation: RelationDeclaration) -> str:
    """A relation nothing gives a table for."""
    rows = f'one row per {list(relation.key)} it maps' if relation.values else 'one row per tuple it relates'
    return (
        f"no data provided for relation '{name}'. Pass it under key '{name}' as a table with "
        f'columns {list(relation.roles)} — {rows}, and no row for one it does not.'
    )


def _column_names(source: Source, dim: str) -> frozenset[str]:
    """What a supplied index carries, or nothing where it is a bare label sequence."""
    table = as_frame(source, (dim,))
    return frozenset(table.collect_schema().names()) if table is not None else frozenset()


def _check_relations_hold_labels(
    program: Program, relations: Mapping[str, pl.LazyFrame], indices: Mapping[str, pl.LazyFrame]
) -> None:
    """Every column of every relation holds labels of its dimension.

    Raises:
        DataError: A column holding a label its dimension lacks.
    """
    for name, relation in program.relations.items():
        for role, dim in relation.columns:
            _check_column_holds_labels(relations[name], name, role, dim, _labels_of(dim, indices[dim]))


def _check_column_holds_labels(rows: pl.LazyFrame, name: str, role: str, dim: str, labels: pl.Series) -> None:
    """Refuse a relation column holding a value that is not a label of its dimension.

    Offenders keep their own type — a python native off polars, never a numpy
    scalar — because the message reprs them.
    """
    known = set(labels.to_list())
    strays: dict[object, None] = {v: None for v in rows.select(role).collect()[role].to_list() if v not in known}
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


def _labels_of(dim: str, index: pl.LazyFrame) -> pl.Series:
    """One dimension's labels."""
    return index.select(dim).collect()[dim]


def _read_relation(source: Source, name: str, relation: RelationDeclaration) -> pl.LazyFrame:
    """One supplied relation as the frame both lanes read: one column per declared column, under its own name.

    Raises:
        DataError: A source no reader accepts, a table short of a column,
            carrying a null, or holding a row twice.
    """
    roles = list(relation.roles)
    table = as_frame(source, tuple(roles))
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
    rows = table.select(*roles).collect()

    holes = rows.filter(pl.any_horizontal(pl.col(c).is_null() for c in roles))
    if holes.height:
        holed = [c for c in roles if holes[c].null_count()]
        where = repr(holed[0]) if len(holed) == 1 else str(holed)
        shown = coordinates_shown(roles, holes.head(5).rows())
        raise DataError(
            f"relation '{name}' carries {holes.height} row(s) with a null in {where}: {shown}. A relation "
            f'is partial by leaving a row out, not by relating a label to nothing — drop the row and '
            f'the label is unmapped, which is what every operator reading the relation already '
            f'means by it.'
        )

    key = list(relation.key)
    twice = rows.group_by(key).len().filter(pl.col('len') > 1).sort(key)
    if twice.height:
        shown = coordinates_shown(key, twice.select(key).head(5).rows())
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


def _parameter_frame(
    name: str, p: ParameterDeclaration, obj: Source, sources: Mapping[str, pl.LazyFrame]
) -> pl.LazyFrame:
    """The caller's object for one parameter as a lazy frame, whatever shape it took.

    Raises:
        DataError: A shape neither a table reader nor [`_spread`][] accepts.
    """
    if is_dense_array(obj):
        raise DataError(
            f"parameter '{name}': an xarray.DataArray is not a source. specsolve reads tables — "
            f'rows under named columns — and hands arrays back rather than taking them. Pass '
            f'array.to_series().reset_index() for a tidy frame, whose columns attach by name. '
            f'Result.to_dataarray() is the way back out.'
        )
    if is_multi_indexed(obj):
        raise DataError(
            f"parameter '{name}': a pandas Series with a MultiIndex is not a source. An index is "
            f'a pandas idea with no counterpart in the frames a build reads, and its depth is a '
            f"second claim about what '{name}' is over. Pass a tidy frame carrying {[*p.dims, 'value']} — "
            f'series.reset_index() is the whole change.'
        )
    table = as_frame(obj, p.dims)
    return table if table is not None else _spread(name, obj, p.dims, sources)


def least_value(name: str, p: ParameterDeclaration, obj: Source) -> float | None:
    """The least value one parameter's source holds, read without any dimension's labels.

    Returns:
        The least value, or ``None`` where the source holds no rows.

    Raises:
        DataError: A shape no reader accepts.
    """
    if isinstance(obj, (bool, int, float)):
        return float(obj)
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return min(map(float, obj), default=None)  # pyrefly: ignore[bad-argument-type]  — a parameter's sequence holds numbers; a label sequence is an index's
    return _parameter_frame(name, p, obj, {}).select(pl.col('value').min()).collect().item()


def _spread(name: str, obj: Source, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame]) -> pl.LazyFrame:
    """A parameter written as plain Python, spread over the dims it declares.

    A ``{label: value}`` map, a sequence in the dimension's own label order,
    or one number standing for every coordinate. A bool stays boolean rather
    than widening to float: a mask's truthiness is read off the column type.

    Raises:
        DataError: A shape that does not fit the declared dims, a sequence
            whose length does not match, or a dimension whose labels nothing
            supplies.
    """
    if isinstance(obj, Mapping):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a dict maps one label to one value', dims))
        return pl.LazyFrame({dims[0]: list(obj.keys()), 'value': list(obj.values())})

    if isinstance(obj, bool):
        return _broadcast(name, pl.lit(obj, dtype=pl.Boolean), dims, sources)
    if isinstance(obj, (int, float)):
        return _broadcast(name, pl.lit(float(obj), dtype=pl.Float64), dims, sources)

    if isinstance(obj, Collection) and not isinstance(obj, (str, bytes)):
        if len(dims) != 1:
            raise DataError(_wrong_rank(name, 'a sequence runs along one dimension', dims))
        labels = _labels(name, dims[0], sources)
        values = list(obj)
        if len(values) != len(labels):
            raise DataError(
                f"parameter '{name}': {len(values)} values against {len(labels)} "
                f"'{dims[0]}' labels. A sequence is positional, so it must have "
                f'one entry per label, in the order the index declares them.'
            )
        return pl.LazyFrame({dims[0]: labels, 'value': values})

    raise DataError(
        f"parameter '{name}': cannot adapt {type(obj).__name__} to a tidy "
        f'table — pass any table polars can read with columns '
        f'{[*dims, "value"]} (polars, pyarrow, pandas), a parquet path, or the '
        f'plain-Python shapes: a dict, a sequence, or one number.'
    )


def _wrong_rank(name: str, said: str, dims: Sequence[str]) -> str:
    """One wording for a plain-Python shape against the dims it cannot cover."""
    return (
        f"parameter '{name}': {said}, and '{name}' is over {dims}. Pass a table "
        f'with columns {[*dims, "value"]} instead.'
    )


def _broadcast(name: str, value: pl.Expr, dims: Sequence[str], sources: Mapping[str, pl.LazyFrame]) -> pl.LazyFrame:
    """One number over every coordinate of *dims*."""
    frame = pl.LazyFrame({'__one__': [0]})
    for dim in dims:
        frame = frame.join(pl.LazyFrame({dim: _labels(name, dim, sources)}), how='cross')
    return frame.drop('__one__').with_columns(value.alias('value'))


def _labels(name: str, dim: str, sources: Mapping[str, pl.LazyFrame]) -> list[Label]:
    """*dim*'s labels, in index order, for a shape that has none of its own.

    Raises:
        DataError: Nothing supplies the labels.
    """
    source = sources.get(dim)
    if source is None:
        raise DataError(
            f"parameter '{name}' is written positionally over '{dim}', so it says what the "
            f'values are but not what they are labelled — and nothing else supplies an index '
            f"for '{dim}'. Pass '{dim}': [...] in sources, or pass '{name}' as a table "
            f"carrying its own '{dim}' column."
        )
    return source.select(dim).collect()[dim].to_list()


def _checked_parameter(
    name: str, p: ParameterDeclaration, table: pl.LazyFrame, sources: Mapping[str, pl.LazyFrame]
) -> pl.LazyFrame:
    """*table* held to what its declaration claims, collected once.

    The collect is the one model-sized materialisation on the way in.

    Raises:
        DataError: The frame lacks a declared dim or ``value``, holds two rows
            for one coordinate or a label its dimension lacks, carries a null
            or NaN value, or types the column differently from the declaration.
    """
    wanted = [*p.dims, 'value']
    available = table.collect_schema().names()
    if missing := set(wanted) - set(available):
        raise DataError(
            f"source for parameter '{name}' is missing columns {sorted(missing)} "
            f"(need dims {list(p.dims)} plus 'value'; has {available}). Rename them to "
            f'the declared dims, or drop the index names to attach positionally.'
        )
    frame = table.select(wanted).collect(engine=polars_engine())
    _check_one_row_per_coordinate(name, p, frame, sources)
    _check_values_are_present(name, p, frame)
    _check_value_dtype(name, p, frame)
    return frame.lazy()


def _check_one_row_per_coordinate(
    name: str, p: ParameterDeclaration, frame: pl.DataFrame, sources: Mapping[str, pl.LazyFrame]
) -> None:
    """A parameter is a function of its dims: one row per coordinate, every label a real one.

    Labels are checked against the dimensions whose index has been read; one
    still missing is refused once every source is in. A parameter with no dims
    takes exactly one row.
    """
    if not p.dims:
        if frame.height != 1:
            raise DataError(
                f"parameter '{name}' is declared with no dims, which means one value "
                f'broadcast everywhere — but its source has {frame.height} rows. '
                f'Declare the dims it is indexed by, or reduce the source to a single row.'
            )
        return

    known = {d: _labels_of(d, sources[d]) for d in p.dims if d in sources}
    answers = frame.select(
        pl.struct(p.dims).is_duplicated().any().alias('#duplicated'),
        *(pl.col(d).is_in(labels.implode()).all().alias(f'#known {d}') for d, labels in known.items()),
    ).row(0, named=True)

    for d, labels in known.items():
        if not answers[f'#known {d}']:
            strangers = frame.filter(~pl.col(d).is_in(labels.implode())).select(pl.col(d).unique())[d].to_list()
            shown = ', '.join(repr(s) for s in strangers[:5])
            more = f' (and {len(strangers) - 5} more)' if len(strangers) > 5 else ''
            raise DataError(
                f"parameter '{name}' has label(s) in dimension '{d}' that are not coordinates "
                f'of it: {shown}{more}.\n'
                f'  {d} has: {sorted(str(k) for k in labels.to_list())[:10]}\n'
                f'A missing row is a zero coefficient, but a label that is not a coordinate is a '
                f'typo: its row joins nothing, so the coordinate it was meant for silently reads '
                f'as absent. Fix the label, or declare it as a coordinate.'
            )

    if not answers['#duplicated']:
        return
    duplicated = frame.group_by(p.dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3)
    shown = '; '.join(
        ', '.join(f'{d}={row[d]!r}' for d in p.dims) + f' ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(
        f"parameter '{name}' has more than one row for a coordinate: {shown}. "
        f'A parameter is a function of its dims, so which value applies is undefined — '
        f'aggregate the source to one row per {list(p.dims)} before attaching it.'
    )


def _check_values_are_present(name: str, p: ParameterDeclaration, frame: pl.DataFrame) -> None:
    """Every row carries a value: a null or a NaN is refused rather than read."""
    value = pl.col('value')
    holed = value.is_null() | value.is_nan() if frame.schema['value'].is_float() else value.is_null()
    holes = int(frame.select(holed.sum()).item())
    if not holes:
        return
    shown = coordinates_shown(p.dims, frame.filter(holed).select(p.dims).head(3).rows()) if p.dims else ''
    at = f': {shown}' if shown else ''
    raise DataError(
        f"parameter '{name}' carries {holes} row(s) with no value — null or NaN{at}. "
        f'In long form the absence of a value is the absence of the row, and such a row '
        f'says the coordinate exists and denies it in the same breath.\n'
        f'  Drop them     polars .drop_nulls("value").drop_nans("value"), pandas .dropna(subset=["value"])\n'
        f'  Supply them   if a number was what was meant'
    )


def coordinates_shown(dims: Sequence[str], rows: Iterable[Sequence[Label]]) -> str:
    """Coordinates as a refusal prints them: ``f='b'; f='c'``."""
    return '; '.join(', '.join(f'{d}={v!r}' for d, v in zip(dims, row, strict=True)) for row in rows)


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


def _check_value_dtype(name: str, p: ParameterDeclaration, frame: pl.DataFrame) -> None:
    """The attached column is the type the declaration claims.

    Asked after the holes, so a column of nothing but nulls — which polars
    types ``Null`` — is told it has no values rather than the wrong kind.
    """
    column = frame.schema['value']
    if column in ACCEPTED_VALUE_TYPES[p.dtype]:
        return
    arrived = next((name for name, types in _COLUMNS.items() if column in types), str(column))
    raise DataError(
        f"parameter '{name}' is declared '{p.dtype}' and its values arrived as '{arrived}'. "
        f'A declared dtype is a claim about the values, and it is checked here — the file '
        f'says what the column is, or the column is not attached.\n'
        f'  Cast the column to {p.dtype}, if the declaration is what you meant\n'
        f'  Or declare what the data has: {{dtype: {arrived}}}'
    )
