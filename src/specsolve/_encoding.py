"""The tidy tables encoded against their dimensions, and checked for what the labels' positions show.

[`tidy_sources`][specsolve._sources.tidy_sources] has read every source and
held each to its shape. [`encoded`][] turns each label into its dimension's
own type, once, and asks of the result what is cheap to ask there: that every
label is one its dimension has, that no coordinate has two rows, and that no
coordinate lacks one where the spec refuses that. What the model makes of the
data after that is [`attach`][specsolve._engine.attaching.attach]'s.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import polars as pl
from mathspec.program import Parameter

from specsolve._attached import AttachedSources, Dimension, label_product, row_major
from specsolve._clock import untimed
from specsolve._collect import collected
from specsolve._messages import coordinate_text, coordinates_text, missing_row_message
from specsolve._names import VALUE
from specsolve._sources import checked_parameter, parameter_table
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mathspec.program import Program, Relation

    from specsolve._clock import Lap
    from specsolve._inputs import Source

#: The fewest rows at which a table's coordinates are counted by position rather than hashed.
POSITIONS_FROM = 50_000


def encoded(program: Program, tidy: Mapping[str, pl.DataFrame], lap: Lap = untimed) -> AttachedSources:
    """The tidy tables of *program* in its dimensions' own types, refused where they do not fit what it declares.

    A dimension's labels become its [`Dimension`][specsolve._attached.Dimension],
    and every column over it is cast to that dimension's type. A ``float``
    parameter's value becomes a ``Float64``.

    Args:
        program: The compiled spec.
        tidy: Its sources, as [`tidy_sources`][specsolve._sources.tidy_sources] returns them.
        lap: Ends a lap of ``attach`` after each step: ``encode`` and
            ``content``, one of each per relation and per parameter.

    Raises:
        DataError: A relation or a parameter holding a label its dimension
            lacks or one coordinate twice, or lacking a row where it declares
            ``missing: refused``.
    """
    dimensions = {d: Dimension.of(d, tidy[d][d]) for d in program.dimensions}
    relations: dict[str, pl.DataFrame] = {}
    for name, relation in program.relations.items():
        relations[name] = _in_dimension_types(tidy[name], relation.columns, dimensions, plain=False)
        lap('attach', 'encode', name)
        _refuse_a_relation_that_does_not_fit(name, relation, tidy[name], relations[name], dimensions)
        lap('attach', 'content', name)
    parameters: dict[str, pl.DataFrame] = {}
    for name, p in program.parameters.items():
        table = _in_dimension_types(tidy[name], [(d, d) for d in p.dims], dimensions)
        parameters[name] = table.with_columns(pl.col(VALUE).cast(pl.Float64)) if p.dtype == 'float' else table
        lap('attach', 'encode', name)
        _refuse_unknown_or_repeated_coordinates(name, 'parameter', p.dims, tidy[name], parameters[name], dimensions)
        if p.missing == 'refused':
            _refuse_a_missing_row('parameter', name, parameters[name], {d: dimensions[d] for d in p.dims})
        lap('attach', 'content', name)
    return AttachedSources(
        parameters={name: table.lazy() for name, table in parameters.items()},
        dimensions=dimensions,
        relations={name: table.lazy() for name, table in relations.items()},
        parameter_rows={name: table.height for name, table in parameters.items()},
    )


def read_values(
    kind: str,
    name: str,
    dims: Sequence[str],
    obj: Source,
    tidy: Mapping[str, pl.DataFrame],
    dimensions: Mapping[str, Dimension],
    dtype: str,
) -> pl.DataFrame:
    """*obj*, ``(dims…, value)`` of *dtype* over *dims*, read and checked as a parameter's source is.

    For values that are no parameter's, such as a start's. The table comes
    back as it was read, in the caller's types.

    Args:
        kind: Whose values they are, as a refusal names them.
        name: The entry they are for.
        dims: The dimensions they are over.
        obj: The caller's object, in any shape a parameter's source takes.
        tidy: The tidy tables, for each dimension's labels as they arrived.
        dimensions: The same dimensions as a build holds them.
        dtype: The declared dtype of the values.

    Raises:
        DataError: A table a parameter over the same dimensions would be refused for.
    """
    declared = Parameter(dims=tuple(dims), dtype=dtype)  # pyrefly: ignore[bad-argument-type] — the two dtypes a start reads
    shaped = parameter_table(name, declared, obj, tidy, kind)
    table = checked_parameter(name, declared, shaped, {d: tidy[d][d] for d in dims if d in tidy}, kind)
    _refuse_unknown_or_repeated_coordinates(
        name, kind, declared.dims, table, _in_dimension_types(table, [(d, d) for d in dims], dimensions), dimensions
    )
    return table


def _in_dimension_types(
    table: pl.DataFrame,
    columns: Sequence[tuple[str, str]],
    dimensions: Mapping[str, Dimension],
    *,
    plain: bool = True,
) -> pl.DataFrame:
    """*table* with each of *columns*, ``(column, dimension)``, cast to its dimension's type; null where a label is none of its.

    Where *plain*, a column that arrived with a dictionary of its own is read
    as plain strings first, so that dictionary cannot block the cast into an
    ``Enum``. A ``UInt64`` column over ``Int64`` labels is read as an
    ``Int64``, null past its range: polars compares the two types as floats,
    where two integers past 2**53 can be one number.
    """
    own = [c for c, _ in columns if plain and table.schema[c] in (pl.Categorical, pl.Enum)]
    if own:
        table = table.with_columns(pl.col(c).cast(pl.String) for c in own)
    casts = [cast for column, dim in columns if (cast := dimensions[dim].encoded(column)) is not None]
    casts += [
        pl.col(column).cast(pl.Int64, strict=False)
        for column, dim in columns
        if table.schema[column] == pl.UInt64 and dimensions[dim].dtype == pl.Int64
    ]
    return table.with_columns(casts) if casts else table


def _refuse_a_relation_that_does_not_fit(
    name: str, relation: Relation, tidy: pl.DataFrame, rows: pl.DataFrame, dimensions: Mapping[str, Dimension]
) -> None:
    """Refuse a relation holding a key twice, a label its column's dimension lacks, or no row where it refuses that.

    *tidy* holds the rows as they arrived, which is how a refusal shows them
    and how a key is told from another, and *rows* the same rows encoded.
    """
    key = list(relation.key)
    if tidy.select(key).is_duplicated().any():
        twice = tidy.group_by(key).len().filter(pl.col('len') > 1).sort(key)
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
    for role, dim in relation.columns:
        strays = tidy.filter(~rows.select(dimensions[dim].holds(pl.col(role))).to_series())
        if strays.height:
            _raise_unknown_relation_labels(name, role, dim, strays[role], dimensions[dim])
    if relation.missing == 'refused':
        wanted = relation.key or relation.roles
        columns = {column: dimensions[relation.dim(column)] for column in wanted}
        _refuse_a_missing_row('relation', name, rows.select(*wanted).unique(), columns)


def _raise_unknown_relation_labels(name: str, role: str, dim: str, strays: pl.Series, held: Dimension) -> None:
    """The refusal of a relation column holding values that are no label of its dimension.

    Offenders keep their own type — a python native off polars, never a numpy
    scalar — because the message reprs them.
    """
    unknown = list(dict.fromkeys(strays.to_list()))
    shown = ', '.join(repr(v) for v in unknown[:5]) + (' …' if len(unknown) > 5 else '')
    spelled = [str(x) for x in held.labels.to_list()]
    raise DataError(
        f"relation '{name}' has value(s) in '{role}' that are not '{dim}' labels: {shown}. '{dim}' takes "
        f'its labels from the data here, and they are {spelled[:8]}{" …" if len(spelled) > 8 else ""}. A '
        f'relation relates the labels that exist — a value matching none of them would place its terms '
        f'nowhere, so it is a typo on one side or a label missing from the other.'
    )


def _refuse_unknown_or_repeated_coordinates(
    name: str,
    kind: str,
    dims: tuple[str, ...],
    tidy: pl.DataFrame,
    table: pl.DataFrame,
    dimensions: Mapping[str, Dimension],
) -> None:
    """A table of values is a function of its dimensions: one row per coordinate, every label a real one.

    *tidy* holds the rows as they arrived, which is how a refusal shows them,
    and *table* the same rows encoded. A table with no row, or over no
    dimension, has neither to check, and is not queried.
    """
    if not dims or not table.height:
        return
    for d in dims:
        held = dimensions[d]
        known = table.select(held.holds(pl.col(d))).to_series()
        if not known.all():
            _raise_unknown_labels(name, kind, d, tidy.filter(~known)[d], held)
    positions = _coordinate_positions(dims, table, dimensions)
    repeated = table.select(dims).is_duplicated().any() if positions is None else positions.n_unique() < table.height
    if not repeated:
        return
    duplicated = tidy.group_by(dims).agg(pl.len().alias('#rows')).filter(pl.col('#rows') > 1).head(3)
    shown = '; '.join(
        f'{coordinate_text({d: row[d] for d in dims})} ({row["#rows"]} rows)'
        for row in duplicated.iter_rows(named=True)
    )
    raise DataError(
        f"{kind} '{name}' has more than one row for a coordinate: {shown}. "
        f'It holds one value per coordinate, so which value applies is undefined — '
        f'aggregate the source to one row per {list(dims)} before attaching it.'
    )


def _raise_unknown_labels(name: str, kind: str, dim: str, strays: pl.Series, held: Dimension) -> None:
    """The refusal of a table of values holding labels its dimension *dim* does not have."""
    unknown = strays.unique().to_list()
    shown = ', '.join(repr(s) for s in unknown[:5])
    more = f' (and {len(unknown) - 5} more)' if len(unknown) > 5 else ''
    raise DataError(
        f"{kind} '{name}' has label(s) in dimension '{dim}' that are not labels "
        f'of it: {shown}{more}.\n'
        f'  {dim} has: {sorted(str(k) for k in held.labels.to_list())[:10]}\n'
        f'A missing row means no value there, but a label the dimension does not have is a '
        f'typo: its row joins nothing, so the coordinate it was meant for silently reads '
        f'as absent. Fix the label, or add it to the labels of {dim}.'
    )


def _coordinate_positions(
    dims: tuple[str, ...], table: pl.DataFrame, dimensions: Mapping[str, Dimension]
) -> pl.Series | None:
    """Each row's coordinate position: where its coordinate sits in the product of its dimensions' labels.

    A coordinate's position is built from its labels' positions, row-major, so
    two rows share one exactly when they share a coordinate. Counting distinct
    positions replaces hashing every key column, which is the check's cost on a
    large dense table. ``None`` asks the caller to hash instead: for one key
    column, which hashes once either way, for a table under
    [`POSITIONS_FROM`][] rows, and for a product past 63 bits, where a
    position cannot be formed. Every label is one its dimension has by the
    time this is asked, so a column of another width than its dimension's is
    read in the dimension's.
    """
    if len(dims) < 2 or table.height < POSITIONS_FROM:
        return None
    if math.prod(dimensions[d].size for d in dims) >= 2**63:
        return None
    sizes = (dimensions[d].size for d in dims)
    labels = (
        pl.col(d) if table.schema[d] == dimensions[d].dtype else pl.col(d).cast(dimensions[d].dtype) for d in dims
    )
    return table.select(
        row_major(sizes, (dimensions[d].position(label) for d, label in zip(dims, labels, strict=True)))
    ).to_series()


def _refuse_a_missing_row(kind: str, name: str, rows: pl.DataFrame, columns: Mapping[str, Dimension]) -> None:
    """Refuse *rows* that hold no row at some combination of *columns*' labels, naming the first.

    No coordinate has two rows by now, so a height short of the product is a
    gap, and only a short table pays for the anti-join that names one.
    """
    if not columns:
        return
    missing = math.prod(held.size for held in columns.values()) - rows.height
    if missing <= 0:
        return
    gaps = label_product(columns).join(rows.lazy().select(*columns), on=list(columns), how='anti')
    first = gaps.head(1).pipe(collected).row(0, named=True)
    raise DataError(missing_row_message(kind, name, coordinate_text(first), missing))
