"""What a caller's ``sources`` become: the tables the engine reads by name.

[`tidy_sources`][specsolve._sources.tidy_sources] has already read and checked
every source; this shapes each and encodes the string dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._collect import collected, collected_all, in_memory
from specsolve._messages import coordinate_text, missing_row_message, negative_window_message
from specsolve._names import LABEL, POSITION, VALUE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mathspec import program


@dataclass(frozen=True)
class AttachedSources:
    """The data a program is built against, after attaching.

    ``parameters`` are tidy ``(dims…, value)``, a ``float`` parameter's value
    a ``Float64`` whatever column it arrived in, and a parameter that declares
    a value for a missing row filled with it; ``dimensions`` are ``(LABEL, POSITION)``;
    ``relations`` hold no row for a key that maps nowhere. ``given`` holds the
    coordinates the data gave for each filled parameter, which a bare name in
    a ``where`` still asks about. ``cardinality`` and ``parameter_rows`` are
    cached table heights.
    ``consecutive`` maps each dimension whose labels are the integers
    ``first, first + 1, …`` in that order to ``first``, so that a label less
    ``first`` is its position.
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, pl.LazyFrame]
    relations: Mapping[str, pl.LazyFrame]
    cardinality: Mapping[str, int]
    parameter_rows: Mapping[str, int]
    consecutive: Mapping[str, int]
    given: Mapping[str, pl.LazyFrame]

    def is_enum_encoded(self, dim: str) -> bool:
        """Whether *dim* was given an ``Enum``."""
        return self.dimensions[dim].collect_schema()[LABEL] == pl.Enum


#: A float parameter's value column, whatever numeric type it arrived in.
_AS_FLOAT = pl.col(VALUE).cast(pl.Float64)


def attach(program: program.Program, sources: Mapping[str, pl.LazyFrame]) -> AttachedSources:
    """Shape the tables ``tidy_sources`` returns into what *program* is written against.

    A dimension's ``Enum`` is built from its labels, then every table that
    carries the dimension is re-encoded against it. Each table is read
    [`in_memory`][], since every read holds one table whole. A dimension's
    cast is built once and the parameters are cast in one batch: building a
    cast to an ``Enum`` of many labels costs a fixed time per expression, and
    PyPSA's spec casts hundreds of tables over the same dozen dimensions.

    Raises:
        DataError: A parameter or a relation that refuses a missing row has none
            at some coordinate.
    """
    with in_memory():
        dimensions = {d: _position_table(d, sources[d]).pipe(collected) for d in program.dimensions}
        relations = {name: sources[name].pipe(collected) for name in program.relations}
        parameters = {name: sources[name].pipe(collected) for name in program.parameters}

    enums = {d: pl.Enum(f[LABEL]) for d, f in dimensions.items() if f.schema[LABEL] == pl.String}
    for d, enum in enums.items():
        dimensions[d] = dimensions[d].with_columns(pl.col(LABEL).cast(enum))
    for name, relation in program.relations.items():
        casts = [pl.col(role).cast(enums[dim]) for role, dim in relation.columns if dim in enums]
        if casts:
            relations[name] = relations[name].with_columns(casts)
    to_enum = {d: pl.col(d).cast(enum) for d, enum in enums.items()}
    uncollected: dict[str, pl.LazyFrame] = {}
    for name, p in program.parameters.items():
        table = _plain_strings(parameters[name], p.dims)
        casts = [to_enum[d] for d in p.dims if d in to_enum]
        if p.dtype == 'float':
            casts.append(_AS_FLOAT)
        if casts:
            uncollected[name] = table.lazy().with_columns(casts)
        else:
            parameters[name] = table
    with in_memory():
        parameters |= dict(zip(uncollected, collected_all(uncollected.values()), strict=True))

    for name, relation in program.relations.items():
        if relation.missing == 'refused':
            key = relation.key or relation.roles
            _refuse_a_missing_row(
                'relation', name, relations[name].select(*key).unique(), key, relation.dim, dimensions
            )
    given: dict[str, pl.LazyFrame] = {}
    for name, p in program.parameters.items():
        if p.missing in ('absent', 'neutral', None):
            continue
        if p.missing == 'refused':
            _refuse_a_missing_row('parameter', name, parameters[name], p.dims, lambda d: d, dimensions)
            continue
        table = parameters[name]
        if table.height < math.prod(dimensions[d].height for d in p.dims):
            given[name] = table.select(*p.dims).lazy()
            parameters[name] = _filled(table, p.dims, p.missing, dimensions)

    return AttachedSources(
        parameters={name: f.lazy() for name, f in parameters.items()},
        dimensions={d: f.lazy() for d, f in dimensions.items()},
        relations={name: f.lazy() for name, f in relations.items()},
        cardinality={d: f.height for d, f in dimensions.items()},
        parameter_rows={name: f.height for name, f in parameters.items()},
        consecutive={d: first for d, f in dimensions.items() if (first := _first_of_a_run(f)) is not None},
        given=given,
    )


def refuse_a_negative_window(data: AttachedSources, name: str) -> None:
    """Refuse parameter *name*, the window of a ``sum_back``, where a row holds a negative width.

    Raises:
        DataError: Naming *name* and each negative width it holds.
    """
    negative = data.parameters[name].filter(pl.col(VALUE) < 0).select(pl.col(VALUE).unique().sort()).pipe(collected)
    if negative.height:
        raise DataError(negative_window_message(name, negative[VALUE].to_list()))


def _product(
    columns: tuple[str, ...], dim: Callable[[str], str], dimensions: Mapping[str, pl.DataFrame]
) -> pl.LazyFrame:
    """Every combination of the labels of the dimension under each of *columns*, one column per name."""
    tables = [dimensions[dim(column)].lazy().select(pl.col(LABEL).alias(column)) for column in columns]
    product = tables[0]
    for table in tables[1:]:
        product = product.join(table, how='cross', maintain_order='left_right')
    return product


def _refuse_a_missing_row(
    kind: str,
    name: str,
    rows: pl.DataFrame,
    columns: tuple[str, ...],
    dim: Callable[[str], str],
    dimensions: Mapping[str, pl.DataFrame],
) -> None:
    """Refuse *rows* that hold no row at some combination of *columns*' labels, naming the first.

    ``tidy_sources`` refuses a duplicate, so a height short of the product is a gap,
    and only a short table pays for the anti-join that names one.
    """
    if not columns:
        return
    missing = math.prod(dimensions[dim(column)].height for column in columns) - rows.height
    if missing <= 0:
        return
    gaps = _product(columns, dim, dimensions).join(rows.lazy().select(*columns), on=list(columns), how='anti')
    first = gaps.head(1).pipe(collected).row(0, named=True)
    raise DataError(missing_row_message(kind, name, coordinate_text(first), missing))


def _filled(
    table: pl.DataFrame, dims: tuple[str, ...], value: bool | float, dimensions: Mapping[str, pl.DataFrame]
) -> pl.DataFrame:
    """*table* with a row at every coordinate of *dims*, *value* where the data has none.

    The rows come back in label order, so a sum over them adds its terms in
    the same order at every build.
    """
    if not dims:
        return table if table.height else pl.DataFrame({VALUE: [value]}, schema={VALUE: table.schema[VALUE]})
    product = _product(dims, lambda d: d, dimensions)
    filled = product.join(table.lazy(), on=list(dims), how='left', maintain_order='left').with_columns(
        pl.col(VALUE).fill_null(value)
    )
    return filled.select(*dims, pl.col(VALUE).cast(table.schema[VALUE])).pipe(collected)


def _first_of_a_run(dimension: pl.DataFrame) -> int | None:
    """The first label of a dimension whose labels are consecutive ascending integers, else ``None``."""
    labels = dimension.get_column(LABEL)
    if not labels.dtype.is_integer() or labels.is_empty():
        return None
    first = int(labels[0])
    return first if np.array_equal(labels.to_numpy(), np.arange(first, first + labels.len())) else None


def _position_table(d: str, labels: pl.LazyFrame) -> pl.LazyFrame:
    """A dimension's ``(LABEL, POSITION)`` from its labels, a label's position being its row."""
    return labels.with_row_index(POSITION).select(pl.col(d).alias(LABEL), pl.col(POSITION).cast(pl.Int64))


def _plain_strings(table: pl.DataFrame, dims: tuple[str, ...]) -> pl.DataFrame:
    """Dimension columns as plain strings, so a writer's own dictionary cannot block the cast into an ``Enum``."""
    categorical = [d for d, dtype in table.schema.items() if d in dims and dtype in (pl.Categorical, pl.Enum)]
    if not categorical:
        return table
    return table.with_columns(pl.col(d).cast(pl.String) for d in categorical)
