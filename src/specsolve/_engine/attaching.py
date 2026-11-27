"""What a caller's ``sources`` become: the tables the engine reads by name.

[`tidy_sources`][specsolve._sources.tidy_sources] has already read and checked
every source; [`attach`][] shapes each, encodes the string dimensions, and
checks the spec's ``assumptions:`` on them. It is the one way data reaches a
build or a read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program as program_module

from specsolve._collect import collected
from specsolve._dimensions import Dimension
from specsolve._engine.assumptions import check_assumptions
from specsolve._engine.coverage import divisors_of
from specsolve._engine.scope import reads_as_zero
from specsolve._engine.specialize import settle
from specsolve._messages import coordinate_text, missing_row_message, negative_window_message
from specsolve._names import LABEL, VALUE
from specsolve.errors import DataError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from mathspec import program


@dataclass(frozen=True)
class AttachedSources:
    """The data a program is built against, after attaching.

    ``parameters`` are tidy ``(dims…, value)``, a ``float`` parameter's value
    a ``Float64`` whatever column it arrived in, and a parameter that declares
    a value for a missing row filled with it; ``dimensions`` hold each
    dimension's labels and what turns one into its position;
    ``relations`` hold no row for a key that maps nowhere. ``given`` holds the
    coordinates the data gave for each filled parameter, which a bare name in
    a ``where`` still asks about. ``parameter_rows`` are cached table heights.
    ``settled`` is each atom of a ``where`` that the
    data settles: ``True`` where it holds everywhere, ``False`` where
    nowhere ([`settle`][]).
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, Dimension]
    relations: Mapping[str, pl.LazyFrame]
    parameter_rows: Mapping[str, int]
    given: Mapping[str, pl.LazyFrame]
    settled: Mapping[program.Condition, bool] = field(default_factory=dict)


def attach(program: program.Program, sources: Mapping[str, pl.DataFrame]) -> AttachedSources:
    """Shape the tables ``tidy_sources`` returns into what *program* is written against.

    A dimension's ``Enum`` is built from its labels, then every table that
    carries the dimension is re-encoded against it. Each dimension's
    cast is built once and shared, since building a cast to an ``Enum`` of
    many labels costs a fixed time per expression. Every atom of a ``where``
    is then read once ([`settle`][]), for the assumptions and the build both,
    and the assumptions are checked.

    Raises:
        DataError: A parameter or a relation that refuses a missing row has none
            at some coordinate, or an ``assumptions:`` entry the data does not
            hold.
    """
    dimensions = {d: Dimension.of(d, sources[d][d]) for d in program.dimensions}
    relations = {name: sources[name] for name in program.relations}
    parameters = {name: sources[name] for name in program.parameters}

    for name, relation in program.relations.items():
        casts = [
            pl.col(role).cast(dimensions[dim].dtype)
            for role, dim in relation.columns
            if dimensions[dim].encoding is not None
        ]
        if casts:
            relations[name] = relations[name].with_columns(casts)
    for name, p in program.parameters.items():
        table = _plain_strings(parameters[name], p.dims)
        casts = [cast for d in p.dims if (cast := dimensions[d].encoding) is not None]
        if p.dtype == 'float':
            casts.append(pl.col(VALUE).cast(pl.Float64))
        parameters[name] = table.with_columns(casts) if casts else table

    for name, relation in program.relations.items():
        if relation.missing == 'refused':
            key = relation.key or relation.roles
            _refuse_a_missing_row(
                'relation', name, relations[name].select(*key).unique(), key, relation.dim, dimensions
            )
    given: dict[str, pl.LazyFrame] = {}
    filled = filled_parameters(program)
    for name, p in program.parameters.items():
        if p.missing in ('absent', None) or (reads_as_zero(p.missing) and name not in filled):
            continue
        if p.missing == 'refused':
            _refuse_a_missing_row('parameter', name, parameters[name], p.dims, lambda d: d, dimensions)
            continue
        table = parameters[name]
        if table.height < math.prod(dimensions[d].size for d in p.dims):
            given[name] = table.select(*p.dims).lazy()
            parameters[name] = _filled(table, p.dims, p.missing, dimensions)

    attached = AttachedSources(
        parameters={name: f.lazy() for name, f in parameters.items()},
        dimensions=dimensions,
        relations={name: f.lazy() for name, f in relations.items()},
        parameter_rows={name: f.height for name, f in parameters.items()},
        given=given,
    )
    attached = replace(attached, settled=settle(program, attached))
    check_assumptions(program, attached)
    return attached


def filled_parameters(declared: program.Program) -> frozenset[str]:
    """The parameters read under a divisor anywhere in *declared*.

    A ``missing: 0`` row of one is filled rather than left sparse: a sparse
    zero is no contribution to a term, but under a divisor it is the number
    divided by, as in ``x / (1 + d)``. A bound fills its own read instead.
    """
    named = (e.expression for e in declared.expressions.values())
    return program_module.parameters_of(*divisors_of(*declared.roots, *named))


def refuse_a_negative_window(data: AttachedSources, name: str) -> None:
    """Refuse parameter *name*, the window of a ``sum_back``, where a row holds a negative width.

    Raises:
        DataError: Naming *name* and each negative width it holds.
    """
    negative = data.parameters[name].filter(pl.col(VALUE) < 0).select(pl.col(VALUE).unique().sort()).pipe(collected)
    if negative.height:
        raise DataError(negative_window_message(name, negative[VALUE].to_list()))


def _product(columns: tuple[str, ...], dim: Callable[[str], str], dimensions: Mapping[str, Dimension]) -> pl.LazyFrame:
    """Every combination of the labels of the dimension under each of *columns*, one column per name."""
    tables = [dimensions[dim(column)].table.select(pl.col(LABEL).alias(column)) for column in columns]
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
    dimensions: Mapping[str, Dimension],
) -> None:
    """Refuse *rows* that hold no row at some combination of *columns*' labels, naming the first.

    ``tidy_sources`` refuses a duplicate, so a height short of the product is a gap,
    and only a short table pays for the anti-join that names one.
    """
    if not columns:
        return
    missing = math.prod(dimensions[dim(column)].size for column in columns) - rows.height
    if missing <= 0:
        return
    gaps = _product(columns, dim, dimensions).join(rows.lazy().select(*columns), on=list(columns), how='anti')
    first = gaps.head(1).pipe(collected).row(0, named=True)
    raise DataError(missing_row_message(kind, name, coordinate_text(first), missing))


def _filled(
    table: pl.DataFrame, dims: tuple[str, ...], value: bool | float, dimensions: Mapping[str, Dimension]
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


def _plain_strings(table: pl.DataFrame, dims: tuple[str, ...]) -> pl.DataFrame:
    """Dimension columns as plain strings, so a writer's own dictionary cannot block the cast into an ``Enum``."""
    categorical = [d for d, dtype in table.schema.items() if d in dims and dtype in (pl.Categorical, pl.Enum)]
    if not categorical:
        return table
    return table.with_columns(pl.col(d).cast(pl.String) for d in categorical)
