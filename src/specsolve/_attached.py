"""The data a program is built against, as the engine reads it: each dimension's labels and positions, and the tables encoded against them.

A label's position is its row among its dimension's labels, and a coordinate's
is its row in the product of its dimensions, row-major ([`row_major`][]). A
solver index, a bound's slot, a set's position and the check that no
coordinate has two rows all read that one rule, so two builds of one model
agree on every index.

[`encoded`][specsolve._encoding.encoded] makes an [`AttachedSources`][] out of
the tidy tables, and [`attach`][specsolve._engine.attaching.attach] completes
it with what the model makes of the data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._names import LABEL, POSITION

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from mathspec import program


def row_major(
    sizes: Iterable[int], positions: Iterable[pl.Expr], dtype: type[pl.Int64 | pl.UInt64] = pl.Int64
) -> pl.Expr:
    """A coordinate's position in the product of its dimensions, from each label's position in its own.

    Contiguous over the full product, the last dimension running fastest.
    *sizes* and *positions* run over the dimensions in the order they are
    declared.
    """
    position: pl.Expr = pl.lit(0, dtype=dtype)
    for size, of_label in zip(sizes, positions, strict=True):
        position = position * size + of_label
    return position.cast(dtype)


@dataclass(frozen=True, eq=False)
class Dimension:
    """One dimension's labels, held once for everything that reads them."""

    name: str
    #: The labels in the order they arrived, as the engine holds them: string
    #: labels, in any polars encoding, as an ``Enum`` in that order, so its
    #: physical code is a label's position, and any other in the type it
    #: arrived in.
    labels: pl.Series
    #: The cast of a column called [`name`][] into [`dtype`][], null where a
    #: value is none of the labels, or ``None`` where the labels are held as
    #: they arrived. Built once and shared by every table that carries the
    #: dimension, since building a cast to an ``Enum`` of many labels costs a
    #: fixed time per expression.
    encoding: pl.Expr | None

    @classmethod
    def of(cls, name: str, labels: pl.Series) -> Dimension:
        """The dimension *name* over *labels*, as [`tidy_sources`][specsolve._sources.tidy_sources] read them."""
        if isinstance(labels.dtype, (pl.Categorical, pl.Enum)):
            labels = labels.cast(pl.String)
        if labels.dtype != pl.String:
            return cls(name, labels, None)
        ordered = pl.Enum(labels)
        return cls(name, labels.cast(ordered), pl.col(name).cast(ordered, strict=False))

    @property
    def size(self) -> int:
        """How many labels the dimension has."""
        return self.labels.len()

    @property
    def dtype(self) -> pl.DataType:
        """The type a column over this dimension holds once it is encoded."""
        return self.labels.dtype

    @cached_property
    def first(self) -> int | None:
        """The first label where the labels are consecutive ascending integers, so a label less it is its position."""
        if not self.labels.dtype.is_integer() or self.labels.is_empty():
            return None
        first = int(self.labels[0])
        return first if np.array_equal(self.labels.to_numpy(), np.arange(first, first + self.size)) else None

    @cached_property
    def table(self) -> pl.LazyFrame:
        """The dimension as ``(LABEL, POSITION)``, a label beside its position."""
        return pl.LazyFrame({LABEL: self.labels, POSITION: pl.int_range(self.size, dtype=pl.Int64, eager=True)})

    def encoded(self, column: str) -> pl.Expr | None:
        """The cast of the column called *column* into [`dtype`][], or ``None`` where it arrives in it."""
        if self.encoding is None:
            return None
        return self.encoding if column == self.name else pl.col(column).cast(self.labels.dtype, strict=False)

    def holds(self, column: pl.Expr) -> pl.Expr:
        """Whether each value of an encoded *column* is one of this dimension's labels; a null is none."""
        if self.encoding is not None:
            return column.is_not_null()
        if self.first is not None:
            return column.is_between(self.first, self.first + self.size - 1).fill_null(False)
        return column.is_in(self.labels.implode()).fill_null(False)

    def position(self, column: pl.Expr) -> pl.Expr:
        """An encoded *column* of this dimension's labels as each label's position."""
        if self.labels.dtype == pl.Enum:
            return column.to_physical().cast(pl.Int64)
        if self.first is not None:
            return column.cast(pl.Int64) - self.first
        return column.replace_strict(
            self.labels, pl.int_range(self.size, dtype=pl.Int64, eager=True), return_dtype=pl.Int64
        )


def label_product(columns: Mapping[str, Dimension]) -> pl.LazyFrame:
    """Every combination of the labels of each of *columns*' dimensions, one column per name, in row-major order."""
    tables = [held.table.select(pl.col(LABEL).alias(column)) for column, held in columns.items()]
    product = tables[0]
    for table in tables[1:]:
        product = product.join(table, how='cross', maintain_order='left_right')
    return product


@dataclass(frozen=True)
class AttachedSources:
    """The data a program is built against.

    ``parameters`` are tidy ``(dims…, value)``, a ``float`` parameter's value
    a ``Float64`` whatever column it arrived in, and a parameter that declares
    a value for a missing row filled with it; ``dimensions`` hold each
    dimension's labels and what turns one into its position;
    ``relations`` hold no row for a key that maps nowhere. Every column over a
    dimension is in that dimension's [`dtype`][specsolve._attached.Dimension.dtype].
    ``given`` holds the
    coordinates the data gave for each filled parameter, which a bare name in
    a ``where`` still asks about. ``parameter_rows`` are cached table heights.
    ``settled`` is each atom of a ``where`` that the
    data settles: ``True`` where it holds everywhere, ``False`` where
    nowhere ([`settle`][specsolve._engine.specialize.settle]).
    """

    parameters: Mapping[str, pl.LazyFrame]
    dimensions: Mapping[str, Dimension]
    relations: Mapping[str, pl.LazyFrame]
    parameter_rows: Mapping[str, int]
    given: Mapping[str, pl.LazyFrame] = field(default_factory=dict)
    settled: Mapping[program.Condition, bool] = field(default_factory=dict)
