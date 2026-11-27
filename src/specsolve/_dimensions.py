"""A dimension as the data and the engine both read it: its labels in order, and what turns a label into its position.

A label's position is its row among its dimension's labels, and a coordinate's
is its row in the product of its dimensions, row-major ([`row_major`][]). A
solver index, a bound's slot, a set's position and the check that no
coordinate has two rows all read that one rule, so two builds of one model
agree on every index.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._names import LABEL, POSITION

if TYPE_CHECKING:
    from collections.abc import Iterable


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
    #: The labels in the order they arrived, as the engine holds them: labels
    #: that arrived as ``String`` as an ``Enum`` in that order, so its physical
    #: code is a label's position, and any other in the type it arrived in.
    labels: pl.Series
    #: The cast of a column called [`name`][] into [`dtype`][], or ``None``
    #: where the labels are held as they arrived. Built once and shared by
    #: every table that carries the dimension, since building a cast to an
    #: ``Enum`` of many labels costs a fixed time per expression.
    encoding: pl.Expr | None

    @classmethod
    def of(cls, name: str, labels: pl.Series) -> Dimension:
        """The dimension *name* over *labels*, as [`tidy_sources`][specsolve._sources.tidy_sources] read them."""
        if labels.dtype != pl.String:
            return cls(name, labels, None)
        ordered = pl.Enum(labels)
        return cls(name, labels.cast(ordered), pl.col(name).cast(ordered))

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

    def position(self, column: pl.Expr) -> pl.Expr:
        """An encoded *column* of this dimension's labels as each label's position."""
        if self.labels.dtype == pl.Enum:
            return column.to_physical().cast(pl.Int64)
        if self.first is not None:
            return column.cast(pl.Int64) - self.first
        return column.replace_strict(
            self.labels, pl.int_range(self.size, dtype=pl.Int64, eager=True), return_dtype=pl.Int64
        )
