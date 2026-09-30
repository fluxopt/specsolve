"""The family base: what every format writes the same way. It renders no format of its own."""

from __future__ import annotations

from dataclasses import dataclass
from typing import IO

import polars as pl

__all__ = ['NUMBERED', 'Names', 'chunk_key', 'digits', 'number', 'sink']


@dataclass(frozen=True)
class Names:
    """What a file calls each column and row.

    ``None`` numbers them, ``x7`` and ``c3``. A series is indexed by the
    solver's own column or row index, so a writer looks a name up rather than
    joining for it.
    """

    columns: pl.Series | None = None
    rows: pl.Series | None = None

    def column(self, col: pl.Expr) -> pl.Expr:
        """The name of the column *col* indexes."""
        return _named(self.columns, 'x', col)

    def row(self, row: pl.Expr) -> pl.Expr:
        """The name of the row *row* indexes."""
        return _named(self.rows, 'c', row)


#: The names a file carries unless the caller asks for the declared ones.
NUMBERED = Names()


def _named(names: pl.Series | None, prefix: str, index: pl.Expr) -> pl.Expr:
    if names is None:
        return pl.concat_str(pl.lit(prefix), digits(index))
    return pl.lit(names).gather(index)


def sink(frame: pl.LazyFrame, f: IO[bytes]) -> None:
    """Append a one-column frame to *f*, one raw line per row.

    Polars writes through the caller's handle, so an ``f.write()`` between two
    sinks lands between them. ``maintain_order`` is stated because its default
    is documented as unstable.
    """
    frame.sink_csv(f, include_header=False, quote_style='never', maintain_order=True)


def chunk_key(axis: pl.Expr, lo: int, slots: int, within: pl.Expr) -> pl.Expr:
    """The one sort key of a chunked section: ``slots`` consecutive keys per *axis* value.

    Chunk-relative, so the product is bounded by a chunk's height rather than
    the model's.
    """
    return ((axis - lo) * slots + within).alias('key')


def number(value: pl.Expr) -> pl.Expr:
    """A float as text, shortest round-trip, so a solver reads back the double the engine computed."""
    return value.cast(pl.String)


def digits(value: pl.Expr) -> pl.Expr:
    """An index as text — never in scientific notation, whatever its size."""
    return value.cast(pl.Int64).cast(pl.String)
