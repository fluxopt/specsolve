"""The scope a query is compiled in: what each name stands for, and what a dimension means as a coordinate.

``variables`` is the build's own dict, not a copy: a constraint compiled after
a variable is built has to see its frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve.relational.collect import collected

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from mathspec import program
    from polars._typing import JoinStrategy, MaintainOrderJoin

    from specsolve.relational.engine.attaching import AttachedSources
    from specsolve.relational.engine.labels import Labelled
    from specsolve.relational.engine.pieces import Piece


def join_on(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    dims: Sequence[str],
    how: JoinStrategy,
    maintain_order: MaintainOrderJoin | None = None,
) -> pl.LazyFrame:
    """``left.join(right)`` keyed by *dims* — a cross join where there are none."""
    if dims:
        return left.join(right, on=list(dims), how=how, maintain_order=maintain_order)
    return left.join(right, how='cross', maintain_order=maintain_order)


#: Carries the single row of the empty coordinate product, since polars cannot
#: hold a frame with one row and no columns.
UNIT = '__unit__'


def ordinal(dim: str) -> str:
    """The frame column carrying *dim*'s position in its declared order."""
    return f'__ord {dim}__'


@dataclass(frozen=True)
class Scope:
    """What a name resolves to here: the program for its declaration, the data and the variable frames for its frame."""

    program: program.Program
    data: AttachedSources
    variables: Mapping[str, Labelled]

    def product(self, dims: tuple[str, ...]) -> pl.LazyFrame:
        """Cross join of the dim tables: labels and ordinals, nothing else.

        Each cross join keeps its left side's order, then its right's, so the
        product arrives row-major, which is label order, on either engine; the
        streaming engine otherwise picks which side it buffers from the
        tables' sizes, and the order with it. [`labels.frame`][] verifies it.
        The empty product is one real row carrying only [`UNIT`][], so a
        ``where`` on a scalar declaration has a row to filter.
        """
        out: pl.LazyFrame | None = None
        for d in dims:
            table = self.data.dimensions[d].select(pl.col('val').alias(d), pl.col('ord').alias(ordinal(d)))
            out = table if out is None else out.join(table, how='cross', maintain_order='left_right')
        if out is None:
            return pl.LazyFrame({UNIT: [0]})
        return out

    def parameter_join(
        self,
        frame: pl.LazyFrame,
        param: str,
        frame_dims: tuple[str, ...],
        alias: str,
        subject: str,
        how: JoinStrategy = 'left',
        maintain_order: MaintainOrderJoin | None = None,
    ) -> pl.LazyFrame:
        """Join *param* onto *frame*, its value column renamed to *alias*.

        A parameter over a dim the frame lacks would be reduced over it, so that is
        refused, naming *subject*.
        """
        declaration = self.program.parameters[param]
        assert not set(declaration.dims) - set(frame_dims), (
            f'{subject} has dims outside the frame dims {list(frame_dims)}'
        )
        table = self.data.parameters[param].rename({'value': alias})
        return join_on(frame, table, declaration.dims, how, maintain_order)

    def row_major(self, dims: tuple[str, ...], ordinals: Callable[[str], pl.Expr]) -> pl.Expr:
        """A coordinate's row-major position in the declared product of *dims*, dense over the full product.

        A label, a bound's slot and a set's position all read this one rule, so two
        builds of one model agree on every index. *ordinals* reads a dim's ordinal
        off the frame in hand.
        """
        position: pl.Expr = pl.lit(0, dtype=pl.Int64)
        for d in dims:
            position = position * self.data.cardinality[d] + ordinals(d)
        return position.cast(pl.Int64)

    def ordinal_of(self, dim: str) -> pl.Expr:
        """A *dim* value column as that dimension's ordinal.

        Attaching encodes an ``Enum`` in ordinal order, so its physical code is the
        ordinal.
        """
        column = pl.col(dim)
        if self.data.is_enum_encoded(dim):
            return column.to_physical().cast(pl.Int64)
        labels = self.data.dimensions[dim].select('val').pipe(collected)['val']
        return column.replace_strict({value: at for at, value in enumerate(labels)}, return_dtype=pl.Int64)

    def widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dim in *want*, saying the same thing.

        A presence frame is silent about the dims it omits, which reads as present
        at all of them.
        """
        return self.spread(presence, [d for d in want if d not in have]).select(*want)

    def spread(self, frame: pl.LazyFrame, dims: Iterable[str]) -> pl.LazyFrame:
        """*frame* repeated at every label of each of *dims*, which it does not carry."""
        for d in dims:
            frame = frame.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return frame

    def spanned(self, pieces: Sequence[Piece]) -> tuple[str, ...]:
        """The dims *pieces* carry between them, in declaration order."""
        return self.in_declaration_order(d for p in pieces for d in p.dims)

    def in_declaration_order(self, dims: Iterable[str]) -> tuple[str, ...]:
        """*dims* in the order the file declares them, duplicates dropped.

        Two frames keyed by the same dims in two orders would join on a key written
        twice.
        """
        wanted = set(dims)
        return tuple(d for d in self.program.dimensions if d in wanted)
