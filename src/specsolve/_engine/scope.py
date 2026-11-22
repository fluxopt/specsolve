"""The scope a query is built in: what each name stands for, and what a dimension means as a coordinate.

``variables`` is the build's own dict, not a copy: a constraint queried after
a variable is built has to see its table.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from specsolve._collect import collected
from specsolve._engine import positional
from specsolve._names import LABEL, POSITION, VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from mathspec import program
    from polars._typing import JoinStrategy, MaintainOrderJoin

    from specsolve._engine.attaching import AttachedSources
    from specsolve._engine.indices import Indexed
    from specsolve._engine.terms import Term


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
#: hold a table with one row and no columns.
UNIT = '__unit__'


def ordinal(dim: str) -> str:
    """The column carrying *dim*'s position in its declared order."""
    return f'__ord {dim}__'


@dataclass(frozen=True)
class Scope:
    """What a name resolves to here: the program for its entry, the data and the variable tables for its rows."""

    program: program.Program
    data: AttachedSources
    variables: Mapping[str, Indexed]
    #: Each column read by position, by what it holds and its name; [`positional`][specsolve._engine.positional] fills it.
    by_position: dict[tuple[str, str], pl.Series] = field(default_factory=dict)

    def absent(self, parameter: str) -> bool:
        """Whether *parameter* declares ``missing: absent`` and the data is short of a row, so a read of it carries a presence."""
        entry = self.program.parameters[parameter]
        if entry.missing != 'absent':
            return False
        return not self.data.covers(parameter, entry.dims)

    def product(self, dims: tuple[str, ...]) -> pl.LazyFrame:
        """Cross join of the dimension tables: labels and ordinals, nothing else.

        Each cross join keeps its left side's order, then its right's, so the
        product arrives row-major, which is index order, on either engine; the
        streaming engine otherwise picks which side it buffers from the
        tables' sizes, and the order with it. [`indices.table`][] verifies it.
        The empty product is one real row carrying only [`UNIT`][], so a
        ``where`` on a scalar entry has a row to filter.
        """
        out: pl.LazyFrame | None = None
        for d in dims:
            table = self.data.dimensions[d].select(pl.col(LABEL).alias(d), pl.col(POSITION).alias(ordinal(d)))
            out = table if out is None else out.join(table, how='cross', maintain_order='left_right')
        if out is None:
            return pl.LazyFrame({UNIT: [0]})
        return out

    def parameter_join(
        self,
        table: pl.LazyFrame,
        parameter: str,
        table_dims: tuple[str, ...],
        alias: str,
        subject: str,
        how: JoinStrategy = 'left',
        maintain_order: MaintainOrderJoin | None = None,
        *,
        by_position: bool = True,
    ) -> pl.LazyFrame:
        """Join *parameter* onto *table*, its value column renamed to *alias*.

        A parameter over a dimension *table* lacks would be reduced over it, so
        that is refused, naming *subject*. A left join of a parameter over no
        dimensions keeps every row of *table*, reading null where the parameter
        has no row, which a cross join with its empty table would not. A
        parameter with a row at every coordinate, and enough of them to be
        [worth reading][specsolve._engine.positional.worth_reading], is read at
        each row's [`position`][Scope.position] instead, which is what a left
        or an inner join of it finds, unless *by_position* is false.
        """
        entry = self.program.parameters[parameter]
        assert not set(entry.dims) - set(table_dims), f'{subject} has dimensions outside {list(table_dims)}'
        reads = by_position and how in ('left', 'inner') and positional.worth_reading(self, parameter)
        position = self.position(entry.dims) if reads else None
        if position is not None:
            return table.with_columns(pl.lit(positional.column(self, parameter)).gather(position).alias(alias))
        values = self.data.parameters[parameter].rename({VALUE: alias})
        if not entry.dims and how == 'left':
            null = pl.LazyFrame({alias: [None]}, schema={alias: values.collect_schema()[alias]})
            values = pl.concat([values, null]).head(1)
        return join_on(table, values, entry.dims, how, maintain_order)

    def position(self, dims: tuple[str, ...]) -> pl.Expr | None:
        """A table row's [`row_major`][Scope.row_major] position over *dims*, read off its dimension columns alone.

        ``None`` where a dimension's labels are neither an ``Enum`` nor a run of
        consecutive integers: their ordinals are then a lookup per row, which
        is the join this is offered in place of.
        """
        if not all(self.data.is_enum_encoded(d) or d in self.data.consecutive for d in dims):
            return None
        return self.row_major(dims, self.ordinal_of)

    def row_major(self, dims: tuple[str, ...], ordinals: Callable[[str], pl.Expr]) -> pl.Expr:
        """A coordinate's row-major position in the declared product of *dims*, contiguous over the full product.

        A solver index, a bound's slot and a set's position all read this one rule, so two
        builds of one model agree on every index. *ordinals* reads a dimension's
        ordinal off the table in hand.
        """
        position: pl.Expr = pl.lit(0, dtype=pl.Int64)
        for d in dims:
            position = position * self.data.cardinality[d] + ordinals(d)
        return position.cast(pl.Int64)

    def ordinal_of(self, dim: str) -> pl.Expr:
        """A *dim* value column as that dimension's ordinal.

        Attaching encodes an ``Enum`` in ordinal order, so its physical code is the
        ordinal, and a run of consecutive integers is its own ordinal less its first.
        """
        column = pl.col(dim)
        if self.data.is_enum_encoded(dim):
            return column.to_physical().cast(pl.Int64)
        if (first := self.data.consecutive.get(dim)) is not None:
            return column.cast(pl.Int64) - first
        ordinals = self.data.dimensions[dim].pipe(collected)
        return column.replace_strict(ordinals.get_column(LABEL), ordinals.get_column(POSITION), return_dtype=pl.Int64)

    def widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dimension in *want*, saying the same thing.

        A presence table is silent about the dimensions it omits, which reads as
        present at all of them.
        """
        return self.spread(presence, [d for d in want if d not in have]).select(*want)

    def spread(self, table: pl.LazyFrame, dims: Iterable[str]) -> pl.LazyFrame:
        """*table* repeated at every label of each of *dims*, which it does not carry."""
        for d in dims:
            table = table.join(self.data.dimensions[d].select(pl.col(LABEL).alias(d)), how='cross')
        return table

    def spanned(self, terms: Sequence[Term]) -> tuple[str, ...]:
        """The dimensions *terms* carry between them, in the order the file declares them."""
        return self.in_dimension_order(d for p in terms for d in p.dims)

    def in_dimension_order(self, dims: Iterable[str]) -> tuple[str, ...]:
        """*dims* in the order the file declares them, duplicates dropped.

        Two tables keyed by the same dimensions in two orders would join on a key
        written twice.
        """
        wanted = set(dims)
        return tuple(d for d in self.program.dimensions if d in wanted)
