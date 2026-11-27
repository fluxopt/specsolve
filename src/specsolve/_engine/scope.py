"""The scope a query is built in: what each name stands for, and what a dimension means as a coordinate.

``variables`` is the build's own dict, not a copy: a constraint queried after
a variable is built has to see its table.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve._dimensions import row_major
from specsolve._names import LABEL, POSITION, VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from mathspec import program
    from polars._typing import JoinStrategy, MaintainOrderJoin

    from specsolve._engine.attaching import AttachedSources
    from specsolve._engine.indices import Indexed
    from specsolve._engine.terms import Term


def missing_value(declared: program.Program, name: str) -> float | bool | None:
    """The value a missing row of the parameter *name* reads as, or ``None`` under ``refused`` or ``absent``.

    ``None`` too for a name that is no parameter, such as a variable a mask tests.
    """
    declaration = declared.parameters.get(name)
    if declaration is None or isinstance(declaration.missing, str):
        return None
    return declaration.missing


def reads_as_zero(missing: program.ParameterMissing | None) -> bool:
    """Whether a missing row reads as zero, so a table short of rows is left sparse rather than filled.

    Every term and bound reads a sparse zero row as no contribution, so only a
    comparison has to see the zero, which [`missing_value`][] gives it.
    """
    return not isinstance(missing, str) and missing is not None and missing == 0


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


def position_column(dim: str) -> str:
    """The column carrying *dim*'s position in its declared order."""
    return f'__position {dim}__'


@dataclass(frozen=True)
class Scope:
    """What a name resolves to here: the program for its entry, the data and the variable tables for its rows."""

    program: program.Program
    data: AttachedSources
    variables: Mapping[str, Indexed]

    def absent(self, parameter: str) -> bool:
        """Whether *parameter* declares ``missing: absent`` and the data is short of a row, so a read of it carries a presence."""
        entry = self.program.parameters[parameter]
        if entry.missing != 'absent':
            return False
        return self.data.parameter_rows[parameter] < math.prod(self.data.dimensions[d].size for d in entry.dims)

    def product(self, dims: tuple[str, ...]) -> pl.LazyFrame:
        """Cross join of the dimension tables: labels and positions, nothing else.

        Each cross join keeps its left side's order, then its right's, so the
        product arrives row-major, which is index order, on either engine; the
        streaming engine otherwise picks which side it buffers from the
        tables' sizes, and the order with it. [`indices.table`][] verifies it.
        The empty product is one real row carrying only [`UNIT`][], so a
        ``where`` on a scalar entry has a row to filter.
        """
        out: pl.LazyFrame | None = None
        for d in dims:
            table = self.data.dimensions[d].table.select(
                pl.col(LABEL).alias(d), pl.col(POSITION).alias(position_column(d))
            )
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
    ) -> pl.LazyFrame:
        """Join *parameter* onto *table*, its value column renamed to *alias*.

        A parameter over a dimension *table* lacks would be reduced over it, so
        that is refused, naming *subject*. A left join of a parameter over no
        dimensions keeps every row of *table*, reading null where the parameter
        has no row, which a cross join with its empty table would not.
        """
        entry = self.program.parameters[parameter]
        assert not set(entry.dims) - set(table_dims), f'{subject} has dimensions outside {list(table_dims)}'
        values = self.data.parameters[parameter].rename({VALUE: alias})
        if not entry.dims and how == 'left':
            null = pl.LazyFrame({alias: [None]}, schema={alias: values.collect_schema()[alias]})
            values = pl.concat([values, null]).head(1)
        return join_on(table, values, entry.dims, how, maintain_order)

    def row_major(self, dims: tuple[str, ...], positions: Callable[[str], pl.Expr]) -> pl.Expr:
        """A coordinate's row-major position in the declared product of *dims*, contiguous over the full product.

        A solver index, a bound's slot and a set's position all read this one rule, so two
        builds of one model agree on every index. *positions* reads a dimension's
        position off the table in hand.
        """
        return row_major((self.data.dimensions[d].size for d in dims), (positions(d) for d in dims))

    def position_of(self, dim: str) -> pl.Expr:
        """A *dim* value column as that dimension's position.

        Attaching encodes an ``Enum`` in label order, so its physical code is the
        position, and a run of consecutive integers is its own position less its first.
        """
        return self.data.dimensions[dim].position(pl.col(dim))

    def widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dimension in *want*, saying the same thing.

        A presence table is silent about the dimensions it omits, which reads as
        present at all of them.
        """
        return self.spread(presence, [d for d in want if d not in have]).select(*want)

    def spread(self, table: pl.LazyFrame, dims: Iterable[str]) -> pl.LazyFrame:
        """*table* repeated at every label of each of *dims*, which it does not carry."""
        for d in dims:
            table = table.join(self.data.dimensions[d].table.select(pl.col(LABEL).alias(d)), how='cross')
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
