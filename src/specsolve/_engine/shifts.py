"""Shifts along one dimension's own order: ``shift``, and ``sum_back``, a sum of shifts.

``shift`` is a pointwise remap of the dimension through its positions and
``sum_back`` a one-to-many one. They share the position arithmetic and the
question of the edge, where the walk runs out of dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import polars as pl
from mathspec import program

from specsolve._collect import collected
from specsolve._engine.relations import GROUP_RANK, GROUP_SIZE, Grouping
from specsolve._engine.scope import join_on
from specsolve._engine.terms import Presence, Term, refuse_a_term_without_the_dims
from specsolve._names import CONSTANT, LABEL, VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from specsolve._engine.scope import Scope


#: Scratch columns.
_OFFSET = '__offset'
_LAG = '__lag'
_WIDTH = '__width'
_POSITION_IN = '__position in__'
_POSITION_OUT = '__position out__'


@dataclass(frozen=True)
class _Order:
    """A grouping as an operator walks along it: the two keyed sides of the remap.

    ``incoming`` and ``outgoing`` read the same rank column, so a change to how
    a walk ranks moves both sides.
    """

    #: The groups the walk stays inside — the whole dimension where no ``by=`` was written.
    grouping: Grouping
    incoming: pl.LazyFrame
    outgoing: pl.LazyFrame

    @classmethod
    def of(cls, scope: Scope, dimension: str, partition: program.Partition | None) -> _Order:
        """Rank *dimension* inside each group of *partition*, or along the whole of it."""
        grouping = Grouping.whole(scope.data, dimension) if partition is None else Grouping.of(scope.data, partition)
        incoming = grouping.table.select(
            pl.col(LABEL).alias(dimension), pl.col(GROUP_RANK).alias(_POSITION_IN), *grouping.key, pl.col(GROUP_SIZE)
        )
        outgoing = grouping.table.select(
            pl.col(LABEL).alias(dimension), pl.col(GROUP_RANK).alias(_POSITION_OUT), *grouping.key
        )
        return cls(grouping, incoming, outgoing)

    def remap(
        self,
        source: pl.LazyFrame,
        carried: Sequence[str],
        dims: tuple[str, ...],
        *,
        moved: pl.Expr,
        prepared: Callable[[pl.LazyFrame], pl.LazyFrame],
    ) -> pl.LazyFrame:
        """*source* with the walked dimension moved by *moved*.

        *dims* is the caller's, since a presence table need not carry the
        term's. *prepared* adds the operator's extra join between the two keyed sides.
        """
        dimension = self.grouping.dimension
        kept = [d for d in dims if d != dimension]
        walked = prepared(source.join(self.incoming, on=list(self.grouping.keys), how='inner').drop(dimension))
        return (
            walked.with_columns(moved.alias(_POSITION_OUT))
            .join(self.outgoing, on=[_POSITION_OUT, *self.grouping.key], how='inner')
            .select(*kept, dimension, *carried)
        )


def walked_dims(scope: Scope, s: program.Translate | program.WindowSum) -> tuple[str, ...]:
    """The dimensions a walk keys each row of its operand by.

    The dimension walked, a partition's joined dimensions, and a named offset's
    or width's own dimensions, less the ones a partition groups into, which it reads
    under the group column ([`_named_amount`][]).
    """
    amount = s.offset if isinstance(s, program.Translate) else s.width
    named = scope.program.parameters[amount].dims if isinstance(amount, str) else ()
    partition = s.partition
    joined = partition.joined_dims if partition is not None else ()
    grouped = {partition.dim(role) for role in partition.grouped} if partition is not None else set()
    return tuple(dict.fromkeys((s.along, *joined, *(d for d in named if d not in grouped))))


def translate_rows(
    scope: Scope, table: pl.LazyFrame, dims: tuple[str, ...], carried: Sequence[str], along: str, offset: int
) -> pl.LazyFrame:
    """*table*'s rows moved *offset* positions along *along*, the end the move vacates dropped.

    The condition form of [`translate_term`][]: a missing row already reads as false.
    """
    order = _Order.of(scope, along, None)
    return order.remap(table, carried, dims, moved=pl.col(_POSITION_IN) + offset, prepared=lambda f: f)


def window_term(scope: Scope, p: Term, s: program.WindowSum, context: str) -> Term:
    """A one-to-many remap of the dimension: a row at *o* contributes at every ``o + lag`` inside the window.

    The lag table is built to the widest window the data asks for. A window
    is a sum, so the operand's presence does not travel: an absent slot leaves
    the sum one term fewer, and a window of only absent slots is an empty sum
    whose row stands. The absence it reports is a coordinate with no window at
    all: one a partition places in no group, or one a ``missing: absent`` width
    has no row for ([`_widths_defined`][]).
    """
    if s.along not in p.dims:
        refuse_a_term_without_the_dims(p, [s.along], context, f'sum_back(along={s.along!r})')
    order = _Order.of(scope, s.along, s.partition)

    width_name = s.width if isinstance(s.width, str) else None
    if width_name is not None:
        widest = int(scope.data.parameters[width_name].select(pl.col(VALUE).max()).pipe(collected).item() or 0)
    else:
        assert not isinstance(s.width, str)
        widest = s.width
    lags = pl.LazyFrame({_LAG: pl.Series(range(min(widest, scope.data.cardinality[s.along])), dtype=pl.Int64)})

    moved = pl.col(_POSITION_IN) + pl.col(_LAG)
    if s.wrap:
        moved = moved % pl.col(GROUP_SIZE)

    def lagged(table: pl.LazyFrame) -> pl.LazyFrame:
        """Every reachable lag beside each row — a named width keeps only the lags its entity reaches."""
        table = table.join(lags, how='cross')
        if width_name is None:
            return table
        widths, keys = _named_amount(scope, order, width_name, _WIDTH)
        return join_on(table, widths, keys, 'inner').filter(pl.col(_LAG) < pl.col(_WIDTH))

    table = order.remap(p.table, p.carried, p.dims, moved=moved, prepared=lagged)
    if width_name is not None and scope.absent(width_name):
        return replace(p, body=table, presences=(_widths_defined(scope, order, width_name),))
    unplaced = (Presence(order.grouping.placed(), order.grouping.keys),) if order.grouping.partial else ()
    return replace(p, body=table, presences=unplaced)


def _widths_defined(scope: Scope, order: _Order, name: str) -> Presence:
    """The coordinates a width has a value for, as a presence: a window elsewhere is undefined, so absent.

    Read off the order's table, which holds only the coordinates a partition
    places, and through the group column for a per-group width, which no
    table carries.
    """
    grouping = order.grouping
    widths, keys = _named_amount(scope, order, name, _WIDTH)
    own = [key for key in keys if key not in grouping.key]
    table = join_on(grouping.table, widths, [key for key in keys if key in grouping.key], 'inner')
    coordinates = table.select(pl.col(LABEL).alias(grouping.dimension), *grouping.joined, *own).unique()
    return Presence(coordinates, (*grouping.keys, *own))


def translate_term(scope: Scope, p: Term, s: program.Translate, context: str, *, over_a_variable: bool) -> Term:
    """A pointwise remap of the dimension: a row at *o* contributes at ``o + offset``.

    Every fill over a constant is written, ``0`` included, so the slot has a
    value. Over a linear term a fill writes nothing: the vacated slot contributes no
    term, and the program admits no nonzero fill over a variable.
    *over_a_variable* says whether the whole operand carries a variable, which
    decides where a fill stands ([`_Edge.vacated`][]).
    """
    if s.along not in p.dims:
        refuse_a_term_without_the_dims(p, [s.along], context, f'shift(along={s.along!r})')
    others = [d for d in p.dims if d != s.along]
    order = _Order.of(scope, s.along, s.partition)
    edge = _Edge.of(scope, order, s)

    named_offset = isinstance(s.offset, str)
    if named_offset:
        moved = pl.col(_POSITION_IN) + pl.col(_OFFSET)
    else:
        assert not isinstance(s.offset, str)
        moved = pl.col(_POSITION_IN) + s.offset
    if s.wrap:
        moved = (moved % pl.col(GROUP_SIZE) + pl.col(GROUP_SIZE)) % pl.col(GROUP_SIZE)

    def offsetted(table: pl.LazyFrame) -> pl.LazyFrame:
        """A per-entity offset is one more equi-join, on keys *table* already carries."""
        if edge.offsets is None:
            return table
        offsets, keys = edge.offsets
        return join_on(table, offsets, keys, 'inner')

    remap = partial(order.remap, moved=moved, prepared=offsetted)

    def travelled_presences() -> tuple[Presence, ...]:
        """Where the variable exists after the shift, and what keys it.

        An existing presence goes through the same map as the rows, the vacated
        positions going back in under a fill. An operand with no presence gets
        one under an acyclic edge, or else its row would survive with the term
        quietly gone; under a wrap or fill only a coordinate in no group is
        absent, and one a named offset has no row for.
        """
        if not p.presences:
            if edge.offsets is not None:
                return (Presence(edge.coordinates(vacated=None), edge.keys),)
            if s.wrap or s.fill is not None:
                return (Presence(order.grouping.placed(), order.grouping.keys),) if order.grouping.partial else ()
            return (Presence(edge.coordinates(vacated=False), edge.keys),)
        return tuple(travelled(x) for x in p.presences)

    def travelled(presence: Presence) -> Presence:
        source, joined_on = presence.table, presence.joined_on
        if joined_on is not None and not set(edge.keys).issubset(joined_on):
            source, joined_on = scope.widen(source, joined_on, p.dims), None
        join_columns = p.dims if joined_on is None else joined_on
        moved_presence = remap(source, [], join_columns)
        if s.wrap or s.fill is None:
            return Presence(moved_presence, joined_on)
        vacated = edge.vacated(scope, join_columns, beside=source if over_a_variable else None)
        return Presence(pl.concat([moved_presence, vacated], how='vertical_relaxed').unique(), joined_on)

    table = remap(p.table, p.carried, p.dims)
    if not s.wrap and s.fill is not None and p.kind == 'constant':
        table = pl.concat([table, edge.filled(scope, others, s.fill)], how='vertical_relaxed')
    return replace(p, body=table, presences=travelled_presences())


@dataclass(frozen=True)
class _Edge:
    """The edge of an acyclic shift along an [`_Order`][]: which coordinates it vacates, and what keys them.

    Keyed by the translated dimension, a named offset's own dimensions and the
    partition's joined dimensions, each once: under either, whether a coordinate is
    the edge depends on the rest of the key. A grouped dimension is not a key.
    """

    order: _Order
    shift: program.Translate
    #: The dimensions a per-entity offset varies over — empty where it is a number.
    offset_dims: tuple[str, ...]
    #: A named offset's values and the keys the order's table reads them by, or ``None`` for a number.
    offsets: tuple[pl.LazyFrame, list[str]] | None

    @classmethod
    def of(cls, scope: Scope, order: _Order, s: program.Translate) -> _Edge:
        if not isinstance(s.offset, str):
            return cls(order, s, (), None)
        dims = scope.program.parameters[s.offset].dims
        offset_dims = tuple(d for d in dims if order.grouping.column_of(d) is None)
        return cls(order, s, offset_dims, _named_amount(scope, order, s.offset, _OFFSET))

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.order.grouping.dimension, *self.offset_dims, *self.order.grouping.joined)))

    def coordinates(self, *, vacated: bool | None) -> pl.LazyFrame:
        """The coordinates the shift vacates, keeps, or, under ``None``, both, under [`keys`][].

        Exact complements, so a fill and the presence it implies agree on the
        edge. Under a partition the edge is each group's; a coordinate in no
        group is in neither, and nor is one a named offset has no row for.
        """
        grouping, s = self.order.grouping, self.shift
        table, position, span = grouping.table, pl.col(GROUP_RANK), pl.col(GROUP_SIZE)
        if self.offsets is not None:
            offsets, keys = self.offsets
            on = [key for key in keys if key in grouping.key]
            table = join_on(table, offsets, on, 'inner')
            offset = pl.col(_OFFSET)
        else:
            assert not isinstance(s.offset, str)
            offset = pl.lit(s.offset, dtype=pl.Int64)
        source = position - offset
        reaches = (source % span + span) % span if s.wrap else source
        outside = (reaches < 0) | (reaches >= span)
        keyed = [d for d in self.keys if d != s.along]
        if vacated is not None:
            table = table.filter(outside if vacated else ~outside)
        return table.select(pl.col(LABEL).alias(s.along), *keyed)

    def filled(self, scope: Scope, others: list[str], fill: float) -> pl.LazyFrame:
        """``(dims…, CONSTANT=fill)`` at every coordinate the shift vacated, dense over *others*."""
        edge = self.vacated(scope, (*others, self.shift.along), beside=None)
        return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias(CONSTANT))

    def vacated(self, scope: Scope, keys: tuple[str, ...], *, beside: pl.LazyFrame | None) -> pl.LazyFrame:
        """The coordinates the shift leaves with nothing to move in, under *keys*.

        Back in a presence set they are present with no term, so the row
        survives. *beside*, a presence over *keys*, keeps only the combinations
        of the other keys it has rows for, so a line along the dimension that
        the operand's variable lacks whole stays absent; ``None`` keeps every
        one, as an operand with no variable reads. The columns come in the
        order [`_Order.remap`][] writes them, for the concat with the moved
        presence.
        """
        along = self.shift.along
        others = [d for d in keys if d != along]
        edge = self.coordinates(vacated=True)
        if beside is not None and others:
            rows = beside.select(*others).unique()
            edge = join_on(rows, edge, [d for d in self.keys if d in others], 'inner')
        else:
            edge = scope.spread(edge, [d for d in others if d not in self.keys])
        return edge.select(*others, along)


def _named_amount(scope: Scope, order: _Order, name: str, alias: str) -> tuple[pl.LazyFrame, list[str]]:
    """A named offset's or width's values, and the keys a table reads them by.

    A per-group amount is read under the group column, since no table carries
    the grouped dimension.
    """
    dims = scope.program.parameters[name].dims
    keys = [order.grouping.column_of(d) or d for d in dims]
    table = scope.data.parameters[name].select(
        *(pl.col(d).alias(key) for d, key in zip(dims, keys, strict=True)),
        pl.col(VALUE).cast(pl.Int64).alias(alias),
    )
    return table, keys
