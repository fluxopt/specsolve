"""Shifts along one dimension's own order: ``shift``, and ``sum_back``, a sum of shifts.

``shift`` is a pointwise remap of the dimension through its ordinal and
``sum_back`` a one-to-many one. They share the ordinal arithmetic and the
question of the edge, where the walk runs out of dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING

import polars as pl

from specsolve.relational.collect import collected
from specsolve.relational.engine.pieces import Piece, Presence, refuse_a_piece_without_the_dims
from specsolve.relational.engine.relations import GROUP_RANK, GROUP_SIZE, Grouping
from specsolve.relational.engine.scope import join_on
from specsolve.relational.names import VALUE

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from mathspec import program

    from specsolve.relational.engine.scope import Scope


#: Scratch columns.
_OFFSET = '__offset'
_LAG = '__lag'
_WIDTH = '__width'
_ORD_IN = '__ord in__'
_ORD_OUT = '__ord out__'


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
            pl.col('val').alias(dimension), pl.col(GROUP_RANK).alias(_ORD_IN), *grouping.key, pl.col(GROUP_SIZE)
        )
        outgoing = grouping.table.select(
            pl.col('val').alias(dimension), pl.col(GROUP_RANK).alias(_ORD_OUT), *grouping.key
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

        *dims* is the caller's, since a presence frame need not carry the
        piece's. *prepared* adds the operator's extra join between the two keyed sides.
        """
        dimension = self.grouping.dimension
        kept = [d for d in dims if d != dimension]
        walked = prepared(source.join(self.incoming, on=list(self.grouping.keys), how='inner').drop(dimension))
        return (
            walked.with_columns(moved.alias(_ORD_OUT))
            .join(self.outgoing, on=[_ORD_OUT, *self.grouping.key], how='inner')
            .select(*kept, dimension, *carried)
        )


def translate_rows(
    scope: Scope, frame: pl.LazyFrame, dims: tuple[str, ...], carried: Sequence[str], along: str, offset: int
) -> pl.LazyFrame:
    """*frame*'s rows moved *offset* positions along *along*, the end the move vacates dropped.

    The predicate form of [`translate_piece`][]: a missing row already reads as false.
    """
    order = _Order.of(scope, along, None)
    return order.remap(frame, carried, dims, moved=pl.col(_ORD_IN) + offset, prepared=lambda f: f)


def window_piece(scope: Scope, p: Piece, s: program.WindowSum, context: str) -> Piece:
    """A one-to-many remap of the dimension: a row at *o* contributes at every ``o + lag`` inside the window.

    The lag table is built to the widest window the data asks for. A window
    vacates nothing, so an operand with no presence gains one only under a
    partition, for the coordinates in no group.
    """
    if s.along not in p.dims:
        refuse_a_piece_without_the_dims(p, [s.along], context, f'sum_back(along={s.along!r})')
    order = _Order.of(scope, s.along, s.partition)

    width_name = s.width if isinstance(s.width, str) else None
    if width_name is not None:
        widest = int(scope.data.parameters[width_name].select(pl.col(VALUE).max()).pipe(collected).item() or 0)
    else:
        assert not isinstance(s.width, str)
        widest = s.width
    lags = pl.LazyFrame({_LAG: pl.Series(range(min(widest, scope.data.cardinality[s.along])), dtype=pl.Int64)})

    moved = pl.col(_ORD_IN) + pl.col(_LAG)
    if s.wrap:
        moved = moved % pl.col(GROUP_SIZE)

    def lagged(frame: pl.LazyFrame) -> pl.LazyFrame:
        """Every reachable lag beside each row — a named width keeps only the lags its entity reaches."""
        frame = frame.join(lags, how='cross')
        if width_name is None:
            return frame
        widths, keys = _named_amount(scope, order, width_name, _WIDTH)
        return frame.join(widths, on=keys, how='inner').filter(pl.col(_LAG) < pl.col(_WIDTH))

    remap = partial(order.remap, moved=moved, prepared=lagged)

    def travelled(presence: Presence) -> Presence:
        keyed_by, source = presence.keyed_by, presence.frame
        if keyed_by is not None and not set(order.grouping.keys).issubset(keyed_by):
            source, keyed_by = scope.widen(source, keyed_by, p.dims), None
        return Presence(remap(source, [], p.dims if keyed_by is None else keyed_by).unique(), keyed_by)

    frame = remap(p.frame, p.carried, p.dims)
    if not p.presences and order.grouping.partial:
        return replace(p, frame=frame, presences=(Presence(order.grouping.placed(), order.grouping.keys),))
    return replace(p, frame=frame, presences=tuple(travelled(x) for x in p.presences))


def translate_piece(scope: Scope, p: Piece, s: program.Translate, context: str) -> Piece:
    """A pointwise remap of the dimension: a row at *o* contributes at ``o + offset``.

    Every fill over a constant is written, ``0`` included, so the slot has a
    value. Over a term a fill writes nothing: the vacated slot contributes no
    term, and lowering refuses every nonzero fill over a variable.
    """
    if s.along not in p.dims:
        refuse_a_piece_without_the_dims(p, [s.along], context, f'shift(along={s.along!r})')
    others = [d for d in p.dims if d != s.along]
    order = _Order.of(scope, s.along, s.partition)
    edge = _Edge.of(scope, order, s)

    named_offset = isinstance(s.offset, str)
    if named_offset:
        moved = pl.col(_ORD_IN) + pl.col(_OFFSET)
    else:
        assert not isinstance(s.offset, str)
        moved = pl.col(_ORD_IN) + s.offset
    if s.wrap:
        moved = (moved % pl.col(GROUP_SIZE) + pl.col(GROUP_SIZE)) % pl.col(GROUP_SIZE)

    def offsetted(frame: pl.LazyFrame) -> pl.LazyFrame:
        """A per-entity offset is one more equi-join, on keys the frame already carries."""
        if edge.offsets is None:
            return frame
        offsets, keys = edge.offsets
        return frame.join(offsets, on=keys, how='inner')

    remap = partial(order.remap, moved=moved, prepared=offsetted)

    def travelled_presences() -> tuple[Presence, ...]:
        """Where the variable exists after the shift, and what keys it.

        An existing presence goes through the same map as the rows, the vacated
        positions going back in under a fill. An operand with no presence gets
        one under an acyclic edge, or else its row would survive with the term
        quietly gone; under a wrap or fill only a coordinate in no group is absent.
        """
        if not p.presences:
            if s.wrap or s.fill is not None:
                return (Presence(order.grouping.placed(), order.grouping.keys),) if order.grouping.partial else ()
            return (Presence(edge.coordinates(vacated=False), edge.keys),)
        return tuple(travelled(x) for x in p.presences)

    def travelled(presence: Presence) -> Presence:
        source, keyed_by = presence.frame, presence.keyed_by
        if keyed_by is not None and not set(edge.keys).issubset(keyed_by):
            source, keyed_by = scope.widen(source, keyed_by, p.dims), None
        moved_presence = remap(source, [], p.dims if keyed_by is None else keyed_by)
        if s.wrap or s.fill is None:
            return Presence(moved_presence, keyed_by)
        vacated = edge.vacated_of(scope, presence, p.dims)
        return Presence(pl.concat([moved_presence, vacated], how='vertical_relaxed').unique())

    frame = remap(p.frame, p.carried, p.dims)
    if not s.wrap and s.fill is not None and p.kind == 'const':
        frame = pl.concat([frame, edge.filled(scope, others, s.fill)], how='vertical_relaxed')
    return replace(p, frame=frame, presences=travelled_presences())


@dataclass(frozen=True)
class _Edge:
    """The edge of an acyclic shift along an [`_Order`][]: which coordinates it vacates, and what keys them.

    Keyed by the translated dimension, a named offset's own dims and the
    partition's joined dims, each once: under either, whether a coordinate is
    the edge depends on the rest of the key. A grouped dimension is not a key.
    """

    order: _Order
    shift: program.Translate
    #: The dims a per-entity offset varies over — empty where it is a number.
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

    def coordinates(self, *, vacated: bool) -> pl.LazyFrame:
        """The coordinates the shift vacates, or keeps, under [`keys`][].

        Exact complements, so a fill and the presence it implies agree on the
        edge. Under a partition the edge is each group's; a coordinate in no
        group is in neither.
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
        return table.filter(outside if vacated else ~outside).select(pl.col('val').alias(s.along), *keyed)

    def filled(self, scope: Scope, others: list[str], fill: float) -> pl.LazyFrame:
        """``(dims…, cval=fill)`` at every coordinate the shift vacated, dense over *others*."""
        edge = scope.spread(self.coordinates(vacated=True), [d for d in others if d not in self.keys])
        return edge.with_columns(pl.lit(fill, dtype=pl.Float64).alias('cval')).select(*others, self.shift.along, 'cval')

    def vacated_of(self, scope: Scope, presence: Presence, dims: tuple[str, ...]) -> pl.LazyFrame:
        """The edge positions ``shift`` leaves with nothing to move in, for one presence.

        Back in the presence set they are present with no term, so the row
        survives. The edge is crossed with the other-dim combinations the
        variable has, so a coordinate its own mask removed stays absent.
        """
        others = [d for d in dims if d != self.shift.along]
        edge = self.coordinates(vacated=True)
        if not others:
            return edge
        have = presence.keys(dims)
        source = presence.frame if all(d in have for d in others) else scope.widen(presence.frame, have, dims)
        keys = [d for d in self.keys if d in others]
        rows = source.select(*others).unique()
        return join_on(rows, edge, keys, 'inner')


def _named_amount(scope: Scope, order: _Order, name: str, alias: str) -> tuple[pl.LazyFrame, list[str]]:
    """A named offset's or width's values, and the keys a frame reads them by.

    A per-group amount is read under the group column, since no frame carries
    the grouped dimension.
    """
    dims = scope.program.parameters[name].dims
    keys = [order.grouping.column_of(d) or d for d in dims]
    frame = scope.data.parameters[name].select(
        *(pl.col(d).alias(key) for d, key in zip(dims, keys, strict=True)),
        pl.col(VALUE).cast(pl.Int64).alias(alias),
    )
    return frame, keys
