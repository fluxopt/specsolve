"""A relation's table as a call reads it — the one place a role becomes a column.

A `Direction` names roles; an operand carries its coordinates under dimension
names. Nothing here reads data or holds state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from mathspec import program

    from specsolve.relational.engine.attaching import AttachedSources

#: What a [`Grouping`][] adds to a dimension table: a coordinate's rank
#: inside its group, and the group's size.
GROUP_RANK = '__pos in group__'
GROUP_SIZE = '__group size__'


def landing(dim: str) -> str:
    """The column a walk's produced column waits under until the consumed one is dropped.

    A self-map produces the dimension it consumes. The spaces make the name
    unrepresentable as a declared one.
    """
    return f'__landing {dim}__'


def group_column(role: str) -> str:
    """The column a [`Grouping`][] carries one group-making value column of the relation under.

    Named for the role, not the dimension: a group may hold two columns over
    one dimension, and the operand may carry that dimension itself.
    """
    return f'__group {role}__'


# ---------------------------------------------------------------------------
# a group or a pullback
# ---------------------------------------------------------------------------


def mapping(relations: Mapping[str, pl.LazyFrame], direction: program.Direction) -> pl.LazyFrame:
    """The table a group or a pullback joins against — the relation, read as the direction names it.

    Consumed and joined columns arrive under their dimensions, produced ones
    under [`landing`][]. A key the direction does not map has no row.
    """
    table = relations[direction.name]
    return table.select(
        *(pl.col(role).alias(direction.dim(role)) for role in (*direction.consumed, *direction.joined)),
        *(pl.col(role).alias(landing(direction.dim(role))) for role in direction.produced),
    )


def landed(mapping: pl.LazyFrame, node: program.GroupSum | program.Pullback) -> pl.LazyFrame:
    """*mapping* at the coordinates it lands on: the joined dimensions and the produced ones, under their names."""
    joined = node.direction.joined_dims
    return mapping.select(*joined, *(pl.col(landing(d)).alias(d) for d in node.direction.produced_dims))


def walk_join(
    frame: pl.LazyFrame,
    mapping: pl.LazyFrame,
    node: program.GroupSum | program.Pullback | program.PulledBackPredicate,
    have: Sequence[str],
    columns: Sequence[str] = (),
) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*frame* traded through *mapping*: one inner equi-join, and the dimensions the result is over.

    The result keeps *have* less the consumed dims, gains the produced ones, and
    keeps *columns*. A self-map consumes and produces one dimension, so the two
    are traded in a single select.
    """
    direction = node.direction
    gained = direction.produced_dims
    keep = [d for d in have if d not in direction.consumed_dims]
    traded = frame.join(mapping, on=[*direction.consumed_dims, *direction.joined_dims], how='inner').select(
        *keep, *(pl.col(landing(d)).alias(d) for d in gained), *columns
    )
    return traded, (*keep, *gained)


# ---------------------------------------------------------------------------
# a partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Grouping:
    """A dimension ranked inside the groups a partition makes — what ``shift``, ``sum_back`` and ``position`` count along.

    A coordinate in no group has no row in [`table`][], so a partitioned call
    reads only grouped coordinates.

    Attributes:
        dimension: The dimension stepped along.
        joined: The partition's other key dimensions, which the operand carries.
        groups: The group-making columns, one per group column, under
            [`group_column`][].
        grouped: The dimension each group column is over, in the same order.
        table: ``(val, ord, joined…, groups…, GROUP_RANK, GROUP_SIZE)``, one
            row per coordinate the partition places in a group.
    """

    dimension: str
    joined: tuple[str, ...]
    groups: tuple[str, ...]
    grouped: tuple[str, ...]
    table: pl.LazyFrame

    @classmethod
    def whole(cls, data: AttachedSources, dimension: str) -> Grouping:
        """*dimension* as one group, so an unpartitioned call reads the same table shape."""
        table = data.dimensions[dimension].with_columns(
            pl.col('ord').alias(GROUP_RANK),
            pl.lit(data.cardinality[dimension], dtype=pl.Int64).alias(GROUP_SIZE),
        )
        return cls(dimension, (), (), (), table)

    @classmethod
    def of(cls, data: AttachedSources, partition: program.Partition) -> Grouping:
        """Rank the dimension *partition* steps along inside the groups its ``within=`` columns make."""
        dimension = partition.along_dim
        joined = partition.joined_dims
        groups = tuple(group_column(role) for role in partition.group)
        rows = data.relations[partition.name].select(
            pl.col(partition.along).alias('val'),
            *(pl.col(role).alias(dim) for role, dim in zip(partition.joined, joined, strict=True)),
            *(pl.col(role).alias(column) for role, column in zip(partition.group, groups, strict=True)),
        )
        key = [*joined, *groups]
        table = (
            data.dimensions[dimension]
            .join(rows, on='val', how='inner')
            .with_columns(
                (pl.col('ord').rank('ordinal').over(key) - 1).cast(pl.Int64).alias(GROUP_RANK),
                pl.len().over(key).cast(pl.Int64).alias(GROUP_SIZE),
            )
        )
        return cls(dimension, joined, groups, tuple(partition.dim(role) for role in partition.group), table)

    @property
    def key(self) -> tuple[str, ...]:
        """What one group is identified by: its joined coordinates and its group columns."""
        return (*self.joined, *self.groups)

    @property
    def keys(self) -> tuple[str, ...]:
        """What a coordinate of the dimension stepped along is identified by: the dimension, and the joined ones."""
        return (self.dimension, *self.joined)

    @property
    def partial(self) -> bool:
        """Whether a coordinate can be in no group."""
        return bool(self.groups)

    def placed(self) -> pl.LazyFrame:
        """The coordinates the partition places in some group, under [`keys`][]; the rest build no row."""
        return self.table.select(pl.col('val').alias(self.dimension), *self.joined).unique()

    def column_of(self, dim: str) -> str | None:
        """The group column carrying *dim*'s labels, or ``None`` where the partition does not group into it.

        Where two group columns are over one dimension, the first declared carries it.
        """
        return next((column for column, over in zip(self.groups, self.grouped, strict=True) if over == dim), None)
