"""Contiguous solver indices for a coordinate product narrowed by a condition.

``COL`` is the solver's column index and ``ROW`` its row index, so two
builds of one model must agree on every solver index
(docs/development/architecture.md, "The relational lane").
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve._collect import collected
from specsolve._engine.conditions import narrowed
from specsolve._engine.scope import UNIT, position_column
from specsolve._names import LABEL

if TYPE_CHECKING:
    from collections.abc import Collection, Sequence

    from mathspec import program

    from specsolve._engine.scope import Scope
    from specsolve._engine.terms import Presence


@dataclass(frozen=True)
class Indexed:
    """One entry's indexed table, and the contiguous run of solver indices it owns."""

    table: pl.LazyFrame
    start: int
    height: int

    def share(self, values: pl.Series) -> pl.Series:
        """This entry's share of a solver vector — a slice, never a join."""
        return values.slice(self.start, self.height)

    def valued(self, dims: tuple[str, ...], values: pl.Series) -> pl.LazyFrame:
        """``(dims…, values.name)``: each coordinate beside its [`share`][] of *values*, in index order.

        Projected after the share is attached, never ``select(*dims)`` first:
        with no dimensions that keeps no column, and polars gives a table with
        no columns no rows, so a scalar's one value would be lost. A share of the
        wrong length raises rather than padding with nulls.
        """
        return self.table.with_columns(self.share(values)).select(*dims, values.name)


def table(
    scope: Scope,
    dims: tuple[str, ...],
    where: program.Mask | None,
    index: str,
    start: int,
    presences: Sequence[Presence] = (),
) -> pl.DataFrame:
    """The coordinate product of *dims* narrowed by *where*, with a contiguous *index* from *start*.

    The index is row-major over the dimensions' positions, so it is the
    solver's own index with no remapping. *presences* are the
    variable presences that narrow a constraint's rows; which
    rows they remove is unknown until data is read, so they take the counted
    path.

    Returns:
        ``(dims…, index)`` in that column order and in index order; the next
        free index is ``start`` plus its height, each column in one chunk, so
        a table laid out against it is written as one parquet row group
        rather than one per chunk the cross join left. With no dimensions it is
        ``([`UNIT`][], index)``, the column a reader keys a scalar on, since selecting
        nothing drops the empty product's one row.
    """
    if where is not None and not presences:
        free = _free_prefix(dims, where.dims)
        if free:
            factored = _factored(scope, dims, free, where, index, start)
            if factored is not None:
                return factored

    surviving = _survivors(scope, dims, where)
    for presence in presences:
        surviving = presence.narrow(surviving, presence.joined_on or ())

    dropped = where is not None or bool(presences)
    numbering = _row_major(scope, dims)
    if not dropped:
        numbering = pl.lit(start, dtype=pl.Int64) + numbering
    position = '#position' if dropped else index
    materialised = in_position_order(
        surviving.select(*(dims or (UNIT,)), numbering.alias(position)).pipe(collected),
        position,
    )
    if not dropped and dims:
        materialised.replace_column(materialised.get_column_index(index), materialised.get_column(index).set_sorted())
        return materialised.rechunk()
    if dropped:
        materialised = materialised.with_row_index(index, offset=start).with_columns(pl.col(index).cast(pl.Int64))
    return materialised.select(*(dims or (UNIT,)), pl.col(index).set_sorted()).rechunk()


def _survivors(scope: Scope, dims: tuple[str, ...], where: program.Mask | None) -> pl.LazyFrame:
    """The coordinates of *dims* the condition keeps, built from the dimensions it reads crossed with the ones it leaves free.

    A condition over some of the dimensions is evaluated on their product only,
    so the full product is never built where the condition leaves a dimension
    free in the middle of the order; dimensions sorted by name put one there,
    as ``snapshot`` in ``(node, snapshot, tech)`` under a condition over
    ``(node, tech)``. The rows come out of order, which the caller's numbering
    sorts.
    """
    free = tuple(d for d in dims if where is not None and d not in where.dims)
    if not free or len(free) == len(dims):
        return narrowed(scope, dims, where)
    read = tuple(d for d in dims if d not in free)
    return narrowed(scope, read, where).join(scope.product(free), how='cross')


def nothing(scope: Scope, dims: tuple[str, ...], index: str) -> pl.DataFrame:
    """[`table`][]'s answer over dimensions one of which has no label, read without a query: no row, in its columns and dtypes."""
    schema = {d: scope.data.dimensions[d].collect_schema()[LABEL] for d in dims}
    return pl.DataFrame(schema=schema | {index: pl.Int64})


def _factored(
    scope: Scope,
    dims: tuple[str, ...],
    free: int,
    where: program.Mask,
    index: str,
    start: int,
) -> pl.DataFrame | None:
    """Solver indices for a condition that reads none of the first *free* dimensions.

    The survivors are the full head product against one surviving suffix set,
    so only the suffix is ranked, and the index is the number the counted path
    gives. ``None`` when nothing survives, for the counted path to answer.
    Which side of the cross join cycles is polars' choice, so
    [`in_position_order`][] verifies it.
    """
    head, kept = dims[:free], dims[free:]
    rank = '#rank'
    survivors = (
        narrowed(scope, kept, where)
        .sort([position_column(d) for d in kept])
        .select(*kept)
        .with_row_index(rank)
        .pipe(collected)
    )
    width = survivors.height
    if width == 0:
        return None

    position = '#position'
    indexed = (
        survivors.lazy()
        .join(narrowed(scope, head, None).select(*head, _row_major(scope, head).alias(position)), how='cross')
        .select(
            *dims,
            (pl.lit(start, dtype=pl.Int64) + pl.col(position) * width + pl.col(rank)).alias(index),
        )
        .pipe(collected)
    )
    return in_position_order(indexed, index).with_columns(pl.col(index).set_sorted()).rechunk()


def _free_prefix(dims: tuple[str, ...], touched: Collection[str]) -> int:
    """How many leading dimensions the condition does not read; 0 when it reads the first dimension or none.

    Only a prefix leaves the survivors contiguous in row-major order.
    """
    free = 0
    while free < len(dims) and dims[free] not in touched:
        free += 1
    return free if free < len(dims) else 0


def _row_major(scope: Scope, dims: tuple[str, ...]) -> pl.Expr:
    """[`Scope.row_major`][] over a product table, which carries its positions."""
    return scope.row_major(dims, lambda d: pl.col(position_column(d)))


def in_position_order(materialised: pl.DataFrame, position: str) -> pl.DataFrame:
    """The table ordered by *position*, sorted only when the engine emitted another order."""
    if materialised.get_column(position).is_sorted():
        return materialised
    return materialised.sort(position)
