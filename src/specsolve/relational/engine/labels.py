"""Dense solver indices for a masked coordinate product.

``var_label`` is the solver's column index and ``row`` its row index, so two
builds of one model must agree on every label (docs/about/architecture.md,
"The relational lane").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve.relational.collect import collect_engine
from specsolve.relational.engine.predicates import masked
from specsolve.relational.engine.scope import UNIT, ordinal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec import program

    from specsolve.relational.engine.pieces import Presence
    from specsolve.relational.engine.scope import Scope


@dataclass(frozen=True)
class Labelled:
    """One declaration's labelled frame, and the contiguous run of labels it owns."""

    frame: pl.LazyFrame
    start: int
    height: int

    def share(self, values: pl.Series) -> pl.Series:
        """This declaration's share of a solver vector — a slice, never a join."""
        return values.slice(self.start, self.height)


def frame(
    scope: Scope,
    dims: tuple[str, ...],
    where: program.Mask | None,
    label: str,
    start: int,
    restrictions: Sequence[Presence] = (),
) -> pl.DataFrame:
    """The masked coord product of *dims* with a dense *label* from *start*.

    A label is row-major over the dims' declared ordinals, so it is the
    solver's own index with no remapping. *restrictions* are the
    variable-presence semi-joins a constraint row must be contained in; which
    rows they remove is unknown until data is read, so they take the counted
    path. With no dims the query selects [`UNIT`][], since selecting nothing
    drops the empty product's one row.

    Returns:
        ``(dims…, label)`` in that column order and in label order; the next
        free label is ``start`` plus its height.
    """
    if where is not None and not restrictions:
        free = _free_prefix(dims, where.dims)
        if free:
            factored = _factored(scope, dims, free, where, label, start)
            if factored is not None:
                return factored

    surviving = masked(scope, dims, where)
    for restriction in restrictions:
        surviving = restriction.restrict(surviving, restriction.keyed_by or ())

    dropped = where is not None or bool(restrictions)
    numbering = _row_major(scope, dims)
    if not dropped:
        numbering = pl.lit(start, dtype=pl.Int64) + numbering
    position = '#position' if dropped else label
    materialised = in_position_order(
        surviving.select(*(dims or (UNIT,)), numbering.alias(position)).collect(engine=collect_engine()),
        position,
    )
    if not dropped and dims:
        materialised.replace_column(materialised.get_column_index(label), materialised.get_column(label).set_sorted())
        return materialised
    if dropped:
        materialised = materialised.with_row_index(label, offset=start).with_columns(pl.col(label).cast(pl.Int64))
    return materialised.select(*dims, pl.col(label).set_sorted())


def declared_height(scope: Scope, dims: tuple[str, ...], where: program.Mask | None) -> int:
    """How many rows a declaration asks for: its coord product under its own mask.

    Less [`frame`][]'s height, it is the rows a propagated absence removed.
    With a mask it costs a pass over the masked product.
    """
    if where is None:
        return math.prod(scope.data.cardinality[d] for d in dims)
    return int(masked(scope, dims, where).select(pl.len()).collect(engine=collect_engine()).item())


def _factored(
    scope: Scope,
    dims: tuple[str, ...],
    free: int,
    where: program.Mask,
    label: str,
    start: int,
) -> pl.DataFrame | None:
    """Labels for a mask that reads none of the first *free* dims.

    The survivors are the full head product against one surviving suffix set,
    so only the suffix is ranked, and the label is the number the counted path
    gives. ``None`` when nothing survives, for the counted path to answer.
    Which side of the cross join cycles is polars' choice, so
    [`in_position_order`][] verifies it.
    """
    head, kept = dims[:free], dims[free:]
    rank = '#rank'
    survivors = (
        masked(scope, kept, where)
        .sort([ordinal(d) for d in kept])
        .select(*kept)
        .with_row_index(rank)
        .collect(engine=collect_engine())
    )
    width = survivors.height
    if width == 0:
        return None

    position = '#position'
    labelled = (
        survivors.lazy()
        .join(masked(scope, head, None).select(*head, _row_major(scope, head).alias(position)), how='cross')
        .select(
            *dims,
            (pl.lit(start, dtype=pl.Int64) + pl.col(position) * width + pl.col(rank)).alias(label),
        )
        .collect(engine=collect_engine())
    )
    return in_position_order(labelled, label).with_columns(pl.col(label).set_sorted())


def _free_prefix(dims: tuple[str, ...], touched: frozenset[str]) -> int:
    """How many leading dims the mask does not read; 0 when it reads the first dim or none.

    Only a prefix leaves the survivors contiguous in declaration order.
    """
    free = 0
    while free < len(dims) and dims[free] not in touched:
        free += 1
    return free if free < len(dims) else 0


def _row_major(scope: Scope, dims: tuple[str, ...]) -> pl.Expr:
    """[`Scope.row_major`][] over a product frame, which carries its ordinals."""
    return scope.row_major(dims, lambda d: pl.col(ordinal(d)))


def in_position_order(materialised: pl.DataFrame, position: str) -> pl.DataFrame:
    """The frame ordered by *position*, sorted only when the engine emitted another order."""
    if materialised.get_column(position).is_sorted():
        return materialised
    return materialised.sort(position)
