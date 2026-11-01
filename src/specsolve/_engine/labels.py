"""Dense solver indices for a masked coordinate product.

``var_label`` is the solver's column index and ``row`` its row index, so two
builds of one model must agree on every label (docs/development/architecture.md,
"The relational lane").
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._collect import collected, collected_all
from specsolve._engine.predicates import masked
from specsolve._engine.scope import UNIT, ordinal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from mathspec import program

    from specsolve._engine.scope import Scope
    from specsolve._engine.terms import Presence


@dataclass(frozen=True)
class Labelled:
    """One declaration's labelled frame, and the contiguous run of labels it owns."""

    frame: pl.LazyFrame
    start: int
    height: int

    def share(self, values: pl.Series) -> pl.Series:
        """This declaration's share of a solver vector — a slice, never a join."""
        return values.slice(self.start, self.height)

    def valued(self, dims: tuple[str, ...], values: pl.Series) -> pl.LazyFrame:
        """``(dims…, values.name)``: each coordinate beside its [`share`][] of *values*, in label order.

        Projected after the share is attached, never ``select(*dims)`` first:
        with no dims that keeps no column, and polars gives a frame with no
        columns no rows, so a scalar's one value would be lost. A share of the
        wrong length raises rather than padding with nulls.
        """
        return self.frame.with_columns(self.share(values)).select(*dims, values.name)


#: The column a domain's coordinates are ordered by before they are numbered.
POSITION = '#position'


@dataclass(frozen=True)
class Domain:
    """The coordinates a declaration spans: the product of its dims, under its mask, inside its restrictions.

    *restrictions* are the variable-presence semi-joins a constraint row must
    be contained in; which rows they remove is unknown until data is read.
    """

    dims: tuple[str, ...]
    where: program.Mask | None = None
    restrictions: Sequence[Presence] = ()


def frames(scope: Scope, domains: Sequence[Domain], label: str) -> list[pl.DataFrame]:
    """Each of *domains*' coordinates with a dense *label* from 0, in the solver's order.

    A label is row-major over the dims' declared ordinals, so it is the
    solver's own index once its declaration's offset is added. A polars query
    costs a fixed time that a small domain never earns back, so the domains
    are collected in one pass, and in one pass before it for the masks
    [`_factored`][] reads. Every path orders a domain's coordinates by
    [`POSITION`][], and a label is the row in that order.

    Returns:
        One frame per domain, ``(dims…, label)`` in that column order and in
        label order. With no dims it is ``([`UNIT`][], label)``, the column a
        reader keys a scalar on, since selecting nothing drops the empty
        product's one row.
    """
    free = {
        i: n
        for i, d in enumerate(domains)
        if d.where is not None and not d.restrictions and (n := _free_prefix(d.dims, d.where.dims))
    }
    ranked = collected_all(_ranked(scope, domains[i].dims[n:], domains[i].where) for i, n in free.items())
    survivors = dict(zip(free, ranked, strict=True))
    plans = [
        _factored(scope, d.dims, free[i], survivors[i]) if i in survivors else _counted(scope, d)
        for i, d in enumerate(domains)
    ]
    built = []
    for d, positioned in zip(domains, collected_all(plans), strict=True):
        ordered = in_position_order(positioned, POSITION)
        numbered = pl.Series(label, np.arange(ordered.height, dtype=np.int64)).set_sorted()
        built.append(pl.DataFrame([*(ordered.get_column(dim) for dim in d.dims or (UNIT,)), numbered]))
    return built


def declared_height(scope: Scope, dims: tuple[str, ...], where: program.Mask | None) -> int:
    """How many rows a declaration asks for: its coord product under its own mask.

    Less the height [`frames`][] numbers for it, it is the rows a propagated absence removed.
    With a mask it costs a pass over the masked product.
    """
    if where is None:
        return math.prod(scope.data.cardinality[d] for d in dims)
    return int(masked(scope, dims, where).select(pl.len()).pipe(collected).item())


def _counted(scope: Scope, domain: Domain) -> pl.LazyFrame:
    """A domain's coordinates, each beside its row-major [`POSITION`][] in the full product."""
    surviving = masked(scope, domain.dims, domain.where)
    for restriction in domain.restrictions:
        surviving = restriction.restrict(surviving, restriction.keyed_by or ())
    return surviving.select(*(domain.dims or (UNIT,)), _row_major(scope, domain.dims).alias(POSITION))


def _ranked(scope: Scope, kept: tuple[str, ...], where: program.Mask | None) -> pl.LazyFrame:
    """The surviving coordinates of the dims a mask reads, ranked in declared order."""
    return masked(scope, kept, where).sort([ordinal(d) for d in kept]).select(*kept).with_row_index('#rank')


def _factored(scope: Scope, dims: tuple[str, ...], free: int, survivors: pl.DataFrame) -> pl.LazyFrame:
    """A domain whose mask reads none of its first *free* dims, positioned without a pass over its product.

    The survivors are the full head product against one surviving suffix set,
    [`_ranked`][], so a coordinate's position is its head's row-major position
    times the suffix count, plus its suffix's rank. Which side of the cross
    join cycles is polars' choice, so [`frames`][] orders by position.
    """
    head = dims[:free]
    return (
        survivors.lazy()
        .join(masked(scope, head, None).select(*head, _row_major(scope, head).alias(POSITION)), how='cross')
        .select(*dims, (pl.col(POSITION) * survivors.height + pl.col('#rank')).alias(POSITION))
    )


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
