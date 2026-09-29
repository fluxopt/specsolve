"""The scope a query is compiled in: what each name stands for, and what a dimension means as a coordinate.

What every query in the lane is written against, and what each helper takes
— the labeller, the mask walk, the reindexing operators, the coverage
guards — none of which compiles an expression. The compiler holds one of
these beside the walk it adds.

``data`` is everything attaching produced, frozen. ``variables`` is the
build's own dict, not a copy: a variable frame appears while its declaration
is built and a constraint compiled afterwards has to see it — the one live
registry in the lane, visible in this signature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from specsolve.relational.engines.polars.fragments import join_on

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping, Sequence

    from mathspec import program
    from polars._typing import JoinStrategy, MaintainOrderJoin

    from specsolve.relational.engines.polars.attaching import AttachedSources
    from specsolve.relational.engines.polars.fragments import TermFragment
    from specsolve.relational.engines.polars.labels import Labelled


#: Carries the single row of the empty coordinate product. Polars cannot hold a
#: frame with one row and no columns — collecting one reports ``(0, 0)`` — so the
#: unit needs a column to exist in, and every path drops it by selecting the
#: dims and the label instead.
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

        **Folded in reverse, then projected back.** polars' streaming engine
        walks a cross join right-major, so folding backwards makes the product
        arrive in declaration row-major order — label order.
        [`labels.frame`][] verifies that rather than trusting it.

        The empty product is one *real* row carrying only [`UNIT`][]: a
        ``where`` on a scalar declaration filters this frame, and nothing
        survives a filter.
        """
        out: pl.LazyFrame | None = None
        for d in reversed(dims):
            table = self.data.dimensions[d].select(pl.col('val').alias(d), pl.col('ord').alias(ordinal(d)))
            out = table if out is None else out.join(table, how='cross')
        if out is None:
            return pl.LazyFrame({UNIT: [0]})
        return out.select(*(c for d in dims for c in (d, ordinal(d))))

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

        A parameter carrying a dim the frame lacks would be reduced over it,
        widening a mask or picking an arbitrary bound, so that is refused;
        *subject* is the caller's word for the declaration to name.

        *how* is ``left`` for a bound, where a missing value is a fact to
        report rather than a row to drop. ``inner`` is the mask walk's story
        ([`compile_predicate`][specsolve.relational.engines.polars.predicates.compile_predicate]).

        *maintain_order* is asked for only by the bounds, which become ``cols``
        and are read in order; every other consumer verifies order where it
        reads.
        """
        declaration = self.program.parameters[param]
        assert not set(declaration.dims) - set(frame_dims), (
            f'{subject} has dims outside the frame dims {list(frame_dims)}'
        )
        table = self.data.parameters[param].rename({'value': alias})
        return join_on(frame, table, declaration.dims, how, maintain_order)

    def row_major(self, dims: tuple[str, ...], ordinals: Callable[[str], pl.Expr]) -> pl.Expr:
        """A coordinate's row-major position in the declared product of *dims*.

        A label, a bound's slot and a set's position all read this one rule, so
        two builds of one model agree on every index. Dense over the *full*
        product rather than the survivors; with no dims, the literal zero of the
        empty product's one row. *ordinals* says how the frame in hand carries a
        dim's ordinal — a product frame has the column beside the label, a
        built variable frame kept only the label and reads it through
        [`ordinal_of`][].
        """
        position: pl.Expr = pl.lit(0, dtype=pl.Int64)
        for d in dims:
            position = position * self.data.cardinality[d] + ordinals(d)
        return position.cast(pl.Int64)

    def ordinal_of(self, dim: str) -> pl.Expr:
        """A *dim* value column as that dimension's ordinal.

        A string dimension is Enum-encoded by attaching over the labels in
        ordinal order, so the physical code already *is* the ordinal. Every
        other dtype uses a dictionary built from the dimension table — one
        entry per label, not per row.
        """
        column = pl.col(dim)
        if self.data.is_enum_encoded(dim):
            return column.to_physical().cast(pl.Int64)
        labels = self.data.dimensions[dim].select('val').collect()['val']
        return column.replace_strict({value: at for at, value in enumerate(labels)}, return_dtype=pl.Int64)

    def widen(self, presence: pl.LazyFrame, have: tuple[str, ...], want: tuple[str, ...]) -> pl.LazyFrame:
        """*presence* over every dim in *want*, saying the same thing.

        A presence frame is silent about the dims it omits, which reads as
        "present at all of them" — so the widening is a cross join with those
        dimensions' own tables, and it changes no answer.
        """
        return self.spread(presence, [d for d in want if d not in have]).select(*want)

    def spread(self, frame: pl.LazyFrame, dims: Sequence[str]) -> pl.LazyFrame:
        """*frame* repeated at every label of each of *dims*, which it does not carry."""
        for d in dims:
            frame = frame.join(self.data.dimensions[d].select(pl.col('val').alias(d)), how='cross')
        return frame

    def spanned(self, fragments: Sequence[TermFragment]) -> tuple[str, ...]:
        """The dims *fragments* carry between them, in declaration order."""
        return self.in_declaration_order(d for p in fragments for d in p.dims)

    def in_declaration_order(self, dims: Iterable[str]) -> tuple[str, ...]:
        """*dims* in the order the file declares them, duplicates dropped.

        A ``where`` leaf stamps its dims as a set, and two frames keyed by the
        same dims in two orders join on a key written twice — so the order
        comes from the one place that has one.
        """
        wanted = set(dims)
        return tuple(d for d in self.program.dimensions if d in wanted)
