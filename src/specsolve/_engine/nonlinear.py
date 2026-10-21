"""Nonlinear terms as defined columns: each a new column ``w`` and the equation ``w == op(a, b)``.

``a`` and ``b`` are affine in the model's columns, one value per coordinate of
the term. The term is replaced by ``w``, so every row stays at most quadratic
and everything nonlinear reaches a sink in one stream of its own,
``(col, op)`` beside ``(col, arg, var, coeff)``, a null ``var`` being the
argument's constant.

A defined column is numbered after every declared variable, so the read-back,
which slices a declaration's share of the primal, never sees one. It is
free: its equation is what holds it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, get_args

import polars as pl

from specsolve._collect import collected
from specsolve._engine import labels
from specsolve._engine.pieces import CompiledExpression, Piece, absence_restrictions
from specsolve._engine.scope import join_on

if TYPE_CHECKING:
    from collections.abc import Sequence

    from specsolve._engine.scope import Scope

#: What a defined column may equal: a binary operator over two arguments, or a function of the first.
Op = Literal['mul', 'div', 'pow', 'exp', 'log', 'sqrt', 'sin', 'cos']
OPS: tuple[Op, ...] = get_args(Op)
NL_OP = pl.Enum(OPS)

#: The two frames a sink reads, as schemas.
NL_SCHEMA = {'col': pl.Int32, 'op': NL_OP}
NL_ARGS_SCHEMA = {'col': pl.Int32, 'arg': pl.UInt8, 'var': pl.Int32, 'coeff': pl.Float64}


@dataclass
class Atoms:
    """The defined columns one build has numbered so far, and their equations.

    ``next`` is set by the assembly once every declared variable has its
    columns, and is the model's column count when the build ends.
    """

    scope: Scope
    next: int = 0
    defined: list[pl.DataFrame] = field(default_factory=list)
    args: list[pl.DataFrame] = field(default_factory=list)

    def atom(self, op: Op, operands: Sequence[CompiledExpression]) -> Piece:
        """``op(*operands)`` as a term over new columns, one per coordinate the operands span.

        A coordinate where a variable under an operand is absent gets no
        column, and the term carries that absence on, as a quotient by an
        absent divisor does. A product of two variables in an operand becomes
        a column of its own first ([`linear`][]).
        """
        operands = [self.linear(c) for c in operands]
        pieces = [p for c in operands for p in (*c.terms, *c.consts)]
        dims = self.scope.spanned(pieces)
        restrictions = absence_restrictions(pieces)
        numbered = labels.frame(self.scope, dims, None, 'col', self.next, restrictions)
        self.next += numbered.height
        held = numbered.lazy()
        parts: list[pl.LazyFrame] = []
        for k, c in enumerate(operands):
            arg = pl.lit(k, dtype=pl.UInt8).alias('arg')
            parts += [
                join_on(held, p.frame, p.dims, 'inner').select('col', arg, pl.col('var_label').alias('var'), 'coeff')
                for p in c.terms
            ]
            parts += [
                join_on(held, p.frame, p.dims, 'inner').select(
                    'col', arg, pl.lit(None, dtype=pl.Int64).alias('var'), pl.col('cval').alias('coeff')
                )
                for p in c.consts
            ]
        self._record(numbered.select('col'), op, pl.concat(parts) if parts else None)
        term = held.select(*dims, pl.col('col').alias('var_label'), pl.lit(1.0, dtype=pl.Float64).alias('coeff'))
        return Piece(dims, term, 'term', presences=tuple(restrictions))

    def linear(self, compiled: CompiledExpression) -> CompiledExpression:
        """*compiled* with each product of two variables replaced by a ``mul`` column.

        One column per distinct pair, wherever it stands, so ``x * y`` written
        twice is one column.
        """
        if not compiled.quads:
            return compiled
        return CompiledExpression((*compiled.terms, *map(self._multiplied, compiled.quads)), compiled.consts)

    def _multiplied(self, q: Piece) -> Piece:
        """One quadratic piece as a term over ``mul`` columns, a column per pair of variables it multiplies."""
        frame = q.frame.pipe(collected)
        pairs = (
            frame.select('var_label', 'var_label_2')
            .unique(maintain_order=True)
            .with_row_index('col', offset=self.next)
            .with_columns(pl.col('col').cast(pl.Int64))
        )
        self.next += pairs.height
        one = pl.lit(1.0, dtype=pl.Float64).alias('coeff')
        args = pl.concat(
            [
                pairs.lazy().select('col', pl.lit(k, dtype=pl.UInt8).alias('arg'), pl.col(side).alias('var'), one)
                for k, side in enumerate(('var_label', 'var_label_2'))
            ]
        )
        self._record(pairs.select('col'), 'mul', args)
        term = frame.join(pairs, on=['var_label', 'var_label_2']).select(
            *q.dims, pl.col('col').alias('var_label'), 'coeff'
        )
        return Piece(q.dims, term.lazy(), 'term', presences=q.presences, region=q.region)

    def _record(self, cols: pl.DataFrame, op: Op, args: pl.LazyFrame | None) -> None:
        """Keep the equations of *cols*: their op, and their arguments with repeated entries summed.

        A null coefficient is a parameter with no row under an argument, and
        adds nothing.
        """
        self.defined.append(cols.select(pl.col('col').cast(pl.Int32), pl.lit(op, dtype=NL_OP).alias('op')))
        if args is None:
            return
        summed = (
            args.filter(pl.col('coeff').is_not_null())
            .group_by('col', 'arg', 'var')
            .agg(pl.col('coeff').sum())
            .sort('col', 'arg', 'var', nulls_last=True)
            .pipe(collected)
        )
        self.args.append(summed.select(*(pl.col(c).cast(t) for c, t in NL_ARGS_SCHEMA.items())))

    def frames(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        """The two streams a sink reads, in column order."""
        defined = pl.concat(self.defined) if self.defined else pl.DataFrame(schema=NL_SCHEMA)
        args = pl.concat(self.args) if self.args else pl.DataFrame(schema=NL_ARGS_SCHEMA)
        return defined, args
