"""Logical plan → polars. Lazy: nothing is read, nothing is executed.

Column conventions, relied on by the engine:

===================  ==========================================
frame                columns
===================  ==========================================
dimension table      ``val``, ``ord``
relation table       its declared columns, each under its own name
parameter table      ``dims…``, ``value``
variable frame       ``dims…``, ``var_label``
===================  ==========================================

A piece's columns are in ``pieces.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np
import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError
from specsolve.relational.collect import collected
from specsolve.relational.engine.pieces import (
    PRESENT,
    CompiledExpression,
    Piece,
    Presence,
    absence_restrictions,
    acted_along,
    both_regions,
    constant_scalar,
    fan_in,
    join_mul,
    join_on,
    join_pow,
    join_quad,
    map_pieces,
    negate,
    ordered_sum,
    propagate_absence,
    refuse_a_piece_without_the_dims,
    region_over,
)
from specsolve.relational.engine.predicates import Carrier, compile_predicate, falsy_if_null, masked
from specsolve.relational.engine.relations import landed, mapping, walk_join
from specsolve.relational.engine.scope import UNIT, Scope
from specsolve.relational.engine.shifts import translate_piece, window_piece

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve.relational.engine.labels import Labelled


def _totalled(p: Piece) -> pl.LazyFrame:
    """Const piece *p* added up to one ``cval`` per coordinate, null where any of its rows is.

    A null summand is a divisor's hole, so the total keeps it for the refusal.
    """
    total = pl.when(pl.col('cval').is_null().any()).then(None).otherwise(ordered_sum('cval')).alias('cval')
    return p.frame.group_by(p.dims).agg(total) if p.dims else p.frame.select(total)


def _presence(held: Labelled, dims: tuple[str, ...], label: str) -> pl.LazyFrame:
    """The coordinates a declaration's rows exist at.

    A scalar's marker is renamed from the *label* column, never a ``pl.lit()``:
    a select of literals alone is length 1, so an absent scalar would come back present.
    """
    if dims:
        return held.frame.select(*dims)
    return held.frame.select(pl.col(label).alias(PRESENT))


@dataclass(frozen=True)
class Solution:
    """What a solve left, for a compiler reading a named expression at it.

    ``dual`` is ``None`` where the solve left no duals, and ``no_duals`` then
    says why, which is what reading one raises.
    """

    primal: pl.Series
    dual: pl.Series | None
    constraints: Mapping[str, Labelled]
    no_duals: str | None


@dataclass(frozen=True)
class Compiler:
    """Turn plan nodes into polars queries over the model's tidy frames.

    ``solution`` is set on the compiler a read builds and on no other: with it
    every variable and every ``dual(c)`` compiles to a value.
    """

    scope: Scope
    solution: Solution | None = None

    # ------------------------------------------------------------------
    # bounds
    # ------------------------------------------------------------------

    def bounds(self, frame: pl.LazyFrame, variable: str, v: program.VariableDeclaration) -> pl.LazyFrame:
        """*frame* with ``lb``/``ub`` columns for the variable *variable* declares as *v*."""
        carrier = Carrier(frame)

        def attach_bound(f: pl.LazyFrame, alias: str, name: str) -> pl.LazyFrame:
            aligned = self._aligned_bound(f, name, v, alias)
            if aligned is not None:
                return aligned
            subject = f"bound parameter '{name}' of variable '{variable}'"
            return self.scope.parameter_join(f, name, v.dims, alias, subject, maintain_order='left')

        def bound(e: program.Expression | None, open_side: float) -> pl.Expr:
            """A bound is a number, a parameter name, or ``None`` where that side is open; lowering admits nothing else."""
            if e is None:
                return pl.lit(open_side, dtype=pl.Float64)
            if isinstance(e, program.Constant):
                return pl.lit(float(e.value), dtype=pl.Float64)
            if isinstance(e, program.Parameter):
                alias = carrier.once(f'__bound {e.name}__', lambda f, a: attach_bound(f, a, e.name))
                return pl.col(alias).cast(pl.Float64)
            msg = f"unsupported node {type(e).__name__} in bounds of variable '{variable}'"
            raise AssertionError(msg)

        lower, upper = bound(v.lower, -math.inf), bound(v.upper, math.inf)
        return carrier.frame.with_columns(lower.alias('lb'), upper.alias('ub'))

    def _aligned_bound(
        self, frame: pl.LazyFrame, param: str, v: program.VariableDeclaration, alias: str
    ) -> pl.LazyFrame | None:
        """*frame* with *param* attached by position, or ``None`` to join.

        Position lines up only for a dense parameter over the variable's
        unmasked dims, in order. Height stands in for density because
        duplicates are refused at the door.
        """
        declaration = self.scope.program.parameters[param]
        if v.where is not None or tuple(declaration.dims) != tuple(v.dims) or not v.dims:
            return None

        expected = math.prod(self.scope.data.cardinality[d] for d in v.dims)
        table = self.scope.data.parameters[param]
        if table.select(pl.len()).pipe(collected).item() != expected:
            return None

        position = self.scope.row_major(v.dims, self.scope.ordinal_of)
        pairs = table.select(position.alias('__at__'), pl.col('value')).pipe(collected)
        return frame.with_columns(pl.Series(alias, _scattered(pairs['__at__'], pairs['value'], expected)))

    # ------------------------------------------------------------------
    # expressions → pieces
    # ------------------------------------------------------------------

    def expression(
        self, expr: program.Expression, context: str, *, quadratic: bool = False, reported: bool = False
    ) -> CompiledExpression:
        """Compile an expression into term, quadratic and const pieces.

        *quadratic* is the position's ceiling: the objective can hold a product
        of two variables and a constraint row cannot. *reported* is set by a
        read, where a quotient is absent wherever its divisor is zero
        ([`_read_divisor`][]). The asserts are the plan-boundary backstop to
        ``mathspec.degree``. No join in the walk maintains order; every
        consumer verifies order where it reads.
        """

        def product(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a * b``, every pairing of both operands' pieces formed once, both mixed products included.

            A constant piece owes a factor's parameters only where the other
            factor carries no variable ([`Piece.parameters`][]).
            """
            assert not ((a.quads and b.terms) or (b.quads and a.terms) or (a.quads and b.quads)), (
                f'in {context}: a product of degree 3 reached the compiler'
            )
            assert quadratic or not (a.terms and b.terms), (
                f'in {context}: a quadratic product in a position compiled as affine'
            )
            quads = tuple(join_quad(t, u) for t in a.terms for u in b.terms)
            quads += tuple(join_mul(q, c, 'quad') for q in a.quads for c in b.consts)
            quads += tuple(join_mul(q, c, 'quad') for q in b.quads for c in a.consts)
            terms = tuple(join_mul(t, c, t.kind) for t in a.terms for c in b.consts)
            terms += tuple(join_mul(t, c, t.kind) for t in b.terms for c in a.consts)
            a_carries, b_carries = bool(a.terms or a.quads), bool(b.terms or b.quads)
            consts = tuple(
                replace(
                    join_mul(x, c, 'const'),
                    parameters=(frozenset() if b_carries else x.parameters)
                    | (frozenset() if a_carries else c.parameters),
                )
                for x in a.consts
                for c in b.consts
            )
            return CompiledExpression(terms, consts, quads)

        def quotient(e: program.Divide) -> CompiledExpression:
            """``a / b``, where *b* carries no variable, added up first to one value per coordinate."""
            a, b = ev(e.numerator), self._added_up(ev(e.divisor), e.divisor, absent='spreads')
            assert not (b.terms or b.quads), f'in {context}: a divisor carrying a variable reached the compiler'
            assert len(b.consts) == 1, 'a variable-free divisor is added up to one piece'
            inv = self._read_divisor(b.consts[0]) if reported else b.consts[0]
            terms = tuple(join_mul(t, inv, t.kind, divide=True) for t in a.terms)
            quads = tuple(join_mul(q, inv, 'quad', divide=True) for q in a.quads)
            consts = tuple(join_mul(x, inv, 'const', divide=True) for x in a.consts)
            return CompiledExpression(terms, consts, quads)

        def power(e: program.Power) -> CompiledExpression:
            """``a ** b``, where neither side carries a variable; each side is added up first."""
            a = self._added_up(ev(e.base), e.base, absent='zero')
            b = self._added_up(ev(e.exponent), e.exponent, absent='zero')
            assert not (a.terms or a.quads or b.terms or b.quads), (
                f'in {context}: a power over variables reached the compiler'
            )
            assert len(a.consts) == 1 and len(b.consts) == 1, 'a variable-free operand is added up to one piece'
            return CompiledExpression((), (join_pow(a.consts[0], b.consts[0]),))

        def shaped(
            e: program.Sum | program.GroupSum | program.Pullback | program.Translate | program.WindowSum,
            rewrite: Callable[[Piece], Piece],
        ) -> CompiledExpression:
            """One shape operator applied to its compiled operand.

            A node that is not one-to-one mixes input slots, so absence reaches
            the operand before the rewrite consumes it.
            """
            inner = ev(e.operand)
            if fan_in(e) != 'one-to-one':
                assert isinstance(e, program.Sum | program.GroupSum | program.WindowSum), 'only a sum fans in'
                inner = propagate_absence(inner, self.scope, acted_along(e))
            return map_pieces(inner, rewrite)

        def region(r: program.Region) -> CompiledExpression:
            """One region's value, kept only where that region applies."""
            on = tuple(d for d in self.scope.program.dimensions if d in r.when.dims)
            truth = masked(self.scope, on, r.when).select(*on) if on else None

            def relaxed(x: Presence, dims: tuple[str, ...]) -> Presence:
                """A region's presence, silent about the coordinates the region does not claim.

                Unwidened, it would unmake rows another region covers. The
                regions are disjoint, so each coordinate is still held to the
                one region that claims it.
                """
                have = x.keys(dims)
                keys = tuple(dict.fromkeys((*have, *on)))
                complement = masked(self.scope, on, ~r.when)
                elsewhere = complement.select(*on) if on else complement.select(UNIT)
                widened = [self.scope.widen(x.frame, have, keys), self.scope.widen(elsewhere, on, keys)]
                return Presence(pl.concat(widened, how='vertical_relaxed').unique(), keys)

            def kept(p: Piece) -> Piece:
                """One piece cut down to the region's coordinates.

                An inner join, so a value narrower than the mask gains the
                mask's dims. A mask reading no dimension filters by its own
                constant instead.
                """
                if truth is None:
                    carrier, condition = compile_predicate(self.scope, p.frame, r.when, p.dims)
                    frame = carrier.filter(falsy_if_null(condition)).select(*p.dims, *p.carried)
                    presences = tuple(relaxed(x, p.dims) for x in p.presences)
                    return replace(p, frame=frame, presences=presences, region=both_regions(p.region, r.when))
                shared = tuple(d for d in p.dims if d in on)
                out_dims = p.dims + tuple(d for d in on if d not in p.dims)
                frame = join_on(p.frame, truth, shared, 'inner').select(*out_dims, *p.carried)
                presences = tuple(relaxed(x, p.dims) for x in p.presences)
                return replace(
                    p, dims=out_dims, frame=frame, presences=presences, region=both_regions(p.region, r.when)
                )

            return map_pieces(ev(r.value), kept)

        def cases(e: program.Cases) -> CompiledExpression:
            """Every region added; the language proved them disjoint at load, so nothing is ranked or subtracted.

            A region worth the literal zero adds nothing, so it builds no
            piece: a piece would carry its region's claim, which a pullback
            cannot follow, and be owed at every row it lands on.
            """
            built = [region(r) for r in e.regions if not _zero(r.value)]
            return CompiledExpression(
                tuple(f for c in built for f in c.terms),
                tuple(f for c in built for f in c.consts),
                tuple(f for c in built for f in c.quads),
            )

        def ev(e: program.Expression) -> CompiledExpression:
            if isinstance(e, program.Constant):
                frame = pl.LazyFrame({'cval': [float(e.value)]}, schema={'cval': pl.Float64})
                return CompiledExpression((), (Piece((), frame, 'const'),))
            if isinstance(e, program.Parameter):
                return CompiledExpression((), (self._parameter_piece(e.name),))
            if isinstance(e, program.Variable):
                if self.solution is None:
                    return CompiledExpression((self._variable_piece(e.name),), ())
                return CompiledExpression((), (self._solved_piece(e.name),))
            if isinstance(e, program.Dual):
                assert self.solution is not None, (
                    f'in {context}: a dual reached a build — the language keeps one out of the math'
                )
                return CompiledExpression((), (self._dual_piece(e.constraint),))
            if isinstance(e, program.Negate):
                return map_pieces(ev(e.operand), negate)
            if isinstance(e, program.Add):
                a, b = ev(e.left), ev(e.right)
                return CompiledExpression(a.terms + b.terms, a.consts + b.consts, a.quads + b.quads)
            if isinstance(e, program.Multiply):
                return product(ev(e.left), ev(e.right))
            if isinstance(e, program.Divide):
                return quotient(e)
            if isinstance(e, program.Power):
                return power(e)
            if isinstance(e, program.Sum):
                return shaped(e, lambda p: self._sum_piece(p, e.over, context))
            if isinstance(e, program.GroupSum):
                return shaped(e, lambda p: self._group_piece(p, e, context))
            if isinstance(e, program.Pullback):
                return shaped(e, lambda p: self._at_piece(p, e, context))
            if isinstance(e, program.Translate):
                return shaped(e, lambda p: translate_piece(self.scope, p, e, context))
            if isinstance(e, program.WindowSum):
                return shaped(e, lambda p: window_piece(self.scope, p, e, context))
            if isinstance(e, program.Cases):
                return cases(e)
            if isinstance(e, program.Named):
                return ev(e.body)
            assert_never(e)

        return ev(expr)

    def _parameter_piece(self, name: str) -> Piece:
        """A parameter as a constant part, keyed by its declared dims."""
        dims = self.scope.program.parameters[name].dims
        frame = self.scope.data.parameters[name].select(*dims, pl.col('value').cast(pl.Float64).alias('cval'))
        return Piece(dims, frame, 'const', parameters=frozenset({name}))

    def _variable_piece(self, name: str) -> Piece:
        """A variable as a term with unit coefficients."""
        dims = self.scope.program.variables[name].dims
        frame = self.scope.variables[name].frame.select(
            *dims, 'var_label', pl.lit(1.0, dtype=pl.Float64).alias('coeff')
        )
        return Piece(dims, frame, 'term', presences=self._variable_presences(name, dims))

    def _variable_presences(self, name: str, dims: tuple[str, ...]) -> tuple[Presence, ...]:
        """Presence only for a masked variable under ``absence: undefined``, keyed explicitly as [`Presence`][] requires."""
        declaration = self.scope.program.variables[name]
        propagates = declaration.where is not None and declaration.absence == 'undefined'
        return (Presence(_presence(self.scope.variables[name], dims, 'var_label'), dims),) if propagates else ()

    def _solved_piece(self, name: str) -> Piece:
        """A variable at its primal — the const piece a read compiles it to, carrying the presence its term would.

        Under ``absence: zero`` it is filled with zero over the unmasked product,
        because a nonlinear read such as ``0.5 ** x`` tells a zero from no row.
        """
        assert self.solution is not None
        held, declaration = self.scope.variables[name], self.scope.program.variables[name]
        dims = declaration.dims
        keys = dims or (UNIT,)
        rows = held.frame.select(*keys).with_columns(held.share(self.solution.primal).alias('cval'))
        if declaration.where is not None and declaration.absence == 'zero':
            everywhere = masked(self.scope, dims, None).select(*keys)
            rows = join_on(everywhere, rows, keys, 'left').with_columns(pl.col('cval').fill_null(0.0))
        return Piece(dims, rows.select(*dims, 'cval'), 'const', presences=self._variable_presences(name, dims))

    def _dual_piece(self, name: str) -> Piece:
        """``dual(name)`` at the solve's row duals — one value per row the constraint built, and present exactly there.

        Raises:
            SpecsolveError: The solve left no duals.
        """
        solution = self.solution
        assert solution is not None
        if solution.dual is None:
            assert solution.no_duals is not None, 'a solve without duals says why'
            raise SpecsolveError(solution.no_duals)
        held, dims = solution.constraints[name], self.scope.program.constraints[name].dims
        frame = held.valued(dims, solution.dual.alias('cval'))
        return Piece(dims, frame, 'const', presences=(Presence(_presence(held, dims, 'row'), dims),))

    def _read_divisor(self, divisor: Piece) -> Piece:
        """*divisor* absent wherever it is zero, which is how a reported quotient reads a zero divisor.

        *divisor* is already one value per coordinate ([`_added_up`][]). The
        presence admits every coordinate but the zeros, not every coordinate
        the divisor has: a divisor parameter short of a row stays a null for
        the refusal to find.
        """
        zeros = divisor.frame.filter(pl.col('cval') == 0)
        if divisor.dims:
            present = masked(self.scope, divisor.dims, None).select(*divisor.dims)
            present = present.join(zeros.select(*divisor.dims), on=list(divisor.dims), how='anti')
        else:
            count = zeros.select(pl.len().alias('__zeros__'))
            present = pl.LazyFrame({PRESENT: [True]}).join(count, how='cross').filter(pl.col('__zeros__') == 0)
            present = present.select(PRESENT)
        return replace(divisor, presences=(*divisor.presences, Presence(present, divisor.dims)))

    def _added_up(
        self, compiled: CompiledExpression, operand: program.Expression, *, absent: Literal['zero', 'spreads']
    ) -> CompiledExpression:
        """*compiled* as one const piece with one value per coordinate — a divisor, or a power's base or exponent.

        A sum reaches ``/`` and ``**`` still holding one row per summand
        ([`_sum_piece`][]), so an operand with a reduction under it is totalled
        first ([`_totalled`][]). Several pieces are added up too: addition does
        not distribute over ``/`` or ``**``. At a build, *absent* is what a
        parameter with no row adds ([`summed_onto`][]): a divisor ``spreads``,
        so the null coefficient is refused naming the parameter, and a power's
        operand reads it as ``zero``. At a read, several pieces add up null
        where no piece has a value, so a divisor's hole is reported rather than
        divided by an invented zero. An operand carrying a variable passes
        through, for the plan-boundary assert behind it.
        """
        if compiled.terms or compiled.quads:
            return compiled
        pieces = compiled.consts
        if len(pieces) == 1:
            if all(fan_in(node) == 'one-to-one' for node in program.walk(operand)):
                return compiled
            return CompiledExpression((), (replace(pieces[0], frame=_totalled(pieces[0])),))
        dims, restrictions = self.scope.spanned(pieces), absence_restrictions(pieces)
        carrier = masked(self.scope, dims, None)
        for restriction in restrictions:
            carrier = restriction.restrict(carrier, restriction.keyed_by or ())
        added = self.summed_onto(pieces, carrier, absent='hole' if self.solution is not None else absent)
        parameters = frozenset[str]().union(*(p.parameters for p in pieces))
        return CompiledExpression(
            (), (Piece(dims, added, 'const', presences=tuple(restrictions), parameters=parameters),)
        )

    def summed_onto(
        self, pieces: Sequence[Piece], carrier: pl.LazyFrame, *, absent: Literal['zero', 'hole', 'spreads']
    ) -> pl.LazyFrame:
        """Const *pieces* added per coordinate onto *carrier* — its columns, then ``cval``.

        *carrier* holds one row per coordinate of
        [`spanned`][specsolve.relational.engine.scope.Scope.spanned], restricted by
        the caller to where every variable under the pieces exists. *absent*
        is what a piece with no value adds: ``zero``; ``hole``, null where no
        piece has a value; ``spreads``, null where any piece has none.
        """
        assert pieces, 'an expression compiles to at least one piece'
        assert all(p.kind == 'const' for p in pieces), 'a read compiles every variable to its value'
        columns = [f'__piece {i}__' for i in range(len(pieces))]
        for p, column in zip(pieces, columns, strict=True):
            carrier = join_on(carrier, constant_scalar(p).rename({'cval': column}), p.dims, 'left')
        total = pl.sum_horizontal(columns, ignore_nulls=absent != 'spreads')
        if absent == 'hole':
            total = pl.when(pl.any_horizontal([pl.col(c).is_not_null() for c in columns])).then(total).otherwise(None)
        return carrier.select(pl.exclude(columns), total.alias('cval'))

    # ------------------------------------------------------------------
    # shape operators — one dim rewritten per piece
    # ------------------------------------------------------------------

    def _sum_piece(self, p: Piece, over: tuple[str, ...], context: str) -> Piece:
        """Drop the summed dims — not an aggregate; the terminal ``sum(coeff)`` at assembly collapses the rows.

        Constructed rather than ``replace``d so presence is dropped: a reduction
        skips absent slots.
        """
        missing = [d for d in over if d not in p.dims]
        if missing and p.kind == 'const':
            refuse_a_piece_without_the_dims(p, missing, context, f'sum(over={list(over)})')
        keep = tuple(d for d in p.dims if d not in over)
        scale = math.prod(self.scope.data.cardinality[d] for d in missing)
        frame = p.frame.select(*keep, *p.carried)
        if scale != 1:
            frame = frame.with_columns(pl.col(p.value_column) * scale)
        return self._empty_sums(Piece(keep, frame, p.kind, region=region_over(p.region, keep), parameters=p.parameters))

    def _group_piece(self, p: Piece, g: program.GroupSum, context: str) -> Piece:
        """Relabel the dims the direction consumes to the ones it produces, through its relation.

        No aggregate, as for [`_sum_piece`][]: a keyed relation neither
        duplicates nor drops a term, and a bare one fans out into the
        many-to-many sum.
        """
        over = g.direction.consumed_dims
        missing = [d for d in over if d not in p.dims]
        if missing:
            refuse_a_piece_without_the_dims(p, missing, context, f'sum(by=) over {list(over)}')
        return self._empty_sums(self._remap_piece(p, g))

    def _empty_sums(self, p: Piece) -> Piece:
        """Reduced *p* with a zero row at every coordinate of its dims that no member lands in.

        An empty sum is zero, not a hole, and [`coverage.constant_side`][]
        cannot tell the two apart, so the zero is written here: a dimension
        with no labels, a group its relation maps no member to, and a member a
        ``cases:`` region leaves out all land nowhere. Asked of the piece
        rather than the relation for that last reason. A parameter short of a
        member is [`coverage.refuse_short_constants`][]' question. Only for a
        constant part: a row with no terms is not built.
        """
        if p.kind != 'const':
            return p
        if not p.dims:
            zero = pl.LazyFrame({'cval': [0.0]}, schema={'cval': pl.Float64})
            return replace(p, frame=pl.concat([p.frame.select(*p.carried), zero.select(*p.carried)]))
        first = self.scope.data.dimensions[p.dims[0]].select(pl.col('val').alias(p.dims[0]))
        universe = self.scope.spread(first, p.dims[1:])
        empty = universe.join(p.frame.select(*p.dims).unique(), on=list(p.dims), how='anti')
        zeros = empty.with_columns(pl.lit(0.0, dtype=pl.Float64).alias('cval')).select(*p.dims, *p.carried)
        return replace(p, frame=pl.concat([p.frame, zeros]))

    def _at_piece(self, p: Piece, a: program.Pullback, context: str) -> Piece:
        """Spread the consumed dims back out over the produced ones — the adjoint of a group.

        The join fans out, so a later reduction can bring two copies of one
        ``var_label`` into a row, where the terminal aggregate adds them. Unlike
        a group it is pointwise, so it reports absence.
        """
        absent = [d for d in a.direction.consumed_dims if d not in p.dims]
        assert not absent, f'in {context}: a pullback through {absent}, which the expression does not span'
        remapped = self._remap_piece(p, a)
        return replace(remapped, presences=self._pulled_back_presences(p, a))

    def _pulled_back_presences(self, p: Piece, a: program.Pullback) -> tuple[Presence, ...]:
        """Where a pullback's variables exist, keyed by the fine dims they now span, as [`Presence`][] requires.

        [`_remap_piece`][]'s inner join swallows both the operand's absence and
        the relation's; unreported, the row survives to assert ``x <= 0`` where
        the model said nothing.
        """
        joined = a.direction.joined_dims
        fine = (*joined, *a.direction.produced_dims)
        table = mapping(self.scope.data.relations, a.direction)
        reachable = landed(table, a).unique()
        if not p.presences:
            total = math.prod(self.scope.data.cardinality[d] for d in fine)
            reached = reachable.select(pl.len()).pipe(collected).item()
            return () if reached == total else (Presence(reachable, fine),)

        def pulled(presence: Presence) -> Presence:
            keys = presence.keys(p.dims)
            if not keys:
                return Presence(presence.restrict(reachable, keys), fine)
            carries_targets = all(i in keys for i in (*a.direction.consumed_dims, *joined))
            source, keys = (
                (presence.frame, keys) if carries_targets else (self.scope.widen(presence.frame, keys, p.dims), p.dims)
            )
            return Presence(*walk_join(source, table, a, keys))

        return tuple(pulled(x) for x in p.presences)

    def _remap_piece(self, p: Piece, node: program.GroupSum | program.Pullback) -> Piece:
        """Trade the dims *node*'s direction consumes for the ones it produces, through its relation."""
        frame, dims = walk_join(p.frame, mapping(self.scope.data.relations, node.direction), node, p.dims, p.carried)
        return Piece(dims, frame, p.kind, region=region_over(p.region, dims), parameters=p.parameters)


def _zero(e: program.Expression) -> bool:
    return isinstance(e, program.Constant) and float(e.value) == 0


def _scattered(at: pl.Series, values: pl.Series, size: int) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """*values* moved to the positions *at* names, one pass, order checked."""
    indices = at.to_numpy()
    written = np.zeros(size, dtype=bool)
    written[indices] = True
    if not written.all():
        msg = 'a parameter passed the density gate but does not cover the coordinate product'
        raise SpecsolveError(msg)

    out = np.empty(size, dtype=np.float64)
    out[indices] = values.to_numpy()
    return out
