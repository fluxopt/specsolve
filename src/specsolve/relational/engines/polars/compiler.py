"""Logical plan → polars. Lazy: nothing is read, nothing is executed.

The language compiles a spec to a plan; this compiles the plan to a query,
in the [`Scope`][specsolve.relational.engines.polars.scope.Scope] the model's names
resolve in.

Column conventions, relied on by the engine:

===================  ==========================================
frame                columns
===================  ==========================================
dimension table      ``val``, ``ord``
relation table       its declared columns, each under its own name
parameter table      ``dims…``, ``value``
variable frame       ``dims…``, ``var_label``
term fragment        ``dims…``, ``var_label``, ``coeff``
quad fragment        ``dims…``, ``var_label``, ``var_label_2``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, assert_never

import numpy as np
import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError
from specsolve.relational.collect import polars_engine
from specsolve.relational.engines.polars.fragments import (
    PRESENT,
    CompiledExpression,
    Presence,
    TermFragment,
    absence_restrictions,
    acted_along,
    both_regions,
    constant_scalar,
    fan_in,
    join_mul,
    join_on,
    join_pow,
    join_quad,
    map_fragments,
    negate,
    propagate_absence,
    refuse_a_fragment_without_the_dims,
    region_over,
)
from specsolve.relational.engines.polars.predicates import Carrier, compile_predicate, falsy_if_null, masked
from specsolve.relational.engines.polars.reindex import translate_fragment, window_fragment
from specsolve.relational.engines.polars.relations import landed, mapping, walk_join
from specsolve.relational.engines.polars.scope import UNIT, Scope

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve.relational.engines.polars.labels import Labelled


def _totalled(p: TermFragment) -> pl.LazyFrame:
    """Const fragment *p* added up to one ``cval`` per coordinate, null where any of its rows is.

    A null summand is a divisor's hole under the sum, so the total keeps it
    for the refusal rather than adding up the rest.
    """
    total = pl.when(pl.col('cval').is_null().any()).then(None).otherwise(pl.col('cval').sum()).alias('cval')
    return p.frame.group_by(p.dims).agg(total) if p.dims else p.frame.select(total)


def _presence(held: Labelled, dims: tuple[str, ...], label: str) -> pl.LazyFrame:
    """The coordinates a declaration's rows exist at.

    A **scalar** declaration has none, and ``select()`` over no dims is the
    empty frame polars cannot represent, so the marker column carries the one
    bit left: whether the row is there at all. It is renamed from the *label*
    column, never a ``pl.lit()`` — a select of literals alone is length 1
    whatever it selects from, so an absent scalar would come back present.
    """
    if dims:
        return held.frame.select(*dims)
    return held.frame.select(pl.col(label).alias(PRESENT))


@dataclass(frozen=True)
class Solution:
    """What a solve left, for a compiler reading a named expression at it.

    Attached, a variable compiles to its primal and ``dual(c)`` to the
    constraint's row duals, as const fragments. ``dual`` is ``None`` where the
    solve left no duals, and ``no_duals`` then says why, which is what reading
    one raises.
    """

    primal: pl.Series
    dual: pl.Series | None
    constraints: Mapping[str, Labelled]
    no_duals: str | None


@dataclass(frozen=True)
class PolarsCompiler:
    """Turn plan nodes into polars queries over the model's tidy frames.

    ``scope`` is what every query is written against
    ([`Scope`][specsolve.relational.engines.polars.scope.Scope]). ``solution`` is
    set on the compiler a read builds and on no other: with it every variable
    and every ``dual(c)`` compiles to a value ([`Solution`][]).
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
        """*frame* with *param* attached **by position**, or ``None`` to join.

        Each parameter row's slot is its [`row_major`][specsolve.relational.engines.polars.scope.Scope.row_major] position and its
        value is scattered there — the table's row order is nothing, and
        ``_scattered`` refuses a product any slot of which nothing wrote.

        Attached by position only where all three hold:

        * the parameter's dims are exactly the variable's, in the same order —
          fewer broadcast, more is already refused, a different order is a
          different row-major walk
        * the variable declares no ``where`` — a mask makes the label frame a
          subset of the product and position stops lining up
        * the parameter is dense over that product, its height equal to the
          product of the cardinalities attaching cached

        Duplicate coordinates would break density without changing the height,
        and the door refuses them before this.
        """
        declaration = self.scope.program.parameters[param]
        if v.where is not None or tuple(declaration.dims) != tuple(v.dims) or not v.dims:
            return None

        expected = math.prod(self.scope.data.cardinality[d] for d in v.dims)
        table = self.scope.data.parameters[param]
        if table.select(pl.len()).collect().item() != expected:
            return None

        position = self.scope.row_major(v.dims, self.scope.ordinal_of)
        pairs = table.select(position.alias('__at__'), pl.col('value')).collect(engine=polars_engine())
        return frame.with_columns(pl.Series(alias, _scattered(pairs['__at__'], pairs['value'], expected)))

    # ------------------------------------------------------------------
    # expressions → fragments
    # ------------------------------------------------------------------

    def expression(
        self, expr: program.Expression, context: str, *, quadratic: bool = False, reported: bool = False
    ) -> CompiledExpression:
        """Compile an expression into term, quadratic and const fragments.

        *quadratic* is the position's ceiling, passed by the caller that knows
        it: the objective can hold a product of two variables and a constraint
        row cannot. *reported* is set by the caller that reads a value rather
        than building a row, where a quotient is absent wherever its divisor
        is zero ([`_read_divisor`][]). The language has already refused what
        it refuses (``mathspec.degree``), so this is the **plan-boundary backstop** —
        a degree-2 node arriving by any other route dies here rather than
        becoming a term whose second variable is silently dropped.

        No join in the walk maintains order; every consumer verifies order
        where it reads.
        """

        def product(a: CompiledExpression, b: CompiledExpression) -> CompiledExpression:
            """``a * b``, distributed over both operands' fragment lists.

            Every pairing is formed and each is formed once, **including both
            mixed products**: where the two factors each carry a variable and a
            constant part, ``a.terms`` against ``b.consts`` and ``b.terms``
            against ``a.consts`` are different terms of the model, and dropping
            either answers something else.

            Degree is the language's, decided before a plan exists to
            compile, so the two shapes with nowhere to go here are
            invariants of a checked plan rather than refusals of a file: a
            cubic product has no third label column, and a quadratic one is
            unrepresentable in a position whose caller compiled it as affine.

            A constant piece of the product owes a factor's parameters only
            where the *other* factor carries no variable: a parameter a
            variable stands with in a product is a coefficient, whichever
            piece it lands in ([`TermFragment.parameters`][]).
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
            """``a / b``, where *b* is one variable-free factor.

            That it is *one* is ``degree.check_binary``'s answer, given at load
            with no data attached, so a divisor that adds never reaches a plan
            from the math. *b* is still added up first, to the one value per
            coordinate the join wants: a sum reaches it one row per summand.
            """
            a, b = ev(e.numerator), self._added_up(ev(e.divisor), e.divisor)
            assert not (b.terms or b.quads), f'in {context}: a divisor carrying a variable reached the compiler'
            assert len(b.consts) == 1, 'a divisor that adds is refused at load'
            inv = self._read_divisor(b.consts[0]) if reported else b.consts[0]
            terms = tuple(join_mul(t, inv, t.kind, divide=True) for t in a.terms)
            quads = tuple(join_mul(q, inv, 'quad', divide=True) for q in a.quads)
            consts = tuple(join_mul(x, inv, 'const', divide=True) for x in a.consts)
            return CompiledExpression(terms, consts, quads)

        def power(e: program.Power) -> CompiledExpression:
            """``a ** b``, where neither side carries a variable.

            The language refuses one that does in the math (``mathspec.degree``),
            before a plan exists to carry it, so a variable under a power is an
            invariant here rather than a refusal — folding its coefficient into
            a base is what the assert stands in front of. At a read a variable
            is its value, and each side is added up first, as a divisor is.
            """
            a, b = self._added_up(ev(e.base), e.base), self._added_up(ev(e.exponent), e.exponent)
            assert not (a.terms or a.quads or b.terms or b.quads), (
                f'in {context}: a power over variables reached the compiler'
            )
            assert len(a.consts) == 1 and len(b.consts) == 1, 'a base or exponent that adds is refused at load'
            return CompiledExpression((), (join_pow(a.consts[0], b.consts[0]),))

        def shaped(
            e: program.Sum | program.GroupSum | program.Pullback | program.Translate | program.WindowSum,
            rewrite: Callable[[TermFragment], TermFragment],
        ) -> CompiledExpression:
            """One shape operator applied to its compiled operand, absence pushed in by the node's own fan-in.

            An output row of a node that is not one-to-one mixes several input
            slots, so absence has to reach the operand before the rewrite
            consumes it ([`propagate_absence`][]); [`fan_in`][] says which.
            """
            inner = ev(e.operand)
            if fan_in(e) != 'one-to-one':
                assert isinstance(e, program.Sum | program.GroupSum | program.WindowSum), 'only a sum fans in'
                inner = propagate_absence(inner, self.scope, acted_along(e))
            return map_fragments(inner, rewrite)

        def region(r: program.Region) -> CompiledExpression:
            """One region's value, kept only where that region applies."""
            on = tuple(d for d in self.scope.program.dimensions if d in r.when.dims)
            truth = masked(self.scope, on, r.when).select(*on) if on else None

            def relaxed(x: Presence, dims: tuple[str, ...]) -> Presence:
                """A region's presence, silent about the coordinates the region does not claim.

                A presence unmakes a constraint row wherever the variable
                under it is absent, and left alone a region's would do that
                across the whole frame — a commitment file's ``otherwise``
                shifts with no ``edge=``, so it is absent at the first
                snapshot, and the row every other region does cover would go
                with it. Widening it by the region's complement says what is
                true instead: outside its own region a region requires
                nothing, and the regions being disjoint, each coordinate is
                still held to the one region that claims it.

                A mask reading no dimension is the same question with a
                one-row answer: the complement is the whole frame where the
                constant is false and empty where it is true, so a region
                that claims nothing widens to requiring nothing.
                """
                have = x.keys(dims)
                keys = tuple(dict.fromkeys((*have, *on)))
                complement = masked(self.scope, on, ~r.when)
                elsewhere = complement.select(*on) if on else complement.select(UNIT)
                widened = [self.scope.widen(x.frame, have, keys), self.scope.widen(elsewhere, on, keys)]
                return Presence(pl.concat(widened, how='vertical_relaxed').unique(), keys)

            def kept(p: TermFragment) -> TermFragment:
                """One fragment cut down to the region's coordinates.

                An inner join rather than a semi-join: a value narrower than
                the mask has to *gain* the mask's dims, so a case that is one
                number still lands a row at every coordinate it claims.

                A mask that reads **no dimension** — a ``when`` of ``true``,
                a scalar switch, and the ``otherwise`` that is the negation of
                either — has no coordinate set to join against, so it filters
                the piece by its own constant instead. The presence still
                relaxes: a constant that is false leaves the region claiming
                nothing, and a region claiming nothing may not unmake a row.
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

            return map_fragments(ev(r.value), kept)

        def cases(e: program.Cases) -> CompiledExpression:
            """Every region added.

            No region is ranked against another and none is subtracted back
            out: the language proved them apart before any data attached, so a
            coordinate is carried by exactly one of them and the rest are
            empty there. Adding is therefore the whole of it, and the same
            concatenation `Add` does.
            """
            built = [region(r) for r in e.regions]
            return CompiledExpression(
                tuple(f for c in built for f in c.terms),
                tuple(f for c in built for f in c.consts),
                tuple(f for c in built for f in c.quads),
            )

        def ev(e: program.Expression) -> CompiledExpression:
            if isinstance(e, program.Constant):
                frame = pl.LazyFrame({'cval': [float(e.value)]}, schema={'cval': pl.Float64})
                return CompiledExpression((), (TermFragment((), frame, 'const'),))
            if isinstance(e, program.Parameter):
                return CompiledExpression((), (self._parameter_fragment(e.name),))
            if isinstance(e, program.Variable):
                if self.solution is None:
                    return CompiledExpression((self._variable_fragment(e.name),), ())
                return CompiledExpression((), (self._solved_fragment(e.name),))
            if isinstance(e, program.Dual):
                assert self.solution is not None, (
                    f'in {context}: a dual reached a build — the language keeps one out of the math'
                )
                return CompiledExpression((), (self._dual_fragment(e.constraint),))
            if isinstance(e, program.Negate):
                return map_fragments(ev(e.operand), negate)
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
                return shaped(e, lambda p: self._sum_fragment(p, e.over, context))
            if isinstance(e, program.GroupSum):
                return shaped(e, lambda p: self._group_fragment(p, e, context))
            if isinstance(e, program.Pullback):
                return shaped(e, lambda p: self._at_fragment(p, e, context))
            if isinstance(e, program.Translate):
                return shaped(e, lambda p: translate_fragment(self.scope, p, e, context))
            if isinstance(e, program.WindowSum):
                return shaped(e, lambda p: window_fragment(self.scope, p, e, context))
            if isinstance(e, program.Cases):
                return cases(e)
            if isinstance(e, program.Named):
                return ev(e.body)
            assert_never(e)

        return ev(expr)

    def _parameter_fragment(self, name: str) -> TermFragment:
        """A parameter as a constant part, keyed by its declared dims.

        One row per coordinate, which the engine enforces by refusing a
        duplicated one.
        """
        dims = self.scope.program.parameters[name].dims
        frame = self.scope.data.parameters[name].select(*dims, pl.col('value').cast(pl.Float64).alias('cval'))
        return TermFragment(dims, frame, 'const', parameters=frozenset({name}))

    def _variable_fragment(self, name: str) -> TermFragment:
        """A variable as a term with unit coefficients.

        Presence is what makes absence *propagate*, and it is attached only
        where the declaration asks for it — decided before any data is read.
        Two declarations carry none: an unmasked variable, which exists at every
        coordinate of its dims and could restrict nothing, and one declaring
        ``absence: zero``, whose missing coordinates hold a quantity that *is*
        zero rather than one with no value. Both then leave the term simply
        absent from the rows it does not reach, which is the same arithmetic —
        only the second had a choice about it.

        ``keyed_by`` is stated rather than left to its ``None`` default,
        because dims are rewritten downstream while the presence frame is not
        — the hazard [`Presence`][] names.
        """
        dims = self.scope.program.variables[name].dims
        frame = self.scope.variables[name].frame.select(
            *dims, 'var_label', pl.lit(1.0, dtype=pl.Float64).alias('coeff')
        )
        return TermFragment(dims, frame, 'term', presences=self._variable_presences(name, dims))

    def _variable_presences(self, name: str, dims: tuple[str, ...]) -> tuple[Presence, ...]:
        declaration = self.scope.program.variables[name]
        propagates = declaration.where is not None and declaration.absence == 'undefined'
        return (Presence(_presence(self.scope.variables[name], dims, 'var_label'), dims),) if propagates else ()

    def _solved_fragment(self, name: str) -> TermFragment:
        """A variable at its primal — the const fragment a read compiles it to, carrying the presence its term would.

        Under ``absence: zero`` a masked variable *is* zero where it has no
        row, and a nonlinear read tells a zero from no row where affine
        arithmetic cannot — ``0.5 ** x`` is 1 at the one and nothing at the
        other — so its value is laid over the unmasked coordinate product,
        zero where the variable is absent. A build's term is right as it is:
        an absent term contributes nothing to a row either way.
        """
        assert self.solution is not None
        held, declaration = self.scope.variables[name], self.scope.program.variables[name]
        dims = declaration.dims
        keys = dims or (UNIT,)
        rows = held.frame.select(*keys).with_columns(held.share(self.solution.primal).alias('cval'))
        if declaration.where is not None and declaration.absence == 'zero':
            everywhere = masked(self.scope, dims, None).select(*keys)
            rows = join_on(everywhere, rows, keys, 'left').with_columns(pl.col('cval').fill_null(0.0))
        return TermFragment(dims, rows.select(*dims, 'cval'), 'const', presences=self._variable_presences(name, dims))

    def _dual_fragment(self, name: str) -> TermFragment:
        """``dual(name)`` at the solve's row duals — one value per row the constraint built, and present exactly there.

        A row a mask or an absent variable unmade has no dual, so unlike a
        variable's the presence is attached whether or not the declaration is
        masked: which rows stand is known only once they are built.

        Raises:
            SpecsolveError: The solve left no duals — the sentence
                [`dual`][specsolve.relational.result.Result.dual] gives.
        """
        solution = self.solution
        assert solution is not None
        if solution.dual is None:
            assert solution.no_duals is not None, 'a solve without duals says why'
            raise SpecsolveError(solution.no_duals)
        held, dims = solution.constraints[name], self.scope.program.constraints[name].dims
        frame = held.frame.select(*dims).with_columns(held.share(solution.dual).alias('cval'))
        return TermFragment(dims, frame, 'const', presences=(Presence(_presence(held, dims, 'row'), dims),))

    def _read_divisor(self, divisor: TermFragment) -> TermFragment:
        """*divisor* absent wherever it is zero, which is how a reported quotient reads a zero divisor.

        *divisor* is already one value per coordinate ([`_added_up`][]). The
        presence admits every coordinate but the zeros, rather than every
        coordinate the divisor has: a divisor parameter short of a row has to
        stay a null for the refusal to find, since a missing row is not absence.
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

    def _added_up(self, compiled: CompiledExpression, operand: program.Expression) -> CompiledExpression:
        """*compiled* as one const fragment with one value per coordinate — a divisor, or a power's base or exponent.

        ``/`` and ``**`` do not distribute over a sum, and a sum reaches them
        still holding one row per summand ([`_sum_fragment`][]), so an operand
        with a reduction under it is added up first ([`_totalled`][]), at a
        build as at a read. One without holds one row per coordinate already,
        and is not scanned again: at a build that scan is as long as the
        operand. Several pieces are added up null where no piece has a value,
        so a divisor with a hole still reports it rather than dividing by a
        zero the fill invented. Several pieces are an operand that adds, which the
        language refuses at a build: there they pass through, as does an
        operand carrying a variable, for the plan-boundary assert behind it.
        """
        if compiled.terms or compiled.quads:
            return compiled
        fragments = compiled.consts
        if len(fragments) == 1:
            if all(fan_in(node) == 'one-to-one' for node in program.walk(operand)):
                return compiled
            return CompiledExpression((), (replace(fragments[0], frame=_totalled(fragments[0])),))
        if self.solution is None:
            return compiled
        dims, restrictions = self.scope.spanned(fragments), absence_restrictions(fragments)
        carrier = masked(self.scope, dims, None)
        for restriction in restrictions:
            carrier = restriction.restrict(carrier, restriction.keyed_by or ())
        added = self.added(fragments, carrier, absent='hole')
        parameters = frozenset[str]().union(*(p.parameters for p in fragments))
        return CompiledExpression(
            (), (TermFragment(dims, added, 'const', presences=tuple(restrictions), parameters=parameters),)
        )

    def added(
        self, fragments: Sequence[TermFragment], carrier: pl.LazyFrame, *, absent: Literal['zero', 'hole', 'spreads']
    ) -> pl.LazyFrame:
        """Const *fragments* added per coordinate onto *carrier* — its columns, then ``cval``.

        *carrier* is the coordinate product the sum stands over, one row per
        coordinate of [`spanned`][specsolve.relational.engines.polars.scope.Scope.spanned], restricted by the caller to where every
        variable under the fragments exists — the rows a constraint over the
        same expression would keep. *absent* is what a piece with no value at
        a coordinate adds: ``zero``, what a read reports; ``hole``, the same
        except null where no piece has a value, what a divisor keeps so the
        hole is reported rather than divided by; ``spreads``, null wherever
        any piece has none, what arithmetic under a ``where`` reads (the
        absence rules).
        """
        assert fragments, 'an expression compiles to at least one fragment'
        assert all(p.kind == 'const' for p in fragments), 'a read compiles every variable to its value'
        columns = [f'__piece {i}__' for i in range(len(fragments))]
        for p, column in zip(fragments, columns, strict=True):
            carrier = join_on(carrier, constant_scalar(p).rename({'cval': column}), p.dims, 'left')
        total = pl.sum_horizontal(columns, ignore_nulls=absent != 'spreads')
        if absent == 'hole':
            total = pl.when(pl.any_horizontal([pl.col(c).is_not_null() for c in columns])).then(total).otherwise(None)
        return carrier.select(pl.exclude(columns), total.alias('cval'))

    # ------------------------------------------------------------------
    # shape operators — one dim rewritten per fragment
    # ------------------------------------------------------------------

    def _sum_fragment(self, p: TermFragment, over: tuple[str, ...], context: str) -> TermFragment:
        """Drop the summed dims — **not an aggregate**.

        The rows that carried them stay and collapse in the terminal
        ``sum(coeff)`` at assembly. Constructed rather than ``replace``d so
        ``presence`` is *dropped*: the absence rules read a reduction as
        skipping absent slots, so summing over a partly-masked dim reports
        nothing.
        """
        missing = [d for d in over if d not in p.dims]
        if missing and p.kind == 'const':
            refuse_a_fragment_without_the_dims(p, missing, context, f'sum(over={list(over)})')
        keep = tuple(d for d in p.dims if d not in over)
        scale = math.prod(self.scope.data.cardinality[d] for d in missing)
        frame = p.frame.select(*keep, *p.carried)
        if scale != 1:
            frame = frame.with_columns(pl.col(p.value_column) * scale)
        return TermFragment(keep, frame, p.kind, region=region_over(p.region, keep), parameters=p.parameters)

    def _group_fragment(self, p: TermFragment, g: program.GroupSum, context: str) -> TermFragment:
        """Relabel the dims the direction consumes to the ones it produces, through its relation.

        No aggregate either: a keyed relation holds one row per key and its
        columns were checked for containment at build time, so the join
        neither duplicates nor drops a term, and rows landing on one ``into``
        tuple are added by the terminal aggregate as ``Sum``'s are. A bare
        relation fans out instead — a member related to several targets lands
        a term in each — which is the many-to-many sum the language reads it
        as. A group is a sum, so it constructs rather than ``replace``s — see
        [`_sum_fragment`][].

        Several reads ride the same join.
        """
        over = g.direction.consumed_dims
        missing = [d for d in over if d not in p.dims]
        if missing:
            refuse_a_fragment_without_the_dims(p, missing, context, f'sum(by=) over {list(over)}')
        grouped = self._remap_fragment(p, g)
        if p.kind != 'const':
            return grouped
        return replace(grouped, frame=pl.concat([grouped.frame, self._empty_groups(grouped, g)]))

    def _empty_groups(self, p: TermFragment, g: program.GroupSum) -> pl.LazyFrame:
        """The produced combinations no member maps to, as constant rows worth zero.

        A group with no members contributes nothing, so on a constant side it
        holds a *value* — the empty sum — and not a hole. The two are the same
        missing row to [`coverage.constant_side`][]'s check,
        which reads what the fragment produced and cannot see why a label is
        absent, so the value is written down here where the reason is known.

        A read producing several columns lands on a *product* of targets, and a
        combination no member sits at is empty for the reason one unreached
        label is — so what the reached set is subtracted from is that product,
        at each coordinate of the joined dimensions the group is read under.

        Only for a constant part: an empty group contributes no *term*, and a
        row left with no terms is not built at all.
        """
        into = g.direction.produced_dims
        universe = self.scope.data.dimensions[into[0]].select(pl.col('val').alias(into[0]))
        for target in into[1:]:
            labels = self.scope.data.dimensions[target].select(pl.col('val').alias(target))
            universe = universe.join(labels, how='cross')
        spanned = [d for d in p.dims if d not in into]
        if spanned:
            universe = p.frame.select(spanned).unique().join(universe, how='cross')
        reached = landed(mapping(self.scope.data.relations, g.direction), g)
        empty = universe.join(reached, on=[*g.direction.joined_dims, *into], how='anti')
        return empty.with_columns(pl.lit(0.0, dtype=pl.Float64).alias('cval')).select(*p.dims, *p.carried)

    def _at_fragment(self, p: TermFragment, a: program.Pullback, context: str) -> TermFragment:
        """Spread the consumed dims back out over the produced ones — the adjoint of a group.

        The same mapping table as [`_group_fragment`][], joined on the other
        columns, so the join **fans out**: one row per consumed tuple lands on
        every produced tuple sharing it. Still one equi-join against a table
        the frame holds, so the locality class does not move.

        A pullback duplicates a label — the same ``var_label`` at every fine
        coordinate of its component — so a later reduction can bring two copies
        into one row, where the terminal aggregate adds them.

        Unlike the group it shares that join with, it **reports absence**:
        pointwise, so what the fine coordinate has is whatever the coarse slot
        it reads has, and a slot with nothing has to take the row with it.
        """
        absent = [d for d in a.direction.consumed_dims if d not in p.dims]
        assert not absent, f'in {context}: a pullback through {absent}, which the expression does not span'
        remapped = self._remap_fragment(p, a)
        return replace(remapped, presences=self._pulled_back_presences(p, a))

    def _pulled_back_presences(self, p: TermFragment, a: program.Pullback) -> tuple[Presence, ...]:
        """Where a pullback's variables exist, keyed by the fine dims they now span.

        Two absences reach the fine coordinate and [`_remap_fragment`][]'s
        inner join swallows both — the operand's own, and the **relation's**,
        where the map has no row for the key. Unreported, the term merely
        vanishes and its row survives to assert `x <= 0` where the model said
        nothing.

        A total map over an operand with nothing to report yields nothing
        rather than a restriction admitting everything. The key is stated
        rather than left implied because a later product widens the fragment's
        dims while this frame keeps the columns that matter — the hazard
        [`Presence`][] names.
        """
        joined = a.direction.joined_dims
        fine = (*joined, *a.direction.produced_dims)
        table = mapping(self.scope.data.relations, a.direction)
        reachable = landed(table, a).unique()
        if not p.presences:
            total = math.prod(self.scope.data.cardinality[d] for d in fine)
            reached = reachable.select(pl.len()).collect().item()
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

    def _remap_fragment(self, p: TermFragment, node: program.GroupSum | program.Pullback) -> TermFragment:
        """Trade the dims *node*'s direction consumes for the ones it produces, through its relation.

        One inner equi-join against [`mapping`][], keyed as [`walk_join`][]
        says. A group consumes the dims its direction is over
        ([`_group_fragment`][]); a pullback reads the same table backwards
        ([`_at_fragment`][]).
        """
        frame, dims = walk_join(p.frame, mapping(self.scope.data.relations, node.direction), node, p.dims, p.carried)
        return TermFragment(dims, frame, p.kind, region=region_over(p.region, dims), parameters=p.parameters)


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
