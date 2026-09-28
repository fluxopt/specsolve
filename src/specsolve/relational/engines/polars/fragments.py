"""The algebra of additive pieces: what an expression compiles *to*.

An LP row is a sum of pieces, so a compiled expression is a list of fragments
and every shape operator rewrites one. This module is that vocabulary and the
arithmetic over it — product, quotient, power, negation, and the absence rule
that decides which rows a reduction is allowed to see.

It holds no state and reads no data: everything here takes fragments and
returns fragments.

Column conventions, relied on by the engine:

===================  ==========================================
frame                columns
===================  ==========================================
term fragment        ``dims…``, ``var_label``, ``coeff``
quad fragment        ``dims…``, ``var_label``, ``var_label_2``, ``coeff``
const fragment       ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never

import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars._typing import JoinStrategy, MaintainOrderJoin


def join_on(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    dims: Sequence[str],
    how: JoinStrategy,
    maintain_order: MaintainOrderJoin | None = None,
) -> pl.LazyFrame:
    """``left.join(right)`` keyed by *dims* — a cross join where there are none.

    The empty coordinate product is one real row, so a scalar piece joins by
    crossing rather than by an empty key.
    """
    if dims:
        return left.join(right, on=list(dims), how=how, maintain_order=maintain_order)
    return left.join(right, how='cross', maintain_order=maintain_order)


#: The right-hand operand's value while a join holds both. The spaces make it
#: unrepresentable as a declared name, so it cannot collide with a dimension or
#: relation the model already has.
_RHS = '__rhs value__'

#: Carries the one bit a *scalar* declaration's presence frame has to say:
#: whether the declaration exists at all. Polars cannot hold a frame with rows
#: and no columns, and a scalar presence has no dim to be keyed by, so the
#: restriction becomes a cross join against this marker.
PRESENT = '__present__'


@dataclass(frozen=True)
class Presence:
    """Where the *variable* under a fragment exists, and what keys it.

    Not which rows the fragment's frame has. A fragment loses rows for two
    unrelated reasons and a constraint row reacts to only one: a **masked
    variable** is genuinely absent, a **sparse parameter** is a compressed
    dense array whose missing rows mean a zero coefficient (the data-attachment rules).
    Multiplied together the frame cannot tell them apart, so the variable's
    coordinates ride alongside.

    ``keyed_by`` is ``None`` where the frame is keyed by the fragment's own
    dims — the implied key survives the dims being rewritten downstream (a
    product broadcasts, a sum drops) where a stated one would silently become
    a claim about columns the frame never had. Only an acyclic ``shift``
    states it: the coordinates it vacates are an edge along *one* dimension,
    so the restriction is one column wide however many dims the fragment
    carries — where keying it by the fragment's dims would materialise the
    whole coordinate product to name an edge.
    """

    frame: pl.LazyFrame
    keyed_by: tuple[str, ...] | None = None

    def keys(self, fragment_dims: tuple[str, ...]) -> tuple[str, ...]:
        """The columns this presence restricts by, for a fragment over *fragment_dims*."""
        return self.keyed_by if self.keyed_by is not None else fragment_dims

    def restrict(self, frame: pl.LazyFrame, on: Sequence[str]) -> pl.LazyFrame:
        """Keep only the rows of *frame* this presence admits.

        Keyed by *on* this is a semi-join. A **scalar** declaration has no
        key, its presence being at most one row saying only whether it
        exists, so the question becomes a cross join: every row survives a
        present scalar and none survives an absent one — absence spreading
        through arithmetic (the operator rules) at no dimension.
        """
        if on:
            return frame.join(self.frame.select(list(on)), on=list(on), how='semi')
        return frame.join(self.frame.select(PRESENT), how='cross').drop(PRESENT)


#: What a fragment is a piece *of*. ``term`` and ``quad`` differ only in how
#: many label columns the coefficient multiplies.
Kind = Literal['term', 'quad', 'const']


@dataclass(frozen=True)
class TermFragment:
    """One additive piece of a compiled expression.

    Terms yield ``(dims…, var_label, coeff)``, quadratic terms
    ``(dims…, var_label, var_label_2, coeff)`` and const parts
    ``(dims…, cval)``. An LP row *is* a sum of pieces, so every shape operator
    rewrites one.
    """

    dims: tuple[str, ...]
    frame: pl.LazyFrame
    kind: Kind

    presences: tuple[Presence, ...] = ()
    """Where the variables under this fragment exist — see [`Presence`][].

    Empty is nothing to report: a constant fragment has no variable, and a
    reduction clears it, ``sum`` skipping absent slots rather than propagating
    them (the absence rules).

    A **tuple**: a quadratic term stands on two variables and is absent where
    either is, so the presences travel side by side and each consumer applies
    them in turn.
    """

    region: program.Mask | None = None
    """The region of a `Cases` this piece was built under.

    ``None`` where the piece stands over the whole frame, which is everything
    outside a ``cases:`` block. Set, it says the piece covers that region *by
    construction* and nowhere else — so the constant side's coverage check
    asks its question there rather than over every row, a null outside being
    another region's coordinate and a null inside still the hole it always was.

    It travels through the arithmetic, and a product of two regions is the
    conjunction: ``ramp_limit * previous_status`` has a value exactly where
    ``previous_status`` does. A reduction that drops a dim the region reads
    drops the region with it ([`region_over`][]): it can no longer say which
    of the rows summed into a coordinate it claimed.
    """

    parameters: frozenset[str] = frozenset()
    """The parameters standing as constant pieces under this fragment — what the coverage check asks of.

    A constant piece is a parameter in a variable-free additive position, ``hi``
    in ``x + hi`` or in ``sum(x) + sum(hi)``. A parameter a variable stands with
    in a product is a coefficient instead, and a sparse coefficient is a zero
    the absence rules allow, so a term carries none and a product strips the
    factor that stood beside a variable ([`PolarsCompiler.expression`][specsolve.relational.engines.polars.compiler.PolarsCompiler.expression]).
    A divisor is never owed here either: it has its own check.
    """

    @property
    def value_column(self) -> str:
        """``coeff`` where a variable is under it, ``cval`` otherwise."""
        return value_column(self.kind)

    @property
    def carried(self) -> list[str]:
        """The non-dim columns a projection has to keep."""
        return carried_columns(self.kind)


def refuse_a_fragment_without_the_dims(p: TermFragment, dims: list[str], context: str, operator: str) -> NoReturn:
    """Refuse a fragment an operator cannot act on, in the right class.

    Two different failures share this shape and must not share a class. A
    **constant part** lacking the dims is a file the language accepts and this
    engine cannot build — `check` passes, so `LanguageError` would be a lie — and
    it is reachable from ordinary YAML wherever a scalar is added beside a term.
    A **term** lacking them is not reachable that way: `dims_of` gives every
    term the frame dims at load, so reaching here means the plan is
    malformed.

    *operator* is the surface spelling, not the plan node: the reader wrote
    ``sum(by=…)``, and ``GroupSum`` is a word their file does not contain.
    """
    if p.kind == 'const':
        raise SpecsolveError(
            f'in {context}: {operator} acts along {dims}, which a constant part of the expression '
            f'does not carry, and specsolve cannot build that. A constant part compiles to its own '
            f'frame, so a fragment with no rows for {dims} has no slots for the operator to act on — '
            f'and under a mask, which slots those are is known only to the rows. Declare the parameter '
            f'over {dims} and supply it there: the model is the same and the number is unchanged.'
        )
    msg = f'in {context}: {operator} along {dims}, which the expression does not span'
    raise AssertionError(msg)


#: The label columns each kind carries, in the order a projection keeps them.
_LABELS: dict[Kind, list[str]] = {'term': ['var_label'], 'quad': ['var_label', 'var_label_2'], 'const': []}


def value_column(kind: Kind) -> str:
    """The value column a fragment of this kind carries."""
    return 'cval' if kind == 'const' else 'coeff'


def carried_columns(kind: Kind) -> list[str]:
    """The non-dim columns a projection of this fragment kind has to keep."""
    return [*_LABELS[kind], value_column(kind)]


@dataclass(frozen=True)
class CompiledExpression:
    """An expression as fragments: variable terms, quadratic terms, a constant part."""

    terms: tuple[TermFragment, ...]
    consts: tuple[TermFragment, ...]
    quads: tuple[TermFragment, ...] = ()


def constant_scalar(p: TermFragment) -> pl.LazyFrame:
    """The const fragment summed per coordinate: ``(dims…, cval)``."""
    if not p.dims:
        return p.frame.select(pl.col('cval').sum())
    return p.frame.group_by(p.dims).agg(pl.col('cval').sum())


def absence_restrictions(fragments: Sequence[TermFragment]) -> list[Presence]:
    """The presence frames a constraint's rows — or a read's — have to be contained in.

    Absence propagates into a comparison and drops the row (the absence
    rules): ``x + y >= 10`` where ``y`` is masked is not ``x >= 10``, it is no
    constraint at all. Only *variable* absence counts — a sparse parameter's
    missing rows mean a zero coefficient — which is why the fragment carries
    [`TermFragment.presences`][] separately from its frame.

    *Having* no dims is not *having nothing to restrict*: a masked scalar
    variable restricts every row of every constraint naming it, all or nothing.
    Each restriction leaves with its key spelled out — the fragment's dims
    where the presence implied them — since labelling cannot know the
    fragment it came from.
    """
    return [Presence(x.frame, x.keys(p.dims)) for p in fragments for x in p.presences]


#: How a node's output rows relate to its input slots, answered by
#: [`fan_in`][] for every node.
FanIn = Literal['one-to-one', 'many-to-one', 'one-to-many']


def fan_in(expression: program.Expression) -> FanIn:
    """How *expression*'s output rows relate to its input slots.

    Both classes other than ``'one-to-one'`` sum several input slots into an
    output row, so [`propagate_absence`][] runs before them.
    """
    if isinstance(expression, program.Named):
        return fan_in(expression.body)
    if isinstance(expression, (program.Sum, program.GroupSum)):
        return 'many-to-one'
    if isinstance(expression, program.WindowSum):
        return 'one-to-many'
    if isinstance(
        expression,
        (
            program.Constant,
            program.Parameter,
            program.Variable,
            program.Dual,
            program.Negate,
            program.Add,
            program.Multiply,
            program.Power,
            program.Divide,
            program.Pullback,
            program.Translate,
            program.Cases,
        ),
    ):
        return 'one-to-one'
    assert_never(expression)


def propagate_absence(compiled: CompiledExpression) -> CompiledExpression:
    """Restrict every fragment to where the *whole* expression exists.

    Addition is fragment concatenation, so ``x + size`` is two independent
    streams — right at row level, where the engine intersects the presences,
    but a **reduction** consumes the expression before any row exists. That is
    the difference between ``sum(x + size, over=f)``, which sums where the
    summand exists, and ``sum(x, over=f) + sum(size, over=f)``, which sums each
    operand over its own domain and reads the absent ``size`` as a zero (the
    absence and operator rules).

    Applied only where the key columns are dims the fragment carries: a
    restriction naming a dim a fragment lacks cannot speak about it.

    **Which operators need it is decided by their fan-in** ([`fan_in`][]),
    which the compiler reads.
    Many-to-one and one-to-many mix several input slots into an output row:
    the row-level intersection at assembly can say the *row* survives, never
    which of the slots behind it did, and a constant read from an absent slot
    is already inside the total by then. One-to-one is one output, one input,
    so the row either survives or does not and the intersection is exactly
    right without a pass here.

    **A fragment is never restricted by its own presence.** Its rows and its
    presence are built from one frame and rewritten in step — a product joins
    the rows and leaves the coordinates, a translation remaps both, a fill adds
    to both — so the rows are inside the coordinates by construction and the
    join could only return them all.

    The presence frame is not deduplicated first: a semi-join asks whether a
    key occurs, and occurring twice is still occurring, so the distinct would
    change no row.
    """
    absent = [(p, x) for p in (*compiled.terms, *compiled.quads, *compiled.consts) for x in p.presences]
    if not absent:
        return compiled

    def restrict(p: TermFragment) -> TermFragment:
        frame = p.frame
        for source, presence in absent:
            if source is p:
                continue
            on = list(presence.keys(source.dims))
            if all(d in p.dims for d in on):
                frame = presence.restrict(frame, on)
        return p if frame is p.frame else replace(p, frame=frame)

    return map_fragments(compiled, restrict)


def map_fragments(
    compiled: CompiledExpression,
    rewrite: Callable[[TermFragment], TermFragment],
) -> CompiledExpression:
    """Apply *rewrite* to every fragment, keeping the kinds apart.

    Rewriting one fragment at a time is what pointwise and bounded-halo
    locality mean; a node needing them together is global, and rejected at
    lowering. A quadratic fragment goes through the same rewrites as a linear
    one and for the same reason — a shape operator moves rows between
    coordinates and never looks at what the row *carries*, which is why the
    operators project through ``carried`` rather than naming columns.
    """
    return CompiledExpression(
        tuple(rewrite(p) for p in compiled.terms),
        tuple(rewrite(p) for p in compiled.consts),
        tuple(rewrite(p) for p in compiled.quads),
    )


def both_regions(a: program.Mask | None, b: program.Mask | None) -> program.Mask | None:
    """The region a product of two pieces stands over — where both of them do."""
    if a is None or b is None:
        return a or b
    return a if a == b else a & b


def region_over(region: program.Mask | None, dims: Sequence[str]) -> program.Mask | None:
    """*region* where a fragment over *dims* can still be cut to it, else ``None``."""
    return region if region is not None and region.dims <= set(dims) else None


def negate(p: TermFragment) -> TermFragment:
    return replace(p, frame=p.frame.with_columns(-pl.col(p.value_column)))


def join_mul(a: TermFragment, c: TermFragment, kind: Kind, divide: bool = False) -> TermFragment:
    """``a * c`` (or ``a / c``) where *c* is a const fragment.

    Joins on shared dims, broadcasts the rest. The right-hand value is renamed
    first: both sides may carry ``cval``, and a suffix collision would multiply
    a column by itself. The dims *c* contributes are broadcast, so the label
    says nothing about them.

    A divide joins **left**, so a coordinate the divisor has no value for
    yields a *null* coefficient instead of silently dropping the term. The row
    may still be masked out downstream, taking the null with it: the question
    is not whether the divisor is dense but whether it is defined where the
    model divides by it. A coordinate where the divisor is *absent* is not
    that question — the quotient is absent there too — so it leaves the frame
    here, before any check could count its null as a short parameter.

    At a build *c* is variable-free and contributes no absence: a sparse
    coefficient zeroes a term, it does not unmake the variable underneath it.
    At a read a const fragment may be a variable at its primal, carrying the
    presence its term would, so the presences of both sides travel out. The
    output dims may be wider than ``a.dims``, which is why the presence key
    travels with the fragment rather than being re-derived from dims here.

    A constant product owes both factors' parameters; a divisor is owed
    nowhere, having its own check; and a term owes none.
    """
    shared = [d for d in a.dims if d in c.dims]
    out_dims = a.dims + tuple(d for d in c.dims if d not in a.dims)
    right = c.frame.rename({'cval': _RHS})
    how = 'left' if divide else 'inner'
    joined = a.frame.join(right, on=shared, how=how) if shared else a.frame.join(right, how='cross')
    if divide:
        for presence in c.presences:
            joined = presence.restrict(joined, presence.keys(c.dims))

    value, rhs = pl.col(a.value_column), pl.col(_RHS)
    combined = value / rhs if divide else value * rhs
    out = value_column(kind)
    frame = joined.with_columns(combined.alias(out)).select(*out_dims, *carried_columns(kind))
    if kind != 'const':
        parameters = frozenset[str]()
    elif divide:
        parameters = a.parameters
    else:
        parameters = a.parameters | c.parameters
    return replace(
        a,
        dims=out_dims,
        frame=frame,
        kind=kind,
        presences=a.presences + c.presences,
        region=both_regions(a.region, c.region),
        parameters=parameters,
    )


def join_pow(a: TermFragment, b: TermFragment) -> TermFragment:
    """``a ** b``, both const fragments — one const fragment out.

    [`join_mul`][]'s shape with ``pow`` in place of ``*``, and the same
    reason for renaming the right-hand value first: both sides carry ``cval``.
    An **inner** join, unlike divide's left: an exponent with no value at a
    coordinate is not a division by a hole, it is a factor the model never
    stated, and a null base or exponent would poison the coefficient it
    multiplies rather than reporting anything.
    """
    shared = [d for d in a.dims if d in b.dims]
    out_dims = a.dims + tuple(d for d in b.dims if d not in a.dims)
    right = b.frame.rename({'cval': _RHS})
    joined = a.frame.join(right, on=shared, how='inner') if shared else a.frame.join(right, how='cross')
    frame = joined.with_columns(pl.col('cval').pow(pl.col(_RHS)).alias('cval')).select(
        *out_dims, *carried_columns('const')
    )
    return TermFragment(
        out_dims,
        frame,
        'const',
        presences=a.presences + b.presences,
        region=both_regions(a.region, b.region),
        parameters=a.parameters | b.parameters,
    )


def join_quad(a: TermFragment, b: TermFragment) -> TermFragment:
    """``a * b`` where both carry a variable — one quadratic fragment.

    A join on the dims the two share: aligned is an equi-join, broadcast joins
    on the coarser side, and the cross join is refused upstream
    (``mathspec.degree``).

    The second label is renamed on the way in, since both sides carry
    ``var_label`` and a suffix collision would pair a variable with itself —
    which ``p * p`` makes a *legal* fragment, so no error would catch it.

    Nothing is canonicalised here: which of ``x * y`` and ``y * x`` a pair is
    depends on column labels, which fragments do not carry until the engine
    places them ([`Assembly._build_objective`][specsolve.relational.engines.polars.assembly.Assembly._build_objective]).
    """
    shared = [d for d in a.dims if d in b.dims]
    out_dims = a.dims + tuple(d for d in b.dims if d not in a.dims)
    right = b.frame.rename({'var_label': 'var_label_2', 'coeff': _RHS})
    joined = a.frame.join(right, on=shared, how='inner') if shared else a.frame.join(right, how='cross')
    frame = joined.with_columns((pl.col('coeff') * pl.col(_RHS)).alias('coeff')).select(
        *out_dims, *carried_columns('quad')
    )
    return TermFragment(
        out_dims, frame, 'quad', presences=a.presences + b.presences, region=both_regions(a.region, b.region)
    )
