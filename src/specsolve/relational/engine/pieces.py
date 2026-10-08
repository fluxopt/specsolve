"""The additive pieces an expression compiles to — linear terms, quadratic terms or a constant — and the arithmetic over them.

Column conventions:

===================  ==========================================
frame                columns
===================  ==========================================
term piece           ``dims…``, ``var_label``, ``coeff``
quad piece           ``dims…``, ``var_label``, ``var_label_2``, ``coeff``
const piece          ``dims…``, ``cval``
===================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never

import polars as pl
from mathspec import program

from specsolve.errors import SpecsolveError
from specsolve.relational.engine.scope import join_on

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars._typing import JoinStrategy

    from specsolve.relational.engine.scope import Scope


#: The right-hand operand's value while a join holds both. The spaces make it
#: unrepresentable as a declared name.
_RHS = '__rhs value__'

#: The one column of a scalar presence frame: polars holds no frame with rows and no columns.
PRESENT = '__present__'


@dataclass(frozen=True)
class Presence:
    """Where the *variable* under a piece exists, and what keys it.

    Not which rows the piece's frame has: a sparse parameter's missing rows
    are a zero coefficient, a masked variable's are absence, and the frame
    cannot tell them apart.

    ``keyed_by`` is ``None`` where the frame is keyed by the piece's own
    dims, an implied key that stays right as downstream operators rewrite
    them. A shift or window edge states a narrower one.
    """

    frame: pl.LazyFrame
    keyed_by: tuple[str, ...] | None = None

    def keys(self, piece_dims: tuple[str, ...]) -> tuple[str, ...]:
        """The columns this presence restricts by, for a piece over *piece_dims*."""
        return self.keyed_by if self.keyed_by is not None else piece_dims

    def restrict(self, frame: pl.LazyFrame, on: Sequence[str]) -> pl.LazyFrame:
        """Keep only the rows of *frame* this presence admits.

        With no key, every row survives a present scalar and none an absent one.
        """
        if on:
            return frame.join(self.frame.select(list(on)), on=list(on), how='semi')
        return frame.join(self.frame.select(PRESENT), how='cross').drop(PRESENT)


#: What a piece carries: linear terms, quadratic terms, or a constant.
Kind = Literal['term', 'quad', 'const']


@dataclass(frozen=True)
class Piece:
    """One additive piece of a compiled expression."""

    dims: tuple[str, ...]
    frame: pl.LazyFrame
    kind: Kind

    presences: tuple[Presence, ...] = ()
    """Where the variables under this piece exist; a quadratic term is absent where either is.

    Empty for a constant piece and after a reduction, which skips absent slots.
    """

    region: program.Mask | None = None
    """The region of a `Cases` this piece covers by construction, or ``None``.

    The coverage check asks its question only there. A product carries the
    conjunction, and a reduction that drops a dim the region reads drops the
    region ([`region_over`][]).
    """

    parameters: frozenset[str] = frozenset()
    """The parameters standing as constant pieces under this piece — what the coverage check asks of.

    A term carries none, since a sparse coefficient is a zero, and a divisor has its own check.
    """

    @property
    def value_column(self) -> str:
        """``coeff`` where a variable is under it, ``cval`` otherwise."""
        return value_column(self.kind)

    @property
    def carried(self) -> list[str]:
        """The non-dim columns a projection has to keep."""
        return carried_columns(self.kind)


def refuse_a_piece_without_the_dims(p: Piece, dims: list[str], context: str, operator: str) -> NoReturn:
    """Refuse a piece an operator cannot act on, in the right class.

    A constant part lacking the dims is valid YAML this engine cannot build, so
    it is a `SpecsolveError`; a term always carries the frame dims from load, so
    one here is a malformed plan. *operator* is the spelling the user wrote.
    """
    if p.kind == 'const':
        raise SpecsolveError(
            f'in {context}: {operator} acts along {dims}, which a constant part of the expression '
            f'does not carry, and specsolve cannot build that. A constant part compiles to its own '
            f'frame, so a piece with no rows for {dims} has no slots for the operator to act on — '
            f'and under a mask, which slots those are is known only to the rows. Declare the parameter '
            f'over {dims} and supply it there: the model is the same and the number is unchanged.'
        )
    msg = f'in {context}: {operator} along {dims}, which the expression does not span'
    raise AssertionError(msg)


#: The label columns each kind carries, in the order a projection keeps them.
_LABELS: dict[Kind, list[str]] = {'term': ['var_label'], 'quad': ['var_label', 'var_label_2'], 'const': []}


def value_column(kind: Kind) -> str:
    """The value column a piece of this kind carries."""
    return 'cval' if kind == 'const' else 'coeff'


def carried_columns(kind: Kind) -> list[str]:
    """The non-dim columns a projection of this piece kind has to keep."""
    return [*_LABELS[kind], value_column(kind)]


@dataclass(frozen=True)
class CompiledExpression:
    """An expression as pieces: variable terms, quadratic terms, a constant part."""

    terms: tuple[Piece, ...]
    consts: tuple[Piece, ...]
    quads: tuple[Piece, ...] = ()


def ordered_sum(column: str) -> pl.Expr:
    """*column* added up in ascending order, so one set of numbers gives one sum to the last bit.

    Floating-point addition is not associative, and the rows reaching a sum
    come out of joins and group-bys in no fixed order. Summed as they come, a
    model's costs and right-hand sides would differ in the last bit from one
    build to the next, and so would its digest, which a saved answer is checked
    against. A plain ``sum``, sorted first or not, adds in an order polars
    chooses; a cumulative sum adds in row order. Zero where nothing is summed,
    as ``sum`` gives.
    """
    return pl.col(column).sort().cum_sum().last().fill_null(0.0)


def constant_scalar(p: Piece) -> pl.LazyFrame:
    """The const piece summed per coordinate: ``(dims…, cval)``."""
    if not p.dims:
        return p.frame.select(ordered_sum('cval'))
    return p.frame.group_by(p.dims).agg(ordered_sum('cval'))


def absence_restrictions(pieces: Sequence[Piece]) -> list[Presence]:
    """The presence frames a constraint's rows — or a read's — have to be contained in.

    Each leaves with its key spelled out, since labelling cannot know the piece it came from.
    """
    return [Presence(x.frame, x.keys(p.dims)) for p in pieces for x in p.presences]


#: How a node's output rows relate to its input slots.
FanIn = Literal['one-to-one', 'many-to-one', 'one-to-many']


def acted_along(expression: program.Sum | program.GroupSum | program.WindowSum) -> tuple[str, ...]:
    """The dims a node that is not one-to-one sums along."""
    if isinstance(expression, program.Sum):
        return expression.over
    if isinstance(expression, program.GroupSum):
        return tuple(expression.direction.consumed_dims)
    return (expression.along,)


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


def propagate_absence(compiled: CompiledExpression, scope: Scope, along: Sequence[str]) -> CompiledExpression:
    """Restrict every piece to where the *whole* expression exists.

    Needed before a node that is not one-to-one ([`fan_in`][]): the row-level
    intersection at assembly cannot say which input slots behind a row survived.
    A piece lacking a dim a restriction is keyed by is first repeated along
    it, never along a dim in *along*, which the operator refuses itself
    ([`refuse_a_piece_without_the_dims`][]). A piece is never restricted
    by its own presence: its rows are inside it by construction.
    """
    absent = [(p, x) for p in (*compiled.terms, *compiled.quads, *compiled.consts) for x in p.presences]
    if not absent:
        return compiled

    def restrict(p: Piece) -> Piece:
        frame, dims = p.frame, p.dims
        for source, presence in absent:
            if source is p:
                continue
            on = list(presence.keys(source.dims))
            missing = [d for d in on if d not in dims]
            if any(d in along for d in missing):
                continue
            if missing:
                dims = scope.in_declaration_order((*dims, *missing))
                frame = scope.spread(frame, missing).select(*dims, *p.carried)
            frame = presence.restrict(frame, on)
        return p if frame is p.frame else replace(p, dims=dims, frame=frame)

    return map_pieces(compiled, restrict)


def map_pieces(
    compiled: CompiledExpression,
    rewrite: Callable[[Piece], Piece],
) -> CompiledExpression:
    """Apply *rewrite* to every piece, keeping the kinds apart.

    A rewrite sees one piece at a time; a node needing them together is global and refused at lowering.
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
    """*region* where a piece over *dims* can still be cut to it, else ``None``."""
    return region if region is not None and region.dims <= set(dims) else None


def negate(p: Piece) -> Piece:
    return replace(p, frame=p.frame.with_columns(-pl.col(p.value_column)))


def _paired(a: Piece, b: Piece, right: pl.LazyFrame, how: JoinStrategy) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*a*'s frame joined to *right*, *b*'s frame with its value renamed, and the dims out, *a*'s first.

    The join is on the dims the two share, so *b* broadcasts over the rest.
    """
    joined = join_on(a.frame, right, [d for d in a.dims if d in b.dims], how)
    return joined, a.dims + tuple(d for d in b.dims if d not in a.dims)


def join_mul(a: Piece, c: Piece, kind: Kind, divide: bool = False) -> Piece:
    """``a * c`` (or ``a / c``) where *c* is a const piece, broadcast over the dims not shared.

    The right-hand value is renamed first: a suffix collision on ``cval`` would
    multiply a column by itself. A divide joins left, so a coordinate the
    divisor has no value for yields a null coefficient rather than dropping the
    term; a coordinate where the divisor is absent leaves the frame, the
    quotient being absent too. The presences of both sides travel out: at a
    read *c* may be a variable at its primal.
    """
    joined, out_dims = _paired(a, c, c.frame.rename({'cval': _RHS}), 'left' if divide else 'inner')
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


def join_pow(a: Piece, b: Piece) -> Piece:
    """``a ** b``, both const pieces — one const piece out.

    An inner join, unlike divide's left: a null base or exponent would poison
    the coefficient it multiplies rather than report anything.
    """
    joined, out_dims = _paired(a, b, b.frame.rename({'cval': _RHS}), 'inner')
    frame = joined.with_columns(pl.col('cval').pow(pl.col(_RHS)).alias('cval')).select(
        *out_dims, *carried_columns('const')
    )
    return Piece(
        out_dims,
        frame,
        'const',
        presences=a.presences + b.presences,
        region=both_regions(a.region, b.region),
        parameters=a.parameters | b.parameters,
    )


def join_quad(a: Piece, b: Piece) -> Piece:
    """``a * b`` where both carry a variable — one quadratic piece.

    The second label is renamed on the way in: a suffix collision would pair a
    variable with itself, which ``p * p`` makes a legal piece. Pairs are
    canonicalised later, once column labels exist
    ([`Assembly._build_objective`][specsolve.relational.engine.assembly.Assembly._build_objective]).
    """
    joined, out_dims = _paired(a, b, b.frame.rename({'var_label': 'var_label_2', 'coeff': _RHS}), 'inner')
    frame = joined.with_columns((pl.col('coeff') * pl.col(_RHS)).alias('coeff')).select(
        *out_dims, *carried_columns('quad')
    )
    return Piece(out_dims, frame, 'quad', presences=a.presences + b.presences, region=both_regions(a.region, b.region))
