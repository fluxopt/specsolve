"""The terms an expression turns into — linear, quadratic or constant — and the arithmetic over them.

Column conventions:

===================  ==========================================
table                columns
===================  ==========================================
linear term          ``dims…``, ``COL``, ``COEFF``
quadratic term       ``dims…``, ``COL``, ``COL_2``, ``COEFF``
constant term        ``dims…``, ``CONSTANT``
===================  ==========================================
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, NoReturn, assert_never

import polars as pl
from mathspec import program

from specsolve._engine.scope import join_on
from specsolve._names import COEFF, COL, COL_2, CONSTANT
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from polars._typing import JoinStrategy

    from specsolve._engine.scope import Scope


#: The right-hand operand's value while a join holds both. The spaces make it
#: unrepresentable as a declared name.
_RHS = '__rhs value__'

#: The one column of a scalar presence table: polars holds no table with rows and no columns.
PRESENT = '__present__'


@dataclass(frozen=True)
class Presence:
    """Where the *variable* under a term exists, and the columns it is joined on.

    Not which rows the term's table has: a neutral parameter's missing rows
    are a zero coefficient, those a variable's condition leaves out are
    absence, and the table cannot tell them apart.

    ``joined_on`` is ``None`` where the table is joined on the term's own
    dimensions, implied columns that stay right as downstream operators rewrite
    them. A shift or window edge states narrower ones.
    """

    table: pl.LazyFrame
    joined_on: tuple[str, ...] | None = None

    def join_columns(self, term_dims: tuple[str, ...]) -> tuple[str, ...]:
        """The columns this presence narrows by, for a term over *term_dims*."""
        return self.joined_on if self.joined_on is not None else term_dims

    def narrow(self, table: pl.LazyFrame, on: Sequence[str]) -> pl.LazyFrame:
        """Keep only the rows of *table* this presence admits.

        With no key, every row survives a present scalar and none an absent one.
        """
        if on:
            return table.join(self.table.select(list(on)), on=list(on), how='semi')
        return table.join(self.table.select(PRESENT), how='cross').drop(PRESENT)


#: What a term carries: one variable, a product of two, or none.
Kind = Literal['linear', 'quadratic', 'constant']


@dataclass(frozen=True)
class Term:
    """One additive term of an expression, over its own dimensions."""

    dims: tuple[str, ...]
    table: pl.LazyFrame
    kind: Kind

    presences: tuple[Presence, ...] = ()
    """Where the variables under this term exist; a quadratic term is absent where either is.

    Empty for a constant term and after a reduction, which skips absent slots.
    """

    region: program.Mask | None = None
    """The region of a `Cases` this term covers by construction, or ``None``.

    The coverage check asks its question only there. A product carries the
    conjunction, and a reduction that drops a dimension the region reads drops the
    region ([`region_over`][]).
    """

    parameters: frozenset[str] = frozenset()
    """The parameters standing as constant terms under this term — what the coverage check asks of.

    A linear or quadratic term carries none, since a sparse coefficient is a zero, and a divisor has its own check.
    """

    @property
    def value_column(self) -> str:
        """``COEFF`` where a variable is under it, ``CONSTANT`` otherwise."""
        return value_column(self.kind)

    @property
    def carried(self) -> list[str]:
        """The non-dimension columns a projection has to keep."""
        return carried_columns(self.kind)


def refuse_a_term_without_the_dims(p: Term, dims: list[str], context: str, operator: str) -> NoReturn:
    """Refuse a term an operator cannot act on, in the right class.

    A constant term lacking the dimensions is valid YAML this engine cannot
    build, so it is a `SpecsolveError`; a linear term is repeated along every
    dimension an operator reads before the operator runs ([`spread_term`][]),
    so one here is a malformed program. *operator* is the spelling the user
    wrote.
    """
    if p.kind == 'constant':
        raise SpecsolveError(
            f'in {context}: {operator} acts along {dims}, which a constant term of the expression '
            f'does not carry, and specsolve cannot build that. A constant term becomes its own '
            f'table, so a term with no rows for {dims} has no slots for the operator to act on — '
            f'and under a condition, which slots those are is known only to the rows. Declare the '
            f'parameter over {dims} and supply it there: the model is the same and the number is unchanged.'
        )
    msg = f'in {context}: {operator} along {dims}, which the expression does not span'
    raise AssertionError(msg)


def spread_term(scope: Scope, p: Term, dims: Sequence[str]) -> Term:
    """*p* repeated at every label of each of *dims* it lacks, the same value at each.

    Its presences leave with their join columns spelled out, since the implied ones,
    the term's dimensions, now name columns their tables do not hold.
    """
    missing = [d for d in dims if d not in p.dims]
    if not missing:
        return p
    out = scope.in_dimension_order((*p.dims, *missing))
    table = scope.spread(p.table, missing).select(*out, *p.carried)
    return replace(p, dims=out, table=table, presences=_keyed(p))


#: The column-index columns each kind carries, in the order a projection keeps them.
_INDEX_COLUMNS: dict[Kind, list[str]] = {
    'linear': [COL],
    'quadratic': [COL, COL_2],
    'constant': [],
}


def value_column(kind: Kind) -> str:
    """The value column a term of this kind carries."""
    return CONSTANT if kind == 'constant' else COEFF


def carried_columns(kind: Kind) -> list[str]:
    """The non-dimension columns a projection of this kind of term has to keep."""
    return [*_INDEX_COLUMNS[kind], value_column(kind)]


@dataclass(frozen=True)
class ExpressionTerms:
    """An expression as the terms it adds up, each kind apart."""

    linear: tuple[Term, ...]
    constant: tuple[Term, ...]
    quadratic: tuple[Term, ...] = ()


def constant_scalar(p: Term) -> pl.LazyFrame:
    """The constant term summed per coordinate: ``(dims…, CONSTANT)``."""
    if not p.dims:
        return p.table.select(pl.col(CONSTANT).sum())
    return p.table.group_by(p.dims).agg(pl.col(CONSTANT).sum())


def presences_of(terms: Sequence[Term]) -> list[Presence]:
    """The presence tables a constraint's rows — or a read's — have to be contained in.

    Each leaves with its join columns spelled out, since indexing cannot know the term it came from.
    """
    return [Presence(x.table, x.join_columns(p.dims)) for p in terms for x in p.presences]


#: How a node's output rows relate to its input slots.
FanIn = Literal['one-to-one', 'many-to-one', 'one-to-many']


def acted_along(expression: program.Sum | program.WindowSum) -> tuple[str, ...]:
    """The dimensions a node that is not one-to-one sums along, a join's axis by the dimension it runs over."""
    if isinstance(expression, program.Sum):
        return tuple(axis.dimension for axis in expression.over)
    return (expression.along,)


def fan_in(expression: program.Expression) -> FanIn:
    """How *expression*'s output rows relate to its input slots.

    Both classes other than ``'one-to-one'`` sum several input slots into an
    output row, so [`propagate_absence`][] runs before them.
    """
    if isinstance(expression, program.ExpressionReference):
        return fan_in(expression.body)
    if isinstance(expression, program.Sum):
        return 'many-to-one'
    if isinstance(expression, program.WindowSum):
        return 'one-to-many'
    if isinstance(
        expression,
        (
            program.Constant,
            program.ParameterReference,
            program.VariableReference,
            program.Dual,
            program.Negate,
            program.Add,
            program.Multiply,
            program.Power,
            program.Divide,
            program.Join,
            program.Translate,
            program.Cases,
        ),
    ):
        return 'one-to-one'
    assert_never(expression)


def propagate_absence(terms: ExpressionTerms, scope: Scope, along: Sequence[str]) -> ExpressionTerms:
    """Narrow every term to where the *whole* expression exists.

    Needed before a node that is not one-to-one ([`fan_in`][]): the row-level
    intersection at assembly cannot say which input slots behind a row survived.
    A term lacking a dimension a presence is joined on is first repeated
    along it, never along a dimension in *along*, which the operator refuses itself
    ([`refuse_a_term_without_the_dims`][]). A term is never narrowed
    by its own presence: its rows are inside it by construction.
    """
    absent = [(p, x) for p in (*terms.linear, *terms.quadratic, *terms.constant) for x in p.presences]
    if not absent:
        return terms

    def narrow(p: Term) -> Term:
        table, dims = p.table, p.dims
        for source, presence in absent:
            if source is p:
                continue
            on = list(presence.join_columns(source.dims))
            missing = [d for d in on if d not in dims]
            if any(d in along for d in missing):
                continue
            if missing:
                dims = scope.in_dimension_order((*dims, *missing))
                table = scope.spread(table, missing).select(*dims, *p.carried)
            table = presence.narrow(table, on)
        return p if table is p.table else replace(p, dims=dims, table=table)

    return map_terms(terms, narrow)


def map_terms(
    terms: ExpressionTerms,
    rewrite: Callable[[Term], Term],
) -> ExpressionTerms:
    """Apply *rewrite* to every term, keeping the kinds apart.

    A rewrite sees one term at a time; a node needing them together is global and refused at load.
    """
    return ExpressionTerms(
        tuple(rewrite(p) for p in terms.linear),
        tuple(rewrite(p) for p in terms.constant),
        tuple(rewrite(p) for p in terms.quadratic),
    )


def both_regions(a: program.Mask | None, b: program.Mask | None) -> program.Mask | None:
    """The region a product of two terms stands over — where both of them do."""
    if a is None or b is None:
        return a or b
    return a if a == b else a & b


def region_over(region: program.Mask | None, dims: Sequence[str]) -> program.Mask | None:
    """*region* where a term over *dims* can still be cut to it, else ``None``."""
    return region if region is not None and region.dims <= set(dims) else None


def negate(p: Term) -> Term:
    return replace(p, table=p.table.with_columns(-pl.col(p.value_column)))


def _paired(a: Term, b: Term, right: pl.LazyFrame, how: JoinStrategy) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """*a*'s table joined to *right*, *b*'s table with its value renamed, and the dimensions out, *a*'s first.

    The join is on the dimensions the two share, so *b* broadcasts over the rest.
    """
    joined = join_on(a.table, right, [d for d in a.dims if d in b.dims], how)
    return joined, a.dims + tuple(d for d in b.dims if d not in a.dims)


def _keyed(p: Term) -> tuple[Presence, ...]:
    """*p*'s presences with their join columns spelled out, before a product gives them a term over more dimensions.

    The implied join columns are the term's own dimensions, and the product's are a superset,
    so carried over unspelled it would name columns the table does not hold.
    """
    return tuple(Presence(x.table, x.join_columns(p.dims)) for x in p.presences)


def times_constant(a: Term, c: Term, kind: Kind, divide: bool = False) -> Term:
    """``a * c`` (or ``a / c``) where *c* is a constant term, broadcast over the dimensions not shared.

    The right-hand value is renamed first: a suffix collision on ``CONSTANT`` would
    multiply a column by itself. A divide joins left, so a coordinate the
    divisor has no value for yields a null coefficient rather than dropping the
    term; a coordinate where the divisor is absent leaves the table, the
    quotient being absent too. The presences of both sides travel out: at a
    read *c* may be a variable at its primal.
    """
    joined, out_dims = _paired(a, c, c.table.rename({CONSTANT: _RHS}), 'left' if divide else 'inner')
    if divide:
        for presence in c.presences:
            joined = presence.narrow(joined, presence.join_columns(c.dims))

    value, rhs = pl.col(a.value_column), pl.col(_RHS)
    combined = value / rhs if divide else value * rhs
    out = value_column(kind)
    table = joined.with_columns(combined.alias(out)).select(*out_dims, *carried_columns(kind))
    if kind != 'constant':
        parameters = frozenset[str]()
    elif divide:
        parameters = a.parameters
    else:
        parameters = a.parameters | c.parameters
    return replace(
        a,
        dims=out_dims,
        table=table,
        kind=kind,
        presences=_keyed(a) + _keyed(c),
        region=both_regions(a.region, c.region),
        parameters=parameters,
    )


def to_the_power(a: Term, b: Term) -> Term:
    """``a ** b``, both constant terms — one constant term out.

    An inner join, unlike divide's left: a null base or exponent would poison
    the coefficient it multiplies rather than report anything.
    """
    joined, out_dims = _paired(a, b, b.table.rename({CONSTANT: _RHS}), 'inner')
    table = joined.with_columns(pl.col(CONSTANT).pow(pl.col(_RHS)).alias(CONSTANT)).select(
        *out_dims, *carried_columns('constant')
    )
    return Term(
        out_dims,
        table,
        'constant',
        presences=_keyed(a) + _keyed(b),
        region=both_regions(a.region, b.region),
        parameters=a.parameters | b.parameters,
    )


def times_variable(a: Term, b: Term) -> Term:
    """``a * b`` where both carry a variable — one quadratic term.

    The second column index is renamed on the way in: a suffix collision would pair a
    variable with itself, which ``p * p`` makes a legal term. Pairs are
    canonicalised later, once column indices exist
    ([`Assembly._build_objective`][specsolve._engine.assembly.Assembly._build_objective]).
    """
    joined, out_dims = _paired(a, b, b.table.rename({COL: COL_2, COEFF: _RHS}), 'inner')
    table = joined.with_columns((pl.col(COEFF) * pl.col(_RHS)).alias(COEFF)).select(
        *out_dims, *carried_columns('quadratic')
    )
    return Term(out_dims, table, 'quadratic', presences=_keyed(a) + _keyed(b), region=both_regions(a.region, b.region))
