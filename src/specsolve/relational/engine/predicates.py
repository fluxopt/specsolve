"""A ``where:`` mask as a query: which rows of a coordinate product survive.

The plan's predicate nodes in; a boolean expression out, with the frame the
walk joined the mask's parameters onto. [`masked`][] is the product a
declaration is instantiated over, cut by its mask.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, assert_never

import polars as pl
from mathspec import program

from specsolve.errors import DataError
from specsolve.messages import position_out_of_range_message, short_groups_message
from specsolve.relational.collect import collected
from specsolve.relational.engine.relations import GROUP_RANK, GROUP_SIZE, Grouping, mapping, walk_join
from specsolve.relational.engine.scope import join_on
from specsolve.relational.engine.shifts import translate_rows

if TYPE_CHECKING:
    import datetime
    from collections.abc import Callable

    from polars._typing import JoinStrategy

    from specsolve.relational.engine.scope import Scope


class Carrier:
    """A frame a walk joins onto, each attachment made at most once."""

    def __init__(self, frame: pl.LazyFrame) -> None:
        self.frame = frame
        self._attached: set[str] = set()

    def once(self, alias: str, attach: Callable[[pl.LazyFrame, str], pl.LazyFrame]) -> str:
        """Join *attach* onto the frame under *alias*, unless it already is, and return *alias*."""
        if alias not in self._attached:
            self.frame = attach(self.frame, alias)
            self._attached.add(alias)
        return alias


def _defined(col: pl.Expr, dtype: program.ParameterDtype) -> pl.Expr:
    """What a bare parameter name in a ``where`` asks of *col*, by its declared dtype.

    A ``bool`` is its own answer, a ``str`` is defined wherever it has a row,
    and a number has to be finite as well.
    """
    if dtype == 'bool':
        return col.is_not_null() & col.cast(pl.Boolean)
    if dtype == 'str':
        return col.is_not_null()
    return col.is_not_null() & col.is_finite()


def masked(scope: Scope, dims: tuple[str, ...], where: program.Mask | None) -> pl.LazyFrame:
    """The masked coordinate product over *dims*, with the ordinals a caller sorts by.

    A joining mask that reads only some of *dims*, and none outside them, is
    evaluated over their product and semi-joined; every other shape filters the
    full product, so an error names all of its dims. Both keep row order.
    """
    out = scope.product(dims)
    if where is None:
        return out
    on = tuple(d for d in dims if d in where.dims)
    if on and len(on) < len(dims) and where.dims <= set(dims):
        keyed = scope.product(on)
        carrier, condition = compile_predicate(scope, keyed, where, on)
        if carrier is keyed:
            return out.filter(falsy_if_null(condition))
        surviving = carrier.filter(falsy_if_null(condition)).select(*on)
        return out.join(surviving, on=list(on), how='semi')
    carrier, condition = compile_predicate(scope, out, where, dims)
    return carrier.filter(falsy_if_null(condition))


def compile_predicate(
    scope: Scope, frame: pl.LazyFrame, mask: program.Mask, dims: tuple[str, ...]
) -> tuple[pl.LazyFrame, pl.Expr]:
    """``(frame with the mask's parameters joined, boolean expression)``.

    Walking joins the parameters, so the condition is built before the frame is
    read. A name the mask is certain of ([`_certain_names`][]) is inner- or
    semi-joined rather than left-joined. No join maintains order: consumers
    verify it ([`labels.in_position_order`][]).
    """
    certain = _certain_names(mask)
    carrier = Carrier(frame)
    serial = itertools.count()

    def join_param(param: str) -> str:
        how: JoinStrategy = 'inner' if param in certain else 'left'
        return carrier.once(
            f'__where {param}__',
            lambda f, alias: scope.parameter_join(f, param, dims, alias, f"where-parameter '{param}'", how),
        )

    def refuse_outside_frame(reading: str, dimension: str) -> None:
        """A mask reading a dim the frame does not span — refused at load, asserted here."""
        assert dimension in dims, f'where-comparison on {reading} is outside the frame dims {list(dims)}'

    def join_ordinal(dimension: str) -> str:
        refuse_outside_frame(f"dimension '{dimension}'", dimension)
        return carrier.once(
            f'__where ord {dimension}__',
            lambda f, alias: f.join(
                scope.data.dimensions[dimension].select(pl.col('val').alias(dimension), pl.col('ord').alias(alias)),
                on=dimension,
                how='left',
            ),
        )

    def join_group_offset(p: program.DimensionPosition) -> str:
        """One column: the row's ordinal minus its own group's target ordinal."""
        assert p.partition is not None, 'an ungrouped position counts along the whole dimension and asks for no table'
        grouping = Grouping.of(scope.data, p.partition)
        for dim in grouping.keys:
            refuse_outside_frame(f"dimension '{dim}'", dim)
        _refuse_short_groups(p, grouping)
        target = pl.lit(p.position) if p.position >= 0 else pl.col(GROUP_SIZE) + p.position
        offset = pl.col(GROUP_RANK) - target
        return carrier.once(
            f'__where ord {p.name} by {p.partition.name}__',
            lambda f, alias: f.join(
                grouping.table.select(pl.col('val').alias(p.name), *grouping.joined, offset.alias(alias)),
                on=list(grouping.keys),
                how='left',
            ),
        )

    def join_relation(relation: str, dims: tuple[str, ...], column: str | None) -> str:
        """*relation* read at *dims* — one value column, or with ``None`` whether a row is there at all."""
        for dim in dims:
            refuse_outside_frame(f"relation '{relation}' reading dimension '{dim}'", dim)
        shape = scope.program.relations[relation]
        roles = shape.key or shape.roles
        assert tuple(shape.dim(role) for role in roles) == dims, f"relation '{relation}' is read at its key"
        read = pl.col(column) if column is not None else pl.lit(value=True)
        return carrier.once(
            f'__where relation {relation}.{column or ""}__',
            lambda f, alias: f.join(
                scope.data.relations[relation].select(
                    *(pl.col(role).alias(dim) for role, dim in zip(roles, dims, strict=True)), read.alias(alias)
                ),
                on=list(dims),
                how='left',
            ),
        )

    def join_reduction(label: str, build: Callable[[str], pl.LazyFrame], on: tuple[str, ...]) -> str:
        """A frame a leaf reduces its own product to, joined onto the walk's by *on*.

        The alias is numbered: two leaves of one mask can reduce the same frame to different answers.
        """
        for dimension in on:
            refuse_outside_frame(f'where {label}', dimension)
        alias = f'__where {label} {next(serial)}__'
        return carrier.once(alias, lambda f, a: join_on(f, build(a), on, 'left'))

    def walk(p: program.Predicate) -> pl.Expr:
        if isinstance(p, program.ParameterComparison):
            return _compare(pl.col(join_param(p.name)), p.op, p.value)
        if isinstance(p, program.ExpressionComparison):
            left, right = (_values(scope, side) for side in (p.left, p.right))
            columns = tuple(
                join_reduction('value', lambda a, v=values: v.rename({'cval': a}), on) for values, on in (left, right)
            )
            return _COLUMN_COMPARISONS[p.op](pl.col(columns[0]), pl.col(columns[1]))
        if isinstance(p, program.CountComparison):
            keys = scope.in_declaration_order(p.dims)
            alias = join_reduction('count', lambda a: _counted(scope, p, a), keys)
            return _COLUMN_COMPARISONS[p.op](pl.col(alias).fill_null(0), pl.lit(p.value))
        if isinstance(p, program.TranslatedPredicate):
            on = scope.in_declaration_order(p.dims)
            alias = join_reduction(f'shift {p.along}', lambda a: _translated(scope, p, on, a), on)
            return falsy_if_null(pl.col(alias))
        if isinstance(p, program.PulledBackPredicate):
            on = scope.in_declaration_order(p.dims)
            alias = join_reduction(f'at {p.direction.name}', lambda a: _pulled_back(scope, p, a), on)
            return falsy_if_null(pl.col(alias))
        if isinstance(p, program.DimensionComparison):
            refuse_outside_frame(f"dimension '{p.name}'", p.name)
            return _compare(_dimension_column(p.name, p.value), p.op, p.value)
        if isinstance(p, program.DimensionPosition):
            if p.partition is not None:
                return falsy_if_null(_COLUMN_COMPARISONS[p.op](pl.col(join_group_offset(p)), pl.lit(0)))
            at = _position_ordinal(p, scope.data.cardinality[p.name])
            return _COLUMN_COMPARISONS[p.op](pl.col(join_ordinal(p.name)), pl.lit(at))
        if isinstance(p, program.RelationComparison):
            column = pl.col(join_relation(p.name, p.dims, p.column))
            if isinstance(p.value, str):
                column = column.cast(pl.String)
            return _compare(column, p.op, p.value)
        if isinstance(p, program.RelationPairComparison):
            left = pl.col(join_relation(p.name, p.dims, p.column))
            right = pl.col(join_relation(p.other, p.dims, p.other_column))
            return _COLUMN_COMPARISONS[p.op](left, right)
        if isinstance(p, program.RelationDefined):
            return pl.col(join_relation(p.name, p.dims, None)).is_not_null()
        if isinstance(p, program.ParameterDefined):
            return _defined(pl.col(join_param(p.name)), scope.program.parameters[p.name].dtype)
        if isinstance(p, program.VariableDefined):
            on = list(scope.program.variables[p.name].dims)
            coordinates = scope.variables[p.name].frame.select(*on)
            if p.name in certain:
                carrier.once(f'__where defined {p.name}__', lambda f, _: f.join(coordinates, on=on, how='semi'))
                return pl.lit(value=True)
            flag = carrier.once(
                f'__where defined {p.name}__',
                lambda f, alias: f.join(
                    coordinates.unique().with_columns(pl.lit(value=True).alias(alias)), on=on, how='left'
                ),
            )
            return falsy_if_null(pl.col(flag))
        if isinstance(p, program.BooleanLiteral):
            return pl.lit(value=p.value)
        if isinstance(p, program.And):
            return walk(p.left) & walk(p.right)
        if isinstance(p, program.Or):
            return walk(p.left) | walk(p.right)
        if isinstance(p, program.Not):
            return ~falsy_if_null(walk(p.operand))
        assert_never(p)

    condition = walk(mask.root)
    return carrier.frame, condition


def _values(scope: Scope, side: program.Expression) -> tuple[pl.LazyFrame, tuple[str, ...]]:
    """One side of a comparison of expressions as ``(dims…, cval)``, and the dims it is keyed by.

    Read by the expression compiler, so it cannot drift from the rows. Pieces
    are added so that a null spreads, which [`falsy_if_null`][] reads as false.
    """
    # in-function: the compiler imports this module
    from specsolve.relational.engine.compiler import Compiler

    compiler = Compiler(scope)
    compiled = compiler.expression(side, 'a where comparing expressions')
    assert not (compiled.terms or compiled.quads), (
        'a where compares expressions the language keeps every variable out of'
    )
    dims = scope.spanned(compiled.consts)
    added = compiler.summed_onto(compiled.consts, masked(scope, dims, None), absent='spreads')
    return added.select(*dims, 'cval'), dims


def _counted(scope: Scope, p: program.CountComparison, alias: str) -> pl.LazyFrame:
    """How many coordinates along ``over`` the predicate admits, per coordinate of the rest.

    A coordinate no row survives at is missing, which the walk reads as zero.
    """
    dims = scope.in_declaration_order(p.predicate.dims)
    keys = [d for d in dims if d != p.over]
    surviving = masked(scope, dims, p.predicate)
    if not keys:
        return surviving.select(pl.len().alias(alias))
    return surviving.group_by(keys).agg(pl.len().alias(alias))


def _translated(scope: Scope, p: program.TranslatedPredicate, dims: tuple[str, ...], alias: str) -> pl.LazyFrame:
    """Where the operand holds *offset* coordinates back along ``along``, true-only.

    The end the move vacates is a missing row, which reads as false.
    """
    assert p.along in dims, f"a shift along '{p.along}' is read at a frame carrying it"
    admitted = masked(scope, dims, p.operand).select(*dims).with_columns(pl.lit(value=True).alias(alias))
    return translate_rows(scope, admitted, dims, [alias], p.along, p.offset)


def _pulled_back(scope: Scope, p: program.PulledBackPredicate, alias: str) -> pl.LazyFrame:
    """Where the operand holds at the coarse coordinate the relation maps each fine one to, true-only.

    A fine coordinate the relation has no row for is a missing row, which reads as false.
    """
    dims = scope.in_declaration_order(p.operand.dims)
    admitted = masked(scope, dims, p.operand).select(*dims).with_columns(pl.lit(value=True).alias(alias))
    read, _ = walk_join(admitted, mapping(scope.data.relations, p.direction), p, dims, [alias])
    return read


def _certain_names(mask: program.Mask) -> frozenset[str]:
    """Parameter and variable names whose absence alone makes the whole mask false.

    Only the ``AND`` spine counts: under ``OR`` or ``NOT`` an absent value can still leave the mask true.
    """
    atoms = (program.ParameterComparison, program.ParameterDefined, program.VariableDefined)
    return frozenset(a.name for a in mask.conjuncts if isinstance(a, atoms))


def _refuse_short_groups(p: program.DimensionPosition, grouping: Grouping) -> None:
    """Refuse a position no coordinate of some group occupies, as [`_position_ordinal`][] does ungrouped.

    A coordinate in no group is not in the grouping's table, so it is never counted short.
    """
    assert p.partition is not None
    needed = p.position + 1 if p.position >= 0 else -p.position
    sizes = grouping.table.select(*grouping.key, GROUP_SIZE).unique().pipe(collected)
    named = (row[0] if len(grouping.key) == 1 else row[:-1] for row in sizes.iter_rows() if row[-1] < needed)
    if short := sorted(str(group) for group in named):
        raise DataError(short_groups_message(p.name, p.partition.name, p.op, p.position, short))


def falsy_if_null(condition: pl.Expr) -> pl.Expr:
    """*condition* with null read as false: a missing row excludes the coordinate."""
    return condition.fill_null(value=False)


def _position_ordinal(p: program.DimensionPosition, cardinality: int) -> int:
    """*p*'s position as an ordinal into a dimension of *cardinality* labels, negative from the end.

    Out of range is an error: a boundary clause that seeds no row leaves the recurrence unanchored.
    """
    at = p.position + cardinality if p.position < 0 else p.position
    if not 0 <= at < cardinality:
        raise DataError(position_out_of_range_message(p.name, p.op, p.position, at, cardinality))
    return at


def _dimension_column(dimension: str, value: float | str | datetime.date) -> pl.Expr:
    """The column a where-comparison on *dimension* reads.

    A string label is compared as ``String``, not ``Enum``: the where-string
    rules order labels bytewise and read an unknown label as matching nothing.
    """
    column = pl.col(dimension)
    return column.cast(pl.String) if isinstance(value, str) else column


#: The comparison operators, evaluated column against column.
_COLUMN_COMPARISONS: dict[program.PredicateOperator, Callable[[pl.Expr, pl.Expr], pl.Expr]] = {
    '==': lambda left, right: left == right,
    '!=': lambda left, right: left != right,
    '<': lambda left, right: left < right,
    '<=': lambda left, right: left <= right,
    '>': lambda left, right: left > right,
    '>=': lambda left, right: left >= right,
}


def _compare(column: pl.Expr, op: program.PredicateOperator, value: float | str | datetime.date) -> pl.Expr:
    """One where-comparison against a literal."""
    return _COLUMN_COMPARISONS[op](column, pl.lit(value))
