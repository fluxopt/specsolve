"""Data read by position instead of joined, where it has a value at every coordinate of its dims.

The join path keys a term by its label columns, so every alignment hashes
them. Where the data has a row at every coordinate, a coordinate's row-major
position ([`Scope.row_major`][specsolve._engine.scope.Scope.row_major])
is the key, and an alignment is an index. A [`Positional`][] term is that reading,
and it always carries the frame the join path would read, so an operator
without a positional arm takes the frame and nothing is written twice to be
right.

A read by position is a polars expression over a frame of positions, so it
runs in the query that places it, on the engine and the memory every other
frame of the build uses.

Only a leaf decides: [`parameter`][], [`variable`][] and [`constant`][]
answer ``None`` where the data is short of a coordinate, and a term over one
such leaf is a frame from there on.
"""

from __future__ import annotations

import math
from functools import cached_property, reduce
from typing import TYPE_CHECKING

import polars as pl

from specsolve._collect import collected
from specsolve._names import VALUE
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve._engine.scope import Scope
    from specsolve._engine.terms import Presence

    #: A term's columns at coordinates, ``-1`` where its variable has none.
    type Columns = Callable[[Coordinates], pl.Expr]
    #: A term's values at coordinates.
    type Values = Callable[[Coordinates], pl.Expr]

#: The column a frame of coordinates carries: each one's row-major position in the product of its dims.
POSITION = '#position'

#: The fewest coordinates a parameter spans for a ``where`` or a bound to read
#: it by position. The read is a query of its own, where a join rides in one
#: that runs anyway, and under this many rows it costs more than the join.
READ_BY_POSITION_FROM = 100_000

#: The most coordinates a masked variable's dims may span per column it has,
#: for its columns to be held by position: an integer per coordinate.
CELLS_PER_COLUMN = 16


class Coordinates:
    """A frame of coordinates of the product of *dims*, each given as its [`POSITION`][] there.

    It holds no row: it is how such a frame is read. A term over some of the
    dims reads them through [`over`][Coordinates.over].
    """

    def __init__(self, dims: tuple[str, ...], cardinality: Mapping[str, int]) -> None:
        self.dims = dims
        self._cardinality = cardinality

    def over(self, dims: tuple[str, ...]) -> pl.Expr:
        """Each coordinate's position in the product of *dims*, which are among this one's, in the order given."""
        if dims == self.dims:
            return pl.col(POSITION)
        position: pl.Expr = pl.lit(0, dtype=pl.Int64)
        for d in dims:
            position = position * self._cardinality[d] + self.ordinal(d)
        return position

    def ordinal(self, dim: str) -> pl.Expr:
        """Each coordinate's ordinal along *dim*, one of this one's dims."""
        after = self.dims[self.dims.index(dim) + 1 :]
        stride = math.prod(self._cardinality[d] for d in after)
        return (pl.col(POSITION) // stride) % self._cardinality[dim]


class Positional:
    """A term held by position: a value at every coordinate of its dims.

    ``values`` is the expression that reads it on a frame of
    [`Coordinates`][] over dims that include its own, and ``columns`` the one
    that reads the column under it, ``None`` for a constant term. ``reach``
    is the last column its variable can hold.
    ``frame`` is the same term as the join path reads it, formed the first
    time something asks. ``absent`` is the presence a coordinate without a
    column stands for, where the term's variable is masked and removes the
    row it is missing from; ``None`` where a missing column is a zero.
    """

    def __init__(
        self,
        values: Values,
        tidy: Callable[[], pl.LazyFrame],
        columns: Columns | None = None,
        reach: int = -1,
        absent: Presence | None = None,
    ) -> None:
        self.values = values
        self.columns = columns
        self.reach = reach
        self.absent = absent
        self._tidy = tidy

    @cached_property
    def frame(self) -> pl.LazyFrame:
        """The term as ``(dims…, label columns, value)``, one row per coordinate it holds."""
        return self._tidy()

    def stands_for(self, presences: Sequence[Presence]) -> bool:
        """Whether *presences*, a term's, are exactly what this body's missing columns say.

        An operator that gives a term another presence leaves the body
        saying less than the term does, and the term is then read as a frame.
        """
        own = () if self.absent is None else (self.absent,)
        return len(presences) == len(own) and all(theirs is mine for theirs, mine in zip(presences, own, strict=True))


def constant(value: float) -> Values | None:
    """A literal number at every coordinate."""
    return lambda at: pl.lit(value, dtype=pl.Float64)


def readable(scope: Scope, name: str) -> bool:
    """Whether parameter *name* can be read by position: it [covers][specsolve._engine.attaching.AttachedSources.covers] its dims."""
    return scope.data.covers(name, scope.program.parameters[name].dims)


def worth_reading(scope: Scope, name: str) -> bool:
    """Whether parameter *name* is [`readable`][] and spans [`READ_BY_POSITION_FROM`][] coordinates or more."""
    dims = scope.program.parameters[name].dims
    return readable(scope, name) and math.prod(scope.data.cardinality[d] for d in dims) >= READ_BY_POSITION_FROM


def column(scope: Scope, name: str) -> pl.Series:
    """Parameter *name*'s value column in the row-major order of its dims, as it was attached; it is [`readable`][]."""
    return _read_once(scope, ('column', name), _column_by_position)


def values(scope: Scope, name: str) -> pl.Series:
    """[`column`][] as the numbers a term reads, held once per build."""
    return _read_once(scope, ('values', name), lambda scope, name: column(scope, name).cast(pl.Float64))


def parameter(scope: Scope, name: str) -> Values | None:
    """Parameter *name* by position, or ``None`` where it is short of a row."""
    if not readable(scope, name):
        return None
    dims = scope.program.parameters[name].dims
    return lambda at: _read(values(scope, name), at, dims)


def variable(scope: Scope, name: str) -> Columns | None:
    """Variable *name*'s columns by position, or ``None`` where position is not worth holding.

    An unmasked variable's column is its start plus its position, so nothing
    is held. A masked one holds an integer per coordinate of its dims, which
    is refused past [`CELLS_PER_COLUMN`][] per column it has.
    """
    declaration, held = scope.program.variables[name], scope.variables[name]
    dims = declaration.dims
    if declaration.where is None:
        return lambda at: held.start + at.over(dims)
    cells = math.prod(scope.data.cardinality[d] for d in dims)
    if not held.height or cells > CELLS_PER_COLUMN * held.height:
        return None
    return lambda at: _read(_read_once(scope, ('variable', name), _columns_by_position), at, dims)


def _read(held: pl.Series, at: Coordinates, dims: tuple[str, ...]) -> pl.Expr:
    """*held*, one value per coordinate of *dims* in row-major order, at each coordinate of the frame.

    Over no dims it is its one value, said as a scalar: a gather of one row
    is a column of one row, which no frame taller than that takes.
    """
    return pl.lit(held).gather(at.over(dims)) if dims else pl.lit(held).first()


def unit(at: Coordinates) -> pl.Expr:
    """The coefficient a variable stands at before anything multiplies it."""
    return pl.lit(1.0, dtype=pl.Float64)


def _read_once(scope: Scope, key: tuple[str, str], read: Callable[[Scope, str], pl.Series]) -> pl.Series:
    """What *read* gives for *key*'s name, read once per build and kept on the scope."""
    if key not in scope.by_position:
        scope.by_position[key] = read(scope, key[1])
    return scope.by_position[key]


def negated(body: Positional, tidy: Callable[[], pl.LazyFrame]) -> Positional:
    """``-body``, *tidy* being the same on the join path."""
    return Positional(lambda at: -body.values(at), tidy, body.columns, body.reach, body.absent)


def product(a: Positional, c: Positional, tidy: Callable[[], pl.LazyFrame]) -> Positional:
    """``a * c`` where *c* is a constant term, *tidy* being the same on the join path.

    Each factor reads the coordinate along its own dims, which is the
    broadcast the join makes by matching the dims the two share.
    """
    return Positional(lambda at: a.values(at) * c.values(at), tidy, a.columns, a.reach, a.absent)


def _column_by_position(scope: Scope, name: str) -> pl.Series:
    """Parameter *name*'s value column in the row-major order of its dims, the table having a row at every coordinate.

    A table whose rows already stand in that order is read as it is, with no
    copy beside it; any other order is moved into one in a single pass.

    Raises:
        SpecsolveError: The table is as tall as the product and still misses
            a coordinate, which the door's refusal of a repeated row rules out.
    """
    dims = scope.program.parameters[name].dims
    cells = math.prod(scope.data.cardinality[d] for d in dims)
    position = scope.row_major(dims, scope.ordinal_of).alias(POSITION)
    held = scope.data.parameters[name].select(position, VALUE).pipe(collected)
    at, column = held.get_column(POSITION), held.get_column(VALUE)
    if at.len() == cells and (at == pl.int_range(cells, dtype=pl.Int64, eager=True)).all():
        return column
    row = pl.repeat(-1, cells, dtype=pl.Int64, eager=True).scatter(
        at, pl.int_range(at.len(), dtype=pl.Int64, eager=True)
    )
    if at.len() != cells or (row < 0).any():
        msg = f"parameter '{name}' is as tall as the product of its dims and does not cover it"
        raise SpecsolveError(msg)
    return column.gather(row)


def _columns_by_position(scope: Scope, name: str) -> pl.Series:
    """Variable *name*'s column at each coordinate of its dims in row-major order, ``-1`` where it has none."""
    dims = scope.program.variables[name].dims
    position = scope.row_major(dims, scope.ordinal_of)
    held = (
        scope.variables[name].frame.select(position.alias(POSITION), pl.col('var_label').cast(pl.Int32)).pipe(collected)
    )
    cells = math.prod(scope.data.cardinality[d] for d in dims)
    return pl.repeat(-1, cells, dtype=pl.Int32, eager=True).scatter(
        held.get_column(POSITION), held.get_column('var_label')
    )


def placed(
    terms: Sequence[tuple[Positional, float]], positions: pl.Series, at: Coordinates
) -> tuple[pl.Series, pl.Series | None, pl.DataFrame]:
    """The matrix entries of *terms*, each a body and its sign, at the coordinates *positions* gives and *at* describes.

    A coordinate where a term is ``absent`` keeps no row, and which those are
    is read first, so every term is read only where a row stands.

    Returns:
        The positions that keep a row, and which of *positions* they are,
        ``None`` for all of them. Then the entries as ``(index, col, coeff)``
        in *index* order, *index* counting the positions kept. A term whose
        variable has no column at a coordinate holds no entry there. Within
        a coordinate the terms are laid out by ``reach``, which is column
        order for terms over different variables, so their share is not
        sorted: each term's entries are read as one block in coordinate
        order, and the blocks are merged on the coordinate, one that ties
        leaving in the order of the blocks.
    """
    kept = None
    there = [_columns_of(body, at) >= 0 for body, _ in terms if body.absent is not None]
    if there:
        standing = pl.DataFrame({POSITION: positions}).select(pl.all_horizontal(there).alias('#kept')).to_series()
        if not standing.all():
            kept, positions = standing, positions.filter(standing)

    index = pl.int_range(positions.len(), dtype=pl.Int64, eager=True).set_sorted()
    coordinates = pl.LazyFrame({'index': index, POSITION: positions})
    blocks = [
        coordinates.select(
            'index',
            _columns_of(body, at).cast(pl.Int32).alias('col'),
            (body.values(at) * sign).cast(pl.Float64).alias('coeff'),
        )
        for body, sign in sorted(terms, key=lambda term: term[0].reach)
    ]
    entries = reduce(lambda merged, block: merged.merge_sorted(block, key='index'), blocks).pipe(collected)
    if entries.height and entries.get_column('col').min() < 0:  # pyrefly: ignore[unsupported-operation] -- an Int32 column of rows has a minimum
        entries = entries.filter(pl.col('col') >= 0)
    return positions, kept, entries


def summed(terms: Sequence[tuple[Positional, float]], at: Coordinates) -> pl.Expr:
    """The sum of *terms*, each a constant body and its sign, on a frame of the coordinates *at* describes.

    Every term has a value at every coordinate, so the sum is owed nothing: no
    coordinate is a gap for a coverage check to find.
    """
    total: pl.Expr = pl.lit(0.0, dtype=pl.Float64)
    for body, sign in terms:
        total = total + body.values(at) * sign
    return total


def _columns_of(body: Positional, at: Coordinates) -> pl.Expr:
    assert body.columns is not None, 'a term placed on a row carries a variable'
    return body.columns(at)
