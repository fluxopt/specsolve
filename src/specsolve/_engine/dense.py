"""A term held by position: a value at every coordinate of its dims, read where asked instead of joined.

The join path keys a term by its label columns, so every alignment hashes
them. Where the data has a row at every coordinate, a coordinate's row-major
position ([`Scope.row_major`][specsolve._engine.scope.Scope.row_major])
is the key, and an alignment is an index. A [`Dense`][] term is that reading,
and it always carries the frame the join path would read, so an operator
without a positional arm takes the frame and nothing is written twice to be
right.

Only a leaf decides: [`parameter`][], [`variable`][] and [`constant`][]
answer ``None`` where the data is short of a coordinate, and a term over one
such leaf is a frame from there on.
"""

from __future__ import annotations

import math
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

from specsolve._collect import collected
from specsolve._names import VALUE
from specsolve.errors import SpecsolveError

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from specsolve._engine.scope import Scope
    from specsolve._engine.terms import Presence

    type Integers = np.ndarray[tuple[int, ...], np.dtype[np.int64]]
    type Floats = np.ndarray[tuple[int, ...], np.dtype[np.float64]]
    #: Solver column indices, in the dtype the hand-off carries them.
    type Entries = np.ndarray[tuple[int, ...], np.dtype[np.int32]]

    #: A term's columns at coordinates, ``-1`` where its variable has none.
    type Columns = Callable[[At], Integers | Entries]
    #: A term's values at coordinates: one each, or one number that stands at all of them.
    type Values = Callable[[At], Floats | float]

#: The most coordinates a masked variable's dims may span per column it has,
#: for its columns to be held by position: an integer per coordinate.
CELLS_PER_COLUMN = 16


class At:
    """Coordinates of the product of *dims*, each as its row-major position there.

    A term over some of the dims reads them through [`over`][At.over], which
    is worked out once per set of dims however many terms ask.
    """

    def __init__(self, dims: tuple[str, ...], positions: Integers, cardinality: Mapping[str, int]) -> None:
        self.dims = dims
        self._cardinality = cardinality
        self._over: dict[tuple[str, ...], Integers] = {dims: positions}
        self._ordinals: dict[str, Integers] = {}

    def __len__(self) -> int:
        return len(self._over[self.dims])

    def over(self, dims: tuple[str, ...]) -> Integers:
        """Each coordinate's position in the product of *dims*, which are among this one's, in the order given."""
        if dims not in self._over:
            position = np.zeros(len(self), dtype=np.int64)
            for d in dims:
                position = position * self._cardinality[d] + self.ordinal(d)
            self._over[dims] = position
        return self._over[dims]

    def only(self, kept: np.ndarray[tuple[int, ...], np.dtype[np.bool_]]) -> At:
        """The coordinates *kept* marks, in the order they are in."""
        return At(self.dims, self._over[self.dims][kept], self._cardinality)

    def ordinal(self, dim: str) -> Integers:
        """Each coordinate's ordinal along *dim*, one of this one's dims."""
        if dim not in self._ordinals:
            after = self.dims[self.dims.index(dim) + 1 :]
            stride = math.prod(self._cardinality[d] for d in after)
            self._ordinals[dim] = (self._over[self.dims] // stride) % self._cardinality[dim]
        return self._ordinals[dim]


class Dense:
    """A term with a value at every coordinate of its dims.

    ``values`` reads it at the coordinates of an [`At`][] over dims that
    include its own, and ``columns`` the column under it there, ``None`` for
    a constant term. ``reach`` is the last column its variable can hold.
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
    return lambda at: value


def covers(scope: Scope, name: str) -> bool:
    """Whether parameter *name* has a row at every coordinate of its dims.

    Duplicates and strangers are refused at the door, so a table as tall as
    the product of its dims has one row at each.
    """
    dims = scope.program.parameters[name].dims
    return scope.data.parameter_rows.get(name) == math.prod(scope.data.cardinality[d] for d in dims)


def values(scope: Scope, name: str) -> Floats:
    """Parameter *name*'s values in the row-major order of its dims, read once per build; it [`covers`][] them."""
    return _held(scope, ('parameter', name), _values_by_position)


def parameter(scope: Scope, name: str) -> Values | None:
    """Parameter *name* by position, or ``None`` where it is short of a row."""
    if not covers(scope, name):
        return None
    dims = scope.program.parameters[name].dims
    return lambda at: values(scope, name)[at.over(dims)]


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
    return lambda at: _held(scope, ('variable', name), _columns_by_position)[at.over(dims)]


def unit(at: At) -> float:
    """The coefficient a variable stands at before anything multiplies it."""
    return 1.0


def _held(scope: Scope, key: tuple[str, str], read: Callable[[Scope, str], np.ndarray]) -> np.ndarray:
    """The array *read* gives for *key*'s name, read once per build and kept on the scope."""
    if key not in scope.by_position:
        scope.by_position[key] = read(scope, key[1])
    return scope.by_position[key]


def negated(body: Dense, tidy: Callable[[], pl.LazyFrame]) -> Dense:
    """``-body``, *tidy* being the same on the join path."""
    return Dense(lambda at: -body.values(at), tidy, body.columns, body.reach, body.absent)


def product(a: Dense, c: Dense, tidy: Callable[[], pl.LazyFrame]) -> Dense:
    """``a * c`` where *c* is a constant term, *tidy* being the same on the join path.

    Each factor reads the coordinate along its own dims, which is the
    broadcast the join makes by matching the dims the two share.
    """
    return Dense(lambda at: a.values(at) * c.values(at), tidy, a.columns, a.reach, a.absent)


def _values_by_position(scope: Scope, name: str) -> Floats:
    """Parameter *name*'s values in the row-major order of its dims, the table having a row at every coordinate.

    A table whose rows already stand in that order is read as it is, with no
    array beside it; any other order is moved into one.

    Raises:
        SpecsolveError: The table is as tall as the product and still misses
            a coordinate, which the door's refusal of a repeated row rules out.
    """
    dims = scope.program.parameters[name].dims
    table = scope.data.parameters[name]
    position, value = scope.row_major(dims, scope.ordinal_of), pl.col(VALUE).cast(pl.Float64)
    cells = math.prod(scope.data.cardinality[d] for d in dims)
    in_order = (position == pl.int_range(pl.len(), dtype=pl.Int64)).all() & (pl.len() == cells)
    if table.select(in_order).pipe(collected).item():
        return table.select(value).pipe(collected).get_column(VALUE).to_numpy()
    held = table.select(position.alias('#position'), value).pipe(collected)
    at = held.get_column('#position').to_numpy()
    written = np.zeros(cells, dtype=bool)
    written[at] = True
    if at.size != cells or not written.all():
        msg = f"parameter '{name}' is as tall as the product of its dims and does not cover it"
        raise SpecsolveError(msg)
    out = np.empty(cells, dtype=np.float64)
    out[at] = held.get_column(VALUE).to_numpy()
    return out


def _columns_by_position(scope: Scope, name: str) -> Entries:
    """Variable *name*'s column at each coordinate of its dims in row-major order, ``-1`` where it has none."""
    dims = scope.program.variables[name].dims
    position = scope.row_major(dims, scope.ordinal_of)
    held = scope.variables[name].frame.select(position.alias('#position'), 'var_label').pipe(collected)
    out = np.full(math.prod(scope.data.cardinality[d] for d in dims), -1, dtype=np.int32)
    out[held.get_column('#position').to_numpy()] = held.get_column('var_label').to_numpy()
    return out


def placed(
    terms: Sequence[tuple[Dense, float]], at: At
) -> tuple[At, np.ndarray[tuple[int, ...], np.dtype[np.bool_]] | None, Integers, Entries, Floats]:
    """The matrix entries of *terms*, each a body and its sign, at the coordinates *at*.

    A coordinate where a term is ``absent`` keeps no row, and which those are
    is read first, so every term is read only where a row stands, and one
    term at a time, so no more than one term's arrays stand beside the
    entries.

    Returns:
        The coordinates that keep a row, and which of *at* they are, ``None``
        for all of them. Then the entries as ``(index, col, coeff)`` in
        *index* order, *index* counting the coordinates kept. A term whose
        variable has no column at a coordinate holds no entry there. Within
        a coordinate the terms are laid out by ``reach``, which is column
        order for terms over different variables, so their share is not
        sorted.
    """
    kept = None
    for body, _ in terms:
        if body.absent is None:
            continue
        column = _columns_of(body, at)
        if column.size and column.min() < 0:
            there = column >= 0
            kept = there if kept is None else kept & there
    if kept is not None:
        at = at.only(kept)

    count = len(terms)
    cols = np.empty(len(at) * count, dtype=np.int32)
    coeffs = np.empty(len(at) * count, dtype=np.float64)
    holed = False
    for slot, (body, sign) in enumerate(sorted(terms, key=lambda term: term[0].reach)):
        column = _columns_of(body, at)
        cols[slot::count] = column
        holed = holed or bool(column.size and column.min() < 0)
        np.multiply(body.values(at), sign, out=coeffs[slot::count])
    index = np.repeat(np.arange(len(at), dtype=np.int64), count)
    if holed:
        held = cols >= 0
        return at, kept, index[held], cols[held], coeffs[held]
    return at, kept, index, cols, coeffs


def _columns_of(body: Dense, at: At) -> Integers | Entries:
    assert body.columns is not None, 'a term placed on a row carries a variable'
    return body.columns(at)
