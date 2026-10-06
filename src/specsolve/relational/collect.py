"""The one place that decides how polars materialises a frame.

A polars built without the streaming engine — the browser's, for one — panics
on the request rather than falling back, so the question is put once. A small
model builds in memory even where it has one: the streaming engine starts up
for every query, and a small build is many small queries.

Every plan here orders its joins by hand, so polars' own join reordering is
off: it moves a join between two others that share its keys, and the one it
runs first can be keyed on a part of them, which multiplies the rows it holds.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ['collect_engine', 'collected', 'collected_all', 'sized']

_AS_WRITTEN = pl.QueryOptFlags(join_order=False)

#: A build whose largest declaration has fewer coordinates than this collects in memory.
IN_MEMORY_BELOW = 250_000

_IN_MEMORY: ContextVar[bool] = ContextVar('_IN_MEMORY', default=False)


@cache
def collect_engine() -> Literal['streaming', 'in-memory']:
    """The engine every collect names: streaming where this polars has it, in-memory otherwise."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return 'in-memory'
    return 'streaming'


@contextmanager
def sized(coordinates: int) -> Iterator[None]:
    """Collect on the in-memory engine inside, where *coordinates* is below [`IN_MEMORY_BELOW`][]."""
    token = _IN_MEMORY.set(coordinates < IN_MEMORY_BELOW)
    try:
        yield
    finally:
        _IN_MEMORY.reset(token)


def _engine() -> Literal['streaming', 'in-memory']:
    return 'in-memory' if _IN_MEMORY.get() else collect_engine()


def collected(frame: pl.LazyFrame) -> pl.DataFrame:
    """*frame* materialised on the engine [`sized`][] chose, its joins in the order they were written."""
    return frame.collect(engine=_engine(), optimizations=_AS_WRITTEN)


def collected_all(frames: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Every one of *frames*, as [`collected`][] would, in one pass that shares their common work."""
    return pl.collect_all(frames, engine=_engine(), optimizations=_AS_WRITTEN)
