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

import math
from contextlib import contextmanager
from contextvars import ContextVar
from functools import cache
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

__all__ = ['collected', 'collected_all', 'collecting_on', 'engine_for', 'streaming_available']

_AS_WRITTEN = pl.QueryOptFlags(join_order=False)

#: A build whose largest declaration has fewer coordinates than this collects in memory.
IN_MEMORY_BELOW = 250_000

_COLLECTING_ON: ContextVar[Literal['streaming', 'in-memory'] | None] = ContextVar('_COLLECTING_ON', default=None)


@cache
def streaming_available() -> bool:
    """Whether this polars has the streaming engine, asked once."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return False
    return True


def engine_for(coordinates: float) -> Literal['streaming', 'in-memory']:
    """The engine a build collects on, given its largest declaration's coordinate count.

    In memory below [`IN_MEMORY_BELOW`][], or where this polars has no
    streaming engine; streaming otherwise.
    """
    return 'streaming' if coordinates >= IN_MEMORY_BELOW and streaming_available() else 'in-memory'


@contextmanager
def collecting_on(engine: Literal['streaming', 'in-memory']) -> Iterator[None]:
    """Every [`collected`][] and [`collected_all`][] inside runs on *engine*."""
    token = _COLLECTING_ON.set(engine)
    try:
        yield
    finally:
        _COLLECTING_ON.reset(token)


def _engine() -> Literal['streaming', 'in-memory']:
    """The engine [`collecting_on`][] set, else the one a frame of unknown size gets."""
    return _COLLECTING_ON.get() or engine_for(math.inf)


def collected(frame: pl.LazyFrame) -> pl.DataFrame:
    """*frame* materialised on the current engine, its joins in the order they were written."""
    return frame.collect(engine=_engine(), optimizations=_AS_WRITTEN)


def collected_all(frames: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Every one of *frames*, as [`collected`][] would, in one pass that shares their common work."""
    return pl.collect_all(frames, engine=_engine(), optimizations=_AS_WRITTEN)
