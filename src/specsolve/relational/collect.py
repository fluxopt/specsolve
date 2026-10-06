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

__all__ = ['CollectEngine', 'collect_engine', 'collected', 'collected_all', 'engine_for', 'on']

type CollectEngine = Literal['streaming', 'in-memory']

_AS_WRITTEN = pl.QueryOptFlags(join_order=False)

#: A build whose largest declaration has fewer coordinates than this collects in memory.
IN_MEMORY_BELOW = 250_000

_ON: ContextVar[CollectEngine | None] = ContextVar('_ON', default=None)


@cache
def collect_engine() -> CollectEngine:
    """The engine every collect names: streaming where this polars has it, in-memory otherwise."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return 'in-memory'
    return 'streaming'


def engine_for(coordinates: int) -> CollectEngine:
    """The engine a build collects on, given its largest declaration's coordinate count.

    In memory below [`IN_MEMORY_BELOW`][], else [`collect_engine`][].
    """
    return 'in-memory' if coordinates < IN_MEMORY_BELOW else collect_engine()


@contextmanager
def on(engine: CollectEngine) -> Iterator[None]:
    """Every [`collected`][] inside runs on *engine*."""
    token = _ON.set(engine)
    try:
        yield
    finally:
        _ON.reset(token)


def _engine() -> CollectEngine:
    return _ON.get() or collect_engine()


def collected(frame: pl.LazyFrame) -> pl.DataFrame:
    """*frame* materialised on the engine [`on`][] names, else [`collect_engine`][], its joins as written."""
    return frame.collect(engine=_engine(), optimizations=_AS_WRITTEN)


def collected_all(frames: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Every one of *frames*, as [`collected`][] would, in one pass that shares their common work."""
    return pl.collect_all(frames, engine=_engine(), optimizations=_AS_WRITTEN)
