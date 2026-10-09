"""The one place that decides how polars materialises a frame.

A collect names polars' ``auto`` engine, so polars chooses it, except inside
[`in_memory`][], where it names the in-memory engine. A polars
built without the streaming engine — the browser's, for one — panics on a
streaming request rather than falling back, so the question is put once and
such a polars is asked for the in-memory engine by name.

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

__all__ = ['collect_engine', 'collected', 'collected_all', 'in_memory', 'is_small']

_AS_WRITTEN = pl.QueryOptFlags(join_order=False)

#: The most columns a model may have for its build to collect in memory.
IN_MEMORY_COLUMNS = 250_000

_in_memory: ContextVar[bool] = ContextVar('in_memory', default=False)


@cache
def collect_engine() -> Literal['auto', 'in-memory']:
    """The engine a collect names: polars' choice where this polars has the streaming engine, in-memory otherwise."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return 'in-memory'
    return 'auto'


def is_small(columns: int) -> bool:
    """Whether a model of *columns* builds [`in_memory`][]: at most [`IN_MEMORY_COLUMNS`][]."""
    return columns <= IN_MEMORY_COLUMNS


@contextmanager
def in_memory(on: bool = True) -> Iterator[None]:
    """Every collect in the block runs on the in-memory engine where *on*, and on [`collect_engine`][] otherwise.

    polars 2.0's streaming engine pays a fixed cost per query, which is most of
    a small model's build; what it buys is a lower peak on a large one. The
    choice is per thread, so each slice of a sweep makes its own.
    """
    token = _in_memory.set(on)
    try:
        yield
    finally:
        _in_memory.reset(token)


def _engine() -> Literal['auto', 'in-memory']:
    """The engine a collect names: in memory inside [`in_memory`][], else [`collect_engine`][]."""
    return 'in-memory' if _in_memory.get() else collect_engine()


def collected(frame: pl.LazyFrame) -> pl.DataFrame:
    """*frame* materialised on [`_engine`][], its joins as written."""
    return frame.collect(engine=_engine(), optimizations=_AS_WRITTEN)


def collected_all(frames: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Every one of *frames*, as [`collected`][] would, in one pass that shares their common work."""
    return pl.collect_all(frames, engine=_engine(), optimizations=_AS_WRITTEN)
