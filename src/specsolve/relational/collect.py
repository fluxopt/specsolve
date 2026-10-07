"""The one place that decides how polars materialises a frame.

A collect names polars' ``auto`` engine, so polars chooses it. A polars built
without the streaming engine — the browser's, for one — panics on a streaming
request rather than falling back, so the question is put once and such a
polars is asked for the in-memory engine by name.

Every plan here orders its joins by hand, so polars' own join reordering is
off: it moves a join between two others that share its keys, and the one it
runs first can be keyed on a part of them, which multiplies the rows it holds.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING, Literal

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ['collect_engine', 'collected', 'collected_all']

_AS_WRITTEN = pl.QueryOptFlags(join_order=False)


@cache
def collect_engine() -> Literal['auto', 'in-memory']:
    """The engine a collect names: polars' choice where this polars has the streaming engine, in-memory otherwise."""
    try:
        pl.LazyFrame({'probe': [0]}).collect(engine='streaming')
    except BaseException:  # a refusal is a pyo3 panic, which is not an Exception
        return 'in-memory'
    return 'auto'


def collected(frame: pl.LazyFrame, *, in_memory: bool = False) -> pl.DataFrame:
    """*frame* materialised on [`collect_engine`][], or the in-memory engine where asked, its joins as written."""
    return frame.collect(engine='in-memory' if in_memory else collect_engine(), optimizations=_AS_WRITTEN)


def collected_all(frames: Iterable[pl.LazyFrame]) -> list[pl.DataFrame]:
    """Every one of *frames*, as [`collected`][] would, in one pass that shares their common work."""
    return pl.collect_all(frames, engine=collect_engine(), optimizations=_AS_WRITTEN)
