"""A polars built without the streaming engine, such as the browser's, still solves and reads back.

polars' Pyodide build refuses ``collect(engine='streaming')`` with a pyo3 panic,
which is a ``BaseException`` rather than an ``Exception``. A refusal chooses the
in-memory engine.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps
from specsolve.relational import collect


class PanicException(BaseException):
    """What pyo3 raises when polars panics: not an ``Exception``, so an ordinary ``except`` misses it."""


@pytest.fixture
def polars_without_streaming(monkeypatch):
    """A polars whose streaming engine is missing, the way the browser build's is."""
    original = pl.LazyFrame.collect

    def refuse(self, *args, engine='auto', **kwargs):
        if engine == 'streaming':
            raise PanicException("activate 'new_streaming, ' feature")
        return original(self, *args, engine=engine, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', refuse)
    collect.polars_engine.cache_clear()
    yield
    collect.polars_engine.cache_clear()


def test_the_probe_sees_the_refusal(polars_without_streaming):
    assert collect.polars_engine() == 'in-memory'


def test_a_solve_and_its_readers_fall_back_to_the_in_memory_engine(
    polars_without_streaming, dispatch_yaml, dispatch_frame_inputs
):
    result = sps.solve(dispatch_yaml, dispatch_frame_inputs)
    assert result.termination_condition == 'optimal'
    assert result.primal('p')['value'].sum() == pytest.approx(result.activity('power_balance')['value'].sum()), (
        'the primals and the activities were read back through the same engine and agree'
    )
    assert (result.dual('power_balance')['value'] > 0).all(), 'every snapshot has a price'


def test_the_streaming_engine_is_used_where_polars_has_one():
    collect.polars_engine.cache_clear()
    assert collect.polars_engine() == 'streaming', 'the polars this suite runs on has the streaming engine'
