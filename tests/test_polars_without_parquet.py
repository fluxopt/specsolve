"""A polars built without parquet, such as the browser's: every file goes through pyarrow, or is refused by name.

polars' Pyodide build has no parquet: each read and write raises an
``AttributeError`` from inside polars. Every call that reads or writes a file
asks first. With pyarrow importable it goes through pyarrow; without, it is
refused with a message that names the call and what still works.
"""

from __future__ import annotations

import sys
import zoneinfo
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve import _parquet as parquet
from specsolve._names import RUN

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


def _missing(*args: object, **kwargs: object) -> None:
    raise AttributeError("'builtins.PyLazyFrame' object has no attribute 'sink_parquet'")


def _take_parquet_away(monkeypatch: pytest.MonkeyPatch, *, pyarrow: bool = False) -> None:
    """Every polars parquet call raise ``AttributeError``, as the browser build's do, pyarrow present or not."""
    for owner, name in (
        (pl.DataFrame, 'write_parquet'),
        (pl.LazyFrame, 'sink_parquet'),
        (pl, 'read_parquet'),
        (pl, 'scan_parquet'),
        (pl, 'read_parquet_schema'),
    ):
        monkeypatch.setattr(owner, name, _missing)
    if not pyarrow:
        monkeypatch.setattr(parquet, 'find_spec', lambda name: None)
    parquet._polars_has_parquet.cache_clear()


@pytest.fixture
def polars_without_parquet(monkeypatch):
    """A polars without parquet, and no pyarrow, for the whole test."""
    _take_parquet_away(monkeypatch)
    yield
    parquet._polars_has_parquet.cache_clear()


def test_the_probe_sees_no_parquet(polars_without_parquet):
    assert not parquet._polars_has_parquet(), 'the probe reads the missing parquet as no parquet'


def test_polars_has_parquet_where_it_was_built_with_it(polars_parquet):
    parquet._polars_has_parquet.cache_clear()
    assert parquet.require('a test') == 'polars', 'the polars this suite runs on reads and writes parquet'


def test_a_solve_from_tables_needs_no_parquet(polars_without_parquet, dispatch_yaml, dispatch_table_inputs):
    result = sps.solve(dispatch_yaml, dispatch_table_inputs)
    assert result.termination_condition == 'optimal'
    sweep = sps.solve_over(dispatch_yaml, _by_scenario(dispatch_table_inputs), sps.EachLabel('scenario'))
    assert sweep.records['termination_condition'].unique().to_list() == ['optimal'], 'every scenario solved'


def _saved(dispatch_yaml: Path, inputs: dict[str, object], tmp_path: Path) -> dict[str, Path]:
    """One of each thing on disk a reader takes, written while parquet still works."""
    sps.solve(dispatch_yaml, inputs).save(tmp_path / 'result')
    sps.solve(dispatch_yaml, inputs, archive=tmp_path / 'archive')
    sps.solve_over(dispatch_yaml, _by_scenario(inputs), sps.EachLabel('scenario'), spill_to=tmp_path / 'sweep')
    inputs['load'].write_parquet(tmp_path / 'load.parquet')
    return {
        'result': tmp_path / 'result',
        'archive': tmp_path / 'archive',
        'sweep': tmp_path / 'sweep',
        'path': tmp_path / 'load.parquet',
    }


def _by_scenario(inputs: dict[str, object]) -> dict[str, object]:
    load = inputs['load']
    assert isinstance(load, pl.DataFrame)
    scenarios = pl.concat([load.with_columns(scenario=pl.lit(name)) for name in ('low', 'high')])
    return {**inputs, 'load': scenarios}


CALLS: list[tuple[str, Callable[..., object], str]] = [
    (
        'a-source-path',
        lambda yaml, inputs, saved, tmp: sps.solve(yaml, {**inputs, 'load': saved['path']}),
        'a source given as the path',
    ),
    ('solve-archive', lambda yaml, inputs, saved, tmp: sps.solve(yaml, inputs, archive=tmp / 'new'), 'archive='),
    (
        'solve-over-archive',
        lambda yaml, inputs, saved, tmp: sps.solve_over(
            yaml, _by_scenario(inputs), sps.EachLabel('scenario'), archive=tmp / 'new'
        ),
        'archive=',
    ),
    (
        'solve-over-spill',
        lambda yaml, inputs, saved, tmp: sps.solve_over(
            yaml, _by_scenario(inputs), sps.EachLabel('scenario'), spill_to=tmp / 'new'
        ),
        'spill_to=',
    ),
    ('result-save', lambda yaml, inputs, saved, tmp: sps.solve(yaml, inputs).save(tmp / 'new'), 'Result.save'),
    (
        'sweep-save',
        lambda yaml, inputs, saved, tmp: sps.solve_over(yaml, _by_scenario(inputs), sps.EachLabel('scenario')).save(
            tmp / 'new'
        ),
        'Sweep.save',
    ),
    ('load-result', lambda yaml, inputs, saved, tmp: sps.load_result(saved['result']), 'reading a saved answer'),
    ('scan-result', lambda yaml, inputs, saved, tmp: sps.scan_result(saved['result']), 'reading a saved answer'),
    ('load-sweep', lambda yaml, inputs, saved, tmp: sps.load_sweep(saved['sweep']), 'reading a saved sweep'),
    ('scan-sweep', lambda yaml, inputs, saved, tmp: sps.scan_sweep(saved['sweep']), 'reading a saved sweep'),
    ('load-archive', lambda yaml, inputs, saved, tmp: sps.load_archive(saved['archive']), 'reading an archive'),
    ('scan-archive', lambda yaml, inputs, saved, tmp: sps.scan_archive(saved['archive']), 'reading an archive'),
    ('load-inputs', lambda yaml, inputs, saved, tmp: sps.load_inputs(saved['archive']), 'reading an archive'),
]


@pytest.mark.parametrize(('call', 'names'), [pytest.param(call, names, id=label) for label, call, names in CALLS])
def test_every_call_that_needs_parquet_says_so_by_name(
    monkeypatch, call, names, dispatch_yaml, dispatch_table_inputs, tmp_path
):
    """The refusal names the call, where polars would raise an ``AttributeError`` from inside itself."""
    saved = _saved(dispatch_yaml, dispatch_table_inputs, tmp_path)
    _take_parquet_away(monkeypatch)
    try:
        with pytest.raises(sps.errors.SpecsolveError, match='this polars was built without it') as raised:
            call(dispatch_yaml, dispatch_table_inputs, saved, tmp_path)
    finally:
        parquet._polars_has_parquet.cache_clear()
    assert names in str(raised.value), 'the refusal names the call'
    assert 'Load pyarrow first' in str(raised.value), 'and names what lifts it'
    assert not (tmp_path / 'new').exists(), 'nothing was written before the refusal'


@pytest.mark.parametrize('call', [pytest.param(call, id=label) for label, call, _ in CALLS])
def test_every_call_that_needs_parquet_goes_through_pyarrow_where_it_can_be_imported(
    monkeypatch, call, dispatch_yaml, dispatch_table_inputs, tmp_path
):
    """The same calls succeed where pyarrow is importable, reading what polars wrote and writing what it reads."""
    pytest.importorskip('pyarrow')
    saved = _saved(dispatch_yaml, dispatch_table_inputs, tmp_path)
    _take_parquet_away(monkeypatch, pyarrow=True)
    try:
        call(dispatch_yaml, dispatch_table_inputs, saved, tmp_path)
    finally:
        parquet._polars_has_parquet.cache_clear()


@pytest.fixture
def no_time_zone_database(monkeypatch):
    """A Python that can name no time zone, as the browser's is until ``tzdata`` is loaded."""
    monkeypatch.setitem(sys.modules, 'tzdata', None)
    zoneinfo.reset_tzpath(to=[])
    zoneinfo.ZoneInfo.clear_cache()
    yield
    zoneinfo.reset_tzpath()
    zoneinfo.ZoneInfo.clear_cache()


def test_a_sweep_is_archived_and_read_back_with_no_time_zone_database(
    monkeypatch, no_time_zone_database, dispatch_yaml, dispatch_table_inputs, tmp_path
):
    """Archiving a sweep read each slice's record out as Python values, and its UTC timestamp needs the database.

    The browser has none, so polars failed on the first slice. The record now
    stays a table from the fold to the file. These are the steps of
    ``tools/pyodide/probe.py``, which is where it was seen.
    """
    pytest.importorskip('pyarrow')
    _take_parquet_away(monkeypatch, pyarrow=True)
    try:
        sources = _by_scenario(dispatch_table_inputs)
        runs = sps.solve_over(dispatch_yaml, sources, sps.EachLabel('scenario'), archive=tmp_path / 'study.zip')
        study = sps.load_archive(tmp_path / 'study.zip', tmp_path / 'study')
        again = sps.solve_over(study.spec, study.sources, study.axis)
    finally:
        parquet._polars_has_parquet.cache_clear()
    assert study.sweep.records.drop(RUN).equals(runs.records.drop(RUN)), 'every record, its timestamp included'
    assert study.sweep.primal('p').equals(runs.primal('p')), 'and every value'
    assert again.records['objective'].to_list() == pytest.approx(runs.records['objective'].to_list()), (
        'and it solves again from what the archive holds'
    )
