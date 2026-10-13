"""A parameter given as a large parquet path is streamed from a copy rather than held in memory.

Every test lowers ``streamed.ROWS`` to 0, so the dispatch example's files stream.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING

import polars as pl
import pytest

import specsolve as sps
from specsolve import streamed
from specsolve.errors import DataError

if TYPE_CHECKING:
    from pathlib import Path

PARAMETERS = {'p_max', 'cost', 'load'}


@pytest.fixture
def paths(dispatch_frame_inputs, tmp_path) -> dict[str, str]:
    """The dispatch example's sources, each written to its own parquet file."""
    out = {}
    for name, frame in dispatch_frame_inputs.items():
        frame.write_parquet(path := tmp_path / f'{name}.parquet')
        out[name] = str(path)
    return out


@pytest.fixture
def stream_everything(monkeypatch) -> None:
    monkeypatch.setattr(streamed, 'ROWS', 0)


def _written(dispatch_yaml: Path, sources: dict[str, str], out: Path) -> str:
    with sps.build(dispatch_yaml, sources) as model:
        model.write(out)
    return out.read_text()


def test_a_streamed_parameter_builds_the_model_a_held_one_builds(dispatch_yaml, paths, tmp_path, monkeypatch):
    held = _written(dispatch_yaml, paths, tmp_path / 'held.lp')
    monkeypatch.setattr(streamed, 'ROWS', 0)
    with sps.build(dispatch_yaml, paths) as model:
        assert model._copies is not None
        assert model._copies.names == PARAMETERS, 'every parameter path streams at a threshold of zero rows'
    assert _written(dispatch_yaml, paths, tmp_path / 'streamed.lp') == held


def test_a_small_parameter_is_held_in_memory(dispatch_yaml, paths):
    with sps.build(dispatch_yaml, paths) as model:
        assert model._copies is not None
        assert model._copies.names == set(), 'no example file reaches the default threshold'


@pytest.mark.usefixtures('stream_everything')
def test_a_file_rewritten_after_the_build_reaches_neither_an_expression_nor_the_archive(
    dispatch_yaml, dispatch_frame_inputs, paths, tmp_path
):
    """The build streams from a copy, so the file the caller named is free to change."""
    with sps.build(dispatch_yaml, paths) as model:
        dispatch_frame_inputs['cost'].with_columns(pl.col('value') * 10).write_parquet(paths['cost'])
        with (
            model.solve(archive=tmp_path / 'archive') as after,
            sps.solve(dispatch_yaml, dispatch_frame_inputs) as before,
        ):
            assert after.objective == pytest.approx(before.objective, rel=1e-9)
            assert after.evaluate('sum(p * cost)')['value'].sum() == pytest.approx(
                before.evaluate('sum(p * cost)')['value'].sum(), rel=1e-9
            ), 'an expression reads the cost the build read'
    held = pl.read_parquet(tmp_path / 'archive' / 'sources' / 'cost.parquet')
    assert held['value'].to_list() == dispatch_frame_inputs['cost']['value'].to_list(), (
        'the archive holds the cost the build read'
    )


@pytest.mark.usefixtures('stream_everything')
def test_a_result_reads_a_streamed_parameter_after_its_model_closes(dispatch_yaml, dispatch_frame_inputs, paths):
    with sps.solve(dispatch_yaml, paths) as result, sps.solve(dispatch_yaml, dispatch_frame_inputs) as held:
        assert result.evaluate('sum(p * cost)')['value'].sum() == pytest.approx(
            held.evaluate('sum(p * cost)')['value'].sum(), rel=1e-9
        )


@pytest.mark.usefixtures('stream_everything')
def test_the_copies_are_deleted_once_nothing_reads_them(dispatch_yaml, paths):
    model = sps.build(dispatch_yaml, paths)
    copies = model._copies
    assert copies is not None
    directory = copies._directory
    assert directory is not None
    result = model.solve()
    model.close()
    del copies
    gc.collect()
    assert directory.exists(), 'an open result still reads from the copies'
    result.close()
    del result, model
    gc.collect()
    assert not directory.exists(), 'nothing reads the copies any more'


@pytest.mark.parametrize(
    'p_max',
    [
        pytest.param({'generator': ['wind', 'wind'], 'value': [100.0, 90.0]}, id='a-coordinate-twice'),
        pytest.param({'generator': ['wind', 'gas'], 'value': [100.0, None]}, id='a-null-value'),
        pytest.param({'generator': ['wind', 'nuclear'], 'value': [100.0, 200.0]}, id='an-unknown-label'),
        pytest.param({'generator': ['wind', 'gas'], 'value': ['100', '200']}, id='a-value-of-another-type'),
    ],
)
def test_a_defect_in_a_streamed_parameter_is_named_as_a_held_one_is(dispatch_yaml, paths, monkeypatch, p_max):
    pl.DataFrame(p_max).write_parquet(paths['p_max'])
    with pytest.raises(DataError) as held:
        sps.build(dispatch_yaml, paths)
    monkeypatch.setattr(streamed, 'ROWS', 0)
    with pytest.raises(DataError) as streaming:
        sps.build(dispatch_yaml, paths)
    assert str(streaming.value) == str(held.value)
