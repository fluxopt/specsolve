"""A dimension's labels are of the type its `dtype` declares, `str` unless declared, or refused at load.

The declared `dtype` is the type of a label. Labels of another type used to be
taken as they came, so a spec could declare one type and its data supply
another, and only a later comparison noticed. A date under `dtype: datetime`
is the one conversion: it reads as the microsecond its day starts on.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import specsolve as sps
from tests.differential import both_lanes_refuse

DAYS = [dt.date(2030, 1, 1), dt.date(2030, 1, 2)]


def _spec(dtype: str | None) -> dict[str, object]:
    return {
        'dimensions': {'d': {} if dtype is None else {'dtype': dtype}},
        'parameters': {'cap': {'dims': ['d']}},
        'variables': {'x': {'dims': ['d'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }


def _sources(labels: pl.Series) -> dict[str, object]:
    return {'d': pl.DataFrame({'d': labels}), 'cap': pl.DataFrame({'d': labels, 'value': [1.0, 2.0]})}


@pytest.mark.parametrize(
    ('dtype', 'labels', 'given'),
    [
        pytest.param('int', pl.Series(['1', '2']), 'String', id='int-given-strings'),
        pytest.param(None, pl.Series([1, 2]), 'Int64', id='str-by-default-given-integers'),
        pytest.param(
            'str', pl.Series([dt.datetime(2030, 1, 1), dt.datetime(2030, 1, 2)]), 'Datetime', id='str-given-datetimes'
        ),
        pytest.param('datetime', pl.Series(['2030-01-01', '2030-01-02']), 'String', id='datetime-given-strings'),
        pytest.param('float', pl.Series([1, 2]), 'Int64', id='float-given-integers'),
        pytest.param('str', pl.Series(DAYS), 'Date', id='str-given-dates'),
    ],
)
def test_labels_of_another_type_than_the_declared_dtype_are_refused(dtype, labels, given):
    """The refusal names the dimension, the type given and the type declared, on both lanes."""
    sentence = both_lanes_refuse(_spec(dtype), _sources(labels), match='declares')
    assert sentence.startswith(f"the labels of dimension 'd' are {given}"), 'it names the dimension and the type given'
    assert f'dtype: {dtype or "str"}' in sentence, 'and the dtype the spec declares, `str` where it declares none'


@pytest.mark.parametrize(
    ('dtype', 'labels'),
    [
        pytest.param('datetime', pl.Series(DAYS), id='dates-under-datetime'),
        pytest.param('int', pl.Series([1, 2], dtype=pl.Int32), id='a-narrower-integer'),
        pytest.param('str', pl.Series(['a', 'b'], dtype=pl.Categorical), id='categorical-strings'),
        pytest.param(
            'datetime',
            pl.Series([dt.datetime(2030, 1, 1), dt.datetime(2030, 1, 2)]).dt.cast_time_unit('ns'),
            id='datetimes-in-nanoseconds',
        ),
    ],
)
def test_labels_the_declared_dtype_holds_build(dtype, labels):
    """Within a dtype any width, encoding or unit builds; a date under `datetime` is its day's first microsecond."""
    with sps.build(_spec(dtype), _sources(labels)) as model:
        assert model.metrics().columns == 2, 'one column per label'


def test_an_empty_labels_table_takes_the_declared_dtype():
    """No label is held, so an empty table, even one polars typed `Null`, reads as the declared type."""
    spec = _spec('int') | {'parameters': {'cap': {'dims': ['d'], 'missing': 'absent'}}}
    sources = {'d': pl.DataFrame({'d': []}), 'cap': pl.DataFrame(schema={'d': pl.Int64, 'value': pl.Float64})}
    tables = sps.tidy(spec, sources)
    assert tables['d'].schema['d'] == pl.Int64, 'the empty labels are read as the declared int'
