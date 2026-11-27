"""A dimension's labels are of the type its `dtype` declares, `str` unless declared, or refused at load.

The declared `dtype` is the type of a label. Labels of another type used to be
taken as they came, so a spec could declare one type and its data supply
another, and only a later comparison noticed. Within a `dtype`, labels of any
width or encoding are read into its one type, and a date under
`dtype: datetime` reads as the microsecond its day starts on.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import specsolve as sps
from tests.differential import both_lanes_refuse

DAYS = [dt.date(2030, 1, 1), dt.date(2030, 1, 2)]
ENUM = pl.Enum(['a', 'b'])


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
        pytest.param('str', pl.Series(['a', 'b'], dtype=pl.Enum(['a', 'b'])), id='enum-strings'),
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


@pytest.mark.parametrize(
    'encoding',
    [
        pytest.param(pl.Categorical, id='categorical'),
        pytest.param(pl.Enum(['a', 'b', 'c']), id='an-enum-in-another-order-with-a-category-to-spare'),
    ],
)
def test_string_labels_with_a_dictionary_of_their_own_are_read_as_plain_strings(encoding):
    """A `Categorical` or an `Enum` is an encoding of strings, and the labels are the strings, in the order of their rows.

    The labels were held in the dictionary they arrived in. `Categorical` labels left a result as `Categorical`,
    which a caller's `String` column does not join, and `Enum` labels failed the build with a polars error (#2255).
    """
    labels = pl.Series('d', ['b', 'a'], dtype=encoding)
    cap = pl.DataFrame({'d': ['a', 'b'], 'value': [1.0, 2.0]})
    with sps.solve(_spec(None), {'d': labels, 'cap': cap}) as result:
        primal = result.primal('x')
    assert primal.schema['d'] == pl.String, 'what leaves is what a caller can join against'
    assert primal.rows() == [('b', 2.0), ('a', 1.0)], 'each label in the order of its row, at its own cap'


MICROSECONDS = pl.Datetime('us')


@pytest.mark.parametrize(
    ('dtype', 'labels', 'held', 'column'),
    [
        pytest.param('int', pl.Series([1, 2], dtype=pl.Int8), pl.Int64, pl.Int8, id='a-narrower-integer'),
        pytest.param('int', pl.Series([1, 2], dtype=pl.UInt32), pl.Int64, pl.UInt32, id='an-unsigned-integer'),
        pytest.param('float', pl.Series([0.5, 1.5], dtype=pl.Float32), pl.Float64, pl.Float32, id='a-narrower-float'),
        pytest.param(None, pl.Series(['a', 'b'], dtype=pl.Categorical), pl.String, pl.Categorical, id='a-categorical'),
        pytest.param(None, pl.Series(['a', 'b'], dtype=ENUM), pl.String, ENUM, id='an-enum'),
        pytest.param('datetime', pl.Series(DAYS), MICROSECONDS, MICROSECONDS, id='dates'),
        pytest.param(
            'datetime',
            pl.Series(DAYS).cast(pl.Datetime('ns', 'UTC')),
            pl.Datetime('us', 'UTC'),
            pl.Datetime('us', 'UTC'),
            id='nanoseconds-in-a-time-zone',
        ),
    ],
)
def test_the_tidy_holds_labels_in_the_declared_dtype_and_a_column_as_it_arrived(dtype, labels, held, column):
    """A dimension's labels are held in the one type its `dtype` names, whatever width or encoding they arrived in.

    A parameter's column over the dimension stays the caller's. A datetime is the exception on both
    sides: it is read in microseconds, and keeps its time zone.
    """
    tables = sps.tidy(_spec(dtype), _sources(labels))
    assert tables['d'].schema['d'] == held, f'the labels of a `{dtype or "str"}` dimension are {held}'
    assert tables['cap'].schema['d'] == column, 'the column of a parameter is not converted beyond a datetime'
    with sps.solve(_spec(dtype), _sources(labels)) as result:
        assert result.primal('x').schema['d'] == held, 'and a result names each label in the type of the labels'


def test_integer_labels_past_the_signed_64_bit_range_are_refused():
    """`dtype: int` labels are held as `Int64`, which an unsigned integer above its range does not fit."""
    labels = pl.Series([1, 2**63], dtype=pl.UInt64)
    sentence = both_lanes_refuse(_spec('int'), _sources(labels), match='Int64')
    assert sentence.startswith("the labels of dimension 'd' hold"), 'it names the dimension'
    assert str(2**63) in sentence, 'and a label that does not fit'


def test_an_empty_labels_table_takes_the_declared_dtype():
    """No label is held, so an empty table, even one polars typed `Null`, reads as the declared type."""
    spec = _spec('int') | {'parameters': {'cap': {'dims': ['d'], 'missing': 'absent'}}}
    sources = {'d': pl.DataFrame({'d': []}), 'cap': pl.DataFrame(schema={'d': pl.Int64, 'value': pl.Float64})}
    tables = sps.tidy(spec, sources)
    assert tables['d'].schema['d'] == pl.Int64, 'the empty labels are read as the declared int'
