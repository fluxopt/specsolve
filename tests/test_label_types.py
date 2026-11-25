"""A label column is read in its dimension's label type, or refused at load by name.

A label compares only with labels of its own type. A table that gives a
dimension's labels in another type, integers as strings or datetimes where the
labels are strings, used to escape the build as a polars error from deep in a
membership test or a join (#2220).
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

import specsolve as sps
from tests.differential import both_lanes_refuse

PORTS = {
    'dimensions': {'unit': {}, 'port': {'dtype': 'int'}},
    'relations': {'home': {'key': 'unit', 'values': 'port'}},
    'parameters': {
        'eff': {'dims': ['unit', 'port']},
        'delay': {'dims': ['unit', 'port'], 'missing': 'absent'},
    },
    'variables': {'p': {'dims': ['unit', 'port'], 'bounds': {'lower': 0}}},
    'constraints': {'cap': {'dims': ['unit', 'port'], 'expression': 'eff * p + delay <= 10'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}

SNAPSHOTS = {
    'dimensions': {'snapshot': {'dtype': 'str'}},
    'parameters': {'load': {'dims': ['snapshot']}},
    'variables': {'p': {'dims': ['snapshot'], 'bounds': {'lower': 0}}},
    'constraints': {'meet': {'dims': ['snapshot'], 'expression': 'p >= load'}},
    'objective': {'sense': 'minimize', 'expression': 'sum(p)'},
}
INSTANTS = [dt.datetime(2013, 5, 1, h) for h in (0, 3)]
SPELLED = ['2013-05-01 00:00:00', '2013-05-01 03:00:00']


def _ports(**replaced: object) -> dict[str, object]:
    return {
        'unit': ['a', 'b'],
        'port': [0, 1],
        'home': pl.DataFrame({'unit': ['a', 'b'], 'port': [0, 1]}),
        'eff': pl.DataFrame({'unit': ['a', 'a', 'b', 'b'], 'port': [0, 1, 0, 1], 'value': 1.0}),
        'delay': pl.DataFrame({'unit': ['a'], 'port': [1], 'value': [1.0]}),
    } | replaced


@pytest.mark.parametrize(
    ('spec', 'sources', 'owner'),
    [
        pytest.param(
            PORTS,
            _ports(delay=pl.DataFrame({'unit': ['a'], 'port': ['1'], 'value': [1.0]})),
            "parameter 'delay' holds 'port' as String",
            id='an-int-dimension-given-as-strings-in-a-parameter',
        ),
        pytest.param(
            PORTS,
            _ports(home=pl.DataFrame({'unit': ['a', 'b'], 'port': ['0', '1']})),
            "relation 'home' holds 'port' as String",
            id='an-int-dimension-given-as-strings-in-a-relation',
        ),
        pytest.param(
            SNAPSHOTS,
            {'snapshot': SPELLED, 'load': pl.DataFrame({'snapshot': INSTANTS, 'value': [1.0, 2.0]})},
            "parameter 'load' holds 'snapshot' as Datetime",
            id='string-labels-given-as-datetimes',
        ),
        pytest.param(
            SNAPSHOTS,
            {
                'snapshot': pl.DataFrame({'snapshot': INSTANTS}),
                'load': pl.DataFrame({'snapshot': SPELLED, 'value': [1.0, 2.0]}),
            },
            "parameter 'load' holds 'snapshot' as String",
            id='datetime-labels-given-as-strings',
        ),
    ],
)
def test_a_label_column_of_another_type_than_its_labels_is_refused_by_name(spec, sources, owner):
    """The refusal names the source, the column, the type given and the labels' type, on both lanes."""
    sentence = both_lanes_refuse(spec, sources, match='holds')
    assert sentence.startswith(owner), 'the refusal names what carried the column and the type it came in'


@pytest.mark.parametrize(
    'empty',
    [
        pytest.param(
            pl.DataFrame(schema={'unit': pl.String, 'port': pl.String, 'value': pl.Float64}), id='typed-as-strings'
        ),
        pytest.param(
            pl.DataFrame(schema={'unit': pl.String, 'port': pl.Null, 'value': pl.Float64}), id='typed-as-null'
        ),
    ],
)
def test_an_empty_table_takes_its_labels_type(empty):
    """An empty table holds no label to misread, so its column is read in the labels' type and the model builds as without it."""
    with sps.build(
        PORTS, _ports(delay=pl.DataFrame(schema={'unit': pl.String, 'port': pl.Int64, 'value': pl.Float64}))
    ) as typed:
        expected = typed.metrics()
    with sps.build(PORTS, _ports(delay=empty)) as model:
        got = model.metrics()
    assert (got.columns, got.rows, got.nonzeros) == (expected.columns, expected.rows, expected.nonzeros), (
        'an empty table of another column type builds the model an empty table of the labels type does'
    )


def test_an_integer_column_of_another_width_is_read_as_its_labels():
    """Int32 beside Int64 labels is the same kind of label, so it builds, as it always has."""
    narrow = pl.DataFrame({'unit': ['a'], 'port': pl.Series([1], dtype=pl.Int32), 'value': [1.0]})
    with sps.build(PORTS, _ports(delay=narrow)) as model:
        assert model.metrics().columns == 4, 'one column per (unit, port)'
