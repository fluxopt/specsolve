"""A dense parameter table is held to one row per coordinate and real labels, on the path a large table takes.

PyPSA hands its spec a dense table per attribute and component, millions of
rows each, and checking them is most of what reading the sources costs. A
large table is checked through each row's position in the product of its
dimensions' labels; these hold it to the same refusals as a small one.
"""

from __future__ import annotations

import polars as pl
import pytest

import specsolve as sps

#: 300 by 200, 60,000 rows, enough to take the large-table path.
GENERATORS = [f'g{i}' for i in range(300)]
HOURS = list(range(200))

SPEC = {
    'dimensions': {'g': {}, 't': {'dtype': 'int'}},
    'parameters': {'cap': {'dims': ['g', 't']}},
    'variables': {'p': {'dims': ['g', 't'], 'bounds': {'lower': 0, 'upper': 'cap'}}},
    'objective': {'sense': 'maximize', 'expression': 'sum(p)'},
}


def _sources(cap: pl.DataFrame) -> dict[str, object]:
    return {'g': GENERATORS, 't': HOURS, 'cap': cap}


def _dense() -> pl.DataFrame:
    return pl.DataFrame({'g': GENERATORS}).join(pl.DataFrame({'t': HOURS}), how='cross').with_columns(value=1.0)


def test_a_dense_table_with_one_row_per_coordinate_builds():
    with sps.build(SPEC, _sources(_dense())) as model:
        assert model.metrics().columns == 60_000, 'one column per row of `cap`'


@pytest.mark.parametrize(
    ('cap', 'refusal'),
    [
        pytest.param(pl.concat([_dense(), _dense().tail(1)]), 'more than one row for a coordinate', id='a-row-twice'),
        pytest.param(
            _dense().with_columns(g=pl.when(pl.int_range(pl.len()) == 7).then(pl.lit('gx')).otherwise('g')),
            "dimension 'g' that are not labels of it: 'gx'",
            id='a-label-outside-its-dimension',
        ),
        pytest.param(
            _dense().with_columns(t=pl.when(pl.int_range(pl.len()) == 7).then(999).otherwise('t')),
            "dimension 't' that are not labels of it: 999",
            id='a-label-outside-an-integer-dimension',
        ),
        pytest.param(
            pl.concat([_dense(), _dense().tail(1)]).with_columns(pl.col('t').cast(pl.Int32)),
            'more than one row for a coordinate',
            id='a-row-twice-under-a-column-of-another-dtype',
        ),
    ],
)
def test_a_dense_table_is_refused_as_a_small_one_is(cap, refusal):
    with pytest.raises(sps.errors.DataError, match=refusal):
        sps.build(SPEC, _sources(cap))
