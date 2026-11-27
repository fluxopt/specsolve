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


def test_a_dimension_is_read_once_however_many_parameters_and_relations_are_over_it(monkeypatch):
    """PyPSA's spec declares hundreds of parameters over a dozen dimensions; each check read its dimensions' labels again."""
    from specsolve import _sources as reading

    reads, real = [], reading._labels_table

    def counted(source, dim, dtype):
        reads.append(dim)
        return real(source, dim, dtype)

    monkeypatch.setattr(reading, '_labels_table', counted)
    spec = {
        **SPEC,
        'dimensions': {**SPEC['dimensions'], 'site': {}},
        'relations': {'sited': {'key': 'g', 'values': 'site'}},
        'parameters': {**SPEC['parameters'], 'cost': {'dims': ['g']}, 'floor': {'dims': ['g', 't']}},
    }
    sources = _sources(_dense()) | {
        'site': ['north', 'south'],
        'sited': pl.DataFrame({'g': GENERATORS, 'site': ['north', 'south'] * (len(GENERATORS) // 2)}),
        'cost': pl.DataFrame({'g': GENERATORS, 'value': 1.0}),
        'floor': _dense().with_columns(value=0.0),
    }
    sps.build(spec, sources).close()
    assert sorted(reads) == ['g', 'site', 't'], (
        f'the labels were read as {sorted(reads)}, where each dimension once is enough'
    )


def test_a_dimension_is_cast_into_its_enum_by_one_expression_however_many_tables_carry_it(monkeypatch):
    """Building a cast to an `Enum` of many labels costs a fixed time per expression, and PyPSA casts hundreds of tables."""
    built, real = [], pl.Expr.cast

    def counted(self, dtype, *args, **kwargs):
        if isinstance(dtype, pl.Enum):
            built.append(tuple(dtype.categories))
        return real(self, dtype, *args, **kwargs)

    monkeypatch.setattr(pl.Expr, 'cast', counted)
    spec = {**SPEC, 'parameters': {**SPEC['parameters'], 'cost': {'dims': ['g']}, 'floor': {'dims': ['g']}}}
    sources = _sources(_dense()) | {
        'cost': pl.DataFrame({'g': GENERATORS, 'value': 1.0}),
        'floor': pl.DataFrame({'g': GENERATORS, 'value': 0.0}),
    }
    sps.build(spec, sources).close()
    casts = sum(1 for categories in built if categories == tuple(GENERATORS))
    assert casts == 1, f'`g` was cast by {casts} expressions, where the one its three tables share is enough'


def test_a_large_table_over_one_dimension_is_checked_by_hashing_its_column():
    """One key column hashes once either way, so a lookup of each row's label into a position only adds to it.

    #2154's A/B had the position path double the tidy of `grid-dims-1-l`, a table of 1.2M rows over one dimension.
    """
    from specsolve._attached import Dimension
    from specsolve._encoding import _coordinate_positions

    labels = pl.Series('g', [f'g{i}' for i in range(60_000)])
    table = pl.DataFrame({'g': labels, 'value': 1.0})
    assert _coordinate_positions(('g',), table, {'g': Dimension.of('g', labels)}) is None, (
        'one column is hashed, not turned into positions'
    )
