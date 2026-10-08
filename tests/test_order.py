"""Row order carries no meaning: the same data builds the same model.

A source table's rows arrive in whatever order its writer chose, and polars
promises no row order from a join or a group-by that was not asked for one. So
each referenced model is built twice from the same sources, and again with the
rows of every parameter and relation table shuffled, and each build must
digest the same: matrix, bounds, costs and right-hand sides, bit for bit. A
saved answer is checked against that digest, so one that moves refuses an
archive whose data nothing changed.

Dimension tables keep their order: a dimension's row order is its coordinate
order, which ``shift`` reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl
import pytest

import specsolve as sps
from specsolve.relational.engine.pieces import ordered_sum, ordered_sum_per
from tests.conftest import expanded
from tests.conftest import port_sources as sources

if TYPE_CHECKING:
    from collections.abc import Mapping


def _digest(port: dict[str, Any], given: Mapping[str, Any]) -> str:
    with sps.build(expanded(port['spec']), given) as model:
        return model._model_digest()


def _shuffled(port: dict[str, Any], seed: int) -> dict[str, Any]:
    """*port*'s sources with the rows of every table but a dimension's shuffled."""
    dimensions = sps.check(expanded(port['spec'])).dimensions
    return {
        name: table.sample(fraction=1.0, shuffle=True, seed=seed)
        if isinstance(table, pl.DataFrame) and name not in dimensions
        else table
        for name, table in sources(port['name']).items()
    }


def test_the_same_sources_build_the_same_model(port: dict[str, Any]) -> None:
    """``osemosys_utopia`` built a different model every time: a cost or a
    right-hand side summed over rows in no fixed order differed in its last
    bit, and an archive of it refused an undeclared read as built from other
    data."""
    given = sources(port['name'])
    assert _digest(port, given) == _digest(port, given)


@pytest.mark.parametrize('seed', [0, 1])
def test_shuffled_tables_build_the_same_model(port: dict[str, Any], seed: int) -> None:
    plain = _digest(port, sources(port['name']))
    assert _digest(port, _shuffled(port, seed)) == plain, 'shuffling the rows of a table built another model'


def test_a_sum_is_the_same_to_the_last_bit_whatever_order_its_rows_arrive_in() -> None:
    """Enough rows and groups that polars' own ``sum`` differs from one shuffle to the next."""
    rng = np.random.default_rng(0)
    rows = 200_000
    frame = pl.DataFrame(
        {
            'group': rng.integers(0, 5_000, rows),
            'value': rng.uniform(-1e3, 1e3, rows) * 10.0 ** rng.integers(-6, 6, rows),
        }
    )

    def totals(rows: pl.DataFrame) -> pl.Series:
        return rows.group_by('group').agg(ordered_sum('value')).sort('group').get_column('value')

    plain = totals(frame)
    for seed in range(3):
        assert totals(frame.sample(fraction=1.0, shuffle=True, seed=seed)).equals(plain), (
            'the same values, shuffled, summed to another total in some group'
        )


def _one_sum_over(expression: str, where: str) -> dict[str, Any]:
    """A model with one variable and *expression* in its objective or its one constraint."""
    spec: dict[str, Any] = {
        'dimensions': {'item': {'dtype': 'int'}},
        'parameters': {'weight': {'dims': ['item']}},
        'variables': {'x': {'dims': [], 'bounds': {'lower': 0, 'upper': 1}}},
        'objective': {'sense': 'minimize', 'expression': 'x'},
    }
    if where == 'objective':
        spec['objective']['expression'] = expression
    else:
        spec['constraints'] = {'bound': {'dims': [], 'expression': expression}}
    return spec


@pytest.mark.parametrize(
    ('expression', 'where'),
    [
        pytest.param('x + sum(weight)', 'objective', id='an-objective-constant'),
        pytest.param('x >= sum(weight)', 'constraint', id='a-right-hand-side'),
        pytest.param('x / sum(weight) <= 1', 'constraint', id='a-divisor'),
    ],
)
def test_a_constant_summed_over_many_rows_builds_the_same_model_whatever_their_order(
    expression: str, where: str
) -> None:
    """One sum over enough rows that adding them in another order moves its last bit."""
    rng = np.random.default_rng(0)
    items = 200_000
    weights = pl.DataFrame(
        {'item': range(items), 'value': rng.uniform(0.0, 1e3, items) * 10.0 ** rng.integers(-6, 6, items)}
    )
    spec = _one_sum_over(expression, where)

    def digest(weight: pl.DataFrame) -> str:
        with sps.build(spec, {'item': list(range(items)), 'weight': weight}) as model:
            return model._model_digest()

    assert digest(weights.sample(fraction=1.0, shuffle=True, seed=0)) == digest(weights), (
        'the same weights, shuffled, built another model'
    )


def test_a_sum_of_nothing_is_zero_as_polars_sum_gives() -> None:
    nothing = pl.DataFrame({'value': []}, schema={'value': pl.Float64})
    assert nothing.select(ordered_sum('value')).item() == nothing.select(pl.col('value').sum()).item() == 0.0


def test_a_sum_per_key_is_the_same_to_the_last_bit_however_many_rows_each_key_has() -> None:
    """Keys of one, two and three rows take the plain sum or the ordered one, and either way the total holds."""
    rng = np.random.default_rng(0)
    sizes = [1] * 2_000 + [2] * 2_000 + [3] * 20_000 + [50] * 500
    keys = np.repeat(np.arange(len(sizes)), sizes)
    frame = pl.DataFrame(
        {'key': keys, 'value': rng.uniform(0.0, 1e3, len(keys)) * 10.0 ** rng.integers(-6, 6, len(keys))}
    )

    def totals(rows: pl.DataFrame) -> pl.Series:
        return ordered_sum_per(rows.lazy(), ['key'], 'value').collect().sort('key').get_column('value')

    plain = totals(frame)
    assert plain.len() == len(sizes), 'one total per key'
    for seed in range(3):
        assert totals(frame.sample(fraction=1.0, shuffle=True, seed=seed)).equals(plain), (
            'the same values, shuffled, summed to another total for some key'
        )
