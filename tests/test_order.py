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
from tests.conftest import expanded
from tests.conftest import port_sources as sources

if TYPE_CHECKING:
    from collections.abc import Mapping

#: Ports whose model moves from one build to the next, and why.
MOVES = {
    'osemosys_utopia': (
        'a cost or a right-hand side summed over rows in no fixed order differs in its last bit, '
        'so an archive refuses an undeclared read as built from other data'
    ),
}


@pytest.fixture
def port_that_holds(port: dict[str, Any], request: pytest.FixtureRequest) -> dict[str, Any]:
    """*port*, expected to fail where ``MOVES`` says its model moves."""
    if reason := MOVES.get(port['name']):
        request.applymarker(pytest.mark.xfail(reason=reason))
    return port


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


def test_the_same_sources_build_the_same_model(port_that_holds: dict[str, Any]) -> None:
    given = sources(port_that_holds['name'])
    assert _digest(port_that_holds, given) == _digest(port_that_holds, given)


@pytest.mark.parametrize('seed', [0, 1])
def test_shuffled_tables_build_the_same_model(port_that_holds: dict[str, Any], seed: int) -> None:
    plain = _digest(port_that_holds, sources(port_that_holds['name']))
    assert _digest(port_that_holds, _shuffled(port_that_holds, seed)) == plain, (
        'shuffling the rows of a table built another model'
    )


def test_the_same_numbers_in_another_row_order_keep_the_loaded_solver() -> None:
    """A coefficient summed over rows in another order differs in its last bit, which is no reason to load again.

    The solver was kept only while every coefficient matched to the last bit,
    so the same numbers in another row order loaded it again at every update
    and lost the warm start.
    """
    rng = np.random.default_rng(0)
    items, periods = 2_000, 50
    spec = {
        'dimensions': {'i': {'dtype': 'int'}, 't': {'dtype': 'int'}},
        'parameters': {'cost': {'dims': ['i', 't']}},
        'variables': {'x': {'dims': ['i'], 'bounds': {'lower': 0, 'upper': 1}}},
        'constraints': {'budget': {'dims': [], 'expression': 'sum(cost * x) <= 1e6'}},
        'objective': {'sense': 'maximize', 'expression': 'sum(x)'},
    }
    cost = pl.DataFrame(
        {
            'i': np.repeat(np.arange(items), periods),
            't': np.tile(np.arange(periods), items),
            'value': rng.uniform(0.0, 1e3, items * periods) * 10.0 ** rng.integers(-6, 6, items * periods),
        }
    )
    with sps.build(spec, {'i': list(range(items)), 't': list(range(periods)), 'cost': cost}) as model:
        model.solve()
        for seed in range(3):
            model.update({'cost': cost.sample(fraction=1.0, shuffle=True, seed=seed)}).solve()
        assert model.diagnostics().loads == 1, 'the same numbers, shuffled, loaded the solver again'
