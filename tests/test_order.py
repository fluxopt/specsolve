"""Row order carries no meaning: the same data builds the same model.

A source table's rows arrive in whatever order its writer chose, and polars
promises no row order from a join or a group-by that was not asked for one. So
each referenced model is built twice from the same sources, and again with the
rows of every parameter and relation table shuffled, and each build must
be the same: matrix, bounds, costs and right-hand sides, bit for bit. One
that moves solves the same data to an answer that differs in its last bits.

Dimension tables keep their order: a dimension's row order is its coordinate
order, which ``shift`` reads.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        'a cost or a right-hand side summed over rows in no fixed order differs in its last bit (#1896)'
    ),
}


@pytest.fixture
def port_that_holds(port: dict[str, Any], request: pytest.FixtureRequest) -> dict[str, Any]:
    """*port*, expected to fail where ``MOVES`` says its model moves."""
    if reason := MOVES.get(port['name']):
        request.applymarker(pytest.mark.xfail(reason=reason))
    return port


def _built(port: dict[str, Any], given: Mapping[str, Any]) -> tuple[bytes, ...]:
    """The built model to the last bit: what a re-solve may not change, then every number it may.

    The objective is read through the dense cost vector, since ``obj`` carries
    no order contract.
    """
    with sps.build(expanded(port['spec']), given) as model:
        handoff = model._engine._model.handoff
        numbers = (
            handoff.cols['lb'],
            handoff.cols['ub'],
            handoff.quad['coeff'],
            handoff.rows['row'],
            handoff.rows['rhs'],
        )
        return (
            handoff.structure,
            f'{handoff.objective_constant}'.encode(),
            handoff._dense_cost().tobytes(),
            *(column.to_numpy().tobytes() for column in numbers),
        )


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
    assert _built(port_that_holds, given) == _built(port_that_holds, given)


@pytest.mark.parametrize('seed', [0, 1])
def test_shuffled_tables_build_the_same_model(port_that_holds: dict[str, Any], seed: int) -> None:
    plain = _built(port_that_holds, sources(port_that_holds['name']))
    assert _built(port_that_holds, _shuffled(port_that_holds, seed)) == plain, (
        'shuffling the rows of a table built another model'
    )
