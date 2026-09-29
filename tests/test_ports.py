"""Referenced models, checked against an optimum that did not come from specsolve.

The differential harness compares two lanes consuming the same resolved AST, so
a shared misreading passes it; this corpus is the net for that class. Each
expected objective was published with the model, produced by somebody else's
code, or reached by a hand-written formulation on another modelling stack.
``examples/ports/references/`` holds the scripts, run out of band, and
``references.json`` records what they said, so the corpus is linopy-free and
runs on the bare-install job.
"""

from __future__ import annotations

from typing import Any

import polars as pl
import pytest
import yaml

import specsolve as sps
from tests.conftest import expanded, port_spec
from tests.conftest import port_sources as sources


def test_port_reaches_the_reference_optimum(port: dict[str, Any]) -> None:
    """The objective, never the primal — ``transport_dantzig`` reaches 153.675
    at a different vertex than the source prints, so a corpus pinned to a
    solution would fail on a solver upgrade that broke nothing. ``rtol`` is per
    port because a published optimum is rounded and a solved one is not."""
    with sps.solve(expanded(port['spec']), sources(port['name'])) as solution:
        assert solution.is_ok, f'{port["name"]} did not solve: {solution.status}'
        assert solution.objective == pytest.approx(port['objective'], rel=port['rtol']), (
            f'{port["name"]} disagrees with {port["provenance"]}'
        )


def test_port_is_inside_the_language(port: dict[str, Any]) -> None:
    """Compiles with no data attached, so a language regression fails separately
    from a semantics one: this breaks when lowering stops accepting the model,
    the test above when it lowers and misses the number."""
    sps.check(expanded(port['spec']))


def test_port_reaches_the_reference_duals(port: dict[str, Any]) -> None:
    """The shadow prices, against the same outside implementation.

    A dual vector is where two implementations disagree quietly: which side of
    the constraint the price belongs to, and what sign an inequality's carries.
    Both of ``transport_dantzig``'s constraints are inequalities pointing
    opposite ways.

    Ports with no ``duals`` block are skipped rather than passing vacuously:
    ``pypsa_unit_commitment`` is a MILP, where a dual solution is undefined and
    specsolve refuses to invent one.
    """
    expected = port.get('duals')
    if not expected:
        pytest.skip(f'{port["name"]} records no duals (a MILP has none)')

    with sps.solve(expanded(port['spec']), sources(port['name'])) as solution:
        for constraint, table in expected.items():
            dims = [c for c in table if c != 'value']
            got = solution.dual(constraint).sort(dims)
            want = pl.DataFrame(table).with_columns(pl.col(d).cast(got.schema[d]) for d in dims).sort(dims)

            assert got[dims].equals(want[dims]), f'{port["name"]}.{constraint}: dual is keyed differently'
            assert got['value'].to_list() == pytest.approx(want['value'].to_list(), rel=port['rtol']), (
                f'{port["name"]}.{constraint} disagrees with {port["provenance"]}'
            )


#: A rule a port claims, and the wrong row it is claimed against: model file,
#: constraint, the expression as shipped, and the misreading.
MISREADINGS = [
    pytest.param(
        'pypsa_store',
        'energy_balance_initial',
        'e == e_initial - store_p',
        'e == e_initial * (1 - standing_loss) - store_p',
        id='pypsa_store-the-initial-level-is-not-decayed',
    ),
    pytest.param(
        'pypsa_cvar',
        'tail_definition',
        '(1 - alpha) * (tail_average - tail_start) >= sum(probability * excess, over=scenario)',
        '(tail_average - tail_start) >= sum(probability * excess, over=scenario)',
        id='pypsa_cvar-the-tail-average-is-scaled-by-the-tail-probability',
    ),
]


@pytest.mark.parametrize(('name', 'constraint', 'shipped', 'misread'), MISREADINGS)
def test_the_instance_can_tell_the_rule_from_its_misreading(
    name: str, constraint: str, shipped: str, misread: str
) -> None:
    """The recorded optimum only guards a rule the *instance* is sensitive to.

    Solve the ported model, then solve it again with one constraint replaced by
    the misreading, and demand the two disagree. On ``pypsa_store`` with an
    ``e_initial`` of 0, both readings of the row reach 3116.36.
    """
    spec = yaml.safe_load(port_spec(name).read_text())
    assert spec['constraints'][constraint]['expression'] == shipped, (
        f'{name}.{constraint} no longer reads `{shipped}` — this probe is pinned to a row that moved'
    )
    spec['constraints'][constraint]['expression'] = misread

    with sps.solve(expanded(port_spec(name)), sources(name)) as solution:
        as_shipped = solution.objective
    with sps.solve(expanded(spec), sources(name)) as solution:
        misreading = solution.objective

    assert misreading != pytest.approx(as_shipped, rel=1e-09), (
        f'{name} reaches {as_shipped!r} whether {constraint} reads `{shipped}` or `{misread}` — '
        f'the instance cannot tell them apart, so its recorded optimum is not evidence for the rule'
    )
