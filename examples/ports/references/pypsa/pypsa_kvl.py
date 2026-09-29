#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_kvl``: PyPSA's own LOPF with lines. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_kvl.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**Kirchhoff's voltage law.** A ``Link``'s flow is a decision variable — a
transport model. A ``Line`` is passive: flow is decided by physics, and around
every independent cycle the reactance-weighted flows must sum to zero.

It also prints the cycle basis PyPSA derived, because the port carries that
basis as data (``cycle_incidence``) and the two must describe the same cycle
space. Computing a cycle basis is data preparation, outside the language.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_kvl.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``r=0`` keeps a line purely reactive: the linearised power flow is a
    function of ``x`` alone, and a resistance would only add losses the DC
    approximation does not model anyway.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    lines: pd.DataFrame = tables['line'].set_index('line')

    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )
    n.add(
        'Line',
        lines.index,
        bus0=lines['line_from'],
        bus1=lines['line_to'],
        x=tables['reactance'].set_index('line')['value'],
        r=0.0,
        s_nom=tables['s_nom'].set_index('line')['value'],
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_prices(n: pypsa.Network) -> dict[str, list]:
    """PyPSA's marginal price per (snapshot, bus), tidy."""
    mp = n.buses_t.marginal_price
    return {
        'snapshot': [s for s in mp.index for _ in mp.columns],
        'bus': [b for _ in mp.index for b in mp.columns],
        'value': [float(v) for row in mp.to_numpy() for v in row],
    }


def cycle_basis(n: pypsa.Network) -> str:
    """The KVL rows PyPSA built, so the port's incidence can be checked by eye.

    PyPSA scales the coefficients for conditioning; the constraint is ``= 0``,
    so any nonzero multiple of a cycle describes the same cycle space. What has
    to match is which lines share a row and with what relative signs.
    """
    return str(n.model.constraints['Kirchhoff-Voltage-Law'])


def main() -> float:
    n = build(load_tables())
    n.optimize.create_model(include_objective_constant=False)
    print(cycle_basis(n))
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_prices(n)})}')
    print(n.lines_t.p0)
    return float(n.objective)


if __name__ == '__main__':
    main()
