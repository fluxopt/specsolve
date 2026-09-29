#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_ramp``: PyPSA's own LOPF. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_ramp.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

The transport model plus generator ramp limits. ``ramp_limit_up`` and
``ramp_limit_down`` are fractions of ``p_nom`` bounding the change between
consecutive snapshots, and PyPSA writes them from the *second* snapshot on —
there is no initial dispatch for the first to ramp from. The network, the loads
and the links are the transport model's.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_ramp.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    links: pd.DataFrame = tables['link'].set_index('link')

    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        ramp_limit_up=tables['ramp_limit_up'].set_index('generator')['value'],
        ramp_limit_down=tables['ramp_limit_down'].set_index('generator')['value'],
    )
    n.add(
        'Link',
        links.index,
        bus0=links['link_from'],
        bus1=links['link_to'],
        p_nom=tables['rating'].set_index('link')['value'],
        p_min_pu=-1.0,
        efficiency=1.0,
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_prices(n: pypsa.Network) -> dict[str, list]:
    """PyPSA's marginal price per (snapshot, bus), tidy — the dual of the nodal balance."""
    mp = n.buses_t.marginal_price
    return {
        'snapshot': [s for s in mp.index for _ in mp.columns],
        'bus': [b for _ in mp.index for b in mp.columns],
        'value': [float(v) for row in mp.to_numpy() for v in row],
    }


def main() -> float:
    """Solve, and print what ``references.json`` records; ramp limits can make the instance infeasible."""
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition} — the ramp limits are tighter than the load swing'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_prices(n)})}')
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
