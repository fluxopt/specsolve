#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_storage``: PyPSA's own LOPF. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_storage.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

Ramp limits plus a ``StorageUnit``. Its state of charge carries energy
between snapshots, charged at ``efficiency_store`` and discharged at
``efficiency_dispatch``, decaying by ``standing_loss`` each step. Left
**non-cyclic**: the horizon starts at ``state_of_charge_initial`` and the end
is free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_storage.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``max_hours`` is the ratio PyPSA stores; the port carries the product it
    implies (``soc_max``), because a bound there takes a name, not arithmetic.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    links: pd.DataFrame = tables['link'].set_index('link')
    storages: pd.DataFrame = tables['storage'].set_index('storage')

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
    p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
    n.add(
        'StorageUnit',
        storages.index,
        bus=storages['storage_bus'],
        p_nom=p_nom,
        max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
        state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
        efficiency_store=tables['efficiency_store'].set_index('storage')['value'],
        efficiency_dispatch=tables['efficiency_dispatch'].set_index('storage')['value'],
        standing_loss=tables['standing_loss'].set_index('storage')['value'],
        cyclic_state_of_charge=False,
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
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_prices(n)})}')
    print(n.generators_t.p)
    print(n.storage_units_t.state_of_charge)
    return float(n.objective)


if __name__ == '__main__':
    main()
