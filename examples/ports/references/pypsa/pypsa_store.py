#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_store``: PyPSA's own ``Store`` component. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_store.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A ``Store`` is not a ``StorageUnit``.** The storage port takes the latter: a
dispatch/store pair of non-negative variables so the two efficiencies can
differ, and a power rating of its own. A ``Store`` has **one signed power**
positive when it supplies the bus, no efficiencies at all, and no power rating —
the only limit on how fast it moves energy is the energy level itself. It is
what every sector-coupled PyPSA model uses for hydrogen, heat and gas.

The energy capacity is extendable here, so ``e_nom`` is a decision rather than a
bound, and the standing loss is 0.05 per snapshot: the tank is charged early and
drawn down late, so the decay shows in the objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_store.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``Store`` takes no power rating: ``e_nom`` bounds the level, and the power
    that moves it is limited only by what the level allows within one snapshot.
    That is why the port declares its store power with no bounds at all.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )

    stores: pd.DataFrame = tables['store'].set_index('store')
    n.add(
        'Store',
        stores.index,
        bus=stores['store_bus'],
        e_nom_extendable=True,
        e_nom_max=tables['e_nom_max'].set_index('store')['value'],
        capital_cost=tables['e_capital_cost'].set_index('store')['value'],
        e_initial=tables['e_initial'].set_index('store')['value'],
        standing_loss=tables['standing_loss'].set_index('store')['value'],
        e_cyclic=False,
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per (snapshot, bus), tidy.

    Read off the model: ``buses_t.marginal_price`` divides it by the snapshot
    weighting.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'snapshot': [int(s) for s, _ in dual.index],
        'bus': [str(b) for _, b in dual.index],
        'value': [float(v) for v in dual.to_numpy()],
    }


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_duals(n)})}')
    print(n.stores[['e_nom_opt']])
    print(n.stores_t.e)
    print(n.stores_t.p)
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
