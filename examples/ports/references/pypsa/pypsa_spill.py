#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_spill``: PyPSA's own storage spillage. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_spill.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**Water a reservoir cannot hold leaves through a second sink.** ``inflow`` adds
energy the model did not choose to store, and ``spill`` is the non-negative
variable that lets the balance close when the reservoir is full — bounded above
by the inflow itself, and existing only where there is inflow to spill.

The reservoir is the cheap unit and gas is dear, so nothing spills for want of
somewhere to sell it: the first two snapshots simply deliver more water than
30 MW of turbine and 60 MWh of reservoir can absorb.

Efficiencies are 1 and the standing loss is 0; the storage port covers the
round-trip terms.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_spill.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``inflow`` is a time-varying attribute, so it arrives pivoted to snapshots
    by names. PyPSA declares the spill variable only for units whose inflow is
    positive somewhere; the port declares it for every unit and bounds it above
    by the inflow, which pins it to zero for the battery. Same model, and the
    port's spelling is the one that keeps the energy balance a single block.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    storages: pd.DataFrame = tables['storage'].set_index('storage')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )

    p_nom: pd.Series = tables['storage_p_nom'].set_index('storage')['value']
    n.add(
        'StorageUnit',
        storages.index,
        bus=storages['storage_bus'],
        p_nom=p_nom,
        max_hours=tables['soc_max'].set_index('storage')['value'] / p_nom,
        state_of_charge_initial=tables['soc_initial'].set_index('storage')['value'],
        inflow=tables['inflow']
        .pivot(index='snapshot', columns='storage', values='value')
        .reindex(columns=storages.index)
        .fillna(0.0),
        cyclic_state_of_charge=False,
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
    print(n.storage_units_t.spill)
    print(n.storage_units_t.state_of_charge)
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
