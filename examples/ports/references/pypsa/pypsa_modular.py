#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_modular``: PyPSA's own modular capacity expansion. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_modular.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**Capacity that comes in whole modules.** ``p_nom_mod`` on an extendable
generator makes PyPSA emit an integer ``Generator-n_mod`` and the equality
``p_nom - n_mod * p_nom_mod == 0``, so the capacity variable survives but may
only land on a multiple of the module size.

One bus and no network. The three module sizes do not divide the peak load, so
an optimum that rounded down would be infeasible and one that ignored the module
sizes would be cheaper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_modular.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``p_nom_extendable`` and a positive ``p_nom_mod`` together are what make the
    capacity modular: PyPSA takes the module count only where a component is in
    both index sets.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom_extendable=True,
        p_nom_mod=tables['p_nom_mod'].set_index('generator')['value'],
        p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
        capital_cost=tables['capital_cost'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'objective_constant {float(n.objective_constant)!r}')
    print(n.generators[['p_nom_opt', 'p_nom_mod']])
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
