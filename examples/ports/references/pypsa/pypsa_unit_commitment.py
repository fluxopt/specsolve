#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_unit_commitment``: PyPSA's own UC. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_unit_commitment.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**The MILP entry in the corpus.** ``committable=True`` gives each generator a
binary ``status`` per snapshot, plus binary ``start_up`` and ``shut_down``. One
bus, no network.

``min_up_time`` and ``min_down_time`` are left at 0; ``pypsa_min_up_down`` ports
them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_unit_commitment.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', 'bus')

    generators: pd.DataFrame = tables['generator'].set_index('generator')

    n.add(
        'Generator',
        generators.index,
        bus='bus',
        committable=True,
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
        start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
        shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
    )

    load: pd.Series = tables['load'].set_index('snapshot')['value']
    n.add('Load', 'load', bus='bus', p_set=load)
    return n


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(n.generators_t.p)
    print(n.generators_t.status)
    return float(n.objective)


if __name__ == '__main__':
    main()
