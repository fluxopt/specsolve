#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_min_up_down``: PyPSA's own minimum up and down times. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_min_up_down.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A unit that has started must stay on.** Beyond the status and the two
transition variables of ``pypsa_unit_commitment``, the windows make commitment
bite: over any ``min_up_time`` consecutive
snapshots a unit may have started at most as often as it is now running, and the
mirror holds for stopping. Each is a backward-looking sum whose length is a
property of the generator, not of the model.

The three units carry different window lengths — 3, 2 and 1.

``up_time_before`` and ``down_time_before`` are both set to 0, against PyPSA's
default of 1 and 0. The default says the unit was already running before the
horizon, which emits a further block pinning the status on for the remainder of
its minimum up time. With both at 0 that block does not appear, every unit begins
the horizon **off**, and the first snapshot's transition rows are the mirror of
the ones ``pypsa_unit_commitment`` ports: a unit committed in the first snapshot
pays for a start and nothing is charged for a stop, where a unit that began the
horizon running pays for no start and is charged if it goes down.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_min_up_down.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    One bus and no network: a model that fails to match should implicate one
    feature, and here it is the window length. ``committable`` is what turns the
    status into a variable at all, and the two ``*_time_before`` values are set
    rather than defaulted for the reason in the module docstring.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', 'hub')

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        committable=True,
        up_time_before=0,
        down_time_before=0,
        p_nom=tables['p_nom'].set_index('generator')['value'],
        p_min_pu=tables['p_min_pu'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        start_up_cost=tables['start_up_cost'].set_index('generator')['value'],
        shut_down_cost=tables['shut_down_cost'].set_index('generator')['value'],
        min_up_time=tables['min_up_time'].set_index('generator')['value'],
        min_down_time=tables['min_down_time'].set_index('generator')['value'],
    )

    n.add('Load', 'l', bus='hub', p_set=tables['load'].set_index('snapshot')['value'])
    return n


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(n.generators_t.status)
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
