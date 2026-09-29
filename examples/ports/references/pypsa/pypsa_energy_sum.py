#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_energy_sum``: PyPSA's own energy-total bounds. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_energy_sum.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A bound on energy, not on power.** ``e_sum_min`` and ``e_sum_max`` reduce a
generator's dispatch across the whole horizon and bound the total — a
contracted delivery, a fuel allowance, a reservoir's season. Every other bound
in the corpus holds within one snapshot.

**The snapshot weightings are not 1.** They are the hours each snapshot stands
for, and they enter twice: once in the energy being bounded, once in the cost
being minimised.

Only two of the three generators carry a bound: the attributes default to
``-inf`` and ``inf``, and PyPSA emits the constraint only where the value is
finite, so the port's tables are short rather than padded with infinities.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_energy_sum.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    The energy bounds arrive as short frames — one row per generator that has
    one — and are reindexed onto the full generator index, which is where the
    infinities PyPSA tests for come from. All three weighting columns are set
    together: ``generators`` scales the energy the bounds see, ``objective``
    scales the cost, and leaving them to disagree would make the number an
    accident of a default.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.snapshot_weightings.loc[:, :] = tables['weighting'].set_index('snapshot')['value'].to_numpy()[:, None]
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    e_sum_max = tables['e_sum_max'].set_index('generator')['value'].reindex(generators.index, fill_value=float('inf'))
    e_sum_min = tables['e_sum_min'].set_index('generator')['value'].reindex(generators.index, fill_value=float('-inf'))
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        e_sum_max=e_sum_max,
        e_sum_min=e_sum_min,
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per (snapshot, bus), tidy — read off the
    model rather than off ``buses_t.marginal_price``.

    PyPSA divides the dual by ``snapshot_weightings.objective`` before
    publishing it as a marginal price, so that the figure reads per unit energy
    rather than per snapshot. This instance weights its snapshots 1, 2, 3, 2,
    and the published price is a flat 60 against a dual of 60, 120, 180, 120.
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
    print(n.generators_t.p)
    print((n.generators_t.p.mul(n.snapshot_weightings.generators, axis=0)).sum())
    return float(n.objective)


if __name__ == '__main__':
    main()
