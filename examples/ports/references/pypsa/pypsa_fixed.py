#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_fixed``: PyPSA's own dispatch and capacity fixing. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_fixed.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A row of data that is present pins its variable; one that is absent leaves it
free.** ``p_set`` fixes a dispatch, ``p_nom_set`` fixes a capacity, and PyPSA
emits each equality only where the value is not NaN — the must-run generator,
the pre-committed schedule, the capacity somebody already signed for.

Both partial tables are at different ranks: ``p_set`` is sparse over
*(snapshot, generator)* and ``p_nom_set`` over *(generator)* alone.

``chp`` is the dearest unit in the fleet and still runs in the two snapshots it
is pinned in, so the fixing shows in the objective.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_fixed.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    Both fixings arrive as short frames and are widened to the shape PyPSA
    reads, with ``NaN`` where the port simply has no row: ``p_nom_set``
    reindexed onto the generator index, ``p_set`` pivoted to snapshots by names.
    NaN is PyPSA's own spelling for *not fixed here* — it is what its mask
    tests — so the widening is the translation, not a defaulting choice.

    Every generator is extendable, because ``p_nom_set`` fixes the capacity
    *variable*: a non-extendable component has none for the equality to bind.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    p_set = (
        tables['p_set']
        .pivot(index='snapshot', columns='generator', values='value')
        .reindex(index=n.snapshots, columns=generators.index)
    )
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom_extendable=True,
        p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
        p_nom_set=tables['p_nom_set'].set_index('generator')['value'].reindex(generators.index, fill_value=np.nan),
        capital_cost=tables['capital_cost'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        p_set=p_set,
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
    print(n.generators[['p_nom_opt']])
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
