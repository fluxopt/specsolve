#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_link_delay``: PyPSA's own delayed link. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_link_delay.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**Power entering a link at one snapshot leaves it at another.** ``delay`` on a
``Link`` withdraws from ``bus0`` at snapshot *t* and injects at ``bus1`` at the
snapshot ``delay`` weighting-units later
(``components/_types/mixin/multiports.py:132``). Two links carry the demand
here: ``ship`` takes two snapshots to arrive and loses 10% on the way, ``wire``
arrives at once — the same expression in the port, with the delay read off a
column.

``cyclic_delay`` is **False**, which is not PyPSA's default. PyPSA's own docs
say *energy is lost at the tail and first snapshots receive nothing from
delayed links*, and ``main`` prints both ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_link_delay.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    Both links are given a ``delay`` — 2 for ``ship`` and 0 for ``wire`` — so the
    column is read rather than a constant applied to everything, and neither
    link is extendable: a delay is about *when* energy arrives, and a capacity
    decision would give a mismatch a second thing to be about.
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

    links: pd.DataFrame = tables['link'].set_index('link')
    n.add(
        'Link',
        links.index,
        bus0=links['link_from'],
        bus1=links['link_to'],
        p_nom=tables['link_p_nom'].set_index('link')['value'],
        efficiency=tables['efficiency'].set_index('link')['value'],
        delay=tables['delay'].set_index('link')['value'],
        cyclic_delay=False,
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
    print('\nwhat each link takes in, and what the demand bus is served by:')
    print(pd.concat({'in': n.links_t.p0, 'out': -n.links_t.p1}, axis=1))
    print(n.generators_t.p)
    print(
        '\nthe first two snapshots receive nothing from `ship`, and nothing shipped in the\n'
        'last two arrives at all — which is what `cyclic_delay=False` means.'
    )
    return float(n.objective)


if __name__ == '__main__':
    main()
