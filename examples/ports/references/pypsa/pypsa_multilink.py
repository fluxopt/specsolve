#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_multilink``: PyPSA's own multi-link. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_multilink.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches. The port holds the link-to-bus
relation as one incidence table — a ``(link, bus, value)`` row per link end,
``-1`` at the input, ``+efficiency`` at each output — and PyPSA holds it wide:
``bus0`` is the input, ``bus1``/``bus2`` the outputs,
``efficiency``/``efficiency2`` their deratings, an empty ``bus2`` where a link
has only two ends. ``build`` opens with that pivot. Nothing here imports
specsolve.

A multi-link is the one PyPSA construct whose schema grows with the data —
every arity adds a column pair — and the port says the same relation as rows.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_multilink.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``; only
    the incidence table changes shape on the way in, pivoted from one row per
    link end into PyPSA's one row per link. The input end is the one with the
    negative value — PyPSA fixes its share at -1, so the pivot asserts it: a
    different input share is sayable in rows and not in these columns. Each
    output end becomes the link's next port, its value the port's efficiency,
    in the incidence table's row order. A link narrower than the instance's
    widest is padded with ``''`` — PyPSA's spelling for a port a link does not
    have — and a filler efficiency of 1.0 that no equation reads.
    """
    incidence = tables['incidence']
    inputs = incidence[incidence['value'] < 0].set_index('link')
    assert (inputs['value'] == -1.0).all(), 'PyPSA fixes the input share of a Link at -1; this instance must too'

    outputs = incidence[incidence['value'] > 0].copy()
    outputs['port'] = outputs.groupby('link', sort=False).cumcount() + 1
    buses = outputs.pivot(index='link', columns='port', values='bus')
    efficiencies = outputs.pivot(index='link', columns='port', values='value')

    links = pd.DataFrame(index=pd.Index(tables['link']['link'], name='link'))
    links['bus0'] = inputs['bus']
    for port in buses.columns:
        links[f'bus{port}'] = buses[port].reindex(links.index).fillna('')
        links['efficiency' if port == 1 else f'efficiency{port}'] = efficiencies[port].reindex(links.index).fillna(1.0)

    n = pypsa.Network()
    n.add('Bus', tables['bus']['bus'])

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        p_nom=tables['gen_p_nom'].set_index('generator')['value'],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
    )
    n.add('Link', links.index, p_nom=tables['p_nom'].set_index('link')['value'], **dict(links.items()))

    load: pd.Series = tables['load'].set_index('bus')['value']
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])
    return n


def nodal_prices(n: pypsa.Network) -> dict[str, list]:
    """PyPSA's marginal price per bus, tidy — the dual of the nodal balance.

    The port has no snapshot dimension, so the network's one default snapshot
    is dropped from the key.
    """
    mp = n.buses_t.marginal_price.iloc[0]
    return {'bus': [str(b) for b in mp.index], 'value': [float(v) for v in mp]}


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_prices(n)})}')
    print(n.links_t.p0)
    print(n.generators_t.p)
    return float(n.objective)


if __name__ == '__main__':
    main()
