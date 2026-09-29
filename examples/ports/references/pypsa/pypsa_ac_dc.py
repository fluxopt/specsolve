#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_ac_dc``: PyPSA's own meshed AC-DC example. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_ac_dc.py

Pinned to the versions that produced the number in ``references.json`` and run
out of band — PyPSA is not a dependency of this project. It reads the same
instance the port attaches and builds the network with PyPSA's own objects.
Nothing here imports specsolve.

**Two coordinates on one dimension.** A generator sits on a bus and also burns a *carrier*,
and the CO2 budget is priced through that second map — PyPSA's
``primary_energy`` global constraint, which charges output over efficiency at
the carrier's rate. The port states the two maps as two coordinates on
``generator``; PyPSA states them as two columns on the generator table and
resolves the second through ``n.carriers``.

**What the recorded number is.** ``n.objective`` on this network is *negative*:
PyPSA credits the capital already standing in ``p_nom``, so its objective is
the change against that starting point. The system cost the port computes —
capital on the chosen capacities plus marginal on the dispatch — is
``n.objective + n.objective_constant``, and that is what is recorded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_ac_dc.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def _wide(frame: pd.DataFrame, columns: str) -> pd.DataFrame:
    """A tidy (snapshot, entity, value) table as the wide frame PyPSA wants."""
    return frame.pivot(index='snapshot', columns=columns, values='value')


def _series(frame: pd.DataFrame, index: str) -> pd.Series:
    return frame.set_index(index)['value']


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    The port carries its cycle basis as ``cycle_incidence`` because computing
    one is a graph algorithm and so data preparation; PyPSA derives its own
    from ``line_x`` / ``line_r``, which is why both are in the instance. The
    two must describe the same cycle space, and the objectives agreeing is
    what says they do.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'].tolist())

    carrier = _series(tables['bus_carrier'], 'bus')
    for bus, ct in carrier.items():
        n.add('Bus', bus, carrier=ct)

    co2 = _series(tables['co2_per_mwh'], 'carrier')
    for name, rate in co2.items():
        n.add('Carrier', name, co2_emissions=rate)

    generators = tables['generator'].set_index('generator')
    p_max_pu = _wide(tables['p_max_pu'], 'generator')
    for g, row in generators.iterrows():
        n.add(
            'Generator',
            g,
            bus=row['bus'],
            carrier=row['carrier'],
            p_nom_extendable=True,
            p_nom=_series(tables['gen_p_nom_existing'], 'generator')[g],
            p_nom_min=_series(tables['p_nom_min'], 'generator')[g],
            marginal_cost=_series(tables['marginal_cost'], 'generator')[g],
            capital_cost=_series(tables['gen_capital_cost'], 'generator')[g],
            efficiency=_series(tables['efficiency'], 'generator')[g],
            p_max_pu=p_max_pu[g],
        )

    for line, ends in tables['line'].set_index('line').iterrows():
        bus0, bus1 = ends['line_from'], ends['line_to']
        n.add(
            'Line',
            line,
            bus0=bus0,
            bus1=bus1,
            x=_series(tables['line_x'], 'line')[line],
            r=_series(tables['line_r'], 'line')[line],
            s_nom=_series(tables['line_s_nom_existing'], 'line')[line],
            s_nom_extendable=True,
            capital_cost=_series(tables['line_capital_cost'], 'line')[line],
        )

    for link, ends in tables['link'].set_index('link').iterrows():
        bus0, bus1 = ends['link_from'], ends['link_to']
        n.add(
            'Link',
            link,
            bus0=bus0,
            bus1=bus1,
            p_nom_extendable=True,
            p_nom=_series(tables['link_p_nom_existing'], 'link')[link],
            p_min_pu=_series(tables['link_p_min_pu'], 'link')[link],
            p_max_pu=_series(tables['link_p_max_pu'], 'link')[link],
            capital_cost=_series(tables['link_capital_cost'], 'link')[link],
        )

    load = _wide(tables['load'], 'bus')
    for bus in load.columns:
        if load[bus].any():
            n.add('Load', bus, bus=bus, p_set=load[bus])

    n.add(
        'GlobalConstraint',
        'co2_limit',
        type='primary_energy',
        carrier_attribute='co2_emissions',
        sense='<=',
        constant=float(tables['co2_limit']),
    )
    return n


def nodal_prices(n: pypsa.Network) -> dict[str, list[float]]:
    """The balance duals, in the port's (snapshot, bus) order."""
    prices = n.buses_t.marginal_price
    return {
        'snapshot': [int(s) for s in prices.index for _ in prices.columns],
        'bus': [b for _ in prices.index for b in prices.columns],
        'value': [float(prices.loc[s, b]) for s in prices.index for b in prices.columns],
    }


def main() -> float:
    """Solve and print what ``references.json`` records: ``n.objective + n.objective_constant``."""
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    system_cost = float(n.objective) + float(n.objective_constant)
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r} + constant {float(n.objective_constant)!r}')
    print(f'system cost {system_cost!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_prices(n)})}')
    print(n.generators.p_nom_opt)
    return system_cost


if __name__ == '__main__':
    main()
