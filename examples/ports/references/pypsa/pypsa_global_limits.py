#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_global_limits``: PyPSA's own global constraints. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_global_limits.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**Four limits, one shape.** Each bounds a sum over a set PyPSA selects by an
attribute:

- ``operational_limit`` — energy a carrier delivers over the horizon (``gas``)
- the per-(bus, carrier) capacity cap — wind capacity at ``east``
- ``transmission_volume_expansion_limit`` — ``p_nom * length`` over the links
- ``transmission_expansion_cost_limit`` — ``capital_cost * p_nom`` over the links

The two link limits carry **different weights over the same set**, and the
weights disagree: ``north_south`` is the long cheap link, ``east_south`` the
short dear one, so a volume limit and a cost limit pull the build in opposite
directions.

``main`` drops each limit in turn to show that it binds, and prints what PyPSA
does with the fifth: see :func:`what_pypsa_drops`.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_global_limits.json'

#: The ``GlobalConstraint`` rows, as ``name -> attributes``. Their constants are
#: the numbers the port's cap parameters carry.
LIMITS: dict[str, dict[str, object]] = {
    'gas_energy': {'type': 'operational_limit', 'carrier_attribute': 'gas', 'sense': '<=', 'constant': 380.0},
    'link_volume': {
        'type': 'transmission_volume_expansion_limit',
        'carrier_attribute': 'DC',
        'sense': '<=',
        'constant': 3400.0,
    },
    'link_cost': {
        'type': 'transmission_expansion_cost_limit',
        'carrier_attribute': 'DC',
        'sense': '<=',
        'constant': 10500.0,
    },
}

#: The per-(bus, carrier) capacity cap, which PyPSA takes as a *column name* on
#: its bus table rather than as a ``GlobalConstraint`` row: ``bus``, the column,
#: and the cap. Deprecated in PyPSA 1.0 — and the only spelling of that limit
#: 1.2.4 emits at all, for the reason :func:`what_pypsa_drops` measures.
BUS_CAPACITY_CAP = ('east', 'nom_max_wind', 17.0)


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(
    tables: dict[str, pd.DataFrame],
    limits: dict[str, dict[str, object]] | None = None,
    bus_capacity_cap: bool = True,
) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``limits`` defaults to all three global-constraint rows and
    ``bus_capacity_cap`` to on; dropping one is how ``main`` measures what it is
    worth. Every generator and every link is extendable, because a limit on
    capacity has nothing to bind on a component whose capacity is data.
    """
    n = pypsa.Network()
    n.set_snapshots(tables['snapshot']['snapshot'])
    n.add('Bus', tables['bus']['bus'])
    n.add('Carrier', tables['carrier']['carrier'])
    n.add('Carrier', 'DC')

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    p_max_pu: pd.DataFrame = tables['p_max_pu'].pivot(index='snapshot', columns='generator', values='value')
    n.add(
        'Generator',
        generators.index,
        bus=generators['gen_bus'],
        carrier=generators['gen_carrier'],
        p_nom_extendable=True,
        p_max_pu=p_max_pu[generators.index],
        marginal_cost=tables['marginal_cost'].set_index('generator')['value'],
        capital_cost=tables['gen_capital_cost'].set_index('generator')['value'],
    )

    links: pd.DataFrame = tables['link'].set_index('link')
    n.add(
        'Link',
        links.index,
        bus0=links['link_from'],
        bus1=links['link_to'],
        carrier='DC',
        p_nom_extendable=True,
        length=tables['link_length'].set_index('link')['value'],
        capital_cost=tables['link_capital_cost'].set_index('link')['value'],
    )

    load: pd.DataFrame = tables['load'].pivot(index='snapshot', columns='bus', values='value')
    for bus in tables['bus']['bus']:
        n.add('Load', f'load_{bus}', bus=bus, p_set=load[bus])

    for name, attributes in (LIMITS if limits is None else limits).items():
        n.add('GlobalConstraint', name, **attributes)

    if bus_capacity_cap:
        bus, column, cap = BUS_CAPACITY_CAP
        n.buses.loc[bus, column] = cap
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


def what_pypsa_drops() -> None:
    """Why the network-wide carrier capacity limit is not one of the four.

    ``tech_capacity_expansion_limit`` is the fifth global limit and the port
    does not carry it, because in pypsa 1.2.4 a single-period network cannot get
    one built. ``global_constraints.py:48`` groups the rows by
    ``["carrier_attribute", "sense", "investment_period"]``; ``investment_period``
    is ``NaN`` where none is given, pandas drops NaN keys, and the row leaves no
    constraint behind — while naming a period raises, there being no investment
    periods to name.
    """
    for label, extra in (('no investment_period', {}), ('investment_period=0', {'investment_period': 0})):
        row = {
            'type': 'tech_capacity_expansion_limit',
            'carrier_attribute': 'wind',
            'sense': '<=',
            'constant': 60.0,
            **extra,
        }
        n = build(load_tables(), {'wind_capacity': row}, bus_capacity_cap=False)
        try:
            n.optimize.create_model()
        except ValueError as error:
            print(f'  {label:22} raises ValueError: {error}')
            continue
        emitted = [name for name in n.model.constraints if name.startswith('GlobalConstraint')]
        print(f'  {label:22} emits {emitted or "no constraint at all"}')


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"nodal_balance": nodal_duals(n)})}')
    print(n.generators[['p_nom_opt']])
    print(n.links[['p_nom_opt', 'length', 'capital_cost']])

    print('\nevery limit binds, and this is what each is worth:')
    for name in (*LIMITS, 'bus_capacity_cap'):
        if name == 'bus_capacity_cap':
            dual = float(n.model.constraints['Bus-nom_max_wind'].dual.values[0])
            without = build(load_tables(), bus_capacity_cap=False)
        else:
            dual = float(n.model.constraints[f'GlobalConstraint-{name}'].dual.values)
            without = build(load_tables(), {k: v for k, v in LIMITS.items() if k != name})
        without.optimize(solver_name='highs')
        print(f'  {name:18} dual {dual:12.4f}   dropped -> {float(without.objective)!r}')

    print('\nthe fifth limit, which pypsa 1.2.4 does not build:')
    what_pypsa_drops()
    return float(n.objective)


if __name__ == '__main__':
    main()
