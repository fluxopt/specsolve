#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_growth_limit``: PyPSA's own carrier growth limit. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_growth_limit.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A limit that couples one period to the one before it.** ``Carrier.max_growth``
caps the capacity of a carrier *first becoming active* in a period, and
``max_relative_growth`` lets that cap grow with what was built last time
(``global_constraints.py:184``):

    new[period] - max_relative_growth * new[period - 1] <= max_growth

The quantity on both sides is **newly built** capacity, not standing capacity — ``vars.where(first_active)``
counts an asset in the period it first exists and never again. And the first
period has no predecessor, so its row is the absolute cap alone.

The three wind units are one per period, which is how a build year becomes a
column here: each is extendable, each first active in its own period, so
``new[period]`` is that unit's capacity. ``gas`` carries the same instance's
fallback and is not limited.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_growth_limit.json'

#: The lifetime every generator gets, long enough that nothing retires inside the horizon.
LIFETIME = 60


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame], growth_limit: bool = True) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    ``growth_limit=False`` drops the two carrier attributes, which is how
    ``main`` measures what the limit is worth. The port's ``build_period`` relation
    is PyPSA's ``build_year``; its ``activity`` table is what ``build_year`` and
    ``lifetime`` derive.
    """
    n = pypsa.Network()
    snapshots: pd.DataFrame = tables['snapshot']
    n.set_snapshots(pd.MultiIndex.from_arrays([snapshots['period_of'], snapshots['snapshot']]))
    n.investment_periods = list(tables['period']['period'])
    n.investment_period_weightings['years'] = 10
    n.investment_period_weightings['objective'] = tables['period_weight'].set_index('period')['value']

    n.add('Bus', 'hub')
    for carrier in tables['carrier']['carrier']:
        limited = growth_limit and carrier == 'wind'
        n.add(
            'Carrier',
            carrier,
            max_growth=float(tables['max_growth']) if limited else float('inf'),
            max_relative_growth=float(tables['max_relative_growth']) if limited else 0.0,
        )

    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        carrier=generators['gen_carrier'],
        p_nom_extendable=True,
        build_year=generators['build_period'],
        lifetime=LIFETIME,
        p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
        marginal_cost=tables['opex'].set_index('generator')['value'],
        capital_cost=tables['capex'].set_index('generator')['value'],
    )

    load: pd.Series = tables['load'].set_index('snapshot')['value']
    n.add('Load', 'l', bus='hub', p_set=load.to_numpy())
    return n


def balance_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per snapshot, keyed the way the port keys it.

    PyPSA's row index is ``(period, timestep)``; the port's is the flat snapshot a
    relation maps into a period, so the pairs are collapsed back to the port's
    labels in the order the instance lists them.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {'snapshot': list(range(len(dual))), 'value': [float(v) for v in dual.to_numpy()]}


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs', multi_investment_periods=True)
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'duals {json.dumps({"power_balance": balance_duals(n)})}')
    print(n.generators[['p_nom_opt', 'build_year']])
    print(n.generators_t.p)
    print('\nthe growth rows, and how tight each is:')
    print(n.model.constraints['Carrier-growth_limit'])
    print(n.model.constraints['Carrier-growth_limit'].dual.to_series())

    without = build(load_tables(), growth_limit=False)
    without.optimize(solver_name='highs', multi_investment_periods=True)
    print(f'\nwithout the growth limit: {float(without.objective)!r}')
    print(without.generators[['p_nom_opt']])
    return float(n.objective)


if __name__ == '__main__':
    main()
