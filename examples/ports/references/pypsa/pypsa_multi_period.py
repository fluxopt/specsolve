#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pypsa==1.2.4", "linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``pypsa_multi_period``: PyPSA's own multi-period investment. See docs/examples/index.md.

    pixi exec -s uv uv run --script examples/ports/references/pypsa/pypsa_multi_period.py

Pinned above to the versions that produced the number in ``references.json``,
and run out of band — PyPSA is not a dependency of this project.

It reads the same instance the port attaches and builds the network with PyPSA's
own objects. Nothing here imports specsolve.

**A build year and a lifetime decide which rows an asset appears in.**
``optimize(multi_investment_periods=True)`` turns the snapshot index into a
``(period, timestep)`` MultiIndex and derives an ``active`` mask per asset per
period from ``build_year`` and ``lifetime``. An inactive asset gets **no
dispatch variable** in that period — not a variable pinned to zero — and its
capacity is paid for only in the periods where it is active.

The three generators cover the three cases: ``coal`` is built in 2030 and
retires before 2040 (lifetime 5), ``gas`` is built in 2030 and lives through both
(lifetime 40), and ``wind`` is built in 2040 and exists in neither row before it.

**Both weightings are set, and only one of them shows.**
``investment_period_weightings.objective`` discounts 2040 to 0.7 and multiplies
*both* the capex of an asset active in that period and the marginal cost of every
snapshot inside it. ``years`` is 10 for each period and does not reach the
objective at all, because ``capital_cost`` is given directly rather than
annuitised from an ``overnight_cost`` (``costs.py:119``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pypsa

DATA = Path(__file__).resolve().parents[2] / 'data' / 'pypsa_multi_period.json'

#: ``build_year`` and ``lifetime`` per generator — PyPSA's spelling of the port's
#: sparse ``active`` table.
LIFETIMES: dict[str, tuple[int, int]] = {'coal': (2030, 5), 'gas': (2030, 40), 'wind': (2040, 30)}


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> pypsa.Network:
    """The port's tables as a PyPSA network, column for column.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.

    The port's flat ``snapshot`` axis carries a relation into ``period``; PyPSA
    wants the same fact as a ``(period, timestep)`` MultiIndex, so the snapshots
    are paired with the period each falls in. ``investment_period_weightings``
    takes the port's ``period_weight`` as its ``objective`` column, and ``years``
    is 10 for both periods — set explicitly, because a default of 1 would make
    the decade an accident rather than a statement.
    """
    n = pypsa.Network()
    snapshots: pd.DataFrame = tables['snapshot']
    periods = list(tables['period']['period'])
    n.set_snapshots(pd.MultiIndex.from_arrays([snapshots['period_of'], snapshots['snapshot']]))
    n.investment_periods = periods
    n.investment_period_weightings['years'] = 10
    n.investment_period_weightings['objective'] = tables['period_weight'].set_index('period')['value']

    n.add('Bus', 'hub')
    generators: pd.DataFrame = tables['generator'].set_index('generator')
    n.add(
        'Generator',
        generators.index,
        bus='hub',
        p_nom_extendable=True,
        build_year=[LIFETIMES[g][0] for g in generators.index],
        lifetime=[LIFETIMES[g][1] for g in generators.index],
        p_nom_max=tables['p_nom_max'].set_index('generator')['value'],
        marginal_cost=tables['opex'].set_index('generator')['value'],
        capital_cost=tables['capex'].set_index('generator')['value'],
    )

    load: pd.Series = tables['load'].set_index('snapshot')['value']
    n.add('Load', 'l', bus='hub', p_set=load.to_numpy())
    return n


def balance_duals(n: pypsa.Network) -> dict[str, list]:
    """The dual of the nodal balance per snapshot, keyed the way the port keys it.

    PyPSA's row index is ``(period, timestep)``; the port's is the flat snapshot
    the relation maps into a period, so the pairs are collapsed back to the port's
    labels in the order the instance lists them.
    """
    dual = n.model.constraints['Bus-nodal_balance'].dual.to_series()
    return {
        'snapshot': list(range(len(dual))),
        'value': [float(v) for v in dual.to_numpy()],
    }


def main() -> float:
    n = build(load_tables())
    status, condition = n.optimize(solver_name='highs', multi_investment_periods=True)
    assert status == 'ok', f'{status}: {condition}'
    print(f'pypsa {pypsa.__version__}')
    print(f'objective {float(n.objective)!r}')
    print(f'objective_constant {float(n.objective_constant)!r}')
    print(f'duals {json.dumps({"power_balance": balance_duals(n)})}')
    print(n.generators[['p_nom_opt', 'build_year', 'lifetime']])
    print(n.generators_t.p)
    print('\nwhich (period, generator) pairs PyPSA gave a dispatch variable:')
    print(
        n.model.variables['Generator-p'].mask.to_pandas() if hasattr(n.model.variables['Generator-p'], 'mask') else '-'
    )
    return float(n.objective)


if __name__ == '__main__':
    main()
