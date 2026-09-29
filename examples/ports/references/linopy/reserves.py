#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.9.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``reserves``: the same LP, hand-written as incidence matrices.

    pixi exec -s uv uv run --script examples/ports/references/linopy/reserves.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The script builds every mapping the YAML states as a relation or a weighted
table — generator/line incidence onto buses, the three-legged offer set onto
generators, markets and tranches, and the overlapping zone weights — as dense
matrices multiplied through by hand.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parents[2] / 'data' / 'reserves.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def indicator(rows: pd.Index, table: pd.DataFrame, member: str, leg: str) -> xr.DataArray:
    """The rows x members 0/1 matrix of one leg, a null leg contributing no entry."""
    out = pd.DataFrame(0.0, index=rows, columns=pd.Index(table[member], name=member))
    for m, target in zip(table[member], table[leg], strict=True):
        if pd.notna(target):
            out.loc[target, m] = 1.0
    out.index.name = rows.name
    return xr.DataArray(out)


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    series = {
        k: tables[k].set_index(tables[k].columns[0])['value']
        for k in (
            'p_max',
            'energy_cost',
            'load',
            'cap',
            'neg_cap',
            'bus_cap',
            'offer_cost',
            'req',
            'tranche_frac',
            'zone_req',
        )
    }
    buses = pd.Index(series['load'].index, name='bus')
    offers = tables['offer'].set_index('offer')
    zones = pd.Index(series['zone_req'].index, name='zone')

    gen_at = indicator(buses, tables['gen_bus'], 'generator', 'bus')
    line_in = indicator(buses, tables['line_to'], 'line', 'bus')
    line_out = indicator(buses, tables['line_from'], 'line', 'bus')
    offer_gen = indicator(pd.Index(series['p_max'].index, name='generator'), tables['gen_of'], 'offer', 'generator')
    offer_market = indicator(pd.Index(series['req'].index, name='market'), tables['market_of'], 'offer', 'market')

    zone_at = pd.DataFrame(0.0, index=zones, columns=series['p_max'].index)
    for gen, zone, share in zip(
        tables['zone_share']['generator'], tables['zone_share']['zone'], tables['zone_share']['value'], strict=True
    ):
        zone_at.loc[zone, gen] = share
    zone_at.columns.name = 'generator'

    tranche_of = tables['tranche_of'].set_index('offer')['tranche']
    gen_of = tables['gen_of'].set_index('offer')['generator']
    r_cap = tranche_of.map(series['tranche_frac']) * gen_of.map(series['p_max'])
    f_cap = tables['line_from'].set_index('line')['bus'].map(series['bus_cap'])

    m = linopy.Model()
    p = m.add_variables(lower=0, coords=[series['p_max'].index], name='p')
    f = m.add_variables(lower=series['neg_cap'], upper=series['cap'], coords=[series['cap'].index], name='f')
    r = m.add_variables(lower=0, upper=r_cap, coords=[offers.index], name='r')

    m.add_constraints(
        (p * gen_at).sum('generator') + (f * line_in).sum('line') - (f * line_out).sum('line') == series['load'],
        name='balance',
    )
    m.add_constraints(f <= f_cap, name='export_cap')
    m.add_constraints((r * offer_market).sum('offer') >= series['req'], name='requirement')
    reserve_of = (r * offer_gen).sum('offer')
    m.add_constraints(p + reserve_of <= series['p_max'], name='headroom')
    m.add_constraints((reserve_of * xr.DataArray(zone_at)).sum('generator') >= series['zone_req'], name='zone_cover')
    m.add_objective((p * series['energy_cost']).sum() + (r * series['offer_cost']).sum())
    return m


def nodal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual, tidy: one price per bus."""
    dual = m.constraints['balance'].dual
    return {
        'bus': [str(b) for b in dual.indexes['bus']],
        'value': [float(v) for v in dual.values.ravel()],
    }


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"balance": nodal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
