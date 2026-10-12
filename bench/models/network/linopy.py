"""`network` as a linopy user writes it.

`mask=` is the YAML's `where: installed > 0`, as a bus x tech plane linopy
broadcasts along the snapshot axis. The joins the YAML declares as relations
are dense bus x line and bus x store incidence matrices, built inside the
timed build as `transport`'s are. `shift(...).fillna(0)` gives the first
snapshot an empty store, which is the YAML's `edge=0`, and `p.fillna(0)` makes
an absent pair contribute zero, as in `nodal`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy
    import pandas as pd
    import xarray as xr

    snapshots = pd.Index(tables['snapshot']['snapshot'], name='snapshot')
    buses = pd.Index(tables['bus']['bus'], name='bus')
    techs = pd.Index(tables['tech']['tech'], name='tech')
    lines = pd.Index(tables['line']['line'], name='line')
    stores = pd.Index(tables['store']['store'], name='store')

    capacity = xr.DataArray(
        tables['installed'].pivot(index='bus', columns='tech', values='value').reindex(index=buses, columns=techs)
    ).fillna(0.0)
    p_max = (
        tables['p_max']
        .set_index(['snapshot', 'bus', 'tech'])['value']
        .to_xarray()
        .reindex(snapshot=snapshots, bus=buses, tech=techs)
    ).fillna(0.0)
    load = xr.DataArray(
        tables['load'].pivot(index='snapshot', columns='bus', values='value').reindex(index=snapshots, columns=buses)
    )
    cost = tables['cost'].set_index('tech')['value'].reindex(techs)
    cap = tables['cap'].set_index('line')['value'].reindex(lines)
    neg_cap = tables['neg_cap'].set_index('line')['value'].reindex(lines)
    e_max = tables['e_max'].set_index('store')['value'].reindex(stores)
    p_store = tables['p_store'].set_index('store')['value'].reindex(stores)
    eta = tables['eta'].set_index('store')['value'].reindex(stores)

    flow_in = pd.DataFrame(0.0, index=buses, columns=lines)
    for line, src, dst in zip(
        tables['line_from']['line'], tables['line_from']['bus'], tables['line_to']['bus'], strict=True
    ):
        flow_in.loc[dst, line] += 1.0
        flow_in.loc[src, line] -= 1.0
    store_at = pd.DataFrame(0.0, index=buses, columns=stores)
    for store, bus in zip(tables['store_bus']['store'], tables['store_bus']['bus'], strict=True):
        store_at.loc[bus, store] = 1.0

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, buses, techs], mask=capacity > 0, name='p')
    f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, lines], name='f')
    charge = m.add_variables(lower=0, upper=p_store, coords=[snapshots, stores], name='charge')
    discharge = m.add_variables(lower=0, upper=p_store, coords=[snapshots, stores], name='discharge')
    soc = m.add_variables(lower=0, upper=e_max, coords=[snapshots, stores], name='soc')

    m.add_constraints(
        p.fillna(0).sum('tech')
        + (f * xr.DataArray(flow_in)).sum('line')
        + ((discharge - charge) * xr.DataArray(store_at)).sum('store')
        == load,
        name='balance',
    )
    m.add_constraints(soc - soc.shift(snapshot=1).fillna(0) - charge * eta + discharge == 0, name='soc_balance')
    m.add_objective((p.fillna(0) * cost).sum())
    return m
