"""`sector` as a linopy user writes it.

`mask=` on `p` is the YAML's `where: installed > 0`, as `nodal`'s is. The
`produces` table becomes a dense tech x carrier array with zeros where a
technology does not deliver a carrier, so `(p * produces).sum('tech')` is the
YAML's `sum(p * produces, over=tech)` on the full snapshot x node x tech x
carrier product. The balance's `mask=` is the YAML's `where: demand`.

The pivots of the tidy tables to dense arrays are inside `build`, so they are
timed. Labels come from the dimension tables.
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
    nodes = pd.Index(tables['node']['node'], name='node')
    techs = pd.Index(tables['tech']['tech'], name='tech')
    carriers = pd.Index(tables['carrier']['carrier'], name='carrier')

    capacity = xr.DataArray(
        tables['installed'].pivot(index='node', columns='tech', values='value').reindex(index=nodes, columns=techs)
    ).fillna(0.0)
    produces = xr.DataArray(
        tables['produces'].pivot(index='tech', columns='carrier', values='value').reindex(index=techs, columns=carriers)
    ).fillna(0.0)
    demand = (
        tables['demand']
        .set_index(['snapshot', 'node', 'carrier'])['value']
        .to_xarray()
        .reindex(snapshot=snapshots, node=nodes, carrier=carriers)
    )
    cost = tables['cost'].set_index('tech')['value'].reindex(techs)

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=capacity, coords=[snapshots, nodes, techs], mask=capacity > 0, name='p')
    m.add_constraints((p.fillna(0) * produces).sum('tech') == demand.fillna(0.0), mask=demand.notnull(), name='balance')
    m.add_objective((p.fillna(0) * cost).sum())
    return m
