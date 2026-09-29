"""`nodal` as a linopy user writes it.

A quarter of the (node, tech) pairs exist. `mask=` is linopy's spelling of the
YAML's `where: installed > 0`, as a node x tech plane linopy broadcasts along
the snapshot axis.

`fillna(0)` makes an absent slot contribute zero to the sum, as the YAML means;
without it the pinned linopy warns and the model depends on a global option.

The pivot of the tidy `installed` table to a dense node x tech frame is inside
`build`, so it is timed. Labels come from the dimension tables, so a tech no
node installed stays in the model's coordinates.
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

    capacity = xr.DataArray(
        tables['installed'].pivot(index='node', columns='tech', values='value').reindex(index=nodes, columns=techs)
    ).fillna(0.0)
    demand = xr.DataArray(
        tables['demand'].pivot(index='snapshot', columns='node', values='value').reindex(index=snapshots, columns=nodes)
    )
    cost = tables['cost'].set_index('tech')['value'].reindex(techs)

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=capacity, coords=[snapshots, nodes, techs], mask=capacity > 0, name='p')
    m.add_constraints(p.fillna(0).sum('tech') == demand, name='balance')
    m.add_objective((p.fillna(0) * cost).sum())
    return m
