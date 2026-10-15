"""`transport` as a linopy user writes it.

Each of the YAML's `sum(x, over=key, by=rel[bus])` is linopy's
`x.groupby(rel).sum()`, with the relation as a key -> bus array. A grouped sum
has a row only for a bus the relation reaches, so `reindex(bus=buses)` puts the
other buses back and `fillna(0)` makes them contribute zero, as the YAML means.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy
    import xarray as xr

    p_max = tables['p_max'].set_index('generator')['value']
    cost = tables['cost'].set_index('generator')['value']
    cap = tables['cap'].set_index('line')['value']
    neg_cap = tables['neg_cap'].set_index('line')['value']
    load = xr.DataArray(tables['load'].pivot(index='snapshot', columns='bus', values='value'))
    snapshots, buses = load.indexes['snapshot'], load.indexes['bus']

    def into_bus(x: Any, relation: str, key: str) -> Any:
        by = xr.DataArray(tables[relation].set_index(key)['bus'].reindex(x.indexes[key]), name='bus')
        return x.groupby(by).sum().reindex(bus=buses).fillna(0)

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
    f = m.add_variables(lower=neg_cap, upper=cap, coords=[snapshots, cap.index], name='f')
    m.add_constraints(
        into_bus(p, 'gen_bus', 'generator') + into_bus(f, 'line_to', 'line') - into_bus(f, 'line_from', 'line') == load,
        name='balance',
    )
    m.add_objective((p * cost).sum())
    return m
