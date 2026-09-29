"""`storage` as a gurobipy user writes it: a variable per (snapshot, entity).

The cyclic state of charge closes the ring with `snapshots[i - 1]`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    load = dict(zip(tables['load']['snapshot'], tables['load']['value'], strict=True))
    e_max = dict(zip(tables['e_max']['store'], tables['e_max']['value'], strict=True))
    p_store = dict(zip(tables['p_store']['store'], tables['p_store']['value'], strict=True))
    eta = dict(zip(tables['eta']['store'], tables['eta']['value'], strict=True))

    snapshots = list(tables['snapshot']['snapshot'])
    generators = list(tables['generator']['generator'])
    stores = list(tables['store']['store'])
    previous = {s: snapshots[i - 1] for i, s in enumerate(snapshots)}

    m = gp.Model(env=env)
    p = {(s, g): m.addVar(ub=p_max[g], obj=cost[g]) for s in snapshots for g in generators}
    charge = {(s, st): m.addVar(ub=p_store[st]) for s in snapshots for st in stores}
    discharge = {(s, st): m.addVar(ub=p_store[st]) for s in snapshots for st in stores}
    soc = {(s, st): m.addVar(ub=e_max[st]) for s in snapshots for st in stores}
    m.addConstrs(
        gp.quicksum(p[s, g] for g in generators)
        + gp.quicksum(discharge[s, st] for st in stores)
        - gp.quicksum(charge[s, st] for st in stores)
        == load[s]
        for s in snapshots
    )
    m.addConstrs(
        soc[s, st] - soc[previous[s], st] - charge[s, st] * eta[st] + discharge[s, st] == 0
        for s in snapshots
        for st in stores
    )
    return m
