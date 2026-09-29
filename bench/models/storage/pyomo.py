"""`storage` as a pyomo user writes it.

The cyclic state of charge is `snapshots[i - 1]`; Python's negative indexing
closes the ring at the first snapshot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    load = dict(zip(tables['load']['snapshot'], tables['load']['value'], strict=True))
    e_max = dict(zip(tables['e_max']['store'], tables['e_max']['value'], strict=True))
    p_store = dict(zip(tables['p_store']['store'], tables['p_store']['value'], strict=True))
    eta = dict(zip(tables['eta']['store'], tables['eta']['value'], strict=True))

    snapshots = list(tables['snapshot']['snapshot'])
    previous = {s: snapshots[i - 1] for i, s in enumerate(snapshots)}

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=snapshots, ordered=True)
    m.generators = pyo.Set(initialize=list(tables['generator']['generator']), ordered=True)
    m.stores = pyo.Set(initialize=list(tables['store']['store']), ordered=True)
    m.p = pyo.Var(m.snapshots, m.generators, bounds=lambda _m, _s, g: (0.0, p_max[g]))
    m.charge = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, p_store[st]))
    m.discharge = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, p_store[st]))
    m.soc = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, e_max[st]))
    m.power_balance = pyo.Constraint(
        m.snapshots,
        rule=lambda _m, s: (
            sum(_m.p[s, g] for g in _m.generators)
            + sum(_m.discharge[s, st] for st in _m.stores)
            - sum(_m.charge[s, st] for st in _m.stores)
            == load[s]
        ),
    )
    m.soc_balance = pyo.Constraint(
        m.snapshots,
        m.stores,
        rule=lambda _m, s, st: (
            _m.soc[s, st] == _m.soc[previous[s], st] + _m.charge[s, st] * eta[st] - _m.discharge[s, st]
        ),
    )
    m.cost = pyo.Objective(expr=sum(cost[g] * m.p[s, g] for s in m.snapshots for g in m.generators), sense=pyo.minimize)
    return m
