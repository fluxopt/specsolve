"""`network` as a gurobipy user writes it: a variable per (snapshot, entity).

The comprehension-of-`addVar` plus `addConstrs(quicksum(...))` form of the
Gurobi examples, with costs on `obj=`. Only installed pairs get a `p`, which
is the YAML's `where: installed > 0`, and the first snapshot has no earlier
state of charge, which is its `edge=0`. No `name=`:
`bench/README.md` says why the harness switches naming off.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _grouped(keys: Any, values: Any) -> dict[Any, list[Any]]:
    """``bus -> what sits on it`` — the adjacency a balance or pool row sums over."""
    out: dict[Any, list[Any]] = collections.defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        out[value].append(key)
    return out


def _by_name(table: Any) -> dict[Any, float]:
    return dict(zip(table[table.columns[0]], table['value'], strict=True))


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    installed = tables['installed']
    pairs = list(zip(installed['bus'], installed['tech'], strict=True))
    p_max = dict(
        zip(
            zip(tables['p_max']['snapshot'], tables['p_max']['bus'], tables['p_max']['tech'], strict=True),
            tables['p_max']['value'],
            strict=True,
        )
    )
    load = dict(
        zip(zip(tables['load']['snapshot'], tables['load']['bus'], strict=True), tables['load']['value'], strict=True)
    )
    cost, cap, neg_cap = _by_name(tables['cost']), _by_name(tables['cap']), _by_name(tables['neg_cap'])
    e_max, p_store, eta = _by_name(tables['e_max']), _by_name(tables['p_store']), _by_name(tables['eta'])
    portfolio = _grouped(installed['tech'], installed['bus'])
    into = _grouped(tables['line.to_bus']['line'], tables['line.to_bus']['to_bus'])
    out_of = _grouped(tables['line.from_bus']['line'], tables['line.from_bus']['from_bus'])
    stores_at = _grouped(tables['store.bus']['store'], tables['store.bus']['bus'])
    pool_of = tables['bus_tech.pool']
    members = _grouped(zip(pool_of['bus'], pool_of['tech'], strict=True), pool_of['pool'])
    limit = _by_name(tables['limit'])
    snapshots = list(tables['snapshot']['snapshot'])
    previous = dict(zip(snapshots[1:], snapshots[:-1], strict=True))
    buses, lines, stores = list(tables['bus']['bus']), list(tables['line']['line']), list(tables['store']['store'])

    m = gp.Model(env=env)
    p = {(s, b, t): m.addVar(ub=p_max[s, b, t], obj=cost[t]) for s in snapshots for b, t in pairs}
    f = {(s, ln): m.addVar(lb=neg_cap[ln], ub=cap[ln]) for s in snapshots for ln in lines}
    charge = {(s, st): m.addVar(ub=p_store[st]) for s in snapshots for st in stores}
    discharge = {(s, st): m.addVar(ub=p_store[st]) for s in snapshots for st in stores}
    soc = {(s, st): m.addVar(ub=e_max[st]) for s in snapshots for st in stores}
    m.addConstrs(
        gp.quicksum(p[s, b, t] for t in portfolio[b])
        + gp.quicksum(f[s, ln] for ln in into[b])
        - gp.quicksum(f[s, ln] for ln in out_of[b])
        + gp.quicksum(discharge[s, st] - charge[s, st] for st in stores_at[b])
        == load[s, b]
        for s in snapshots
        for b in buses
    )
    m.addConstrs(
        soc[s, st] == (soc[previous[s], st] if s in previous else 0) + eta[st] * charge[s, st] - discharge[s, st]
        for s in snapshots
        for st in stores
    )
    m.addConstrs(gp.quicksum(p[s, b, t] for b, t in members[pool]) <= limit[pool] for s in snapshots for pool in limit)
    return m
