"""`transport` as a gurobipy user writes it: a variable per (snapshot, entity).

The bus balance sums over the generators at the bus and over each end of the
lines touching it, through adjacency lists built once inside the timed build.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _grouped(keys: Any, values: Any) -> dict[Any, list[Any]]:
    """``bus -> the entities on it`` — the adjacency a balance row sums over."""
    out: dict[Any, list[Any]] = collections.defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        out[value].append(key)
    return out


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    cap = dict(zip(tables['cap']['line'], tables['cap']['value'], strict=True))
    neg_cap = dict(zip(tables['neg_cap']['line'], tables['neg_cap']['value'], strict=True))
    load = {
        (s, b): v
        for s, b, v in zip(tables['load']['snapshot'], tables['load']['bus'], tables['load']['value'], strict=True)
    }

    snapshots = list(tables['snapshot']['snapshot'])
    generators = list(tables['generator']['generator'])
    lines = list(tables['line']['line'])
    buses = list(tables['bus']['bus'])
    at_bus = _grouped(tables['gen_bus']['generator'], tables['gen_bus']['bus'])
    into_bus = _grouped(tables['line_to']['line'], tables['line_to']['bus'])
    out_of_bus = _grouped(tables['line_from']['line'], tables['line_from']['bus'])

    m = gp.Model(env=env)
    p = {(s, g): m.addVar(ub=p_max[g], obj=cost[g]) for s in snapshots for g in generators}
    f = {(s, line): m.addVar(lb=neg_cap[line], ub=cap[line]) for s in snapshots for line in lines}
    m.addConstrs(
        gp.quicksum(p[s, g] for g in at_bus[b])
        + gp.quicksum(f[s, line] for line in into_bus[b])
        - gp.quicksum(f[s, line] for line in out_of_bus[b])
        == load[s, b]
        for s in snapshots
        for b in buses
    )
    return m
