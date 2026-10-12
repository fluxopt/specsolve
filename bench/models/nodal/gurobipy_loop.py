"""`nodal` as a gurobipy user writes it: a variable per (snapshot, installed pair).

The comprehension-of-`addVar` plus `addConstrs(quicksum(...))` form of the
Gurobi examples, with costs on `obj=`. Only installed pairs get a variable,
which is the YAML's `where: installed > 0`. No `name=`: `bench/README.md` says
why the harness switches naming off.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(env: Any, tables: Mapping[str, Any]) -> Any:
    import gurobipy as gp

    installed = tables['installed']
    pairs = list(zip(installed['node'], installed['tech'], strict=True))
    capacity = dict(zip(pairs, installed['value'], strict=True))
    cost = dict(zip(tables['cost']['tech'], tables['cost']['value'], strict=True))
    demand = dict(
        zip(
            zip(tables['demand']['snapshot'], tables['demand']['node'], strict=True),
            tables['demand']['value'],
            strict=True,
        )
    )
    portfolio: dict[Any, list[Any]] = collections.defaultdict(list)
    for node, tech in pairs:
        portfolio[node].append(tech)
    snapshots = list(tables['snapshot']['snapshot'])
    nodes = list(tables['node']['node'])

    m = gp.Model(env=env)
    p = {(s, n, t): m.addVar(ub=capacity[n, t], obj=cost[t]) for s in snapshots for n, t in pairs}
    m.addConstrs(gp.quicksum(p[s, n, t] for t in portfolio[n]) == demand[s, n] for s in snapshots for n in nodes)
    return m
