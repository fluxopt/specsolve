"""`sector` as a gurobipy user writes it: a variable per (snapshot, installed pair).

The comprehension-of-`addVar` plus `addConstrs(quicksum(...))` form of the
Gurobi examples, with costs on `obj=`. Only installed pairs get a variable,
which is the YAML's `where: installed > 0`, and only the demand table's keys
get a balance row, which is its `where: demand`. No `name=`: `bench/README.md`
says why the harness switches naming off.
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
    demand_table = tables['demand']
    demand = dict(
        zip(
            zip(demand_table['snapshot'], demand_table['node'], demand_table['carrier'], strict=True),
            demand_table['value'],
            strict=True,
        )
    )
    delivers: dict[Any, list[tuple[Any, float]]] = collections.defaultdict(list)
    for tech, carrier, factor in zip(
        tables['produces']['tech'], tables['produces']['carrier'], tables['produces']['value'], strict=True
    ):
        delivers[tech].append((carrier, factor))
    sources: dict[tuple[Any, Any], list[tuple[Any, float]]] = collections.defaultdict(list)
    for node, tech in pairs:
        for carrier, factor in delivers[tech]:
            sources[node, carrier].append((tech, factor))
    snapshots = list(tables['snapshot']['snapshot'])

    m = gp.Model(env=env)
    p = {(s, n, t): m.addVar(ub=capacity[n, t], obj=cost[t]) for s in snapshots for n, t in pairs}
    m.addConstrs(gp.quicksum(f * p[s, n, t] for t, f in sources[n, c]) == d for (s, n, c), d in demand.items())
    return m
