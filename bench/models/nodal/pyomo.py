"""`nodal` as a pyomo user writes it: a `ConcreteModel` with rules.

The index set is the installed (node, tech) pairs, which is the YAML's
`where: installed > 0`. Each node's portfolio is a plain dict built inside the
timed build, and the balance rule sums over it.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

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

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.nodes = pyo.Set(initialize=list(tables['node']['node']), ordered=True)
    m.pairs = pyo.Set(initialize=pairs, dimen=2, ordered=True)
    m.p = pyo.Var(m.snapshots, m.pairs, bounds=lambda _m, _s, n, t: (0.0, capacity[n, t]))
    m.balance = pyo.Constraint(
        m.snapshots, m.nodes, rule=lambda _m, s, n: sum(_m.p[s, n, t] for t in portfolio[n]) == demand[s, n]
    )
    m.cost = pyo.Objective(
        expr=sum(cost[t] * m.p[s, n, t] for s in m.snapshots for n, t in m.pairs), sense=pyo.minimize
    )
    return m
