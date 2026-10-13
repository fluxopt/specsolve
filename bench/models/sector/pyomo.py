"""`sector` as a pyomo user writes it: a `ConcreteModel` with rules.

The index set of `p` is the installed (node, tech) pairs, which is the YAML's
`where: installed > 0`. The balance is indexed by the demand table's keys,
which is its `where: demand`. What each (node, carrier) can draw on is a plain
dict built inside the timed build: the installed technologies at the node that
deliver the carrier, each with its `produces` factor.
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

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.pairs = pyo.Set(initialize=pairs, dimen=2, ordered=True)
    m.rows = pyo.Set(initialize=list(demand), dimen=3, ordered=True)
    m.p = pyo.Var(m.snapshots, m.pairs, bounds=lambda _m, _s, n, t: (0.0, capacity[n, t]))
    m.balance = pyo.Constraint(
        m.rows,
        rule=lambda _m, s, n, c: sum(f * _m.p[s, n, t] for t, f in sources[n, c]) == demand[s, n, c],
    )
    m.cost = pyo.Objective(
        expr=sum(cost[t] * m.p[s, n, t] for s in m.snapshots for n, t in m.pairs), sense=pyo.minimize
    )
    return m
