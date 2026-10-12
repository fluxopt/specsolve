"""`network` as a pyomo user writes it: a `ConcreteModel` with rules.

The index set of `p` is the installed (bus, tech) pairs, which is the YAML's
`where: installed > 0`. What each bus sums over, its portfolio, the lines
into and out of it and its stores, is plain dicts built inside the timed
build. The first snapshot has no earlier state of charge, which is the YAML's
`edge=0`.
"""

from __future__ import annotations

import collections
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def _grouped(keys: Any, values: Any) -> dict[Any, list[Any]]:
    """``bus -> what sits on it`` — the adjacency a balance row sums over."""
    out: dict[Any, list[Any]] = collections.defaultdict(list)
    for key, value in zip(keys, values, strict=True):
        out[value].append(key)
    return out


def _by_name(table: Any) -> dict[Any, float]:
    return dict(zip(table.iloc[:, 0], table['value'], strict=True))


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

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
    into = _grouped(tables['line_to']['line'], tables['line_to']['bus'])
    out_of = _grouped(tables['line_from']['line'], tables['line_from']['bus'])
    stores_at = _grouped(tables['store_bus']['store'], tables['store_bus']['bus'])
    snapshots = list(tables['snapshot']['snapshot'])
    previous = dict(zip(snapshots[1:], snapshots[:-1], strict=True))

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=snapshots, ordered=True)
    m.buses = pyo.Set(initialize=list(tables['bus']['bus']), ordered=True)
    m.pairs = pyo.Set(initialize=pairs, dimen=2, ordered=True)
    m.lines = pyo.Set(initialize=list(tables['line']['line']), ordered=True)
    m.stores = pyo.Set(initialize=list(tables['store']['store']), ordered=True)
    m.p = pyo.Var(m.snapshots, m.pairs, bounds=lambda _m, s, b, t: (0.0, p_max[s, b, t]))
    m.f = pyo.Var(m.snapshots, m.lines, bounds=lambda _m, _s, ln: (neg_cap[ln], cap[ln]))
    m.charge = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, p_store[st]))
    m.discharge = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, p_store[st]))
    m.soc = pyo.Var(m.snapshots, m.stores, bounds=lambda _m, _s, st: (0.0, e_max[st]))
    m.balance = pyo.Constraint(
        m.snapshots,
        m.buses,
        rule=lambda _m, s, b: (
            sum(_m.p[s, b, t] for t in portfolio[b])
            + sum(_m.f[s, ln] for ln in into[b])
            - sum(_m.f[s, ln] for ln in out_of[b])
            + sum(_m.discharge[s, st] - _m.charge[s, st] for st in stores_at[b])
            == load[s, b]
        ),
    )
    m.soc_balance = pyo.Constraint(
        m.snapshots,
        m.stores,
        rule=lambda _m, s, st: (
            _m.soc[s, st]
            == (_m.soc[previous[s], st] if s in previous else 0) + _m.charge[s, st] * eta[st] - _m.discharge[s, st]
        ),
    )
    m.cost = pyo.Objective(
        expr=sum(cost[t] * m.p[s, b, t] for s in m.snapshots for b, t in m.pairs), sense=pyo.minimize
    )
    return m
