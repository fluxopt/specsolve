"""`transport` as a pyomo user writes it: a `ConcreteModel` with rules.

The adjacency the YAML declares as relations is built as plain dicts inside
the timed build, and the balance rule sums over them.
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


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

    p_max = dict(zip(tables['p_max']['generator'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['generator'], tables['cost']['value'], strict=True))
    cap = dict(zip(tables['cap']['line'], tables['cap']['value'], strict=True))
    neg_cap = dict(zip(tables['neg_cap']['line'], tables['neg_cap']['value'], strict=True))
    load = {
        (s, b): v
        for s, b, v in zip(tables['load']['snapshot'], tables['load']['bus'], tables['load']['value'], strict=True)
    }
    at_bus = _grouped(tables['gen_bus']['generator'], tables['gen_bus']['bus'])
    into_bus = _grouped(tables['line_to']['line'], tables['line_to']['bus'])
    out_of_bus = _grouped(tables['line_from']['line'], tables['line_from']['bus'])

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.generators = pyo.Set(initialize=list(tables['generator']['generator']), ordered=True)
    m.lines = pyo.Set(initialize=list(tables['line']['line']), ordered=True)
    m.buses = pyo.Set(initialize=list(tables['bus']['bus']), ordered=True)
    m.p = pyo.Var(m.snapshots, m.generators, bounds=lambda _m, _s, g: (0.0, p_max[g]))
    m.f = pyo.Var(m.snapshots, m.lines, bounds=lambda _m, _s, line: (neg_cap[line], cap[line]))
    m.balance = pyo.Constraint(
        m.snapshots,
        m.buses,
        rule=lambda _m, s, b: (
            sum(_m.p[s, g] for g in at_bus[b])
            + sum(_m.f[s, line] for line in into_bus[b])
            - sum(_m.f[s, line] for line in out_of_bus[b])
            == load[s, b]
        ),
    )
    m.cost = pyo.Objective(expr=sum(cost[g] * m.p[s, g] for s in m.snapshots for g in m.generators), sense=pyo.minimize)
    return m
