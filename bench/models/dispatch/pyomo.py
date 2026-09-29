"""`dispatch` as a pyomo user writes it: a `ConcreteModel` with rules.

The `Set` / `Var(bounds=…)` / `Constraint(rule=…)` / `Objective(expr=…)` form
of pyomo's documentation. The YAML's `where: p_max > 0` leaves the retired
generator out of the index set.
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

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.generators = pyo.Set(initialize=[g for g in tables['generator']['generator'] if p_max[g] > 0], ordered=True)
    m.p = pyo.Var(m.snapshots, m.generators, bounds=lambda _m, _s, g: (0.0, p_max[g]))
    m.power_balance = pyo.Constraint(m.snapshots, rule=lambda _m, s: sum(_m.p[s, g] for g in _m.generators) == load[s])
    m.cost = pyo.Objective(expr=sum(cost[g] * m.p[s, g] for s in m.snapshots for g in m.generators), sense=pyo.minimize)
    return m
