"""`fleet` as a pyomo user writes it: twelve `Var` blocks and seven `Constraint`s.

Every quantity is its own component, so a `ConcreteModel` pays its
per-component cost twelve times.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

#: The twelve, in the order the model file declares them.
QUANTITIES = (
    'p', 'p_up', 'p_down', 'reserve_up', 'reserve_down', 'charge',
    'discharge', 'soc', 'spill', 'curtail', 'import_', 'export_',
)  # fmt: skip


def build(tables: Mapping[str, Any]) -> Any:
    import pyomo.environ as pyo

    p_max = dict(zip(tables['p_max']['unit'], tables['p_max']['value'], strict=True))
    cost = dict(zip(tables['cost']['unit'], tables['cost']['value'], strict=True))
    demand = dict(zip(tables['demand']['snapshot'], tables['demand']['value'], strict=True))

    m = pyo.ConcreteModel()
    m.snapshots = pyo.Set(initialize=list(tables['snapshot']['snapshot']), ordered=True)
    m.units = pyo.Set(initialize=list(tables['unit']['unit']), ordered=True)
    for name in QUANTITIES:
        setattr(m, name, pyo.Var(m.snapshots, m.units, bounds=lambda _m, _s, u: (0.0, p_max[u])))

    m.balance = pyo.Constraint(
        m.snapshots,
        rule=lambda _m, s: sum(_m.p[s, u] + _m.discharge[s, u] + _m.import_[s, u] for u in _m.units) == demand[s],
    )
    m.ramp_up = pyo.Constraint(m.snapshots, m.units, rule=lambda _m, s, u: _m.p_up[s, u] <= p_max[u])
    m.ramp_down = pyo.Constraint(m.snapshots, m.units, rule=lambda _m, s, u: _m.p_down[s, u] <= p_max[u])
    m.reserve = pyo.Constraint(
        m.snapshots, m.units, rule=lambda _m, s, u: _m.reserve_up[s, u] + _m.reserve_down[s, u] <= p_max[u]
    )
    m.storage = pyo.Constraint(
        m.snapshots, m.units, rule=lambda _m, s, u: _m.soc[s, u] + _m.charge[s, u] - _m.discharge[s, u] <= p_max[u]
    )
    m.spillage = pyo.Constraint(
        m.snapshots, m.units, rule=lambda _m, s, u: _m.spill[s, u] + _m.curtail[s, u] <= p_max[u]
    )
    m.exchange = pyo.Constraint(
        m.snapshots, m.units, rule=lambda _m, s, u: _m.import_[s, u] + _m.export_[s, u] <= p_max[u]
    )
    m.cost = pyo.Objective(
        expr=sum(cost[u] * (m.p[s, u] + m.discharge[s, u] + m.import_[s, u]) for s in m.snapshots for u in m.units),
        sense=pyo.minimize,
    )
    return m
