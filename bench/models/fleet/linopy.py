"""`fleet` as a linopy user writes it: twelve declarations, not one big one.

The case charges a per-declaration cost twelve times over, and linopy pays it
once per `add_variables`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy

    p_max = tables['p_max'].set_index('unit')['value']
    cost = tables['cost'].set_index('unit')['value']
    demand = tables['demand'].set_index('snapshot')['value']
    over = [demand.index, p_max.index]

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=over, name='p')
    p_up = m.add_variables(lower=0, upper=p_max, coords=over, name='p_up')
    p_down = m.add_variables(lower=0, upper=p_max, coords=over, name='p_down')
    reserve_up = m.add_variables(lower=0, upper=p_max, coords=over, name='reserve_up')
    reserve_down = m.add_variables(lower=0, upper=p_max, coords=over, name='reserve_down')
    charge = m.add_variables(lower=0, upper=p_max, coords=over, name='charge')
    discharge = m.add_variables(lower=0, upper=p_max, coords=over, name='discharge')
    soc = m.add_variables(lower=0, upper=p_max, coords=over, name='soc')
    spill = m.add_variables(lower=0, upper=p_max, coords=over, name='spill')
    curtail = m.add_variables(lower=0, upper=p_max, coords=over, name='curtail')
    import_ = m.add_variables(lower=0, upper=p_max, coords=over, name='import_')
    export_ = m.add_variables(lower=0, upper=p_max, coords=over, name='export_')

    m.add_constraints(p.sum('unit') + discharge.sum('unit') + import_.sum('unit') == demand, name='balance')
    m.add_constraints(p_up <= p_max, name='ramp_up')
    m.add_constraints(p_down <= p_max, name='ramp_down')
    m.add_constraints(reserve_up + reserve_down <= p_max, name='reserve')
    m.add_constraints(soc + charge - discharge <= p_max, name='storage')
    m.add_constraints(spill + curtail <= p_max, name='spillage')
    m.add_constraints(import_ + export_ <= p_max, name='exchange')
    m.add_objective((p * cost).sum() + (discharge * cost).sum() + (import_ * cost).sum())
    return m
