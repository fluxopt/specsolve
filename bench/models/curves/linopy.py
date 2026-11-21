"""`curves` as a linopy user writes it: one `add_piecewise_formulation` over every unit.

`method='lp'` is the curve as its segment lines, as the YAML's `method: lp` is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import warnings

    import linopy

    warnings.filterwarnings('ignore', category=linopy.EvolvingAPIWarning)  # the API is new, and this is its use

    p_max = tables['p_max'].set_index('unit')['value']
    load = tables['load'].set_index('snapshot')['value']
    bp_x = tables['bp_x'].set_index(['unit', 'bp'])['value'].to_xarray().rename(bp='_breakpoint')
    bp_y = tables['bp_y'].set_index(['unit', 'bp'])['value'].to_xarray().rename(bp='_breakpoint')

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
    m.add_piecewise_formulation((op_cost, bp_y, '>='), (p, bp_x), method='lp', name='cost_curve')
    m.add_constraints(p.sum('unit') == load, name='power_balance')
    m.add_objective(op_cost.sum())
    return m
