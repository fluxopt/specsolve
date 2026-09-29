"""`dispatch` as a linopy user writes it — `examples/ports/references/linopy/dispatch.py`.

The same model against the ladder's parquet. Where the YAML's
`where: p_max > 0` gives a retired generator no columns, this bounds them to
zero: same polytope, same objective.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


def build(tables: Mapping[str, Any]) -> Any:
    import linopy

    p_max = tables['p_max'].set_index('generator')['value']
    cost = tables['cost'].set_index('generator')['value']
    load = tables['load'].set_index('snapshot')['value']

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    m.add_constraints(p.sum('generator') == load, name='power_balance')
    m.add_objective((p * cost).sum())
    return m
