#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.10.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``piecewise``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/piecewise.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The line worth comparing is the cost curve. The YAML's ``piecewise:`` block
links ``p`` and ``op_cost`` through per-generator breakpoints; here the same
convex hull is linopy's ``add_piecewise_formulation``, fed the breakpoint
tables pivoted wide and wrapped in its ``breakpoints`` factory. ``op_cost``
is *bounded below* by the curve (``'>='``) rather than pinned to it — under
minimisation the same convex hull, and what keeps the formulation a pure LP
with duals, as ``method: convex`` does on the YAML side; the pinned form makes
linopy reach for segment binaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'piecewise.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    p_max: pd.Series = tables['p_max'].set_index('generator')['value']
    load: pd.Series = tables['load'].set_index('snapshot')['value']
    curve_x: pd.DataFrame = tables['bp_x'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
    curve_y: pd.DataFrame = tables['bp_y'].pivot(index='generator', columns='bp', values='value').reindex(p_max.index)
    bp_x = linopy.breakpoints(curve_x, dim='generator')
    bp_y = linopy.breakpoints(curve_y, dim='generator')

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    op_cost = m.add_variables(lower=0, coords=[load.index, p_max.index], name='op_cost')
    m.add_piecewise_formulation((p, bp_x), (op_cost, bp_y, '>='))
    m.add_constraints(p.sum('generator') == load, name='balance')
    m.add_objective(op_cost.sum())
    return m


def marginal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual — the marginal cost read off the active segment."""
    dual = m.constraints['balance'].dual
    return {'snapshot': [int(v) for v in dual.indexes['snapshot']], 'value': [float(v) for v in dual.values]}


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"balance": marginal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
