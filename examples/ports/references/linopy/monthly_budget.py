#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.10.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``monthly_budget``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/monthly_budget.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The line worth comparing is the budget. The YAML groups by a relation over
snapshot — ``sum(p, over=snapshot, by=month_of[month])`` —
and linopy carries the same idea natively: ``p.groupby(month).sum()``, the
way PyPSA's own optimization layer aggregates. The difference is where the
calendar lives — a declared relation in the YAML, a data array the model
author threads through by hand here.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parents[2] / 'data' / 'monthly_budget.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    p_max: pd.Series = tables['p_max'].set_index('generator')['value']
    cost: pd.Series = tables['cost'].set_index('generator')['value']
    load: pd.Series = tables['load'].set_index('snapshot')['value']
    cap = xr.DataArray(tables['monthly_cap'].pivot(index='month', columns='generator', values='value'))
    month = xr.DataArray(tables['month_of'].set_index('snapshot')['month'])

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[load.index, p_max.index], name='p')
    m.add_constraints(p.sum('generator') == load, name='balance')
    m.add_constraints(p.groupby(month).sum() <= cap, name='monthly_budget')
    m.add_objective((p * cost).sum())
    return m


def budget_prices(m: linopy.Model) -> dict[str, list]:
    """The budget dual, tidy — what one more unit of a month's cap is worth."""
    dual = m.constraints['monthly_budget'].dual.transpose('month', 'generator')
    return {
        'month': [str(a) for a in dual.indexes['month'] for _ in dual.indexes['generator']],
        'generator': [str(b) for _ in dual.indexes['month'] for b in dual.indexes['generator']],
        'value': [float(v) for v in dual.values.ravel()],
    }


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"monthly_budget": budget_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
