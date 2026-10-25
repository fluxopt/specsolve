#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.10.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``multi_period``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/multi_period.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The line worth comparing is ``within_cap``. The YAML reads a per-period
variable at per-snapshot rows — ``at(p_nom, by=snapshot[period])`` —
and linopy says the same with a vectorised selection,
``p_nom.sel(period=period)``, each snapshot picking its period's capacity.
The ragged calendar (four snapshots in 2030, two in 2050) is just the values
of that selector.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd
import xarray as xr

DATA = Path(__file__).resolve().parents[2] / 'data' / 'multi_period.json'


def load_tables() -> dict[str, pd.DataFrame]:
    """The instance, one frame per parameter — what a caller of either library holds."""
    return {k: pd.DataFrame(v) if isinstance(v, dict) else v for k, v in json.loads(DATA.read_text()).items()}


def build(tables: dict[str, pd.DataFrame]) -> linopy.Model:
    """The instance's tables as a linopy model, row for row.

    ``tables`` is the same mapping the specsolve call attaches as ``sources``.
    """
    load: pd.Series = tables['load'].set_index('snapshot')['value']
    weight: pd.Series = tables['weight'].set_index('snapshot')['value']
    opex: pd.Series = tables['opex'].set_index('generator')['value']
    capex = xr.DataArray(tables['capex'].pivot(index='period', columns='generator', values='value'))
    period = xr.DataArray(tables['snapshot.period'].set_index('snapshot')['period'])

    m = linopy.Model()
    p = m.add_variables(lower=0, coords=[load.index, opex.index], name='p')
    p_nom = m.add_variables(lower=0, upper=100, coords=[capex.indexes['period'], opex.index], name='p_nom')
    m.add_constraints(p <= p_nom.sel(period=period), name='within_cap')
    m.add_constraints(p.sum('generator') == load, name='balance')
    m.add_objective((p * opex * weight).sum() + (p_nom * capex).sum())
    return m


def marginal_prices(m: linopy.Model) -> dict[str, list]:
    """The balance dual — the price a snapshot pays, capacity rent included."""
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
