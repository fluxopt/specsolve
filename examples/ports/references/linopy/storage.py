#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["linopy==0.10.0", "pandas>=2.2", "xarray==2026.7.0", "highspy==1.15.1"]
# ///
"""Reference for ``storage``: the same LP, hand-written in linopy.

    pixi exec -s uv uv run --script examples/ports/references/linopy/storage.py

A teaching model, so what verifies it is agreement with an independent
formulation, not a published figure — see ``dispatch.py`` next door.

The line worth comparing is the cyclic state of charge. The YAML says
``shift(soc, along=snapshot, offset=1, edge='wrap')``; here the wrap is xarray's
``roll``, which sends the last snapshot's charge back to the first. Same
recurrence, stated against a different substrate — an array axis instead of an
ordered dimension.
"""

from __future__ import annotations

import json
from pathlib import Path

import linopy
import pandas as pd

DATA = Path(__file__).resolve().parents[2] / 'data' / 'storage.json'


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
    snapshots = load.index

    m = linopy.Model()
    p = m.add_variables(lower=0, upper=p_max, coords=[snapshots, p_max.index], name='p')
    charge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='charge')
    discharge = m.add_variables(lower=0, upper=30, coords=[snapshots], name='discharge')
    soc = m.add_variables(lower=0, upper=100, coords=[snapshots], name='soc')

    m.add_constraints(p.sum('generator') + discharge - charge == load, name='power_balance')
    m.add_constraints(soc == soc.roll(snapshot=1) + 0.9 * charge - discharge, name='soc_balance')
    m.add_objective((p * cost).sum())
    return m


def marginal_prices(m: linopy.Model) -> dict[str, list]:
    """The power-balance dual — what a unit of load costs each snapshot.

    With cyclic storage the peak price falls below the peaker's cost: the
    battery shaves it, and the price says by how much.
    """
    dual = m.constraints['power_balance'].dual
    return {'snapshot': [int(v) for v in dual.indexes['snapshot']], 'value': [float(v) for v in dual.values]}


def main() -> float:
    m = build(load_tables())
    status, condition = m.solve(solver_name='highs')
    assert status == 'ok', f'{status}: {condition}'
    print(f'linopy {linopy.__version__}')
    print(f'objective {float(m.objective.value)!r}')
    print(f'duals {json.dumps({"power_balance": marginal_prices(m)})}')
    return float(m.objective.value)


if __name__ == '__main__':
    main()
